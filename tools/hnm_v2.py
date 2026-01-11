#!/usr/bin/env python3
"""
tools/hnm_v2.py

Hard-Negative Mining V2 (HNM-v2)

Scoring pipeline:
 - Loads candidate chains JSONL (each record contains qid, chain_id, input_text/prompt, steps, optional step_targets)
 - Loads a trained DualHeadRM model (or path to checkpoint); uses model to produce ORM logits and PRM per-step logits
 - Computes aggregated PRM score (mean or product as option) and ORM probability
 - Computes hardness = prm_mean * (1 - orm_prob) (configurable)
 - For each qid, select top_k hardest negatives (or above threshold)
 - Write selected negatives to out_jsonl for merging into training set

Example:
python tools/hnm_v2.py \
  --candidates data/collected/gsm8k_candidate_chains.jsonl \
  --model_ckpt runs/checkpoints/best_model.pt \
  --tokenizer ../opt1p3b_fp16 \
  --out data/hnm_v2_hardneg_topk.jsonl \
  --top_k 4
"""
import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional
import math
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Local model import (assumes same API as earlier)
from src.models.dual_head_rm import DualHeadRM


# ---------------------------
# Lightweight dataset for scoring
# ---------------------------
class CandidateDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer_name: str, max_length: int = 512, max_steps: Optional[int] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length
        self.max_steps = max_steps
        self.records = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                rec = json.loads(ln)
                # normalize fields
                rec["input_text"] = rec.get("input_text", rec.get("prompt", ""))
                rec["qid"] = rec.get("qid", rec.get("id", ""))
                rec["chain_id"] = rec.get("chain_id", rec.get("cid", 0))
                # optional per-step scores
                rec["step_targets"] = rec.get("step_targets", rec.get("prm_steps", []))
                self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["input_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {
            "qid": rec["qid"],
            "chain_id": rec["chain_id"],
            "raw": rec,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            # pass step_targets if present (optional)
            "step_targets": torch.tensor(rec.get("step_targets", []), dtype=torch.float32) if rec.get("step_targets") else None
        }
        return item


# collate
def collate_fn(batch: List[Dict[str, Any]]):
    out = {}
    out["qid"] = [b["qid"] for b in batch]
    out["chain_id"] = [b["chain_id"] for b in batch]
    out["raw"] = [b["raw"] for b in batch]
    out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
    out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch], dim=0)
    # step_targets: variable length -> keep as list (not used by model, only optional)
    out["step_targets"] = [b["step_targets"] for b in batch]
    return out


# ---------------------------
# Aggregation utilities
# ---------------------------
def aggregate_prm(step_preds: torch.Tensor, step_mask: torch.Tensor, mode: str = "mean", safe_eps: float = 1e-9):
    """
    step_preds: (B, S) logits or probabilities depending on model output.
    step_mask: (B, S) mask 0/1
    mode: "mean" or "prod" or "median"
    """
    # assume step_preds are raw logits; convert to probabilities
    probs = torch.sigmoid(step_preds)
    masked = probs * step_mask
    denom = step_mask.sum(dim=1).clamp_min(1.0)
    if mode == "mean":
        return (masked.sum(dim=1) / denom).clamp(0.0, 1.0)
    elif mode == "prod":
        # use (1 - p) product trick to avoid underflow: prod(p_i) but numerically stable in log-space
        masked_vals = probs * (step_mask.bool())
        # compute log-prob ignoring masked positions by replacing zeros with 1.0 in multiplication (log(1)=0)
        safe_probs = probs * step_mask + (1 - step_mask)  # masked positions -> 1.0
        logp = torch.log(safe_probs.clamp(min=safe_eps))
        return torch.exp(logp.sum(dim=1))
    elif mode == "median":
        # compute median across valid positions (fallback to mean)
        res = []
        for i in range(probs.size(0)):
            valid = probs[i][step_mask[i].bool()]
            if valid.numel() == 0:
                res.append(torch.tensor(0.0, device=probs.device))
            else:
                res.append(valid.median())
        return torch.stack(res)
    else:
        raise ValueError("Unknown mode: " + mode)


# ---------------------------
# Hardness metric
# ---------------------------
def compute_hardness(prm_score: torch.Tensor, orm_prob: torch.Tensor, metric: str = "prm_times_negorm"):
    """
    prm_score, orm_prob: (B,)
    metric options:
      - "prm_times_negorm" = prm_mean * (1 - orm_prob)  [default]
      - "abs_diff" = abs(prm_mean - orm_prob)
      - "margin" = (prm_mean - orm_prob)  (large positive means PRM says good but ORM says bad)
      - "ensemble_uncertainty" = prm*(1-prm) + orm*(1-orm) (uncertainty)
    """
    if metric == "prm_times_negorm":
        return prm_score * (1.0 - orm_prob)
    if metric == "abs_diff":
        return torch.abs(prm_score - orm_prob)
    if metric == "margin":
        return prm_score - orm_prob
    if metric == "ensemble_uncertainty":
        return prm_score * (1.0 - prm_score) + orm_prob * (1.0 - orm_prob)
    raise ValueError(metric)


# ---------------------------
# Main scoring & selection logic
# ---------------------------
def run_hnm(
    candidates_jsonl: str,
    model_ckpt: str,
    tokenizer_name: str,
    out_jsonl: str,
    device: str = "cuda",
    batch_size: int = 32,
    top_k: int = 4,
    prune_threshold: Optional[float] = None,
    prm_agg: str = "mean",
    hardness_metric: str = "prm_times_negorm",
    max_examples: Optional[int] = None,
    verbose: bool = True
):
    # load dataset
    ds = CandidateDataset(candidates_jsonl, tokenizer_name)
    if max_examples:
        indices = list(range(min(max_examples, len(ds))))
        subset = torch.utils.data.Subset(ds, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # load model
    # Model constructor should mirror your DualHeadRM signature
    model = DualHeadRM(
        base_model_name=model_ckpt,  # if your DualHeadRM loads pretrained inside
        max_steps=ds.tokenizer.model_max_length,  # not ideal but we only need heads shapes; adjust per your constructor
        pooling="eos",
        device=device
    ).to(device)

    # If model_ckpt is a state dict rather than model name, try load
    if os.path.exists(model_ckpt) and model_ckpt.endswith(".pt"):
        state = torch.load(model_ckpt, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception:
            # maybe the saved dict was model.module or similar; attempt best-effort
            try:
                model.load_state_dict(state, strict=False)
            except Exception as e:
                print("[HNM] Warning: could not load checkpoint cleanly:", e)

    model.eval()

    # per-qid accumulator
    qid_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Scoring candidates", disable=not verbose):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids, attention_mask=attention_mask, step_mask=None)  # model should accept step_mask None
            orm_logits = out["orm_logits"].view(-1)  # (B,)
            prm_logits = out["prm_logits"]  # (B, S) logits

            # Build step_mask if model returned it or fallback to all-ones
            # try detecting step length
            S = prm_logits.size(1)
            step_mask = torch.ones_like(prm_logits)
            # aggregate PRM
            prm_agg_score = aggregate_prm(prm_logits, step_mask, mode=prm_agg)

            orm_prob = torch.sigmoid(orm_logits)

            hardness = compute_hardness(prm_agg_score, orm_prob, metric=hardness_metric)

            # record per-sample details
            for i in range(len(batch["qid"])):
                qid = batch["qid"][i]
                chain_id = batch["chain_id"][i]
                raw = batch["raw"][i]
                rec = {
                    "qid": qid,
                    "chain_id": chain_id,
                    "raw": raw,
                    "orm_logit": float(orm_logits[i].detach().cpu().item()),
                    "orm_prob": float(orm_prob[i].detach().cpu().item()),
                    "prm_score": float(prm_agg_score[i].detach().cpu().item()),
                    "hardness": float(hardness[i].detach().cpu().item())
                }
                # optional pruning by absolute prm or hardness
                if prune_threshold is not None and rec["hardness"] < prune_threshold:
                    continue
                qid_buckets[qid].append(rec)

    # select top_k per qid
    selected = []
    for qid, recs in qid_buckets.items():
        # sort by hardness descending
        recs_sorted = sorted(recs, key=lambda x: x["hardness"], reverse=True)
        choose = recs_sorted[:top_k]
        for r in choose:
            # convert to training-format negative: ensure orm_label=0 and step_targets reflect low prm if you want
            neg = r["raw"]
            neg["orm_label"] = 0
            # optionally set prm_solution_gold to prm_score or to low value to indicate wrong
            neg["prm_solution_gold"] = r["prm_score"]
            # add hardness meta
            neg.setdefault("meta", {})
            neg["meta"]["hnm_hardness"] = r["hardness"]
            neg["meta"]["hnm_orm_prob"] = r["orm_prob"]
            neg["meta"]["hnm_prm_score"] = r["prm_score"]
            selected.append(neg)

    # write out
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for s in selected:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[HNM-v2] Selected {len(selected)} hard negatives -> {out_jsonl}")
    return out_jsonl


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True, help="JSONL file with candidate chains")
    p.add_argument("--model_ckpt", required=True, help="Model checkpoint path (state_dict .pt) or base model name")
    p.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    p.add_argument("--out", required=True, help="Output JSONL for selected hard negatives")
    p.add_argument("--device", default="cuda", help="device")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=4, help="select top_k per qid")
    p.add_argument("--prune_threshold", type=float, default=None, help="optional min hardness to keep")
    p.add_argument("--prm_agg", choices=["mean", "prod", "median"], default="mean")
    p.add_argument("--metric", choices=["prm_times_negorm", "abs_diff", "margin", "ensemble_uncertainty"], default="prm_times_negorm")
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--no_verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hnm(
        candidates_jsonl=args.candidates,
        model_ckpt=args.model_ckpt,
        tokenizer_name=args.tokenizer,
        out_jsonl=args.out,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        prune_threshold=args.prune_threshold,
        prm_agg=args.prm_agg,
        hardness_metric=args.metric,
        max_examples=args.max_examples,
        verbose=not args.no_verbose
    )