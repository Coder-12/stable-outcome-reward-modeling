#!/usr/bin/env python3
"""
merge_hybrid_dataset_adv_v3.py

Advanced merging pipeline for ORM-Repair V6 hybrid training:
 - Inputs: positive_gold.jsonl, synthetic_neg.jsonl, gpt_neg.jsonl (any subset)
 - Outputs: merged_{timestamp}.jsonl and merged_{timestamp}_summary.json

Features:
 - Schema validation & repair (keeps required fields)
 - Deduplication by (qid, chain_id) and content-hash
 - Per-qid sampling / stratified mixing to avoid leaking too many negatives for a single qid
 - Global balancing by desired ratios (pos : syn : gpt)
 - Optional filtering by step count / PRM / chain_confidence thresholds
 - Audit metadata summarizing distributions and actions
"""

import json, os, sys, hashlib, time, random
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional
from datetime import datetime

random.seed(12345)

# ---------------------------
# Utilities
# ---------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception as e:
                print(f"[WARN] Skipping invalid JSON line in {path}: {e}")
    return out

def write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

def content_hash(rec: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    if keys is None:
        keys = ["prompt", "final_answer", "steps"]
    s = []
    for k in keys:
        v = rec.get(k, "")
        if isinstance(v, (list, dict)):
            v = json.dumps(v, sort_keys=True)
        s.append(str(v))
    h = hashlib.sha1("|".join(s).encode("utf-8")).hexdigest()
    return h

def ensure_schema(rec: Dict[str, Any]) -> Dict[str, Any]:
    # required fields to keep for RM training
    r = dict(rec)  # shallow copy
    # canonical fields
    r["qid"] = r.get("qid") or r.get("id") or r.get("question_id") or str(hashlib.sha1(json.dumps(r, sort_keys=True).encode()).hexdigest())[:16]
    r["chain_id"] = r.get("chain_id") or r.get("cid") or str(uuid4_short())
    r["prompt"] = r.get("prompt") or r.get("input_text") or ""
    r["steps"] = r.get("steps") or r.get("step_text") or []
    # step_targets: ensure list of floats if present
    if "step_targets" in r and isinstance(r["step_targets"], list):
        r["step_targets"] = [float(x) for x in r["step_targets"]]
    else:
        # leave absent or empty list
        r["step_targets"] = r.get("step_targets", [])
    # labels
    r["orm_label"] = int(r.get("orm_label", r.get("label", 0)))
    # chain_confidence
    r["chain_confidence"] = float(r.get("chain_confidence", r.get("meta", {}).get("chain_confidence", 1.0)))
    # prm_solution_gold
    if "prm_solution_gold" in r:
        try:
            r["prm_solution_gold"] = float(r["prm_solution_gold"])
        except:
            r["prm_solution_gold"] = 0.0
    return r

def uuid4_short():
    return hashlib.sha1(str(random.random()).encode()).hexdigest()[:12]

# ---------------------------
# Merge logic
# ---------------------------
def merge_datasets(
    pos_files: List[str],
    syn_files: List[str],
    gpt_files: List[str],
    out_prefix: str = "merged",
    target_counts: Optional[Dict[str, int]] = None,
    ratio: Optional[Dict[str, float]] = None,
    per_qid_max_neg: int = 3,
    dedupe_on_content: bool = True,
    min_steps: int = 1,
    max_steps: Optional[int] = None,
    min_chain_confidence: float = 0.0,
    shuffle_output: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Args:
      pos_files, syn_files, gpt_files: list of file paths
      target_counts: explicit desired counts e.g. {"pos":1000,"syn":1200,"gpt":800}
      ratio: mixing ratio e.g. {"pos":1.0,"syn":1.0,"gpt":0.5} (relative)
      per_qid_max_neg: max negatives to keep per qid from combined neg pools
      dedupe_on_content: dedupe by content hash
      min_steps / max_steps: filter by step count if steps present
      min_chain_confidence: remove very-low-confidence chains
    Returns:
      summary dict with paths and counts
    """
    # Load
    def _load_list(files):
        out = []
        for f in files or []:
            if not f:
                continue
            if not os.path.exists(f):
                print(f"[WARN] File not found: {f}")
                continue
            out.extend(read_jsonl(f))
        return out

    pos = _load_list(pos_files)
    syn = _load_list(syn_files)
    gpt = _load_list(gpt_files)

    if verbose:
        print(f"[LOAD] pos={len(pos)} syn={len(syn)} gpt={len(gpt)}")

    # Ensure schema & annotate source
    def annotate_and_schema(records, src_name):
        out = []
        for r in records:
            rr = ensure_schema(r)
            rr["__src__"] = src_name
            # compute simple STeP count
            rr["__n_steps__"] = len(rr.get("steps") or rr.get("step_targets") or [])
            out.append(rr)
        return out

    pos = annotate_and_schema(pos, "pos")
    syn = annotate_and_schema(syn, "syn")
    gpt = annotate_and_schema(gpt, "gpt")

    # Optional filtering
    def apply_filters(records):
        filtered = []
        for r in records:
            if r["__n_steps__"] < min_steps:
                continue
            if max_steps is not None and r["__n_steps__"] > max_steps:
                continue
            if r.get("chain_confidence", 1.0) < min_chain_confidence:
                continue
            filtered.append(r)
        return filtered

    pos = apply_filters(pos)
    syn = apply_filters(syn)
    gpt = apply_filters(gpt)

    if verbose:
        print(f"[AFTER FILTER] pos={len(pos)} syn={len(syn)} gpt={len(gpt)}")

    # Deduplicate by (qid, chain_id) and optionally by content hash
    seen_pairs = set()
    seen_hashes = set()
    def uniqueify(records):
        out = []
        for r in records:
            pair = (r["qid"], str(r["chain_id"]))
            if pair in seen_pairs:
                continue
            h = content_hash(r)
            if dedupe_on_content and h in seen_hashes:
                continue
            seen_pairs.add(pair)
            seen_hashes.add(h)
            out.append(r)
        return out

    pos = uniqueify(pos)
    syn = uniqueify(syn)
    gpt = uniqueify(gpt)

    if verbose:
        print(f"[DEDUP] pos={len(pos)} syn={len(syn)} gpt={len(gpt)}")

    # Prepare per-qid negative limiting to avoid negative explosion per qid
    neg_by_qid = defaultdict(list)
    for r in syn + gpt:
        neg_by_qid[r["qid"]].append(r)

    # Enforce per-qid cap
    capped_negs = []
    for qid, recs in neg_by_qid.items():
        if len(recs) <= per_qid_max_neg:
            capped_negs.extend(recs)
        else:
            capped_negs.extend(random.sample(recs, per_qid_max_neg))

    # Re-split capped into syn/gpt groups (source preserved)
    syn_capped = [r for r in capped_negs if r["__src__"] == "syn"]
    gpt_capped = [r for r in capped_negs if r["__src__"] == "gpt"]

    # Decide target counts
    total_pos = len(pos)
    if target_counts:
        tgt_pos = target_counts.get("pos", total_pos)
        tgt_syn = target_counts.get("syn", 0)
        tgt_gpt = target_counts.get("gpt", 0)
    elif ratio:
        # compute relative to pos
        base = ratio.get("pos", 1.0)
        if base == 0:
            base = 1.0
        scale = total_pos / base
        tgt_pos = total_pos
        tgt_syn = int(round(ratio.get("syn", 0.0) * scale))
        tgt_gpt = int(round(ratio.get("gpt", 0.0) * scale))
    else:
        # default: keep all pos, keep up to available negatives
        tgt_pos = total_pos
        tgt_syn = len(syn_capped)
        tgt_gpt = len(gpt_capped)

    if verbose:
        print(f"[TARGET] tgt_pos={tgt_pos} tgt_syn={tgt_syn} tgt_gpt={tgt_gpt}")

    # Sample positive (prefer keep all but sample if requested smaller)
    if tgt_pos >= len(pos):
        final_pos = pos
    else:
        final_pos = random.sample(pos, tgt_pos)

    # Stratified sampling for negatives: sample per-qid to avoid too many negs from few qids
    def stratified_sample(negs, tgt_n):
        if tgt_n <= 0:
            return []
        # group by qid
        by_qid = defaultdict(list)
        for r in negs:
            by_qid[r["qid"]].append(r)
        qids = list(by_qid.keys())
        chosen = []
        # round-robin pick one per qid until target reached
        idx = 0
        while len(chosen) < tgt_n and qids:
            qid = qids[idx % len(qids)]
            bucket = by_qid[qid]
            if bucket:
                chosen.append(bucket.pop(random.randrange(len(bucket))))
            else:
                qids.remove(qid)
                idx -= 1
            idx += 1
        # if not enough collected, fallback to random sample from remaining
        if len(chosen) < tgt_n:
            remaining = [r for recs in by_qid.values() for r in recs]
            need = tgt_n - len(chosen)
            if remaining:
                chosen.extend(random.sample(remaining, min(len(remaining), need)))
        return chosen

    final_syn = stratified_sample(syn_capped, min(tgt_syn, len(syn_capped)))
    final_gpt = stratified_sample(gpt_capped, min(tgt_gpt, len(gpt_capped)))

    merged = final_pos + final_syn + final_gpt
    if shuffle_output:
        random.shuffle(merged)

    # Final dedupe pass (just in case)
    final_out = []
    seen = set()
    for r in merged:
        h = content_hash(r)
        if h in seen:
            continue
        seen.add(h)
        final_out.append(r)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_jsonl = f"{out_prefix}_{timestamp}.jsonl"
    out_summary = f"{out_prefix}_{timestamp}_summary.json"

    write_jsonl(out_jsonl, final_out)

    # Produce summary
    counts = {"pos": len(final_pos), "syn": len(final_syn), "gpt": len(final_gpt), "total": len(final_out)}
    by_qid_counts = Counter([r["qid"] for r in final_out])
    step_lens = [r.get("__n_steps__", 0) for r in final_out]
    mean_steps = sum(step_lens) / (len(step_lens) or 1)
    summary = {
        "timestamp": timestamp,
        "input_counts": {"pos": len(pos), "syn": len(syn), "gpt": len(gpt)},
        "target_counts": {"pos": tgt_pos, "syn": tgt_syn, "gpt": tgt_gpt},
        "final_counts": counts,
        "per_qid_max_neg": per_qid_max_neg,
        "dedupe_on_content": dedupe_on_content,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "min_chain_confidence": min_chain_confidence,
        "mean_steps": mean_steps,
        "unique_qids": len(by_qid_counts),
        "qids_with_many_chains": {k: v for k, v in by_qid_counts.items() if v > 3},
    }

    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        print(f"[WRITE] {out_jsonl} ({len(final_out)} records)")
        print(f"[SUMMARY] -> {out_summary}")

    return {"out_jsonl": out_jsonl, "out_summary": out_summary, "summary": summary}

# ---------------------------
# CLI
# ---------------------------
def parse_args_and_run():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pos", nargs="*", default=[], help="Paths to positive/gold jsonl files")
    p.add_argument("--syn", nargs="*", default=[], help="Paths to synthetic negatives jsonl files")
    p.add_argument("--gpt", nargs="*", default=[], help="Paths to GPT negatives jsonl files")
    p.add_argument("--out_prefix", type=str, default="merged", help="Output prefix")
    p.add_argument("--per_qid_max_neg", type=int, default=3)
    p.add_argument("--min_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--min_chain_confidence", type=float, default=0.0)
    p.add_argument("--ratio_pos", type=float, default=1.0)
    p.add_argument("--ratio_syn", type=float, default=1.0)
    p.add_argument("--ratio_gpt", type=float, default=0.5)
    p.add_argument("--shuffle", action="store_true", default=True)
    args = p.parse_args()

    ratio = {"pos": args.ratio_pos, "syn": args.ratio_syn, "gpt": args.ratio_gpt}
    summary = merge_datasets(
        pos_files=args.pos,
        syn_files=args.syn,
        gpt_files=args.gpt,
        out_prefix=args.out_prefix,
        ratio=ratio,
        per_qid_max_neg=args.per_qid_max_neg,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        min_chain_confidence=args.min_chain_confidence,
        shuffle_output=args.shuffle,
        verbose=True,
    )
    print(json.dumps(summary["summary"], indent=2))

if __name__ == "__main__":
    parse_args_and_run()