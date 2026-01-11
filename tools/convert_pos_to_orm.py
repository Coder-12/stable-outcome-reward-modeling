#!/usr/bin/env python3
"""
tools/convert_pos_to_orm.py

Convert "raw" positive CoT JSONL (full reasoning in `prompt`) into
ORM-positive format expected by the training pipeline.

Input:  data/processed/orm_train.jsonl
Output: data/processed/orm_train_converted.jsonl

Behavior:
 - Extract numbered steps (prefers lines starting with "1.", "2." etc).
 - Fallback splitting when numbering not present.
 - Extract final answer from:
     - "Final Answer: <text>"
     - meta.final_answer or meta.gold_answer (looks for "#### 42" style)
 - Populate ORM fields (label, orm_label, steps, step_confidence, step_targets etc).
 - Keep meta.original_qid / meta.original_chain_id for traceability.
 - Ensure unique qid and chain_id in output.
"""
from __future__ import annotations
import re
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any
import sys

ISO_NOW = lambda: datetime.now(timezone.utc).isoformat()

# ---------- helper regex ----------
NUM_STEP_RE = re.compile(r'^\s*\d+\.\s*(.+)$')  # lines starting with "1. "
FINAL_ANSWER_RE = re.compile(r'Final Answer\s*[:\-]\s*(.+)', re.IGNORECASE)
FOOTER_ANSWER_RE = re.compile(r'####\s*(.+)')  # patterns like "#### 42"
INLINE_ANSWER_RE = re.compile(r'Answer\s*[:\-]\s*(.+)', re.IGNORECASE)

# ---------- extraction functions ----------
def extract_steps_from_prompt(text: str) -> List[str]:
    """
    Extract numbered steps. If none found, fallback to splitting by blank lines
    and using first N meaningful lines (min 4).
    Returns list of cleaned steps (no numbering, trimmed).
    """
    if not text:
        return []
    lines = [ln.rstrip() for ln in text.splitlines()]
    steps = []
    # Try to capture multi-line numbered steps: gather consecutive NUM_STEP_RE matches
    for ln in lines:
        m = NUM_STEP_RE.match(ln)
        if m:
            steps.append(m.group(1).strip())
    if len(steps) >= 1:
        # Some steps may include embedded equations with <<..>>; keep as-is but stripped
        return steps

    # fallback: take non-empty lines and group paragraphs
    paras = []
    cur = []
    for ln in lines:
        if ln.strip() == "":
            if cur:
                paras.append(" ".join([l.strip() for l in cur]).strip())
                cur = []
        else:
            cur.append(ln)
    if cur:
        paras.append(" ".join([l.strip() for l in cur]).strip())

    # prune lines that are too short/headers like "Verification:"
    paras = [p for p in paras if len(p) > 10]

    # If we have paragraphs, try to find those containing arithmetic steps by heuristics:
    if paras:
        # Keep up to first 8 paras as steps, but require at least 4.
        chosen = paras[:8]
        if len(chosen) < 4:
            # further split longer paras into sentences as fallback
            sentences = []
            for p in paras:
                parts = re.split(r'\.\s+', p)
                for s in parts:
                    s = s.strip()
                    if s:
                        sentences.append(s if s.endswith('.') else s + '.')
                if len(sentences) >= 8:
                    break
            chosen = sentences[:max(4, len(sentences))]
        return [s.strip() for s in chosen[:8]]

    # worst-case: split by lines and take first 4 non-empty lines
    nonempty = [ln.strip() for ln in lines if ln.strip()]
    return nonempty[:max(4, min(8, len(nonempty)))]


def extract_final_answer(prompt_text: str, meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Try to extract final answer string and method used.
    Returns (final_answer, method_tag)
    """
    # 1) search within prompt for "Final Answer:" style
    if prompt_text:
        for ln in reversed(prompt_text.splitlines()):
            if not ln.strip():
                continue
            m = FINAL_ANSWER_RE.search(ln)
            if m:
                return m.group(1).strip(), "final_answer_in_prompt"
            m2 = FOOTER_ANSWER_RE.search(ln)
            if m2:
                return m2.group(1).strip(), "footer_hash_in_prompt"
    # 2) look into meta.final_answer or meta.gold_answer
    if isinstance(meta, dict):
        for key in ("final_answer", "final_answer_norm", "gold_answer"):
            val = meta.get(key)
            if not val:
                continue
            # if it's a long gold with markers "#### 42", try to extract number after ####
            if isinstance(val, str):
                m = FOOTER_ANSWER_RE.search(val)
                if m:
                    return m.group(1).strip(), f"{key}_footer_hash"
                # fallback: if last token seems numeric or short, pick last line
                lines = [l.strip() for l in val.splitlines() if l.strip()]
                if lines:
                    last = lines[-1]
                    # if last line is just a number or short answer
                    if len(last) < 100:
                        # clean any leading "####" or "Final Answer:" remnants
                        last_clean = re.sub(r'^(Final Answer[:\-]\s*)|^(####\s*)', '', last, flags=re.IGNORECASE).strip()
                        return last_clean, f"{key}_lastline"
    # 3) try inline "Answer: " matches in prompt text
    if prompt_text:
        m = INLINE_ANSWER_RE.search(prompt_text)
        if m:
            return m.group(1).strip(), "inline_answer_re"

    # 4) if nothing found, return empty and tag
    return "", "not_found"

# ---------- main conversion ----------
def convert_record(rec: Dict[str, Any], enforce_min_steps: int = 4) -> Tuple[Dict[str, Any], List[str]]:
    """
    Convert a single raw record to ORM-positive format.
    Returns (converted_record, issues_list)
    """
    issues = []
    original_qid = rec.get("qid")
    original_chain_id = rec.get("chain_id")

    prompt_text = rec.get("prompt") or rec.get("input_text") or ""
    meta = rec.get("meta") or {}

    steps = extract_steps_from_prompt(prompt_text)
    if not steps or len(steps) < enforce_min_steps:
        issues.append("insufficient_steps_extracted")
        # attempt to be helpful: attempt more aggressive extraction
        # split by sentence-ending punctuation and take first 4
        sents = re.split(r'(?<=[\.\?\!])\s+', prompt_text)
        sents = [s.strip() for s in sents if len(s.strip()) > 5]
        steps = sents[:max(enforce_min_steps, len(sents))]
        if len(steps) < enforce_min_steps:
            # last resort: take first 4 non-empty lines
            lines = [ln.strip() for ln in prompt_text.splitlines() if ln.strip()]
            steps = lines[:enforce_min_steps]

    final_answer, fa_method = extract_final_answer(prompt_text, meta)
    if not final_answer:
        issues.append("final_answer_missing")
        # try meta fields more aggressively
        if isinstance(meta, dict):
            for k in ("final_answer", "final_answer_norm", "gold_answer", "final_text"):
                if meta.get(k):
                    final_answer = meta.get(k)
                    break
    final_answer = (final_answer or "").strip()

    # Prepare ORM fields
    # ensure we have at least 4 steps, crop to 8 max
    if len(steps) < enforce_min_steps:
        # pad with placeholder steps (unlikely)
        while len(steps) < enforce_min_steps:
            steps.append("(verification step)")
    if len(steps) > 8:
        steps = steps[:8]

    # Clean steps: remove leading numbering if any remains and excessive whitespace
    cleaned_steps = []
    for s in steps:
        s2 = re.sub(r'^\s*\d+\.\s*', '', s).strip()
        cleaned_steps.append(s2)

    # Create new unique qid and chain_id but preserve originals in meta
    new_qid = str(uuid.uuid4())
    new_chain_id = f"pos-{uuid.uuid4()}"

    record_out: Dict[str, Any] = {
        "qid": new_qid,
        "chain_id": new_chain_id,
        "label": 1,
        "orm_label": 1,
        "input_text": "\n".join(f"{i+1}. {s}" for i, s in enumerate(cleaned_steps)) + ("\n\nFinal Answer: " + final_answer if final_answer else ""),
        "prompt": "\n".join(f"{i+1}. {s}" for i, s in enumerate(cleaned_steps)) + ("\n\nFinal Answer: " + final_answer if final_answer else ""),
        "steps": cleaned_steps,
        # For positives, step_targets should be high (1.0 = correct step)
        "step_targets": [1.0 for _ in cleaned_steps],
        "step_mask": [1 for _ in cleaned_steps],
        # high confidence for gold steps
        "step_confidence": [0.9 for _ in cleaned_steps],
        "chain_confidence": 0.95,
        "final_answer": final_answer,
        # prm_solution_gold: for positives, strong signal ~1.0
        "prm_solution_gold": 1.0,
        "meta": {
            "original_qid": original_qid,
            "original_chain_id": original_chain_id,
            "gold_answer": meta.get("gold_answer") or meta.get("final_answer") or final_answer,
            "source_meta": meta,
            "converted_by": "convert_pos_to_orm.py"
        },
        "created_at": ISO_NOW()
    }

    return record_out, issues

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Convert raw positive CoT JSONL into ORM-positive JSONL")
    ap.add_argument("--in_file", default="data/processed/orm_train.jsonl")
    ap.add_argument("--out_file", default="data/processed/orm_train_converted.jsonl")
    ap.add_argument("--report", default="data/qc/orm_positives_convert_report.json")
    ap.add_argument("--min_steps", type=int, default=4)
    args = ap.parse_args()

    in_p = Path(args.in_file)
    out_p = Path(args.out_file)
    report_p = Path(args.report)

    if not in_p.exists():
        print(f"[ERROR] input not found: {in_p}", file=sys.stderr)
        raise SystemExit(2)

    out_tmp = out_p.with_suffix(out_p.suffix + ".tmp")

    total = 0
    converted = 0
    issues_counter: Dict[str, int] = {}
    step_counts = []
    seen_qids = set()
    seen_chain_ids = set()

    with in_p.open("r", encoding="utf-8") as fin, out_tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                issues_counter["parse_error"] = issues_counter.get("parse_error", 0) + 1
                continue

            out_rec, issues = convert_record(rec, enforce_min_steps=args.min_steps)
            # track
            for it in issues:
                issues_counter[it] = issues_counter.get(it, 0) + 1

            # ensure unique qid/chain_id
            while out_rec["qid"] in seen_qids:
                out_rec["qid"] = str(uuid.uuid4())
            while out_rec["chain_id"] in seen_chain_ids:
                out_rec["chain_id"] = f"pos-{uuid.uuid4()}"

            seen_qids.add(out_rec["qid"])
            seen_chain_ids.add(out_rec["chain_id"])

            step_counts.append(len(out_rec["steps"]))

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            converted += 1

    # atomic replace
    out_tmp.replace(out_p)

    # write report
    report = {
        "input_file": str(in_p),
        "output_file": str(out_p),
        "total_input_records": total,
        "total_converted": converted,
        "issues": issues_counter,
        "step_count_min": min(step_counts) if step_counts else None,
        "step_count_mean": (sum(step_counts) / len(step_counts)) if step_counts else None,
        "step_count_max": max(step_counts) if step_counts else None,
        "unique_qids": len(seen_qids),
        "unique_chain_ids": len(seen_chain_ids),
        "created_at": ISO_NOW()
    }
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with report_p.open("w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)

    print("Conversion complete.")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()