#!/usr/bin/env python3
"""
validate_and_prepare_for_gpt.py

- Load ORM selective JSONL (e.g. data/orm_train_selective_for_gpt.jsonl)
- Load GSM8K gold JSONL (data/collected/gsm8k_training_chains_1000_FINAL_gold_data.jsonl)
  which has {"id": "...", "question": "...", ...}
- For each ORM record:
    - try to look up question by qid -> id
    - if found: attach question and mark source "gsm"
    - else: extract a short question string robustly from the ORM prompt and mark source "extracted"
- Writes:
    - prepared output: data/orm_train_selective_for_gpt_prepared.jsonl
    - diagnostics: data/orm_train_selective_for_gpt_unmatched.jsonl (if any)
- Prints summary and exits with non-zero code if critical failures found (optional).
"""
import json
import re
from pathlib import Path
from typing import Dict, Optional
import argparse

def load_gsm_map(gsm_path: Path) -> Dict[str,str]:
    m = {}
    with gsm_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            kid = obj.get("id") or obj.get("qid")
            qtxt = obj.get("question") or obj.get("prompt") or obj.get("raw_question")
            if kid and qtxt:
                m[str(kid)] = qtxt
    return m

QUESTION_LIKE_PATTERNS = [
    r"(?:(?:what|how|why|when|who|which)\b.*\?)",   # typical question sentence
    r"(?:calculate|determine|find|compute|what is|how many|how much)[^\n]{0,200}",
    r"[A-Z][^\n]{20,250}\?"  # long sentence ending with ?
]

QUESTION_RE = re.compile("|".join("(" + p + ")" for p in QUESTION_LIKE_PATTERNS), re.IGNORECASE)

def extract_question_from_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    # try to find clear question-like sentence
    for m in QUESTION_RE.finditer(prompt.replace("\n"," ")):
        q = m.group(0).strip()
        # trim long trailing context
        if len(q) > 300:
            q = q[:300].rsplit(" ",1)[0] + "..."
        return q
    # fallback heuristics: return first 1-2 meaningful lines
    lines = [l.strip() for l in prompt.splitlines() if l.strip()]
    if not lines:
        return ""
    # if first line is a CoT-step like "1. Start with ..." try to take next lines that look like question words
    # else join first 2 lines
    if len(lines) == 1:
        return lines[0][:300]
    return (" ".join(lines[:2]))[:300]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orm", required=True, help="ORM selective JSONL (input)")
    ap.add_argument("--gsm", required=True, help="GSM8K GOLD JSONL (id->question mapping)")
    ap.add_argument("--out_prepared", default="data/orm_train_selective_for_gpt_prepared.jsonl")
    ap.add_argument("--out_unmatched", default="data/orm_train_selective_for_gpt_unmatched.jsonl")
    ap.add_argument("--fail_on_unmatched", action="store_true", help="Exit non-zero if any unmatched (useful to enforce 100% mapping)")
    ap.add_argument("--max_preview", type=int, default=8)
    args = ap.parse_args()

    orm_path = Path(args.orm)
    gsm_path = Path(args.gsm)
    out_p = Path(args.out_prepared)
    out_un = Path(args.out_unmatched)

    gsm_map = load_gsm_map(gsm_path)
    print(f"[INFO] GSM map loaded: {len(gsm_map)} entries from {gsm_path}")

    total = 0
    matched = 0
    unmatched = []
    with orm_path.open("r", encoding="utf-8") as fin, \
         out_p.open("w", encoding="utf-8") as fout:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            total += 1
            rec = json.loads(ln)
            qid = rec.get("qid") or rec.get("id")
            question = None
            src = None
            if qid is not None and str(qid) in gsm_map:
                question = gsm_map[str(qid)]
                src = "gsm"
                matched += 1
            else:
                # try alternate: maybe the ORM record meta has gold id or gold_answer that maps
                # fallback: extract from prompt robustly
                question = extract_question_from_prompt(rec.get("prompt","") or rec.get("input_text",""))
                src = "extracted"
                unmatched.append({"qid": qid, "reason": "no_gsm_match", "extracted": question, "rec_preview": rec if total<=args.max_preview else None})
            # attach question to rec
            rec_prepared = dict(rec)  # shallow copy
            rec_prepared["_gpt_question"] = question
            rec_prepared["_gpt_question_source"] = src
            fout.write(json.dumps(rec_prepared, ensure_ascii=False) + "\n")

    # write unmatched diagnostics
    with out_un.open("w", encoding="utf-8") as fu:
        for u in unmatched:
            fu.write(json.dumps(u, ensure_ascii=False) + "\n")

    print(f"[DONE] Processed {total} ORM records")
    print(f"  matched to GSM gold questions: {matched}")
    print(f"  unmatched (used extracted fallback): {len(unmatched)}")
    print(f"Prepared file -> {out_p}")
    print(f"Unmatched diagnostics -> {out_un}")

    if args.fail_on_unmatched and len(unmatched) > 0:
        print("[ERROR] There are unmatched qids. Exiting with failure (use --fail_on_unmatched to enforce).")
        raise SystemExit(2)

if __name__ == '__main__':
    main()