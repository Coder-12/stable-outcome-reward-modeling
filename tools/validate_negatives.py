#!/usr/bin/env python3
import json, sys
from pathlib import Path

REQ_FIELDS = {
    "qid": str,
    "chain_id": str,
    "label": int,
    "orm_label": int,
    "input_text": str,
    "prompt": str,
    "steps": list,
    "step_targets": list,
    "step_mask": list,
    "step_confidence": list,
    "chain_confidence": (int, float),
    "final_answer": str,
    "prm_solution_gold": (int, float),
    "meta": dict,
    "created_at": str
}

def validate_record(rec):
    missing = []
    wrong_type = []
    for k, t in REQ_FIELDS.items():
        if k not in rec:
            missing.append(k)
        else:
            if not isinstance(rec[k], t):
                wrong_type.append((k, type(rec[k]).__name__, getattr(t, "__name__", str(t))))
    # step lengths
    steps = rec.get("steps") or []
    if rec.get("step_confidence") and len(rec["step_confidence"]) != len(steps):
        wrong_type.append(("step_confidence_len", len(rec["step_confidence"]), len(steps)))
    if rec.get("step_targets") and len(rec["step_targets"]) != len(steps):
        wrong_type.append(("step_targets_len", len(rec["step_targets"]), len(steps)))
    return missing, wrong_type

def main(path):
    path = Path(path)
    if not path.exists():
        print("File not found:", path); return 1
    total = 0; bad = 0
    sample_bad = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                bad += 1
                sample_bad.append(("parse_error", line[:200], str(e)))
                continue
            missing, wrong = validate_record(rec)
            if missing or wrong:
                bad += 1
                sample_bad.append((missing, wrong, rec.get("qid")))
    print(f"Checked {total} records — issues: {bad}")
    if sample_bad:
        print("First 10 issues:")
        for i, item in enumerate(sample_bad[:10], 1):
            print(i, item)
    return 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python tools/validate_negatives.py <jsonl_file>"); sys.exit(2)
    sys.exit(main(sys.argv[1]))