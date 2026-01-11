#!/usr/bin/env python3
"""
Merge Synthetic Negatives (ADV_V7 + V1) into a unified ORM-Repair dataset.

This script:
- Loads two JSONL negative files
- Dedupes using SHA256 on CoT text
- Shuffles to balance distributions
- Outputs a merged JSONL ready for ORM training

Usage:
  python merge_synth_negatives.py \
      --file1 data/repair_v6_synth_neg.jsonl \
      --file2 data/repair_v6_synth_v1_neg.jsonl \
      --out data/repair_v6_synth_all_neg.jsonl
"""

import json
import hashlib
import random
import argparse
from pathlib import Path


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def merge_files(file1, file2, out_file):
    print(f"[INFO] Loading: {file1}")
    data1 = load_jsonl(file1)
    print(f"[INFO] Loaded {len(data1)} from ADV_V7")

    print(f"[INFO] Loading: {file2}")
    data2 = load_jsonl(file2)
    print(f"[INFO] Loaded {len(data2)} from V1")

    all_data = data1 + data2
    print(f"[INFO] Combined count before dedupe: {len(all_data)}")

    dedupe_set = set()
    merged = []

    for rec in all_data:
        cot_text = rec.get("input_text", "")
        h = sha256_text(cot_text)

        if h in dedupe_set:
            continue
        dedupe_set.add(h)
        merged.append(rec)

    print(f"[INFO] Deduped count: {len(merged)}")

    # Shuffle for distribution robustness
    random.shuffle(merged)

    # Save final output
    out_path = Path(out_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Wrote {len(merged)} merged negatives → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Merge synthetic negative datasets")
    p.add_argument("--file1", required=True, help="Path to first JSONL file (ADV_V7)")
    p.add_argument("--file2", required=True, help="Path to second JSONL file (V1)")
    p.add_argument("--out", required=True, help="Output merged JSONL")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_files(args.file1, args.file2, args.out)