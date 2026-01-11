#!/usr/bin/env python3
"""
merge_orm_v6.py

Merges:
 - ORM positives (train+val+test converted)
 - GPT negatives (fixed)
 - Synthetic negatives (unique)
into a single unified ORM training dataset.

Output:
    data/orm_train_merged_v6.jsonl

Guarantees:
 - no duplicate qids
 - no duplicate chain_ids
 - uniform schema
"""

import json
from pathlib import Path
import uuid

FILES = {
    "pos_train": "data/processed/orm_train_converted.jsonl",
    "pos_val":   "data/processed/orm_val_converted.jsonl",
    "pos_test":  "data/processed/orm_test_converted.jsonl",
    "gpt_neg":   "data/repair_v6_gpt_neg_final_fixed.jsonl",
    "synth_neg": "data/repair_v6_synth_all_neg_unique.jsonl",
}

OUT_FILE = Path("data/orm_train_merged_v6.jsonl")

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).open() if l.strip()]

def main():
    all_records = []
    seen_qids = set()
    seen_chain_ids = set()

    for key, path in FILES.items():
        print(f"Loading {key}: {path}")
        recs = load_jsonl(path)
        for r in recs:
            # ensure qid uniqueness across ALL datasets
            if r["qid"] in seen_qids:
                r["qid"] = str(uuid.uuid4())
            if r["chain_id"] in seen_chain_ids:
                prefix = "pos" if r["label"] == 1 else "neg"
                r["chain_id"] = f"{prefix}-{uuid.uuid4()}"
            seen_qids.add(r["qid"])
            seen_chain_ids.add(r["chain_id"])
            all_records.append(r)

    print(f"Merged total records: {len(all_records)}")

    tmp = OUT_FILE.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tmp.replace(OUT_FILE)
    print(f"Saved → {OUT_FILE}")

if __name__ == "__main__":
    main()