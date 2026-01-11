#!/usr/bin/env python3
"""
split_orm_dataset.py

Splits orm_train_merged_v6.jsonl into:
 - train (90%)
 - val   (5%)
 - test  (5%)

Stratified by label (1/0).
"""

import json
import random
from pathlib import Path

INPUT = Path("data/orm_train_merged_v6.jsonl")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.90
VAL_RATIO   = 0.05
TEST_RATIO  = 0.05

def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]

def write_jsonl(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    data = load_jsonl(INPUT)
    print(f"Loaded {len(data)} records.")

    # stratify by label
    pos = [r for r in data if r["label"] == 1]
    neg = [r for r in data if r["label"] == 0]

    random.shuffle(pos)
    random.shuffle(neg)

    def split_list(lst):
        n = len(lst)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        return (
            lst[:n_train],
            lst[n_train:n_train+n_val],
            lst[n_train+n_val:]
        )

    pos_train, pos_val, pos_test = split_list(pos)
    neg_train, neg_val, neg_test = split_list(neg)

    train = pos_train + neg_train
    val   = pos_val + neg_val
    test  = pos_test + neg_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print("Final split sizes:")
    print("Train:", len(train))
    print("Val:  ", len(val))
    print("Test: ", len(test))

    write_jsonl(OUT_DIR/"orm_train.jsonl", train)
    write_jsonl(OUT_DIR/"orm_val.jsonl", val)
    write_jsonl(OUT_DIR/"orm_test.jsonl", test)

    print(f"Saved splits under {OUT_DIR}/")

if __name__ == "__main__":
    main()