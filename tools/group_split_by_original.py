#!/usr/bin/env python3
"""
Group-split ORM dataset by original_qid to avoid cross-split leakage.
"""

import json, random
from collections import defaultdict
from pathlib import Path

random.seed(12345)

IN_FILE = Path("data/orm_train_merged_v6.jsonl")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.90
VAL_RATIO   = 0.05
TEST_RATIO  = 0.05

# ----- STEP 1: group all records by original_qid -----
groups = defaultdict(list)

with IN_FILE.open() as fh:
    for line in fh:
        r = json.loads(line)
        orig = (
            r.get("meta",{}).get("original_qid") or
            r.get("meta",{}).get("gold_qid") or
            r["qid"]  # fallback if no original tracking
        )
        groups[orig].append(r)

orig_keys = list(groups.keys())
random.shuffle(orig_keys)

n = len(orig_keys)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

train_keys = orig_keys[:n_train]
val_keys   = orig_keys[n_train : n_train+n_val]
test_keys  = orig_keys[n_train+n_val :]

def write_split(keys, outpath):
    with outpath.open("w") as f:
        for k in keys:
            for r in groups[k]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

write_split(train_keys, OUT_DIR/"orm_train_new.jsonl")
write_split(val_keys,   OUT_DIR/"orm_val_new.jsonl")
write_split(test_keys,  OUT_DIR/"orm_test_new.jsonl")

print("Grouped split complete.")
print("Train groups:", len(train_keys))
print("Val groups:  ", len(val_keys))
print("Test groups: ", len(test_keys))