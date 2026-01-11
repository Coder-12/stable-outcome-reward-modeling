#!/usr/bin/env python3
import json
from pathlib import Path

BAD_QIDS = {"0be285f9-f0aa-4f54-9819-2f5641f83efe"}

inp = "data/repair_v6_synth_all_neg.jsonl"
out = "data/repair_v6_synth_all_neg_clean.jsonl"

with open(inp, "r") as fin, open(out, "w") as fout:
    removed = 0
    kept = 0
    for line in fin:
        line=line.strip()
        if not line:
            continue
        try:
            rec=json.loads(line)
        except:
            continue
        if rec.get("qid") in BAD_QIDS:
            removed += 1
            continue
        fout.write(json.dumps(rec, ensure_ascii=False)+"\n")
        kept += 1

print(f"Removed {removed}, kept {kept}, wrote → {out}")