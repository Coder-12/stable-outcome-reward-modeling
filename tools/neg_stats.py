#!/usr/bin/env python3
import json, sys
from collections import Counter
fn = sys.argv[1]
cnt = 0
step_counts = Counter()
chain_conf_stats = []
final_equals_gold = 0
with open(fn) as f:
    for L in f:
        L=L.strip()
        if not L: continue
        cnt += 1
        rec=json.loads(L)
        steps = rec.get("steps", [])
        step_counts[len(steps)] += 1
        chain_conf_stats.append(rec.get("chain_confidence", 0.0))
        if rec.get("meta",{}).get("gold_answer") and str(rec.get("final_answer","")).strip() == str(rec["meta"]["gold_answer"]).strip():
            final_equals_gold += 1
print("Total", cnt)
print("Step counts:", step_counts)
if chain_conf_stats:
    print("chain_conf range:", min(chain_conf_stats), max(chain_conf_stats))
print("final_equals_gold:", final_equals_gold)