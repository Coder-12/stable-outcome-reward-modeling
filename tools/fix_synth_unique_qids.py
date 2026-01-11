#!/usr/bin/env python3
"""
fix_synth_unique_qids.py

Goal:
- Preserve ALL 4413 synthetic negatives
- Assign a NEW unique qid for every row (UUIDv4)
- Assign a NEW chain_id (UUIDv4)
- Keep ALL fields identical (steps, confs, label, reasoning)
- Produce zero duplicates → ORM-stable dataset

Usage:
python tools/fix_synth_unique_qids.py \
  --in_file data/repair_v6_synth_all_neg.jsonl \
  --out_file data/repair_v6_synth_all_neg_unique.jsonl
"""

import json, uuid, argparse
from pathlib import Path
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    in_p = Path(args.in_file)
    out_p = Path(args.out_file)

    logger.info(f"Loading: {in_p}")
    lines = [l.strip() for l in in_p.open("r", encoding="utf-8") if l.strip()]
    records = [json.loads(l) for l in lines]
    n = len(records)
    logger.info(f"Loaded {n} synthetic negatives")

    out_lines = []
    for rec in records:
        # assign NEW qid
        rec["qid"] = str(uuid.uuid4())

        # assign NEW chain_id
        rec["chain_id"] = f"synv1-{uuid.uuid4()}"

        out_lines.append(json.dumps(rec, ensure_ascii=False))

    out_p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    logger.success(f"[DONE] Wrote {len(out_lines)} → {out_p}")
    logger.success("All qids and chain_ids regenerated. ORM-safe dataset.")


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))
    main()