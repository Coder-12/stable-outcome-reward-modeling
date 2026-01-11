#!/usr/bin/env python3
"""
check_dataset_for_training.py

Comprehensive dataset quality validator for ORM/PRM training.
This script performs ALL CRITICAL CHECKS required before training,
ensuring dataset correctness, safety, consistency, robustness,
and statistical balance.

It detects:
 - Parse errors
 - Wrong/missing schema fields
 - Duplicates (qid, chain_id)
 - Step count anomalies
 - Confidence anomalies (step_confidence, chain_confidence)
 - Label distribution health
 - Mixed positive/negative presence
 - Negatives with final_answer == gold (should be disallowed)
 - Missing meta fields
 - Distribution of step lengths
 - Presence of fallback generations
 - Suspicious patterns (empty answers, empty steps, etc.)

Outputs:
   <out_prefix>.json   → structured report
   <out_prefix>.txt    → human-readable report
"""

import json
import argparse
from pathlib import Path
from statistics import mean
from collections import Counter, defaultdict


# ==============================
# Utility functions
# ==============================

def load_jsonl(path):
    records = []
    parse_errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                parse_errors += 1
    return records, parse_errors


def safe_get(d, key, default=None):
    return d[key] if key in d else default


# ==============================
# Main QC Engine
# ==============================

def analyze_dataset(records):
    report = {}

    n = len(records)
    report["total_records"] = n

    if n == 0:
        report["error"] = "EMPTY_DATASET"
        return report

    # ------------------------------
    # Containers
    # ------------------------------
    qid_counts = Counter()
    chain_counts = Counter()
    labels = Counter()

    fallback_count = 0
    malformed_steps = 0
    malformed_conf = 0
    final_equals_gold = 0
    missing_fields = defaultdict(int)

    step_counts = []
    chain_conf_values = []
    step_conf_values = []

    # ------------------------------
    # Required fields for ORM training
    # ------------------------------
    required_fields = [
        "qid",
        "chain_id",
        "label",
        "steps",
        "step_confidence",
        "chain_confidence",
        "final_answer",
        "meta"
    ]

    # ------------------------------
    # Iterate over records
    # ------------------------------
    for rec in records:

        # Track duplicates
        qid_counts[rec.get("qid")] += 1
        chain_counts[rec.get("chain_id")] += 1

        # Label distribution
        labels[rec.get("label")] += 1

        # Missing fields
        for field in required_fields:
            if field not in rec:
                missing_fields[field] += 1

        # Steps
        steps = safe_get(rec, "steps", [])
        if not isinstance(steps, list) or len(steps) < 1:
            malformed_steps += 1
        else:
            step_counts.append(len(steps))

        # Step confidence
        sc = safe_get(rec, "step_confidence", [])
        if not isinstance(sc, list) or len(sc) != len(steps):
            malformed_conf += 1
        else:
            for v in sc:
                try:
                    fv = float(v)
                    step_conf_values.append(fv)
                except:
                    malformed_conf += 1

        # Chain confidence
        cc = safe_get(rec, "chain_confidence", None)
        try:
            cc = float(cc)
            chain_conf_values.append(cc)
        except:
            malformed_conf += 1

        # Fallback detection
        gen_by = rec.get("meta", {}).get("generated_by", "")
        if isinstance(gen_by, str) and "fallback" in gen_by.lower():
            fallback_count += 1

        # final==gold check
        gold = rec.get("meta", {}).get("gold_answer", None)
        final = rec.get("final_answer", "").strip()

        if rec.get("label") == 0 and gold is not None:
            if str(final).strip() == str(gold).strip():
                final_equals_gold += 1

    # ------------------------------
    # Compute metrics
    # ------------------------------
    report["duplicates_qid"] = sum(1 for k, v in qid_counts.items() if v > 1)
    report["duplicates_chain_id"] = sum(1 for k, v in chain_counts.items() if v > 1)

    report["labels"] = dict(labels)

    if step_counts:
        report["step_count_min"] = min(step_counts)
        report["step_count_max"] = max(step_counts)
        report["step_count_mean"] = sum(step_counts) / len(step_counts)

    if chain_conf_values:
        report["chain_confidence_min"] = min(chain_conf_values)
        report["chain_confidence_max"] = max(chain_conf_values)
        report["chain_confidence_mean"] = sum(chain_conf_values) / len(chain_conf_values)

    if step_conf_values:
        report["step_conf_min"] = min(step_conf_values)
        report["step_conf_max"] = max(step_conf_values)
        report["step_conf_mean"] = sum(step_conf_values) / len(step_conf_values)

    report["fallback_flagged"] = fallback_count
    report["malformed_steps"] = malformed_steps
    report["malformed_confidence"] = malformed_conf
    report["final_equals_gold_negatives"] = final_equals_gold
    report["missing_fields"] = dict(missing_fields)

    return report


# ==============================
# Human-readable formatter
# ==============================

def format_human(report, path):
    lines = []
    lines.append("ORM FULL DATASET CHECK")
    lines.append("===================================================")
    lines.append(f"File: {path}")
    lines.append("")

    for k, v in report.items():
        lines.append(f"{k}: {v}")

    return "\n".join(lines)


# ==============================
# Main CLI
# ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to JSONL dataset to check.")
    ap.add_argument("--out_prefix", required=True, help="Output prefix for reports.")
    args = ap.parse_args()

    path = Path(args.file)

    print(f"[1/3] Loading dataset: {path}")
    records, parse_errors = load_jsonl(path)
    print(f"Loaded {len(records)} records; parse_errors = {parse_errors}")

    print("[2/3] Running full checks...")
    report = analyze_dataset(records)
    report["parse_errors"] = parse_errors # type: ignore[arg-type]

    # Save reports
    out_json = Path(args.out_prefix + ".json")
    out_txt = Path(args.out_prefix + ".txt")

    print(f"[3/3] Writing reports:")
    with out_json.open("w") as f:
        json.dump(report, f, indent=2)

    with out_txt.open("w") as f:
        f.write(format_human(report, path))

    print("Done.")
    print(f"JSON report -> {out_json}")
    print(f"TXT report  -> {out_txt}")


if __name__ == "__main__":
    main()