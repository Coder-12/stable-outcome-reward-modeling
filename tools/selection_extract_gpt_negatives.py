#!/usr/bin/env python3
"""
Selective GPT-Negative Prompt Extractor V2
------------------------------------------

Goal: Extract ~500 high-impact prompts for GPT negative generation.

Features:
- Multi-phase sampling:
    Phase 1: Strong-match rules (strict regex)
    Phase 2: Medium-match rules (looser heuristics)
    Phase 3: Fallback reservoir sampling to meet quota
- Category quotas:
        A: 150
        B: 100
        C: 60
        D: 120
        E: 120
- Ensures total ~500 samples (auto-adjust)
- Avoid duplicates
- Deterministic with --seed
"""

import re, json, random
from collections import defaultdict
from pathlib import Path

# -----------------------
# CATEGORY TARGETS
# -----------------------
TARGETS = {
    "A": 150,
    "B": 100,
    "C": 60,
    "D": 120,
    "E": 120,
}
TOTAL_TARGET = sum(TARGETS.values())   # ~550, fallback will cut to ~500

# -----------------------
# STRONG MATCH RULES
# -----------------------
STRONG = {
    "A": [
        r"\d+\s*(plus|minus|\*|x|\-|added|subtracted|difference|product|sum)",
        r"what is.*\d+",
        r"how many.*\d+",
    ],
    "B": [
        r"probab",
        r"chance",
        r"likelihood",
        r"\d+%|p\s*=\s*\d+\.\d+",
    ],
    "C": [
        r"kilometers?|meters?|cm|km|liters?|grams?|kg|lbs|mph|km/h",
        r"convert",
        r"conversion",
    ],
    "D": [
        r"solve for|find x|equation|variable|simplify|quadratic",
    ],
    "E": [
        r"based on the information",
        r"from the passage",
        r"according to",
        r"Assume that",  # common reasoning phrase
    ],
}

# -----------------------
# MEDIUM MATCH RULES
# -----------------------
MEDIUM = {
    "A": [
        r"\d+.*\d+",            # any two numbers in prompt
    ],
    "B": [
        r"\d+%|\d+\.\d+",       # numbers that look like probabilities
    ],
    "C": [
        r"\b(cm|mm|km|kg|g|ml|l)\b",
    ],
    "D": [
        r"compute|evaluate|expression|term|value",
    ],
    "E": [
        r"\. .+?\.",  # two sentences = reasoning-like
    ],
}

def matches_any(text, patterns):
    return any(re.search(p, text.lower()) for p in patterns)


def categorize_prompt(prompt):
    """Classify prompt into zero or more categories."""
    cats = []
    p = prompt.lower()
    # strong matches first
    for c, rules in STRONG.items():
        if matches_any(p, rules):
            cats.append((c, "strong"))
    # medium matches
    for c, rules in MEDIUM.items():
        if (c, "strong") not in cats and matches_any(p, rules):
            cats.append((c, "medium"))
    return cats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    random.seed(args.seed)

    # Load input JSONL
    data = []
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"[INFO] Loaded {len(data)} records")

    # Storage
    selected = defaultdict(list)
    strong_pool = defaultdict(list)
    medium_pool = defaultdict(list)

    # Categorize
    for rec in data:
        prompt = rec.get("prompt") or rec.get("input_text") or ""
        cats = categorize_prompt(prompt)

        for c, lvl in cats:
            if lvl == "strong":
                strong_pool[c].append(rec)
            else:
                medium_pool[c].append(rec)

    # Phase 1: Strong matches
    for c in TARGETS:
        want = TARGETS[c]
        pool = strong_pool[c]
        if len(pool) > want:
            selected[c].extend(random.sample(pool, want))
        else:
            selected[c].extend(pool)

    # Phase 2: Medium matches (fill gaps)
    for c in TARGETS:
        need = TARGETS[c] - len(selected[c])
        if need <= 0:
            continue
        pool = medium_pool[c]
        if len(pool) > need:
            selected[c].extend(random.sample(pool, need))
        else:
            selected[c].extend(pool)

    # Phase 3: Fallback fill to reach ~500
    flat = [rec for recs in selected.values() for rec in recs]
    used_ids = set(id(x) for x in flat)

    remaining = [r for r in data if id(r) not in used_ids]
    random.shuffle(remaining)

    total_current = len(flat)
    need_total = max(500, total_current)   # ensure minimum ~500

    if total_current < need_total:
        fill = remaining[: need_total - total_current]
        flat.extend(fill)

    # Deduplicate by qid
    final = []
    seen = set()
    for rec in flat:
        qid = rec.get("qid") or rec.get("id") or json.dumps(rec, sort_keys=True)
        if qid not in seen:
            final.append(rec)
            seen.add(qid)

    print(f"[DONE] Final selected samples: {len(final)}")

    # Save
    with open(args.output, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")

    print(f"[SAVED] → {args.output}")


if __name__ == "__main__":
    main()