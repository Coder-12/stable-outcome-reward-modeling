#!/usr/bin/env python3
"""
Synthetic Negative Generator — Advanced V7 (ORM-Repair)

This module creates *high-quality, diverse, structurally sound* incorrect reasoning chains.
These synthetic negatives complement GPT-generated negatives for ORM-Repair V6 Hybrid training.

Capabilities:
- Multi-stage corruption pipeline
- Arithmetic corruption, numeric mutation, wrong final answer
- Reasoning step injection, contradiction steps, distractor steps
- Confidence corruption + PRM corruption
- Meta-logging of corruption ops for debugging
- Schema-preserving final output
"""

import json, random, uuid, math, re
from copy import deepcopy
from typing import List, Dict, Any

random.seed(777)

# ============================================================
# NUMERIC EXTRACTION UTILITIES
# ============================================================

NUM_RE = re.compile(r"-?\d+\.?\d*")

def extract_numbers(text: str):
    """Return list of floats found in text."""
    return [float(x) for x in NUM_RE.findall(text or "")]

def mutate_number(val: float):
    """Apply advanced numeric corruption: off-by-one, scaling, noisy offset."""
    ops = []

    # randomly choose corruption type
    p = random.random()
    if p < 0.3:
        ops.append("off_by_one")
        return val + random.choice([-1, 1]), ops

    if p < 0.6:
        ops.append("scale")
        scale = random.choice([0.5, 1.5, 2.0])
        return val * scale, ops

    ops.append("noise")
    noise = random.uniform(-5, 5)
    return val + noise, ops


def corrupt_numbers_in_text(text: str):
    """Replace one or more numbers with corrupted mutated versions."""
    nums = extract_numbers(text)
    if not nums:
        return text, []

    ops_log = []
    chosen = random.sample(nums, k=min(len(nums), random.randint(1, 2)))

    corrupted = {}
    for num in chosen:
        new_val, ops = mutate_number(num)
        corrupted[str(num)] = str(round(new_val, 3))
        ops_log.extend(ops)

    new_text = text
    for orig, new in corrupted.items():
        new_text = new_text.replace(orig, new)

    return new_text, ops_log


# ============================================================
# WRONG FINAL ANSWER GENERATION
# ============================================================

def generate_wrong_final_answer(correct_answer: str | float | int | None):
    """Produce an incorrect final numeric answer."""
    try:
        val = float(correct_answer)
    except:
        return "0"  # fallback wrong answer

    wrong = val + random.choice([-1, 1]) * random.uniform(1, 20)
    return str(round(wrong, 3))


# ============================================================
# STEP-LEVEL CORRUPTION
# ============================================================

def inject_wrong_step(step: str):
    """Produce a faulty reasoning step by mutating numbers in the step text."""
    corrupted, ops = corrupt_numbers_in_text(step)
    if corrupted == step:
        # If no number found, inject logical error
        corrupted = step + " (But mistakenly assumes an incorrect relation.)"
        ops.append("logic_error")
    return corrupted, ops


def inject_contradiction():
    """A contradictory reasoning step."""
    contradictions = [
        "However, we now incorrectly assume the total increases instead of decreases.",
        "Assume incorrectly that doubling the value reduces it by half.",
        "Here we misinterpret the subtraction as addition."
    ]
    return random.choice(contradictions), ["contradiction"]


def inject_distractor():
    distractors = [
        "Introduce an irrelevant variable that has no impact on the solution.",
        "Assume an extra step unrelated to the original problem.",
        "Consider an unnecessary conversion that changes the reasoning flow."
    ]
    return random.choice(distractors), ["distractor"]


def corrupt_steps(steps: List[str]):
    """Apply multi-stage corruptions to reasoning steps."""
    new_steps = []
    all_ops = []

    for s in steps:
        if random.random() < 0.55:
            corrupted, ops = inject_wrong_step(s)
        else:
            corrupted = s
            ops = []

        new_steps.append(corrupted)
        all_ops.extend(ops)

        # occasional contradiction steps
        if random.random() < 0.15:
            cstep, cops = inject_contradiction()
            new_steps.append(cstep)
            all_ops.extend(cops)

        # occasional distractor
        if random.random() < 0.10:
            dstep, dops = inject_distractor()
            new_steps.append(dstep)
            all_ops.extend(dops)

    # random truncation
    if random.random() < 0.20:
        keep = random.randint(max(1, len(new_steps)//2), len(new_steps))
        new_steps = new_steps[:keep]
        all_ops.append("truncate_steps")

    return new_steps, all_ops


# ============================================================
# CONFIDENCE & PRM CORRUPTION
# ============================================================

def corrupt_confidence(c: float):
    """Confidence scaling and noise."""
    scale = random.uniform(0.05, 0.35)
    noisy = c * scale
    return max(0.0, min(1.0, noisy))


def corrupt_prm_step_values(vals: List[float]):
    """Push PRM step targets toward incorrectness."""
    return [max(0.0, min(1.0, v * random.uniform(0.1, 0.5))) for v in vals]


# ============================================================
# MAIN SYNTH NEGATIVE MAKER
# ============================================================

def make_advanced_synthetic_negative(record: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a full synthetic negative in state-of-the-art ORM-Repair format."""
    r = deepcopy(record)
    ops_log = []

    # force negative (wrong chain)
    r["orm_label"] = 0
    r["label"] = 0
    r["synthetic_mode"] = "ADV_V7"

    # preserve qid but produce unique chain id
    r["chain_id"] = f"syn-{uuid.uuid4()}"

    # -----------------------------------------------------
    # CORRUPT PROMPT (light corruption)
    # -----------------------------------------------------
    prompt = r.get("prompt", "")
    corrupted_prompt, ops = corrupt_numbers_in_text(prompt)
    r["prompt"] = corrupted_prompt
    ops_log.extend(ops)

    # -----------------------------------------------------
    # CORRUPT STEPS
    # -----------------------------------------------------
    orig_steps = r.get("steps") or r.get("step_text") or []
    new_steps, step_ops = corrupt_steps(orig_steps)
    r["corrupted_reasoning_steps"] = new_steps
    ops_log.extend(step_ops)

    # -----------------------------------------------------
    # WRONG FINAL ANSWER
    # -----------------------------------------------------
    gold = r.get("meta", {}).get("gold_answer") or r.get("final_answer")
    r["final_answer"] = generate_wrong_final_answer(gold)
    ops_log.append("wrong_final_answer")

    # -----------------------------------------------------
    # PRM corruption
    # -----------------------------------------------------
    if "step_targets" in r and isinstance(r["step_targets"], list):
        r["step_targets"] = corrupt_prm_step_values(r["step_targets"])
        ops_log.append("corrupt_prm_steps")

    # -----------------------------------------------------
    # Confidence corruption
    # -----------------------------------------------------
    cc = r.get("chain_confidence", 1.0)
    r["chain_confidence"] = corrupt_confidence(cc)
    ops_log.append("corrupt_confidence")

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------
    r["corruption_ops"] = ops_log

    return r


# ============================================================
# BULK GENERATION
# ============================================================

def generate_from_file(input_jsonl, out_jsonl, ratio=1.0, max_per_file=None):
    out = []
    with open(input_jsonl, "r") as f:
        lines = f.readlines()

    random.shuffle(lines)
    count = 0

    for L in lines:
        if max_per_file and count >= max_per_file:
            break

        rec = json.loads(L)

        if random.random() < ratio:
            neg = make_advanced_synthetic_negative(rec)
            out.append(neg)
            count += 1

    with open(out_jsonl, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"[ADV_V7] wrote {len(out)} synthetic negatives → {out_jsonl}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--max", type=int, default=None)
    args = p.parse_args()

    generate_from_file(args.infile, args.outfile, ratio=args.ratio, max_per_file=args.max)