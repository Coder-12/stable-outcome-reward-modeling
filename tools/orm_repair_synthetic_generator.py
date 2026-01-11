#!/usr/bin/env python3
"""
ORM-Repair Synthetic Generator V1 — Full CoT Synthetic Negative Generator
=====================================================================
Generates independent, LLM-like incorrect reasoning chains (4-8 steps) that
are schema-compatible with your ORM-Repair training pipeline (same fields
as GPT negatives). This is the "V1" generator (complement to ADV_V7
corruption engine) — it *creates* fresh, plausible wrong chains from templates
and stochastic perturbations without calling any external API.

Key features:
- Generates N negatives (defaults to input file size or explicit --count)
- Many diverse templates (numeric, logic, probability, unit conversion,
  algebraic, commonsense) to mimic LLM-style reasoning
- Ensures 4-8 steps, final_answer != gold (if gold present), confidences
  in requested range [0.0,0.4]
- Produces training-ready records (JSONL) matching your ORM-Repair schema
- CLI options: input file (optional), count, seed, out path, verbosity,
  max attempts per record when trying to avoid final==gold

Notes on integration:
- Output fields match the RM-friendly format used in your pipeline.
- If you provide an input JSONL containing gold answers and qids, the
  generator will use them to craft context-aware wrong chains; otherwise
  it will produce standalone negatives with placeholder gold.

No external dependencies besides stdlib.
"""

from __future__ import annotations
import argparse
import json
import math
import random
import uuid
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------- Helpers ---------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

NUM_RE = re.compile(r"-?\d+\.?\d*")

# Safe float -> string formatting for final answers
def fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(round(x, 3))

# ----------------------------- Templates -------------------------------
# Each template returns (steps:list[str], final_answer:str, extra_meta:dict)

def template_basic_arithmetic(context: Dict[str, Any]) -> Tuple[List[str], str, Dict[str, Any]]:
    # context may include 'gold' as numeric
    base = random.randint(5, 80)
    add = random.randint(1, 40)
    mult = random.choice([2,3,4,5])
    steps = [
        f"Start with the number {base}.",
        f"Add {add} to get {base + add}.",
        f"Multiply the result by {mult} to get { (base + add) * mult }.",
        "(Mis)apply a division by an unrelated constant 2.",
    ]
    # intentionally wrong final answer
    final = fmt_num(((base + add) * mult) / 2)
    return steps, final, {"template":"basic_arithmetic"}


def template_fraction_mistake(context: Dict[str, Any]) -> Tuple[List[str], str, Dict[str, Any]]:
    a = random.randint(2, 9)
    b = random.randint(2, 9)
    steps = [
        f"We take fraction {a}/{b}.",
        "We mistakenly invert the fraction when combining terms.",
        "We add the inverted fraction to itself.",
        "We simplify incorrectly by cancelling unrelated factors.",
    ]
    # wrong final
    final = fmt_num((a/b) + 1)
    return steps, final, {"template":"fraction_mistake"}


def template_unit_conversion_error(context: Dict[str, Any]) -> Tuple[List[str], str, Dict[str, Any]]:
    val = random.randint(1, 100)
    steps = [
        f"We start with {val} meters.",
        "Mistakenly treat meters as centimeters (x100).",
        "Convert the number to kilometers but forget to divide properly.",
        "Round to nearest integer and provide final value.",
    ]
    final = fmt_num(val * 100 / 1000 + random.uniform(-5,5))
    return steps, final, {"template":"unit_conversion_error"}


def template_logic_false_assumption(context: Dict[str, Any]) -> Tuple[List[str], str, Dict[str, Any]]:
    # Non-numeric, commonsense-like
    entity = random.choice(["train schedule","weather forecast","population"])
    steps = [
        f"Assume that the {entity} always grows linearly.",
        "Therefore doubling the time doubles the value.",
        "Ignore seasonal variations and round the result.",
        "Conclude with the projected incorrect value.",
    ]
    final = str(random.randint(10, 500))
    return steps, final, {"template":"logic_false_assumption", "entity": entity}


def template_probability_error(context: Dict[str, Any]) -> Tuple[List[str], str, Dict[str, Any]]:
    p = random.uniform(0.05, 0.95)
    # intentionally wrong composition
    steps = [
        f"Single event probability is {round(p,2)}.",
        "Assume independence incorrectly and multiply probabilities.",
        "Misapply complement rule incorrectly.",
        "Give a final probability as percentage.",
    ]
    final = fmt_num(max(0.0, min(1.0, p * p * random.uniform(0.5,1.5))))
    return steps, final, {"template":"probability_error"}

TEMPLATES = [
    template_basic_arithmetic,
    template_fraction_mistake,
    template_unit_conversion_error,
    template_logic_false_assumption,
    template_probability_error,
]

# --------------------------- Generation Utils --------------------------

def perturb_step_text(step: str) -> str:
    # small perturbations: inject distractor, numeric noise, or phrase
    r = random.random()
    if r < 0.25:
        # numeric tweak
        def repl(m):
            v = float(m.group(0))
            v2 = v + random.choice([-1,1]) * random.randint(1,5)
            return fmt_num(v2)
        return NUM_RE.sub(repl, step)
    elif r < 0.6:
        # inject distractor phrase
        return step + " However, assume an unrelated constant mistakenly."
    else:
        return step


def make_confidences(n_steps: int) -> Tuple[float, List[float]]:
    # chain_conf between 0.01 and 0.35 (prefer low)
    chain_conf = round(random.uniform(0.01, 0.30), 3)
    step_conf = [ round(random.uniform(0.01, 0.35), 3) for _ in range(n_steps) ]
    # ensure same length
    return chain_conf, step_conf


def make_step_targets(n_steps: int) -> List[float]:
    return [ round(random.random() * 0.25, 3) for _ in range(n_steps) ]

# -------------------------- Main generator --------------------------------

def generate_single_negative(qid: Optional[str], gold: Optional[str], seed: Optional[int]=None, max_attempts: int=5) -> Dict[str, Any]:
    # choose template
    template = random.choice(TEMPLATES)
    steps, final, meta = template({"gold": gold})

    # ensure between 4-8 steps: if fewer, add perturbations; if more, truncate
    if len(steps) < 4:
        # duplicate + perturb
        while len(steps) < 4:
            s = random.choice(steps)
            steps.append(perturb_step_text(s))
    if len(steps) > 8:
        steps = steps[:8]

    # apply small perturb to steps
    steps = [ perturb_step_text(s) for s in steps ]

    # ensure final != gold (if gold provided), otherwise it's fine
    attempt = 0
    final_ans = str(final).strip()
    while gold is not None and str(gold).strip() != "" and final_ans == str(gold).strip() and attempt < max_attempts:
        # regenerate perturbation
        steps = [ perturb_step_text(s) for s in steps ]
        final_ans = str(final) + str(random.randint(1,9))
        attempt += 1

    # confidences and prm
    chain_conf, step_conf = make_confidences(len(steps))
    step_targets = make_step_targets(len(steps))

    cot_text = "\n".join(f"{i+1}. {s}" for i,s in enumerate(steps)) + f"\n\nFinal Answer: {final_ans}"

    rec = {
        "qid": qid or str(uuid.uuid4()),
        "chain_id": f"synv1-{uuid.uuid4()}",
        "label": 0,
        "orm_label": 0,
        "input_text": cot_text,
        "prompt": cot_text,
        "steps": steps,
        "step_targets": step_targets,
        "step_mask": [1]*len(steps),
        "step_confidence": [float(x) for x in step_conf],
        "chain_confidence": float(chain_conf),
        "final_answer": final_ans,
        "prm_solution_gold": round(random.random()*0.25, 3),
        "meta": {"gold_answer": gold, "generated_by": "ORM-Repair-Synth-V1", **meta},
        "created_at": now_iso()
    }
    return rec

# -------------------------- Bulk generation --------------------------------

def generate_bulk_from_input(in_path: Optional[str], out_path: str, count: Optional[int], seed: Optional[int]=None, verbose: bool=False) -> None:
    random.seed(seed or 777)

    inputs: List[Dict[str, Any]] = []
    if in_path:
        with open(in_path, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                try:
                    inputs.append(json.loads(line))
                except Exception:
                    # allow bare prompts (strings)
                    inputs.append({"prompt": line})

    n_in = len(inputs)
    if count is None:
        # default: if input provided -> generate same length; else 4679
        count = n_in if n_in>0 else 4679

    out = []
    i = 0
    inp_idx = 0
    attempts = 0
    while i < count:
        # pick input record if available
        if n_in>0:
            src = inputs[inp_idx % n_in]
            qid = src.get('qid') or src.get('id') or None
            gold = None
            if isinstance(src.get('meta'), dict) and 'gold_answer' in src.get('meta'):
                gold = src['meta']['gold_answer']
            elif 'gold_answer' in src:
                gold = src['gold_answer']
            elif 'final_answer' in src:
                gold = src['final_answer']
            inp_idx += 1
        else:
            qid = None
            gold = None

        rec = generate_single_negative(qid=qid, gold=gold)
        out.append(rec)
        i += 1
        if verbose and i%50==0:
            print(f"generated {i}/{count}")

    # write to out_path
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[V1] Wrote {len(out)} synthetic negatives → {out_path}")

# -------------------------- CLI ----------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='ORM-Repair Synthetic Generator V1')
    p.add_argument('--in_file', type=str, default=None, help='Optional input JSONL (to use qids and gold answers)')
    p.add_argument('--out_file', type=str, required=True)
    p.add_argument('--count', type=int, default=None, help='Number of negatives to generate (default = input size or 4679)')
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--verbose', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    generate_bulk_from_input(args.in_file, args.out_file, args.count, seed=args.seed, verbose=args.verbose)

if __name__ == '__main__':
    main()
