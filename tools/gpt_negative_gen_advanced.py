#!/usr/bin/env python3
"""
gpt_negative_gen_v2_1_gpt.py

GPT-based hardened ORM negative generator (V2.1, arithmetic-corruption mode).

Features:
- Uses prepared input with _gpt_question (validate_and_prepare_for_gpt.py output).
- Calls OpenAI (OpenAI class) to generate JSON negatives.
- Enforces: 4-8 reasoning_steps, step_confidence & chain_confidence in [0.0,0.4], final_answer != gold.
- Arithmetic corruption: ensures at least one numeric mistake inside the reasoning steps.
- Robust parsing pipeline: json.loads -> extract JSON block -> repair_and_parse fallback.
- Optional LLM-based repair (--enable_llm_repair).
- Dry-run (--dry_run) writes placeholders without spending credits.
- Preserves count: writes exactly N outputs (fallback generated only for failing items).
- Chunked flushing + atomic replace.
"""

from __future__ import annotations
import os
import sys
import json
import time
import random
import uuid
import re
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
from loguru import logger

# Use user's repair tool if available
try:
    from tools.json_repair import repair_and_parse
except Exception:
    def repair_and_parse(raw: str, verbose: bool = False):
        try:
            return json.loads(raw), {"method": "direct", "ok": True}
        except Exception as e:
            return {}, {"fallback": True, "error": str(e)}

# Try to import OpenAI client wrapper (new SDK exposes OpenAI)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------- Config (tweakable) ----------------
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMPERATURE = 0.2
DEFAULT_DOUBLE_SAMPLE = 1
DEFAULT_ATTEMPTS = 3
DEFAULT_USE_RESPONSE_FORMAT = True
DEFAULT_TRUNCATE_GOLD = 240
DEFAULT_CHUNK_SIZE = 20
# numeric corruption delta choices (relative or absolute)
NUMERIC_DELTAS = ("+1", "-1", "*2", "/2", "+7", "-7")  # deterministic-ish corruption choices

# small compact schema used in prompt (one-line)
_COMPACT_SCHEMA = '{"prompt":"","reasoning_steps":[],"final_answer":"","chain_confidence":0.0,"step_confidence":[]}'

# regexes
JSON_BLOCK_RE = re.compile(r"json\s*(\{.*?\})\s*", re.DOTALL | re.IGNORECASE)
BRACE_BLOCK_RE = re.compile(r"(\{(?:[^{}]|\{[^{}]*\})*\})", re.DOTALL)
NUMBER_RE = re.compile(r"(?<!\w)(\d+(?:\.\d+)?)(?!\w)")

# ---------------- client holder ----------------
_worker_client = None

def _init_client(api_key: str):
    global _worker_client
    if OpenAI is None:
        logger.warning("OpenAI client not available (OpenAI package missing). GPT calls will fail if attempted.")
        _worker_client = None
        return
    _worker_client = OpenAI(api_key=api_key)
    logger.debug("OpenAI client initialized")

# ---------------- helpers ----------------
def sha256_of_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def safe_json_dumps(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False)

def extract_json_block(raw: str) -> Optional[str]:
    if not raw:
        return None
    m = JSON_BLOCK_RE.search(raw)
    if m:
        return m.group(1)
    m2 = BRACE_BLOCK_RE.search(raw)
    if m2:
        return m2.group(1)
    return None

def robust_parse(raw: str, repair_verbose: bool = False):
    """
    Try direct json parse -> extract json block -> repair_and_parse(raw)
    Returns (parsed_dict_or_empty, report)
    """
    if not raw:
        return {}, {"method":"empty_raw"}
    # direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, {"method":"direct", "ok": True}
    except Exception:
        pass
    # extract block
    blk = extract_json_block(raw)
    if blk:
        try:
            parsed = json.loads(blk)
            if isinstance(parsed, dict):
                return parsed, {"method":"extracted_block", "ok": True}
        except Exception as e:
            p, r = repair_and_parse(blk, verbose=repair_verbose)
            if isinstance(p, dict) and p:
                return p, {"method":"block_repair", **r}
            return {}, {"method":"block_failed", "err": str(e)}
    # last resort: repair full raw
    p, r = repair_and_parse(raw, verbose=repair_verbose)
    if isinstance(p, dict) and p:
        return p, {"method":"repair_full", **r}
    return {}, {"method":"failed_all"}

# deterministic wrong final answer helper
def get_wrong_final_answer(gold: str) -> str:
    gold_s = "" if gold is None else str(gold).strip()
    # try numeric
    try:
        g = float(gold_s)
        # deterministic offset based on digits/hash
        offset = 7.0 if abs(g) < 1e6 else 123.0
        wrong = g + offset
        if wrong.is_integer():
            return str(int(wrong))
        return str(wrong)
    except Exception:
        # if not numeric, return a fixed wrong token
        return "WRONG_ANS_999"

def ensure_final_differs(final: str, gold: str) -> str:
    if gold is None:
        return final if final else get_wrong_final_answer("")
    if str(final).strip() == str(gold).strip() or final == "":
        return get_wrong_final_answer(gold)
    return final

# Arithmetic corruption: find a numeric token in steps and corrupt one deterministically
def corrupt_one_numeric_in_steps(steps: List[str], qid: str) -> List[str]:
    """
    Choose one step to corrupt which contains a numeric token.
    If none contains a number, append an incorrect arithmetic line to the chain (to ensure at least one error).
    Corruption is deterministic per qid (but varied) using hash.
    """
    if not steps:
        return steps
    seed = int(hashlib.sha256(qid.encode("utf-8")).hexdigest(), 16) % (2**31)
    rnd = random.Random(seed)
    # collect indices with numbers
    indexed = []
    for i, s in enumerate(steps):
        if NUMBER_RE.search(s):
            indexed.append(i)
    if indexed:
        idx = rnd.choice(indexed)
        s = steps[idx]
        # pick the first number occurrence
        def repl(m):
            num_str = m.group(1)
            try:
                if "." in num_str:
                    val = float(num_str)
                else:
                    val = int(num_str)
            except Exception:
                return num_str
            # pick delta deterministically
            choice = rnd.choice(NUMERIC_DELTAS)
            if choice.startswith("+"):
                new = val + int(choice[1:])
            elif choice.startswith("-"):
                new = val - int(choice[1:])
            elif choice.startswith("*"):
                new = val * int(choice[1:])
            elif choice.startswith("/"):
                # avoid zero
                new = val / int(choice[1:]) if int(choice[1:]) != 0 else val
            else:
                new = val + 1
            # format: if original int -> int
            if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
                try:
                    return str(int(new))
                except Exception:
                    return str(new)
            return str(new)
        # replace only first numeric token
        new_s = NUMBER_RE.sub(repl, s, count=1)
        steps[idx] = new_s + "  (INTENTIONAL_ARITH_ERROR)"
        return steps
    else:
        # no numeric found: append an incorrect arithmetic step
        added = "INTENTIONAL_ARITH_ERROR: 5 + 7 = 3 (incorrect arithmetic injected)"
        steps.append(added)
        return steps

# compact prompt builder (cost-aware)
def compact_neg_prompt(question: str, gold: str, truncate_gold: int = DEFAULT_TRUNCATE_GOLD) -> str:
    if gold is None:
        gold = ""
    g = str(gold).strip()
    if len(g) > truncate_gold:
        g = g[:truncate_gold].rsplit(" ", 1)[0] + "..."
    inst = (
        "Return exactly ONE valid JSON object following this one-line schema:\n"
        f"{_COMPACT_SCHEMA}\n\n"
        "Constraints: reasoning_steps: 4-8 short lines; final_answer MUST BE INCORRECT (not equal to gold); "
        "chain_confidence: 0.0-0.4; step_confidence: same length as reasoning_steps, each 0.0-0.4. "
        "Do NOT output any text outside the JSON.\n\n"
    )
    return inst + "QUESTION:\n" + question.strip() + "\n\nCORRECT_ANSWER (do not use):\n" + g

# model call (with retries/backoff)
def call_model(prompt_text: str, model: str, max_tokens: int = DEFAULT_MAX_TOKENS,
               temperature: float = DEFAULT_TEMPERATURE, attempts: int = DEFAULT_ATTEMPTS,
               use_response_format: bool = DEFAULT_USE_RESPONSE_FORMAT):
    global _worker_client
    if _worker_client is None:
        raise RuntimeError("OpenAI client not initialized")
    last_exc = None
    base_backoff = 0.4
    for attempt in range(attempts):
        try:
            messages = [
                {"role": "system", "content": "Output EXACTLY one JSON object only. No extra text."},
                {"role": "user", "content": prompt_text}
            ]
            # prefer response_format if the installed client supports it
            if use_response_format:
                resp = _worker_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    temperature=temperature
                )
                raw = resp.choices[0].message.content.strip()
            else:
                resp = _worker_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature
                )
                raw = resp.choices[0].message.content.strip()
            return raw
        except Exception as e:
            last_exc = e
            serr = str(e).lower()
            # rate limit / tokens / TPM
            if "rate limit" in serr or "429" in serr or "tpm" in serr:
                sleep = base_backoff * (2 ** attempt) + random.random() * 0.12
                logger.debug(f"Rate limit — sleeping {sleep:.2f}s (attempt {attempt}) err={str(e)[:200]}")
                time.sleep(sleep)
                continue
            # transient network/timeouts
            if "timeout" in serr or "connection" in serr or "temporar" in serr:
                sleep = base_backoff * (1.5 ** attempt)
                logger.debug(f"Transient error — sleeping {sleep:.2f}s (attempt {attempt}) err={str(e)[:200]}")
                time.sleep(sleep)
                continue
            logger.debug(f"Non-retryable model error: {str(e)[:200]}")
            break
    raise last_exc

# validate parsed candidate meets strict requirements
def validate_candidate(parsed: Dict[str, Any], gold: str) -> (bool, List[str]):
    issues = []
    if not isinstance(parsed, dict):
        return False, ["not_dict"]
    rs = parsed.get("reasoning_steps")
    fa = parsed.get("final_answer", "")
    cc = parsed.get("chain_confidence", None)
    sc = parsed.get("step_confidence", None)

    if not isinstance(rs, list) or not (4 <= len(rs) <= 8):
        issues.append("bad_steps_count")
    if isinstance(fa, str) and fa.strip() == "":
        issues.append("empty_final")
    try:
        if cc is None:
            issues.append("missing_chain_conf")
        else:
            cf = float(cc)
            if not (0.0 <= cf <= 0.4):
                issues.append("chain_conf_out_of_range")
    except Exception:
        issues.append("chain_conf_not_float")
    if not isinstance(sc, list) or len(sc) != (len(rs) if isinstance(rs, list) else 0):
        issues.append("step_conf_length_mismatch")
    else:
        try:
            for v in sc:
                vv = float(v)
                if not (0.0 <= vv <= 0.4):
                    issues.append("step_conf_out_of_range")
                    break
        except Exception:
            issues.append("step_conf_not_float")
    if isinstance(fa, str) and isinstance(gold, str) and gold.strip() != "" and fa.strip() == gold.strip():
        issues.append("final_equals_gold")
    return (len(issues) == 0), issues

# build fallback synthetic negative (deterministic per qid)
def build_fallback_negative(rec: Dict[str, Any]):
    qid = rec.get("qid") or rec.get("id") or str(uuid.uuid4())
    question = rec.get("_gpt_question") or rec.get("prompt") or rec.get("input_text", "")
    fallback_steps = [
        "1. (Incorrect) Assume unrelated quantity.",
        "2. (Incorrect) Perform an unrelated operation on that quantity.",
        "3. (Incorrect) Apply a mismatched formula.",
        "4. (Incorrect) Produce a plausible wrong final computation."
    ]
    step_conf = [0.02] * len(fallback_steps)
    final_ans = get_wrong_final_answer(rec.get("meta", {}).get("gold_answer", ""))
    cot_text = "\n".join(fallback_steps) + f"\n\nFinal Answer: {final_ans}"
    rec_out = {
        "qid": qid,
        "chain_id": f"neg-{uuid.uuid4()}",
        "label": 0,
        "orm_label": 0,
        "input_text": cot_text,
        "prompt": cot_text,
        "steps": fallback_steps,
        "step_targets": [0.02 for _ in fallback_steps],
        "step_mask": [1 for _ in fallback_steps],
        "step_confidence": step_conf,
        "chain_confidence": 0.02,
        "final_answer": final_ans,
        "prm_solution_gold": 0.02,
        "meta": {"gold_answer": rec.get("meta", {}).get("gold_answer", ""), "generated_by": "fallback"},
        "created_at": datetime.now(UTC).isoformat()
    }
    return rec_out

# ---------------- per-record processing ----------------
def process_record(rec: Dict[str, Any],
                   model: str,
                   api_key: Optional[str],
                   double_sample: int = DEFAULT_DOUBLE_SAMPLE,
                   use_response_format: bool = DEFAULT_USE_RESPONSE_FORMAT,
                   enable_llm_repair: bool = False,
                   dry_run: bool = False) -> Dict[str, Any]:
    qid = rec.get("qid") or rec.get("id") or str(uuid.uuid4())
    question = rec.get("_gpt_question") or rec.get("meta", {}).get("gold_question") or extract_fallback_question(rec)
    gold = rec.get("meta", {}).get("gold_answer") or rec.get("meta", {}).get("final_answer") or ""
    if dry_run:
        # produce dry-run record (no API cost)
        return {"qid": qid, "status": "dry_run", "question": question, "gold": gold}

    prompt_text = compact_neg_prompt(question, gold)

    best_parsed = None
    best_report = None

    for attempt_idx in range(max(1, int(double_sample))):
        raw = ""
        try:
            raw = call_model(prompt_text, model=model, max_tokens=DEFAULT_MAX_TOKENS,
                             temperature=DEFAULT_TEMPERATURE, attempts=DEFAULT_ATTEMPTS,
                             use_response_format=use_response_format)
        except Exception as e:
            logger.debug(f"[qid {qid}] model call exception: {e}")
            raw = ""

        parsed, report = robust_parse(raw, repair_verbose=False)
        if isinstance(parsed, dict) and parsed:
            # alias fields
            if "steps" in parsed and "reasoning_steps" not in parsed:
                parsed["reasoning_steps"] = parsed.get("steps")
            if "answer" in parsed and "final_answer" not in parsed:
                parsed["final_answer"] = parsed.get("answer")

            # ensure step_conf is present
            if "step_confidence" not in parsed:
                # give a default low-confidence distribution if missing
                parsed["step_confidence"] = [0.15] * len(parsed.get("reasoning_steps", []))

            # enforce arithmetic corruption on parsed candidate: modify one step deterministically
            parsed_steps = parsed.get("reasoning_steps", [])
            if isinstance(parsed_steps, list):
                parsed_steps = corrupt_one_numeric_in_steps(list(parsed_steps), qid)
                parsed["reasoning_steps"] = parsed_steps

            # ensure final answer differs
            parsed["final_answer"] = ensure_final_differs(parsed.get("final_answer", ""), gold)

            valid, issues = validate_candidate(parsed, gold)
            if valid:
                best_parsed = parsed
                best_report = {"method": "direct_parse", **report}
                break
            else:
                logger.debug(f"[qid {qid}] parsed candidate invalid: {issues}; report={report}")

        # optional llm-based repair (costly)
        if enable_llm_repair and raw:
            repair_prompt = (
                "Repair the following output to be EXACTLY a single JSON object following this one-line schema:\n"
                f"{_COMPACT_SCHEMA}\n\nOutput ONLY the repaired JSON object.\n\nRAW:\n" + raw
            )
            try:
                repaired_raw = call_model(repair_prompt, model=model, max_tokens=DEFAULT_MAX_TOKENS,
                                          temperature=0.0, attempts=2, use_response_format=False)
            except Exception as e:
                repaired_raw = ""
            parsed2, report2 = robust_parse(repaired_raw, repair_verbose=False)
            if isinstance(parsed2, dict) and parsed2:
                # apply same post-processing (corruption + final diff)
                if "steps" in parsed2 and "reasoning_steps" not in parsed2:
                    parsed2["reasoning_steps"] = parsed2.get("steps")
                if "step_confidence" not in parsed2:
                    parsed2["step_confidence"] = [0.15] * len(parsed2.get("reasoning_steps", []))
                parsed2["reasoning_steps"] = corrupt_one_numeric_in_steps(list(parsed2.get("reasoning_steps", [])), qid)
                parsed2["final_answer"] = ensure_final_differs(parsed2.get("final_answer", ""), gold)
                valid2, issues2 = validate_candidate(parsed2, gold)
                if valid2:
                    best_parsed = parsed2
                    best_report = {"method": "llm_repair", **report2}
                    break
            logger.debug(f"[qid {qid}] llm repair attempt failed")

    # if still no valid parsed candidate: fallback (but keep count)
    if not best_parsed:
        logger.warning(f"[qid {qid}] falling back to synthetic negative")
        return build_fallback_negative(rec)

    # assemble RM-friendly record
    steps = best_parsed.get("reasoning_steps", [])
    final_ans = str(best_parsed.get("final_answer", "")).strip()
    step_conf = best_parsed.get("step_confidence", [0.02] * len(steps))
    chain_conf = float(best_parsed.get("chain_confidence", 0.0))

    cot_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) + f"\n\nFinal Answer: {final_ans}"
    record_out = {
        "qid": qid,
        "chain_id": f"neg-{uuid.uuid4()}",
        "label": 0,
        "orm_label": 0,
        "input_text": cot_text,
        "prompt": cot_text,
        "steps": steps,
        "step_targets": [max(0.0, min(1.0, random.random() * 0.25)) for _ in steps],
        "step_mask": [1] * len(steps),
        "step_confidence": [float(x) for x in step_conf],
        "chain_confidence": chain_conf,
        "final_answer": final_ans,
        "prm_solution_gold": float(max(0.0, min(1.0, random.random() * 0.25))),
        "meta": {"gold_answer": gold, "generated_by": f"ORM-Repair-GPT-V2.1 (arith_corrupt) ({model})", "parse_report": best_report},
        "created_at": datetime.now(UTC).isoformat()
    }
    return record_out

def extract_fallback_question(rec: Dict[str, Any]) -> str:
    # prefer _gpt_question then meta.gold_question then first meaningful line of prompt/input_text
    if "_gpt_question" in rec and rec["_gpt_question"]:
        return rec["_gpt_question"]
    mg = rec.get("meta", {}).get("gold_question")
    if mg:
        return mg
    prompt = rec.get("prompt") or rec.get("input_text") or ""
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    return lines[0][:800] if lines else prompt[:800]

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="gpt_negative_gen_v2_1_gpt.py — arithmetic-corruption hardened GPT negatives")
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--double_sample", type=int, default=DEFAULT_DOUBLE_SAMPLE)
    ap.add_argument("--use_response_format", action="store_true", default=DEFAULT_USE_RESPONSE_FORMAT)
    ap.add_argument("--enable_llm_repair", action="store_true", default=False)
    ap.add_argument("--dry_run", action="store_true", default=False)
    ap.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = ap.parse_args()

    in_p = Path(args.in_file)
    out_p = Path(args.out_file)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print("[ERROR] OPENAI_API_KEY required unless --dry_run")
        raise SystemExit(2)

    # init client (single-process to avoid TPM issues)
    _init_client(api_key)

    # read inputs
    lines = [l.strip() for l in in_p.open("r", encoding="utf-8") if l.strip()]
    records = [json.loads(l) for l in lines]
    total = len(records)
    logger.info(f"Loaded {total} records from {in_p}")

    tmp_out = out_p.with_suffix(out_p.suffix + ".tmp")
    written = 0
    with tmp_out.open("w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            try:
                rec_out = process_record(rec,
                                         model=args.model,
                                         api_key=api_key,
                                         double_sample=args.double_sample,
                                         use_response_format=args.use_response_format,
                                         enable_llm_repair=args.enable_llm_repair,
                                         dry_run=args.dry_run)
            except Exception as e:
                logger.exception(f"Error processing record qid={rec.get('qid')}: {e}")
                # produce a fallback record to preserve counts
                rec_out = build_fallback_negative(rec)
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            written += 1
            if written % args.chunk_size == 0:
                fout.flush()
                logger.info(f"Flushed {written}/{total} -> {tmp_out}")

    # atomic replace
    os.replace(tmp_out, out_p)
    logger.success(f"[DONE] Wrote {written}/{total} negatives -> {out_p}")

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", backtrace=False, diagnose=False)
    main()