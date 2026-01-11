#!/usr/bin/env python3
"""
tools/json_repair.py

Production-grade JSON repair & parsing utilities for noisy LLM outputs.

Goal:
 - Accept raw string output from an LLM (often "JSON-like" but not valid JSON)
 - Attempt multiple robust repair strategies in sequence
 - Return a parsed dict and a repair report/log
 - Never raise: always return a dict (fallback) so upstream pipelines remain robust

Usage:
    from tools.json_repair import repair_and_parse
    parsed, report = repair_and_parse(raw_model_text)
"""

import json
import re
import html
import logging
from typing import Tuple, Optional, Any

logger = logging.getLogger("json_repair")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[json_repair] %(message)s"))
    logger.addHandler(ch)


# --- small helpers ---------------------------------------------------------
def strip_markdown_wrappers(text: str) -> str:
    """Remove leading/trailing triple-backtick blocks and language hints."""
    if not text:
        return text
    t = text.strip()
    # remove leading fences like ```json or ````
    # handle multiple blocks: take first {...} if present after strip
    # if text starts with code fence -> remove outermost fence
    if t.startswith("```"):
        # remove first fence
        parts = t.split("```", 2)
        # parts: ['', 'json\n{...}', '...'] or ['', '\n{...}', '']
        if len(parts) >= 2:
            inner = parts[1]
            # if inner starts with 'json' remove it
            if inner.lstrip().startswith("json"):
                inner = inner.split("\n", 1)[1] if "\n" in inner else ""
            return inner.strip()
    # also remove single-line fences like `{"a": 1}` (backticks)
    if t.startswith("`") and t.endswith("`") and len(t) > 2:
        return t.strip("`").strip()
    return t


# LaTeX removal patterns
_LATEX_INLINE = re.compile(r"\\\(.+?\\\)", flags=re.DOTALL)
_LATEX_DISPLAY = re.compile(r"\\\[.+?\\\]", flags=re.DOTALL)
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\{.*?\}", flags=re.DOTALL)  # e.g. \frac{a}{b} or \text{...}


# control char removal
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")


def remove_latex(text: str) -> str:
    """Strip or simplify common LaTeX fragments that break JSON parsing."""
    if not text:
        return text
    t = text
    # replace inline/display math with their inner content stripped of backslashes
    t = _LATEX_INLINE.sub(lambda m: re.sub(r"\\", "", m.group(0)), t)
    t = _LATEX_DISPLAY.sub(lambda m: re.sub(r"\\", "", m.group(0)), t)
    # simplify common commands like \frac{a}{b} -> a/b, \text{xyz} -> xyz
    def _cmd_rep(m):
        s = m.group(0)
        # remove leading backslash and braces content naive
        inner = re.sub(r"\\[a-zA-Z]+\{", "", s).rstrip("}")
        inner = inner.replace("}{", "/")
        inner = inner.replace("{", "").replace("}", "")
        inner = inner.replace("\\", "")
        return inner
    t = _LATEX_COMMAND.sub(_cmd_rep, t)
    return t


def escape_backslashes_and_control(text: str) -> str:
    """Escape backslashes and remove control characters - done before json.loads attempt."""
    if not text:
        return text
    t = text
    # remove dangerous control characters
    t = _CONTROL_CHARS.sub(" ", t)
    # replace lone backslashes with double backslash (but don't double already double)
    # simple heuristic: replace single "\" occurrences that are not followed by another "\" with "\\"
    t = t.replace("\\\\", "\\\\\\\\")  # temporarily double existing doubles to avoid re-escaping twice
    t = t.replace("\\", "\\\\")
    # restore previously quadrupled sequences back to double
    t = t.replace("\\\\\\\\", "\\\\")
    return t


def fix_unescaped_newlines_in_strings(text: str) -> str:
    """
    Replace literal newlines inside JSON string values with escaped '\\n'.
    This is a heuristic: we find quoted substrings and escape newlines inside them.
    """
    if not text:
        return text

    def _esc_inner(match):
        content = match.group(0)  # including quotes
        # content like "...." (may contain newlines)
        inner = content[1:-1]
        inner_escaped = inner.replace("\n", "\\n").replace("\r", "\\r")
        # also escape unescaped quotes inside
        inner_escaped = inner_escaped.replace('"', '\\"')
        return f'"{inner_escaped}"'

    # regex to find double-quoted strings (including multiline)
    s = re.sub(r'"(?:\\.|[^"\\])*"', _esc_inner, text, flags=re.DOTALL)
    return s


def remove_trailing_commas(text: str) -> str:
    """Remove trailing commas in objects or arrays which break strict JSON parsers."""
    if not text:
        return text
    # object trailing commas: { "a": 1, } -> { "a": 1 }
    t = re.sub(r",\s*}", "}", text)
    t = re.sub(r",\s*]", "]", t)
    return t


def extract_balanced_braces(text: str) -> Optional[str]:
    """Find the first balanced {...} substring in text (greedy)."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    stack = []
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            stack.append(i)
        elif c == "}":
            stack.pop()
            if not stack:
                return text[start:i+1]
    return None


def try_load_json(txt: str) -> Optional[Any]:
    """Try a simple json.loads and return parsed object or None."""
    try:
        return json.loads(txt)
    except Exception:
        return None


# --- repair pipeline -------------------------------------------------------
def repair_and_parse(raw: str, verbose: bool = False) -> Tuple[dict, dict]:
    """
    Attempt to parse raw LLM text into a python dict.

    Returns:
        parsed (dict): parsed dict if successful, else a safe fallback dict.
        report (dict): information about which repair steps were attempted and status.
    """
    report = {"attempts": []}
    if raw is None:
        raw = ""

    orig = raw
    raw = strip_markdown_wrappers(raw)

    # 0) quick direct attempt
    report["orig_len"] = len(orig)
    if verbose:
        logger.info("Attempting direct json.loads()")
    parsed = try_load_json(raw)
    report["attempts"].append({"method": "direct", "ok": parsed is not None})
    if parsed is not None:
        return parsed, report

    # 1) remove LaTeX
    if verbose:
        logger.info("Removing LaTeX fragments and trying again")
    step1 = remove_latex(raw)
    parsed = try_load_json(step1)
    report["attempts"].append({"method": "remove_latex", "ok": parsed is not None})
    if parsed is not None:
        return parsed, report

    # 2) extract balanced braces
    if verbose:
        logger.info("Extracting balanced {...} block")
    blk = extract_balanced_braces(step1)
    if blk:
        parsed = try_load_json(blk)
        report["attempts"].append({"method": "extract_balanced", "ok": parsed is not None})
        if parsed is not None:
            return parsed, report
    else:
        report["attempts"].append({"method": "extract_balanced", "ok": False})

    # 3) escape backslashes + remove control chars
    if verbose:
        logger.info("Escaping backslashes and control chars")
    step3 = escape_backslashes_and_control(step1)
    parsed = try_load_json(step3)
    report["attempts"].append({"method": "escape_backslashes", "ok": parsed is not None})
    if parsed is not None:
        return parsed, report

    # If we had a block, try escaping it
    if blk:
        blk2 = escape_backslashes_and_control(blk)
        parsed = try_load_json(blk2)
        report["attempts"].append({"method": "escape_backslashes_on_block", "ok": parsed is not None})
        if parsed is not None:
            return parsed, report

    # 4) fix unescaped newlines in strings
    if verbose:
        logger.info("Fixing newlines inside quoted strings")
    step4 = fix_unescaped_newlines_in_strings(step3)
    parsed = try_load_json(step4)
    report["attempts"].append({"method": "fix_newlines_in_strings", "ok": parsed is not None})
    if parsed is not None:
        return parsed, report

    if blk:
        blk3 = fix_unescaped_newlines_in_strings(blk2 if 'blk2' in locals() else blk)
        parsed = try_load_json(blk3)
        report["attempts"].append({"method": "fix_newlines_on_block", "ok": parsed is not None})
        if parsed is not None:
            return parsed, report

    # 5) remove trailing commas
    step5 = remove_trailing_commas(step4)
    parsed = try_load_json(step5)
    report["attempts"].append({"method": "remove_trailing_commas", "ok": parsed is not None})
    if parsed is not None:
        return parsed, report

    # 6) try replacing single quotes with double quotes (naive but helps many cases)
    if verbose:
        logger.info("Replacing single quotes with double quotes (heuristic)")
    step6 = step5.replace("'", '"')
    parsed = try_load_json(step6)
    report["attempts"].append({"method": "single_to_double_quotes", "ok": parsed is not None})
    if parsed is not None:
        return parsed, report

    # 7) as last attempt, do HTML-unescape then try again
    if verbose:
        logger.info("HTML-unescaping then final attempt")
    step7 = html.unescape(step6)
    parsed = try_load_json(step7)
    report["attempts"].append({"method": "html_unescape", "ok": parsed is not None})
    if parsed is not None:
        return parsed, report

    # 8) If everything failed, return fallback safe dict with debug traces
    logger.warning("[json_repair] Could not parse output into JSON; returning fallback dict.")
    fallback = {
        "prompt": None,
        "reasoning_steps": [],
        "final_answer": None,
        "chain_confidence": 0.0,
        "raw": orig[:2000],
    }
    report["fallback"] = True
    return fallback, report


# Simple CLI for quick manual tests (run: python tools/json_repair.py)
if __name__ == "__main__":
    import sys
    sample = sys.stdin.read() if not sys.stdin.isatty() else None
    if sample:
        parsed, report = repair_and_parse(sample, verbose=True)
        print("PARSED:", parsed)
        print("REPORT:", report)
    else:
        print("Usage: pipe raw text into this script to test repair_and_parse().")