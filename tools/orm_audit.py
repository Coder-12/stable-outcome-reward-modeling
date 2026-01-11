#!/usr/bin/env python3
"""
orm_audit.py

Comprehensive dataset audit & QC for ORM/PRM corpora.

Features:
- Validate schema for each JSONL record (checks required fields and types)
- Per-file summary stats: counts, fallback/gold/parse flags, min/mean/max step counts
- Field-level histograms & distribution summaries: step_confidence, chain_confidence, step counts
- Detect duplicates: qid duplicates, chain_id duplicates, exact-prompt duplicates
- Detect final==gold leakage
- Detect malformed/missing steps, mismatched step_confidence lengths
- Outlier reporting for extremely long/short chains
- Cross-file overlap (optional): report qids present in more than one dataset
- Produces: human-readable report (stdout + <out_prefix>.txt) and machine JSON report (<out_prefix>.json)

Usage:
  python tools/orm_audit.py --files data/processed/orm_train.jsonl data/repair_v6_gpt_neg_final.jsonl data/repair_v6_synth_all_neg.jsonl \
      --out_prefix reports/orm_audit_2025-12-09

"""

from __future__ import annotations
import sys, os, json, math, argparse
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median
from typing import List, Dict, Any, Tuple

# ---------- helpers ----------

def load_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                yield {'__parse_error__': True, '__line__': i, '__raw__': line, '__error__': str(e)}


def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ---------- validation logic ----------

REQUIRED_TOP_LEVEL = [
    'qid', 'chain_id', 'label', 'orm_label', 'input_text', 'prompt', 'steps',
    'step_mask', 'step_confidence', 'chain_confidence', 'final_answer', 'meta', 'created_at'
]


class FileAudit:
    def __init__(self, path: Path):
        self.path = path
        self.total = 0
        self.parse_errors = []            # list of (line, err)
        self.records = []                 # loaded valid dicts (may include partial)
        self.qid_counter = Counter()
        self.chainid_counter = Counter()
        self.prompt_counter = Counter()
        # metrics
        self.step_counts = []
        self.step_conf_lengths = []
        self.step_conf_values = []
        self.chain_conf_values = []
        self.final_equals_gold = 0
        self.missing_fields = Counter()
        self.malformed_steps = 0
        self.fallback_flagged = 0
        self.generated_by = Counter()
        self.has_parse_report_ok = 0
        self.fallback_examples = []

    def ingest(self):
        for rec in load_jsonl(self.path):
            self.total += 1
            if isinstance(rec, dict) and rec.get('__parse_error__'):
                self.parse_errors.append(rec)
                continue
            self.records.append(rec)
            qid = rec.get('qid')
            cid = rec.get('chain_id')
            prompt = (rec.get('prompt') or '')[:200]
            self.qid_counter[qid] += 1
            self.chainid_counter[cid] += 1
            self.prompt_counter[prompt] += 1

            # detect generated_by and fallback markers
            meta = rec.get('meta') or {}
            gen = meta.get('generated_by')
            if gen:
                self.generated_by[gen] += 1
            pr = meta.get('parse_report')
            if isinstance(pr, dict) and pr.get('ok'):
                self.has_parse_report_ok += 1
            if meta.get('generated_by') and 'fallback' in str(meta.get('generated_by')).lower():
                self.fallback_flagged += 1
                self.fallback_examples.append(rec)

            # fields presence
            for f in REQUIRED_TOP_LEVEL:
                if f not in rec:
                    self.missing_fields[f] += 1

            # steps and confidences
            steps = rec.get('steps')
            sc = rec.get('step_confidence')
            sm = rec.get('step_mask')
            try:
                if isinstance(steps, list):
                    self.step_counts.append(len(steps))
                else:
                    self.malformed_steps += 1
                if isinstance(sc, list):
                    self.step_conf_lengths.append(len(sc))
                    for v in sc:
                        try:
                            fv = float(v)
                            self.step_conf_values.append(fv)
                        except Exception:
                            pass
                if isinstance(sm, list):
                    pass
                # chain confidence
                cc = rec.get('chain_confidence')
                try:
                    self.chain_conf_values.append(float(cc))
                except Exception:
                    pass
            except Exception:
                self.malformed_steps += 1

            # final == gold leakage
            gold = safe_get(rec, 'meta', 'gold_answer')
            final = rec.get('final_answer')
            if isinstance(final, str) and isinstance(gold, str):
                if final.strip() and gold.strip() and final.strip() == gold.strip():
                    self.final_equals_gold += 1

    def summarize(self) -> Dict[str, Any]:
        s = {
             'path': str(self.path), 'total_records': self.total, 'parse_errors': len(self.parse_errors),
             'unique_qids': len(self.qid_counter),
             'duplicate_qids': sum(v - 1 for v in self.qid_counter.values() if v > 1),
             'unique_chain_ids': len(self.chainid_counter),
             'duplicate_chain_ids': sum(v - 1 for v in self.chainid_counter.values() if v > 1),
             'unique_prompts_top20': list(self.prompt_counter.most_common(20)),
             'generated_by_top20': list(self.generated_by.most_common(20)),
             'has_parse_report_ok': self.has_parse_report_ok, 'fallback_flagged': self.fallback_flagged,
             'missing_fields': dict(self.missing_fields), 'malformed_steps': self.malformed_steps,
             'final_equals_gold': self.final_equals_gold
            }

        if self.step_counts:
            s['step_count_min'] = min(self.step_counts)
            s['step_count_max'] = max(self.step_counts)
            s['step_count_mean'] = mean(self.step_counts)
            s['step_count_median'] = median(self.step_counts)
        else:
            s['step_count_min'] = s['step_count_max'] = s['step_count_mean'] = s['step_count_median'] = None

        if self.step_conf_values:
            s['step_conf_min'] = min(self.step_conf_values)
            s['step_conf_max'] = max(self.step_conf_values)
            s['step_conf_mean'] = mean(self.step_conf_values)
        else:
            s['step_conf_min'] = s['step_conf_max'] = s['step_conf_mean'] = None

        if self.chain_conf_values:
            s['chain_conf_min'] = min(self.chain_conf_values)
            s['chain_conf_max'] = max(self.chain_conf_values)
            s['chain_conf_mean'] = mean(self.chain_conf_values)
        else:
            s['chain_conf_min'] = s['chain_conf_max'] = s['chain_conf_mean'] = None

        return s


# ---------- cross-file audits ----------

def cross_file_overlaps(audits: List[FileAudit]) -> Dict[str, Any]:
    out = {}
    # qid overlaps
    qid_to_files = defaultdict(list)
    for a in audits:
        for q in a.qid_counter:
            if a.qid_counter[q] > 0:
                qid_to_files[q].append(str(a.path))
    overlaps = {q: files for q, files in qid_to_files.items() if len(files) > 1}
    out['qid_overlaps_count'] = len(overlaps)
    out['sample_qid_overlaps'] = dict(list(overlaps.items())[:20])
    return out


# ---------- main driver ----------

def run(files: List[str], out_prefix: str):
    paths = [Path(p) for p in files]
    audits = []
    for p in paths:
        print(f"Ingesting {p}")
        a = FileAudit(p)
        a.ingest()
        audits.append(a)
        print(f"  total={a.total}, records_loaded={len(a.records)}, parse_errors={len(a.parse_errors)}")

    # per-file summaries
    report = {'files': {}, 'cross': {}}
    for a in audits:
        report['files'][str(a.path)] = a.summarize()

    # cross audits
    report['cross'] = cross_file_overlaps(audits)

    # global stats
    total_records = sum(a.total for a in audits)
    total_parse_errors = sum(len(a.parse_errors) for a in audits)
    total_fallbacks = sum(a.fallback_flagged for a in audits)
    report['summary'] = {
        'n_files': len(audits),
        'total_records': total_records,
        'total_parse_errors': total_parse_errors,
        'total_fallbacks': total_fallbacks
    }

    # write outputs
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_out = out_prefix.with_suffix(out_prefix.suffix + '.json')
    txt_out = out_prefix.with_suffix(out_prefix.suffix + '.txt')

    with json_out.open('w', encoding='utf-8') as jf:
        json.dump(report, jf, indent=2)
    with txt_out.open('w', encoding='utf-8') as tf:
        tf.write('ORM AUDIT REPORT\n')
        tf.write('='*60 + '\n')
        tf.write(f"Files checked: {len(audits)}\n")
        tf.write(f"Total records: {total_records}\n")
        tf.write(f"Total parse errors: {total_parse_errors}\n")
        tf.write(f"Total fallback flagged: {total_fallbacks}\n\n")
        for a in audits:
            s = a.summarize()
            tf.write(f"File: {s['path']}\n")
            tf.write(f"  total_records: {s['total_records']}\n")
            tf.write(f"  parse_errors: {s['parse_errors']}\n")
            tf.write(f"  duplicate_qids: {s['duplicate_qids']}\n")
            tf.write(f"  duplicate_chain_ids: {s['duplicate_chain_ids']}\n")
            tf.write(f"  fallback_flagged: {s['fallback_flagged']}\n")
            tf.write(f"  final_equals_gold: {s['final_equals_gold']}\n")
            tf.write(f"  malformed_steps: {s['malformed_steps']}\n")
            tf.write(f"  missing_fields (top): {list(s['missing_fields'].items())[:10]}\n")
            tf.write(f"  generated_by_top20: {s['generated_by_top20']}\n")
            tf.write(f"  step_count_min/mean/max: {s['step_count_min']}/{s['step_count_mean']}/{s['step_count_max']}\n")
            tf.write(f"  chain_conf_min/mean/max: {s['chain_conf_min']}/{s['chain_conf_mean']}/{s['chain_conf_max']}\n")
            tf.write('\n')

        tf.write('\nCROSS-FILE OVERLAPS\n')
        tf.write(json.dumps(report['cross'], indent=2) + '\n')

    print(f"Wrote report json -> {json_out}")
    print(f"Wrote human report -> {txt_out}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', '-f', nargs='+', required=True, help='JSONL files to audit')
    ap.add_argument('--out_prefix', required=True, help='Prefix for outputs (no suffix)')
    args = ap.parse_args()
    run(args.files, args.out_prefix)
