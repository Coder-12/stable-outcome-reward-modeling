#!/usr/bin/env bash
set -euo pipefail

# ===========================
#   INPUT FILES
# ===========================
POS="data/processed/orm_train_converted.jsonl"
GPT_NEG="data/repair_v6_gpt_neg_final_fixed.jsonl"
SYN_NEG="data/repair_v6_synth_all_neg_unique.jsonl"

OUT="data/orm_train_merged_v6.jsonl"
TMP="data/orm_train_merged_v6.tmp"

echo "=========================================="
echo "     ORM MERGE SCRIPT (v6 - Stable)"
echo "=========================================="
echo "Positives       : $POS"
echo "GPT negatives   : $GPT_NEG"
echo "Synthetic negs  : $SYN_NEG"
echo "Output          : $OUT"
echo "------------------------------------------"

# Sanity checks
for f in "$POS" "$GPT_NEG" "$SYN_NEG"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing file: $f" >&2
        exit 2
    fi
done

echo "[1/5] Initializing tmp output..."
> "$TMP"

echo "[2/5] Adding POSITIVES..."
cat "$POS" >> "$TMP"

echo "[3/5] Adding GPT NEGATIVES..."
cat "$GPT_NEG" >> "$TMP"

echo "[4/5] Adding SYNTHETIC NEGATIVES..."
cat "$SYN_NEG" >> "$TMP"

echo "[5/5] Atomic replace..."
mv "$TMP" "$OUT"

echo ""
echo "=========================================="
echo " MERGE COMPLETE"
echo " Output: $OUT"
echo "=========================================="

echo ""
echo "====== Verification ======"
echo "Total records (wc -l): $(wc -l < "$OUT")"
echo "Unique qids: $(jq -r '.qid' "$OUT" | sort | uniq | wc -l)"
echo "Duplicate qids: $(( $(wc -l < "$OUT") - $(jq -r '.qid' "$OUT" | sort | uniq | wc -l) ))"
echo "Duplicate chain_ids: $(( $(jq -r '.chain_id' "$OUT" | wc -l) - $(jq -r '.chain_id' "$OUT" | sort | uniq | wc -l) ))"
echo ""
echo "Reminder: Expected counts → POS 4679 + GPT 339 + SYN 4413 = 9431"