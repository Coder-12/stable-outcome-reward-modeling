#!/usr/bin/env bash
set -e

# ================================
#  GLOBAL CONFIG
# ================================
QC_DIR="data/qc"

echo "============================================"
echo "        ORM DATA QUALITY AUDIT SUITE"
echo "============================================"
echo "QC reports will be saved under: $QC_DIR/"
echo ""

# ================================
#  Helper: run audit on a dataset
# ================================
run_audit() {
    local name="$1"
    local file="$2"

    echo "🔍 Auditing: $name"
    echo "     File: $file"
    echo "     Report: $QC_DIR/${name}_qc.json"
    python tools/orm_audit.py \
        --files "$file" \
        --out_prefix "$QC_DIR/${name}_qc"
    echo ""
}

# ================================
#  RUN AUDITS
# ================================

# 1. GPT-generated negative dataset (339)
run_audit "gpt_negatives" "data/repair_v6_gpt_neg_final_fixed.jsonl"

# 2. Synthetic negative dataset (4413)
run_audit "synthetic_negatives" "data/repair_v6_synth_all_neg_unique.jsonl"

# 3. ORM positive train dataset (gold)
run_audit "orm_positives_train_converted" "data/processed/orm_train_converted.jsonl"

# 4. ORM positive val dataset (gold)
run_audit "orm_positives_val_converted" "data/processed/orm_val_converted.jsonl"

# 5. ORM positive test dataset (gold)
run_audit "orm_positives_test_converted" "data/processed/orm_test_converted.jsonl"

echo "============================================"
echo " QC Completed. Reports saved under $QC_DIR/"
echo "============================================"