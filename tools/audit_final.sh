#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo " ORM FINAL SPLIT AUDIT SUITE"
echo "============================================"

python tools/orm_audit.py \
  --files data/processed/orm_train_new.jsonl \
  --out_prefix data/qc/final_orm_train

python tools/orm_audit.py \
  --files data/processed/orm_val_new.jsonl \
  --out_prefix data/qc/final_orm_val

python tools/orm_audit.py \
  --files data/processed/orm_test_new.jsonl \
  --out_prefix data/qc/final_orm_test

echo "============================================"
echo " Completed. Reports under data/qc/"
echo "============================================"