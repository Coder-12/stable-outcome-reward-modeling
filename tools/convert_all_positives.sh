#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo " CONVERT ALL POSITIVE DATA TO ORM FORMAT"
echo "============================================"

mkdir -p data/processed

# 1. Convert TRAIN
echo "[1/3] Converting TRAIN positives..."
python tools/convert_pos_to_orm.py \
  --in_file data/processed/orm_train.jsonl \
  --out_file data/processed/orm_train_converted.jsonl \
  --report data/qc/orm_train_convert_report.json

# 2. Convert VAL
echo "[2/3] Converting VAL positives..."
python tools/convert_pos_to_orm.py \
  --in_file data/processed/orm_val.jsonl \
  --out_file data/processed/orm_val_converted.jsonl \
  --report data/qc/orm_val_convert_report.json

# 3. Convert TEST
echo "[3/3] Converting TEST positives..."
python tools/convert_pos_to_orm.py \
  --in_file data/processed/orm_test.jsonl \
  --out_file data/processed/orm_test_converted.jsonl \
  --report data/qc/orm_test_convert_report.json

echo "============================================"
echo " DONE — All positive sets converted."
echo "============================================"