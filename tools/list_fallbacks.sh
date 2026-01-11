#!/usr/bin/env bash
#set -e
#OUTDIR="data/qc"
#mkdir -p "$OUTDIR"
#
## list qids that contain "fallback" in meta (the 20)
#jq -r 'select(.meta.parse_report?.fallback == true or (.meta | tostring | test("fallback"))) | .qid' data/repair_v6_gpt_neg_final.jsonl > "$OUTDIR/gpt_fallback_qids.txt"
#
#echo "Fallback QIDs -> $OUTDIR/gpt_fallback_qids.txt"
#echo ""
#echo "Sample fallback records (first 5):"
#grep -Ff "$OUTDIR/gpt_fallback_qids.txt" data/repair_v6_gpt_neg_final.jsonl | head -n 10 > "$OUTDIR/gpt_fallback_samples.jsonl"
#jq '.' "$OUTDIR/gpt_fallback_samples.jsonl" | sed -n '1,200p'


set -e

IN=data/repair_v6_gpt_neg_final.jsonl
OUT=data/tmp_gpt_neg_regen_20.jsonl
QIDS=data/qc/gpt_fallback_qids.txt

echo "" > $OUT
while read -r q; do
    jq --arg q "$q" 'select(.qid == $q)' $IN >> $OUT
done < $QIDS

echo "Prepared 20-record regeneration input → $OUT"