#!/usr/bin/env bash
# E6 fm-weight search driver (NOT committed). Runs the 22 (tier, workload, fm)
# combos sequentially through timegan_s36_e6_fmsearch.py. Continues past a
# single-run failure (logs it, moves on) rather than halting the whole batch.
set -u
cd "$(dirname "$0")/../../.."  # repo root

PY=python3
export PYTHONPATH="$(pwd)/scripts/utils"

RUNS=(
  "tier2 gpt2 0.125"
  "tier2 gpt2 0.25"
  "tier2 gpt2 0.5"
  "tier2 resnet152 0.125"
  "tier2 resnet152 0.25"
  "tier2 resnet152 0.5"
  "tier2 resnet152 2.0"
  "tier2 resnet152 4.0"
  "tier3_matched gpt2 0.125"
  "tier3_matched gpt2 0.25"
  "tier3_matched gpt2 0.5"
  "tier3_matched resnet152 0.125"
  "tier3_matched resnet152 0.25"
  "tier3_matched resnet152 0.5"
  "tier3_matched resnet152 2.0"
  "tier3_matched resnet152 4.0"
  "tier3_matched whisper 0.125"
  "tier3_matched whisper 0.25"
  "tier3_matched whisper 0.5"
  "tier3_matched yolo 0.125"
  "tier3_matched yolo 0.25"
  "tier3_matched yolo 0.5"
)

TOTAL=${#RUNS[@]}
I=0
FAILED=()

for spec in "${RUNS[@]}"; do
  I=$((I+1))
  read -r TIER WL FM <<< "$spec"
  echo "=============================================================="
  echo "[$I/$TOTAL] tier=$TIER workload=$WL fm=$FM  $(date -u +%FT%TZ)"
  echo "=============================================================="
  $PY scripts/phase4/timegan/timegan_s36_e6_fmsearch.py --tier "$TIER" --workload "$WL" --fm "$FM"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED: tier=$TIER workload=$WL fm=$FM rc=$rc"
    FAILED+=("$spec")
  fi
done

echo "=============================================================="
echo "E6 BATCH DONE: $((TOTAL - ${#FAILED[@]}))/$TOTAL succeeded"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED RUNS:"
  for f in "${FAILED[@]}"; do echo "  $f"; done
fi
echo "=============================================================="
