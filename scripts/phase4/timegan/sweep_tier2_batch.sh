#!/bin/bash
set -euo pipefail
cd ~/generative-ai-workload-modeling
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracegen

mkdir -p logs

# 20 variants: 5 workloads x 4 perturbations around each workload's
# frozen S27_HYPERPARAMS point (vr +/-0.2, fm x0.5 / x2.0). No
# sweep_tier3_batch.sh precedent exists -- Tier 3's Step 5 ablation
# was run as 20 sequential per-variant invocations, interleaved by
# perturbation type (vr_low x5, vr_high x5, fm_low x5, fm_high x5),
# not grouped by workload. Mirrored here. Each invocation trains one
# (workload, variant) pair via timegan_s36_tier2_sweep.py's
# --workload/--vr/--fm CLI.
#
# NOTE: values are the mechanical formula applied uniformly (vr+/-0.2,
# fm*0.5/*2.0). Tier 3's actual executed grid deviates from this
# formula in exactly one place -- whisper's fm_low used fm=0.3, not
# the formula's fm=0.25 (0.5*0.5) -- per docs/S36_TIER3_EXTENSION_REFERENCE.md
# Section 4.3/4.4. This script uses the formula value (0.25) for
# Tier 2, not Tier 3's one-off 0.3, since Tier 2 is a fresh ablation
# on different data, not a replay of Tier 3's numbers. Flagged
# explicitly in the Step 6 Phase 1 report for a final call.

run_variant() {
  local workload="$1" variant="$2" vr="$3" fm="$4"
  echo "=== Starting ${workload} ${variant} (vr=${vr} fm=${fm}) at $(date -u +%FT%TZ) ==="
  python scripts/phase4/timegan/timegan_s36_tier2_sweep.py \
    --workload "$workload" --vr "$vr" --fm "$fm" 2>&1 \
    | tee "logs/s36_tier2_sweep_${workload}_${variant}.log"

  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "=== FAILED on ${workload} ${variant} at $(date -u +%FT%TZ) ==="
    exit 1
  fi
  echo "=== Completed ${workload} ${variant} at $(date -u +%FT%TZ) ==="
}

# vr_low: lambda_var_reg - 0.2 (same fm)
run_variant bert      vr_low 0.3 1.5
run_variant gpt2      vr_low 0.1 2.0
run_variant resnet152 vr_low 0.2 1.0
run_variant whisper   vr_low 0.3 0.5
run_variant yolo      vr_low 0.1 1.2

# vr_high: lambda_var_reg + 0.2 (same fm)
run_variant bert      vr_high 0.7 1.5
run_variant gpt2      vr_high 0.5 2.0
run_variant resnet152 vr_high 0.6 1.0
run_variant whisper   vr_high 0.7 0.5
run_variant yolo      vr_high 0.5 1.2

# fm_low: lambda_fm_stat * 0.5 (same vr)
run_variant bert      fm_low 0.5 0.75
run_variant gpt2      fm_low 0.3 1.0
run_variant resnet152 fm_low 0.4 0.5
run_variant whisper   fm_low 0.5 0.25
run_variant yolo      fm_low 0.3 0.6

# fm_high: lambda_fm_stat * 2.0 (same vr)
run_variant bert      fm_high 0.5 3.0
run_variant gpt2      fm_high 0.3 4.0
run_variant resnet152 fm_high 0.4 2.0
run_variant whisper   fm_high 0.5 1.0
run_variant yolo      fm_high 0.3 2.4

echo "=== ALL 20 VARIANTS COMPLETE at $(date -u +%FT%TZ) ==="
touch logs/s36_tier2_sweep_complete.marker
