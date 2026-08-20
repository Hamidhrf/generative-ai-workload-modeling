#!/bin/bash
set -euo pipefail
cd ~/generative-ai-workload-modeling
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tracegen

mkdir -p logs

for workload in bert gpt2 resnet152 whisper yolo; do
  echo "=== Starting $workload at $(date -u +%FT%TZ) ==="
  python scripts/phase4/timegan/timegan_s36_tier2.py \
    --workload "$workload" 2>&1 \
    | tee "logs/s36_tier2_${workload}_frozen.log"

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "=== FAILED on $workload at $(date -u +%FT%TZ) ==="
    exit 1
  fi
  echo "=== Completed $workload at $(date -u +%FT%TZ) ==="
done

echo "=== ALL 5 WORKLOADS COMPLETE at $(date -u +%FT%TZ) ==="
touch logs/s36_tier2_batch_complete.marker
