#!/bin/bash

# Experiment Tracker for Phase 1
# Shows which experiments are complete and which remain

echo "=========================================="
echo "Phase 1 Experiment Progress Tracker"
echo "=========================================="
echo ""

DATA_DIR="data/raw/phase1"

# Define all experiments
declare -a WORKLOADS=("resnet50" "distilbert" "whisper")
declare -a REPLICAS=(1 3 8)

TOTAL=9
COMPLETED=0

echo "Status of all 9 experiments:"
echo ""

for workload in "${WORKLOADS[@]}"; do
    echo "[$workload]"
    for replica in "${REPLICAS[@]}"; do
        # Check if CSV files exist for this experiment
        COUNT=$(find "$DATA_DIR" -name "${workload}_r${replica}_*.csv" 2>/dev/null | wc -l)
        
        if [ "$COUNT" -gt 0 ]; then
            echo "  ✓ ${replica} replicas - COMPLETE ($COUNT files)"
            ((COMPLETED++))
        else
            echo "  ☐ ${replica} replicas - PENDING"
        fi
    done
    echo ""
done

echo "=========================================="
echo "Progress: $COMPLETED / $TOTAL experiments complete"
REMAINING=$((TOTAL - COMPLETED))
echo "Remaining: $REMAINING experiments"
echo "=========================================="
echo ""

if [ "$REMAINING" -gt 0 ]; then
    echo "Remaining experiments to run:"
    echo ""
    for workload in "${WORKLOADS[@]}"; do
        for replica in "${REPLICAS[@]}"; do
            COUNT=$(find "$DATA_DIR" -name "${workload}_r${replica}_*.csv" 2>/dev/null | wc -l)
            if [ "$COUNT" -eq 0 ]; then
                echo "  python3 tools/run_single_experiment.py $workload $replica"
            fi
        done
    done
else
    echo "All experiments complete!"
fi
