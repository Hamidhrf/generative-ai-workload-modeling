#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: ./verify_any_experiment.sh <workload> <replicas>"
    echo "Example: ./verify_any_experiment.sh distilbert 3"
    exit 1
fi

WORKLOAD=$1
REPLICAS=$2
PATTERN="${WORKLOAD}_r${REPLICAS}"

echo "Checking experiment: $PATTERN"
echo ""

# Count files
FILE_COUNT=$(ls data/raw/phase1/${PATTERN}_*.csv 2>/dev/null | wc -l)
echo "CSV files found: $FILE_COUNT (expected: 15)"

# Check row counts
for metric in cpu_usage memory_usage inference_latency_avg gpu_utilization; do
    FILE=$(ls data/raw/phase1/${PATTERN}_${metric}_*.csv 2>/dev/null | head -1)
    if [ -f "$FILE" ]; then
        ROWS=$(wc -l < "$FILE")
        echo "  $metric: $ROWS rows"
    fi
done

echo ""
if [ "$FILE_COUNT" -eq 15 ]; then
    echo "✓ All metrics collected"
else
    echo "⚠ Missing metrics"
fi
