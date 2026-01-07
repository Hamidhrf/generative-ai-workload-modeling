#!/bin/bash

echo "=========================================="
echo "CSV Data Quality Verification"
echo "=========================================="
echo ""

CSV_DIR="data/raw/phase1"

# Find the most recent CPU usage file
CPU_FILE=$(ls -t $CSV_DIR/resnet50_r1_cpu_usage_*.csv 2>/dev/null | head -1)
MEM_FILE=$(ls -t $CSV_DIR/resnet50_r1_memory_usage_*.csv 2>/dev/null | head -1)
LAT_FILE=$(ls -t $CSV_DIR/resnet50_r1_inference_latency_avg_*.csv 2>/dev/null | head -1)
GPU_FILE=$(ls -t $CSV_DIR/resnet50_r1_gpu_utilization_*.csv 2>/dev/null | head -1)

echo "[1/5] Checking CPU usage data..."
echo "File: $(basename $CPU_FILE)"
head -5 "$CPU_FILE"
echo "..."
tail -3 "$CPU_FILE"
echo "Row count: $(wc -l < $CPU_FILE)"
echo ""

echo "[2/5] Checking memory usage data..."
echo "File: $(basename $MEM_FILE)"
head -5 "$MEM_FILE"
echo "..."
tail -3 "$MEM_FILE"
echo "Row count: $(wc -l < $MEM_FILE)"
echo ""

echo "[3/5] Checking inference latency data..."
echo "File: $(basename $LAT_FILE)"
head -5 "$LAT_FILE"
echo "..."
tail -3 "$LAT_FILE"
echo "Row count: $(wc -l < $LAT_FILE)"
echo ""

echo "[4/5] Checking GPU utilization..."
echo "File: $(basename $GPU_FILE)"
head -5 "$GPU_FILE"
echo "..."
tail -3 "$GPU_FILE"
echo "Row count: $(wc -l < $GPU_FILE)"
echo ""

echo "[5/5] Summary of all CSV files:"
ls -lh "$CSV_DIR"/resnet50_r1_*.csv | awk '{print $9, $5}' | sed 's|data/raw/phase1/||'
echo ""

echo "=========================================="
echo "Data Quality Assessment"
echo "=========================================="
echo ""

# Check for non-zero values in CPU
CPU_NONZERO=$(tail -n +2 "$CPU_FILE" | cut -d',' -f2- | grep -v '^$' | grep -v '^0' | wc -l)
echo "CPU data points with non-zero values: $CPU_NONZERO"

# Check for non-zero values in memory
MEM_NONZERO=$(tail -n +2 "$MEM_FILE" | cut -d',' -f2- | grep -v '^$' | grep -v '^0' | wc -l)
echo "Memory data points with non-zero values: $MEM_NONZERO"

# Check for non-zero values in inference latency
LAT_NONZERO=$(tail -n +2 "$LAT_FILE" | cut -d',' -f2- | grep -v '^$' | grep -v '^0' | wc -l)
echo "Latency data points with non-zero values: $LAT_NONZERO"

echo ""
if [ "$CPU_NONZERO" -gt 100 ] && [ "$MEM_NONZERO" -gt 100 ]; then
    echo "✓✓✓ DATA QUALITY: EXCELLENT ✓✓✓"
    echo "metrics are collecting properly"
else
    echo "WARNING: Low non-zero data points detected"
    echo "Check if workload was actually running during collection"
fi
echo ""
echo "Expected row count: ~720 (60 min × 12 samples/min at 5s interval)"
echo "=========================================="
