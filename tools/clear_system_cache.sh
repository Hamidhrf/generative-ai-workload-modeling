#!/bin/bash

# System Cache Cleanup Script
# Run before each experiment for clean baseline measurements

echo "=========================================="
echo "System Cache Cleanup"
echo "=========================================="
echo ""

# 1. Check current memory state
echo "=== Memory State BEFORE Clearing ==="
free -h
echo ""

# 2. Clear all caches (PageCache, dentries, inodes)
echo "Clearing system caches..."
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sleep 2

# 3. Check memory state after clearing
echo ""
echo "=== Memory State AFTER Clearing ==="
free -h

# 4. System health checks
echo ""
echo "=== System Health Checks ==="
echo "Kubernetes node:"
kubectl get nodes --no-headers | awk '{print "  " $1 ": " $2}'

echo "System pods:"
SYSTEM_PODS=$(kubectl get pods -n kube-system | grep Running | wc -l)
echo "  $SYSTEM_PODS running"

echo "GPU status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | awk '{print "  " $0}'

# 5. Verify no workloads running
echo ""
echo "=== Workload Status ==="
WORKLOAD_COUNT=$(kubectl get pods -l 'app in (resnet50,distilbert,whisper)' 2>/dev/null | grep -v NAME | wc -l)
if [ "$WORKLOAD_COUNT" -eq 0 ]; then
    echo "✓ No workload pods running (clean slate)"
else
    echo "⚠ Warning: $WORKLOAD_COUNT workload pods found"
    kubectl get pods -l 'app in (resnet50,distilbert,whisper)'
fi

echo ""
echo "=========================================="
echo "✓ System Ready for Experiment"
echo "=========================================="
echo ""
