#!/bin/bash

echo "Phase 1 Pre-Experiment Checklist"
echo "================================="
echo ""

# 1. Memory check
echo "[1/5] Memory Status:"
free -h | grep Mem
MEMORY_PCT=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
if [ "$MEMORY_PCT" -gt 85 ]; then
    echo " Memory at ${MEMORY_PCT}% - free up memory first"
    exit 1
else
    echo "✓ Memory OK (${MEMORY_PCT}%)"
fi

# 2. Prometheus check
echo ""
echo "[2/5] Prometheus Status:"
PROM_PODS=$(kubectl get pods -n monitoring -l app=prometheus --no-headers 2>/dev/null | grep Running | wc -l)
if [ "$PROM_PODS" -eq 0 ]; then
    echo " Prometheus not running"
    exit 1
else
    echo "✓ Prometheus running"
fi

# 3. Scrape interval check
echo ""
echo "[3/5] Checking Prometheus scrape interval:"
kubectl get configmap prometheus-config -n monitoring -o yaml | grep -A 2 "scrape_interval"

# 4. Grafana check (should be disabled)
echo ""
echo "[4/5] Grafana Status:"
GRAFANA_REPLICAS=$(kubectl get deployment grafana -n monitoring -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "not found")
if [ "$GRAFANA_REPLICAS" == "0" ]; then
    echo "✓ Grafana disabled (0 replicas)"
elif [ "$GRAFANA_REPLICAS" == "not found" ]; then
    echo " Grafana deployment not found"
else
    echo " Grafana still running ($GRAFANA_REPLICAS replicas)"
    echo "   Run: kubectl scale deployment grafana -n monitoring --replicas=0"
fi

# 5. No running workloads
echo ""
echo "[5/5] Checking for running workload pods:"
WORKLOAD_PODS=$(kubectl get pods -l 'app in (resnet50,distilbert,whisper)' --no-headers 2>/dev/null | wc -l)
if [ "$WORKLOAD_PODS" -eq 0 ]; then
    echo "✓ No workload pods running"
else
    echo " Found $WORKLOAD_PODS workload pods running:"
    kubectl get pods -l 'app in (resnet50,distilbert,whisper)' --no-headers
fi

echo ""
echo "================================="
echo "Checklist Complete"
echo "================================="
