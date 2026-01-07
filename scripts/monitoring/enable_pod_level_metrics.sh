#!/bin/bash

set -e

echo "=========================================="
echo "Enabling Pod-Level GPU and PSI Metrics"
echo "=========================================="
echo ""

# Step 1: Update PSI metrics in experiment script
echo "[1/4] Updating PSI metrics to container-level..."
sed -i "s/'cpu_psi': 'rate(node_pressure_cpu_waiting_seconds_total\[1m\])',/'cpu_psi': f'rate(container_pressure_cpu_waiting_seconds_total{{pod=~\"{workload}.*\"}}[1m])',/g" tools/run_single_experiment.py
sed -i "s/'memory_psi': 'rate(node_pressure_memory_waiting_seconds_total\[1m\])',/'memory_psi': f'rate(container_pressure_memory_waiting_seconds_total{{pod=~\"{workload}.*\"}}[1m])',/g" tools/run_single_experiment.py
sed -i "s/'io_psi': 'rate(node_pressure_io_waiting_seconds_total\[1m\])',/'io_psi': f'rate(container_pressure_io_waiting_seconds_total{{pod=~\"{workload}.*\"}}[1m])',/g" tools/run_single_experiment.py
echo "✓ PSI metrics updated"

# Step 2: Backup current DCGM config
echo ""
echo "[2/4] Backing up current DCGM config..."
kubectl get daemonset dcgm-exporter -n monitoring -o yaml > dcgm-exporter-backup.yaml
echo "✓ Backup saved to dcgm-exporter-backup.yaml"

# Step 3: Update DCGM Exporter
echo ""
echo "[3/4] Updating DCGM Exporter for pod-level metrics..."

cat > dcgm-exporter-pod-metrics.yaml << 'EOFDCGM'
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
    spec:
      nodeSelector:
        nvidia.com/gpu: "true"
      runtimeClassName: nvidia
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.8-3.1.5-ubuntu20.04
        env:
        - name: DCGM_EXPORTER_LISTEN
          value: ":9400"
        - name: DCGM_EXPORTER_KUBERNETES_GPU_ID_TYPE
          value: "device-name"
        - name: DCGM_EXPORTER_COLLECTORS
          value: "/etc/dcgm-exporter/dcp-metrics-included.csv"
        ports:
        - containerPort: 9400
          protocol: TCP
        securityContext:
          privileged: true
        volumeMounts:
        - name: pod-gpu-resources
          mountPath: /var/lib/kubelet/pod-resources
          readOnly: true
      volumes:
      - name: pod-gpu-resources
        hostPath:
          path: /var/lib/kubelet/pod-resources
EOFDCGM

kubectl apply -f dcgm-exporter-pod-metrics.yaml
kubectl rollout status daemonset dcgm-exporter -n monitoring --timeout=120s
echo "✓ DCGM Exporter updated"

# Step 4: Verify pod-level metrics
echo ""
echo "[4/4] Verifying pod-level metrics..."
sleep 30

# Check PSI
PSI_COUNT=$(curl -s "http://172.22.174.58:30090/api/v1/label/__name__/values" | jq -r '.data[]' | grep "container_pressure" | wc -l)
echo "Container PSI metrics available: $PSI_COUNT"

if [ "$PSI_COUNT" -gt 0 ]; then
    echo "✓ Pod-level PSI metrics confirmed"
else
    echo "✗ Pod-level PSI metrics not found"
fi

# Check GPU pod labels
echo ""
echo "Checking GPU metrics for pod labels..."
GPU_LABELS=$(curl -s "http://172.22.174.58:30090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL" | jq '.data.result[0].metric | keys[]' | grep -i pod | wc -l)

if [ "$GPU_LABELS" -gt 0 ]; then
    echo "✓ Pod-level GPU metrics enabled!"
    echo ""
    echo "GPU metric sample:"
    curl -s "http://172.22.174.58:30090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL" | jq '.data.result[0].metric'
else
    echo "⚠ Pod-level GPU metrics not yet available"
    echo "  This may take a few minutes to propagate"
    echo "  Or may require GPU workload to be running"
fi

echo ""
echo "=========================================="
echo "Update Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update GPU metrics in script if pod labels confirmed"
echo "2. Delete incomplete ResNet50 × 1 data"
echo "3. Re-run experiment with pod-level metrics"
echo ""
