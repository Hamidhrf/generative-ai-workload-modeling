#!/bin/bash

# Monitoring Stack Deployment Script
# Deploys Prometheus, Grafana, and all exporters for thesis data collection

set -e

echo "=========================================="
echo "Deploying Monitoring"
echo "=========================================="
echo ""

# Change to repository root
cd "$(dirname "$0")/../.."

# Step 0: Ensure storage provisioner exists
echo "[0/9] Checking storage provisioner..."
if ! kubectl get storageclass local-path &>/dev/null; then
    echo "Installing local-path storage provisioner..."
    kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml
    kubectl wait --for=condition=ready pod -l app=local-path-provisioner -n local-path-storage --timeout=120s
    kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
    echo "✓ Storage provisioner installed"
else
    echo "✓ Storage provisioner already exists"
fi
echo ""

# Step 1: Create monitoring namespace
echo "[1/9] Creating monitoring namespace..."
kubectl apply -f k8s/monitoring/namespace.yaml
echo "✓ Namespace created"
echo ""

# Step 2: Deploy Prometheus
echo "[2/9] Deploying Prometheus..."
kubectl apply -f k8s/monitoring/prometheus-config.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
echo "✓ Prometheus deployed"
echo ""

# Step 3: Deploy Node Exporter
echo "[3/9] Deploying Node Exporter (CPU/RAM/Disk metrics)..."
kubectl apply -f k8s/monitoring/node-exporter.yaml
echo "✓ Node Exporter deployed"
echo ""

# Step 3.5: Add GPU label to node (required for DCGM scheduling)
echo "[3.5/9] Ensuring GPU label on node..."
GPU_NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
if kubectl get node $GPU_NODE -o jsonpath='{.metadata.labels.nvidia\.com/gpu}' 2>/dev/null | grep -q "true"; then
    echo "✓ GPU label already exists"
else
    kubectl label nodes $GPU_NODE nvidia.com/gpu=true
    echo "✓ GPU label added to node $GPU_NODE"
fi
echo ""

# Step 4: Deploy DCGM Exporter
echo "[4/9] Deploying DCGM Exporter (GPU metrics)..."
kubectl apply -f k8s/monitoring/dcgm-exporter.yaml
echo "✓ DCGM Exporter deployed"
echo ""

# Step 5: Deploy kube-state-metrics
echo "[5/9] Deploying kube-state-metrics (K8s object state)..."
kubectl apply -f k8s/monitoring/kube-state-metrics.yaml
echo "✓ kube-state-metrics deployed"
echo ""

# Step 6: Deploy Kepler
echo "[6/9] Deploying Kepler (Power consumption)..."
kubectl apply -f k8s/monitoring/kepler.yaml
echo "✓ Kepler deployed"
echo ""

# Step 7: Deploy Grafana
echo "[7/9] Deploying Grafana..."
kubectl apply -f k8s/monitoring/grafana-deployment.yaml
echo "✓ Grafana deployed"
echo ""

# Step 8: Wait for all pods to be ready
echo "[8/9] Waiting for all monitoring pods to be ready..."
echo "This may take 2-3 minutes..."
sleep 20

kubectl wait --for=condition=ready pod \
  -l app=prometheus \
  -n monitoring \
  --timeout=300s || echo " Prometheus might still be starting..."

kukubectl wait --for=condition=ready pod \
  -l app=grafana \
  -n monitoring \
  --timeout=300s || echo " Grafana might still be starting..."

kubectl wait --for=condition=ready pod \
  -l app=node-exporter \
  -n monitoring \
  --timeout=300s || echo " Node Exporter might still be starting..."

echo ""
echo "=========================================="
echo "Monitoring Stack Deployed!"
echo "=========================================="
echo ""

# Get node IP for access
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

echo " Access URLs:"
echo "  Prometheus:  http://${NODE_IP}:30090"
echo "  Grafana:     http://${NODE_IP}:30030"
echo ""
echo " Grafana Credentials:"
echo "  Username: admin"
echo "  Password: admin"
echo ""

# Show pod status
echo "[9/9] Final Status Check:"
kubectl get pods -n monitoring -o wide
echo ""

echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Access Grafana: http://${NODE_IP}:30030"
echo "2. Login with admin/admin"
echo "3. Import dashboards from dashboards/ folder"
echo "4. Verify Prometheus is collecting metrics:"
echo "   - Go to Prometheus: http://${NODE_IP}:30090"
echo "   - Check Status > Targets"
echo ""
echo " For Thesis Data Collection:"
echo "  - GPU metrics: dcgm_* metrics"
echo "  - CPU/RAM: node_* metrics"
echo "  - Container: container_* metrics"
echo "  - Power: kepler_* metrics"
echo "  - Latency: (custom metrics from your apps)"
echo ""