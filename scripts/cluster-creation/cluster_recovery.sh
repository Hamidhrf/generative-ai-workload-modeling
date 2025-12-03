#!/bin/bash

# Kubernetes Cluster Recovery Script
# Run this after reboot if cluster has issues

echo "=========================================="
echo "Kubernetes Cluster Recovery"
echo "=========================================="

# Wait for CRI-O to be ready
echo "[1/6] Waiting for CRI-O to be ready..."
timeout=60
counter=0
while ! sudo systemctl is-active --quiet crio; do
    if [ $counter -ge $timeout ]; then
        echo "✗ CRI-O failed to start"
        sudo systemctl status crio
        exit 1
    fi
    echo "Waiting for CRI-O... ($counter/$timeout)"
    sleep 2
    ((counter+=2))
done
echo "✓ CRI-O is running"

# Wait for kubelet to be ready
echo "[2/6] Waiting for kubelet to be ready..."
counter=0
while ! sudo systemctl is-active --quiet kubelet; do
    if [ $counter -ge $timeout ]; then
        echo "✗ Kubelet failed to start"
        sudo systemctl status kubelet
        exit 1
    fi
    echo "Waiting for kubelet... ($counter/$timeout)"
    sleep 2
    ((counter+=2))
done
echo "✓ Kubelet is running"

# Wait for API server to be responsive
echo "[3/6] Waiting for API server..."
counter=0
while ! kubectl get nodes &>/dev/null; do
    if [ $counter -ge 120 ]; then
        echo "✗ API server not responding"
        exit 1
    fi
    echo "Waiting for API server... ($counter/120)"
    sleep 5
    ((counter+=5))
done
echo "✓ API server is responsive"

# Check node status
echo "[4/6] Checking node status..."
kubectl get nodes

# Wait for critical pods
echo "[5/6] Waiting for critical system pods..."
echo "This may take 2-3 minutes..."
sleep 30

kubectl wait --for=condition=ready pod \
    -l k8s-app=kube-dns \
    -n kube-system \
    --timeout=300s 2>/dev/null || echo "CoreDNS might still be starting..."

# Show cluster status
echo "[6/6] Cluster Status:"
echo ""
echo "=== Nodes ==="
kubectl get nodes
echo ""
echo "=== System Pods ==="
kubectl get pods -n kube-system
echo ""
echo "=== Calico Pods ==="
kubectl get pods -n calico-system
echo ""

# Check for not-ready pods
NOT_READY=$(kubectl get pods -A | grep -v "Running\|Completed" | grep -v "READY" | wc -l)
if [ $NOT_READY -gt 0 ]; then
    echo "⚠ Warning: $NOT_READY pods are not ready yet"
    echo "Wait a few more minutes and check: kubectl get pods -A"
else
    echo "✓ All pods are running"
fi

echo ""
echo "=========================================="
echo "Recovery Complete!"
echo "=========================================="
echo ""
echo "If you see issues, run:"
echo "  kubectl get pods -A"
echo "  kubectl describe pod <pod-name> -n <namespace>"
echo "  sudo journalctl -u kubelet -n 100"
echo "  sudo journalctl -u crio -n 100"
echo ""