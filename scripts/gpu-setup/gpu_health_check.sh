#!/usr/bin/env bash
set -euo pipefail

echo "===================================================="
echo " GPU HEALTH CHECK & AUTO-REPAIR SCRIPT (CRI-O)"
echo "===================================================="
echo

### 1. CHECK GPU VISIBILITY AT HOST LEVEL
echo "[1] Checking NVIDIA driver (nvidia-smi)..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi not working. GPU drivers not loaded."
    exit 1
fi
echo "✓ OK: NVIDIA driver is working."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo

### 2. CHECK NVIDIA CONTAINER RUNTIME
echo "[2] Checking NVIDIA container runtime..."
if ! command -v nvidia-container-runtime &> /dev/null; then
    echo "ERROR: nvidia-container-runtime not found."
    echo "Installing nvidia-container-toolkit..."
    
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
else
    echo "✓ OK: nvidia-container-runtime found."
fi
echo

### 3. CHECK CRI-O RUNTIME CONFIG
echo "[3] Checking CRI-O NVIDIA runtime configuration..."

CRIO_CONF="/etc/crio/crio.conf.d/99-nvidia.conf"

if [[ ! -f "$CRIO_CONF" ]]; then
    echo "Creating CRI-O NVIDIA runtime config..."
    
    sudo mkdir -p /etc/crio/crio.conf.d/
    
    cat <<EOF | sudo tee $CRIO_CONF
[crio.runtime.runtimes.nvidia]
runtime_path = "/usr/bin/nvidia-container-runtime"
runtime_type = "oci"
runtime_root = "/run/nvidia-container-runtime"
monitor_path = "/usr/libexec/crio/conmon"
EOF
    
    echo "Restarting CRI-O..."
    sudo systemctl restart crio
    sleep 3
    
    if ! sudo systemctl is-active --quiet crio; then
        echo "ERROR: CRI-O failed to restart"
        sudo systemctl status crio
        exit 1
    fi
    echo "✓ CRI-O restarted successfully."
else
    echo "✓ OK: CRI-O NVIDIA runtime config exists."
fi
echo

### 4. VERIFY CRI-O RUNTIME
echo "[4] Verifying CRI-O can see NVIDIA runtime..."
if sudo crictl info 2>/dev/null | grep -q nvidia; then
    echo "✓ OK: NVIDIA runtime registered in CRI-O."
else
    echo "⚠ Warning: NVIDIA runtime not visible in crictl info."
    echo "This may be normal. Proceeding..."
fi
echo

### 5. CHECK DEVICE PLUGIN DAEMONSET
echo "[5] Checking NVIDIA device plugin DaemonSet..."
PLUGIN=$(kubectl get ds -n kube-system nvidia-device-plugin-daemonset 2>/dev/null || true)

if [[ -z "$PLUGIN" ]]; then
    echo "Device plugin missing. Installing..."
    kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml
    sleep 5
else
    echo "✓ Device plugin exists."
fi
echo

### 6. CHECK DEVICE PLUGIN POD STATUS
echo "[6] Checking device plugin pod status..."
kubectl wait --for=condition=ready pod -l name=nvidia-device-plugin-ds -n kube-system --timeout=30s || true

READY=$(kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || echo "false")

if [[ "$READY" != "true" ]]; then
    echo "Device plugin not ready. Checking logs..."
    POD_NAME=$(kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -n "$POD_NAME" ]]; then
        echo "Last 10 lines of device plugin logs:"
        kubectl logs -n kube-system "$POD_NAME" --tail=10 || true
    fi
    
    echo "Attempting to restart device plugin..."
    kubectl delete ds -n kube-system nvidia-device-plugin-daemonset --force --grace-period=0 || true
    sleep 2
    kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml
    sleep 10
else
    echo "✓ OK: Device plugin pod is running."
fi
echo

### 7. CHECK GPU RESOURCE REGISTRATION IN KUBELET
echo "[7] Checking kubelet GPU registration..."
NODE_NAME=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
ALLOC=$(kubectl get node "$NODE_NAME" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")

if [[ "$ALLOC" == "0" ]] || [[ -z "$ALLOC" ]]; then
    echo "GPU not registered. Restarting kubelet..."
    sudo systemctl restart kubelet
    sleep 10
    ALLOC=$(kubectl get node "$NODE_NAME" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")
fi

if [[ "$ALLOC" -ge "1" ]]; then
    echo "✓ OK: kubelet registered GPU count = $ALLOC"
else
    echo "✗ FAIL: GPU still not registered."
    echo "Checking node status:"
    kubectl describe node "$NODE_NAME" | grep -A 5 "Allocatable:"
    exit 1
fi
echo

### 8. GPU TEST POD
echo "[8] Running GPU test pod..."

kubectl delete pod gpu-test --force --grace-period=0 >/dev/null 2>&1 || true
sleep 2

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  runtimeClassName: nvidia
  containers:
  - name: cuda-test
    image: nvidia/cuda:12.2.2-base-ubuntu22.04
    command: ["bash", "-c", "nvidia-smi && echo 'GPU Test Passed!' && sleep 5"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

echo "Waiting for pod to start..."
sleep 5

# Wait for pod to complete
kubectl wait --for=condition=ready pod/gpu-test --timeout=30s || true
sleep 3

LOGS=$(kubectl logs gpu-test 2>/dev/null || echo "No logs available")

if echo "$LOGS" | grep -q "NVIDIA-SMI"; then
    echo "✓ SUCCESS: GPU test passed."
    echo ""
    echo "GPU Info from test pod:"
    echo "$LOGS" | head -15
else
    echo "✗ ERROR: GPU test failed."
    echo "Pod status:"
    kubectl get pod gpu-test
    echo ""
    echo "Pod logs:"
    echo "$LOGS"
    echo ""
    echo "Pod description:"
    kubectl describe pod gpu-test | tail -30
    exit 1
fi

# Cleanup test pod
kubectl delete pod gpu-test --force --grace-period=0 >/dev/null 2>&1 || true

echo
echo "===================================================="
echo " ✓ GPU SETUP HEALTHY"
echo "===================================================="
echo "Summary:"
echo "  - NVIDIA Driver: Working"
echo "  - CRI-O Runtime: Configured for NVIDIA"
echo "  - Device Plugin: Running"
echo "  - GPU Resources: $ALLOC GPU(s) available"
echo "  - Test Pod: Passed"
echo ""
echo "Your cluster is ready for GPU workloads!"
echo "===================================================="