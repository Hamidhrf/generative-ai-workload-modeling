# GPU Setup Guide for Kubernetes with CRI-O

Complete guide to enable NVIDIA GPU support in Kubernetes cluster.

## Prerequisites

-  Stable Kubernetes cluster with CRI-O
-  NVIDIA GPU installed in the system
-  NVIDIA drivers installed on the host

## Quick Setup

### Step 1: Create RuntimeClass

```bash
# Apply the NVIDIA RuntimeClass
kubectl apply -f nvidia-runtimeclass.yaml

# Verify
kubectl get runtimeclass
```

### Step 2: Run GPU Health Check

```bash
# Make executable
chmod +x gpu_health_check.sh

# Run the health check and auto-setup
sudo ./gpu_health_check.sh
```

The script will automatically:
-  Check NVIDIA drivers
-  Install nvidia-container-toolkit if needed
-  Configure CRI-O for NVIDIA runtime
-  Install NVIDIA device plugin
-  Register GPUs with kubelet
-  Run a test pod

## Manual Setup (if needed)

### Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### Configure CRI-O

```bash
# Create NVIDIA runtime config
sudo mkdir -p /etc/crio/crio.conf.d/

cat <<EOF | sudo tee /etc/crio/crio.conf.d/99-nvidia.conf
[crio.runtime.runtimes.nvidia]
runtime_path = "/usr/bin/nvidia-container-runtime"
runtime_type = "oci"
runtime_root = "/run/nvidia-container-runtime"
EOF

# Restart CRI-O
sudo systemctl restart crio
```

### Install NVIDIA Device Plugin

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml
```

## Verification

### Check GPU Registration

```bash
# Check node capacity
kubectl describe node | grep nvidia.com/gpu

# Should show:
#  nvidia.com/gpu: 1
```

### Run Test Pod

```bash
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
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

# Check logs
kubectl logs gpu-test
```

## Using GPU in Deployments

### Example Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-gpu-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-gpu-app
  template:
    metadata:
      labels:
        app: my-gpu-app
    spec:
      runtimeClassName: nvidia  # Important!
      containers:
      - name: app
        image: your-image:tag
        resources:
          limits:
            nvidia.com/gpu: 1  # Request 1 GPU
```

### Key Points

1. **Always specify `runtimeClassName: nvidia`** in pod spec
2. **Request GPU in resources**: `nvidia.com/gpu: 1`
3. **GPU is a limit-only resource** (no requests, only limits)

## Troubleshooting

### GPU not showing up

```bash
# Check device plugin logs
kubectl logs -n kube-system -l name=nvidia-device-plugin-ds

# Check CRI-O can see runtime
sudo crictl info | grep nvidia

# Restart components
sudo systemctl restart crio
sudo systemctl restart kubelet
```

### Pod stuck in Pending

```bash
# Check pod events
kubectl describe pod <pod-name>

# Common issues:
# - Missing runtimeClassName: nvidia
# - GPU already allocated to another pod
# - Device plugin not running
```

### Test from host

```bash
# Test nvidia-smi works
nvidia-smi

# Test container runtime
sudo ctr run --rm --runtime=io.containerd.runc.v2 \
  --gpus 0 nvidia/cuda:12.2.2-base-ubuntu22.04 \
  test nvidia-smi
```

## Monitoring GPU Usage

### Real-time monitoring

```bash
# On host
watch -n 1 nvidia-smi

# In pod (if GPU allocated)
kubectl exec -it <pod-name> -- nvidia-smi
```

### GPU metrics

The NVIDIA device plugin exposes metrics that can be scraped by Prometheus:
- GPU utilization
- Memory usage
- Temperature
- Power consumption

## Reboot Safety

The GPU setup survives reboots because:
-  NVIDIA drivers load on boot
-  nvidia-container-toolkit is permanently installed
-  CRI-O config in `/etc/crio/crio.conf.d/` persists
-  Device plugin DaemonSet auto-restarts
-  RuntimeClass is a Kubernetes resource (stored in etcd)

### After Reboot Checklist

```bash
# Wait 2-3 minutes after boot, then:
kubectl get nodes  # Node should be Ready
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds  # Should be Running
kubectl describe node | grep nvidia.com/gpu  # Should show GPU count

# Or just run:
sudo ./gpu-health-check.sh
```

## Common Issues

### Issue: "runtime class nvidia not found"
**Solution**: Apply the RuntimeClass: `kubectl apply -f nvidia-runtimeclass.yaml`

### Issue: Device plugin crashes
**Solution**: Check CRI-O config and restart: `sudo ./gpu-health-check.sh`

### Issue: GPU shows 0 after reboot
**Solution**: Run health check: `sudo ./gpu-health-check.sh`

## References

- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [CRI-O Documentation](https://cri-o.io/)
- [Kubernetes RuntimeClass](https://kubernetes.io/docs/concepts/containers/runtime-class/)