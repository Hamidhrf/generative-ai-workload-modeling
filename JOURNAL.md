# Technical Journal: Kubernetes Cluster Setup for Generative AI Workload Modeling

**Research Context**: Master's Thesis in Digital Transformation at Fachhochschule Dortmund  
**Objective**: Deploy and analyze three AI applications (ResNet50, DistilBERT, Whisper) on Kubernetes with GPU support  
**Date Range**: December 2-3, 2025  
**Environment**: Ubuntu 24.04 LTS, Single-node cluster, NVIDIA A16 GPU

---

## Executive Summary

This journal documents the complete setup of a production-ready Kubernetes cluster with GPU support for AI inference workload analysis. The setup required careful consideration of container runtime selection, reboot stability, and GPU integration with CRI-O.

**Key Achievements**:
- Single-node Kubernetes 1.34 cluster with CRI-O runtime
- Full reboot stability with automatic service recovery
- NVIDIA A16 GPU integration for AI workloads
- Production-ready configuration for research experiments

---

## Day 1: Initial Cluster Design Decisions

### 1.1 Container Runtime Selection: CRI-O vs Containerd

**Decision**: Use CRI-O as container runtime

**Reasoning**:
- Previous attempts with Containerd encountered version compatibility issues
- Containerd pause image version mismatches with Kubernetes 1.34
- CRI-O provides native OCI compliance without Docker dependencies
- Better alignment with Kubernetes architecture

**Technical Context**:
- Kubernetes deprecated Docker support in v1.24
- CRI (Container Runtime Interface) allows pluggable runtimes
- Options evaluated: CRI-O, Containerd, Docker with cri-dockerd

**Initial Challenge**: CRI-O package for Ubuntu 24.04 from pkgs.k8s.io had broken package state on first installation attempt.

**Resolution**: Used `apt-get install --reinstall` to properly install CRI-O binary after detecting missing `/usr/bin/crio`.

### 1.2 Kubernetes Version Selection

**Decision**: Kubernetes 1.34.0

**Reasoning**:
- Latest stable release at time of setup
- Required for compatibility with modern device plugins
- Matches CRI-O version 1.31 (compatible pairing)

**Technical Details**:
- Installed from `pkgs.k8s.io/core:/stable:/v1.34/deb/`
- Version string: `1.34.0-1.1`
- Components: kubeadm, kubelet, kubectl

### 1.3 Network Plugin Selection

**Decision**: Calico v3.29.1

**Reasoning**:
- Industry-standard CNI for Kubernetes
- Supports network policies (required for research isolation)
- Compatible with single-node clusters
- VXLAN encapsulation for pod networking

**Configuration**:
- Pod CIDR: `10.244.0.0/16`
- Service CIDR: `10.96.0.0/12`
- Deployment method: Tigera Operator

---

## Day 1: Reboot Stability Requirements

### 2.1 Problem Statement

**Challenge**: University VM environment experiences frequent reboots due to:
- Maintenance windows
- Power management
- System updates
- Infrastructure constraints

**Requirement**: Cluster must survive reboots without manual intervention.

### 2.2 Swap Management Strategy

**Issue**: Kubernetes requires swap to be disabled for proper memory management.

**Implementation**:
```bash
# Three-layer approach for swap persistence:
# 1. Immediate disable
sudo swapoff -a

# 2. Filesystem persistence (fstab)
sudo sed -i 's|^/swap.img|#/swap.img|g' /etc/fstab

# 3. Crontab failsafe
(crontab -l 2>/dev/null; echo "@reboot /sbin/swapoff -a") | crontab -
```

**Reasoning**: Multiple layers ensure swap stays disabled even if one mechanism fails.

### 2.3 Kernel Module Persistence

**Configuration**: `/etc/modules-load.d/k8s.conf`
```
overlay
br_netfilter
```

**Purpose**:
- `overlay`: Container filesystem driver
- `br_netfilter`: Bridge netfilter for iptables

**Verification**: Modules load automatically via systemd on boot.

### 2.4 Network Parameters Persistence

**Configuration**: `/etc/sysctl.d/k8s.conf`
```
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
```

**Purpose**:
- Enable iptables processing for bridge traffic
- Allow pod-to-pod communication
- Persist across reboots via sysctl

### 2.5 Service Enablement

**Critical Services**:
```bash
sudo systemctl enable crio
sudo systemctl enable kubelet
```

**Reasoning**: SystemD ensures automatic service startup on boot, critical for cluster availability.

---

## Day 1: Initial Cluster Setup Process

### 3.1 Installation Sequence

**Step 1: System Preparation**
- Kernel parameters configured
- Swap disabled with persistence
- Package repositories added

**Step 2: CRI-O Installation**
```bash
# Repository addition
curl -fsSL https://pkgs.k8s.io/addons:/cri-o:/stable:/v1.31/deb/Release.key | \
    sudo gpg --dearmor -o /etc/apt/keyrings/cri-o-apt-keyring.gpg

# Installation with reinstall flag for broken package states
sudo apt-get install -y --reinstall cri-o
```

**Challenge Encountered**: Initial installation showed package installed but binary missing at `/usr/bin/crio`.

**Root Cause**: Broken package state from previous cleanup attempts.

**Solution**: Explicit `--reinstall` flag forces fresh binary installation.

**Step 3: CNI Configuration**
- Default CRI-O CNI configs removed (conflict with Calico)
- CNI plugins directory: `/opt/cni/bin`
- Calico manages its own CNI configuration

**Step 4: Kubernetes Components**
```bash
# Version-locked installation
sudo apt-get install -y \
    kubelet=1.34.0-1.1 \
    kubectl=1.34.0-1.1 \
    kubeadm=1.34.0-1.1

# Prevent automatic upgrades
sudo apt-mark hold kubelet kubeadm kubectl
```

**Reasoning for Version Lock**: Ensures cluster stability, prevents unintended upgrades during research period.

### 3.2 Kubeadm Configuration

**Configuration File**: `/tmp/kubeadm-config.yaml`

**Key Design Decisions**:

1. **CRI Socket Specification**:
```yaml
criSocket: "unix:///var/run/crio/crio.sock"
```
Explicit socket prevents runtime ambiguity.

2. **Single Node Configuration**:
- Control plane endpoint: `172.22.174.58:6443`
- Node name: `controlplane`
- Taint removal for workload scheduling

3. **Network Configuration**:
```yaml
networking:
  podSubnet: "10.244.0.0/16"
  serviceSubnet: "10.96.0.0/12"
  dnsDomain: "cluster.local"
```

4. **Kubelet Configuration**:
```yaml
cgroupDriver: "systemd"
```
Critical for CRI-O compatibility.

5. **Removed Deprecated Fields**:
- `tcpCloseWaitTimeout` (no longer supported in v1.34)
- `tcpEstablishedTimeout` (no longer supported in v1.34)

**Version Compatibility Issue Resolved**: Initial config specified Kubernetes v1.32.0 but installed kubeadm was v1.34, causing initialization failure.

### 3.3 Cluster Initialization

```bash
sudo kubeadm init --config=/tmp/kubeadm-config.yaml
```

**Post-Init Configuration**:
1. Kubeconfig setup for user access
2. Control plane taint removal (single-node requirement)
3. Calico installation
4. Metrics server deployment

**Verification**:
- All system pods Running
- Node status: Ready
- CoreDNS operational

---

## Day 1: Reboot Test #1 Results

### 4.1 Test Procedure

1. Verified cluster operational state
2. Executed `sudo reboot`
3. Waited 3 minutes for system recovery
4. Checked cluster status

### 4.2 Observations

**Successful Components**:
 CRI-O service auto-started
 Kubelet service auto-started
 Static pods (API server, etcd, scheduler, controller-manager) recovered
 DaemonSet pods (Calico, CoreDNS) restarted
 Swap remained disabled (0B)
 Network connectivity maintained

**Issue Encountered**: Kubeconfig file permission error
```
error: open /home/hamid/.kube/config: permission denied
```

**Root Cause**: During reboot, file permissions changed (likely from root-owned kubeconfig copy).

**Solution**:
```bash
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

**Lesson Learned**: Include permission fix in setup script to prevent post-reboot issues.

### 4.3 Reboot Stability Assessment

**Result**:  PASSED

All cluster components recovered successfully. Minor permission issue is cosmetic and easily fixed.

---

## Day 2: GPU Integration Architecture

### 5.1 Requirements Analysis

**Research Objective**: Deploy GPU-accelerated AI inference services for performance analysis.

**Hardware**: NVIDIA A16 GPU
- Compute Capability: 8.6 (Ampere architecture)
- Memory: 16GB GDDR6
- Driver Version: 580.95.05

**Software Requirements**:
1. NVIDIA Container Toolkit
2. GPU device plugin for Kubernetes
3. RuntimeClass configuration
4. Proper CRI-O integration

### 5.2 GPU Integration Strategy for CRI-O

**Challenge**: Most documentation covers Containerd, not CRI-O.

**Architecture Decision**: Use NVIDIA Container Toolkit with CRI-O runtime configuration.

**Key Components**:
1. **NVIDIA Container Runtime**: `/usr/bin/nvidia-container-runtime`
2. **CRI-O Runtime Handler**: `nvidia`
3. **Kubernetes RuntimeClass**: Maps pod requests to runtime handler
4. **Device Plugin**: Advertises GPU resources to kubelet

---

## Day 2: GPU Integration Implementation

### 6.1 NVIDIA Container Toolkit Installation

**Installation**:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

**Verification**: `nvidia-container-runtime --version`

### 6.2 CRI-O Runtime Configuration Challenge

**Initial Approach**: Manual runtime configuration
```toml
[crio.runtime.runtimes.nvidia]
runtime_path = "/usr/bin/nvidia-container-runtime"
runtime_type = "oci"
runtime_root = "/run/nvidia-container-runtime"
```

**Problem #1**: CRI-O failed to start
```
level=fatal msg="validating runtime config: monitor fields translation: 
failed to translate monitor fields for runtime nvidia: 
exec: 'conmon': executable file not found in $PATH"
```

**Root Cause**: NVIDIA runtime configuration missing `monitor_path` for conmon process manager.

**Solution #1**: Add monitor_path
```toml
monitor_path = "/usr/libexec/crio/conmon"
```

**Problem #2**: NVIDIA Container Toolkit couldn't find base runtime
```
error constructing low-level runtime: 
error locating runtime: no runtime binary found from candidate list: [runc crun]
```

**Root Cause Analysis**:
- CRI-O stores runc/crun in `/usr/libexec/crio/`
- NVIDIA Container Runtime expects them in standard PATH locations
- No automatic path resolution

**Solution #2**: Create symbolic links
```bash
sudo ln -sf /usr/libexec/crio/runc /usr/bin/runc
sudo ln -sf /usr/libexec/crio/crun /usr/bin/crun
```

**Reasoning**: Symlinks provide compatibility without modifying CRI-O installation.

### 6.3 CDI (Container Device Interface) Configuration

**Purpose**: Modern device specification format for container runtimes.

**Generation**:
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

**What CDI Provides**:
- Device node specifications (/dev/nvidia*)
- Required libraries and binaries
- Hook configurations for device setup

**Integration**: CRI-O automatically reads CDI specs from `/etc/cdi/`.

### 6.4 Kubernetes RuntimeClass Configuration

**File**: `nvidia-runtimeclass.yaml`
```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
```

**Purpose**: Allows pods to request GPU-enabled runtime via `runtimeClassName: nvidia`.

**Critical Realization**: We forgot to apply this initially, causing device plugin failures.

### 6.5 NVIDIA Device Plugin Deployment

**Challenge**: Device plugin couldn't detect GPUs through CRI-O.

**Error Message**:
```
E1202 23:34:37.031998 factory.go:87] Incompatible strategy detected auto
I1202 23:34:37.032040 main.go:346] No devices found. Waiting indefinitely.
```

**Troubleshooting Process**:

**Attempt 1**: CDI discovery strategy
- Device plugin v0.16.2 doesn't support CDI strategy yet
- Error: `invalid --device-discovery-strategy option cdi`

**Attempt 2**: NVML discovery strategy
- Device plugin pod itself needs GPU runtime access
- Added `runtimeClassName: nvidia` to device plugin pod spec
- Success!

**Final Working Configuration**:
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  template:
    spec:
      runtimeClassName: nvidia  # Critical!
      containers:
      - name: nvidia-device-plugin-ctr
        image: nvcr.io/nvidia/k8s-device-plugin:v0.16.2
        env:
        - name: DEVICE_DISCOVERY_STRATEGY
          value: "nvml"
        securityContext:
          privileged: true
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
        - name: dev
          mountPath: /dev
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
      - name: dev
        hostPath:
          path: /dev
```

**Key Elements**:
1. **runtimeClassName**: Device plugin itself runs with GPU runtime
2. **NVML strategy**: Uses NVIDIA Management Library for discovery
3. **/dev mount**: Direct access to device nodes
4. **privileged**: Required for device management

---

## Day 2: GPU Verification and Testing

### 7.1 GPU Registration Verification

**Command**: `kubectl describe node | grep nvidia.com/gpu`

**Output**:
```
Capacity:
  nvidia.com/gpu:     1
Allocatable:
  nvidia.com/gpu:     1
Allocated resources:
  nvidia.com/gpu     0           0
```

**Interpretation**:
- 1 GPU detected and registered
- 1 GPU available for allocation
- 0 GPUs currently allocated

### 7.2 GPU Test Pod

**Configuration**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  runtimeClassName: nvidia  # Required for GPU access
  containers:
  - name: cuda-test
    image: nvidia/cuda:12.2.2-base-ubuntu22.04
    command: ["bash", "-c", "nvidia-smi && echo 'GPU Test SUCCESS!'"]
    resources:
      limits:
        nvidia.com/gpu: 1  # GPU request
```

**Result**:  SUCCESS

**nvidia-smi Output**:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA A16                     Off |   00000000:00:10.0 Off |                    0 |
|  0%   34C    P8             12W /   62W |      14MiB /  15356MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

**Key Observations**:
- GPU accessible from containerized environment
- Driver version matches host (580.95.05)
- CUDA 13.0 available
- Memory: 15GB available
- Power consumption: Idle state (12W/62W)

---

## Day 2: Reboot Test #2 - GPU Persistence

### 8.1 Test Procedure

1. Verified GPU registration
2. Executed `sudo reboot`
3. Waited 3 minutes for system recovery
4. Checked GPU status

### 8.2 Results

**System Recovery**:
 CRI-O started with NVIDIA runtime configured
 Kubelet registered with GPU device plugin
 Device plugin DaemonSet restarted automatically
 GPU re-registered successfully

**Verification**:
```bash
kubectl describe node | grep nvidia.com/gpu
# Output: nvidia.com/gpu: 1
```

**Test Pod Re-deployment**:
- GPU test pod successfully scheduled
- nvidia-smi accessible in container
- GPU fully operational

**Result**:  GPU configuration persists across reboots

---

## Technical Architecture Summary

### 9.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Host System (Ubuntu 24.04)           │
│  ┌────────────────────────────────────────────────────┐ │
│  │ NVIDIA Driver 580.95.05 + CUDA 13.0                │ │
│  │ /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm      │ │
│  └────────────────────────────────────────────────────┘ │
│                          ↓                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ NVIDIA Container Toolkit                           │ │
│  │ - nvidia-container-runtime                         │ │
│  │ - nvidia-container-cli                             │ │
│  │ - CDI specs: /etc/cdi/nvidia.yaml                  │ │
│  └────────────────────────────────────────────────────┘ │
│                          ↓                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ CRI-O Runtime 1.31.5                               │ │
│  │ - Config: /etc/crio/crio.conf.d/99-nvidia.toml     │ │
│  │ - nvidia runtime handler                           │ │
│  │ - runc/crun symlinks in /usr/bin                   │ │
│  └────────────────────────────────────────────────────┘ │
│                          ↓                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Kubelet 1.34.0                                     │ │
│  │ - Device plugin registration                       │ │
│  │ - GPU resource management                          │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│           Kubernetes Control Plane (Single Node)        │
│  ┌────────────────────────────────────────────────────┐ │
│  │ API Server + etcd + Scheduler + Controller Manager │ │
│  └────────────────────────────────────────────────────┘ │
│                          ↓                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ NVIDIA Device Plugin DaemonSet                     │ │
│  │ - RuntimeClass: nvidia                             │ │
│  │ - Discovery: NVML                                  │ │
│  │ - Advertises: nvidia.com/gpu: 1                    │ │
│  └────────────────────────────────────────────────────┘ │
│                          ↓                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ AI Workload Pods                                   │ │
│  │ - runtimeClassName: nvidia                         │ │
│  │ - resources.limits.nvidia.com/gpu: 1               │ │
│  │ - Direct GPU access via /dev/nvidia*               │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 9.2 Configuration Files Summary

**Persistent Configuration Files**:

1. **CRI-O GPU Runtime**: `/etc/crio/crio.conf.d/99-nvidia.toml`
```toml
[crio]
  [crio.runtime]
    [crio.runtime.runtimes]
      [crio.runtime.runtimes.nvidia]
        runtime_path = "/usr/bin/nvidia-container-runtime"
        runtime_type = "oci"
        runtime_root = "/run/nvidia-container-runtime"
        monitor_path = "/usr/libexec/crio/conmon"
```

2. **NVIDIA Runtime Config**: `/etc/nvidia-container-runtime/config.toml`
```toml
[nvidia-container-runtime]
  runtimes = ["/usr/libexec/crio/runc", "/usr/libexec/crio/crun"]
  
[nvidia-container-cli]
  debug = "/var/log/nvidia-container-toolkit.log"
```

3. **CDI Specification**: `/etc/cdi/nvidia.yaml`
- Auto-generated device specifications
- Contains device nodes, hooks, and library paths

4. **Kernel Modules**: `/etc/modules-load.d/k8s.conf`
```
overlay
br_netfilter
```

5. **Network Parameters**: `/etc/sysctl.d/k8s.conf`
```
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
```

6. **Swap Disable**: Managed via:
- `/etc/fstab` (commented swap entries)
- User crontab (@reboot hook)

7. **Symbolic Links**:
- `/usr/bin/runc` → `/usr/libexec/crio/runc`
- `/usr/bin/crun` → `/usr/libexec/crio/crun`

---

## Lessons Learned and Best Practices

### 10.1 Container Runtime Selection

**Lesson**: CRI-O requires more initial configuration than Containerd but provides better OCI compliance.

**Best Practice**: For Ubuntu 24.04 + Kubernetes 1.34, use CRI-O 1.31 with explicit version matching.

### 10.2 Reboot Stability

**Lesson**: Multiple redundant mechanisms (fstab + crontab + sysctl) ensure reliability in dynamic environments.

**Best Practice**: Test reboot stability early in development cycle, not during production deployment.

### 10.3 GPU Integration with CRI-O

**Critical Insights**:
1. Device plugin itself needs GPU runtime access (`runtimeClassName: nvidia`)
2. RuntimeClass must be applied before device plugin deployment
3. Symlinks required for nvidia-container-runtime to find base runtimes
4. conmon monitor path must be explicit in CRI-O runtime configuration

**Best Practice**: Follow this sequence:
1. Install NVIDIA drivers
2. Install nvidia-container-toolkit
3. Configure CRI-O runtime
4. Apply RuntimeClass
5. Deploy device plugin
6. Verify GPU registration
7. Test with sample pod

### 10.4 Debugging Methodology

**Effective Approach**:
1. Check pod status and events
2. Examine pod logs
3. Review kubelet logs (`journalctl -u kubelet`)
4. Verify CRI-O logs (`journalctl -u crio`)
5. Test host-level GPU access (`nvidia-smi`)
6. Validate runtime configuration (`crictl info`)

---

## Final Configuration Checklist

### 11.1 Verification Commands

**Cluster Health**:
```bash
kubectl get nodes                           # Should show: Ready
kubectl get pods -A                         # All Running
kubectl cluster-info                        # Endpoints accessible
kubectl top nodes                           # Metrics available
```

**GPU Health**:
```bash
nvidia-smi                                  # Host GPU visible
kubectl describe node | grep nvidia.com/gpu # GPU registered
kubectl get runtimeclass                    # nvidia handler exists
kubectl get ds -n kube-system | grep nvidia # Device plugin running
```

**Reboot Persistence**:
```bash
free -h | grep Swap                         # Should show: 0B
systemctl is-enabled crio kubelet           # Both: enabled
ls -la /usr/bin/runc /usr/bin/crun         # Symlinks present
cat /etc/crio/crio.conf.d/99-nvidia.toml   # Config exists
```

### 11.2 Ready for Production Research

**Current Status**:  Production Ready

**Capabilities**:
-  Stable single-node Kubernetes cluster
-  Survives system reboots automatically
-  GPU accessible to containerized workloads
-  Metrics collection operational
-  Network policies supported (Calico)
-  Ready for AI inference deployment

**Next Steps for Research**:
1. Deploy ResNet50 inference service
2. Deploy DistilBERT inference service
3. Deploy Whisper inference service
4. Implement workload monitoring
5. Conduct performance analysis experiments

---

## Appendix: Complete Command Reference

### A.1 Cluster Setup Commands

```bash
# System preparation
sudo swapoff -a
(crontab -l 2>/dev/null; echo "@reboot /sbin/swapoff -a") | crontab -
sudo sed -i 's|^/swap.img|#/swap.img|g' /etc/fstab

# Kernel modules
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF
sudo modprobe overlay
sudo modprobe br_netfilter

# Network parameters
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF
sudo sysctl --system

# CRI-O installation
curl -fsSL https://pkgs.k8s.io/addons:/cri-o:/stable:/v1.31/deb/Release.key | \
    sudo gpg --dearmor -o /etc/apt/keyrings/cri-o-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/cri-o-apt-keyring.gpg] https://pkgs.k8s.io/addons:/cri-o:/stable:/v1.31/deb/ /" | \
    sudo tee /etc/apt/sources.list.d/cri-o.list
sudo apt-get update
sudo apt-get install -y --reinstall cri-o
sudo systemctl enable crio --now

# Kubernetes installation
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.34/deb/Release.key | \
    sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.34/deb/ /" | \
    sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubelet=1.34.0-1.1 kubectl=1.34.0-1.1 kubeadm=1.34.0-1.1
sudo apt-mark hold kubelet kubeadm kubectl

# Cluster initialization
sudo kubeadm init --config=kubeadm-config.yaml
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Remove control-plane taint
kubectl taint nodes --all node-role.kubernetes.io/control-plane-

# Install Calico
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/tigera-operator.yaml
curl https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/custom-resources.yaml -O
sed -i 's|cidr: 192.168.0.0/16|cidr: 10.244.0.0/16|g' custom-resources.yaml
kubectl apply -f custom-resources.yaml

# Install metrics server
kubectl apply -f https://raw.githubusercontent.com/techiescamp/cka-certification-guide/refs/heads/main/lab-setup/manifests/metrics-server/metrics-server.yaml
```

### A.2 GPU Setup Commands

```bash
# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# CRI-O runtime configuration
cat <<EOF | sudo tee /etc/crio/crio.conf.d/99-nvidia.toml
[crio]
  [crio.runtime]
    [crio.runtime.runtimes]
      [crio.runtime.runtimes.nvidia]
        runtime_path = "/usr/bin/nvidia-container-runtime"
        runtime_type = "oci"
        runtime_root = "/run/nvidia-container-runtime"
        monitor_path = "/usr/libexec/crio/conmon"
EOF

# Runtime symlinks
sudo ln -sf /usr/libexec/crio/runc /usr/bin/runc
sudo ln -sf /usr/libexec/crio/crun /usr/bin/crun

# CDI generation
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Restart CRI-O
sudo systemctl restart crio

# Apply RuntimeClass
kubectl apply -f nvidia-runtimeclass.yaml

# Deploy device plugin
kubectl apply -f nvidia-device-plugin-daemonset.yaml
```

---

## Conclusion

This journal documents a complete, production-ready Kubernetes cluster setup optimized for AI research workloads. The configuration balances:

- **Stability**: Survives reboots without manual intervention
- **Performance**: Direct GPU access for AI inference
- **Simplicity**: Single-node design suitable for research environment
- **Reliability**: Multiple redundancy mechanisms for critical configuration

The setup is now ready for the deployment and analysis of AI inference workloads as part of the Master's thesis on Digital Transformation at Fachhochschule Dortmund.

**Total Setup Time**: ~8 hours (including troubleshooting and documentation)  
**Final Status**: Production Ready  
**Date Completed**: December 3, 2025



---

## December 9, 2025 - Monitoring Stack Deployment

### Objective
Deploy production-ready monitoring infrastructure for collecting AI workload performance metrics (Phase 1 of thesis).

### Work Completed

#### 1. Monitoring Stack Components Deployed
**Core Infrastructure:**
- **Prometheus** (v2.48.0)
  - 50GB persistent storage
  - 30-day data retention
  - 15-second scrape interval
  - NodePort access: 30090

- **Grafana** (v10.2.2)
  - 10GB persistent storage
  - Auto-configured Prometheus datasource
  - NodePort access: 30030
  - Default credentials: admin/admin

**Metrics Exporters:**
- **Node Exporter** (v1.7.0)
  - Host-level metrics: CPU, RAM, disk, network
  - PSI (Pressure Stall Information) enabled for load detection
  - DaemonSet deployment (runs on all nodes)

- **DCGM Exporter** (v3.3.5)
  - GPU metrics: utilization, memory, temperature, power
  - NVIDIA A16 GPU monitoring
  - Requires `nvidia.com/gpu=true` node label

- **kube-state-metrics** (v2.10.1)
  - Kubernetes object state metrics
  - Pod, deployment, node status tracking

- **kubelet/cAdvisor** (built-in)
  - Container-level resource usage
  - Per-pod CPU, RAM, network metrics
  - Auto-scraped by Prometheus

#### 2. Storage Infrastructure
**Problem Encountered:**
- Initial deployment failed - Prometheus and Grafana pods stuck in "Pending"
- Root cause: No storage provisioner available for PersistentVolumeClaims

**Solution Implemented:**
- Installed Rancher local-path-provisioner (v0.0.24)
- Set as default StorageClass
- Created 50GB PVC for Prometheus data
- Created 10GB PVC for Grafana data
- Storage location: `/opt/local-path-provisioner/`

**Reboot Safety:**
- Storage provisioner auto-starts (DaemonSet)
- PVCs automatically rebind to existing volumes
- All data persists across reboots

#### 3. GPU Monitoring Configuration
**Issue #1: DCGM Exporter Not Scheduling**
- DaemonSet required `nvidia.com/gpu=true` node label
- Node only had GPU capacity, not label
- Result: "Desired Number of Nodes Scheduled: 0"

**Solution:**
```bash
kubectl label nodes controlplane nvidia.com/gpu=true
```

**Issue #2: Missing RuntimeClass**
- DCGM Exporter needs GPU runtime access
- Added `runtimeClassName: nvidia` to pod spec
- Enabled privileged mode for device access

#### 4. Kepler Power Monitoring (Skipped)
**Attempted Deployment:**
- Kepler v0.7.10 for power consumption metrics
- Required RAPL (Running Average Power Limit) zones

**Failure Analysis:**
```
Error: "failed to initialize service rapl: no RAPL zones found"
```

**Root Cause:**
- AMD EPYC 7643 CPU
- Kernel module `amd_energy` not available in kernel 6.14.0-36
- `/sys/class/powercap/` empty

**Decision:**
- Skipped Kepler deployment
- Power metrics are optional in thesis requirements
- Can revisit later if needed
- Core metrics (CPU, RAM, GPU) sufficient for Phase 1

#### 5. Deployment Script Enhancements
**Updated `scripts/monitoring/deploy-monitoring-stack.sh`:**

**New Step 0: Storage Provisioner Check**
```bash
if ! kubectl get storageclass local-path &>/dev/null; then
    # Install local-path-provisioner
    # Set as default
fi
```

**New Step 3.5: GPU Label Verification**
```bash
GPU_NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
if ! kubectl get node $GPU_NODE -o jsonpath='{.metadata.labels.nvidia\.com/gpu}' | grep -q "true"; then
    kubectl label nodes $GPU_NODE nvidia.com/gpu=true
fi
```

**Fixed Typo:**
- Line 68: `bectl wait` → `kubectl wait`

#### 6. System Configuration
**fstab Cleanup:**
- Removed duplicate swap entries
- Added debugfs mount for eBPF support: `debugfs /sys/kernel/debug debugfs defaults 0 0`
- Persists across reboots

### Technical Decisions

#### Why Local-Path Provisioner?
-  Lightweight, perfect for single-node clusters
-  No external dependencies
-  Simple local storage on host filesystem
-  Sufficient for research/development workloads

#### Why Skip Kepler?
-  RAPL unavailable on current kernel/CPU combination
-  Power metrics are optional per thesis requirements
-  Core metrics (CPU, RAM, GPU, latency) are sufficient
-  Can implement power estimation later if needed

#### Node Label vs Removing nodeSelector?
-  Kept nodeSelector in DCGM YAML (proper approach)
-  Added GPU label to node (Kubernetes best practice)
-  Allows future multi-node expansion
-  Clear separation of GPU vs non-GPU nodes

### Verification

#### All Pods Running:
```
NAME                                  READY   STATUS    RESTARTS   AGE
dcgm-exporter-xxxxx                   1/1     Running   0          Xm
grafana-xxxxx                         1/1     Running   0          Xm
kube-state-metrics-xxxxx              1/1     Running   0          Xm
node-exporter-xxxxx                   1/1     Running   0          Xm
prometheus-xxxxx                      1/1     Running   0          Xm
```

#### Prometheus Targets (All UP):
-  prometheus (self-monitoring)
-  node-exporter
-  dcgm-exporter
-  kube-state-metrics
-  kubelet
-  kubelet-cadvisor

#### Access URLs:
- Prometheus: http://172.22.174.58:30090
- Grafana: http://172.22.174.58:30030

### Metrics Available for Thesis

#### Resource Consumption:
| Metric | Source | Query Example |
|--------|--------|---------------|
| CPU utilization | Node Exporter | `node_cpu_seconds_total` |
| RAM consumption | Node Exporter | `node_memory_MemAvailable_bytes` |
| GPU utilization | DCGM Exporter | `dcgm_gpu_utilization` |
| GPU memory | DCGM Exporter | `dcgm_fb_used_bytes` |
| Container CPU | kubelet/cAdvisor | `container_cpu_usage_seconds_total` |
| Container RAM | kubelet/cAdvisor | `container_memory_usage_bytes` |

#### System State:
| Metric | Source | Query Example |
|--------|--------|---------------|
| Load detection (PSI) | Node Exporter | `node_pressure_cpu_waiting_seconds_total` |
| Pod status | kube-state-metrics | `kube_pod_status_phase` |
| Node capacity | kube-state-metrics | `kube_node_status_capacity` |

#### QoS Metrics (To Be Implemented):
- Application latency (custom metrics from inference apps)
- Request throughput
- Queue depth

### Reboot Stability Verification

**Components that auto-recover:**
-  Kubernetes cluster (systemd services)
-  Storage provisioner (DaemonSet)
-  All monitoring pods (Deployments/DaemonSets)
-  Persistent volumes (data on disk)
-  Node labels (stored in etcd)

**Post-reboot checklist:**
```bash
kubectl get nodes                    # Should be Ready
kubectl get pods -n monitoring       # All Running
kubectl get pvc -n monitoring        # All Bound
kubectl get storageclass             # local-path exists
```

### Lessons Learned

1. **Storage provisioner is essential** - Always deploy before stateful applications
2. **Node labels vs capacity** - GPU capacity doesn't equal GPU label
3. **RAPL availability varies** - AMD/Intel, kernel version dependent
4. **Test end-to-end** - Delete namespace and redeploy to verify scripts
5. **Persistent configuration** - Labels and storage bindings survive reboots

### Next Steps

**Immediate (Tomorrow):**
1. Create Grafana dashboards for thesis metrics
2. Verify Prometheus is collecting all target metrics
3. Test data export capabilities

**Phase 1 Continuation:**
1. Deploy AI inference workloads (ResNet50, DistilBERT, Whisper)
2. Instrument applications with latency metrics
3. Collect baseline performance data (uncontended state)
4. Generate load scenarios (modest, high)
5. Export time-series data for model training

**Optional:**
- Revisit Kepler if power metrics become critical
- Add alerting rules (Alertmanager)
- Implement deep observability (Pixie) if needed

### Status

**Phase 1: Workload Setup**
-  Kubernetes cluster operational
-  GPU support enabled
-  Monitoring infrastructure deployed ← **COMPLETED TODAY**
-  AI workloads deployment (next)
-  Data collection experiments (after workloads)

**Infrastructure Maturity:** Production-ready for research workload analysis

---
```

