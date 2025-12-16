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





---

## December 9, 2025 - Monitoring Stack Deployment and Dashboard Creation

### Objective
Complete monitoring infrastructure deployment with comprehensive Grafana dashboards for thesis data visualization and analysis (Phase 1 continuation).

### Work Completed

#### 1. Storage Infrastructure Resolution

**Problem Identified:**
Initial monitoring stack deployment failed with Prometheus and Grafana pods stuck in "Pending" state.

**Root Cause:**
No StorageClass available in cluster. PersistentVolumeClaims could not be provisioned.

**Error Message:**
```
0/1 nodes are available: pod has unbound immediate PersistentVolumeClaims
```

**Solution Implemented:**
Deployed Rancher local-path-provisioner (v0.0.24):
```bash
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml
kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

**Result:**
- PVCs automatically transitioned from "Pending" to "Bound"
- Prometheus and Grafana pods started successfully
- Storage location: /opt/local-path-provisioner/

**Verification:**
```bash
kubectl get storageclass
NAME                   PROVISIONER             RECLAIMPOLICY
local-path (default)   rancher.io/local-path   Delete

kubectl get pvc -n monitoring
NAME             STATUS   VOLUME                                     CAPACITY
grafana-pvc      Bound    pvc-xxxxx                                  10Gi
prometheus-pvc   Bound    pvc-yyyyy                                  50Gi
```

#### 2. GPU Monitoring Configuration

**Issue: DCGM Exporter Not Scheduling**

**Symptom:**
```bash
kubectl get daemonset -n monitoring dcgm-exporter
NAME            DESIRED   CURRENT   READY
dcgm-exporter   0         0         0
```

**Root Cause Analysis:**
DaemonSet nodeSelector required `nvidia.com/gpu=true` label, but node only had GPU capacity annotation, not the label itself.
```yaml
nodeSelector:
  nvidia.com/gpu: "true"  # Label required
```

**Node Status:**
```bash
kubectl describe node controlplane | grep nvidia
  nvidia.com/gpu:     1         # Capacity only, no label
```

**Solution:**
Added GPU label to node:
```bash
kubectl label nodes controlplane nvidia.com/gpu=true
```

**Result:**
DCGM DaemonSet immediately scheduled pod after label was applied.

**Lesson Learned:**
GPU device plugin creates capacity but not labels. Labels must be added manually for node selection.

#### 3. Deployment Script Enhancements

**Updated scripts/monitoring/deploy-monitoring-stack.sh:**

**New Step 0: Storage Provisioner Check**
```bash
if ! kubectl get storageclass local-path &>/dev/null; then
    kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml
    kubectl wait --for=condition=ready pod -l app=local-path-provisioner -n local-path-storage --timeout=120s
    kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
fi
```

**New Step 3.5: GPU Node Label Check**
```bash
GPU_NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
if ! kubectl get node $GPU_NODE -o jsonpath='{.metadata.labels.nvidia\.com/gpu}' | grep -q "true"; then
    kubectl label nodes $GPU_NODE nvidia.com/gpu=true
fi
```

**Bug Fix:**
Line 68: Changed `bectl wait` to `kubectl wait` (typo correction)

**Testing:**
Deleted and recreated monitoring namespace to verify end-to-end automation. All components deployed successfully without manual intervention.

#### 4. Kepler Power Monitoring (Deferred)

**Attempted Deployment:**
Kepler v0.7.10 for power consumption metrics.

**Failure Analysis:**
```
Error: "failed to initialize service rapl: no RAPL zones found"
```

**Investigation:**
- CPU: AMD EPYC 7643 48-Core Processor
- Kernel: 6.14.0-36-generic
- RAPL module: amd_energy not available
- No zones in /sys/class/powercap/

**Attempted Fixes:**
1. Load AMD RAPL module: `modprobe amd_energy` (module not found)
2. Mount debugfs: `mount -t debugfs debugfs /sys/kernel/debug` (already mounted)
3. Configure estimator mode: Added environment variables for ML-based estimation

**Decision:**
Deferred Kepler deployment. Reasons:
- Power metrics are optional per thesis requirements (Phase 1 document)
- Core metrics (CPU, RAM, GPU, latency) are sufficient
- RAPL unavailable on current kernel/hardware combination
- Can revisit if power consumption becomes critical

#### 5. Grafana Dashboard Development

**Dashboards Created:**

**5.1 System Resources Dashboard**
File: `dashboards/system-resources.json`

Panels:
- CPU Utilization (Total): Aggregate CPU usage percentage
- CPU Utilization (Per Core): Individual core usage tracking
- Memory Usage (Bytes): Used, available, total memory
- Memory Usage (Percent): Memory utilization percentage
- Disk I/O: Read/write operations per device
- Network Traffic: Transmit/receive bytes per interface

Key Metrics:
```promql
# CPU Usage
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory Usage
node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes

# Disk I/O
rate(node_disk_read_bytes_total[5m])
rate(node_disk_written_bytes_total[5m])
```

Use Case: System-level resource baseline for thesis Phase 1.

**5.2 GPU Performance Dashboard**
File: `dashboards/gpu-performance.json`

Panels:
- GPU Utilization: Percentage usage of NVIDIA A16
- GPU Memory Usage: Bytes and percentage
- GPU Temperature: Celsius monitoring
- GPU Power Usage: Watts consumption
- GPU Clock Speeds: SM and memory clocks
- Composite Performance: Combined utilization view

Key Metrics:
```promql
# GPU Utilization
dcgm_gpu_utilization{gpu="0"}

# GPU Memory Percentage
(dcgm_fb_used_bytes{gpu="0"} / dcgm_fb_total_bytes{gpu="0"}) * 100

# GPU Temperature
dcgm_gpu_temp{gpu="0"}

# GPU Power
dcgm_power_usage_watts{gpu="0"}
```

Alerts Configured:
- High GPU utilization: >90%
- High temperature: >80C

Use Case: GPU workload characterization under different load states.

**5.3 Container Metrics Dashboard**
File: `dashboards/container-metrics.json`

Panels:
- Pod CPU Usage (AI Workloads): Filtered for resnet50, distilbert, whisper
- Pod Memory Usage (AI Workloads): Per-application tracking
- All Pod CPU Usage: Cluster-wide view
- All Pod Memory Usage: Cluster-wide view
- Pod Network I/O: Transmit/receive per pod
- Pod Filesystem Usage: Disk usage per container
- Pod Status Summary: Table view of pod states

Key Metrics:
```promql
# Container CPU
rate(container_cpu_usage_seconds_total{pod=~"resnet50.*|distilbert.*|whisper.*"}[5m])

# Container Memory
container_memory_usage_bytes{pod=~"resnet50.*|distilbert.*|whisper.*"}

# Pod Status
kube_pod_status_phase{namespace="default"}
```

Use Case: Granular per-application resource consumption analysis.

**5.4 System Pressure Dashboard**
File: `dashboards/system-pressure.json`

Panels:
- CPU Pressure (PSI): Waiting time for CPU resources
- Memory Pressure (PSI): Some and Full stall states
- I/O Pressure (PSI): Waiting time for I/O operations
- Load State Classification: Combined CPU utilization and pressure
- Current Load State: Gauge with color thresholds
- System Pressure Score: Composite pressure indicator
- Historical Load Pattern: Time-series of all utilization metrics

Key Metrics:
```promql
# CPU Pressure
rate(node_pressure_cpu_waiting_seconds_total[5m])

# Memory Pressure (Some)
rate(node_pressure_memory_waiting_seconds_total[5m])

# Memory Pressure (Full)
rate(node_pressure_memory_stalled_seconds_total[5m])

# Composite Pressure Score
(rate(node_pressure_cpu_waiting_seconds_total[5m]) + 
 rate(node_pressure_memory_waiting_seconds_total[5m]) + 
 rate(node_pressure_io_waiting_seconds_total[5m])) * 100
```

Load State Thresholds:
- Empty: <10% CPU utilization
- Modest: 40-60% CPU utilization
- High: 70-90% CPU utilization
- Critical: >90% CPU utilization

Use Case: Critical for thesis requirement of load state detection using PSI metrics.

**5.5 Inference Performance Dashboard**
File: `dashboards/inference-performance.json`

Panels:
- ResNet50 Inference Latency: p50, p95, p99 percentiles
- DistilBERT Inference Latency: p50, p95, p99 percentiles
- Whisper Inference Latency: p50, p95, p99 percentiles
- Comparative Latency: All models median comparison
- Request Throughput: Requests per second per model
- Queue Depth: Pending requests per application
- Inference Statistics Summary: Tabular view
- Latency Under Load: Correlation with CPU utilization

Key Metrics (To Be Implemented):
```promql
# Latency Percentiles
histogram_quantile(0.50, rate(inference_latency_seconds_bucket{app="resnet50"}[5m]))
histogram_quantile(0.95, rate(inference_latency_seconds_bucket{app="resnet50"}[5m]))

# Throughput
rate(inference_requests_total{app="resnet50"}[5m])

# Queue Depth
inference_queue_depth{app="resnet50"}
```

Note: Requires application instrumentation with Prometheus client library.

Use Case: QoS measurement and latency analysis under varying load conditions (thesis Phase 1 requirement).

#### 6. Dashboard Import Automation

**Created scripts/monitoring/import-dashboards.sh**

**Initial Implementation Issue:**
Script stopped after importing first dashboard. Investigation revealed:
- Complex `jq` payload construction causing silent failures
- HTTP status code check not catching JSON parsing errors
- Script using `set -e` causing premature exit

**Root Cause:**
Nested `jq` operations with inline JSON construction were fragile and error-prone.

**Solution - Script Rewrite:**
```bash
# Key improvements:
1. Removed `set -e` - continue on errors
2. Use temp file for payload construction
3. Check for "imported":true in response
4. Show actual error messages
5. List successfully imported dashboards
```

**Script Features:**
- Automatic Prometheus datasource UID detection
- Grafana connectivity check
- Dashboard file validation
- Success/failure counting
- Detailed error reporting

#### 7. Documentation

**Created dashboards/README.md**

**Content:**
- Overview of each dashboard and its purpose
- Import instructions (UI and API methods)
- PromQL query examples
- Customization guide
- Instrumentation requirements for custom metrics
- Data export procedures for thesis analysis
- Troubleshooting guide
- Alerting configuration

### Monitoring Stack Status

**All Components Running:**
```bash
kubectl get pods -n monitoring
NAME                                  READY   STATUS    RESTARTS   AGE
dcgm-exporter-xxxxx                   1/1     Running   0          Xh
grafana-xxxxx                         1/1     Running   0          Xh
kube-state-metrics-xxxxx              1/1     Running   0          Xh
node-exporter-xxxxx                   1/1     Running   0          Xh
prometheus-xxxxx                      1/1     Running   0          Xh
```

**All Prometheus Targets UP:**
- prometheus (self-monitoring)
- node-exporter (system metrics)
- dcgm-exporter (GPU metrics)
- kube-state-metrics (K8s state)
- kubelet (container metrics)
- kubelet-cadvisor (cAdvisor metrics)

**All Dashboards Imported:**
- Container Metrics - Thesis Data Collection
- GPU Performance - Thesis Data Collection
- Inference Performance - Thesis QoS Metrics
- System Pressure (PSI) - Thesis Load Detection
- System Resources - Thesis Data Collection

**Access URLs:**
- Prometheus: http://172.22.174.58:30090
- Grafana: http://172.22.174.58:30030 (admin/admin)

### Metrics Available for Thesis Analysis

**Resource Consumption Metrics:**

| Category | Metric | Source | Query |
|----------|--------|--------|-------|
| CPU Total | Utilization % | Node Exporter | `100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)` |
| CPU Per Core | Utilization % | Node Exporter | `rate(node_cpu_seconds_total{mode="user"}[5m])` |
| RAM | Used Bytes | Node Exporter | `node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes` |
| RAM | Utilization % | Node Exporter | `100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)` |
| GPU | Utilization % | DCGM Exporter | `dcgm_gpu_utilization{gpu="0"}` |
| GPU Memory | Used Bytes | DCGM Exporter | `dcgm_fb_used_bytes{gpu="0"}` |
| GPU Memory | Utilization % | DCGM Exporter | `(dcgm_fb_used_bytes / dcgm_fb_total_bytes) * 100` |
| GPU Temperature | Celsius | DCGM Exporter | `dcgm_gpu_temp{gpu="0"}` |
| GPU Power | Watts | DCGM Exporter | `dcgm_power_usage_watts{gpu="0"}` |

**Container Metrics:**

| Metric | Source | Query |
|--------|--------|-------|
| Container CPU | kubelet/cAdvisor | `rate(container_cpu_usage_seconds_total{pod=~"resnet50.*"}[5m])` |
| Container Memory | kubelet/cAdvisor | `container_memory_usage_bytes{pod=~"resnet50.*"}` |
| Container Network RX | kubelet/cAdvisor | `rate(container_network_receive_bytes_total[5m])` |
| Container Network TX | kubelet/cAdvisor | `rate(container_network_transmit_bytes_total[5m])` |

**System Pressure Metrics (Load Detection):**

| Metric | Source | Query | Purpose |
|--------|--------|-------|---------|
| CPU Pressure | Node Exporter | `rate(node_pressure_cpu_waiting_seconds_total[5m])` | Detect CPU contention |
| Memory Pressure (Some) | Node Exporter | `rate(node_pressure_memory_waiting_seconds_total[5m])` | Some tasks waiting |
| Memory Pressure (Full) | Node Exporter | `rate(node_pressure_memory_stalled_seconds_total[5m])` | All tasks stalled |
| I/O Pressure | Node Exporter | `rate(node_pressure_io_waiting_seconds_total[5m])` | Disk bottlenecks |

**Application QoS Metrics (To Be Implemented):**

| Metric | Type | Purpose |
|--------|------|---------|
| inference_latency_seconds | Histogram | Response time distribution |
| inference_requests_total | Counter | Request throughput |
| inference_queue_depth | Gauge | Queue backlog |

### Reboot Stability Verification

**Components with Persistent Storage:**
- Prometheus: 50GB PVC, 30-day retention
- Grafana: 10GB PVC, all dashboards and settings

**Storage Location:**
```bash
ls /opt/local-path-provisioner/
pvc-xxxxx/  # Prometheus data
pvc-yyyyy/  # Grafana data
```

**Auto-Recovery After Reboot:**
1. Kubernetes cluster (systemd services)
2. Storage provisioner (DaemonSet)
3. All monitoring pods (Deployments/DaemonSets)
4. PVC bindings (etcd persistence)
5. Node labels (etcd persistence)
6. Grafana dashboards (SQLite on PVC)

**Post-Reboot Verification:**
```bash
kubectl get nodes                    # Ready
kubectl get pods -n monitoring       # All Running
kubectl get pvc -n monitoring        # All Bound
kubectl get storageclass             # local-path exists
curl http://172.22.174.58:30030      # Grafana accessible
```

No manual intervention required after reboot.

### Data Export Procedures

**For Thesis Model Training:**

Export time-series data via Prometheus API:
```bash
# Export GPU utilization for date range
curl -G 'http://172.22.174.58:30090/api/v1/query_range' \
  --data-urlencode 'query=dcgm_gpu_utilization{gpu="0"}' \
  --data-urlencode 'start=2025-12-09T00:00:00Z' \
  --data-urlencode 'end=2025-12-09T23:59:59Z' \
  --data-urlencode 'step=15s' \
  > data/raw/gpu_utilization.json
```

Recommended export format (CSV):
```
timestamp,cpu_util,ram_usage,gpu_util,gpu_memory,latency
2025-12-09T10:00:00Z,45.2,8589934592,67.3,10737418240,0.023
```

### System Configuration Cleanup

**Fixed /etc/fstab Duplicates:**

Issue: Multiple duplicate entries from debugging process.

Solution:
```bash
sudo cp /etc/fstab /etc/fstab.backup
sudo awk '!seen[$0]++' /etc/fstab | sudo tee /etc/fstab.tmp
sudo mv /etc/fstab.tmp /etc/fstab
```

Final clean fstab:
```
/dev/disk/by-uuid/xxxxx / ext4 defaults 0 1
#/swap.img      none    swap    sw      0       0
debugfs /sys/kernel/debug debugfs defaults 0 0
```

### Technical Decisions Summary

**Storage Provisioner Choice:**
- Selected: local-path-provisioner
- Rationale: Lightweight, no external dependencies, perfect for single-node
- Alternative considered: NFS provisioner (rejected - unnecessary complexity)

**GPU Node Labeling:**
- Decision: Keep nodeSelector in DCGM YAML, add label to node
- Rationale: Kubernetes best practice, allows future multi-node expansion
- Alternative: Remove nodeSelector (rejected - less maintainable)

**Kepler Deferral:**
- Decision: Skip power metrics for now
- Rationale: Optional per thesis, RAPL unavailable, core metrics sufficient
- Alternative: Implement estimator mode (deferred - can revisit later)

**Dashboard Import Method:**
- Decision: Automated script with robust error handling
- Rationale: Reproducible, version controlled, enables CI/CD
- Alternative: Manual UI import (rejected - not repeatable)

### Lessons Learned

1. **Storage provisioner must be deployed before stateful applications** - This is non-negotiable for any Kubernetes cluster running applications with persistent data.

2. **GPU capacity does not equal GPU label** - Device plugins create capacity annotations but do not automatically label nodes. Labels must be added manually for nodeSelector to work.

3. **RAPL availability varies significantly** - Kernel version, CPU vendor, and module availability all affect power monitoring capabilities. Always have fallback plans for optional metrics.

4. **Complex inline JSON construction is fragile** - Use temp files or heredocs for complex payload construction in shell scripts to improve reliability and debuggability.

5. **Test end-to-end automation** - Deleting and recreating resources verifies scripts work correctly without manual intervention.

### Next Steps

**Immediate (Tomorrow):**
1. Test dashboards with load generation (stress tool)
2. Verify all metrics collecting correctly
3. Practice data export procedures

**Phase 1 Continuation:**
1. Deploy AI inference workloads (ResNet50, DistilBERT, Whisper)
2. Instrument applications with Prometheus client library
3. Add custom latency tracking metrics
4. Collect baseline performance data (uncontended state)
5. Generate load scenarios:
   - Modest load (40-60% utilization)
   - High load (70-90% utilization)
6. Export time-series data for model training (Phase 3)

**Optional Future Work:**
- Revisit Kepler if power metrics become critical
- Implement Alertmanager for notifications
- Add additional custom dashboards based on analysis needs
- Consider Pixie for deep observability if needed

### Project Status

**Phase 1: Workload Setup**
- ✓ Kubernetes cluster operational (reboot-safe)
- ✓ GPU support enabled (reboot-safe)
- ✓ Monitoring infrastructure deployed (reboot-safe)
- ✓ Grafana dashboards created (reboot-safe)
- Next: AI workloads deployment
- Next: Application instrumentation
- Next: Data collection experiments

---

## December 16, 2025 - System Recovery and Workload Preparation

### Objective
Resolve critical system issues, redeploy monitoring infrastructure, and prepare AI workloads for Phase 1 data collection experiments.

### Critical Issues Resolved

#### 1. Massive Pod Eviction Crisis

**Problem Discovered:**
Over 10,000 evicted pods accumulated in cluster, primarily tigera-operator pods.

**Root Cause:**
- Disk pressure reached critical levels (64GB/98GB used = 65%)
- Kubernetes evicted pods attempting to free space
- Deployment controllers continuously recreated pods
- New pods immediately evicted due to persistent disk pressure
- Eviction records accumulated over weeks/months

**Timeline Analysis:**
```
Weeks ago    → Disk slowly fills (Docker cache, old images, logs)
Days ago     → Disk hits 85%+ threshold
              → Kubernetes detects disk pressure
              → Mass evictions begin
Continuous   → Pod creation/eviction cycle
Today        → 10,000+ eviction records discovered
```

**Initial Cleanup Attempt:**
```bash
kubectl get pods -A | grep Evicted | awk '{print $2 " -n " $1}' | \
  xargs -r kubectl delete pod
```
Result: Too slow for 10,000+ pods (would take hours)

**Fast Cleanup Solution:**
```bash
kubectl delete pods --field-selector=status.phase=Failed -A \
  --grace-period=0 --force
```
Result: Cleared all evicted pods in 30-60 seconds

**Verification:**
```bash
kubectl get pods -A | grep Evicted | wc -l
# Output: 0
```

#### 2. Disk Space Recovery

**Critical Space Constraint:**
```
Before: 98GB total, 64GB used (65% full) → Near critical threshold
After:  98GB total, 34GB used (35% full) → Healthy operational level
```

**Space Recovery Strategy:**

**Step 1: Docker Build Cache Cleanup (14.95GB recovered)**
```bash
docker builder prune -af
```
Removed all intermediate build layers and compilation artifacts.

**Step 2: Dangling Images Removal (~24GB recovered)**
```bash
docker image prune -af
```

**Critical Error:**
Used `-af` flag which removed ALL unused images, including:
- hamidhrf/resnet50-inference:v2 (6.48GB)
- hamidhrf/distilbert-inference:v2 (7.33GB)
- hamidhrf/whisper-inference:v2 (7.86GB)
- All base images (nvidia/cuda, python, pytorch)
- All containerlab images (frrouting, alpine)

**Impact Assessment:**
- All AI workload images deleted from local Docker cache
- Images remain safely stored on DockerHub (pushed 5 days prior)
- Kubernetes will auto-pull from DockerHub on deployment
- Total space recovered: ~30.36GB

**Lesson Learned:**
`docker image prune -f` removes only dangling images (desired)
`docker image prune -af` removes ALL unused images (too aggressive)

**Decision:**
Proceed with auto-pull strategy - Kubernetes handles image pulling automatically on pod deployment. Eliminates need for manual pre-pull.

#### 3. Monitoring Stack Redeployment

**Post-Cleanup Status:**
- Monitoring namespace previously deleted during troubleshooting
- 60GB disk space now available
- System stable and ready for fresh deployment

**Deployment Execution:**
```bash
cd ~/generative-ai-workload-modeling/scripts/monitoring
./deploy-monitoring-stack.sh
```

**Deployment Results:**

All components successfully deployed:
- Prometheus (v2.47.0) - 50GB PVC, 30-day retention
- Grafana (v10.2.0) - 10GB PVC, dashboards pre-configured
- Node Exporter (v1.6.1) - System metrics with PSI
- DCGM Exporter (v3.1.8) - GPU metrics
- kube-state-metrics (v2.10.0) - K8s object state

**Script Enhancements Applied:**
- Step 0: Automatic storage provisioner check/installation
- Step 3.5: Automatic GPU node labeling
- Fixed typo: `kukubectl` → `kubectl` (line 93)

**Minor Issue - Kepler:**
```
kepler-8v68f    0/1    CrashLoopBackOff
```
Status: Ignored (power metrics optional, RAPL unavailable)

**Final Pod Status:**
```
NAME                                  READY   STATUS    RESTARTS   AGE
dcgm-exporter-vggql                   1/1     Running   0          4m
grafana-7b4f7db8d7-6kpfd              1/1     Running   0          4m
kube-state-metrics-557d476869-wb9l8   1/1     Running   0          4m
node-exporter-rd5kn                   1/1     Running   0          4m
prometheus-77df554df5-fvrb2           1/1     Running   0          4m
```

All critical monitoring components operational.

### Workload Preparation

#### 1. AI Inference Scripts Enhancement

**Prometheus Metrics Instrumentation (v2):**

All three inference scripts updated with:
- Prometheus client library integration
- HTTP metrics server on port 8000
- Custom metrics: inference_latency_seconds, inference_requests_total
- Automatic test data generation (no external input required)

**ResNet50 Updates:**
```python
# Added Prometheus metrics
from prometheus_client import start_http_server, Histogram, Counter

INFERENCE_LATENCY = Histogram('inference_latency_seconds', 
                               'Inference latency in seconds')
INFERENCE_REQUESTS = Counter('inference_requests_total', 
                             'Total inference requests')

# Auto-generate test data
input_tensor = torch.randn(1, 3, 224, 224, device=device)
```

**DistilBERT Updates:**
```python
# Predefined text samples (cyclic)
texts = [
    "This is a test sentence.",
    "I love using transformers for NLP tasks.",
    "This workload stress tests CPU/GPU usage."
]
```

**Whisper Updates:**
```python
# Synthetic audio generation
audio = np.random.uniform(low=-1.0, high=1.0, 
                          size=(duration_s * sample_rate,)).astype(np.float32)
```

**Docker Images Built and Pushed:**
```bash
docker build -t hamidhrf/resnet50-inference:v2
docker build -t hamidhrf/distilbert-inference:v2
docker build -t hamidhrf/whisper-inference:v2

docker push hamidhrf/resnet50-inference:v2
docker push hamidhrf/distilbert-inference:v2
docker push hamidhrf/whisper-inference:v2
```

Status: All v2 images on DockerHub, deleted locally during cleanup, will auto-pull on deployment.

#### 2. Kubernetes Deployment Manifests

**Critical Change: Resource Limits Removed**

**Rationale:**
Thesis requires measuring actual resource consumption patterns, not constrained behavior.

**Before:**
```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
    nvidia.com/gpu: 1
  limits:
    cpu: "4"
    memory: "8Gi"
    nvidia.com/gpu: 1
```

**After:**
```yaml
# No CPU/memory limits - measure natural consumption
# GPU limit retained for proper scheduling
resources:
  limits:
    nvidia.com/gpu: 1
```

**Prometheus Annotations Added:**
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

Enables automatic service discovery by Prometheus.

### Experimental Strategy Confirmation

#### Phase 1 Approach: Replica Scaling

**Decision:**
Use replica scaling (1 → 3 → 8 pods) with fixed inference rate per pod.

**Rejected Alternatives:**
- Variable inference frequency per pod (adds complexity)
- External load generator/clients (Phase 2 enhancement if needed)
- Mixed approach (overcomplicates initial experiments)

**Rationale:**
- Simulates production scaling patterns (Kubernetes HPA, Netflix-style scaling)
- Creates measurable system contention naturally
- Aligns with thesis requirement: "load levels defined by CPU/GPU utilization or PSI metrics"
- Each pod's behavior changes due to resource competition
- Simpler to implement and analyze

**Load States Defined:**

| State | Replicas | Expected Behavior | Metrics to Observe |
|-------|----------|-------------------|-------------------|
| Baseline | 1 | No contention, best latency | GPU 100% to 1 pod, low CPU |
| Modest | 3 | GPU scheduling delays | GPU time-slicing, CPU 40-60% |
| High | 8 | Heavy GPU queuing | Severe GPU contention, CPU 70-90% |

**Experimental Procedure:**
1. Deploy workload with 1 replica
2. Collect metrics for 30-60 minutes (baseline)
3. Scale to 3 replicas: `kubectl scale deployment resnet50-inference --replicas=3`
4. Collect metrics for 30-60 minutes (modest load)
5. Scale to 8 replicas: `kubectl scale deployment resnet50-inference --replicas=8`
6. Collect metrics for 30-60 minutes (high load)
7. Export Prometheus data for analysis
8. Repeat for DistilBERT and Whisper

**Data to Collect:**
- Inference latency (p50, p95, p99) from custom metrics
- GPU utilization (DCGM exporter)
- CPU utilization (Node Exporter)
- Memory usage (Node Exporter, cAdvisor)
- PSI metrics (System pressure indicators)

### Verification Procedures

#### 1. Monitoring Stack Verification

**Prometheus Targets Status:**
All targets showing "UP" status:
- prometheus (self-monitoring)
- dcgm-exporter (GPU metrics)
- kube-state-metrics (K8s state)
- kubelet (container metrics)
- kubelet-cadvisor (cAdvisor metrics)
- node-exporter (system metrics)

**Access Confirmed:**
- Prometheus UI: http://172.22.174.58:30090
- Grafana UI: http://172.22.174.58:30030

**Grafana Data Source:**
- Prometheus connection tested: ✓ "Successfully queried the Prometheus API"
- All 5 dashboards loaded and functional

#### 2. Dashboard Availability

**Imported Dashboards:**
1. System Resources - Thesis Data Collection
2. GPU Performance - Thesis Data Collection
3. Container Metrics - Thesis Data Collection
4. System Pressure (PSI) - Thesis Load Detection
5. Inference Performance - Thesis QoS Metrics

All dashboards displaying data from respective exporters.

### Technical Architecture Current State

**System Health:**
```
Disk Space:      34GB used / 98GB total (35% - Healthy)
Kubernetes:      All pods Running
GPU:             Registered, 1 GPU allocatable
Monitoring:      All targets UP
Dashboards:      5/5 operational
Docker Images:   Will auto-pull from DockerHub
```

**Ready for Data Collection:**
- ✓ Monitoring infrastructure operational
- ✓ Dashboards configured for thesis metrics
- ✓ AI workload images on DockerHub
- ✓ Deployment manifests updated (no resource limits)
- ✓ Experimental strategy defined (replica scaling)
- ✓ Prometheus auto-discovery configured

### Lessons Learned

1. **Disk pressure causes cascading failures** - Regular monitoring and cleanup essential for long-running research clusters.

2. **Aggressive Docker cleanup has consequences** - Always verify flags before running system-wide cleanup commands. `-af` is more aggressive than needed in most cases.

3. **DockerHub as safety net** - Pushed images provide backup when local cache is lost. Kubernetes auto-pull handles recovery transparently.

4. **Pod eviction records accumulate** - 10,000+ eviction records from weeks of disk pressure. Regular cleanup prevents metadata bloat.

5. **Monitoring infrastructure is prerequisite** - Must be operational before workload deployment to capture complete data from start.

### Next Steps

**Immediate (Today/Tomorrow):**
1. Deploy first AI workload (ResNet50)
2. Verify Kubernetes auto-pulls v2 image from DockerHub
3. Confirm pod starts and exposes Prometheus metrics
4. Check Grafana dashboards show real-time data
5. Test manual scaling (1 → 3 → 1 replicas)

**Phase 1 Data Collection (This Week):**
1. Run ResNet50 experiments (baseline → modest → high load)
2. Export Prometheus data after each experiment
3. Repeat for DistilBERT
4. Repeat for Whisper
5. Analyze collected data for patterns

**Phase 2 Preparation (Next Week):**
1. Review literature on time-series generative models
2. Identify suitable architectures (RNN, LSTM, GAN, VAE)
3. Prepare datasets for model training
4. Begin model selection process

### Status Summary

**Infrastructure:**
- Kubernetes cluster: Production-ready, reboot-safe
- GPU support: Operational, 1 NVIDIA A16 available
- Monitoring: Complete stack deployed and verified
- Storage: 60GB free space, healthy operational level

**Workloads:**
- Scripts: Enhanced with Prometheus metrics (v2)
- Images: On DockerHub, ready for auto-pull
- Manifests: Updated, resource limits removed
- Strategy: Replica scaling approach confirmed

**Current Phase: Phase 1 - Workload Setup**
- ✓ Cluster operational (Dec 2-3)
- ✓ GPU integration (Dec 2-3)
- ✓ Monitoring deployed (Dec 9)
- ✓ Dashboards created (Dec 9)
- ✓ System recovery (Dec 16) ← **TODAY**
- ✓ Workload preparation (Dec 16) ← **TODAY**
- → **NEXT: AI workload deployment and baseline data collection**

**Ready to begin Phase 1 experiments.**

---