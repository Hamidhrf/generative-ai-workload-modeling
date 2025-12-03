#!/bin/bash

# Complete Kubernetes Cluster Cleanup Script
# Removes all Kubernetes components, CRI-O, Containerd, and configurations

set -e

echo "=========================================="
echo "Starting Complete Kubernetes Cleanup"
echo "=========================================="

# Reset kubeadm if it was initialized
echo "[1/12] Resetting kubeadm..."
if command -v kubeadm &> /dev/null; then
    sudo kubeadm reset -f
fi

# Stop all services
echo "[2/12] Stopping all Kubernetes and container runtime services..."
sudo systemctl stop kubelet 2>/dev/null || true
sudo systemctl stop containerd 2>/dev/null || true
sudo systemctl stop crio 2>/dev/null || true
sudo systemctl stop cri-o 2>/dev/null || true

# Disable services
echo "[3/12] Disabling services..."
sudo systemctl disable kubelet 2>/dev/null || true
sudo systemctl disable containerd 2>/dev/null || true
sudo systemctl disable crio 2>/dev/null || true
sudo systemctl disable cri-o 2>/dev/null || true

# Remove Kubernetes packages
echo "[4/12] Removing Kubernetes packages..."
sudo apt-mark unhold kubelet kubeadm kubectl 2>/dev/null || true
sudo apt-get purge -y kubeadm kubectl kubelet kubernetes-cni 2>/dev/null || true
sudo apt-get autoremove -y

# Remove CRI-O
echo "[5/12] Removing CRI-O..."
sudo apt-get purge -y cri-o cri-o-runc cri-tools 2>/dev/null || true

# Remove CRI-O binaries (in case of manual installation)
sudo rm -f /usr/bin/crio
sudo rm -f /usr/bin/crio-status
sudo rm -f /usr/local/bin/crio
sudo rm -f /usr/local/bin/crio-status

# Remove Containerd
echo "[6/12] Removing Containerd..."
sudo apt-get purge -y containerd containerd.io 2>/dev/null || true
sudo rm -rf /usr/local/bin/containerd* 2>/dev/null || true
sudo rm -rf /usr/local/bin/ctr 2>/dev/null || true
sudo rm -rf /usr/local/sbin/runc 2>/dev/null || true

# Remove all Kubernetes directories and files
echo "[7/12] Removing Kubernetes directories..."
sudo rm -rf /etc/kubernetes
sudo rm -rf /var/lib/kubelet
sudo rm -rf /var/lib/etcd
sudo rm -rf /etc/cni
sudo rm -rf /opt/cni
sudo rm -rf /var/lib/cni
sudo rm -rf /run/flannel
sudo rm -rf /etc/kube-flannel
sudo rm -rf ~/.kube

# Remove CRI-O directories
echo "[8/12] Removing CRI-O directories..."
sudo rm -rf /etc/crio
sudo rm -rf /var/lib/crio
sudo rm -rf /var/run/crio
sudo rm -rf /usr/local/bin/crio
sudo rm -rf /usr/lib/cri-o-runc

# Remove Containerd directories
echo "[9/12] Removing Containerd directories..."
sudo rm -rf /etc/containerd
sudo rm -rf /var/lib/containerd
sudo rm -rf /run/containerd
sudo rm -rf /etc/systemd/system/containerd.service

# Remove crictl
sudo rm -rf /usr/local/bin/crictl
sudo rm -rf /etc/crictl.yaml

# Clean up network
echo "[10/12] Cleaning up network interfaces..."
sudo ip link delete cni0 2>/dev/null || true
sudo ip link delete flannel.1 2>/dev/null || true
sudo ip link delete tunl0 2>/dev/null || true
sudo ip link delete vxlan.calico 2>/dev/null || true
sudo ip link delete cilium_host 2>/dev/null || true
sudo ip link delete cilium_net 2>/dev/null || true
sudo ip link delete cilium_vxlan 2>/dev/null || true

# Remove iptables rules
echo "[11/12] Flushing iptables rules..."
sudo iptables -F && sudo iptables -t nat -F && sudo iptables -t mangle -F && sudo iptables -X

# Remove kernel modules and sysctl settings
echo "[12/12] Removing kernel modules and sysctl settings..."
sudo rm -f /etc/modules-load.d/k8s.conf
sudo rm -f /etc/sysctl.d/k8s.conf
sudo rm -f /etc/modules-load.d/crio.conf
sudo rm -f /etc/sysctl.d/99-kubernetes-cri.conf

# Remove repositories
sudo rm -f /etc/apt/sources.list.d/kubernetes.list
sudo rm -f /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list
sudo rm -f /etc/apt/sources.list.d/devel:kubic:libcontainers:stable:cri-o:*.list
sudo rm -f /etc/apt/sources.list.d/cri-o.list
sudo rm -f /etc/apt/keyrings/kubernetes-apt-keyring.gpg
sudo rm -f /etc/apt/keyrings/cri-o-apt-keyring.gpg
sudo rm -f /etc/apt/keyrings/libcontainers-archive-keyring.gpg
sudo rm -f /etc/apt/keyrings/libcontainers-crio-archive-keyring.gpg
sudo rm -f /etc/apt/trusted.gpg.d/k8s.gpg
sudo rm -f /etc/apt/trusted.gpg.d/libcontainers.gpg

# Reload systemd
sudo systemctl daemon-reload

# Update apt
sudo apt-get update

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo "All Kubernetes components and container runtimes have been removed."
echo ""
echo "Verification:"
echo "-------------"

# Verify packages removed
if dpkg -l | grep -E 'kube|kubernetes' > /dev/null 2>&1; then
    echo "⚠ Warning: Some Kubernetes packages may still be present"
    dpkg -l | grep -E 'kube|kubernetes'
else
    echo "✓ All Kubernetes packages removed"
fi

# Verify CRI-O removed
if [ -f /usr/bin/crio ] || [ -f /usr/local/bin/crio ]; then
    echo "⚠ Warning: CRI-O binaries still present"
else
    echo "✓ CRI-O binaries removed"
fi

# Verify network interfaces clean
if ip link show | grep -E "cni|flannel|calico|tunl|cilium" > /dev/null 2>&1; then
    echo "⚠ Warning: Some network interfaces may still exist"
    ip link show | grep -E "cni|flannel|calico|tunl|cilium"
else
    echo "✓ Network interfaces cleaned"
fi

# Verify directories removed
if [ -d /etc/kubernetes ] || [ -d /var/lib/kubelet ] || [ -d /etc/crio ]; then
    echo "⚠ Warning: Some directories still exist"
else
    echo "✓ Key directories removed"
fi

# Verify repositories removed
if ls /etc/apt/sources.list.d/ 2>/dev/null | grep -E "kubernetes|cri-o" > /dev/null; then
    echo "⚠ Warning: Repository files still present"
    ls /etc/apt/sources.list.d/ | grep -E "kubernetes|cri-o"
else
    echo "✓ Repository files removed"
fi

echo ""
echo "System is ready for fresh installation."
echo "Recommended: Reboot before installing"
echo "  sudo reboot"
echo ""