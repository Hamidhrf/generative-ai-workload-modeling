# first try versin 7
# in case of problems use version 6 but you need to install CRI-O manaually after running the script and then run the script again.
# version 5 was stable version without rebooting survival but the CRI part was fine


#====================================
# VERSION 5
#====================================

# #!/bin/bash

# # Single Node Kubernetes Cluster Setup with CRI-O
# # For Ubuntu 24.04

# set -e

# # Configuration
# CONTROL_PLANE_IP="172.22.174.58"
# KUBERNETES_VERSION="v1.34"
# KUBERNETES_INSTALL_VERSION="1.34.0-1.1"
# CRIO_VERSION="1.31"
# CRICTL_VERSION="v1.31.1"

# echo "=========================================="
# echo "Kubernetes Single Node Setup with CRI-O"
# echo "=========================================="
# echo "Control Plane IP: $CONTROL_PLANE_IP"
# echo "Kubernetes Version: $KUBERNETES_VERSION"
# echo "CRI-O Version: $CRIO_VERSION"
# echo ""

# # Step 1: Enable iptables bridged traffic
# echo "[1/9] Configuring kernel parameters..."
# cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
# overlay
# br_netfilter
# EOF

# sudo modprobe overlay
# sudo modprobe br_netfilter

# cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
# net.bridge.bridge-nf-call-iptables = 1
# net.bridge.bridge-nf-call-ip6tables = 1
# net.ipv4.ip_forward = 1
# EOF

# sudo sysctl --system

# # Step 2: Disable swap
# echo "[2/9] Disabling swap..."
# sudo swapoff -a

# # Add crontab entry to disable swap on reboot
# (crontab -l 2>/dev/null; echo "@reboot /sbin/swapoff -a") | crontab - || true

# # Comment out ALL swap entries in fstab (multiple patterns to catch all cases)
# sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
# sudo sed -i 's|^/swap.img|#/swap.img|g' /etc/fstab
# sudo sed -i 's|^/swapfile|#/swapfile|g' /etc/fstab

# # Verify swap is disabled
# if [ "$(swapon --show | wc -l)" -eq 0 ]; then
#     echo "✓ Swap disabled successfully"
# else
#     echo "⚠ Warning: Swap might still be active"
#     swapon --show
# fi

# # Step 3: Install CRI-O
# echo "[3/9] Installing CRI-O runtime..."

# # Add CRI-O repositories
# sudo apt-get update
# sudo apt-get install -y software-properties-common curl apt-transport-https ca-certificates gpg

# # Add CRI-O repository - using pkgs.k8s.io
# sudo mkdir -p /etc/apt/keyrings
# curl -fsSL https://pkgs.k8s.io/addons:/cri-o:/stable:/v${CRIO_VERSION}/deb/Release.key | \
#     sudo gpg --dearmor -o /etc/apt/keyrings/cri-o-apt-keyring.gpg

# echo "deb [signed-by=/etc/apt/keyrings/cri-o-apt-keyring.gpg] https://pkgs.k8s.io/addons:/cri-o:/stable:/v${CRIO_VERSION}/deb/ /" | \
#     sudo tee /etc/apt/sources.list.d/cri-o.list

# # Install CRI-O
# sudo apt-get update
# sudo apt-get install -y cri-o

# # Remove default CNI configs (will conflict with Calico)
# sudo rm -f /etc/cni/net.d/*

# # Start and enable CRI-O
# sudo systemctl daemon-reload
# sudo systemctl enable crio --now
# sudo systemctl start crio

# # Verify CRI-O is running
# if sudo systemctl is-active --quiet crio; then
#     echo "✓ CRI-O installed and running successfully"
# else
#     echo "✗ CRI-O failed to start"
#     exit 1
# fi

# # Step 4: Install crictl
# echo "[4/9] Installing crictl..."
# curl -LO https://github.com/kubernetes-sigs/cri-tools/releases/download/${CRICTL_VERSION}/crictl-${CRICTL_VERSION}-linux-amd64.tar.gz
# sudo tar zxvf crictl-${CRICTL_VERSION}-linux-amd64.tar.gz -C /usr/local/bin
# rm -f crictl-${CRICTL_VERSION}-linux-amd64.tar.gz

# # Configure crictl to use CRI-O
# cat <<EOF | sudo tee /etc/crictl.yaml
# runtime-endpoint: unix:///var/run/crio/crio.sock
# image-endpoint: unix:///var/run/crio/crio.sock
# timeout: 10
# debug: false
# EOF

# echo "crictl installed and configured successfully"

# # Step 5: Install Kubeadm, Kubelet, and Kubectl
# echo "[5/9] Installing Kubernetes components..."

# # Add Kubernetes repository
# curl -fsSL https://pkgs.k8s.io/core:/stable:/$KUBERNETES_VERSION/deb/Release.key | \
#     sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

# echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/$KUBERNETES_VERSION/deb/ /" | \
#     sudo tee /etc/apt/sources.list.d/kubernetes.list

# # Update and install
# sudo apt-get update
# sudo apt-get install -y kubelet="$KUBERNETES_INSTALL_VERSION" kubectl="$KUBERNETES_INSTALL_VERSION" kubeadm="$KUBERNETES_INSTALL_VERSION"
# sudo apt-mark hold kubelet kubeadm kubectl

# # Configure kubelet with node IP
# sudo apt-get install -y jq
# local_ip="$CONTROL_PLANE_IP"
# cat <<EOF | sudo tee /etc/default/kubelet
# KUBELET_EXTRA_ARGS=--node-ip=$local_ip
# EOF

# echo "Kubernetes components installed successfully"

# # Step 6: Create kubeadm config
# echo "[6/9] Creating kubeadm configuration..."
# cat <<EOF > /tmp/kubeadm-config.yaml
# apiVersion: kubeadm.k8s.io/v1beta4
# kind: InitConfiguration
# localAPIEndpoint:
#   advertiseAddress: "$CONTROL_PLANE_IP"
#   bindPort: 6443
# nodeRegistration:
#   name: "controlplane"
#   criSocket: "unix:///var/run/crio/crio.sock"
# ---
# apiVersion: kubeadm.k8s.io/v1beta4
# kind: ClusterConfiguration
# kubernetesVersion: "v1.34.0"
# controlPlaneEndpoint: "$CONTROL_PLANE_IP:6443"
# apiServer:
#   extraArgs:
#   - name: "enable-admission-plugins"
#     value: "NodeRestriction"
#   - name: "audit-log-path"
#     value: "/var/log/kubernetes/audit.log"
# controllerManager:
#   extraArgs:
#   - name: "node-cidr-mask-size"
#     value: "24"
# scheduler:
#   extraArgs:
#   - name: "leader-elect"
#     value: "true"
# networking:
#   podSubnet: "10.244.0.0/16"
#   serviceSubnet: "10.96.0.0/12"
#   dnsDomain: "cluster.local"
# ---
# apiVersion: kubelet.config.k8s.io/v1beta1
# kind: KubeletConfiguration
# cgroupDriver: "systemd"
# syncFrequency: "1m"
# ---
# apiVersion: kubeproxy.config.k8s.io/v1alpha1
# kind: KubeProxyConfiguration
# mode: "ipvs"
# conntrack:
#   maxPerCore: 32768
#   min: 131072
# EOF

# echo "Kubeadm configuration created"

# # Step 7: Initialize Kubernetes cluster
# echo "[7/9] Initializing Kubernetes cluster..."
# sudo kubeadm init --config=/tmp/kubeadm-config.yaml

# # Setup kubeconfig
# mkdir -p $HOME/.kube
# sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
# sudo chown $(id -u):$(id -g) $HOME/.kube/config

# echo "Cluster initialized successfully"

# # Step 8: Remove taint from control plane (single node)
# echo "[8/9] Configuring single node (removing control-plane taint)..."
# sleep 5
# kubectl taint nodes --all node-role.kubernetes.io/control-plane- || true

# # Step 9: Install Calico network plugin
# echo "[9/9] Installing Calico network plugin..."

# # Install Tigera operator
# kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/tigera-operator.yaml

# # Download and customize Calico resources
# curl https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/custom-resources.yaml -O

# # Update CIDR to match our pod network
# sed -i 's|cidr: 192.168.0.0/16|cidr: 10.244.0.0/16|g' custom-resources.yaml

# # Apply Calico custom resources
# kubectl apply -f custom-resources.yaml

# echo ""
# echo "=========================================="
# echo "Waiting for all pods to be ready..."
# echo "=========================================="
# echo "This may take a few minutes..."
# sleep 30

# # Wait for pods
# kubectl wait --for=condition=ready pod --all -n kube-system --timeout=300s || true

# echo ""
# echo "=========================================="
# echo "Cluster Setup Complete!"
# echo "=========================================="
# echo ""
# echo "Cluster Information:"
# kubectl cluster-info
# echo ""
# echo "Node Status:"
# kubectl get nodes
# echo ""
# echo "All Pods:"
# kubectl get pods -A
# echo ""
# echo "=========================================="
# echo "Installing Metrics Server..."
# echo "=========================================="

# # Install metrics server
# kubectl apply -f https://raw.githubusercontent.com/techiescamp/cka-certification-guide/refs/heads/main/lab-setup/manifests/metrics-server/metrics-server.yaml

# echo ""
# echo "=========================================="
# echo "Setup Complete!"
# echo "=========================================="
# echo ""
# echo "Your single-node Kubernetes cluster is ready!"
# echo "Control Plane IP: $CONTROL_PLANE_IP"
# echo ""
# echo "To verify the cluster:"
# echo "  kubectl get nodes"
# echo "  kubectl get pods -A"
# echo "  kubectl top nodes  (wait 1-2 minutes for metrics)"
# echo ""
# echo "Kubeconfig is located at: ~/.kube/config"
# echo ""







#====================================
# VERSION 6
#====================================

# #!/bin/bash

# # Single Node Kubernetes Cluster Setup with CRI-O
# # For Ubuntu 24.04

# set -e

# # Configuration
# CONTROL_PLANE_IP="172.22.174.58"
# KUBERNETES_VERSION="v1.34"
# KUBERNETES_INSTALL_VERSION="1.34.0-1.1"
# CRIO_VERSION="1.31"
# CRICTL_VERSION="v1.31.1"

# echo "=========================================="
# echo "Kubernetes Single Node Setup with CRI-O"
# echo "=========================================="
# echo "Control Plane IP: $CONTROL_PLANE_IP"
# echo "Kubernetes Version: $KUBERNETES_VERSION"
# echo "CRI-O Version: $CRIO_VERSION"
# echo ""

# # Step 1: Enable iptables bridged traffic
# echo "[1/9] Configuring kernel parameters..."
# cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
# overlay
# br_netfilter
# EOF

# sudo modprobe overlay
# sudo modprobe br_netfilter

# cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
# net.bridge.bridge-nf-call-iptables = 1
# net.bridge.bridge-nf-call-ip6tables = 1
# net.ipv4.ip_forward = 1
# EOF

# sudo sysctl --system

# # Step 2: Disable swap
# echo "[2/9] Disabling swap..."
# sudo swapoff -a

# # Add crontab entry to disable swap on reboot
# (crontab -l 2>/dev/null; echo "@reboot /sbin/swapoff -a") | crontab - || true

# # Comment out ALL swap entries in fstab (multiple patterns to catch all cases)
# sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
# sudo sed -i 's|^/swap.img|#/swap.img|g' /etc/fstab
# sudo sed -i 's|^/swapfile|#/swapfile|g' /etc/fstab

# # Verify swap is disabled
# if [ "$(swapon --show | wc -l)" -eq 0 ]; then
#     echo "✓ Swap disabled successfully"
# else
#     echo "⚠ Warning: Swap might still be active"
#     swapon --show
# fi

# # Step 3: Install CRI-O
# echo "[3/9] Installing CRI-O runtime..."

# # Add CRI-O repositories
# sudo apt-get update
# sudo apt-get install -y software-properties-common curl apt-transport-https ca-certificates gpg

# # Add CRI-O repository - using pkgs.k8s.io
# sudo mkdir -p /etc/apt/keyrings
# curl -fsSL https://pkgs.k8s.io/addons:/cri-o:/stable:/v${CRIO_VERSION}/deb/Release.key | \
#     sudo gpg --dearmor -o /etc/apt/keyrings/cri-o-apt-keyring.gpg

# echo "deb [signed-by=/etc/apt/keyrings/cri-o-apt-keyring.gpg] https://pkgs.k8s.io/addons:/cri-o:/stable:/v${CRIO_VERSION}/deb/ /" | \
#     sudo tee /etc/apt/sources.list.d/cri-o.list

# # Install CRI-O
# sudo apt-get update
# sudo apt-get install -y cri-o

# # Remove default CNI configs (will conflict with Calico)
# sudo rm -f /etc/cni/net.d/*

# # Start and enable CRI-O
# sudo systemctl daemon-reload
# sudo systemctl enable crio --now
# sudo systemctl start crio

# # Verify CRI-O is running
# if sudo systemctl is-active --quiet crio; then
#     echo "✓ CRI-O installed and running successfully"
# else
#     echo "✗ CRI-O failed to start"
#     exit 1
# fi

# # Step 4: Install crictl
# echo "[4/9] Installing crictl..."
# curl -LO https://github.com/kubernetes-sigs/cri-tools/releases/download/${CRICTL_VERSION}/crictl-${CRICTL_VERSION}-linux-amd64.tar.gz
# sudo tar zxvf crictl-${CRICTL_VERSION}-linux-amd64.tar.gz -C /usr/local/bin
# rm -f crictl-${CRICTL_VERSION}-linux-amd64.tar.gz

# # Configure crictl to use CRI-O
# cat <<EOF | sudo tee /etc/crictl.yaml
# runtime-endpoint: unix:///var/run/crio/crio.sock
# image-endpoint: unix:///var/run/crio/crio.sock
# timeout: 10
# debug: false
# EOF

# echo "crictl installed and configured successfully"

# # Step 5: Install Kubeadm, Kubelet, and Kubectl
# echo "[5/9] Installing Kubernetes components..."

# # Add Kubernetes repository
# curl -fsSL https://pkgs.k8s.io/core:/stable:/$KUBERNETES_VERSION/deb/Release.key | \
#     sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

# echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/$KUBERNETES_VERSION/deb/ /" | \
#     sudo tee /etc/apt/sources.list.d/kubernetes.list

# # Update and install
# sudo apt-get update
# sudo apt-get install -y kubelet="$KUBERNETES_INSTALL_VERSION" kubectl="$KUBERNETES_INSTALL_VERSION" kubeadm="$KUBERNETES_INSTALL_VERSION"
# sudo apt-mark hold kubelet kubeadm kubectl

# # Enable kubelet service (will start after kubeadm init)
# sudo systemctl enable kubelet

# # Configure kubelet with node IP
# sudo apt-get install -y jq
# local_ip="$CONTROL_PLANE_IP"
# cat <<EOF | sudo tee /etc/default/kubelet
# KUBELET_EXTRA_ARGS=--node-ip=$local_ip
# EOF

# echo "✓ Kubernetes components installed successfully"

# # Step 6: Create kubeadm config
# echo "[6/9] Creating kubeadm configuration..."
# cat <<EOF > /tmp/kubeadm-config.yaml
# apiVersion: kubeadm.k8s.io/v1beta4
# kind: InitConfiguration
# localAPIEndpoint:
#   advertiseAddress: "$CONTROL_PLANE_IP"
#   bindPort: 6443
# nodeRegistration:
#   name: "controlplane"
#   criSocket: "unix:///var/run/crio/crio.sock"
# ---
# apiVersion: kubeadm.k8s.io/v1beta4
# kind: ClusterConfiguration
# kubernetesVersion: "v1.34.0"
# controlPlaneEndpoint: "$CONTROL_PLANE_IP:6443"
# apiServer:
#   extraArgs:
#   - name: "enable-admission-plugins"
#     value: "NodeRestriction"
#   - name: "audit-log-path"
#     value: "/var/log/kubernetes/audit.log"
# controllerManager:
#   extraArgs:
#   - name: "node-cidr-mask-size"
#     value: "24"
# scheduler:
#   extraArgs:
#   - name: "leader-elect"
#     value: "true"
# networking:
#   podSubnet: "10.244.0.0/16"
#   serviceSubnet: "10.96.0.0/12"
#   dnsDomain: "cluster.local"
# ---
# apiVersion: kubelet.config.k8s.io/v1beta1
# kind: KubeletConfiguration
# cgroupDriver: "systemd"
# syncFrequency: "1m"
# ---
# apiVersion: kubeproxy.config.k8s.io/v1alpha1
# kind: KubeProxyConfiguration
# mode: "ipvs"
# conntrack:
#   maxPerCore: 32768
#   min: 131072
# EOF

# echo "Kubeadm configuration created"

# # Step 7: Initialize Kubernetes cluster
# echo "[7/9] Initializing Kubernetes cluster..."
# sudo kubeadm init --config=/tmp/kubeadm-config.yaml

# # Setup kubeconfig
# mkdir -p $HOME/.kube
# sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
# sudo chown $(id -u):$(id -g) $HOME/.kube/config

# echo "Cluster initialized successfully"

# # Step 8: Remove taint from control plane (single node)
# echo "[8/9] Configuring single node (removing control-plane taint)..."
# sleep 5
# kubectl taint nodes --all node-role.kubernetes.io/control-plane- || true

# # Step 9: Install Calico network plugin
# echo "[9/9] Installing Calico network plugin..."

# # Install Tigera operator
# kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/tigera-operator.yaml

# # Download and customize Calico resources
# curl https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/custom-resources.yaml -O

# # Update CIDR to match our pod network
# sed -i 's|cidr: 192.168.0.0/16|cidr: 10.244.0.0/16|g' custom-resources.yaml

# # Apply Calico custom resources
# kubectl apply -f custom-resources.yaml

# echo ""
# echo "=========================================="
# echo "Waiting for all pods to be ready..."
# echo "=========================================="
# echo "This may take a few minutes..."
# sleep 30

# # Wait for pods
# kubectl wait --for=condition=ready pod --all -n kube-system --timeout=300s || true

# echo ""
# echo "=========================================="
# echo "Cluster Setup Complete!"
# echo "=========================================="
# echo ""
# echo "Cluster Information:"
# kubectl cluster-info
# echo ""
# echo "Node Status:"
# kubectl get nodes
# echo ""
# echo "All Pods:"
# kubectl get pods -A
# echo ""
# echo "=========================================="
# echo "Installing Metrics Server..."
# echo "=========================================="

# # Install metrics server
# kubectl apply -f https://raw.githubusercontent.com/techiescamp/cka-certification-guide/refs/heads/main/lab-setup/manifests/metrics-server/metrics-server.yaml

# echo ""
# echo "=========================================="
# echo "Setup Complete!"
# echo "=========================================="
# echo ""
# echo "Your single-node Kubernetes cluster is ready!"
# echo "Control Plane IP: $CONTROL_PLANE_IP"
# echo ""
# echo "To verify the cluster:"
# echo "  kubectl get nodes"
# echo "  kubectl get pods -A"
# echo "  kubectl top nodes  (wait 1-2 minutes for metrics)"
# echo ""
# echo "Kubeconfig is located at: ~/.kube/config"
# echo ""







#====================================
# VERSION 7
#====================================
#!/bin/bash

# Single Node Kubernetes Cluster Setup with CRI-O - Version 7
# For Ubuntu 24.04
# Includes: Reboot-safety + Robust CRI-O installation

set -e

# Configuration
CONTROL_PLANE_IP="172.22.174.58"
KUBERNETES_VERSION="v1.34"
KUBERNETES_INSTALL_VERSION="1.34.0-1.1"
CRIO_VERSION="1.31"
CRICTL_VERSION="v1.31.1"

echo "=========================================="
echo "Kubernetes Single Node Setup with CRI-O"
echo "=========================================="
echo "Control Plane IP: $CONTROL_PLANE_IP"
echo "Kubernetes Version: $KUBERNETES_VERSION"
echo "CRI-O Version: $CRIO_VERSION"
echo ""

# Step 1: Enable iptables bridged traffic
echo "[1/9] Configuring kernel parameters..."
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF

sudo sysctl --system

# Step 2: Disable swap
echo "[2/9] Disabling swap..."
sudo swapoff -a

# Add crontab entry to disable swap on reboot
(crontab -l 2>/dev/null; echo "@reboot /sbin/swapoff -a") | crontab - || true

# Comment out ALL swap entries in fstab (multiple patterns to catch all cases)
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
sudo sed -i 's|^/swap.img|#/swap.img|g' /etc/fstab
sudo sed -i 's|^/swapfile|#/swapfile|g' /etc/fstab

# Verify swap is disabled
if [ "$(swapon --show | wc -l)" -eq 0 ]; then
    echo "✓ Swap disabled successfully"
else
    echo "⚠ Warning: Swap might still be active"
    swapon --show
fi

# Step 3: Install CRI-O
echo "[3/9] Installing CRI-O runtime..."

# Add CRI-O repositories
sudo apt-get update
sudo apt-get install -y software-properties-common curl apt-transport-https ca-certificates gpg

# Add CRI-O repository - using pkgs.k8s.io
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/addons:/cri-o:/stable:/v${CRIO_VERSION}/deb/Release.key | \
    sudo gpg --dearmor -o /etc/apt/keyrings/cri-o-apt-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/cri-o-apt-keyring.gpg] https://pkgs.k8s.io/addons:/cri-o:/stable:/v${CRIO_VERSION}/deb/ /" | \
    sudo tee /etc/apt/sources.list.d/cri-o.list

# Install CRI-O (use --reinstall to handle broken package states)
sudo apt-get update
sudo apt-get install -y --reinstall cri-o

# Verify binary exists (critical check)
if [ ! -f /usr/bin/crio ]; then
    echo "✗ CRI-O binary not found after installation!"
    echo "Attempting to fix package state..."
    sudo apt-get purge -y cri-o
    sudo apt-get autoremove -y
    sudo apt-get install -y cri-o
    
    if [ ! -f /usr/bin/crio ]; then
        echo "✗ CRI-O installation failed. Binary still missing."
        exit 1
    fi
fi

echo "✓ CRI-O binary verified at /usr/bin/crio"

# Remove default CNI configs (will conflict with Calico)
sudo rm -f /etc/cni/net.d/*

# Start and enable CRI-O
sudo systemctl daemon-reload
sudo systemctl enable crio --now
sudo systemctl start crio

# Wait a moment for CRI-O to fully start
sleep 3

# Verify CRI-O is running
if sudo systemctl is-active --quiet crio; then
    echo "✓ CRI-O installed and running successfully"
    crio --version
else
    echo "✗ CRI-O failed to start"
    echo "Checking status:"
    sudo systemctl status crio --no-pager
    echo ""
    echo "Checking logs:"
    sudo journalctl -u crio -n 20 --no-pager
    exit 1
fi

# Step 4: Install crictl
echo "[4/9] Installing crictl..."
curl -LO https://github.com/kubernetes-sigs/cri-tools/releases/download/${CRICTL_VERSION}/crictl-${CRICTL_VERSION}-linux-amd64.tar.gz
sudo tar zxvf crictl-${CRICTL_VERSION}-linux-amd64.tar.gz -C /usr/local/bin
rm -f crictl-${CRICTL_VERSION}-linux-amd64.tar.gz

# Configure crictl to use CRI-O
cat <<EOF | sudo tee /etc/crictl.yaml
runtime-endpoint: unix:///var/run/crio/crio.sock
image-endpoint: unix:///var/run/crio/crio.sock
timeout: 10
debug: false
EOF

echo "crictl installed and configured successfully"

# Step 5: Install Kubeadm, Kubelet, and Kubectl
echo "[5/9] Installing Kubernetes components..."

# Add Kubernetes repository
curl -fsSL https://pkgs.k8s.io/core:/stable:/$KUBERNETES_VERSION/deb/Release.key | \
    sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/$KUBERNETES_VERSION/deb/ /" | \
    sudo tee /etc/apt/sources.list.d/kubernetes.list

# Update and install
sudo apt-get update
sudo apt-get install -y kubelet="$KUBERNETES_INSTALL_VERSION" kubectl="$KUBERNETES_INSTALL_VERSION" kubeadm="$KUBERNETES_INSTALL_VERSION"
sudo apt-mark hold kubelet kubeadm kubectl

# Enable kubelet service (will start after kubeadm init)
sudo systemctl enable kubelet

# Configure kubelet with node IP
sudo apt-get install -y jq
local_ip="$CONTROL_PLANE_IP"
cat <<EOF | sudo tee /etc/default/kubelet
KUBELET_EXTRA_ARGS=--node-ip=$local_ip
EOF

echo "✓ Kubernetes components installed successfully"

# Step 6: Create kubeadm config
echo "[6/9] Creating kubeadm configuration..."
cat <<EOF > /tmp/kubeadm-config.yaml
apiVersion: kubeadm.k8s.io/v1beta4
kind: InitConfiguration
localAPIEndpoint:
  advertiseAddress: "$CONTROL_PLANE_IP"
  bindPort: 6443
nodeRegistration:
  name: "controlplane"
  criSocket: "unix:///var/run/crio/crio.sock"
---
apiVersion: kubeadm.k8s.io/v1beta4
kind: ClusterConfiguration
kubernetesVersion: "v1.34.0"
controlPlaneEndpoint: "$CONTROL_PLANE_IP:6443"
apiServer:
  extraArgs:
  - name: "enable-admission-plugins"
    value: "NodeRestriction"
  - name: "audit-log-path"
    value: "/var/log/kubernetes/audit.log"
controllerManager:
  extraArgs:
  - name: "node-cidr-mask-size"
    value: "24"
scheduler:
  extraArgs:
  - name: "leader-elect"
    value: "true"
networking:
  podSubnet: "10.244.0.0/16"
  serviceSubnet: "10.96.0.0/12"
  dnsDomain: "cluster.local"
---
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: "systemd"
syncFrequency: "1m"
---
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
mode: "ipvs"
conntrack:
  maxPerCore: 32768
  min: 131072
EOF

echo "Kubeadm configuration created"

# Step 7: Initialize Kubernetes cluster
echo "[7/9] Initializing Kubernetes cluster..."
sudo kubeadm init --config=/tmp/kubeadm-config.yaml

# Setup kubeconfig
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

echo "Cluster initialized successfully"

# Step 8: Remove taint from control plane (single node)
echo "[8/9] Configuring single node (removing control-plane taint)..."
sleep 5
kubectl taint nodes --all node-role.kubernetes.io/control-plane- || true

# Step 9: Install Calico network plugin
echo "[9/9] Installing Calico network plugin..."

# Install Tigera operator
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/tigera-operator.yaml

# Download and customize Calico resources
curl https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/custom-resources.yaml -O

# Update CIDR to match our pod network
sed -i 's|cidr: 192.168.0.0/16|cidr: 10.244.0.0/16|g' custom-resources.yaml

# Apply Calico custom resources
kubectl apply -f custom-resources.yaml

echo ""
echo "=========================================="
echo "Waiting for all pods to be ready..."
echo "=========================================="
echo "This may take a few minutes..."
sleep 30

# Wait for pods
kubectl wait --for=condition=ready pod --all -n kube-system --timeout=300s || true

echo ""
echo "=========================================="
echo "Cluster Setup Complete!"
echo "=========================================="
echo ""
echo "Cluster Information:"
kubectl cluster-info
echo ""
echo "Node Status:"
kubectl get nodes
echo ""
echo "All Pods:"
kubectl get pods -A
echo ""
echo "=========================================="
echo "Installing Metrics Server..."
echo "=========================================="

# Install metrics server
kubectl apply -f https://raw.githubusercontent.com/techiescamp/cka-certification-guide/refs/heads/main/lab-setup/manifests/metrics-server/metrics-server.yaml

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Your single-node Kubernetes cluster is ready!"
echo "Control Plane IP: $CONTROL_PLANE_IP"
echo ""
echo "To verify the cluster:"
echo "  kubectl get nodes"
echo "  kubectl get pods -A"
echo "  kubectl top nodes  (wait 1-2 minutes for metrics)"
echo ""
echo "Kubeconfig is located at: ~/.kube/config"
echo ""