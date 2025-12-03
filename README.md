# Generative AI Workload Modeling

**Research Thesis Project** - Digital Transformation  
**Institution**: Fachhochschule Dortmund  
**Author**: Hamidreza Fathollahzadeh
**Focus**: Performance analysis of AI inference workloads on Kubernetes

---

##  Project Overview

This research project analyzes the performance characteristics of three generative AI inference workloads deployed on Kubernetes:
1. **ResNet50** - Image classification inference
2. **DistilBERT** - Natural language processing inference  
3. **Whisper** - Speech-to-text inference

The goal is to model workload behavior, resource consumption patterns, and performance metrics under various configurations.

---

##  Current Status

 **Kubernetes Cluster**: Operational (v1.34.0)  
 **Container Runtime**: CRI-O 1.31.5  
 **GPU Support**: NVIDIA A16 (Driver 580.95.05)  
 **Network Plugin**: Calico (10.244.0.0/16)  
 **Reboot Stability**: Verified  


---

##  Repository Structure

```
generative-ai-workload-modeling/
│
├── scripts/                      # Automation scripts
│   ├── cluster-creation/         # Cluster setup & recovery
│   │   ├── cleanup.sh
│   │   ├── setup-cluster.sh
│   │   └── cluster-recovery.sh
│   ├── gpu-setup/                # GPU configuration
│   │   ├── gpu-health-check.sh
│   │   └── nvidia-runtimeclass.yaml
│   └── workloads/                # Workload deployment automation
│
├── k8s/                          # Kubernetes manifests
│   ├── base/                     # Base cluster configs
│   │   └── custom-resources.yaml (Calico)
│   ├── gpu/                      # GPU configurations
│   │   └── nvidia-runtimeclass.yaml
│   └── workloads/                # AI application deployments
│       ├── resnet50-deployment.yaml
│       ├── distilbert-deployment.yaml
│       └── whisper-deployment.yaml
│
├── tools/                        # Monitoring & observability
│   ├── grafana-v11.2.0/         # Grafana dashboards
│   └── prometheus/               # Metrics collection
│
├── dashboards/                   # Grafana dashboard configs
│   └── (monitoring dashboards)
│
├── models/                       # AI model configurations
│   └── (model definitions & configs)
│
├── notebooks/                    # Jupyter notebooks
│   └── (analysis & visualization)
│
├── data/                         # Experiment data
│   ├── raw/                      # Raw metrics (gitignored)
│   ├── processed/                # Processed results
│   └── benchmarks/               # Performance benchmarks
│
├── docs/                         # Documentation
│   ├── setup-guide.md
│   └── troubleshooting.md
│
├── JOURNAL.md                    # Technical development journal
├── README.md                     # This file
├── .gitignore                    # Git exclusions
└── environment.yml               # Conda environment
```

---

##  Quick Start

### Prerequisites
- Ubuntu 24.04 LTS
- NVIDIA GPU with drivers installed
- Minimum 2 vCPU, 2GB RAM
- kubectl access to cluster

### Cluster Setup
```bash
# Complete cluster setup with GPU
cd scripts/cluster-creation
sudo ./setup-cluster.sh

# Verify cluster
kubectl get nodes
kubectl describe node | grep nvidia.com/gpu
```

### GPU Health Check
```bash
cd scripts/gpu-setup
sudo ./gpu-health-check.sh
```

---

##  Infrastructure Details

### Cluster Configuration
- **Type**: Single-node (control-plane + worker)
- **IP**: 172.22.174.58 (static)
- **Runtime**: CRI-O 1.31.5 with NVIDIA runtime
- **CNI**: Calico with VXLAN encapsulation
- **Storage**: Local persistent volumes

### GPU Configuration
- **Device**: NVIDIA A16
- **Memory**: 16GB GDDR6
- **Driver**: 580.95.05
- **CUDA**: 13.0
- **Runtime**: nvidia-container-runtime
- **Device Plugin**: v0.16.2

### Network Configuration
- **Pod Network**: 10.244.0.0/16
- **Service Network**: 10.96.0.0/12
- **DNS**: CoreDNS in cluster

---

##  Monitoring Stack (Upcoming)

### Components
- **Prometheus**: Metrics collection & storage
- **Grafana**: Visualization & dashboards
- **Node Exporter**: System metrics
- **DCGM Exporter**: GPU metrics
- **kube-state-metrics**: Kubernetes object metrics

### Dashboards (Planned)
- GPU utilization & temperature
- Container resource usage
- Inference latency & throughput
- Workload performance comparison

---

##  AI Workloads

### 1. ResNet50 (Image Classification)
- **Framework**: PyTorch/TensorFlow
- **Input**: Images (224x224)
- **Output**: Classification scores
- **Metrics**: Inference time, throughput, GPU utilization

### 2. DistilBERT (NLP)
- **Framework**: Transformers (Hugging Face)
- **Input**: Text sequences
- **Output**: Embeddings/classifications
- **Metrics**: Token processing speed, memory usage

### 3. Whisper (Speech-to-Text)
- **Framework**: OpenAI Whisper
- **Input**: Audio files
- **Output**: Transcriptions
- **Metrics**: Real-time factor, GPU memory, accuracy

---

##  Development Workflow

### Daily Work Cycle
```bash
# Morning: Pull latest
git pull origin main

# During work: Frequent commits
git add .
git commit -m "type: descriptive message"

# Evening: Push progress
git push origin main
```

### Commit Types
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation
- `test:` - Testing
- `perf:` - Performance improvements
- `refactor:` - Code restructuring
- `chore:` - Maintenance

---

##  Documentation

- **[JOURNAL.md](JOURNAL.md)** - Detailed technical journal
- **[docs/setup-guide.md](docs/setup-guide.md)** - Setup instructions
- **[docs/troubleshooting.md](docs/troubleshooting.md)** - Common issues

---

##  Testing & Validation

### Cluster Health
```bash
kubectl get nodes
kubectl get pods -A
kubectl cluster-info
```

### GPU Verification
```bash
nvidia-smi
kubectl describe node | grep nvidia.com/gpu
```

### Test Workload
```bash
kubectl apply -f k8s/test/gpu-test-pod.yaml
kubectl logs gpu-test
```

---

##  Research Objectives

1. **Workload Characterization**: Profile AI inference patterns
2. **Resource Modeling**: Map resource requirements
3. **Performance Analysis**: Measure latency, throughput, efficiency
4. **Scaling Behavior**: Analyze under different loads
5. **GPU Utilization**: Optimize GPU resource allocation

---

##  Related Links

- **GitHub**: [generative-ai-workload-modeling](https://github.com/Hamidhrf/generative-ai-workload-modeling)
- **Kubernetes Docs**: [kubernetes.io/docs](https://kubernetes.io/docs)
- **CRI-O**: [cri-o.io](https://cri-o.io)
- **Calico**: [docs.tigera.io/calico](https://docs.tigera.io/calico)

---

##  License

This is an academic research project for educational purposes.

---

##  Contact

**Hamidreza Fathollahzadeh**  
Master's Student - Digital Transformation  
Fachhochschule Dortmund

---

**Last Updated**: December 3, 2025  
**Cluster Version**: Kubernetes 1.34.0 with CRI-O 1.31.5