# THESIS BRIEF v2.1
## Generative Modeling of Application Workloads for Synthetic Trace Generation

**Purpose:** Single reference document for writing the full research thesis. Every fact, number, and decision has been verified against project records.

**Type:** Research Project Thesis 
**Author:** Hamidreza Fathollahzadeh (Matriculation: 7219187)
**Program:** Master's in Digital Transformation (MDT), Fachhochschule Dortmund
**Supervisor:** Prof. Dr. Stephan Recker
**Project Duration:** November 2025 - April 2026

---


## PART 1: PROJECT FACTS 

### 1.1 Infrastructure

| Component | Specification |
|-----------|--------------|
| VM | Ubuntu 24.04 LTS, University-hosted |
| CPU | 16 vCPUs |
| RAM | 62.5 GB |
| GPU | NVIDIA A16, 16 GB GDDR6, Compute 8.6 (Ampere) |
| GPU Sharing | Time-slicing, 10 virtual slices |
| Kubernetes | v1.34.0 |
| Container Runtime | CRI-O 1.31.5 |
| Monitoring | Prometheus (15-day retention), Grafana, DCGM Exporter, Node Exporter |
| Repository | github.com/Hamidhrf/generative-ai-workload-modeling |

Single-node cluster (control plane + worker), control-plane taint removed.

### 1.2 Workloads

**Phase 1 v1 (Dec 2025 - Jan 2026): 3 workloads, 13 experiments**
ResNet-50, DistilBERT, Whisper (small). Limited replica counts. Total ~60 pod traces.

**Phase 1 v3 (Feb 2026): 5 workloads, 50 experiments (FINAL)**

| Workload | Type | Framework | Resource Profile | Key Behavior |
|----------|------|-----------|-----------------|-------------|
| BERT | NLP Classification | HuggingFace | Moderate GPU | Linear GPU scaling (2%-27%), stable latency |
| GPT-2 | Text Generation | HuggingFace | Heavy GPU | Non-linear GPU saturation (21%-98%), exponential latency |
| ResNet-152 | Image Classification | PyTorch | Balanced | Linear GPU scaling, consistent performance |
| Whisper | Speech-to-Text | OpenAI Whisper | CPU-Intensive | CPU-bound (PSI=0.63 at r=10), pod crashes |
| YOLO | Object Detection | Ultralytics | Light GPU | Very light resource usage (GPU 1%-15%) |

**Why expanded:** Initial training on 3 workloads showed insufficient diversity and data.

### 1.3 Experimental Design

- **Replicas:** r=1 through r=10 (all integers) per workload
- **Experiments:** 50 total (5 x 10)
- **Pod traces:** 275 total (55 per workload: 1+2+...+10)
- **Duration:** 60 min per experiment, 5-second intervals, 715 timesteps per trace

**Business Day load pattern:** 6 phases simulating daily traffic:
Phase 0 (0-8 min): Night/Warmup, Phase 1 (8-15 min): Morning ramp-up, Phase 2 (15-25 min): Midday high, Phase 3 (25-35 min): Afternoon sustained, Phase 4 (35-50 min): Peak/Evening, Phase 5 (50-60 min): Cooldown.

Fixed inference rate per pod. Contention from shared GPU/CPU, not increased per-pod rate. GPU time-slicing: 10 slices, each pod requests 1.

### 1.4 Metrics (12 -> 10 -> 7)

**12 collected:** pod_cpu_usage, pod_memory_bytes, pod_psi_cpu, pod_psi_memory, pod_psi_io, pod_latency_avg, pod_throughput, gpu_utilization, gpu_memory_used, gpu_memory_total, gpu_power_watts, gpu_temperature

**Removed (zero variance):** pod_psi_memory, pod_psi_io

**Dropped from training (near-constant):** gpu_memory_total, gpu_memory_used, gpu_temperature (reconstructed in post-processing)

**7 trained by S36:** pod_cpu_usage, pod_memory_bytes, pod_psi_cpu, pod_latency_avg, pod_throughput, gpu_utilization, gpu_power_watts

pod_memory_bytes trained but replaced in post-processing (near-constant in real data).

### 1.5 Known Limitations

- **Latency:** Only pod_latency_avg (not p50/p95/p99). Prometheus expired. App-level percentiles in CSV but pod-level unrecoverable.
- **Whisper r=10 pod_9:** Partial data, extreme CPU stress, zero throughput. Kept as realistic degradation.
- **GPU metrics system-level:** Not per-pod. Inherent to GPU time-slicing.

---

## PART 2: MODEL DEVELOPMENT

### 2.1 Phase 2: Model Selection

**LSTM Baseline:** Mean VR=0.612. BERT 0.594, GPT-2 0.676, ResNet-152 0.607, Whisper 0.748, YOLO 0.438.

**TimeVAE:** v1 VR=0.613, v2 VR=0.517 (undertrained), v3 VR=0.148 (flat traces).

**Key finding: MSE loss ceiling at VR~0.61.** All non-adversarial architectures converge here. MSE produces conditional mean, suppresses impulsive metrics. Adversarial training needed.

### 2.2 Phase 4: TimeGAN (39 Stages)

**Era 1 (S1-S9):** Encoder-based GANs. Train/test mismatch caused flat traces. WGAN-GP tried.

**Era 2 (S10-S15):** S10 fixed unconditional discriminator. **S14 BREAKTHROUGH: encoder removal.** Direct noise-to-trace mapping. VR=1.204. S15 removed zero-mean preprocessing.

**Era 3 (S16-S22):** S20 segment-based generation (6 phases x 120 steps). S21 precomputed FM targets. VR=0.914.

**Era 4 (S23-S34):** Cross-workload unification. S34 baseline: 5 workloads, 7 metrics, VR=1.205.

**Era 5 (S35-S39) Ablation:**

| Stage | Change | Mean VR | Dist. from 1.0 | Result |
|-------|--------|---------|----------------|--------|
| S34 | Baseline | 1.205 | 0.205 | Reference |
| S35 | Data augmentation 4x | 1.359 | 0.359 (+75%) | Failed |
| S36 | Spectral norm | 1.116 | 0.116 (-43%) | **WINNER** |
| S37 | Dropout 0.3 | 1.124 | 0.124 | Mixed |
| S38 | Multi-layer FM | 1.147 | 0.147 (+26%) | Failed |
| S39 | Variance tuning | 1.328 | 0.328 (+183%) | Catastrophic |

### 2.3 Final Model: S36

Generator: Encoder-free segment-based LSTM (hidden=128, layers=2, latent=64). Discriminator: BiLSTM with spectral norm. Per-workload hyperparams. 20 warmup + 150 adversarial epochs.

| Workload | VR | Wasserstein | Status |
|----------|-----|------------|--------|
| BERT | 1.022 | 1.543 | Pass |
| GPT-2 | 1.137 | 4.736 | Pass |
| ResNet-152 | 0.985 | 2.973 | Pass |
| Whisper | 1.397 | 5.515 | Pass |
| YOLO | 1.039 | 1.197 | Pass |
| **Mean** | **1.116** | **3.193** | **5/5** |

### 2.4 Post-Processing
Cosine boundary blending, adaptive filtering, memory reconstruction, dropped metric reconstruction. Zero clamping violations.

### 2.5 Extrapolation
GPT-2 to r=50, Whisper to r=20-30, BERT/ResNet/YOLO saturate by r=15-20.

---

## PART 3: THESIS STRUCTURE 

### Front Matter 
Title page, Abstract, Declaration, TOC, List of Figures, List of Tables

### Chapter 1: Introduction 
1.1 Motivation, 1.2 Problem Statement and Research Questions, 1.3 Contributions, 1.4 Thesis Structure

### Chapter 2: Background and Related Work 
2.1 Intro, 2.2 Kubernetes and Container Orchestration, 2.3 GPU Sharing for Inference, 2.4 Workload Characterization and Digital Twins, 2.5 PSI Metrics, 2.6 Time-Series Generative Models (GAN foundations, TimeGAN, DoppelGANger, conditional generation), 2.7 Evaluation of Synthetic Time Series, 2.8 Gap Analysis

### Chapter 3: Methodology 
3.1 Intro, 3.2 Infrastructure, 3.3 Workload Selection, 3.4 Experimental Design, 3.5 Data Collection, 3.6 Preprocessing, 3.7 Evaluation Framework

### Chapter 4: Model Development 
4.1 Intro, 4.2 LSTM Baseline, 4.3 TimeVAE Experiments, 4.4 TimeGAN Architecture, 4.5 Training Evolution (S1-S39), 4.6 Ablation Study, 4.7 Post-Processing, 4.8 Validation

### Chapter 5: Results and Discussion 
5.1 Intro, 5.2 Final Model Performance, 5.3 Scaling Curves, 5.4 Model Comparison, 5.5 Extrapolation, 5.6 Post-Processing Impact, 5.7 Discussion and Limitations

### Chapter 6: Conclusion and Future Work 
6.1 Contributions, 6.2 Limitations, 6.3 Future Work

### Back Matter
References (~30), Appendices

---

## PART 4: REFERENCES 

### Foundational ML/DL
[1] Goodfellow et al., "Generative Adversarial Nets," NeurIPS, 2014. arxiv:1406.2661
[2] Kingma, Welling, "Auto-Encoding Variational Bayes," ICLR, 2014. arxiv:1312.6114
[3] Hochreiter, Schmidhuber, "Long Short-Term Memory," Neural Comp., 1997. doi:10.1162/neco.1997.9.8.1735
[4] Vaswani et al., "Attention Is All You Need," NeurIPS, 2017. arxiv:1706.03762

### GAN Training Stability
[5] Arjovsky et al., "Wasserstein GAN," ICML, 2017. arxiv:1701.07875
[6] Gulrajani et al., "Improved Training of Wasserstein GANs," NeurIPS, 2017. arxiv:1704.00028
[7] Miyato et al., "Spectral Normalization for GANs," ICLR, 2018. openreview.net/forum?id=B1QRgziT-

### Time-Series Generation
[8] Yoon et al., "Time-series Generative Adversarial Networks," NeurIPS, 2019 (TimeGAN)
[9] Desai et al., "TimeVAE," arXiv:2111.08095, 2021
[10] Lin et al., "DoppelGANger," ACM IMC, 2020. doi:10.1145/3419394.3423643
[11] Naiman et al., "Koopman VAEs," ICLR, 2024. arxiv:2310.02619
[12] Bengio et al., "Scheduled Sampling," NeurIPS, 2015. arxiv:1506.03099

### Surveys and Evaluation
[13] Brophy et al., "GANs in Time Series: Systematic Review," ACM CSUR, 2023. doi:10.1145/3559540
[14] Ribeiro et al., "Measuring Fidelity/Utility of Time Series GANs," IEEE ISCC, 2024. doi:10.1109/ISCC61673.2024.10733699

### Workload Traces
[15] Bergsma et al., "Generating Cloud Workloads using RNNs," ACM SOSP, 2021. doi:10.1145/3477132.3483590
[16] Leznik et al., "Multivariate Time Series Synthesis Using GANs," ACM ICPE, 2021. doi:10.1145/3427921.3450257
[17] Pang, Kant, "Synthetic Storage Trace Generation," ACM TOS, 2026. doi:10.1145/3767317

### Kubernetes
[18] Burns et al., "Borg, Omega, and Kubernetes," ACM Queue, 2016
[19] Carrion, "Kubernetes Scheduling: Taxonomy," ACM CSUR, 2022. doi:10.1145/3539606
[20] Rejiba, Chamanara, "Custom Scheduling in Kubernetes," ACM CSUR, 2022. doi:10.1145/3544788
[21] Gough, "Evaluation of Kubernetes Schedulers," ACM PEARC, 2024. doi:10.1145/3626203.3670520

### GPU Sharing
[22] Dhakal et al., "GSLICE: Controlled GPU Sharing," ACM SoCC, 2020. doi:10.1145/3419111.3421284
[23] Yeh et al., "KubeShare," ACM HPDC, 2020. doi:10.1145/3369583.3392679
[24] Li et al., "MISO: Multi-Instance GPU," ACM SoCC, 2022. doi:10.1145/3542929.3563510
[25] Ye et al., "DL Workload Scheduling in GPU Datacenters," ACM CSUR, 2024. doi:10.1145/3638757
[26] Wei et al., "Jormungandr: GPU Allocation," ACM SYSTOR, 2024. doi:10.1145/3688351.3689156

### Digital Twins
[27] Bhardwaj, Benson, "KubeKlone," ACM APNet, 2023. doi:10.1145/3542637.3542642
[28] Borsatti et al., "KubeTwin," IEEE TNSM, 2024. doi:10.1109/TNSM.2024.3405175

### Conditional Generation
[29] Sun et al., "Decision-Aware Conditional GANs," ACM, 2023. doi:10.1145/3604237.3626855
[30] Lu et al., "Conditional GAN for Time Series," IEEE TKDE, 2024. doi:10.1109/TKDE.2023.3310909

### AI Models (cite in methodology only)
BERT (Devlin, NAACL 2019), GPT-2 (Radford, OpenAI 2019), ResNet (He, CVPR 2016), Whisper (Radford, ICML 2023), YOLO (Jocher, 2023)

### Tools
Kubernetes docs, Prometheus docs, KWOK (kwok.sigs.k8s.io), PSI (Weiner, Facebook 2018), DCGM Exporter (NVIDIA)

---

## PART 5: FIGURES 

### Figures
1. System architecture — TikZ (system_architecture_tikz.tex / system_architecture.pdf)
2. Business Day load pattern (load_pattern.pdf)
3. TimeGAN architecture — TikZ (timegan_architecture_tikz.tex / timegan_architecture.pdf)
4. Encoder removal before/after (encoder_removal.pdf)
5. Real vs Synthetic traces at r=5 (real_vs_synthetic_r5.pdf)
6. Post-processing before/after (postprocessing_before_after.pdf)
7. Ablation study bar chart (ablation_study.pdf)
8. Scaling curves — GPU util + latency x 5 workloads (scaling_curves.pdf)
9. Model comparison — LSTM/TimeVAE/S36 (model_comparison.pdf)
10. Extrapolation curves — GPT-2 + Whisper to r=50 (extrapolation_curves.pdf)
11. Wasserstein distance heatmap (wasserstein_heatmap.pdf)
12. Wasserstein distance per replica count (wasserstein_per_replica.pdf)

### Tables
1. Infrastructure specs
2. Workload characteristics
3. Metrics with units
4. LSTM results
5. TimeVAE results
6. TimeGAN milestones (S1-S39 selected)
7. Ablation (S34-S39)
8. S36 final results
9. Per-workload hyperparams
10. Extrapolation assessment
11. Timeline

---


**Version:** 2.1 | **Updated:** April 20, 2026