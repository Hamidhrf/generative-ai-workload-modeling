# THESIS BRIEF v2.2
## Generative Modeling of Application Workloads for Synthetic Trace Generation

**Type**: Research Project Thesis
**Author**: Hamidreza Fathollahzadeh
**Program**: Master's in Digital Transformation (MDT), Fachhochschule Dortmund
**Supervisor**: Prof. Dr. Stephan Recker
**Project Duration**: November 2025 – May 2026

---

## PART 1: PROJECT FACTS (VERIFIED)

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
| Monitoring | Prometheus (15-day retention, 5 s scrape), Grafana, DCGM Exporter, Node Exporter |
| Repository | github.com/Hamidhrf/generative-ai-workload-modeling |

Single-node cluster (control plane + worker), control-plane taint removed.

### 1.2 Workloads

**Phase 1 v1 (Dec 2025 – Jan 2026)**: 3 workloads, 13 experiments (ResNet-50,
DistilBERT, Whisper). Limited replica counts. ~60 pod traces. Superseded.

**Phase 1 v3 (Feb 2026): 5 workloads, 50 experiments — FINAL**

| Workload | Type | Framework | Profile | Key Behavior |
|----------|------|-----------|---------|-------------|
| BERT | NLP Classification | HuggingFace | Moderate GPU | Linear GPU scaling (2%–27%), stable latency |
| GPT-2 | Text Generation | HuggingFace | Heavy GPU | Non-linear GPU saturation (21%–98%), exponential latency |
| ResNet-152 | Image Classification | PyTorch | Balanced | Linear GPU scaling, consistent performance |
| Whisper | Speech-to-Text | OpenAI Whisper | CPU-Intensive | CPU-bound (PSI=0.63 at r=10), pod crashes |
| YOLO | Object Detection | Ultralytics | Light GPU | Very light resource usage (GPU 1%–15%) |

### 1.3 Experimental Design

- **Replicas**: r=1 through r=10 (all integers) per workload
- **Experiments**: 50 total (5 × 10)
- **Pod traces**: 275 total (55 per workload: 1+2+...+10)
- **Duration**: 60 min per experiment, 5-second intervals, 715 timesteps per trace (resampled to 720)

**Business-day load pattern** (6 phases):
- Phase 0 (0–8 min): Night/Warmup
- Phase 1 (8–15 min): Morning ramp-up
- Phase 2 (15–25 min): Midday high
- Phase 3 (25–35 min): Afternoon sustained
- Phase 4 (35–50 min): Peak/Evening
- Phase 5 (50–60 min): Cooldown

Fixed inference rate per pod. Contention from shared GPU/CPU, not increased per-pod rate.
GPU time-slicing: 10 slices, each pod requests 1.

### 1.4 Metrics (12 → 10 → 7)

**12 collected**: pod_cpu_usage, pod_memory_bytes, pod_psi_cpu, pod_psi_memory,
pod_psi_io, pod_latency_avg, pod_throughput, gpu_utilization, gpu_memory_used,
gpu_memory_total, gpu_power_watts, gpu_temperature

**Removed (zero variance)**: pod_psi_memory, pod_psi_io

**Dropped from training (near-constant)**: gpu_memory_total, gpu_memory_used,
gpu_temperature — reconstructed in post-processing

**7 trained by S36**: pod_cpu_usage, pod_memory_bytes, pod_psi_cpu,
pod_latency_avg, pod_throughput, gpu_utilization, gpu_power_watts

TRAINED_INDICES into the 10-metric array: `[0, 1, 2, 3, 4, 5, 8]`
(gpu_power_watts is index 8)

pod_memory_bytes is trained but replaced by statistical reconstruction in post-processing
(near-constant within experiments).

### 1.5 Train/Validation Split

49 pods to training, 6 pods to validation per workload (90/10). Split at pod level
(not segment level) to prevent leakage. Fixed seed=42. No separate test set.

With 6 segments per pod: 1,470 training segments, 180 validation segments.

Data source: `data/processed/phase4/unified/combined_dataset.npz` +
`combined_normalization.json` (not per-workload raw NPZ files).

### 1.6 Known Limitations

- **Latency**: Only pod_latency_avg collected at pod level. p50/p95/p99 not
  recoverable (Prometheus 15-day retention expired). App-level percentile CSVs
  exist but pod-level percentiles cannot be derived.
- **Whisper r=10 pod_9**: Partial data, extreme CPU stress, zero throughput. Kept as
  realistic degradation.
- **GPU metrics system-level**: Not per-pod. Per-pod values derived as 1/r of
  system value. This is a modeling assumption, not a measured quantity.
- **Single-seed model selection**: All ablation stages run with one seed.
- **No separate test set**: Validation set serves as both development metric and
  final evaluation set.

---

## PART 2: MODEL DEVELOPMENT

### 2.1 Baselines

**LSTM Baseline**: Mean VR=0.613. BERT 0.594, GPT-2 0.676, ResNet-152 0.607,
Whisper 0.748, YOLO 0.438. All 0/5 pass. Phase-conditioned autoregressive LSTM,
2 layers, hidden=128. Trained with MSE.

**TimeVAE**:
- v1: Mean VR=0.613. BERT 0.388, GPT-2 0.781, ResNet-152 0.557, Whisper 0.703, YOLO 0.635.
- v2: Mean VR=0.517. BERT 0.458, GPT-2 0.645, ResNet-152 0.405, Whisper 0.679, YOLO 0.396.
- v3: Mean VR=0.148. BERT 0.114, GPT-2 0.254, ResNet-152 0.066, Whisper 0.210, YOLO 0.096.

**Key finding**: MSE loss ceiling at VR~0.61. Both LSTM and TimeVAE v1 reach the
same value despite different architectures. MSE trains for the conditional mean,
suppressing variability. Adversarial training required.

Note: LSTM and TimeVAE v1/v3 evaluated on smaller per-workload metric sets
(4–6 metrics). S36 evaluated on 7 metrics. Numbers not directly comparable in
absolute terms but the qualitative ordering is robust.

### 2.2 Phase 4: TimeGAN (39 Stages)

**Era 1 (S1–S9)**: Encoder-based GANs. Train/test mismatch — discriminator sees
encoded real data during training but random noise at generation. Flat traces.
WGAN-GP tried. Problem not resolved.

**Era 2 (S10–S15)**:
- S10: Fixed unconditional discriminator (accepted r as input but never used it).
- **S14 BREAKTHROUGH**: Encoder removed entirely. Direct noise-to-trace mapping.
  Mean VR=1.204. BERT 0.718, GPT-2 1.232, ResNet-152 0.992, Whisper 1.654,
  YOLO 1.422. BERT still fails threshold (0.718); 4/5 pass.
  Note: G maps (z, r) only at S14, not (z, r, p) — phase index added later.
- S15: Removed zero-mean preprocessing (mismatch with sigmoid output). Switched
  to min-max normalization. Mean VR=1.144.

**Era 3 (S16–S22)**:
- S20: Segment-based generation (6 phases × 120 steps). LSTM sequence length
  reduced from 720 to 120.
- S21: Precomputed FM targets from real data (stable reference vs. evolving
  discriminator representation). Mean VR~1.000 on partial 5–6 metric set.

**Era 4 (S23–S34)**: Cross-workload unification. Standardized architecture across
all workloads (same structure, same 7 metrics, per-workload loss weights only).
S29 tested as unified cross-workload model — underperformed on Whisper (−39%)
and YOLO (−24%), per-workload approach retained.
S34 unified baseline: 5 workloads, 7 metrics, mean VR=1.205. All 5 pass.

**Era 5 (S35–S39) Ablation**:

| Stage | Change | Mean VR | Dist. from 1.0 | Pass | Result |
|-------|--------|---------|----------------|------|--------|
| S34 | Baseline | 1.205 | 0.205 | 5/5 | Reference |
| S35 | 4× data augmentation | 1.359 | 0.359 (+75%) | 5/5 | Excess variance |
| S36 | Spectral norm on D FC layers | 1.116 | 0.116 (−43%) | 5/5 | **WINNER** |
| S37 | Discriminator dropout p=0.3 | 1.124 | 0.124 (−40%) | 5/5 | Comparable |
| S38 | Multi-layer feature matching | 1.118 | 0.118 (−42%) | 5/5 | Comparable |
| S39 | Variance-loss tuning | 1.328 | 0.328 (+60%) | 5/5 | Excess variance |

S36 selected: lowest deviation from ideal, no additional hyperparameter,
most consistent per-workload performance. S37 and S38 are comparable but
differences are within single-seed noise.

### 2.3 Final Model: S36

**Generator**: Encoder-free segment-based LSTM (hidden=128, layers=2, latent=64).
Conditioned on r and p through Linear-Tanh projection layers that initialize
LSTM hidden and cell states. Output: 120 × 7 per segment, sigmoid activation.

**Discriminator**: BiLSTM (2 × 128 hidden), spectral normalization on FC output
layers only (not BiLSTM, not generator).

**Training**: 20 warmup epochs (FM + variance only, no adversarial) +
150 adversarial epochs. Adam: G lr=1e-3, D lr=2e-4, beta=(0.0, 0.9),
weight_decay=1e-5. Batch size=32. Grad clip: G=1.0, D=5.0.

**Per-workload hyperparameters**:

| Workload | lambda_vr | lambda_fm | Checkpoint dir |
|----------|-----------|-----------|----------------|
| BERT | 0.05 | 0.15 | s36_bert_vr05_fm15 |
| GPT-2 | 0.03 | 0.20 | s36_gpt2_vr03_fm20 |
| ResNet-152 | 0.04 | 0.10 | s36_resnet152_vr04_fm10 |
| Whisper | 0.05 | 0.05 | s36_whisper_vr05_fm05 |
| YOLO | 0.03 | 0.12 | s36_yolo_vr03_fm12 |

**S36 final per-workload results**:

| Workload | VR | Wasserstein | Pass (>= 0.8) |
|----------|----|-------------|---------------|
| BERT | 1.022 | 1.543 | Yes |
| GPT-2 | 1.137 | 4.736 | Yes |
| ResNet-152 | 0.985 | 2.973 | Yes |
| Whisper | 1.397 | 5.515 | Yes |
| YOLO | 1.039 | 1.197 | Yes |
| **Mean** | **1.116** | **3.193** | **5/5** |

Result source: `outputs/phase4/timegan_s36/s36_eval_results.json`
(summary.mean_vr_smooth = 1.1159913)

Wasserstein source: `outputs/phase4/validation/s36/validation_report.json`

### 2.4 Post-Processing

All steps in `scripts/phase4/postprocess_s36.py`:

1. **Cosine boundary blending** (k=20): Smooth transitions between the 6 independently
   generated segments.
2. **Adaptive per-metric filtering**: Uniform rolling-mean (kernel=15) applied only to
   smooth-classified metrics. NOT Savitzky-Golay. Metrics classified by visual
   inspection of r=5 plots. Per-workload overrides for Whisper and GPT-2.
3. **Memory reconstruction**: pod_memory_bytes replaced with flat trace at
   per-(workload, r) empirical mean from real data.
4. **Dropped metric reconstruction**:
   - gpu_memory_total: hardware constant (15,356 MiB)
   - gpu_memory_used, gpu_temperature: sampled from per-workload distributions
   - pod_psi_memory, pod_psi_io: set to zero
5. **Physical validity clamping**: Zero clamping violations in S36 output.

### 2.5 Extrapolation

Generated traces are physically plausible up to:

| Workload | Useful Range | Reason |
|----------|-------------|--------|
| GPT-2 | up to r ~ 50 | Near saturation at r=10 (98% GPU), full curve learned |
| Whisper | up to r ~ 20–30 | CPU PSI still rising at r=10, extrapolation plausible |
| BERT | up to r ~ 20 | Below saturation at r=10, model has not seen transition |
| ResNet-152 | up to r ~ 20 | Same as BERT |
| YOLO | up to r ~ 15 | Very light usage, limited informative range |

Extrapolation rests on the capacity scaling assumption: infrastructure scales
proportionally with r. The model does not encode cluster topology; this is
left to the KWOK configuration.

---

## PART 3: THESIS STRUCTURE

### Front Matter
Title page, Abstract (p. i), Declaration (p. ii), Contents (pp. iii–v),
List of Figures (p. vi), List of Tables (p. vii)

### Chapter 1: Introduction (pp. 1–5)
1.1 Motivation, 1.2 Problem Statement and Research Questions,
1.3 Contributions, 1.4 Thesis Structure

### Chapter 2: Background and Related Work (pp. 6–14)
2.1 Kubernetes and Container Orchestration, 2.2 GPU Sharing for Inference,
2.3 Workload Characterization and Digital Twins, 2.4 Pressure Stall Information,
2.5 Time-Series Generative Models (LSTM, VAE, GAN, TimeGAN, DoppelGANger,
Conditional Generation), 2.6 Evaluation of Synthetic Time Series, 2.7 Gap Analysis

### Chapter 3: Methodology (pp. 15–24)
3.1 Infrastructure, 3.2 Workload Selection, 3.3 Experimental Design
(3.3.1 Replica Scaling, 3.3.2 Business-Day Load Pattern, 3.3.3 Uniform Rate Justification),
3.4 Data Collection, 3.5 Preprocessing (ZV removal, near-constant handling,
memory treatment, normalization, segment windowing, training structure, train/val split),
3.6 Evaluation Framework (VR, Wasserstein, visual comparison)

### Chapter 4: Model Development (pp. 25–39)
4.1 LSTM Baseline, 4.2 TimeVAE Experiments, 4.3 TimeGAN Architecture
(Generator, Discriminator, Training Losses, Spectral Normalization, Training Schedule),
4.4 Training Evolution (Era 1–5), 4.5 Ablation Study, 4.6 Post-Processing,
4.7 Validation

### Chapter 5: Results and Discussion (pp. 40–50)
5.1 Final Model Performance (per-workload, per-metric), 5.2 Scaling Curves
(+ Capacity Scaling Assumption), 5.3 Model Comparison, 5.4 Extrapolation,
5.5 Post-Processing Pipeline, 5.6 Discussion and Limitations (RQs revisited,
limitations)

### Chapter 6: Conclusion and Future Work (pp. 51–54)
6.1 Summary, 6.2 Contributions, 6.3 Future Work

### Back Matter
Bibliography (pp. 55–58, 35 entries), Appendix A: Scaling Curves for Remaining
Metrics (pp. 59–60)

---

## PART 4: REFERENCES (35 ENTRIES, IEEEtranN)

Note: IEEEtranN silently drops bare doi= fields. All DOI entries use
url = {https://doi.org/<doi>} companion fields in the .bib file.

### Core ML/DL
- Goodfellow et al., "Generative Adversarial Nets," NeurIPS, 2014.
- Kingma & Welling, "Auto-Encoding Variational Bayes," ICLR, 2014.
- Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Comp., 1997. doi:10.1162/neco.1997.9.8.1735
- Miyato et al., "Spectral Normalization for GANs," ICLR, 2018.
- Arjovsky et al., "Wasserstein GAN," ICML, 2017.
- Gulrajani et al., "Improved Training of Wasserstein GANs," NeurIPS, 2017.
- Bengio et al., "Scheduled Sampling," NeurIPS, 2015.

### Time-Series Generation
- Yoon et al., "TimeGAN," NeurIPS, 2019.
- Desai et al., "TimeVAE," arXiv:2111.08095, 2021.
- Lin et al., "DoppelGANger," ACM IMC, 2020. doi:10.1145/3419394.3423643
- Naiman et al., "Koopman VAEs," ICLR, 2024.
- Brophy et al., "GANs in Time Series: Systematic Review," ACM CSUR, 2023. doi:10.1145/3559540
- Ribeiro et al., "Fidelity/Utility of Time Series GANs," IEEE ISCC, 2024. doi:10.1109/ISCC61673.2024.10733699

### Workload Traces
- Bergsma et al., "Generating Cloud Workloads using RNNs," SOSP, 2021. doi:10.1145/3477132.3483590
- Leznik et al., "Multivariate TS Synthesis Using GANs," ICPE, 2021. doi:10.1145/3427921.3450257
- Pang & Kant, "Synthetic Storage Trace Generation," ACM TOS, 2026. doi:10.1145/3767317

### Kubernetes
- Burns et al., "Borg, Omega, and Kubernetes," ACM Queue, 2016. doi:10.1145/2890784
- Carrion, "Kubernetes Scheduling: Taxonomy," ACM CSUR, 2022. doi:10.1145/3539606
- Rejiba & Chamanara, "Custom Scheduling in Kubernetes," ACM CSUR, 2022. doi:10.1145/3544788
- Gough, "Evaluation of Kubernetes Schedulers," PEARC, 2024. doi:10.1145/3626203.3670520

### GPU Sharing
- Dhakal et al., "GSLICE," ACM SoCC, 2020. doi:10.1145/3419111.3421284
- Yeh et al., "KubeShare," ACM HPDC, 2020. doi:10.1145/3369583.3392679
- Li et al., "MISO," ACM SoCC, 2022. doi:10.1145/3542929.3563510
- Ye et al., "DL Workload Scheduling Survey," ACM CSUR, 2024. doi:10.1145/3638757
- Wei et al., "Jormungandr," ACM SYSTOR, 2024. doi:10.1145/3688351.3689156

### Digital Twins
- Bhardwaj & Benson, "KubeKlone," APNet, 2022. doi:10.1145/3542637.3542642
- Borsatti et al., "KubeTwin," IEEE TNSM, 2024. doi:10.1109/TNSM.2024.3405175

### Conditional Generation
- Sun et al., "Decision-Aware Conditional GANs," ICAIF, 2023. doi:10.1145/3604237.3626855
- Lu et al., "Conditional GAN for Time Series," IEEE TKDE, 2024. doi:10.1109/TKDE.2023.3310909

### KWOK / PSI
- KWOK: https://kwok.sigs.k8s.io (accessed Apr. 2026)
- Weiner, "Pressure stall information," 2018 (accessed Apr. 2026)

### AI Models (methodology citations)
- Devlin et al., "BERT," NAACL, 2019.
- Radford et al., "Language Models Are Unsupervised Multitask Learners," OpenAI, 2019. (GPT-2)
- He et al., "Deep Residual Learning," CVPR, 2016. doi:10.1109/CVPR.2016.90
- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," ICML, 2023. (Whisper)
- Jocher et al., "Ultralytics YOLO," 2023. (accessed Apr. 2026)

---

## PART 5: FIGURES (12, ALL IN figures/)

| File | Figure | Location in thesis |
|------|--------|--------------------|
| system_architecture.pdf | Fig. 3.1: System architecture and monitoring stack | Ch. 3 |
| load_pattern.pdf | Fig. 3.2: Business-day load pattern | Ch. 3 |
| timegan_architecture.pdf | Fig. 4.1: TimeGAN S36 architecture | Ch. 4 |
| encoder_removal.pdf | Fig. 4.2: Effect of encoder removal | Ch. 4 |
| ablation_study.pdf | Fig. 4.3: Ablation study variance ratios | Ch. 4 |
| postprocessing_before_after.pdf | Fig. 4.4: Effect of post-processing | Ch. 4 |
| real_vs_synthetic_r5.pdf | Fig. 4.5: Real vs. synthetic traces at r=5 | Ch. 4 |
| wasserstein_heatmap.pdf | Fig. 5.1: Wasserstein distance heatmap | Ch. 5 |
| wasserstein_per_replica.pdf | Fig. 5.2: Mean Wasserstein vs. replica count | Ch. 5 |
| scaling_curves.pdf | Fig. 5.3: Real vs. synthetic scaling curves | Ch. 5 |
| model_comparison.pdf | Fig. 5.4: Trace comparison across models | Ch. 5 |
| extrapolation_curves.pdf | Fig. 5.5: Extrapolation curves GPT-2 + Whisper | Ch. 5 |
| (appendix figure) | Fig. A.1: Scaling curves remaining metrics | Appendix A |

Generation scripts: `scripts/phase4/thesis_figures/`

---

**Version**: 2.2 | **Updated**: June 2026 