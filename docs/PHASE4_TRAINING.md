# Phase 4: TimeGAN Model Training — Complete Documentation

**Project:** Generative Modeling of Application Workloads for Synthetic Trace Generation  
**Author:** Hamidreza Fathollahzadeh  
**Institution:** Fachhochschule Dortmund  
**Date Range:** February 2026 — March 2026  
**Final Model:** S36 (Spectral Normalization)

---

## Executive Summary

Phase 4 implemented and iteratively refined a TimeGAN-based generative model for synthesizing realistic Kubernetes pod-level resource traces. Starting from a baseline LSTM and progressing through 39 model stages (S1-S39), the work culminated in S36 — a segment-based conditional GAN with spectral normalization — achieving a mean Variance Ratio (VR) of 1.116 across all 5 AI inference workloads, with every workload exceeding the thesis target of VR >= 0.8.

The development followed a systematic ablation methodology: each stage introduced exactly one architectural or training change, enabling clear causal attribution of improvements and regressions.

---

## Dataset

The training dataset consists of 176 pod-level traces collected from 35 experiments across 5 AI inference workloads deployed on a single-node Kubernetes cluster (16 vCPU, 62.5GB RAM, NVIDIA A16 GPU).

**Workloads:** BERT, GPT2, ResNet152, Whisper, YOLO  
**Replica counts:** r=1, 2, 3, 5, 6, 8, 10 per workload  
**Trace dimensions:** 715 timesteps x 10 metrics (at 5-second intervals, 60 minutes)  
**Trained metrics:** 7 per workload after dropping low/zero-variance features (gpu_memory_total, gpu_memory_used, gpu_temperature)

**Normalization:** Min-max scaling to [0, 1] per workload per metric.  
**Conditioning variables:** replica_count (normalized as (r-1)/9) and phase_index (0-5).

---

## Stage History

### Era 1: Architecture Exploration (S1-S9)

These early stages explored the fundamental GAN architecture, identifying critical failure modes that shaped all subsequent work.

**S1-S4: Encoder-based GAN with reconstruction loss.**  
The generator used an LSTM encoder to compress real traces into a latent vector z, then an LSTM decoder to reconstruct. This produced a train/test mismatch: during training the encoder saw real traces, but during generation z was sampled from random noise. The decoder had never seen random z values, causing it to output flat per-phase means — the "flat trace" failure mode that persisted through S1-S12.

**S5-S7: WGAN-GP with conditional discriminator.**  
Switched to Wasserstein GAN with gradient penalty. Added replica count conditioning to the discriminator. Discovered that WGAN-GP with LSTM critics is slow due to CuDNN double-backward limitations — workaround: disable CuDNN during gradient penalty computation. Results still showed flat synthetic traces due to the encoder mismatch.

**S8: Encoder dropout breakthrough.**  
Added dropout on the encoder during training, forcing the decoder to partially rely on noise. This broke the reconstruction lock and produced the first traces with genuine temporal dynamics. Mean VR improved to 1.678, but with excessive variance (BERT psi_cpu VR=5.0) and poor phase transitions (Whisper jump ratio 42x).

**S9: Variance penalty introduction.**  
Added adaptive variance cap to penalize over-generation. Tuned cap from 2.0 to 3.5 to accommodate low-variance workloads (ResNet152). Mean VR improved to 0.878, but ResNet152 collapsed to VR=0.324 — the variance cap was too aggressive for near-zero variance metrics.

### Era 2: Architectural Fixes (S10-S15)

**S10: Truly conditional discriminator.**  
Discovered that the discriminator accepted but ignored replica count — it was injected after the classification layer. Fixed by injecting r_norm into the classifier input. This was a critical fix: without it, the adversarial signal could not penalize replica-blind outputs.

**S11-S12: Huber variance regression.**  
Replaced the adaptive variance cap with Huber-loss variance regression that pulls fake std toward real std. Eliminated the cap sensitivity problem but flat traces persisted.

**S14: Encoder removal — the breakthrough.**  
Root cause identification: the encoder created an irrecoverable train/test mismatch. Fix: remove the encoder entirely. The generator now maps (z, r_norm, phase_idx) directly to traces. Train and generate paths are identical — z is always sampled from N(0,1). This was the single most important architectural change in the entire project. Flat traces disappeared immediately. Mean VR = 1.204.

**S15: Zero-mean preprocessing removal.**  
Discovered that zero-mean preprocessing caused value level mismatches between the sigmoid-bounded generator output [0,1] and unbounded targets. Removed zero-mean; generator targets are now raw min-max normalized [0,1] traces. This aligned the output activation with the target space.

### Era 3: Segment-Based Architecture (S16-S22)

**S16-S17: Per-phase feature matching and gradient clipping.**  
Added statistical matching loss (fm_stat) computed per phase per replica count group. Added gradient clipping to stabilize training. Incremental improvements.

**S18-S19: Per-workload hyperparameter tuning.**  
Tuned lambda_var_reg, lambda_fm_stat, lambda_smooth, and n_disc_steps per workload. Found that Whisper requires lower fm_stat weight (0.5 vs 1.0-2.0) and fewer discriminator steps (1 vs 2) due to its CPU-bound nature creating different optimization dynamics.

**S20: Segment-based generation.**  
Structural fix for phase boundary jump ratios. Instead of generating 720 timesteps end-to-end, split each trace into 6 phase segments of 120 timesteps. Each segment is generated independently, conditioned on phase_idx. This eliminated the jump ratio problem (from 8-42x down to 0.006-0.155x) because boundaries are now explicit concatenation points.

**S21: Precomputed FM targets + scale-invariant loss.**  
S20 computed feature matching statistics per batch, but with batch=32 and 42 phase-replica groups, most batches had 0-1 samples per group causing sparse gradients. S21 precomputes mean and std across the full training set. Combined with scale-invariant mean matching (normalized by |target_mean| + target_std), this ensured that low-range metrics (e.g., GPT2 psi_cpu at 0-0.002) contributed equally to the loss.

**S21 was selected as the first model meeting the thesis target.** Mean VR = 0.9146 (BERT=0.672, GPT2=0.996, ResNet152=0.960, Whisper=1.284, YOLO=0.661). Aggregate VR >= 0.8, though BERT and YOLO individually fell below.

**S22: Hyperparameter tuning of S21.** Minor improvements on some workloads, no structural change.

### Era 4: Cross-Workload and Unified Models (S27-S34)

**S23-S26: Autocorrelation loss experiments.**  
Attempted to add autocorrelation matching loss to fix temporal smoothing failures. S23 showed improvements for YOLO and Whisper but broke BERT. S24-S26 variants showed that autocorrelation loss caused regressions on GPT2 and Whisper level collapse. Abandoned as too risky — documented as a defensible negative result.

**S27: Per-workload tuned models.**  
Applied the optimal hyperparameters found in S18-S22 to all 5 workloads with the S21 architecture. Trained 5 separate generators. Achieved the best per-workload results to date, but Whisper was excluded from the final evaluation due to inconsistent metrics.

**S28-S29: Unified cross-workload model.**  
Experimented with a single generator trained on all workloads simultaneously, using workload embedding (nn.Embedding(5,16)). S29 achieved mean VR = 1.474 with workload embedding, demonstrating cross-workload knowledge transfer. However, YOLO remained below 0.8 threshold.

**S30-S33: GPU-bound and continuity-aware variants.**  
S31 tested continuity-aware loss for GPU-bound workloads (BERT, GPT2, ResNet152, YOLO), achieving VR > 0.8 on all 4 but with suspiciously high values (3.3-4.1) suggesting excessive variance. S33 achieved the best 4-workload results using the unified architecture without Whisper.

**S34: Unified baseline — all 5 workloads.**  
Combined the S27 per-workload hyperparameters with the unified 7-metric architecture and included all 5 workloads (adding Whisper back). This became the baseline for the final ablation study. Mean VR = 1.205, all 5 workloads passing VR >= 0.8.

### Era 5: Ablation Study and Final Selection (S35-S39)

With S34 as the controlled baseline, a systematic ablation study tested 5 improvement strategies, each changing exactly one variable.

**S35: Data augmentation (4x dataset).**  
Augmented 176 traces to 704 using sliding window and noise injection. Result: Mean VR = 1.359, distance from 1.0 increased by 75%. **Failed.** Augmented data introduced artifacts that confused the generator.

**S36: Spectral normalization on discriminator.**  
Added spectral normalization (Miyato et al., 2018) to the discriminator's classifier layers. This constrains the Lipschitz constant, preventing discriminator dominance and improving gradient flow to the generator. Result: Mean VR = 1.116, distance from 1.0 reduced by 43%. **Winner.** 3/5 workloads improved, YOLO showed the largest gain (1.346 to 1.039).

**S37: Discriminator dropout (0.3).**  
Added dropout to discriminator. Mean VR = 1.089 (closest to 1.0) but 3/5 workloads degraded individually. The mean was pulled down by one metric. **Rejected** due to inconsistency.

**S38: Multi-layer feature matching.**  
Feature matching from 3 discriminator layers instead of 1. Mean VR = 1.147, 26% worse than S36. **Failed.**

**S39: Targeted per-workload variance tuning.**  
Aggressive per-workload lambda_var_reg adjustment. Mean VR = 1.328, 183% worse than S36. **Catastrophic failure.** Over-regularization destroyed quality.

**S36 selected as the final model.**

---

## Final Model: S36

### Architecture

Generator: GeneratorSeg (segment-based, encoder-free)
- LSTM decoder: hidden_dim=128, num_layers=2, latent_dim=64
- Replica embedding: Linear(1, 16) + Tanh
- Phase embedding: Embedding(7, 8)
- Output: Sigmoid activation (bounded [0,1])
- Segment length: 120 timesteps
- Phases: 6 (full trace = 6 segments concatenated, trimmed to 715)

Discriminator: DiscriminatorSpectral
- Bidirectional LSTM: hidden_dim=32, num_layers=1
- Classifier: 2x Linear layers with spectral normalization
- Conditional on replica count (injected into classifier input)

### Per-Workload Hyperparameters

| Workload | lambda_var_reg | lambda_fm_stat | lambda_smooth | n_disc_steps |
|----------|---------------|----------------|---------------|-------------|
| BERT | 0.5 | 1.5 | 0.05 | 2 |
| GPT2 | 0.3 | 2.0 | 0.1 | 2 |
| ResNet152 | 0.4 | 1.0 | 0.1 | 2 |
| Whisper | 0.5 | 0.5 | 0.05 | 1 |
| YOLO | 0.3 | 1.2 | 0.1 | 2 |

### Training

- Warmup: 20 epochs (fm_stat + var_reg only, no adversarial loss)
- Adversarial: 150 epochs (WGAN-GP)
- Optimizer: Adam, lr=1e-3 (G), lr=2e-4 (D), betas=(0.0, 0.9)
- Checkpoint: best Wasserstein distance
- Batch size: 32

### Results

| Workload | Variance Ratio | Status |
|----------|---------------|--------|
| BERT | 1.022 | Pass |
| GPT2 | 1.137 | Pass |
| ResNet152 | 0.985 | Pass |
| Whisper | 1.397 | Pass |
| YOLO | 1.039 | Pass |
| **Mean** | **1.116** | **5/5 Pass** |

---

## Post-Processing Pipeline

Three post-processing steps are applied to raw S36 output before Kwok export:

**1. Cosine boundary blending (window=20 timesteps).** Eliminates sharp discontinuities at segment boundaries (at 10, 20, 30, 40, 50 minute marks) using a raised-cosine weight curve.

**2. Per-metric adaptive filtering.** Metrics classified as "smooth" (throughput, GPU power) receive a rolling mean filter (window=15) to remove GAN jitter. "Noisy" metrics (CPU, PSI, latency) are left unchanged.

**3. Pod memory reconstruction.** Pod memory is near-constant in real data (container memory = model weights loaded once at startup). The GAN output is replaced with a constant value plus tiny Gaussian noise, computed from real per-workload per-replica statistics.

Dropped metrics (gpu_memory_total, gpu_memory_used, gpu_temperature) are reconstructed as constants from real data statistics.

---

## Validation Results

### Scaling Curves (r=1-10 validated, r=15-50 extrapolation)

The model was validated against real data at all 7 measured replica counts. Extrapolation was explored at r=15, 20, 30, 50.

**Validated zone (r=1-10):** Synthetic scaling curves track real data across all metrics and workloads. Error bars overlap for most points.

**Extrapolation behavior:**
- GPT2: smooth continuation to r=50, physically plausible (latency increases, throughput decreases)
- Whisper: reasonable to r=20-30, then saturates
- ResNet152, YOLO: saturates by r=15-20 (flat extrapolation)
- BERT: saturates immediately after r=10 (lightweight workload, minimal contention signal)

### Wasserstein Distances

Zero clamping violations across all workloads and replica counts.

| Workload | Mean Wasserstein Distance |
|----------|--------------------------|
| BERT | 1.543 |
| YOLO | 1.197 |
| ResNet152 | 2.973 |
| GPT2 | 4.736 |
| Whisper | 5.515 |

Best-matched metrics: CPU Usage, CPU Pressure (PSI), Latency (all near-zero distances). Worst-matched: Pod Memory (reconstruction offset), GPU Power at r=1 (low-contention regime).

---

## Key Learnings

1. **Encoder removal was the single most important fix.** The train/test mismatch from encoding real traces during training but sampling random noise during generation caused all flat-trace failures in S1-S12.

2. **Zero-mean preprocessing caused value level mismatches.** Sigmoid-bounded generator outputs cannot match unbounded targets. Raw min-max normalization to [0,1] is the correct approach.

3. **Discriminator must be truly conditional.** Injecting conditioning variables after the classification layer means the discriminator ignores them. They must be part of the classifier input.

4. **Segment-based generation solves boundary jumps.** Generating 6 independent 120-step segments and concatenating is more effective than generating 720 steps end-to-end.

5. **Precomputed FM targets are essential for small datasets.** Per-batch statistics with 176 traces and 42 phase-replica groups are too noisy. Population-level precomputation provides stable training signals.

6. **Data augmentation fails with small GAN datasets.** Augmenting 176 to 704 traces introduced artifacts. The original data was sufficient for the architecture.

7. **Spectral normalization is the best single improvement.** Among 5 ablation candidates, spectral norm provided the largest and most consistent gain.

8. **Mean VR can be misleading.** S37 had the best mean VR (1.089) but degraded 3/5 workloads. Per-workload analysis is essential.

9. **Pod memory should be reconstructed, not generated.** Near-constant metrics are better served by statistical reconstruction than GAN synthesis.

10. **Extrapolation is bounded by the contention signal.** The model can only extrapolate meaningfully for workloads that showed strong contention gradients in training (GPT2, Whisper). Lightweight workloads (BERT, YOLO) saturate immediately beyond the training range.

---

## File Locations

### Scripts
- Training: `scripts/phase4/timegan/timegan_s{N}.py` (S21-S39)
- Evaluation: `scripts/phase4/evaluation/eval_s{N}.py`
- Comparison plots: `scripts/phase4/comparison/comparison_plots_s{N}.py`
- Post-processing: `scripts/phase4/postprocess_s36.py`
- Validation: `scripts/phase4/validate_s36.py`
- Utilities: `scripts/utils/boundary_smoothing.py`

### Models
- S21: `models/phase4/timegan_s21/s21_seg_vr03_fm10_ae150/{workload}/generator.pt`
- S27: `models/phase4/timegan_s27/s27_seg_vr03_fm10_ae150/{workload}/generator.pt`
- S34: `models/phase4/timegan_s34/s34_{workload}_vr{X}_fm{Y}/generator.pt`
- S36: `models/phase4/timegan_s36/s36_{workload}_vr{X}_fm{Y}/generator.pt`

### Data
- Unified dataset: `data/processed/phase4/unified/combined_dataset.npz`
- Normalization: `data/processed/phase4/unified/combined_normalization.json`
- Per-workload: `data/processed/phase4/raw/{workload}_traces.npz`

### Outputs
- Validation report: `outputs/phase4/validation/s36/validation_report.json`
- Scaling curves: `outputs/phase4/validation/s36/{workload}_scaling_curves.png`
- Wasserstein heatmaps: `outputs/phase4/validation/s36/{workload}_wasserstein_heatmap.png`
- Post-processed plots: `outputs/phase4/comparison_plots/s36_postprocessed/`

---

**Document Version:** 1.0  
**Last Updated:** March 24, 2026  
**Status:** Phase 4 Complete