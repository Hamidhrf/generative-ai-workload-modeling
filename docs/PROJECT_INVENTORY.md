# PROJECT_INVENTORY.md

Generated on 2026-04-24 against the working tree on branch `phase4-model-training`.
No branch checkouts were performed; other-branch inspection used `git ls-tree` / `git show`
so no uncommitted work was touched. Sizes are in bytes unless stated otherwise.

---

## Section 1: Git Branch Overview

| Branch | Last commit (date, author, subject) | Total commits |
|---|---|---|
| `main` | 2026-02-18 13:44 +0100 · Hamidhrf · `Merge Phase 1 v3: 38 experiments, 220 pod traces` | 24 |
| `phase1-v3-archive` | 2026-02-18 13:31 +0100 · Hamidhrf · `BACKUP: All Phase 1 v3 work before cleanup` | 23 |
| `phase2` | 2026-01-31 19:24 +0100 · Hamidhrf · `Phase 2: Complete model comparison - VAE, LSTM-AE, TimeGAN results` | 22 |
| `phase4-model-training` | 2026-04-20 15:01 +0200 · Hamidhrf · `feat: add thesis figures and brief` | 50 |

**Main/current development branch:** `phase4-model-training` — it has the most commits, the most recent commit, and 1,399 tracked files vs. ~145–154 in the others (so it contains all the Phase 4 TimeGAN code, models, results, figures, and thesis draft). `main` has been frozen since Phase 1 v3 was merged; `phase2` is a branch-point snapshot from Jan 31; `phase1-v3-archive` is an explicit backup tag-as-branch of Phase 1 v3 state.

All four branches were present locally and on `origin`.

---

## Section 2: Directory Structure

### 2.1 Top-level contents per branch (tracked files only)

Diffs computed against `git ls-tree <branch> --name-only`:

**`main` ↔ `phase1-v3-archive`:** identical top-level tree.

**`main` ↔ `phase2`:** `phase2` adds `data/`, and lacks `run_single_experiment_v2.py`.

**`main` ↔ `phase4-model-training`:** `phase4-model-training` replaces the top-level `JOURNAL.md` with a `docs/JOURNAL.md`, removes `run_single_experiment_v2.py`, and adds `.gitignore.bak`, `data/`, `figures/`, `models/`, `notebooks/`, `reports/`.

Top-level layout (tracked + untracked) on `phase4-model-training`:
```
.claude/                  current_vm_config.yaml    environment.yml
.gitignore                dashboards/               figures/
.gitignore.bak            data/                     k8s/
README.md                 docs/                     models/
notebooks/                outputs/                  reports/
scripts/                  tools/
```

### 2.2 Tree to depth 3 for `phase4-model-training` (working tree)

```
generative-ai-workload-modeling
├── current_vm_config.yaml
├── dashboards/ (6 files incl. README + 5 Grafana JSONs)
├── data/
│   ├── processed/
│   │   ├── phase1_v1/
│   │   ├── phase1_v3/
│   │   └── phase4/          (raw / diff / zeromean / windows / diff_windows / unified + preprocessing_summary.json)
│   └── raw/
│       ├── phase1/
│       ├── phase1_v2/
│       └── phase1_v3/
├── docs/ (14 markdown files — see Section 5)
├── environment.yml
├── figures/ (12 thesis PDFs)
├── k8s/
│   ├── base/
│   ├── gpu/
│   ├── monitoring/ (prometheus, grafana, kepler, dcgm, node-exporter, kube-state-metrics)
│   └── workloads/ (bert, gpt2, resnet152, whisper, yolo deployment YAMLs)
├── models/
│   └── phase4/ (lstm/, timevae/, timegan_s1..s39 excl. s13/s30)
├── notebooks/            (empty)
├── outputs/
│   ├── conditional_timegan/            phase2 era
│   ├── data_analysis/
│   ├── final_conditional_timegan/
│   ├── improved_conditional_vae/
│   ├── model_comparison/
│   ├── phase2_baseline/
│   ├── phase2_eda/
│   ├── phase4/                         (lstm, timevae, timegan_s* dirs, comprehensive_eval, comparison_plots, postprocessed, quality_evaluation_s27, validation)
│   ├── temporal_vae_extreme/
│   ├── timegan/                        phase2 era
│   ├── timegan_windowed/
│   └── timegan_workload_specific/
├── README.md
├── reports/
│   └── phase4_eda_10m/
├── scripts/
│   ├── cluster-creation/
│   ├── gpu-setup/
│   ├── models/phase4/                  (empty dir)
│   ├── monitoring/
│   ├── phase2/                         (baseline, eda, model_comparison.py, temporalvae, timegan, timevae, utils)
│   ├── phase4/                         (augment, comparison, evaluation, generate_s27, lstm, normalized_scaling, postprocess_s36, preprocess, thesis_figures, timegan, timevae, validate_s36)
│   ├── utils/boundary_smoothing.py
│   └── workloads/                      (bert_base, gpt2, resnet152, whisper, yolo — each with inference_variable.py)
└── tools/
```

The two archival branches (`main`, `phase1-v3-archive`) share the pre-Phase-4 subset of the above (no `figures/`, no `models/`, no `data/processed/phase4/`, no `scripts/phase4/`, no `scripts/phase2/phase4/`, no `reports/`). The `phase2` branch sits between those and the Phase 4 tree (it already has `data/` but no Phase 4 scripts or outputs).

---

## Section 3: Code Files (Python / Shell)

118 files total on `phase4-model-training` (all `.py` + `.sh`).  Size in KB (rounded).

### 3.1 `tools/`

| Path | KB | Description (from header) |
|---|---|---|
| `tools/analyze_experiments_v3.py` | 9.3 | Phase 1 v3 Experiment Analysis — statistical analysis of collected metrics |
| `tools/build_all.sh` | 2.3 | Build and push all Phase 1 v3 Docker images |
| `tools/clear_system_cache.sh` | 1.5 | System cache cleanup before experiments |
| `tools/pre_experiment_checklist.sh` | 1.9 | Phase 1 pre-experiment checklist |
| `tools/quick_backup.sh` | 1.6 | Quick thesis backup script |
| `tools/run_experiment_v3.py` | 22 | Phase 1 v3 experiment runner with interactive UX |
| `tools/visualize_experiment_v3.py` | 5.0 | Quick visualization of Phase 1 v3 data |

### 3.2 `scripts/cluster-creation/`, `scripts/gpu-setup/`, `scripts/monitoring/`, `scripts/workloads/`, `scripts/utils/`

| Path | KB | Description |
|---|---|---|
| `scripts/cluster-creation/cluster_recovery.sh` | 2.8 | Kubernetes Cluster Recovery Script |
| `scripts/cluster-creation/delete_cluster.sh` | 5.6 | Complete Kubernetes Cluster Cleanup |
| `scripts/cluster-creation/setup_cluster.sh` | 25 | Cluster setup (versioned, CRI-O notes) |
| `scripts/gpu-setup/gpu_health_check.sh` | 6.4 | GPU Health Check & Auto-Repair (CRI-O) |
| `scripts/monitoring/deploy-monitoring-stack.sh` | 4.5 | Deploy Prometheus / Grafana / exporters |
| `scripts/monitoring/enable_pod_level_metrics.sh` | 3.8 | Enable pod-level metrics |
| `scripts/monitoring/import-dashboards.sh` | 4.3 | Import Grafana dashboards |
| `scripts/utils/boundary_smoothing.py` | 13 | Boundary smoothing post-processing utility |
| `scripts/workloads/bert_base/inference_variable.py` | 7.2 | BERT-base variable load profile inference |
| `scripts/workloads/gpt2/inference_variable.py` | 7.0 | GPT-2 variable load profile inference |
| `scripts/workloads/resnet152/inference_variable.py` | 4.8 | ResNet152 variable load profile inference |
| `scripts/workloads/whisper/inference_variable.py` | 6.6 | Whisper variable load profile inference |
| `scripts/workloads/yolo/inference_variable.py` | 4.9 | YOLOv8 variable load profile inference |

### 3.3 `scripts/phase2/` (legacy Phase 2 model zoo)

| Path | KB | Description |
|---|---|---|
| `scripts/phase2/baseline/lstm_baseline.py` | 20 | **LSTM Baseline Model for Pod-Level Trace Generation** (Phase 2) |
| `scripts/phase2/eda/phase2_eda_podlevel.py` | 18 | Phase 2 EDA of 60 pod traces |
| `scripts/phase2/model_comparison.py` | 23 | Comprehensive Model Comparison for trace generation |
| `scripts/phase2/temporalvae/hyperparameter_search.py` | 8.6 | Hyperparameter grid search for Temporal VAE (beta, KL warmup) |
| `scripts/phase2/temporalvae/temporal_vae_improved.py` | 30 | Improved Temporal VAE (Conv1D + LSTM, posterior-collapse fix) |
| `scripts/phase2/timegan/conditional_timegan.py` | 27 | Conditional TimeGAN w/ replica count conditioning |
| `scripts/phase2/timegan/final_conditional_timegan.py` | 25 | FINAL workload-specific conditional windowed TimeGAN |
| `scripts/phase2/timegan/timegan_windowed.py` | 21 | Sliding window TimeGAN |
| `scripts/phase2/timegan/timegan_workload.py` | 27 | TimeGAN (SeriesGAN-inspired stabilization) |
| `scripts/phase2/timegan/timegan_workload_specific.py` | 18 | Workload-specific TimeGAN |
| `scripts/phase2/timevae/data_handler.py` | 11 | TimeVAE data handler (absolute normalization) |
| `scripts/phase2/timevae/evaluate_timevae.py` | 16 | TimeVAE evaluation / generation |
| `scripts/phase2/timevae/improved_conditional_timevae.py` | 26 | Improved conditional VAE for all workload types |
| `scripts/phase2/timevae/timevae_architecture.py` | 12 | TimeVAE architecture spec |
| `scripts/phase2/timevae/train_timevae.py` | 15 | TimeVAE training |
| `scripts/phase2/utils/absolute_normalizer.py` | 11 | Absolute normalization (vs. system-relative) |
| `scripts/phase2/utils/inspect_csv_headers.py` | 3.9 | Inspect CSV headers across experiments |
| `scripts/phase2/utils/phase1_data_loader.py` | 12 | Phase 1 pod-level trace loader |
| `scripts/phase2/utils/test_data_loader.py` | 8.4 | Validation test for phase1 data loader |
| `scripts/phase2/utils/vm_config_handler.py` | 6.5 | VM hardware config handler for conditioning |

### 3.4 `scripts/phase4/` (the Phase-4 training code)

Top-level Phase 4 scripts (flagged if they match priority keywords: `preprocess`, `generate`, `postprocess`, `validate`):

| Path | KB | Description | Priority |
|---|---|---|---|
| `scripts/phase4/preprocess_phase4.py` | 22 | Phase 4 preprocessing — three transformation methods (raw, diff, zeromean + windows variants) | **preprocess** |
| `scripts/phase4/augment_dataset_s35.py` | 9.1 | S35 data augmentation script | train-adjacent |
| `scripts/phase4/generate_s27_traces.py` | 12 | S27 trace generator w/ intelligent reconstruction | **generate** |
| `scripts/phase4/postprocess_s36.py` | 26 | S36 post-processing pipeline (cosine blend + adaptive filter + memory reconstruction) | **postprocess** |
| `scripts/phase4/validate_s36.py` | 24 | S36 validation suite (vs. real data + extrapolation) | **validate** |
| `scripts/phase4/normalized_scaling.py` | 3.1 | Normalized scaling plot helper |

#### `scripts/phase4/timegan/` — 38 training scripts (S13 and final "s30" missing an outputs/models dir)

| Path | KB | Stage title |
|---|---|---|
| `timegan_s1.py` | 37 | Stage 1 — Minimal GAN |
| `timegan_s2.py` | 39 | Stage 2 — Stabilization |
| `timegan_s3.py` | 43 | Stage 3 — Richer generator loss (FM + moment matching) |
| `timegan_s4.py` | 44 | Stage 4 — WGAN-GP |
| `timegan_s5.py` | 55 | Ablation full redo S1–S7 |
| `timegan_s6.py` | 42 | S6 — Discriminator warmup |
| `timegan_s7.py` | 43 | S7 — Reduced D capacity + warmup |
| `timegan_s8.py` | 43 | S8 — Noise injection + data aug + fixed epoch budget |
| `timegan_s9.py` | 50 | S9 — Per-metric variance control + fixed aug |
| `timegan_s10.py` | 52 | S10 — Adaptive per-metric variance cap + grad clip |
| `timegan_s11.py` | 49 | S11 — Conditional discriminator + variance regression |
| `timegan_s12.py` | 53 | S12 — Per-replica feature matching + stronger recon |
| `timegan_s14.py` | 48 | **S14 — Encoder-Free Generator (GeneratorB)** — "encoder-removal breakthrough" |
| `timegan_s15.py` | 46 | S15 — Remove zero-mean, raw normalized targets |
| `timegan_s16.py` | 45 | S16 — Per-phase statistical FM |
| `timegan_s17.py` | 46 | S17 — D gradient clipping + stronger phase FM |
| `timegan_s18.py` | 48 | S18 — Per-workload FM/VarReg tuning + Whisper stat ckpt |
| `timegan_s19.py` | 48 | S19 — YOLO reverted + Whisper best-W ckpt |
| `timegan_s20.py` | 54 | S20 — Segment-based generation |
| `timegan_s21.py` | 56 | S21 — Segment-based + precomputed FM targets |
| `timegan_s22.py` | 56 | S22 — Segment-based tuned |
| `timegan_s23.py` | 57 | S23 — (docstring mislabelled "S21") segment-based + FM |
| `timegan_s24.py` | 47 | S24 — Segment + autocorrelation loss |
| `timegan_s25.py` | 48 | S25 — Fixed autocorrelation loss |
| `timegan_s26.py` | 49 | S26 — Correctly normalised autocorr loss |
| `timegan_s27.py` | 41 | **S27** — S21 architecture + pod memory + GPU power |
| `timegan_s28.py` | 43 | S28 — Complete training (all 8 meaningful metrics) |
| `timegan_s29.py` | 28 | S29 — Unified cross-workload model |
| `timegan_s30.py` | 20 | S30 — Unified GPU-bound only (excludes Whisper) — **no output/model dir** |
| `timegan_s31.py` | 17 | S31 — Continuity-aware for GPU-bound |
| `timegan_s32.py` | 28 | S32 — GPU-bound unified + boundary smoothing |
| `timegan_s33.py` | 20 | S33 — Per-workload (GPU-bound only) |
| `timegan_s34.py` | 20 | S34 — S33 + Whisper (all 5 workloads) |
| `timegan_s35.py` | 20 | S35 — S34 + 4× data augmentation |
| `timegan_s36.py` | 21 | **S36 — S34 + Spectral Normalization** (final model) |
| `timegan_s37.py` | 21 | S37 — S36 + discriminator dropout (p=0.3) |
| `timegan_s38.py` | 22 | S38 — S36 + multi-layer feature matching |
| `timegan_s39.py` | 21 | S39 — S36 + targeted variance tuning |

#### `scripts/phase4/timevae/` — **timevae** flagged

| Path | KB | Description |
|---|---|---|
| `timevae_v1.py` | 39 | Phase 4 TimeVAE conditional VAE (v1) |
| `timevae_v2.py` | 42 | Phase 4 TimeVAE v2 (two targeted fixes over v1) |
| `timevae_v3.py` | 36 | Phase 4 TimeVAE v3 (fully-connected decoder) |

#### `scripts/phase4/lstm/` — **lstm** flagged

| Path | KB | Description |
|---|---|---|
| `validate_step4_phase_lstm_v6.py` | 51 | **Phase 4 Step 4: Phase-Conditioned LSTM (v6, matched-r plots, revised metric set)** — contains the trained LSTM baseline |

#### `scripts/phase4/evaluation/` — **evaluate/metrics** flagged

| Path | KB | Description |
|---|---|---|
| `comprehensive_eval_s27_s33_s34.py` | 20 | Comprehensive S27 vs S33 vs S34 eval (ACF, cross-corr, distribution) |
| `comprehensive_eval_s27_vs_s33.py` | 19 | Comprehensive S27 vs S33 eval |
| `eval_s21.py` | 18 | S21 vs real evaluation plots |
| `eval_s32.py` | 16 | S32 — GPU-bound + boundary smoothing assessment |
| `eval_s33.py` | 12 | S33 evaluation with boundary smoothing |
| `eval_s34.py` | 13 | S34 evaluation with boundary smoothing |
| `eval_s35.py` | 15 | S35 vs S34 comparison |
| `eval_s36.py` | 16 | S36 vs S34 comparison |
| `eval_s37.py` | 16 | S37 vs S36 comparison |
| `eval_s38.py` | 16 | S38 vs S36 comparison |
| `eval_s39.py` | 16 | S39 vs S36 comparison |
| `evaluate_s27_quality.py` | 25 | S27 quality evaluation (multi-plot) |
| `evaluate_s29.py` | 8.5 | S29 per-workload quality assessment |
| `evaluate_s31.py` | 15 | S31 GPU-bound assessment vs S29/S27 |

#### `scripts/phase4/comparison/`

| Path | KB | Description |
|---|---|---|
| `compare_s27_smoothing.py` | 17 | S27 smoothed vs unsmoothed comparison |
| `comparison_plots_s21_vs_real.py` | 14 | Final production plots for S21 |
| `comparison_plots_s27.py` | 10 | S27 comparison plots |
| `comparison_plots_s28.py` | 9.0 | S28 comparison plots |
| `comparison_plots_s29.py` | 7.6 | S29 unified comparison |
| `comparison_plots_s31.py` | 13 | S31 continuity plots |
| `comparison_plots_s33.py` | 19 | S33 real vs raw vs smoothed |
| `comparison_plots_s35.py` | 11 | S35 comparison |
| `comparison_plots_s36.py` | 11 | S36 comparison |

#### `scripts/phase4/thesis_figures/`

| Path | KB | Description |
|---|---|---|
| `fig_extrapolation.py` | 9.5 | Standalone extrapolation curves figure |
| `fig_load_pattern.py` | 2.7 | Load-pattern diagram |
| `fig_real_vs_synthetic.py` | 5.2 | Real vs synthetic at r=5 |
| `fig_scaling_curves.py` | 6.6 | Scaling curves figure |
| `fig_system_architecture.py` | 5.3 | System architecture diagram |
| `fig_timegan_architecture.py` | 5.7 | TimeGAN S36 architecture diagram |
| `generate_thesis_figures.py` | 45 | Generates all 9 figures + 1 LaTeX table |

---

## Section 4: Result Files and Logs

**Totals (tracked + untracked, excl. `.git/`):** 2,354 files, ~899 MB.

By extension:

| ext | files | bytes |
|---|---|---|
| csv | 1,467 | 395 MB |
| pt  |   380 | 465 MB |
| json|   454 | 6.2 MB |
| npz |    32 | 68 MB |
| npy |    20 | 8.6 MB |
| pkl |     1 | 674 B |

By top-level directory:

| dir | files | MB |
|---|---|---|
| `data/`       | 1,528 | 448.8 |
| `models/`     |   725 | 424.2 |
| `outputs/`    |    92 |  26.0 |
| `dashboards/` |     5 |  <1 |
| `reports/`    |     3 |  <1 |
| `.claude/`    |     1 |  <1 |

Raw CSV trace data dominates `data/raw/phase1*` (about 400 MB of CSVs per replica per metric).

### 4.1 Phase-4 model checkpoint directories (`models/phase4/`)

All TimeGAN stage dirs contain one `generator.pt` per `{stage}_{workload}_vr{X}_fm{Y}/` subdir. Typical per-stage directory size: 6–10 MB (5 workloads × single generator.pt). Exceptions:
- `timegan_s1`..`timegan_s6`, `timegan_s8`, `timegan_s9` — 10–20 MB each (older, larger checkpoints with more saved state).
- `timegan_s5` — **75 MB** (largest; full ablation redo, multiple sub-runs).
- `timevae/` — **45 MB** across `timevae_v1/`, `timevae_v2/`, `timevae_v3/`, each with 5 per-workload subdirs containing `config.json`, `model.pt`, `posthoc_tables.json`.
- `lstm/step4_phase/{bert,gpt2,resnet152,whisper,yolo}/` — 9.4 MB; each has `config.json`, `model.pt`, `posthoc_tables.json`.
- `timegan_s36/` — 28 KB on disk (generator.pt only), subdirs:
  - `s36_bert_vr05_fm15/generator.pt`
  - `s36_gpt2_vr03_fm20/generator.pt`
  - `s36_resnet152_vr04_fm10/generator.pt`
  - `s36_whisper_vr05_fm05/generator.pt`
  - `s36_yolo_vr03_fm12/generator.pt`

The `vrXX_fmYY` suffix encodes per-workload `lambda_var_reg`/`lambda_fm_stat` (e.g. `vr05_fm15` = 0.5 / 1.5), confirmed by reading `scripts/phase4/timegan/timegan_s36.py:73-104`.

### 4.2 Phase-4 result JSONs in `outputs/phase4/`

All JSON files (depth ≤ 3):

| Path | Size |
|---|---|
| `comprehensive_eval/comprehensive_results.json` | 11 entries (list). Keys: `model, workload, metric_names, vr_mean, vr_per_metric, acf_real/synth/error, corr_real/synth/error, kl_divs, kl_mean, wass_dists, wass_mean, boundary_disc, real_mean, synth_mean, real_std, synth_std` |
| `comprehensive_eval/comprehensive_results_s27_s33_s34.json` | 12 entries (list), same schema |
| `lstm/step4_phase/step4_results.json` | 5 entries (one per workload) — per-workload `var_ratio_mean`, `var_ratio_per_metric`, `autocorr_diff`, `phase_jump_ratio`, `real_jump_ratio`, `syn_jump_ratio`, `metrics_generated`, `metrics_posthoc`, `n_epochs`, `elapsed_s`, `posthoc_tables` |
| `timegan_s1/timegan_s1_results.json` … `timegan_s34/...` | per-stage results.json arrays |
| `timegan_s27/s27_seg_vr03_fm10_ae150/results.json` | |
| `timegan_s29/s29_unified_vr04_fm12_ae150/results.json` | |
| `timegan_s29/.../s29_vs_s27_comparison.json` | |
| `timegan_s31/s31_gpu_bound_cont03/{config.json,results.json}` | |
| `timegan_s32/s32_gpu_bound_smooth02/results.json` | |
| `timegan_s33/{s33_eval_results.json, s33_summary.json}` | |
| `timegan_s34/{s34_eval_results.json, s34_summary.json}` | |
| `timegan_s35/{s35_eval_results.json, s35_summary.json}` | |
| `timegan_s36/{s36_eval_results.json, s36_summary.json}` | `s36_eval_results.json` top-level keys: `window_size, smoothing_enabled, dataset_size, improvement, workloads{bert,gpt2,resnet152,whisper,yolo}, summary{mean_vr_smooth, pass_count, total}` |
| `timegan_s37/{s37_eval_results.json, s37_summary.json}` | |
| `timegan_s38/s38_summary.json` | **no `s38_eval_results.json`** |
| `timegan_s39/s39_eval_results.json` | **no `s39_summary.json`** |
| `timevae/timevae_v1/timevae_results.json` | 5 entries, per-workload VR + posthoc |
| `timevae/timevae_v3/timevae_v3_results.json` | 5 entries (timevae_v2 has no results JSON — only `plots/`) |
| `validation/s36/validation_report.json` | 67 KB. Keys: `model="S36 (spectral normalization)", post_processing, workloads{bert, gpt2, resnet152, whisper, yolo}`. Per-workload: `validated_replicas, extrapolation_replicas, mean_wasserstein_all` |

### 4.3 Key small-JSON top-level keys

`data/processed/phase4/preprocessing_summary.json` — list of 25 entries (5 workloads × 5 methods `{raw, diff, zeromean, windows, diff_windows}`). Each entry: `workload, method, n_samples, n_train, n_val, trace_shape, replica_distribution`. All raw/diff/zeromean variants use `n_train=49, n_val=6` (90/10 split per workload; 220 pod traces total).

`outputs/phase4/timegan_s36/s36_eval_results.json` — 5 workloads, each with `vr_raw`, `vr_smooth`, `pass_raw`, `pass_smooth`, `vr_per_metric_smooth` (7 values), `metric_names` (`pod_cpu_usage, pod_memory_bytes, pod_psi_cpu, pod_latency_avg, pod_throughput, gpu_utilization, gpu_power_watts`). Summary mean_vr_smooth = **1.1159913**.

`outputs/phase4/timegan_s37/s37_eval_results.json` — same schema. Summary mean_vr_smooth = **1.1243255**. `pass_count = 5/5`.

`outputs/phase4/timegan_s38/s38_summary.json` — training-only summary (no post-eval JSON) with `stage, description, workloads, improvement="multi_layer_feature_matching", baseline="s36", results{bert, gpt2, resnet152, whisper, yolo}` where each entry has `best_val, elapsed_s, model_path`.

`outputs/phase4/timegan_s39/s39_eval_results.json` — per-workload S39 evaluation vs S36 (3456 B).

`outputs/phase4/validation/s36/validation_report.json` — per-workload `mean_wasserstein_all`:
- bert 1.543307, gpt2 4.73626, resnet152 2.972559, whisper 5.514604, yolo 1.197167.

`models/phase4/lstm/step4_phase/bert/config.json` (confirms LSTM baseline architecture): `hidden_dim=128, num_layers=2, latent_dim=64, regime_dim=32, pod_dim=32, replica_embed_dim=8, phase_embed_dim=4, dropout=0.1, batch_size=16, epochs=150, learning_rate=0.001, weight_decay=1e-5, patience=20, grad_clip=1.0, seed=42, phase_boundaries=[0,96,180,300,420,600], seq_len=715`.

### 4.4 Postprocessed trace arrays (`outputs/phase4/postprocessed/s36/`)

15 `.npy` files: `{bert,gpt2,resnet152,whisper,yolo}_r{1,5,10}_s36_postprocessed.npy` — output of the S36 post-processing pipeline at replica counts 1, 5, 10.

### 4.5 Processed datasets (`data/processed/phase4/`)

- `raw/{workload}_traces.npz` × 5 + `{workload}_normalization.json` × 5 (per-workload 55-trace dataset).
- `diff/`, `zeromean/`, `windows/`, `diff_windows/` variants of the same.
- `unified/combined_dataset.npz` (16 MB), `unified/combined_dataset_s35_augmented.npz` (23 MB), `unified/combined_normalization.json`.
- `preprocessing_summary.json` (see above).

### 4.6 Phase-2 era outputs (pre-Phase-4)

`outputs/{timegan,timegan_windowed,timegan_workload_specific,conditional_timegan,final_conditional_timegan,improved_conditional_vae,model_comparison,phase2_baseline,phase2_eda,temporal_vae_extreme}/` — each holds older `best_model.pt`, `all_results.json`, `training_curves.png`, etc. Most relevant for baseline claims: `outputs/phase2_baseline/{lstm_baseline_model.pt, metrics.json, train_losses.npy, val_losses.npy, baseline_summary.txt, training_curves.png, sample_predictions.png}`.

### 4.7 Reports (`reports/phase4_eda_10m/`)

`dataset_summary.csv`, `eda_summary.txt`, `normalization_check.csv`, `normalization_params.txt`, `quality_checks.csv`, `split_coverage.txt` (shows the Phase 4 train/val split: **49 train / 6 val per workload**, replicas r=1..10).

CSV headers (first-row only, truncated):
- `reports/phase4_eda_10m/dataset_summary.csv` — (not read; small file)
- `reports/phase4_eda_10m/quality_checks.csv` — (not read; small file)
- `reports/phase4_eda_10m/normalization_check.csv` — (not read; small file)

### 4.8 Dashboards JSONs (`dashboards/*.json`)

Grafana dashboard defs: `container-metrics.json`, `gpu-performance.json`, `inference-performance.json`, `system-pressure.json`, `system-resources.json` — each under 10 KB, not result data.

---

## Section 5: Documentation Files

All tracked `.md`, `.txt`, `.tex`, `.pdf` files on `phase4-model-training` (excluding data timestamp `.txt` stubs under `data/raw/`):

### 5.1 `docs/` and root README

| Path | Size | mtime | First 3 lines |
|---|---|---|---|
| `docs/JOURNAL.md` | 198 KB | 2026-02-24 | `# Technical Journal: Kubernetes Cluster Setup for Generative AI Workload Modeling` / (blank) / `**Research Context**: Master's Thesis ... Fachhochschule Dortmund` — **journal (flag)** |
| `docs/phase1_experiment_guide.md` | 1.6 KB | 2026-01-06 | `# Phase 1 Experiment Execution Guide` / (blank) / `## Quick Start Commands` |
| `docs/phase2_model_selection_framework.md` | 20 KB | 2026-02-18 | `# Phase 2: Model Selection Framework ...` / `## Master's Thesis: ...` |
| `docs/phase2_quickstart.md` | 8.7 KB | 2026-02-18 | `# Phase 2 Quick-Start Guide` / `## Immediate Actions for This Week` |
| `docs/PHASE4B_HANDOFF.md` | 13 KB | 2026-02-23 | `# Phase 4b Handoff: LSTM Baseline Training` / (blank) / `**Date:** February 23, 2026` — **handoff (flag)** |
| `docs/PHASE4_COMPLETE_SUMMARY.md` | 35 KB | 2026-03-09 | `# Phase 4 Complete Summary: Generative Model Development` / `**Master's Thesis - Digital Transformation**` — **summary (flag)** |
| `docs/PHASE4_EXECUTIVE_SUMMARY.md` | 6.5 KB | 2026-03-09 | `# Phase 4 Executive Summary - Quick Reference` / (blank) / `**Project:** Generative AI Workload Modeling` — **summary (flag)** |
| `docs/PHASE4_JOURNAL_ENTRY.md` | 4.7 KB | 2026-03-24 | `# Journal Entry: Phase 4 Complete — Model Training and Validation` / `**Date:** March 24, 2026` — **journal (flag)** |
| `docs/PHASE4_TRAINING.md` | 16 KB | 2026-03-24 | `# Phase 4: TimeGAN Model Training — Complete Documentation` |
| `docs/PROJECT_DECISIONS.md` | 6.6 KB | 2026-02-18 | `## Decision #2: Normalization + VM Config Conditioning` / (blank) / `**Date**: January 23, 2026` |
| `docs/setup-guide.md` | 274 B | 2025-12-03 | `# Kubernetes Cluster Setup Guide` / (blank) / `See [JOURNAL.md](../JOURNAL.md) for complete technical documentation.` |
| `docs/THESIS_BRIEF.md` | 12 KB | 2026-04-20 | `# THESIS BRIEF v2.0` / `## Generative Modeling of Application Workloads for Synthetic Trace Generation` — **may be draft (flag: v2.0 label)** |
| `docs/THESIS_DRAFT_v1.md` | 122 KB | 2026-02-18 | `# Generative Modeling of Application Workloads for Synthetic Trace Generation` / `**Master's Thesis in Digital Transformation**` — **draft/v1 (flag)** |
| `docs/troubleshooting.md` | 243 B | 2025-12-03 | `# Troubleshooting Guide` / (blank) / `## Common Issues` |
| `README.md` | 7.1 KB | 2025-12-03 | `# Generative AI Workload Modeling` / (blank) / `**Research Thesis Project** - Digital Transformation` |

### 5.2 Other docs / text outputs

| Path | Size | mtime | First 3 lines |
|---|---|---|---|
| `dashboards/README.md` | 4.2 KB | 2025-12-09 | `# Grafana Dashboards for Data Collection` / (blank) / `This directory contains Grafana dashboard configurations ...` |
| `k8s/monitoring/README.md` | 8.4 KB | 2025-12-09 | `# Monitoring Stack for Thesis Data Collection` / (blank) / `Complete observability stack for collecting AI workload performance metrics.` |
| `scripts/gpu-setup/README.md` | 5.3 KB | 2025-12-03 | `# GPU Setup Guide for Kubernetes with CRI-O` / (blank) / `Complete guide to enable NVIDIA GPU support in Kubernetes cluster.` |
| `outputs/phase2_baseline/baseline_summary.txt` | 428 B | 2026-02-18 | `Date: 26.01.2026` / `Normalized Test MSE: 0.006197` / `Best Val Loss: 0.002613` |
| `outputs/phase2_eda/EDA_REPORT.md` | 3.6 KB | 2026-02-18 | `# Phase 2 EDA - Pod-Level Analysis Report` / `**Date**: January 23, 2026` / `**Dataset**: 60 Individual Pod Traces from Phase 1` |
| `reports/phase4_eda_10m/eda_summary.txt` | 429 B | 2026-02-23 | `======================================================================` / `PHASE 4: DATA VALIDATION (10 METRICS)` / `======================================================================` |
| `reports/phase4_eda_10m/normalization_params.txt` | 3.3 KB | 2026-02-23 | (blank) / `BERT:` / `======================================================================` |
| `reports/phase4_eda_10m/split_coverage.txt` | 3.1 KB | 2026-02-23 | (blank) / `BERT:` / `==================================================` — documents train/val coverage per replica count |
| `tools/requirements.txt` | 63 B | 2026-01-06 | `requests>=2.31.0` / `pandas>=2.0.0` / `numpy>=1.24.0` |

### 5.3 PDF figures (`figures/`)

All under `figures/` on `phase4-model-training`, mtimes 2026-04-16 to 2026-04-20:
`ablation_study.pdf` (34 KB), `encoder_removal.pdf` (44 KB), `extrapolation_curves.pdf` (42 KB), `load_pattern.pdf` (26 KB), `model_comparison.pdf` (47 KB), `postprocessing_before_after.pdf` (33 KB), `real_vs_synthetic_r5.pdf` (120 KB), `scaling_curves.pdf` (35 KB), `system_architecture.pdf` (41 KB), `timegan_architecture.pdf` (114 KB), `wasserstein_heatmap.pdf` (48 KB), `wasserstein_per_replica.pdf` (28 KB).

No `.tex` files anywhere in the tree. No other `.pdf` files.

### 5.4 Timestamp stub `.txt` files (low information)

~72 files under `data/raw/phase1*/*/` named `{workload}_r{N}_YYYYMMDD_HHMMSS_timestamps.txt`, each ~110–230 B, containing start/end timestamps for that experiment. Not flagged further.

---

## Section 6: Notable Files Found (Claim Verification Map)

1. **S36 final model per-workload VR and Wasserstein**
   - VR: `outputs/phase4/timegan_s36/s36_eval_results.json` (per-workload `vr_smooth`, `vr_raw`, `vr_per_metric_smooth`, and `summary.mean_vr_smooth=1.1159913`).
   - Wasserstein: `outputs/phase4/validation/s36/validation_report.json` (per-workload `mean_wasserstein_all`).
   - **Likely found in:** `outputs/phase4/timegan_s36/s36_eval_results.json` and `outputs/phase4/validation/s36/validation_report.json`.

2. **S38 ablation results ("Failed" label, actual VR numbers)**
   - Training summary only: `outputs/phase4/timegan_s38/s38_summary.json` — has `best_val` per workload but **no `s38_eval_results.json`** (unlike S36, S37). Thesis brief labels S38 "Failed" with mean VR 1.147 (`docs/THESIS_BRIEF.md:110`). Raw VR numbers are not obviously present as a committed evaluation JSON; verify via `scripts/phase4/evaluation/eval_s38.py` or regenerate.
   - **Likely found in:** `outputs/phase4/timegan_s38/s38_summary.json` (training stats) + `docs/THESIS_BRIEF.md` table + `scripts/phase4/evaluation/eval_s38.py`. **No `s38_eval_results.json` exists**, so the VR=1.147 claim cannot be cross-checked against a committed eval JSON.

3. **S37 dropout ablation mean VR (1.089 vs 1.124)**
   - `outputs/phase4/timegan_s37/s37_eval_results.json` → `summary.mean_vr_smooth = 1.1243254542` (≈1.124).
   - **Likely found in:** `outputs/phase4/timegan_s37/s37_eval_results.json`. The 1.089 figure is **not** in this JSON (summary value is 1.124).

4. **S14 encoder-removal breakthrough results**
   - Training script: `scripts/phase4/timegan/timegan_s14.py` (header: "TimeGAN S14 - Encoder-Free Generator (GeneratorB)").
   - Per-workload results: `outputs/phase4/timegan_s14/s14_vr03_fm05_ae150/results.json` (var_ratio_mean values: bert 0.718, gpt2 1.232, resnet152 0.992, whisper 1.654, yolo 1.422).
   - **Likely found in:** `outputs/phase4/timegan_s14/s14_vr03_fm05_ae150/results.json`.

5. **LSTM baseline architecture (layers, hidden size, training details)**
   - Phase-4 baseline (final): `scripts/phase4/lstm/validate_step4_phase_lstm_v6.py` + per-workload `models/phase4/lstm/step4_phase/{workload}/config.json` (hidden_dim=128, num_layers=2, latent_dim=64, regime_dim=32, pod_dim=32, replica_embed_dim=8, phase_embed_dim=4, dropout=0.1, batch_size=16, epochs=150, lr=0.001, weight_decay=1e-5, patience=20, grad_clip=1.0, seed=42). Aggregate VR results in `outputs/phase4/lstm/step4_phase/step4_results.json`.
   - Phase-2 baseline: `scripts/phase2/baseline/lstm_baseline.py` with `outputs/phase2_baseline/{lstm_baseline_model.pt, metrics.json, baseline_summary.txt}`.
   - **Likely found in:** `scripts/phase4/lstm/validate_step4_phase_lstm_v6.py`, `models/phase4/lstm/step4_phase/*/config.json`, `outputs/phase4/lstm/step4_phase/step4_results.json`.

6. **TimeVAE v1/v2/v3 per-workload VR values**
   - v1: `outputs/phase4/timevae/timevae_v1/timevae_results.json` — bert 0.388, gpt2 0.781, resnet152 0.557, whisper 0.703, yolo 0.635.
   - v2: **no results JSON committed** (`outputs/phase4/timevae/timevae_v2/` has only `plots/`). Training script is `scripts/phase4/timevae/timevae_v2.py`.
   - v3: `outputs/phase4/timevae/timevae_v3/timevae_v3_results.json` — bert 0.114, gpt2 0.254, resnet152 0.066, whisper 0.210, yolo 0.096.
   - **Likely found in:** `outputs/phase4/timevae/timevae_v1/timevae_results.json`, `outputs/phase4/timevae/timevae_v3/timevae_v3_results.json`. **v2 VR values are not committed as a JSON** — verify against `docs/PHASE4_COMPLETE_SUMMARY.md` table (TimeVAE v2 = 0.527 per that doc).

7. **Train/validation/test split**
   - Defined in `scripts/phase4/preprocess_phase4.py:337` via `train_test_split(indices, train_size=0.9, random_state=random_state)`. No separate held-out **test** split — only 49-train / 6-val per workload per method. Per-workload split coverage documented in `reports/phase4_eda_10m/split_coverage.txt` and summarized in `data/processed/phase4/preprocessing_summary.json`.
   - **Likely found in:** `scripts/phase4/preprocess_phase4.py` (lines 337–375) + `reports/phase4_eda_10m/split_coverage.txt` + `data/processed/phase4/preprocessing_summary.json`.

8. **Per-workload hyperparameter search (grid ranges for vr and fm weights)**
   - Final per-workload values hard-coded as the `S27_HYPERPARAMS` dict in `scripts/phase4/timegan/timegan_s36.py:73-104` (lambda_var_reg × lambda_fm_stat per workload; comment labels them "proven optimal"). The dict is also replicated in S27, S33–S39 scripts.
   - A grid search over (vr, fm) weights per workload is **not obviously present as a sweep JSON** on this branch. The directory naming of model runs (`s36_bert_vr05_fm15`, `s36_gpt2_vr03_fm20`, etc.) encodes the chosen values. There is a separate Phase-2 grid search at `scripts/phase2/temporalvae/hyperparameter_search.py` (over beta / KL warmup, not vr/fm). Also `scripts/phase4/evaluation/eval_s32.py` compares vr/fm variants.
   - **Likely found in:** `scripts/phase4/timegan/timegan_s36.py` (S27_HYPERPARAMS dict). A formal grid-search artifact is **not obviously present in project**.

9. **Adaptive filter window size selection per metric**
   - `scripts/phase4/postprocess_s36.py:76-95` — hard-coded dicts `METRIC_BEHAVIOR` (7 metrics labelled `smooth`/`noisy`/`flat`) and `WORKLOAD_OVERRIDES` (whisper: pod_throughput+gpu_power_watts = noisy; gpt2: gpu_utilization = smooth). `adaptive_filter` (line 271) uses `kernel_size = 15` (hard-coded) for all "smooth" metrics, with edge-aware padding.
   - Classification "from visual analysis of r=5 plots" per in-code comment (line 70).
   - **Likely found in:** `scripts/phase4/postprocess_s36.py` (lines 70–95, 270–305).

10. **Post-processing pipeline code**
    - `scripts/phase4/postprocess_s36.py` (26 KB) — cosine boundary blend (`cosine_blend_boundaries`, window=20), adaptive per-metric filter, memory reconstruction, dropped-metric reconstruction, plus a `GeneratorSeg` definition mirroring S36 training. Helper: `scripts/utils/boundary_smoothing.py`. Outputs: `outputs/phase4/postprocessed/s36/*.npy` (15 files) and the figure `figures/postprocessing_before_after.pdf`.
    - **Likely found in:** `scripts/phase4/postprocess_s36.py` (+ `scripts/utils/boundary_smoothing.py`).

---

## Section 7: Unexpected or Surprising Files

- **No `s38_eval_results.json`.** S36, S37, S34, S35 all have `{stage}_eval_results.json` alongside `{stage}_summary.json`, but S38 only has `s38_summary.json`. The thesis brief's S38 "Failed, VR=1.147" line therefore has no committed JSON backing within `outputs/phase4/`.
- **No `s39_summary.json`.** Mirror-image gap: S39 has `s39_eval_results.json` but no training-summary JSON.
- **`outputs/phase4/timegan_s36/s36_summary.json` references `models/phase4/timegan_s39/...` paths.** The S36 summary file's `results[*].model_path` fields point at S39 model directories — likely a copy-paste artifact; actual S36 checkpoints live under `models/phase4/timegan_s36/s36_*_vrXX_fmYY/generator.pt` as expected.
- **S13 is skipped everywhere** (no `timegan_s13.py`, no outputs, no models). S1→S12 then S14. S30 script exists (`timegan_s30.py`) but **has no outputs or model directory** — training appears to have never produced committed artifacts.
- **S23 has a copy-paste docstring** — opens with "TimeGAN S21 - Segment-Based Generation with Precomputed FM Targets" even though the filename is S23.
- **S18 and S19 output directories are named `s17_vr03_fm10_ae150/`** — old tag name, not `s18_...`/`s19_...`. Easy to mis-read.
- **`timevae_v2/` has no `timevae_v2_results.json`** — only a `plots/` dir, unlike v1/v3.
- **`notebooks/` is an empty tracked dir** on the working tree (contains no files).
- **`.gitignore.bak`** is a committed backup of the old `.gitignore` (1.1 KB, mtime 2026-03-24).
- **Phase-2 era outputs still present on `phase4-model-training`:** `outputs/{timegan, timegan_windowed, conditional_timegan, final_conditional_timegan, temporal_vae_extreme, …}/`. Multiple `best_model.pt` files (5–7 MB each) would clutter a released snapshot.
- **`scripts/models/phase4/` is an empty directory** (distinct from the populated `models/phase4/`).
- **Massive `.npz` augmented dataset:** `data/processed/phase4/unified/combined_dataset_s35_augmented.npz` (23 MB) is tracked alongside the smaller 16 MB `combined_dataset.npz` — S35 augmentation was kept committed even though the S35 stage is labelled "Failed" in the ablation table.
- **`docs/JOURNAL.md` (198 KB)** is by far the largest doc file and may contain outdated Phase-1/2 era decisions; `docs/THESIS_DRAFT_v1.md` (122 KB) is similarly dated 2026-02-18 (before the S30+ stages) and is labelled `_v1`.
- **`dashboards/`, `k8s/`, `scripts/cluster-creation/`, `scripts/gpu-setup/`, `scripts/monitoring/`, `scripts/workloads/`, `tools/run_experiment_v3.py`** are all Phase 1 data-collection infrastructure, carried forward on `phase4-model-training` though not directly relevant to the thesis's generative modelling claims.

---

_End of inventory._
