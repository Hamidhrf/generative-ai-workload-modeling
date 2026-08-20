# S36 Tier 2 Extension Reference

Standalone reference for the H100 Tier 2 (MIG all-1g.12gb) extension
of the S36 workload trace generation work. Mirrors the structure of
`S36_TIER3_EXTENSION_REFERENCE.md`. Written to be self-contained
enough that the extension paper's Tier 2 sections can be written from
this document alone.

## 1. Motivation and scope

Third and final tier of the S36 cross-tier extension. Asks whether
the frozen thesis recipe transfers to MIG hard-partitioned GPU
sharing on H100, following the same apples-to-apples discipline
established for Tier 3: the training recipe (architecture,
hyperparameters, seed, epoch counts, loss functions) is byte-frozen
across tiers; only paths retarget and, this tier only, the source
column for `gpu_utilization` changes from aggregated to per-slice.
The ablation is a clearly-separated second stage, not a silent
change to the frozen baseline. A16 numbers throughout this document
are the thesis's reported figures; Tier 3 numbers are from
`S36_TIER3_EXTENSION_REFERENCE.md`; Tier 2 numbers are from this
extension's runs.

The methodological headline for Tier 2 is that this is the first
tier of the study where per-pod GPU attribution is natively
available. Under GPU time-slicing (A16 Phase 1 v3 and Tier 3),
whole-GPU DCGM counters must be divided by replica count to impute a
per-pod value — an assumption that every pod uses an equal share of
the physical GPU at each timestep. Under MIG all-1g.12gb, each pod
runs on a dedicated 1g.12gb slice, and DCGM per-instance labels
attach the pod name directly to per-slice metrics. Per-pod GPU
utilization becomes a direct measurement rather than an inference,
and cross-pod variance in that measurement becomes a physical signal
the model can learn from.

The choice to use the per-slice source column for training (rather
than the aggregated file's sum-across-slices, divided by MIG's fixed
7-slice count for scale comparability) is central to Tier 2's
contribution and is defended in Section 2. The three-way A16 / Tier
3 / Tier 2 comparison is the primary paper artifact.

## 2. Dataset

Tier 2 collected 35 experiments (5 workloads x r=1..7) on the H100
devLab with GPU Operator v26.3.3 in MIG single-strategy mode
(`nvidia.com/mig.config=all-1g.12gb`, 7 x 1g.12gb slices). Each
experiment ran the same 60-minute Business Day 6-phase load profile
as A16 Phase 1 v3 and Tier 3, with Prometheus scraping every 5
seconds. Zero collection failures; whisper r=7 completed cleanly
despite CPU contention (16 vCPUs shared across 7 pods). Full
collection details in `TIER2_NOTES.md`.

Total per-workload pod count is 28 (1+2+...+7), versus Tier 3's 55
(1+2+...+10). Total training-dataset pod count is 140 versus Tier
3's 275. Sequence length is the same 715 timesteps after
preprocessing (Tier 2's raw traces are 720 ticks; truncated to 715
for compatibility with the frozen recipe's segment structure — the
5 discarded timesteps are the tail of Phase 5 cooldown).

**Cross-tier dataset landscape.**

| Tier | Cluster | GPU sharing | r range | Traces / workload | Per-pod GPU |
|---|---|---|---|---|---|
| A16 Phase 1 v3 | A16 VM | time-slicing 10 | 1..10 | 55 | inferred (whole / r) |
| Tier 3 H100 | H100 devLab | time-slicing 10 | 1..10 | 55 | inferred (whole / r) |
| Tier 2 H100 | H100 devLab | MIG 1g.12gb x 7 | 1..7 | 28 | direct per-slice |

The replica-count embedding trained on A16's r=1..10 stays inside
its trained range for Tier 2's r=1..7, so no embedding-extrapolation
concern.

### 2.1 Source-column choice for `gpu_utilization`

Tier 2's collector writes two DCGM engine-active files per
experiment: aggregated (`<workload>_r<n>_gpu_utilization_*.csv`) and
per-slice (`<workload>_r<n>_gpu_utilization_per_slice_*.csv`). The
aggregated file sums across active slices client-side and scales by
100, producing 0..700 values at r=7 with all slices busy. The
per-slice file exposes each active MIG instance's engine-active
percentage as a separate row per (timestamp, pod), preserving pod
identity via the DCGM `pod` label. Both are honest for their
intended questions — aggregated for whole-experiment activity,
per-slice for per-pod measurement.

Training used the per-slice source. Alternative approaches (dividing
the aggregated file by 7 for scale comparability with A16's 0..100
range) were rejected because dividing sums-across-slices by a fixed
denominator produces a value that is identical across all pods at
each timestep by construction. That is the same equal-share
imputation as time-slicing's `system_value / replica_count`, just
with a bigger denominator. Applying that transformation would
manually destroy the per-pod variance that MIG hardware makes
available — the exact signal the tier was designed to capture. Fresh
Tier 2 normalization is fit on the training partition, so
scale-comparability with A16 is handled through per-workload min-max
scaling rather than through source-column massaging. This choice is
central to the paper's Section 4 claim that per-slice attribution
reveals distinct sources of per-pod variance that time-sliced tiers'
imputation destroys.

The aggregated file is retained for a Section 4.6 diagnostic that
quantifies precisely what the /7 transformation would have
destroyed. See that section for numbers.

### 2.2 Raw per-metric mean audit (BERT reference)

Three-way audit on BERT, all 10 metrics, denormalized units, means
computed across the full trace and averaged across all pods in the
respective r-sweep:

| Metric | A16 | Tier 2 | Tier 3 |
|---|---|---|---|
| pod_cpu_usage | 0.0962 | 0.0640 | 0.0679 |
| pod_memory_bytes | 8.36e8 | 1.12e9 | 1.04e9 |
| pod_psi_cpu | 0.0001 | 0.0001 | 0.0001 |
| pod_latency_avg | 0.0083 | 0.0049 | 0.0057 |
| pod_throughput | 2.3045 | 2.3934 | 2.3705 |
| gpu_utilization | 2.6959 | 1.8374 | 0.8952 |
| gpu_memory_used | 556.02 | 698.00 | 1186.39 |
| gpu_memory_total | 2720.31 | 2751.75 | 17330.64 |
| gpu_power_watts | 7.26 | 24.38 | 23.14 |
| gpu_temperature | 12.26 | 12.52 | 10.28 |

Note on `gpu_memory_total`: this metric is divided by replica count
in `create_pod_traces` uniformly across all three tiers, so the
reported values are total-VRAM/replica averaged over the r-sweep,
not raw hardware capacity. The affected metric is in
`DROP_PER_WORKLOAD` and is not trained on; the post-processing stage
reconstructs it statistically from real per-workload data. Values
are reported here for completeness but should not be interpreted as
distributional signal.

Note on `gpu_utilization` semantics differing by tier: A16 and Tier
3 report per-pod-imputed whole-GPU-utilization values (system value
divided by r). Tier 2 reports per-pod per-slice engine-active
percentages (each pod's own 1g.12gb slice). These are semantically
different measurements of "how much GPU is this pod using" and are
not directly numerically comparable — a Tier 2 pod at 1.8% on its
own slice is roughly consistent with the same pod producing a
0.9% whole-H100 reading if that pod were on time-sliced hardware.
The relationship is scope-dependent, not a simple scaling.

### 2.3 Cross-pod spread audit at r=7

Per-pod `gpu_utilization` time-mean values at r=7 (all 7 slices
active) across the 5 workloads, denormalized units, one row per
workload:

| Workload | Per-pod means (7 values) | Mean | Spread (max-min) | Stdev |
|---|---|---|---|---|
| bert | 1.84, 1.81, 1.85, 1.85, 1.82, 1.82, 1.82 | 1.830 | 0.039 | 0.014 |
| gpt2 | 16.03, 15.69, 16.01, 15.54, 15.55, 16.12, 15.61 | 15.795 | 0.577 | 0.232 |
| resnet152 | 1.95, 1.94, 1.94, 2.00, 1.94, 1.99, 1.99 | 1.964 | 0.068 | 0.028 |
| whisper | 5.85, 5.73, 5.57, 5.77, 5.35, 5.96, 5.75 | 5.710 | 0.615 | 0.185 |
| yolo | 0.93, 0.93, 0.96, 0.96, 0.93, 0.94, 0.96 | 0.944 | 0.028 | 0.012 |

Two workloads produce visibly wider per-pod spread than the other
three: GPT-2 (spread 0.577) and whisper (0.615). The mechanisms
differ. GPT-2 is autoregressive with variable per-inference
generation length, so per-pod work legitimately varies pod-to-pod
even under perfect MIG GPU isolation. Whisper preserves per-pod
variance through host CPU contention (16 vCPUs shared across 7
pods), which the r=1 versus r=7 latency check confirms — whisper
r=7 mean latency is 2.86x r=1, while the four GPU-bound workloads
sit at 0.92 to 1.07x. Both signals were destroyed by A16 and Tier
3's per-pod imputation and are recovered here by direct
per-slice measurement. This is the empirical basis for the Section
2.1 source-column choice.

### 2.4 Front-gap handling

Front-gap patterns on Tier 2 affect 2 of 35 experiments (gpt2 r=7,
whisper r=7), 10 series total, all in the request-derived
`pod_latency_avg` and `pod_throughput` metrics. Maximum observed gap
is 15 seconds, well under the `MAX_FRONT_GAP_S=180` tripwire.
Handled by left-pad forward-fill on the same per-(metric, pod)
basis as Tier 3. Magnitude sits between A16's 1/50 (2%) and Tier
3's 13/50 (26%), closer to A16. Consistent with MIG hard-partitioning
reducing the cross-pod contention that produces request-completion
delays under time-slicing.

## 3. Model and hyperparameters

Architecture and training recipe are byte-frozen from the thesis and
from Tier 3. Encoder-free segment-based LSTM generator with
spectral-normalized LSTM discriminator. Generator conditioning is
88-dimensional (latent 64 + replica-count embedding 16 + phase
embedding 8). LSTM hidden 128, 2 layers, dropout 0.1. Output 7
metrics after `DROP_PER_WORKLOAD` (drops `gpu_memory_used`,
`gpu_memory_total`, `gpu_temperature`, all reconstructed statistically
in post-processing). Segments are 6 phases x 120 timesteps = 720,
truncated to 715.

**Per-workload frozen hyperparameters** (same across A16, Tier 3,
Tier 2):

| Workload | lambda_var_reg | lambda_fm_stat | Tag |
|---|---|---|---|
| bert | 0.5 | 1.5 | vr05/fm15 |
| gpt2 | 0.3 | 2.0 | vr03/fm20 |
| resnet152 | 0.4 | 1.0 | vr04/fm10 |
| whisper | 0.5 | 0.5 | vr05/fm05 |
| yolo | 0.3 | 1.2 | vr03/fm12 |

Training regime: 20 warmup + 150 adversarial epochs (170 total),
seed=42, WGAN-GP with lambda=10, Adam (generator lr=1e-3,
discriminator lr=2e-4, betas=(0.0, 0.9)). Discriminator update
frequency n_disc_steps=2 for all workloads except whisper, which
uses n_disc_steps=1. Identical to A16 and Tier 3.

The ablation grid is a 4-neighbor local search around each
workload's frozen point: `lambda_var_reg +/- 0.2` and
`lambda_fm_stat x 0.5` and `x 2.0`. 5 workloads x 4 variants = 20
runs. Grid is strictly formula-defined for Tier 2 with no per-workload
manual deviations (contrast: Tier 3's ablation had a one-off
`fm=0.3` for whisper `fm_low` instead of the formula's 0.25; Tier 2
restored the formula throughout).

## 4. Results

### 4.1 Headline VR (smooth) — three-way

| Workload | A16 (thesis) | Tier 3 frozen | Tier 3 best-tuned | Best variant (T3) | Tier 2 frozen | Tier 2 best variant | Tier 2 best value |
|---|---|---|---|---|---|---|---|
| bert | 1.022 | 1.885 | 1.885 | baseline | 1.142 | frozen | 1.142 |
| gpt2 | 1.137 | 0.920 | 1.000 | fm_low | 0.612 | fm_low (fm=1.0) | 0.616 |
| resnet152 | 0.985 | 1.064 | 0.940 | fm_high | 0.369 | fm_high (fm=2.0) | 0.432 |
| whisper | 1.397 | 0.759 | 0.994 | fm_low | 1.106 | fm_low (fm=0.25) | 1.294 |
| yolo | 1.039 | 1.069 | 1.011 | fm_low | 0.774 | fm_low (fm=0.6) | 0.957 |
| **Mean** | **1.116** | **1.140** | **1.166** | — | **0.800** | — | **0.888** |

Pass criterion is one-sided `VR_smooth > 0.8`, matching the Tier 3
reference doc's Section 4.1 convention.

Frozen pass count: 2/5 on Tier 2 (bert, whisper) versus Tier 3's 4/5
and A16's 5/5. Ablation pass count: 3/5 (adds yolo via fm_low,
0.774 → 0.957). Two workloads — gpt2 and resnet152 — fail all 5
configurations in the local ablation neighborhood and are documented
as honest limitations in Section 6.

### 4.2 Wasserstein distance (model versus real, pooled per workload)

Per-workload mean Wasserstein distance across the 7 trained metrics,
pooled across r=1..7 for Tier 2 and r=1..10 for A16 and Tier 3:

| Workload | A16 | Tier 3 | Tier 2 |
|---|---|---|---|
| bert | 1.543 | 11.741 | 12.004 |
| gpt2 | 4.736 | 21.903 | 19.286 |
| resnet152 | 2.973 | 14.566 | 19.137 |
| whisper | 5.515 | 25.972 | 24.322 |
| yolo | 1.197 | 8.628 | 9.910 |
| **Mean** | **3.193** | **16.562** | **16.932** |

Tier 2 and Tier 3 Wasserstein magnitudes track each other closely,
both roughly 5x A16. Interpretation carries over from the Tier 3
reference doc's Section 4.2: pooled means are dominated by
large-absolute-unit metrics (`pod_memory_bytes`, `gpu_power_watts`),
so the ~5x ratio is a shared raw-unit scale effect between the two
H100 tiers rather than uniformly worse fidelity relative to A16.

### 4.3 Ablation (20 variants)

Full ablation table, VR (smooth) per (workload, variant), best value
in each row bolded:

| Workload | frozen | vr_low | vr_high | fm_low | fm_high |
|---|---|---|---|---|---|
| bert | **1.142** | 0.914 | 0.828 | 1.052 | 0.961 |
| gpt2 | 0.612 | 0.448 | 0.507 | **0.616** | 0.485 |
| resnet152 | 0.369 | 0.361 | 0.393 | 0.302 | **0.432** |
| whisper | 1.106 | 1.166 | 0.824 | **1.294** | 1.016 |
| yolo | 0.774 | 0.508 | 0.550 | **0.957** | 0.638 |

Three workloads (gpt2, whisper, yolo) best-recover through `fm_low`
— a systematically-lower `lambda_fm_stat` than the A16-tuned frozen
values. This suggests Tier 2 as a whole benefits from lower
feature-matching regularization strength than A16, plausibly because
MIG isolation reduces per-pod variance for GPU-bound workloads and
the frozen feature-matching targets are consequently tuned to a
noisier signal than Tier 2 actually produces. Direction is empirical
here; a globally-retuned `lambda_fm_stat` for Tier 2 is a candidate
future work item (Section 6).

BERT's frozen configuration is already its own best across all 5
variants — no local perturbation improves it. Consistent with Tier
3's finding that bert's low frozen VR (there, 1.885; here, 1.142)
is architecture-limited rather than tuning-limited.

### 4.4 Per-workload narratives

**BERT** (frozen VR 1.142, pass). Passes frozen with mild overshoot,
same direction as Tier 3 (1.885) but substantially smaller
magnitude. The `pod_memory_bytes` small-denominator effect that
drove Tier 3's severe overshoot is muted on Tier 2, presumably
because Tier 2's per-pod memory values sit in a similar range to
Tier 3's (1.12e9 vs 1.04e9 mean, per Section 2.2). None of the 4
ablation variants improves on frozen; the local neighborhood does
not contain a better configuration.

**GPT-2** (frozen VR 0.612, fail; best variant 0.616, fail). GPT-2
was expected to benefit from `vr_high` given the autoregressive
per-pod variance signal preserved on Tier 2 (Section 2.3). The
empirical ablation direction was opposite: `vr_high` reduced VR to
0.507, `fm_low` marginally improved it to 0.616, but no
configuration approached the pass threshold. Two candidate
mechanisms remain undistinguished by the local search: (a) the
A16-tuned recipe is a poor fit for the specific autoregressive
variance signature that per-slice measurement exposes, requiring a
wider hyperparameter search to recover; (b) the frozen S36
architecture cannot fit Tier 2's GPT-2 data at any local
neighborhood point around the A16 configuration. Documented as an
open question in Section 6.

**ResNet152** (frozen VR 0.369, fail; best variant 0.432, fail).
Worst VR of any (tier, workload) combination in the whole
three-tier study. Local ablation neighborhood is far from
sufficient — reaching the 0.8 threshold would require ~85%
improvement, versus the ~30% improvement Tier 3's whisper achieved
through `fm_low`. `fm_high` provided the best local recovery
(0.302 → 0.432, roughly a 17% relative gain over frozen), which is
directionally consistent with Tier 2's overall `fm_low` trend being
wrong for resnet152. Interpretation is uncertain: either the recipe
needs a substantially different hyperparameter regime for Tier 2
resnet152 (candidate: substantially higher fm_stat than the frozen
1.0), or the architecture is genuinely mismatched to Tier 2's
resnet152 data distribution. This is the primary honest limitation
of the Tier 2 extension.

**Whisper** (frozen VR 1.106, pass; best variant fm_low VR 1.294,
pass with mild overshoot). Whisper flips from Tier 3's failure
(0.759) to Tier 2's clean pass, reversing Tier 3's regime-crossing
story. Physical interpretation: Tier 3's whisper failure was
attributed to a CPU-comfortable regime on H100 (PSI 0.32) versus
the CPU-saturated A16 regime (PSI 0.63) the frozen `fm=0.5` was
tuned for. On Tier 2, MIG's host CPU sharing (16 vCPUs across 7
pods at r=7) preserves substantial CPU contention (r=7 latency
2.86x r=1) and consequently substantial per-pod variance, moving
whisper back toward the A16-like regime the frozen recipe expected.
Whisper's `fm_low` variant provides an additional 17% improvement,
tightening the fit further. This is the cleanest cross-tier physical
story in the extension.

**YOLO** (frozen VR 0.774, fail; best variant fm_low VR 0.957,
pass). Marginal frozen failure recovers cleanly through `fm_low`, a
23% relative improvement. Same direction as whisper and gpt2 —
lower feature-matching regularization is broadly better for Tier 2
— but starting from a closer-to-threshold position, the local
recovery suffices. YOLO is Tier 2's cleanest ablation-recovery
story.

### 4.5 Cross-tier dataset-distance Wasserstein

Real-data-to-real-data Wasserstein distance between tiers, per
metric per workload, denormalized units. Selected rows below (full
table at `outputs/analysis/cross_tier/three_way_wasserstein_table.md`):

| Workload | Metric | A16 vs Tier 2 | A16 vs Tier 3 |
|---|---|---|---|
| bert | pod_cpu_usage | 0.032 | 0.028 |
| bert | gpu_utilization | 1.184 | 0.895 |
| bert | pod_latency_avg | 0.003 | 0.005 |
| gpt2 | gpu_utilization | 3.614 | 5.780 |
| whisper | pod_cpu_usage | 1.205 | 0.184 |
| whisper | gpu_utilization | 4.743 | 0.320 |
| whisper | pod_psi_cpu | 0.391 | 0.008 |
| resnet152 | gpu_utilization | 1.986 | 0.762 |
| yolo | gpu_utilization | 1.006 | 0.183 |

Two patterns emerge. First, Tier2-vs-Tier3 distances (not shown
above but present in the full table) are consistently smaller than
either-vs-A16, reflecting that both extension tiers were collected
on the same H100 devLab with the same v4 containers, while A16 is
older hardware and older collector code. The sharing-mode
differences we're studying (time-slicing versus MIG) are visible
against a smaller baseline of collection-pipeline noise than the
generation differences (A16 versus H100). Second, whisper is the
consistent outlier — largest A16-vs-Tier2 distances on
`pod_cpu_usage`, `pod_psi_cpu`, and `gpu_utilization`, all 5-25x
its nearest sibling workload. Physical explanation: whisper is the
one CPU-bound workload, host CPU is shared even under MIG, so
CPU-side distributional differences hit whisper harder than the
four GPU-bound workloads.

### 4.6 Per-slice versus averaged diagnostic

Quantifies precisely what the alternative /7 approach (dividing the
aggregated file by MIG's fixed 7-slice count and broadcasting to all
pods) would have destroyed. At r=7 for each workload, real per-pod
cross-pod variance in `gpu_utilization` versus the /7 broadcast
(which produces variance = 0 exactly by construction):

| Workload | mean cross-pod variance (real per-slice) | var of pod means (real) | fraction of total variance |
|---|---|---|---|
| bert | 0.042 | 0.00019 | 0.014% |
| gpt2 | 2.184 | 0.05390 | 0.111% |
| resnet152 | 0.035 | 0.00077 | 0.049% |
| whisper | 0.298 | 0.03439 | 1.011% |
| yolo | 0.001 | 0.00014 | 0.041% |

Absolute cross-pod variance is largest for GPT-2 (2.184) — 7x
whisper's — consistent with the autoregressive per-pod variance
mechanism. Whisper has the highest variance fraction (1.01%)
despite lower absolute variance than GPT-2, because whisper's
denominator (total signal variance) is smaller. All fractions sit
under 1.1%, so /7 would have destroyed a small fraction of the
signal in relative terms. The destroyed signal is nonetheless
paper-worthy content in absolute terms: GPT-2's 2.18 destroyed
variance is a real number quantifying what time-sliced tiers'
imputation structurally loses, independent of whether the loss
materially affects downstream S36 fidelity on this dataset.

### 4.7 Note on `gpu_utilization` cross-tier scope

Per Section 2.2, Tier 2's `gpu_utilization` measurements are
per-pod per-slice engine-active percentages (each pod's own
1g.12gb slice, DCGM GPM counter), while A16 and Tier 3 report
per-pod-imputed values derived from whole-GPU DCGM legacy SM
occupancy divided by replica count. Numerically both are in a
0..100 scale after their respective normalizations, but the
underlying physical quantity differs. Cross-tier comparative plots
of `gpu_utilization` should note this scope difference; values are
not directly comparable, and the observation that Tier 2's raw
values are lower than Tier 3's (Section 2.2) reflects the
per-slice denominator (1 slice out of 7) rather than lower actual
GPU utilization per pod.

## 5. Reproducibility

Retraining the frozen and ablation checkpoints from scratch:

```
# On A16 VM (172.22.174.58), tracegen conda env, branch extension-h100
cd ~/generative-ai-workload-modeling

# Frozen retrain (5 workloads, ~30 min total)
bash scripts/phase4/timegan/train_tier2_batch.sh

# Frozen eval + validate
PYTHONPATH=$(pwd)/scripts/utils python \
  scripts/phase4/evaluation/eval_s36_tier2.py
PYTHONPATH=$(pwd)/scripts/utils python \
  scripts/phase4/validate_s36_tier2.py

# Ablation (20 variants, ~2 hours)
bash scripts/phase4/timegan/sweep_tier2_batch.sh

# Ablation eval
PYTHONPATH=$(pwd)/scripts/utils python \
  scripts/phase4/evaluation/eval_s36_tier2_sweep.py
```

Preprocessing and unification pipeline (not needed if the processed
data is already in place):

```
python scripts/phase4/preprocess_tier2.py
python scripts/phase4/unify_tier2.py
```

Cross-tier Wasserstein diagnostic:

```
python scripts/analysis/a16_vs_tier2_wasserstein.py
```

**Paths.** Raw data: `data/raw/extension_tier2/<workload>_r<n>/`.
Processed: `data/processed/tier2/<workload>_traces.npz`. Unified:
`data/processed/tier2/unified/combined_dataset.npz` +
`combined_normalization.json`. Checkpoints:
`models/phase4/timegan_s36_tier2/s36_<workload>_<tag>/generator.pt`
for frozen, `models/phase4/timegan_s36_tier2_sweep/` for the 20
ablation variants. Eval outputs: `outputs/phase4/timegan_s36_tier2/`
and `outputs/phase4/timegan_s36_tier2_sweep/`. Validation report:
`outputs/phase4/validation/s36_tier2/`.

**Seeds.** All training uses `seed=42`. The train/val split (125/15
of 140 pods) is derived from a `numpy.random.default_rng(42)`
shuffle, mirroring the Tier 3 mechanism.

## 6. Open items and honest limitations

**GPT-2 does not recover in local ablation.** Frozen VR 0.612, best
variant VR 0.616. The physical hypothesis that autoregressive
per-pod variance under MIG isolation would benefit from `vr_high`
(reduced variance regularization) was not supported empirically —
`vr_high` reduced VR to 0.507. `fm_low` provided the only marginal
improvement. A candidate future-work direction is a wider
hyperparameter search around Tier 2's `lambda_fm_stat` direction for
GPT-2 specifically, potentially with values well below the ablation
neighborhood's fm=1.0.

**ResNet152 is the primary honest limitation.** Frozen VR 0.369,
best variant VR 0.432 through `fm_high`. Local ablation is
insufficient to reach the pass threshold. Two candidate mechanisms
(hyperparameter regime substantially different from A16 versus
architecture-data mismatch) remain undistinguished. This is worth a
paper-body sentence noting that the frozen recipe's transfer to MIG
partitioning is workload-dependent and can fail hard.

**Training loss does not track VR at the ablation neighborhood
scale.** Whisper's fm_low variant had the worst training val loss
(0.995) of its 4 ablation variants but produced whisper's best VR
(1.294). Training loss optimizes a composite adversarial +
variance-regularizer + feature-matching objective that is proximal
to VR but not identical; the two can decorrelate at fine-grained
hyperparameter differences. Worth a methods-section note.

**Per-slice file's role beyond `gpu_utilization`.** The MIG per-slice
files also contain per-pod `gpu_dram_active`, `gpu_pipe_tensor_active`,
and `gpu_total_energy_consumption` measurements that were not
included in the trained metric set (which is frozen from the thesis
at 7 metrics + 3 statistically reconstructed). These are informative
per-pod signals that time-sliced tiers cannot produce. Future work:
extend the trained metric set to include per-pod DRAM and tensor-core
activity, retrain on Tier 2, and compare fidelity against
model-augmented A16 and Tier 3 baselines that would necessarily
impute the additional metrics from whole-GPU DCGM.

**`fm_low` as a global Tier 2 pattern.** Three of five workloads
(gpt2, whisper, yolo) best-recover through `fm_low`, suggesting
Tier 2 as a whole benefits from lower `lambda_fm_stat` than the
A16-tuned frozen values. A globally-retuned `lambda_fm_stat` for
Tier 2 is a candidate future direction that would test whether
gpt2 and resnet152 recover under a more aggressive Tier-2-specific
retune — potentially reducing the 2/5 honest-limitation count to
1/5 (resnet152 alone) or 0/5.

## 7. Session provenance

Extension executed across a planning chat (Claude Opus 4.7) and
execution chats via Claude Code (Sonnet) on the A16 VM
(172.22.174.58). Planner produced consolidated single-shot prompts;
executor ran on the VM against branch `extension-h100`. Every
executor round-trip verified outputs before committing; three
verification failures during the extension caused planner review
before proceeding (sequence-length 720 versus 715 in Step 1,
normalization schema design in Step 3, hardcoded replica range in
Step 5).

**Commit range for Tier 2 extension** (branch `extension-h100`,
ordered): preprocessing pipeline + processed data (2d0508c), sanity
audit + per-slice-vs-averaged diagnostic (1fa0853), unified dataset
+ normalization (a4dc2cf), frozen retrain 5 checkpoints (753f3fa),
eval + validation (9dadfa4), ablation 20 variants + sweep eval
(3410555), cross-tier Wasserstein synthesis (8ed7b31), reference
doc + `TIER2_NOTES.md` augmentation (75449ef).

**Data provenance.** Raw collection performed 2026-08-14 through
2026-08-16 on the H100 devLab (172.22.174.66) via
`tools/run_experiment_v4.py` under `tools/run_tier2_batch.sh`.
Complete collection details in `TIER2_NOTES.md`.

**References.**
- `docs/S36_TIER3_EXTENSION_REFERENCE.md` — template and Tier 3
  headline numbers used throughout Section 4.
- `docs/plans/S36_TIER2_RETRAIN_PLAN.md` — plan document reviewed
  and green-lit before Step 1.
- `TIER2_NOTES.md` — Tier 2 collection reference; Section 9
  augmentation from this extension's per-workload observations.
- The thesis's Chapter 4 and Chapter 5 for A16 Phase 1 v3 baseline
  numbers and S36 architecture details.

**End of reference.**