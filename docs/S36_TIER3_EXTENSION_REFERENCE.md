## Post-E1 correction (2026-08-22)

The central interpretation in this document is superseded; the
methodology, collection details, and per-tier numbers below remain valid
and are still the paper's source of record.

Superseded claims:
1. Sharing-mode framing. See the matched-size control (E1): the MIG
   degradation is reproduced by size-matched time-slicing, so transfer is
   gated by training-data volume, not sharing mode or hardware
   generation. The A16-to-H100-time-slicing (hardware) transfer holds at
   full data and is unaffected.
2. bert Tier 3 VR 1.885. This is a pod_memory_bytes small-denominator
   artifact (H100 has ~6x the VRAM, real memory variance near zero).
   pod_memory_bytes is reconstructed in post-processing, not generated,
   and is excluded from the composite fidelity score; the 6-metric bert
   Tier 3 composite is ~0.99.
3. Whisper Tier 3 "regime crossing" physical story and the fm 0.5->0.3
   recovery narrative. Withdrawn (matched-r CPU contention is
   near-identical across the two H100 configs; the effect is hardware,
   A16-vs-H100).
4. Raw pooled Wasserstein (~5x) figures. Retired in favour of
   scale-normalized per-metric distances.

Current framing: see S36_EXTENSION_ANCHOR.md. Body below is retained as
the record of the pre-E1 analysis.

---

# S36 Tier 3 Extension: Reference Document

Consolidated reference for the H100 Tier 3 extension of the S36
workload trace generation work. Sole reference needed for writing the
thesis extension chapter or paper draft on this material.

## 1. Motivation and scope

The thesis's S36 model was trained and validated exclusively on A16
GPU data (Phase 1 v3). This extension asks whether the same trained
recipe — architecture, hyperparameters, training schedule, all frozen
exactly as in the thesis — transfers to a different GPU generation
(H100) without re-tuning. "Apples-to-apples" here means literally
reusing the thesis's `S27_HYPERPARAMS` values, seed, epoch counts, and
architecture unchanged, and only ever adapting *paths* (data
locations, output directories) when porting scripts to the new
dataset; the one exception — a bounded hyperparameter ablation — is
reported as a clearly separate, explicitly-labeled addition, not a
silent change to the "frozen" baseline. The A16 numbers used for
comparison throughout are the thesis's own reported figures, not
recomputed.

## 2. Dataset

**Tier 3 collection summary.** 50 experiments = 5 workloads (bert,
gpt2, resnet152, whisper, yolo) × replica counts r=1..10, collected on
H100 with time-slicing 10 (10-way GPU sharing), each experiment
running a business-day 6-phase synthetic load profile, Prometheus
scraping every 5s, single-node Kubernetes cluster.

**Contrast with A16 Phase 1 v3.** Identical protocol at the experiment
level (same workload set, same r=1..10 range, same 6-phase load
profile, same 5s scrape interval) — only the underlying GPU hardware
differs. Physical differences confirmed via Step 2's raw (denormalized)
per-metric mean audit on BERT as a representative workload:

| Metric | A16 raw mean | Tier 3 raw mean | Ratio |
|---|---|---|---|
| `gpu_memory_total` | 2720.3 | 17330.6 | 6.37x |
| `gpu_power_watts` | 7.26 | 23.14 | 3.19x |
| `gpu_utilization` | 2.70 | 0.90 | 0.33x (lower on Tier 3) |
| `gpu_temperature` | 12.26 | 10.28 | 0.84x |

`gpu_memory_total` is a workload-independent hardware constant — the
A16↔Tier3 Wasserstein distance for this metric is identical (14,610)
across all 5 workloads (Section 4.6), confirming it purely reflects
total VRAM difference (A16 ≈ 15GB-class allocation, H100 ≈ 95GB-class
allocation, consistent with H100's much larger memory). H100 needs
proportionally *less* GPU utilization to do the same inference work —
this is the opposite direction of what a naive "H100 is more powerful
so metrics scale up uniformly" assumption would predict, and matters
for interpreting both the VR and Wasserstein results below.

**Preprocessing pipeline.** `scripts/phase4/preprocess_tier3.py`,
adapted from the recovered `scripts/phase4/preprocess_v3_10metric.py`
(recovered verbatim from git history, commit `83eb884^`, confirmed to
reproduce the tracked `data/processed/phase1_v3/*.npz` files
byte-for-byte before any Tier 3 work began). The adaptation's one
substantive logic change: `align_timestamps()` was rewritten from an
intersection-based join to a union-of-timestamps join with
per-(metric, pod) forward-fill, needed because Tier 3 experiments
under high replica-count contention exhibit a front-gap in
`pod_latency_avg`/`pod_throughput` — these two metrics are computed
from *completed* inference requests, and under heavy contention the
first request can take tens of seconds to complete, while
cAdvisor/DCGM resource metrics report from t=0 regardless. A
`MAX_FRONT_GAP_S=180` tripwire logs loudly (never triggered — max
observed gap was 165s) but never hard-fails.

**The A16 vs Tier 3 preprocessing asymmetry, with physical reasoning.**
Both tiers exhibit *some* version of this front-gap pattern under
contention, but the underlying failure mode differs and warrants
different handling:
- **A16**: exactly 1 experiment, 1 (metric, pod) series affected
  (whisper r=10, `pod_cpu_usage`, one pod, 325s gap) — handled by
  zero-fill (`nan_to_num`), which is *defensible* here because before a
  pod is actually running, CPU usage genuinely is ~0 (no process
  consuming CPU yet).
- **Tier 3**: 13 of 50 experiments, 154 (metric, pod) series affected,
  concentrated in `pod_latency_avg`/`pod_throughput` (request-derived
  metrics) — handled by forward-fill from the first observed value,
  which is *correct* here because zero-fill would misrepresent "no
  data yet" as "0ms latency" / "0 req/s throughput," actively
  misleading rather than merely imprecise.

Both choices are physically motivated by what each specific metric
means, not an arbitrary inconsistency. The initial diagnosis (pooled
per-metric timestamp intersection) undercounted the Tier 3 pattern
(predicted 3 experiments, actual 13) because pooling across all pods
in a metric masks a lag in any single pod as long as others start on
time — the per-pod re-audit that found the true 13/50 (Tier 3) and
1/50 (A16) numbers is the one to cite.

**Unified dataset.** `scripts/phase4/unify_tier3.py` concatenates the
5 per-workload `.npz` files into
`data/processed/tier3/unified/combined_dataset.npz`: 275 traces total
(55 × 5 workloads, shape `(275, 715, 10)`), workload order
`[bert, gpt2, resnet152, whisper, yolo]` matching A16's
`workload_ids` ordering, train/val split via `seed=42`,
`np.random.default_rng(42).shuffle` then an 11% val fraction —
245/30, matching A16's exact split ratio.

## 3. Model and hyperparameters

**S36 architecture** (frozen from thesis, unchanged for Tier 3):
encoder-free, segment-based LSTM generator with spectral-normalized
LSTM discriminator (`DiscriminatorSpectral`, S34 architecture + spectral
norm). Generator conditioning is 88-dimensional: latent noise (64) +
replica-count embedding (16) + phase embedding (8). LSTM hidden size
128, 2 layers, dropout 0.1. Output: 7 metrics (the 3
near-constant-per-workload GPU metrics — `gpu_memory_used`,
`gpu_memory_total`, `gpu_temperature` — are dropped per
`DROP_PER_WORKLOAD`, identical set for all 5 workloads). Segments are
120 timesteps × 6 phases = 720, truncated to the real trace length of
715.

**Per-workload frozen hyperparameters** (from thesis; verified
byte-for-byte against `S27_HYPERPARAMS` in `timegan_s36_tier3.py`
before any Tier 3 training ran):

| Workload | lambda_var_reg | lambda_fm_stat | Tag |
|---|---|---|---|
| bert | 0.5 | 1.5 | vr05/fm15 |
| gpt2 | 0.3 | 2.0 | vr03/fm20 |
| resnet152 | 0.4 | 1.0 | vr04/fm10 |
| whisper | 0.5 | 0.5 | vr05/fm05 |
| yolo | 0.3 | 1.2 | vr03/fm12 |

**Training regime.** 20 warmup epochs + 150 adversarial epochs (170
total), seed=42, WGAN-GP adversarial loss with gradient penalty
(λ=10), Adam optimizers (generator lr=1e-3, discriminator lr=2e-4,
β=(0.0, 0.9)) — identical to A16.

**Ablation grid** (a bounded addition beyond the frozen recipe,
clearly separate from the headline apples-to-apples result): 4
one-off neighbors per workload around the frozen point —
`lambda_var_reg ± 0.2` (constant step; all frozen vr values sit in
[0.3, 0.5]), `lambda_fm_stat × 0.5` and `× 2.0` (multiplicative step;
frozen fm values span a wider range, 0.5–2.0, across workloads). 5
workloads × 4 variants = 20 additional training + eval runs.

## 4. Results

### 4.1 Headline: cross-tier VR

| Workload | A16 VR (thesis) | Tier 3 VR (frozen) | Tier 3 VR (best-tuned) | Best variant | Pass 0.8 (frozen / best) |
|---|---|---|---|---|---|
| bert | 1.022 | 1.885 | 1.885 | baseline | PASS / PASS |
| gpt2 | 1.137 | 0.920 | 1.000 | fm_low | PASS / PASS |
| resnet152 | 0.985 | 1.064 | 0.940 | fm_high | PASS / PASS |
| whisper | 1.397 | 0.759 | 0.994 | fm_low | FAIL / PASS |
| yolo | 1.039 | 1.069 | 1.011 | fm_low | PASS / PASS |
| **Mean** | **1.116** | **1.140** | **1.166** | — | **4/5 / 5/5** |

The frozen recipe passes 4/5 workloads on Tier 3 (whisper is the one
failure). A minimal local hyperparameter search recovers whisper to a
pass and leaves every other workload's pass status unchanged.

### 4.2 Cross-tier Wasserstein

| Workload | A16 W (thesis) | Tier 3 W (frozen) | Delta | Ratio |
|---|---|---|---|---|
| bert | 1.543 | 11.741 | +10.198 | 7.61x |
| gpt2 | 4.736 | 21.903 | +17.167 | 4.62x |
| resnet152 | 2.973 | 14.566 | +11.593 | 4.90x |
| whisper | 5.515 | 25.972 | +20.457 | 4.71x |
| yolo | 1.197 | 8.628 | +7.431 | 7.21x |
| **Mean** | **3.193** | **16.562** | **+13.369** | **5.19x** |

**Scale-artifact caveat.** Wasserstein here is raw-physical-unit
distance, not scale-invariant like VR. `Pod Memory` and `GPU Power`
dominate the pooled per-workload mean (tens to hundreds vs. sub-1 for
CPU/PSI/Latency-type metrics) purely because they're measured in
larger absolute units, and Tier 3's absolute GPU-metric magnitudes are
already known-larger (Section 2 table). The near-uniform ~5x inflation
across all 5 workloads — despite very different VR behavior per
workload — is more consistent with this shared scale effect than five
independent fidelity failures, though this analysis cannot fully
separate the two explanations.

### 4.3 Hyperparameter ablation on Tier 3

VR (smooth) is the threshold metric throughout (reproduces the
thesis's A16 figures exactly). **Bold** = best variant per workload
(closest to 1.0, not highest).

**bert** (baseline: vr=0.5, fm=1.5) — frozen baseline is already
optimal; every swept neighbor moves further from 1.0:

| Variant | vr | fm | VR (smooth) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|
| **baseline** | 0.5 | 1.5 | **1.885** | PASS | **0.885** |
| vr_low | 0.3 | 1.5 | 1.899 | PASS | 0.899 |
| vr_high | 0.7 | 1.5 | 1.939 | PASS | 0.939 |
| fm_low | 0.5 | 0.75 | 2.284 | PASS | 1.284 |
| fm_high | 0.5 | 3.0 | 1.922 | PASS | 0.922 |

**gpt2** (baseline: vr=0.3, fm=2.0):

| Variant | vr | fm | VR (smooth) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|
| baseline | 0.3 | 2.0 | 0.920 | PASS | 0.080 |
| vr_low | 0.1 | 2.0 | 1.007 | PASS | 0.007 |
| vr_high | 0.5 | 2.0 | 1.136 | PASS | 0.136 |
| **fm_low** | 0.3 | 1.0 | **1.000** | PASS | **0.0003** |
| fm_high | 0.3 | 4.0 | 0.889 | PASS | 0.111 |

**resnet152** (baseline: vr=0.4, fm=1.0) — `fm_low` is the only variant
in the whole sweep that flips a passing baseline to failing:

| Variant | vr | fm | VR (smooth) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|
| baseline | 0.4 | 1.0 | 1.064 | PASS | 0.064 |
| vr_low | 0.2 | 1.0 | 1.108 | PASS | 0.108 |
| vr_high | 0.6 | 1.0 | 1.359 | PASS | 0.359 |
| fm_low | 0.4 | 0.5 | 0.786 | **FAIL** | 0.214 |
| **fm_high** | 0.4 | 2.0 | **0.940** | PASS | **0.060** |

**whisper** (baseline: vr=0.5, fm=0.5) — the ablation's headline
result, see Section 4.4:

| Variant | vr | fm | VR (smooth) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|
| baseline | 0.5 | 0.5 | 0.759 | **FAIL** | 0.241 |
| vr_low | 0.3 | 0.5 | 0.736 | **FAIL** | 0.264 |
| vr_high | 0.7 | 0.5 | 0.931 | PASS | 0.069 |
| **fm_low** | 0.5 | 0.3 | **0.994** | PASS | **0.006** |
| fm_high | 0.5 | 1.0 | 0.888 | PASS | 0.112 |

**yolo** (baseline: vr=0.3, fm=1.2):

| Variant | vr | fm | VR (smooth) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|
| baseline | 0.3 | 1.2 | 1.069 | PASS | 0.069 |
| vr_low | 0.1 | 1.2 | 1.077 | PASS | 0.077 |
| vr_high | 0.5 | 1.2 | 0.874 | PASS | 0.126 |
| **fm_low** | 0.3 | 0.6 | **1.011** | PASS | **0.011** |
| fm_high | 0.3 | 2.4 | 0.917 | PASS | 0.083 |

**Best-variant summary.** Using the task's bar for a "meaningful"
improvement (baseline `|Δ|` minus best-variant `|Δ|` > 0.1):

| Workload | Best variant | Improvement over baseline | Meaningful (>0.1)? |
|---|---|---|---|
| bert | baseline itself | 0.000 | No |
| gpt2 | fm_low | 0.080 | No — just under the bar |
| resnet152 | fm_high | 0.004 | No — negligible |
| whisper | fm_low | **0.235** | **Yes** |
| yolo | fm_low | 0.058 | No — modest |

Only whisper clears the meaningful-improvement bar, by a wide margin.
`fm_low` is the best variant for 4 of 5 workloads (all but bert, where
it's the *worst* variant) — no single tuning direction (vr up/down, fm
up/down) is uniformly beneficial; the direction that helps is
workload-specific.

### 4.4 Whisper narrative

Whisper is the one workload where the frozen A16 recipe fails on Tier
3 (VR 0.759 < 0.8 threshold), and it is also the workload with the
largest documented A16↔Tier3 real-hardware behavioral difference:
A16 whisper crashed at PSI 0.63 with ~1576ms latency at r=10 (10 pods,
10-slot time-slicing config); Tier 3 whisper completed the identical
configuration cleanly, with PSI ≈ 0.32 and ~643ms latency (~2.5x
lower). Under this framing, the frozen recipe's `lambda_fm_stat=0.5`
was implicitly tuned against A16's higher-variance, higher-contention
whisper distribution. On Tier 3's lower-variance, lower-latency
whisper data, that same feature-matching weight over-constrains the
generator; reducing it to `fm=0.3` (leaving `lambda_var_reg` at the
frozen 0.5) recovers VR to 0.994 — the single closest-to-1.0 result
across all 25 data points (baseline + 20 sweep runs) in this entire
extension. The physical interpretation: lower real-distribution
variance calls for a lower FM-loss weight, since a strong
feature-matching penalty pushes the generator to match A16-scale
variance the Tier 3 data no longer has. This is also consistent with
Section 4.6's dataset-distance appendix, where whisper shows the
largest A16↔Tier3 real-data distance of all 5 workloads on exactly the
request-derived metrics (`pod_cpu_usage`, `pod_latency_avg`) most
relevant to this story.

### 4.5 Bert narrative

Bert shows the opposite pattern: the highest VR overshoot of any
workload on Tier 3 (1.885, vs. A16's 1.022), and — per the ablation —
the frozen baseline is already the best point in its entire local
hyperparameter neighborhood; every tested variant moves further from
1.0. The per-metric breakdown (Step 5) shows `pod_memory_bytes`
VR = 7.26 as the dominant single-metric contributor, far above every
other metric for any workload in this extension. Physical
interpretation: bert's memory footprint is a tiny, nearly-constant
fraction of available VRAM on H100 (which has roughly 6x A16's total
GPU memory — Section 2), so the real `pod_memory_bytes` variance
across pods/replicas is close to zero on Tier 3; any small amount of
generator output noise on this metric then produces a large *relative*
variance ratio, since VR is a ratio against a near-zero real-variance
denominator. This reads as a small-denominator VR instability specific
to this metric on this hardware, not a generation-quality failure —
worth reporting honestly rather than either dismissing bert's high VR
or treating it as a genuine 1.9x-real-variance finding.

### 4.6 Dataset distance appendix

Produced by `scripts/analysis/a16_vs_tier3_wasserstein.py` — real A16
vs. real Tier 3 distributional distance, **not a model fidelity
metric**, computed on denormalized (raw-unit) traces,
`data/processed/phase1_v3/{workload}_traces.npz` vs.
`data/processed/tier3/{workload}_traces.npz`, 55×715=39,325 samples
per tier per workload per metric.

| Workload | pod_cpu_usage | pod_memory_bytes | pod_psi_cpu | pod_latency_avg | pod_throughput | gpu_utilization | gpu_memory_used | gpu_memory_total | gpu_power_watts | gpu_temperature |
|---|---|---|---|---|---|---|---|---|---|---|
| bert | 0.02834 | 2.084e+08 | 1.884e-05 | 0.002583 | 0.1363 | 1.801 | 632 | 1.461e+04 | 15.88 | 1.986 |
| gpt2 | 0.1023 | 2.688e+08 | 3.663e-05 | 0.5483 | 0.1909 | 2.325 | 630.7 | 1.461e+04 | 15.74 | 2.214 |
| resnet152 | 0.01843 | 1.775e+09 | 1.766e-05 | 0.007736 | 0.09437 | 2.556 | 446.2 | 1.461e+04 | 15.49 | 1.946 |
| whisper | 1.239 | 3.412e+08 | 0.3466 | 0.7504 | 0.2753 | 7.592 | 709.2 | 1.461e+04 | 15.4 | 2.573 |
| yolo | 0.02896 | 1.315e+09 | 8.935e-06 | 0.009154 | 0.1319 | 1.138 | 286.9 | 1.461e+04 | 15.68 | 1.685 |

Flagged outliers (>10x the cross-workload median for that metric):
whisper on `pod_cpu_usage` (42.8x median), `pod_psi_cpu` (18,394x
median — but both absolute values are tiny, near-zero-to-small-nonzero,
not a large absolute shift), and `pod_latency_avg` (82.0x median);
gpt2 also elevated on `pod_latency_avg` (59.9x median). `gpu_memory_total`
is identical (14,610) across all 5 workloads by construction — a fixed
hardware constant, correctly workload-independent.

Whisper's real A16-vs-Tier3 data is genuinely more different than the
other four workloads', concentrated in the same request-derived
metrics implicated in Section 4.4's narrative — consistent with, not
proof of, whisper's model-fit difficulty partly reflecting real
distribution shift rather than purely a hyperparameter mismatch.

## 5. Reproducibility

### 5.1 Code paths

All on branch `extension-h100`:
- `scripts/phase4/preprocess_v3_10metric.py` — recovered from git
  history (`83eb884^`), historical reference, unmodified.
- `scripts/phase4/preprocess_tier3.py` — adapted (paths + the
  union/forward-fill logic change described in Section 2).
- `scripts/phase4/unify_tier3.py` — new.
- `scripts/phase4/timegan/timegan_s36_tier3.py` — adapted (paths only).
- `scripts/phase4/timegan/timegan_s36_tier3_sweep.py` — adapted, +
  `--workload`/`--vr`/`--fm` CLI for the ablation.
- `scripts/phase4/evaluation/eval_s36_tier3.py` — adapted (paths only).
- `scripts/phase4/evaluation/eval_s36_tier3_sweep.py` — adapted, +
  `--workload`/`--model-dir`/`--variant-tag` CLI for the ablation.
- `scripts/phase4/postprocess_s36_tier3.py` — adapted (paths only;
  required because `validate_s36.py` delegates model/data loading to
  this module — see Section 5.4).
- `scripts/phase4/validate_s36_tier3.py` — adapted (paths + import
  target only).
- `scripts/analysis/a16_vs_tier3_wasserstein.py` — new (Section 4.6).
- `scripts/analysis/tier3_sanity.py` — from earlier work (pre-dates
  this session), cross-tier loader validation.

### 5.2 Data paths

- Raw: `data/raw/extension_tier3/`
- Processed: `data/processed/tier3/`, `data/processed/tier3/unified/`
- Models: `models/phase4/timegan_s36_tier3/` (frozen recipe, 5
  checkpoints), `models/phase4/timegan_s36_tier3_sweep/` (20 ablation
  variants)
- Eval: `outputs/phase4/timegan_s36_tier3/`,
  `outputs/phase4/timegan_s36_tier3_sweep/`,
  `outputs/phase4/validation/s36_tier3/`

### 5.3 Environment

A16 VM (172.22.174.58), `tracegen` conda env, PyTorch 2.9.0, CUDA
12.8. Reused the thesis environment unchanged.

### 5.4 Known workarounds

- `eval_s36.py` (and its Tier 3 derivatives) has a stale `sys.path`
  expecting `scripts/phase4/utils/boundary_smoothing.py`; the actual
  file is at `scripts/utils/boundary_smoothing.py`. This predates the
  Tier 3 work — the original script would fail the same way if run
  fresh. Workaround at invocation, no code change:
  `PYTHONPATH=$(pwd)/scripts/utils`.
- `validate_s36.py` delegates all model-loading and real-data-loading
  to `postprocess_s36.py` (imports `generate_postprocessed_traces`,
  `load_generator`, `load_real_data`, etc.). Several call sites inside
  `validate_s36.py` invoke `generate_postprocessed_traces(...)`
  *without* forwarding a `data_dir` override, so adapting
  `validate_s36.py`'s own path defaults alone is insufficient — any
  adaptation of `validate_s36.py` must include a corresponding
  `postprocess_s36.py` adaptation (its `MODEL_PATHS` dict and 6
  `data_dir` defaults), or the synthetic-generation side silently keeps
  using the old paths.
- `timegan_s36.py`/`eval_s36.py` docstrings/console output say
  "176 traces (original, no augmentation)" — this is stale boilerplate,
  not the actual dataset size (see Section 6.2).

## 6. Open items and future work

### 6.1 Not done in this extension

- **Tier 2 (MIG) retrain**: same S36 pipeline, but `gpu_utilization`
  has different semantics under MIG (`DCGM_FI_PROF_GR_ENGINE_ACTIVE ×
  100`, summed across active slices — up to 700 at r=7 fully busy).
  Needs its own plan; not a drop-in path retarget.
- **Tier 1 (r=1 only) retrain**: single replica point, not enough data
  for a variance-ratio metric.
- **Kwok trace generation** at r=50, r=100 using the Tier 3 models
  (extrapolation zone was generated during Step 5's validation run but
  not analyzed in depth here).
- **Jan Hagemann's fidelity analysis workstream** — consumes these
  artifacts, parallel effort.
- **Cross-tier multi-tier unified training** (A16 + Tier 2 + Tier 3 in
  one model) — future direction, not attempted here.
- **Latency percentile per-pod recovery** — S36 doesn't train on
  percentiles (known, low priority per NEXT_STEPS), but the raw
  CSV export includes `app_latency_p50/p95/p99` on Tier 3 that may be
  recoverable from H100 data if ever needed.

### 6.2 Methodology honesty items for the paper

- **Whisper's validation set is 3 pods** (vs. 6–7 for the other four
  workloads, inherited from the unified 245/30 split's random
  assignment) — its VR estimate has less statistical support than the
  other workloads'.
- **Bert `pod_memory_bytes` VR = 7.26** reflects a small-denominator
  instability (Section 4.5), not a model failure — report the number
  honestly rather than omitting or reframing it.
- **Tier 3 Wasserstein numbers run ~5x A16's across every workload**,
  but this is largely a raw-unit scale artifact (`Pod Memory` and
  `GPU Power` dominate the pooled mean, and those metrics are
  mechanically larger in absolute terms on H100) — not read as a
  fidelity signal on its own (Section 4.2).
- **Sample-count discrepancy**: some thesis-era script docstrings/console
  output say "176 traces," but the actual unified dataset
  (`combined_dataset.npz`) has 275 traces (55 × 5 workloads, full
  r=1..10 coverage). 275 is correct; "176" is stale boilerplate text
  left over from an earlier iteration of the codebase, not a real
  discrepancy in the data itself.

## 7. Session provenance

This document consolidates work done in a single planning + execution
session (Aug 18–19, 2026). Detailed working records — including every
diagnostic run, script diff, and intermediate finding referenced above
— live under `docs/plans/S36_TIER3_*.md`.
