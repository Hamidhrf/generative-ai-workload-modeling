# S36 Tier 2 (MIG) Retrain Plan

Plan document for the H100 Tier 2 (MIG all-1g.12gb) extension of the
S36 workload trace generation work. Mirrors the structure of
`S36_TIER3_RETRAIN_PLAN.md`, adapted for Tier 2's two data
differences (per-pod GPU attribution native via MIG; replica range
r=1..7 not r=1..10). Reviewed and green-lit before any execution.

## 1. Purpose and scope

Third and final tier of the S36 cross-tier extension. Asks whether
the frozen thesis recipe — architecture, hyperparameters, seed,
epoch counts unchanged, only paths retargeted — transfers to MIG
hard-partitioned GPU sharing on H100. The apples-to-apples
discipline is the same as Tier 3: reuse `S27_HYPERPARAMS` values
byte-for-byte, only ever adapting paths and (this tier only) the
source column for `gpu_utilization`. The ablation is a clearly
separated, explicitly-labeled second stage, not a silent change to
the frozen baseline. A16 numbers throughout are the thesis's
reported figures; Tier 3 numbers are from
`S36_TIER3_EXTENSION_REFERENCE.md`, not recomputed.

The methodological headline for Tier 2, per `TIER2_NOTES.md`
Section 1, is that this is the first tier of the study where
per-pod GPU attribution is natively available (MIG per-instance
DCGM labels carry pod identity). A16 imputes per-pod as
`system_value / replica_count`; Tier 3 does the same. Tier 2
measures per-pod directly. This is central to the paper's
contribution and drives the source-column decision in Section 3
below.

## 2. Dataset

**Tier 2 collection summary.** 35 experiments = 5 workloads (bert,
gpt2, resnet152, whisper, yolo) x replica counts r=1..7, collected
on H100 devLab (172.22.174.66) with MIG single-strategy
all-1g.12gb (7 x 1g.12gb slices), each experiment running the same
60-minute Business Day 6-phase load profile as A16 Phase 1 v3 and
Tier 3, Prometheus scraping every 5s. Zero failures; whisper r=7
completed cleanly. Raw data at `data/raw/extension_tier2/`, 34
files per experiment (33 CSV + 1 timestamps.txt).

**Contrast with A16 Phase 1 v3 and Tier 3.**

| Tier | GPU sharing | r range | Traces / workload | Per-pod GPU attribution |
|---|---|---|---|---|
| A16 Phase 1 v3 | time-slicing 10 | 1..10 | 55 | inferred (whole / r) |
| Tier 3 H100 | time-slicing 10 | 1..10 | 55 | inferred (whole / r) |
| Tier 2 H100 | MIG 1g.12gb x 7 | 1..7 | 28 | direct per-slice |

Total Tier 2 pods: 28 x 5 = 140 (vs. Tier 3's 275). Sequence length
same 715 timesteps. Replica-count embedding trained on r=1..10 for
A16 stays inside range for r=1..7 — no extrapolation into the
embedding's untrained region.

**Front-gap handling.** The union + forward-fill fix from Tier 3's
`preprocess_tier3.py` carries over unchanged in shape. Magnitude on
Tier 2 is TBD — Step 1 audit reports the actual (metric, pod)
count of affected series and the max observed gap, same
`MAX_FRONT_GAP_S=180` tripwire as Tier 3.

## 3. Source column decision for `gpu_utilization` (per-slice)

The `gpu_utilization` training column for Tier 2 is populated from
`<workload>_r<n>_gpu_utilization_per_slice_*.csv`, not from the
aggregated `<workload>_r<n>_gpu_utilization_*.csv`. Concretely:
group per-slice rows by `pod` column, drop rows where pod is empty
(idle slices), and produce a per-pod trace column in the same shape
as every other per-pod metric already has. Per-slice values are
stored as `DCGM_FI_PROF_GR_ENGINE_ACTIVE * 100` (scaled in the
PromQL query itself, per Task 3 of the diagnostic audit), so raw
range is 0..100 per slice — no client-side scaling required. Fresh
Tier 2 normalization stats are fit on the training partition in
Step 3.

The aggregated `gpu_utilization` file is retained as a Section 4.7
comparison artifact (Step 2 diagnostic below), not used for
training. Its `sum-across-slices` design is honest for its intended
question ("how much total engine activity did this experiment
produce") but broadcasting an averaged value to every pod would
manually destroy per-pod variance — the same imputation Tier 2 was
designed to eliminate. Per-slice preserves the physical per-pod
signal that MIG hardware provides; that signal is the study's
contribution.

Pod-to-slice mapping is native per `TIER2_NOTES.md` Section 5.5 —
DCGM v26.3.3 populates the `pod` label directly on per-instance
series. Empirically confirmed by Task 1: 7 pods x 720 timesteps x 1
row per (pod, timestamp) = 5040 rows, ratio 7.0 exact. No mapping
resolution work needed; `df.groupby('pod')` suffices.

## 4. Model and hyperparameters

**Architecture** (frozen from thesis, unchanged from Tier 3):
encoder-free segment-based LSTM generator with spectral-normalized
LSTM discriminator (`DiscriminatorSpectral`). Generator
conditioning 88-dimensional: latent 64 + replica-count embedding 16
+ phase embedding 8. LSTM hidden 128, 2 layers, dropout 0.1.
Output 7 metrics after `DROP_PER_WORKLOAD` (drops
`gpu_memory_used`, `gpu_memory_total`, `gpu_temperature`). Segments
120 timesteps x 6 phases = 720, truncated to 715.

**Per-workload frozen hyperparameters** (same as thesis and Tier 3):

| Workload | lambda_var_reg | lambda_fm_stat | Tag |
|---|---|---|---|
| bert | 0.5 | 1.5 | vr05/fm15 |
| gpt2 | 0.3 | 2.0 | vr03/fm20 |
| resnet152 | 0.4 | 1.0 | vr04/fm10 |
| whisper | 0.5 | 0.5 | vr05/fm05 |
| yolo | 0.3 | 1.2 | vr03/fm12 |

**Training regime.** 20 warmup + 150 adversarial epochs (170
total), seed=42, WGAN-GP with lambda=10, Adam (generator lr=1e-3,
discriminator lr=2e-4, betas=(0.0, 0.9)). Identical to A16 and
Tier 3.

**Ablation grid.** Same 4-neighbor local search as Tier 3:
`lambda_var_reg +/- 0.2`, `lambda_fm_stat x 0.5` and `x 2.0`.
5 workloads x 4 variants = 20 additional runs. Run unconditionally
regardless of frozen pass/fail outcome — the ablation shows
recoverability of any overshoot, which is itself paper content.

## 5. Seven-step execution plan

Each step is atomic: has defined inputs, deliverables, and a
success check. Sequential dependencies noted. All commits go to
branch `extension-h100`, one commit per logical artifact. Long
compute runs go through tmux with `tee` to log files (SSH
resilience per project convention).

### Step 1: Preprocessing per-workload

**Objective.** Produce 5 per-workload `.npz` files with per-pod
trace tensors, using the per-slice source for `gpu_utilization`.

**Adapt from.** `scripts/phase4/preprocess_tier3.py`.

**Substantive change (not just paths).** GPU utilization branch
reads the per-slice CSV, filters to non-empty `pod`, groups by
`pod` to produce per-pod columns aligned with the other per-pod
metrics. Every other metric branch retains Tier 3 logic
byte-identical (paths retargeted from `extension_tier3` to
`extension_tier2`).

**Front-gap audit.** Reuse the per-pod re-audit script logic from
Tier 3 Section 2 that found 13/50 (Tier 3) and 1/50 (A16). Report
the (metric, pod) count of affected series and the max observed
gap for Tier 2. Tripwire `MAX_FRONT_GAP_S=180` unchanged.

**Deliverables.**
- `scripts/phase4/preprocess_tier2.py` (new script)
- `data/processed/tier2/{bert,gpt2,resnet152,whisper,yolo}_traces.npz`
- Per-pod counts match r for every (workload, r) cell (55 not
  expected — Tier 2 has 28 pods/workload; verify 28 = 1+2+...+7)

**Success check.** Per-workload trace tensor shape
`(28, 715, 10)`; workload_ids ordering matches `[bert, gpt2,
resnet152, whisper, yolo]`; the 10 metrics in the documented order
(pod_cpu_usage, pod_memory_bytes, pod_psi_cpu, pod_latency_avg,
pod_throughput, gpu_utilization, gpu_memory_used, gpu_memory_total,
gpu_power_watts, gpu_temperature). GPU utilization column values
in 0..100 range per pod (not 0..700 range that aggregated would
give).

**Commit.** `feat: Tier 2 preprocessing pipeline + processed data`.

### Step 2: Sanity audit and per-slice-vs-averaged diagnostic

**Objective.** Verify preprocessing correctness. Produce the
per-slice-vs-averaged comparison that quantifies what the /7
approach would have destroyed.

**Adapt from.** `scripts/analysis/tier3_sanity.py` for the audit
part.

**Audit checks.**
- Cross-tier raw per-metric mean audit on BERT (representative
  workload), extending Tier 3's Section 2 table from 2-column
  (A16, Tier 3) to 3-column (A16, Tier 2, Tier 3). Concrete
  physical interpretation for each row expected in the analysis
  writeup (`gpu_memory_total` should read Tier 2's ~11007 MiB
  per-slice constant vs. A16's ~15000-class vs. Tier 3's
  ~95000-class; `gpu_utilization` mean per pod should be
  significantly lower than Tier 3 due to MIG isolation eliminating
  cross-pod contention).
- Per-pod value distribution audit for Tier 2 `gpu_utilization` at
  r=7 across all 5 workloads: report min/max/mean/stdev per pod and
  cross-pod spread. Expect tight per-pod spread for GPU-bound
  workloads (empirically ~1.80-1.84 for BERT r=7 per Task 1).

**New diagnostic (Step 2 addition).** For each workload at r=7 (7
active slices, most demanding contention case):
1. Load per-slice per-pod values for `gpu_utilization` (real, from
   per-slice CSV, one distinct time series per pod).
2. Compute what a /7 (or sum/max_slices) approach would produce —
   sum across pods at each timestep, divide by 7, broadcast to all
   pods. By construction, cross-pod variance = 0 exactly at every
   timestep.
3. Report the destroyed per-pod signal quantitatively: (a) cross-pod
   variance at each timestep, real vs. /7 (real > 0, /7 = 0
   exactly), (b) per-pod deviation from the pod-averaged value at
   the peak-utilization timestep, and (c) the aggregate per-workload
   per-pod variance across the trace.

The diagnostic is standalone paper content — it quantifies what the
time-sliced tiers' imputation is structurally losing, regardless of
whether it materially affects downstream S36 fidelity. Result
belongs in the Tier 2 reference doc's Section 4.7 (new subsection).

**Deliverables.**
- `scripts/analysis/tier2_sanity.py`
- `scripts/analysis/tier2_per_slice_vs_averaged.py`
- Diagnostic plots (2-3 figures showing per-pod real vs. averaged)
- Written audit findings dropped into `EXTENSION_JOURNAL.md`

**Commit.** `analysis: Tier 2 sanity + per-slice vs averaged
diagnostic`.

### Step 3: Unified dataset

**Objective.** Concatenate 5 per-workload files into a single
unified training dataset with train/val split and fresh
normalization.

**Adapt from.** `scripts/phase4/unify_tier3.py`.

**Deliverables.**
- `scripts/phase4/unify_tier2.py`
- `data/processed/tier2/unified/combined_dataset.npz` — shape
  `(140, 715, 10)`, 140 = 28 x 5 workloads
- `data/processed/tier2/unified/combined_normalization.json` —
  fresh per-metric min/max/mean/stdev fit on the training partition
  only
- Train/val split via `seed=42`,
  `np.random.default_rng(42).shuffle` then ~11% val fraction:
  125/15 (matches Tier 3's 245/30 ratio)

**Success check.** Workload ordering matches `[bert, gpt2,
resnet152, whisper, yolo]` (same as A16 and Tier 3). Normalization
stats are Tier-2-specific (not copied from A16 or Tier 3).

**Commit.** `feat: Tier 2 unified dataset + normalization`.

### Step 4: Frozen retrain

**Objective.** Train S36 per-workload on Tier 2 data with
byte-identical frozen hyperparameters, producing 5 checkpoints.

**Adapt from.** `scripts/phase4/timegan/timegan_s36_tier3.py`.

**Substantive change.** None (paths only).

**Compute plan.** A16 VM (172.22.174.58), `tracegen` conda env.
Sequential per-workload tmux sessions: `s36_tier2_bert`,
`s36_tier2_gpt2`, etc. Est. 30-45 min per workload, ~4h total. All
runs `tee` to `logs/s36_tier2_<workload>_frozen.log`.

**Verification before launch.** Confirm `S27_HYPERPARAMS` values in
`timegan_s36_tier2.py` byte-match those in the frozen list (Section
4 table above). Same seed=42.

**Deliverables.**
- `scripts/phase4/timegan/timegan_s36_tier2.py`
- `models/phase4/timegan_s36_tier2/s36_{workload}_*/generator.pt`
  x 5

**Commit.** After all 5 finish and pass a smoke-test generation:
`feat: S36 Tier 2 frozen retrain (5 checkpoints)`.

### Step 5: Evaluation and validation

**Objective.** Produce per-workload VR (smooth) and Wasserstein
numbers on the val partition. Compare to A16 (thesis) and Tier 3
(frozen retrain) figures.

**Adapt from.** `scripts/phase4/evaluation/eval_s36_tier3.py`,
`scripts/phase4/postprocess_s36_tier3.py`,
`scripts/phase4/validate_s36_tier3.py`.

**Known workarounds** (from Tier 3 reference Section 5.4, carry
over): `PYTHONPATH=$(pwd)/scripts/utils` for the stale
`boundary_smoothing` import; `postprocess_s36_tier2.py` must be
adapted alongside `validate_s36_tier2.py` (the latter delegates
model/data loading to the former).

**Deliverables.**
- `scripts/phase4/evaluation/eval_s36_tier2.py`
- `scripts/phase4/postprocess_s36_tier2.py`
- `scripts/phase4/validate_s36_tier2.py`
- `outputs/phase4/timegan_s36_tier2/` — per-workload eval JSON +
  plots
- `outputs/phase4/validation/s36_tier2/` — validation report

**Prediction to falsify.** With per-slice per-pod values on
isolated MIG slices, four GPU-bound workloads (bert, gpt2,
resnet152, yolo) have low real per-pod variance (Task 1: BERT r=7
per-pod means 1.80-1.84 across 7 pods, tight cluster). Frozen S36
trained on higher-variance A16 data should overshoot VR on these
four (analogous mechanism to Tier 3 BERT `pod_memory_bytes`
small-denominator effect but at whole-workload scale). Whisper
preserves real per-pod variance via CPU contention (r=7 latency
2.86x r=1 per Task 5), so may pass frozen or read close to A16
baseline — mirror-image of Tier 3's whisper story. Rough guess:
1-2/5 pass frozen. If the outcome deviates substantially from this
prediction, understand why before writing the paper narrative.

**Commit.** `eval: S36 Tier 2 frozen retrain evaluation + validation`.

### Step 6: Hyperparameter ablation

**Objective.** Bounded local hyperparameter search around each
workload's frozen point. 4 variants x 5 workloads = 20 additional
training + eval runs.

**Adapt from.** `scripts/phase4/timegan/timegan_s36_tier3_sweep.py`
and `scripts/phase4/evaluation/eval_s36_tier3_sweep.py`. Same
`--workload`/`--vr`/`--fm`/`--model-dir`/`--variant-tag` CLIs.

**Ablation grid** (identical to Tier 3):
- `vr_low` = frozen `lambda_var_reg` - 0.2
- `vr_high` = frozen `lambda_var_reg` + 0.2
- `fm_low` = frozen `lambda_fm_stat` x 0.5
- `fm_high` = frozen `lambda_fm_stat` x 2.0

**Rationale for running unconditionally.** Even if the frozen
retrain in Step 4-5 passes 5/5 (unlikely given the per-pod variance
prediction), the ablation quantifies local sensitivity and
recoverability — content that goes into the reference doc's ablation
tables. If frozen fails on multiple workloads (predicted), the
ablation shows the "meaningful improvement" bar analogous to Tier 3
Section 4.3.

**Compute plan.** A16 VM, `tracegen`, sequential tmux sessions.
Est. ~4h total.

**Deliverables.**
- `models/phase4/timegan_s36_tier2_sweep/` (20 checkpoints)
- `outputs/phase4/timegan_s36_tier2_sweep/` (20 eval JSON + plots)

**Commit.** `feat: S36 Tier 2 ablation (20 variants)`.

### Step 7: Cross-tier synthesis and reference doc

**Objective.** Produce the three-way A16/Tier2/Tier3 synthesis and
the standalone `S36_TIER2_EXTENSION_REFERENCE.md` doc. Update
`TIER2_NOTES.md` with post-load observations.

**Deliverables.**
- `scripts/analysis/a16_vs_tier2_wasserstein.py` — mirror of the
  Tier 3 cross-tier Wasserstein script, denormalized-unit distance,
  per-workload per-metric table
- `docs/S36_TIER2_EXTENSION_REFERENCE.md` — same section structure
  as `S36_TIER3_EXTENSION_REFERENCE.md`: Motivation, Dataset, Model
  and hyperparameters, Results (headline VR + Wasserstein + ablation
  + per-workload narratives + dataset-distance appendix + per-slice
  vs averaged appendix as new Section 4.7), Reproducibility, Open
  items, Session provenance
- Extension of `S36_TIER3_EXTENSION_REFERENCE.md`'s headline VR
  table from 2-column (A16, Tier 3) to 3-column (A16, Tier 2, Tier
  3) — or a new combined summary table in the Tier 2 reference doc
  that supersedes both
- `TIER2_NOTES.md` augmentation: (a) new subsection under Section 6
  or 13 documenting the S36 retrain's per-slice source choice with
  reference to Section 1's methodological purpose and Section 6.2's
  file-design intent; (b) Section 9 "Per-workload observations"
  filled from post-load audit + eval findings
- Draft memo to Prof. Recker covering all three tiers, positioned as
  the state of the extension paper's methods/results sections

**Success criterion.** The reference doc is self-contained enough
that writing the paper's Tier 2 sections needs no other document.
Same bar as the Tier 3 reference doc met.

**Commit sequence.** Analysis script, then reference doc, then
`TIER2_NOTES.md` augmentation, then memo — one commit each.

## 6. Compute plan

All S36 training and evaluation on A16 VM (172.22.174.58),
`tracegen` conda env, PyTorch 2.9.0, CUDA 12.8, unchanged from
thesis. No H100 needed for S36 itself — Tier 2 raw data already
collected. Total S36 compute: ~4h frozen retrain (Step 4) + ~4h
ablation (Step 6) = ~8h. Preprocessing, analysis, and doc writing
are lightweight and interactive.

Long-running training goes through tmux sessions with `tee` to log
files, per project SSH resilience convention. Session naming
pattern: `s36_tier2_<workload>` for frozen, `s36_tier2_sweep_<n>`
for ablation batches.

## 7. Deliverables checklist

Step 1: `preprocess_tier2.py`, 5 processed npz files, front-gap
audit findings.

Step 2: `tier2_sanity.py`, `tier2_per_slice_vs_averaged.py`,
diagnostic plots, journal entry.

Step 3: `unify_tier2.py`, unified npz, fresh normalization JSON.

Step 4: `timegan_s36_tier2.py`, 5 frozen generator checkpoints.

Step 5: `eval_s36_tier2.py`, `postprocess_s36_tier2.py`,
`validate_s36_tier2.py`, per-workload eval JSONs, validation
report.

Step 6: `timegan_s36_tier2_sweep.py`, `eval_s36_tier2_sweep.py`, 20
ablation checkpoints, 20 eval JSONs.

Step 7: `a16_vs_tier2_wasserstein.py`,
`S36_TIER2_EXTENSION_REFERENCE.md`, `TIER2_NOTES.md` augmentation,
Recker memo draft.

## 8. Open items to resolve during execution

- Front-gap magnitude on Tier 2 — expected somewhere between A16
  (1/50) and Tier 3 (13/50) but empirically TBD; report the actual
  number in Step 1.
- Whisper's specific Tier 2 behavior — CPU contention preserved
  (Task 5: 2.86x r=1 latency at r=7), so may behave A16-like or may
  surprise; document what actually happens rather than what was
  predicted.
- If frozen fails on more than 3/5 workloads, whether the ablation
  neighborhood is wide enough to recover — Tier 3's neighborhood was
  local (vr +/- 0.2, fm x 0.5/2.0) and was sufficient for whisper's
  0.759 -> 0.994 recovery, but Tier 2 overshoots may sit further
  from the local optimum. If ablation recovers 5/5, no expansion
  needed; if 2+ still fail after ablation, note for future work.
- The per-slice vs averaged diagnostic's numerical outcome will
  itself dictate how prominently it features in the paper — if the
  destroyed variance is small in absolute terms, it stays as an
  appendix; if large, it earns paper-body treatment.

## 9. Provenance and references

Plan draft: this session (Aug 20, 2026). Reviewed and green-lit
before execution.

References:
- `docs/S36_TIER3_EXTENSION_REFERENCE.md` — template for structure
  and apples-to-apples discipline
- `docs/plans/S36_TIER3_RETRAIN_PLAN.md` — mirrored plan structure
- `TIER2_NOTES.md` — Tier 2 collection, cluster state, schema
  decisions, and open questions this plan resolves for its S36
  scope
- Diagnostic pass on A16 VM (branch `extension-h100`, executed via
  Claude Code) — Task 1 (per-slice schema audit), Task 3 (collector
  write-path audit), Task 5 (r=1 vs r=7 latency sanity check)

**End of plan.**