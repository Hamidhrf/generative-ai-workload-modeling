# S36 Tier 3 Retrain Plan

**Goal.** Retrain S36 per-workload on H100 Tier 3 data so we can put a
headline cross-tier VR table into the paper — five workload rows, two
columns (A16 Phase 1 v3 baseline / H100 Tier 3), plus a Wasserstein
appendix comparing the two real datasets directly.

**Compute.** A16 VM (172.22.174.58). S36 is small (a few hundred K
params), takes ~30–45 min per workload, ~4 hours total for five
workloads. `tracegen` conda env, unchanged. No H100 needed.

**Data status confirmed.**
- Tier 3 raw CSVs are on A16 under `data/raw/extension_tier3/`
  (pulled from `extension-h100`, 50 experiments × 23 files each, verified).
- Phase 1 v3 processed traces live at
  `data/processed/phase1_v3/{workload}_traces.npz`.
- The training file S36 actually consumes is
  `data/processed/phase4/unified/combined_dataset.npz`
  (275 traces = 55/workload × 5 workloads, r=1..10 full).
- `phase4/raw/` is a byte-identical duplicate of `phase1_v3/`.
- Empirically confirmed by Claude Code: `combined_dataset.npz` is a
  lossless concatenation of the five per-workload files, no hidden
  transformation. Missing unifier is a pure `np.concatenate` — trivial
  to rewrite.
- Sequence length: 715 timesteps for both Phase 1 v3 and Tier 3
  (verified for Tier 3 bert r=5).
- Metric handling: system GPU metrics (gpu_utilization, gpu_power_watts)
  are divided by replica count, then broadcast to every pod. Recovered
  logic in `preprocess_v3_10metric.py`, lines 169–174 of the archived
  version at commit `83eb884^`.

**Normalization policy (option a).** Fresh Tier 3 stats. Model sees
Tier 3 in its own distributional context. VR is directly comparable to
A16's VR of 1.116 as "how well does S36 fit *this* distribution".
A16↔Tier 3 dataset distance goes in the appendix as a separate
question, so the paper doesn't conflate model fit with distribution shift.

---

## Step 1 — Recover the preprocessor from git

```
git show 83eb884^:scripts/phase4/preprocess_v3_10metric.py \
  > scripts/phase4/preprocess_v3_10metric.py
```

Sanity check: file should be roughly 200–300 lines, contain
`SYSTEM_GPU_METRICS`, `POD_METRICS`, `DROP_PER_WORKLOAD`, and the
`per_pod = data[metric]["value"].values / replica_count` line at ~170.

Do not modify it yet. First run it against `phase1_v3` raw data (if the
A16 VM still has those CSVs, which it should — main branch data) and
verify it reproduces `data/processed/phase1_v3/{workload}_traces.npz`
byte-for-byte. If it does, we know the preprocessor is trustworthy and
any Tier 3 differences are dataset-driven, not code-driven.

Fallback if raw Phase 1 v3 CSVs aren't on A16 either: skip the
reproduction check and trust the recovered script based on the git
history evidence. Note this in the paper's methodology section.

## Step 2 — Adapt preprocessor for Tier 3

Two paths:

**Path A (preferred, minimal-diff).** Copy the recovered file to
`scripts/phase4/preprocess_tier3.py`. Change only the input and output
paths — everything else identical. Diff should be under 10 lines.

```
INPUT  data/raw/phase1_v3/{workload}_r{n}/       ->  data/raw/extension_tier3/{workload}_r{n}/
OUTPUT data/processed/phase1_v3/{workload}_...   ->  data/processed/tier3/{workload}_...
```

**Path B (parametrized).** Add a `--input-dir` / `--output-dir` CLI to
the recovered file, single script serves both datasets. Cleaner
long-term but bigger change, more risk of introducing a subtle
difference from the original logic. Skip unless there's a clear reason.

Go with A. Ship it, note the duplication in a comment.

**Deliverable.** Five files at
`data/processed/tier3/{workload}_traces.npz`, each shape (55, 715, 10)
with keys matching phase1_v3: `traces`, `replica_counts`, `train_idx`,
`val_idx`, `metadata`, `metric_names`. Plus normalization JSONs at
`data/processed/tier3/{workload}_normalization.json`.

**Verification.** After preprocessing runs, print for each workload:
- shape (must be (55, 715, 10))
- `sorted(set(replica_counts))` (must be [1..10])
- `sum(1 for r in replica_counts if r == k)` for k in 1..10 (must equal k)
- min/max/mean of each of the 10 metrics
- count of NaN / inf (must be 0)

## Step 3 — Write the unifier

`scripts/phase4/unify_tier3.py`, ~30 lines. Concatenates the five
per-workload files and adds `workload_ids` + `workload_names`. Match
`combined_dataset.npz` schema exactly.

```python
import numpy as np
from pathlib import Path

WORKLOADS = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']  # match A16 id ordering
IN_DIR = Path('data/processed/tier3')
OUT_DIR = Path('data/processed/tier3/unified')
OUT_DIR.mkdir(parents=True, exist_ok=True)

traces_all, replicas_all, workload_ids_all, metadata_all = [], [], [], []
metric_names = None

for wid, wl in enumerate(WORKLOADS):
    d = np.load(IN_DIR / f'{wl}_traces.npz', allow_pickle=True)
    traces_all.append(d['traces'])
    replicas_all.append(d['replica_counts'])
    workload_ids_all.append(np.full(len(d['traces']), wid, dtype=np.int32))
    metadata_all.append(d['metadata'])
    if metric_names is None:
        metric_names = d['metric_names']
    else:
        assert list(d['metric_names']) == list(metric_names), \
            f'metric_names mismatch: {wl}'

traces = np.concatenate(traces_all, axis=0)
replicas = np.concatenate(replicas_all, axis=0)
workload_ids = np.concatenate(workload_ids_all, axis=0)
metadata = np.concatenate(metadata_all, axis=0)

# Train/val split: match A16 policy. Original used random_state=42,
# ~89/11 split. Reproduce that ratio; document seed.
rng = np.random.default_rng(42)
idx = np.arange(len(traces))
rng.shuffle(idx)
n_val = int(round(len(idx) * 0.11))  # matches 245/30 A16 split ratio
val_idx = np.sort(idx[:n_val])
train_idx = np.sort(idx[n_val:])

np.savez(
    OUT_DIR / 'combined_dataset.npz',
    traces=traces,
    replica_counts=replicas,
    workload_ids=workload_ids,
    workload_names=np.array(WORKLOADS),
    metric_names=metric_names,
    metadata=metadata,
    train_idx=train_idx,
    val_idx=val_idx,
)
print(f'wrote {OUT_DIR / "combined_dataset.npz"}: {traces.shape}')
```

Two callouts:
1. Workload id ordering `['bert','gpt2','resnet152','whisper','yolo']` matches
   the A16 `workload_names` array Claude Code verified.
2. The 89/11 split is the A16 default. Same seed, same ratio, so any
   VR difference is data-driven not split-driven.

**Verification.** Print unified shape (must be (275, 715, 10)), and
per-workload counts (must be 55 each).

## Step 4 — Retrain S36 per workload

Copy `scripts/phase4/timegan/timegan_s36.py` to
`scripts/phase4/timegan/timegan_s36_tier3.py`. Two changes:

1. Data path: point `DATA_PATH` at
   `data/processed/tier3/unified/combined_dataset.npz`.
2. Output dir: point at `models/phase4/timegan_s36_tier3/`.

Everything else — architecture, per-workload hyperparams, warmup and
adversarial epoch counts, seeds — stays as-is. Same recipe from the
thesis. This is the whole point of the "apples-to-apples" framing.

**Per-workload hyperparams (from thesis, do not change).**
- BERT: vr05, fm15
- GPT-2: vr03, fm20
- ResNet-152: vr04, fm10
- Whisper: vr05, fm05
- YOLO: vr03, fm12

**Runs.** Five, sequentially. Each writes to
`models/phase4/timegan_s36_tier3/s36_<workload>_<hparams>/generator.pt`.

**Recommendation.** Run each inside `tmux` and pipe stdout to a log
file — no more Claude Code approval-click gaps mid-training. One-liner
template:

```
tmux new -d -s s36_tier3_bert \
  "conda activate tracegen && \
   python scripts/phase4/timegan/timegan_s36_tier3.py \
     --workload bert --vr 0.05 --fm 15 \
     2>&1 | tee logs/s36_tier3_bert.log"
```

Do them one at a time (not parallel) to keep GPU contention out of the
timing signal. Roughly 45 min each. Whisper first? BERT first? Doesn't
matter — order doesn't affect any result. My call: BERT first because
it's the fastest workload, so if anything is broken in the pipeline
we see it in 30 min not 45.

## Step 5 — Evaluate

Copy `scripts/phase4/eval_s36.py` to `eval_s36_tier3.py`. Point at Tier
3 unified data and Tier 3 model directory. Two outputs:

1. `outputs/phase4/timegan_s36_tier3/vr_per_workload.json` — VR per
   workload, mean, and pass/fail vs threshold 0.8.
2. `outputs/phase4/timegan_s36_tier3/wasserstein_per_workload.json` —
   Wasserstein distance per workload (and optionally per metric).

Verify by inspection: any VR outside roughly [0.5, 2.0] is a red flag
worth investigating before trusting the number.

## Step 6 — Headline cross-tier table

Trivial once steps 1–5 done. Assemble from thesis numbers + Step 5
output:

```
Workload    | A16 VR (thesis) | H100 Tier 3 VR (new)
BERT        | 1.022           | ?
GPT-2       | 1.137           | ?
ResNet-152  | 0.985           | ?
Whisper     | 1.397           | ?
YOLO        | 1.039           | ?
Mean        | 1.116           | ?
```

Save to `outputs/phase4/timegan_s36_tier3/cross_tier_vr_table.md`.
Same table format for Wasserstein.

## Step 7 — Appendix: dataset distance (A16 vs Tier 3 real)

Separate small script, `scripts/analysis/a16_vs_tier3_wasserstein.py`.
For each (workload, metric):

1. Load real A16 traces from
   `data/processed/phase1_v3/{workload}_traces.npz` — shape (55, 715, 10).
2. Load real Tier 3 traces from
   `data/processed/tier3/{workload}_traces.npz` — same shape.
3. Flatten each to 1D per metric (55 × 715 = 39325 samples per metric per tier).
4. `scipy.stats.wasserstein_distance(a16_metric, tier3_metric)`.
5. Assemble a (5 workloads × 10 metrics) heatmap or table.

Save to `outputs/phase4/timegan_s36_tier3/dataset_distance_a16_tier3.{png,json}`.

Purpose: separates "how much did switching hardware move the
distribution" from "how well does S36 fit each dataset". Reviewer will
absolutely ask. Cheap to compute, defensible answer ready.

## Step 8 — Commit and memo

Two commits on `extension-h100`:

**Commit A** — pipeline + models:
- `scripts/phase4/preprocess_v3_10metric.py` (recovered from git)
- `scripts/phase4/preprocess_tier3.py`
- `scripts/phase4/unify_tier3.py`
- `scripts/phase4/timegan/timegan_s36_tier3.py`
- `scripts/phase4/eval_s36_tier3.py`
- `data/processed/tier3/` (five .npz + normalization JSONs + unified)
- `models/phase4/timegan_s36_tier3/` (five generator.pt files)

**Commit B** — analysis and headline outputs:
- `scripts/analysis/a16_vs_tier3_wasserstein.py`
- `outputs/phase4/timegan_s36_tier3/*.json`
- `outputs/phase4/timegan_s36_tier3/*.md`
- `outputs/phase4/timegan_s36_tier3/*.png`

**Memo to Prof. Recker.** After both commits land, one memo covering:
- what was retrained and why
- cross-tier VR table (headline result)
- Whisper A16→Tier 3 finding: same 10-slot config, 10 pods, r=10 —
  A16 crashed at PSI 0.63; Tier 3 completed cleanly at PSI 0.317 with
  ~2.5x lower latency
- dataset distance appendix
- what's next (Tier 2 retrain, Kwok simulation)

---

## Risk register

**R1: Recovered preprocessor doesn't reproduce phase1_v3 outputs
byte-for-byte.** Likely if there was a "fixed" version for Whisper
r=10 that was never committed. Mitigation: check if the recovered
script handles Whisper r=10's partial pod_9 correctly. If not, patch
minimally, document the patch, and re-run.

**R2: Tier 3 preprocessing outputs unexpectedly different shapes.**
Should be caught by Step 2's verification prints. If it happens,
almost certainly a CSV schema drift between A16 and H100 Tier 3
collectors — the `application` → `app` label rename that
`load_experiment.py` handles is the obvious suspect. Fix in the
preprocessor.

**R3: VR values on Tier 3 collapse (e.g. mean < 0.5).** Would mean
S36 hyperparams don't transfer to the Tier 3 distribution. Not
catastrophic — it's a real finding in its own right. But before
concluding that, sanity-check: are all 5 workloads' VR bad, or just
some? If mixed, likely a single-workload preprocessing bug. If
uniform, likely a real transferability finding worth writing about.

**R4: Tier 2 retrain gets deprioritized.** Tier 3 is the direct A16
comparison; Tier 2 (MIG) is a different sharing mode with different
`gpu_utilization` semantics (`DCGM_FI_PROF_GR_ENGINE_ACTIVE × 100`,
summed across active slices — up to 700 at r=7). That difference
alone is worth a paper subsection. Don't skip Tier 2, but Tier 3
first.

## Out of scope for this plan

- Tier 2 retrain (separate plan, similar shape, with the DCGM proxy
  documented).
- Tier 1 retrain (r=1 only, single point — not enough for VR).
- Latency percentile recovery (low priority per NEXT_STEPS, S36
  doesn't train on it).
- Kwok trace generation (depends on this retrain being done).
- Jan's fidelity analysis (parallel workstream, uses same artifacts).
- Cross-tier multi-tier training (a unified model on A16+Tier 3+Tier 2
  is a future direction, not this plan).

## Timeline estimate

- Steps 1–3 (preprocess + unify + verify): 2–3 hours.
- Step 4 (five retrains): 4 hours wall clock, sequential.
- Steps 5–7 (eval + tables + appendix): 2 hours.
- Step 8 (commits + memo): 1 hour.

**Total: one working day if nothing surprises.** Two days is realistic
budget with debugging headroom.