# S36 Tier 3 Retrain — Step 5 Report

Evaluated the five Tier 3 S36 checkpoints from Step 4. No commits.

## Setup — three discrepancies from the plan's assumptions

**1. Script location.** `scripts/phase4/eval_s36.py` doesn't exist —
the real file is `scripts/phase4/evaluation/eval_s36.py`. Adapted
that one.

**2. Wasserstein is not computed by this script at all.** The plan
and this task both assumed `eval_s36.py` produces VR *and*
Wasserstein. It doesn't — `eval_s36.py` only computes Variance Ratio
(`compute_vr_per_metric`, raw and smoothed). There is no Wasserstein
logic anywhere in the file. Traced where the thesis's Wasserstein
numbers actually came from: a completely different, 667-line script,
`scripts/phase4/validate_s36.py`, writing
`outputs/phase4/validation/s36/validation_report.json`. Confirmed by
recomputing bert's mean Wasserstein directly from that file's raw
per-replica, per-metric values: **1.5433** ≈ thesis's **1.543**, exact
match. That script's `"post_processing"` field literally says `"cosine
blend (w=20) + adaptive filter + memory reconstruction"` — a
substantially heavier pipeline with its own parameters, not a
`eval_s36.py`-shaped "just retarget 4 paths" adaptation. Given the
explicit "no logic changes, no anything" discipline for this step, I
did not adapt `validate_s36.py` or invent a standalone Wasserstein
computation. **The Wasserstein table below is left empty pending your
decision** (see "What's needed for Wasserstein" at the end).

**3. Pre-existing import bug, unrelated to Tier 3.** `eval_s36.py`
does `sys.path.insert(0, str(Path(__file__).parent.parent /
"utils"))` expecting `scripts/phase4/utils/boundary_smoothing.py`.
That directory doesn't exist — the real file is at
`scripts/utils/boundary_smoothing.py`. This would break the *original*
`eval_s36.py` too if run fresh today (the existing A16
`s36_eval_results.json` must predate a later `scripts/` reorg). Not
something introduced by the Tier 3 adaptation, and not one of the four
paths I was told to retarget, so I didn't edit the file for it —
worked around it purely at invocation time with
`PYTHONPATH=$(pwd)/scripts/utils`, zero code diff impact. Confirmed
`boundary_smoothing.py` itself is a generic numpy utility with no
A16/Tier3-specific assumptions, so this is a safe, logic-neutral
workaround.

## Diff, `eval_s36.py` → `eval_s36_tier3.py`

Header comment + exactly 3 substantive changes (data/norm paths,
5 model_paths dict values, out_dir) — no logic, threshold, or
smoothing changes:

```diff
1a2,6
> Adapted from eval_s36.py for H100 Tier 3 data. Only
> DATA_COMBINED, NORM_COMBINED, the model_paths dict values,
> and out_dir changed. No logic, threshold, or smoothing
> changes -- frozen per the S36 Tier 3 retrain plan.
29,31c34,36
< # Paths
< DATA_COMBINED = Path("data/processed/phase4/unified/combined_dataset.npz")
< NORM_COMBINED = Path("data/processed/phase4/unified/combined_normalization.json")
---
> # Paths (Tier 3 H100)
> DATA_COMBINED = Path("data/processed/tier3/unified/combined_dataset.npz")
> NORM_COMBINED = Path("data/processed/tier3/unified/combined_normalization.json")
201,205c206,210
<         "bert": "models/phase4/timegan_s36/s36_bert_vr05_fm15/generator.pt",
<         ... (4 more, timegan_s36/)
---
>         "bert": "models/phase4/timegan_s36_tier3/s36_bert_vr05_fm15/generator.pt",
>         ... (4 more, timegan_s36_tier3/)
325c330
<     out_dir = Path("outputs/phase4/timegan_s36")
---
>     out_dir = Path("outputs/phase4/timegan_s36_tier3")
```

`load_s34_results()` still points at the A16 S34 baseline
(`outputs/phase4/timegan_s34/s34_eval_results.json`) — left untouched,
it's just an extra informational "S36 vs S34" console comparison the
script prints; it doesn't affect the saved `s36_eval_results.json`
values used below, and removing/redirecting it would be a logic
change outside the requested scope.

## Run

```
PYTHONPATH=$(pwd)/scripts/utils python scripts/phase4/evaluation/eval_s36_tier3.py
```

All 5 workloads evaluated cleanly, `EXIT_0`. Output:
`outputs/phase4/timegan_s36_tier3/s36_eval_results.json`. (The
`s36_summary.json` also in that directory is leftover from Step 4's
training, not an eval output — ignore it.)

## Cross-tier VR comparison

Using `vr_smooth` (the smoothed VR — confirmed this is the number
that reproduces the thesis's A16 figures exactly, e.g. bert 1.022,
mean 1.116).

| Workload | A16 VR (thesis) | Tier 3 VR (new) | delta | Tier 3 pass? |
|---|---|---|---|---|
| bert | 1.022 | 1.885 | +0.863 | PASS |
| gpt2 | 1.137 | 0.920 | -0.217 | PASS |
| resnet152 | 0.985 | 1.064 | +0.079 | PASS |
| whisper | 1.397 | 0.759 | -0.638 | **FAIL** |
| yolo | 1.039 | 1.069 | +0.030 | PASS |
| **Mean** | **1.116** | **1.140** | **+0.024** | **4/5** |

(`vr_raw`, unsmoothed, for reference: bert 1.931, gpt2 0.942,
resnet152 1.069, whisper 0.776, yolo 1.075 — same pass/fail pattern.)

## Cross-tier Wasserstein comparison — UPDATE (Step 5 continuation)

Computed via `validate_s36_tier3.py`, adapted from `validate_s36.py`
(see new section below for the adaptation writeup).

| Workload | A16 W (thesis) | Tier 3 W (new) | delta | ratio (Tier3/A16) |
|---|---|---|---|---|
| bert | 1.543 | 11.741224 | +10.198 | 7.61x |
| gpt2 | 4.736 | 21.903472 | +17.167 | 4.62x |
| resnet152 | 2.973 | 14.565862 | +11.593 | 4.90x |
| whisper | 5.515 | 25.972153 | +20.457 | 4.71x |
| yolo | 1.197 | 8.628075 | +7.431 | 7.21x |
| **Mean** | **3.193** | **16.562157** | **+13.369** | **5.19x** |

No single workload individually crosses the "10x or more" flag you
asked me to watch for (closest: bert at 7.61x, yolo at 7.21x), but
**every one of the 5 workloads is elevated by roughly the same 4.6x–7.6x
factor** — a strikingly uniform pattern across workloads that otherwise
have very different VR behavior (recall whisper failed VR while bert's
VR is unusually high). Reporting the number and the pattern per your
instruction, not interpreting it as good/bad — but flagging the
uniformity itself as the notable thing, since a metric-scale
explanation (below) fits a uniform inflation better than a genuine
per-workload transferability story would.

**Metric-comparability caveat, not a verdict:** unlike VR (a ratio,
scale-invariant), Wasserstein distance here is computed in raw
physical units per metric, then pooled across all 7 metrics into one
mean (`mean_wasserstein_all` = flat average of every individual
per-replica, per-metric Wasserstein value, no per-metric
normalization). See the per-metric appendix below — **`Pod Memory` and
`GPU Power` dominate the pooled mean for every workload** (tens to
hundreds vs. sub-1 for CPU/PSI/Latency-type metrics), simply because
they're measured in larger absolute units. Step 2's raw-mean audit
already found Tier 3's `gpu_memory_total`/`gpu_power_watts` running
several times higher than A16's in absolute terms (H100 vs A16
hardware) — a metric with mechanically larger raw magnitude will
mechanically produce a larger raw-unit Wasserstein distance even at
equivalent *relative* fit quality. Whether the ~5x uniform inflation
here is fully explained by that scale effect or partly reflects real
fit differences isn't something this script's output can distinguish
— flagging as a caveat for interpretation, not resolving it.

## Per-metric VR breakdown (smoothed) — anything collapsing or exploding?

| Workload | pod_cpu_usage | pod_memory_bytes | pod_psi_cpu | pod_latency_avg | pod_throughput | gpu_utilization | gpu_power_watts |
|---|---|---|---|---|---|---|---|
| bert | 0.904 | **7.259** | 1.265 | 0.875 | 0.773 | 0.919 | 1.203 |
| gpt2 | 1.254 | 0.528 | 0.607 | 1.021 | 0.986 | 1.333 | 0.709 |
| resnet152 | 1.173 | 1.525 | **0.184** | 1.132 | 0.702 | 0.912 | 1.822 |
| whisper | 0.798 | 0.354 | 0.833 | 0.885 | 0.746 | 1.029 | 0.667 |
| yolo | 0.834 | 0.432 | 1.064 | 1.037 | 0.874 | 0.639 | **2.605** |

Bold = outside the workload-level [0.3, 3.0] sanity band, called out
per the "flag if a specific metric collapses or explodes" guidance:

- **bert `pod_memory_bytes` VR = 7.259.** Not a collapse — the
  opposite: synthetic memory variance is ~7x real variance for this
  workload. This is the single most extreme per-metric number in the
  whole run. Worth a look before trusting bert's headline VR=1.885,
  since a single metric this far off can pull the workload mean up
  substantially (7 metrics averaged; this one alone likely accounts
  for a large share of why bert's mean VR is so much higher than
  A16's 1.022).
- **resnet152 `pod_psi_cpu` VR = 0.184.** Close to what the guidance
  called "near 0" (not literally near-zero, but clearly the lowest
  value in the whole table and below the 0.3 sanity floor) —
  synthetic PSI variance is suppressed relative to real. Same
  metric (`pod_psi_cpu`) is near-constant-zero in most raw data
  (see Step 1/2 audits: A16 pod_psi_cpu means were ~0.0001), so a low
  VR here may just reflect that both real and synthetic values are
  tiny and noisy near the metric's floor — flagging rather than
  interpreting.

No workload-level VR fell outside [0.3, 3.0] (all 5 are between 0.759
and 1.885). Per your instruction, not characterizing any of this as
"good" or "bad" — just surfacing it for your interpretation.

## Wasserstein adaptation writeup (Step 5 continuation)

`validate_s36.py` was adapted, but the real A16-specific paths weren't
all in that file. Read the full dependency chain first:
`validate_s36.py` imports `generate_postprocessed_traces`,
`load_generator`, `load_real_data`, `load_normalization_params`,
`denormalize_trace`, `WORKLOADS`, `ALL_METRICS`, `TRAINED_INDICES`,
`TRAINED_NAMES` from `postprocess_s36.py` — and **that's** where the
`MODEL_PATHS` dict and every `data_dir` default actually live. Traced
every call site: `get_synthetic_stats()` and the synthetic-generation
line inside `compute_wasserstein_distances()` both call
`generate_postprocessed_traces(wname, r, n_samples=n_samples,
device=device)` **without** passing `data_dir` — so even after
retargeting `validate_s36.py`'s own `data_dir` defaults, the synthetic
side of the pipeline would have silently kept loading A16 models and
A16 real data via `postprocess_s36.py`'s own hardcoded defaults. Fixing
this required adapting both files.

**`postprocess_s36.py` → `postprocess_s36_tier3.py`** (7 substantive
changes + header comment): `MODEL_PATHS` dict (5 entries, all
`timegan_s36/` → `timegan_s36_tier3/`) and all 6 `data_dir` default
parameters across `load_normalization_params`, `load_real_data`,
`compute_memory_stats`, `compute_dropped_metric_stats`,
`generate_postprocessed_traces`, `plot_before_after` (all
`data/processed/phase4/unified` → `data/processed/tier3/unified`).
Confirmed correct: `denormalize_trace` here properly uses `min`/`max`
(unlike `eval_s36.py`'s broken `mean`/`std` version from the original
Step 5 report — no bug to work around in this pipeline). The two CLI
argparse defaults for `postprocess_s36.py`'s own standalone
`--output-dir`/`--plot-dir` were left untouched — dead code in our
execution path since we only import this module, never run its CLI
directly.

**`validate_s36.py` → `validate_s36_tier3.py`** (4 substantive changes
+ header comment): the `from postprocess_s36 import (...)` line
retargeted to `postprocess_s36_tier3`, `get_real_stats`'s and
`compute_wasserstein_distances`'s `data_dir` defaults, and the
`--output-dir` CLI default (`outputs/phase4/validation/s36` →
`outputs/phase4/validation/s36_tier3`).

Post-processing logic — cosine blend (`window=20`), adaptive
per-metric filtering, pod-memory reconstruction, dropped-metric
reconstruction, `REPLICA_COUNTS_PER_WORKLOAD` (which validated replicas
to check), `n_samples=5`, extrapolation replica counts — all
byte-identical to the original in both files. Full diffs (both files,
in full) were reviewed before running; nothing beyond paths, imports,
and header comments changed.

One frozen-parameter note, not a change: `REPLICA_COUNTS_PER_WORKLOAD`
excludes r=4,7,9 (r=4,7,8,9 for whisper) — an A16-specific curation
choice preserved as-is. Tier 3's raw data actually has full r=1–10 for
every workload (confirmed in Steps 2–3), so this run validates against
fewer replicas than Tier 3 data would support, deliberately mirroring
A16's original validated-replica selection for comparability.

**Run:** `python scripts/phase4/validate_s36_tier3.py` (no CLI
overrides — full default recipe: 5 samples/replica, extrapolation zone
r=15,20,30,50 included). `EXIT_0`, ~few minutes. Zero clamping
violations for all 5 workloads. Output:
`outputs/phase4/validation/s36_tier3/validation_report.json` (+ 11
plot PNGs, not reviewed in detail here — scaling curves and Wasserstein
heatmaps per workload, available for visual inspection if wanted).

## Per-metric Wasserstein appendix (Tier 3, mean across validated replicas)

| Workload | CPU Usage | CPU Pressure (PSI) | Latency (Avg) | Throughput | Pod Memory (MB) | GPU Utilization | GPU Power |
|---|---|---|---|---|---|---|---|
| bert | 0.0089 | 0.0001 | 0.0001 | 0.6234 | **73.2468** | 0.3163 | 7.9930 |
| gpt2 | 0.0776 | 0.0000 | 0.0789 | 0.1397 | **134.3822** | 5.6733 | 12.9725 |
| resnet152 | 0.0101 | 0.0000 | 0.0001 | 0.7073 | **91.9870** | 0.4763 | 8.7801 |
| whisper | 0.2288 | 0.0154 | 0.0339 | 0.2985 | **167.0446** | 1.8716 | 12.3121 |
| yolo | 0.0052 | 0.0000 | 0.0015 | 0.5164 | **42.5736** | 0.2802 | 17.0196 |

Bold = `Pod Memory`, consistently the dominant contributor to every
workload's pooled mean by one to two orders of magnitude over every
other metric — this is the scale-dominance effect described above, not
a per-workload anomaly. `GPU Power` is the second-largest contributor
across all 5. No metric shows a value near zero that looks like a
computation failure (CPU Pressure/Latency are near-zero because those
metrics are themselves near-zero/fractional in raw units for most
workloads — consistent with what Step 1/2's raw-data audits already
found, not new). No per-replica outlier was extreme enough to flag
beyond what's already visible in the per-workload min/max spread
computed above (e.g. bert Pod Memory ranges 0.07–144.6 across its 7
validated replicas — wide, but not an isolated single-replica spike;
consistent with memory reconstruction using a per-workload real-data
level that shifts somewhat by replica count).

## Status

VR evaluation: **4/5 pass** the VR ≥ 0.8 threshold (whisper fails,
vr_smooth=0.759). Wasserstein evaluation: **complete for all 5
workloads**, mean Tier 3 W = 16.562 vs. A16's 3.193 — every workload
elevated ~4.6x–7.6x, uniformly, most plausibly a metric-scale effect
(Pod Memory/GPU Power dominate the raw-unit pooled mean, and Tier 3's
absolute GPU metric magnitudes are known-larger from Step 2's audit)
rather than 5 independent per-workload fit failures. Whisper's VR
failure (0.759 < 0.8) is the same "expected, not a bug" finding framed
in the task — its Wasserstein (25.972) is also the single highest of
the 5 workloads, consistent with the same underlying signal surfacing
through both metrics. Nothing committed.

---

# Full hyperparameter ablation on Tier 3 (Frame C)

20-run (lambda_var_reg, lambda_fm_stat) sweep, 4 one-off neighbors per
workload around the frozen thesis point, all 5 workloads. Goal:
does local hyperparameter tuning on Tier 3 hardware improve fit over
the frozen A16 recipe? No commits.

## Setup

**Grid** (sanity-printed before any run started): constant vr step
(±0.2, since vr values all sit in [0.3, 0.5]), multiplicative fm step
(0.5x / 2.0x, since fm spans a wider absolute range 0.5–2.0 across
workloads). All 20 configs matched the task's reference values
exactly. One tag-encoding note, not a bug: bert's `fm_low` (fm=0.75)
truncates to tag `fm07` under the existing `int(fm*10):02d` formula
(0.75×10=7.5 exactly in floating point; Python's `int()` truncates
toward zero) — kept the existing truncation convention for consistency
with prior tags rather than inventing a rounding rule; the actual
lambda_fm_stat value used in training was the full 0.75, only the
directory name loses precision.

**`timegan_s36_tier3_sweep.py`** (copied from `timegan_s36_tier3.py`,
62 diff lines incl. header comment): added `--workload`/`--vr`/`--fm`
CLI that builds a local hyperparameter dict overriding
`S27_HYPERPARAMS[workload]` for a single run, restricts training to
that one workload, and retargets output to
`models/phase4/timegan_s36_tier3_sweep/` /
`outputs/phase4/timegan_s36_tier3_sweep/` so sweep checkpoints never
collide with Step 4's frozen-hyperparam ones. Architecture, seeds,
warmup/adversarial epoch counts (20/150), and data paths untouched.

**`eval_s36_tier3_sweep.py`** (copied from `eval_s36_tier3.py`, 78
diff lines incl. header comment): added `--workload`/`--model-dir`/
`--variant-tag` CLI to evaluate one checkpoint directly (bypassing the
frozen `model_paths` dict), writing its own
`outputs/phase4/timegan_s36_tier3_sweep/eval_<workload>_<variant>.json`
per run instead of overwriting the shared results file. VR
computation, denormalization, smoothing (window=5), and the 0.8
threshold are byte-identical to `eval_s36_tier3.py`. Included the
`PYTHONPATH` workaround for the `boundary_smoothing` import bug
identified in Step 5.

## Run

20/20 training runs completed, **zero failures**, interleaved order
(vr_low ×5, vr_high ×5, fm_low ×5, fm_high ×5) as specified. Total
wall clock ≈ 15,855s ≈ **4.40 hours** (task estimate: ~4.3h). Whisper
runs were consistently faster (~460s each, `n_disc_steps=1`) than the
other four (~865–890s each, `n_disc_steps=2`) — same pattern observed
in Step 4. All 20 checkpoints verified present before evaluation
started. 20/20 evaluations completed, zero failures, all against the
same Tier 3 validation set used in Step 5.

## Per-workload tables

VR (smooth) is the threshold metric (same convention as Step 5 — this
is what reproduces the thesis's A16 figures exactly). **Bold** = best
variant per workload, defined as closest to 1.0 (not highest).

**Workload: bert** (baseline: vr=0.5, fm=1.5)

| Variant | vr | fm | VR (smooth) | VR (raw) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|---|
| **baseline (Step 4)** | 0.5 | 1.5 | **1.885** | 1.931 | PASS | **0.885** |
| vr_low | 0.3 | 1.5 | 1.899 | 1.933 | PASS | 0.899 |
| vr_high | 0.7 | 1.5 | 1.939 | 1.961 | PASS | 0.939 |
| fm_low | 0.5 | 0.75 | 2.284 | 2.306 | PASS | 1.284 |
| fm_high | 0.5 | 3.0 | 1.922 | 2.037 | PASS | 0.922 |

Bert is the one workload where **the frozen baseline is already the
best of the 5** — every swept neighbor moves further from 1.0, not
closer. (Recall from Step 5: bert's `pod_memory_bytes` VR was already
a large outlier at baseline — consistent with this being a workload
where the model already over-fits variance, and nudging either
hyperparameter in any direction pushes it further.)

**Workload: gpt2** (baseline: vr=0.3, fm=2.0)

| Variant | vr | fm | VR (smooth) | VR (raw) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|---|
| baseline (Step 4) | 0.3 | 2.0 | 0.920 | 0.942 | PASS | 0.080 |
| vr_low | 0.1 | 2.0 | 1.007 | 1.024 | PASS | 0.007 |
| vr_high | 0.5 | 2.0 | 1.136 | 1.161 | PASS | 0.136 |
| **fm_low** | 0.3 | 1.0 | **1.000** | 1.015 | PASS | **0.0003** |
| fm_high | 0.3 | 4.0 | 0.889 | 0.912 | PASS | 0.111 |

**Workload: resnet152** (baseline: vr=0.4, fm=1.0)

| Variant | vr | fm | VR (smooth) | VR (raw) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|---|
| baseline (Step 4) | 0.4 | 1.0 | 1.064 | 1.069 | PASS | 0.064 |
| vr_low | 0.2 | 1.0 | 1.108 | 1.122 | PASS | 0.108 |
| vr_high | 0.6 | 1.0 | 1.359 | 1.365 | PASS | 0.359 |
| fm_low | 0.4 | 0.5 | 0.786 | 0.810 | **FAIL** | 0.214 |
| **fm_high** | 0.4 | 2.0 | **0.940** | 0.961 | PASS | **0.060** |

resnet152's `fm_low` variant is the only case in the whole sweep where
a variant **flips a passing baseline to failing** (0.786 < 0.8).

**Workload: whisper** (baseline: vr=0.5, fm=0.5)

| Variant | vr | fm | VR (smooth) | VR (raw) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|---|
| baseline (Step 4) | 0.5 | 0.5 | 0.759 | 0.776 | **FAIL** | 0.241 |
| vr_low | 0.3 | 0.5 | 0.736 | 0.744 | **FAIL** | 0.264 |
| vr_high | 0.7 | 0.5 | 0.931 | 0.946 | PASS | 0.069 |
| **fm_low** | 0.5 | 0.3 | **0.994** | 1.025 | PASS | **0.006** |
| fm_high | 0.5 | 1.0 | 0.888 | 0.905 | PASS | 0.112 |

**Workload: yolo** (baseline: vr=0.3, fm=1.2)

| Variant | vr | fm | VR (smooth) | VR (raw) | Pass 0.8? | \|Δ from 1.0\| |
|---|---|---|---|---|---|---|
| baseline (Step 4) | 0.3 | 1.2 | 1.069 | 1.075 | PASS | 0.069 |
| vr_low | 0.1 | 1.2 | 1.077 | 1.096 | PASS | 0.077 |
| vr_high | 0.5 | 1.2 | 0.874 | 0.918 | PASS | 0.126 |
| **fm_low** | 0.3 | 0.6 | **1.011** | 1.028 | PASS | **0.011** |
| fm_high | 0.3 | 2.4 | 0.917 | 0.927 | PASS | 0.083 |

## Summary

**Best variant per workload, and whether the margin is "meaningful"**
(task's bar: baseline `|Δ from 1.0|` minus best-variant `|Δ from 1.0|`
> 0.1):

| Workload | Best variant | Best \|Δ\| | Baseline \|Δ\| | Improvement | Meaningful (>0.1)? |
|---|---|---|---|---|---|
| bert | baseline itself | 0.885 | 0.885 | 0.000 | No — nothing beats baseline |
| gpt2 | fm_low | 0.0003 | 0.080 | 0.080 | No — just under the bar |
| resnet152 | fm_high | 0.060 | 0.064 | 0.004 | No — negligible |
| whisper | fm_low | 0.006 | 0.241 | **0.235** | **Yes** |
| yolo | fm_low | 0.011 | 0.069 | 0.058 | No — modest |

**Aggregate:**
- **4 of 5 workloads** got at least somewhat closer to 1.0 with local
  tuning (gpt2, resnet152, whisper, yolo); **bert did not** — its
  frozen baseline is already the best point in its own local
  neighborhood.
- By the strict ">0.1 meaningful improvement" bar, **only whisper**
  clears it, and by a wide margin (0.235). Every other workload's best
  variant is a marginal or negligible improvement over frozen.
- **Pass/fail status changes:** comparing baseline to *best variant*,
  exactly **1 of 5 workloads changes status** — whisper flips FAIL →
  PASS. (resnet152 has a variant, `fm_low`, that flips PASS → FAIL,
  but that's not its best variant — its best variant, `fm_high`, stays
  PASS, same as baseline. So resnet152 doesn't count as a status
  change under best-variant framing, but it does show the workload is
  sensitive to fm in that direction.)
- **`fm_low` is the best variant for 4 of 5 workloads** (gpt2,
  whisper, yolo, and would be resnet152's too if resnet152's fm_low
  weren't the one variant that fails — resnet152's actual best is
  fm_high instead). Bert is the outlier where fm_low is its *worst*
  variant (1.284 from 1.0, furthest of all 4). No single direction
  (vr up/down, fm up/down) is uniformly beneficial across workloads —
  the direction that helps is workload-specific.

**Whisper, explicitly:** yes, multiple variants lift it above 0.8 —
`vr_high` (0.931), `fm_low` (0.994), and `fm_high` (0.888) all pass;
only `vr_low` (0.736) stays failing, and is in fact worse than
baseline. Whisper is also the workload with by far the largest
baseline-to-best improvement in the whole sweep (0.235), and its best
variant (`fm_low`, vr=0.5/fm=0.3) lands almost exactly on VR=1.0
(0.994) — the single closest fit of all 25 (baseline + 20 sweep) data
points in this report. This is consistent with (not proof of) the
idea that the frozen A16 `lambda_fm_stat=0.5` for whisper may be
tuned to A16's own whisper distribution rather than being a universal
optimum — Tier 3's whisper data wants noticeably less feature-matching
pressure (`fm=0.3`) to fit well. Reporting the finding; whether it
changes the "frozen thesis hyperparams" framing for the paper is your
call, not decided here.

Nothing committed.
