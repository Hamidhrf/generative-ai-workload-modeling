# S36 Tier 3 Retrain — Step 6 Headline Tables

Assembled entirely from Steps 4/5 + ablation outputs already on disk.
No new training or evaluation. No commits.

## Table 1: Cross-tier VR comparison (Option Y — with best-tuned column)

| Workload | A16 VR (thesis) | Tier 3 VR (frozen) | Tier 3 VR (best-tuned) | Best variant | Pass 0.8 (frozen / best) |
|---|---|---|---|---|---|
| bert | 1.022 | 1.885 | 1.885 | baseline | PASS / PASS |
| gpt2 | 1.137 | 0.920 | 1.000 | fm_low | PASS / PASS |
| resnet152 | 0.985 | 1.064 | 0.940 | fm_high | PASS / PASS |
| whisper | 1.397 | 0.759 | 0.994 | fm_low | FAIL / PASS |
| yolo | 1.039 | 1.069 | 1.011 | fm_low | PASS / PASS |
| **Mean** | **1.116** | **1.140** | **1.166** | — | **4/5 / 5/5** |

*(Source: `S36_TIER3_STEP5_REPORT.md`, "Full hyperparameter ablation on
Tier 3" section, and its underlying `outputs/phase4/timegan_s36_tier3/s36_eval_results.json`
+ `outputs/phase4/timegan_s36_tier3_sweep/eval_*.json`.)*

**Interpretation.** The frozen thesis recipe already passes 4/5
workloads on Tier 3, with whisper as the single failure (0.759).
Restricting local hyperparameter search to a minimal one-off-neighbor
grid (4 variants per workload, no broader search) recovers whisper to
a pass (0.994) and leaves the other four workloads' pass status
unchanged — three of them get marginally closer to 1.0 in the
process, one (bert) does not improve at all under any tested variant.
This is a narrow, targeted result: it shows the frozen recipe is not
uniformly optimal across hardware, and that a small amount of local
tuning closes the one gap that existed — it does not establish that
Tier 3 requires systematically different hyperparameters in general
(the sweep never explored beyond ±1 step from the frozen point).

## Table 2: Cross-tier Wasserstein comparison (frozen only — sweep did not include W)

| Workload | A16 W (thesis) | Tier 3 W (frozen) | Delta | Ratio |
|---|---|---|---|---|
| bert | 1.543 | 11.741 | +10.198 | 7.61x |
| gpt2 | 4.736 | 21.903 | +17.167 | 4.62x |
| resnet152 | 2.973 | 14.566 | +11.593 | 4.90x |
| whisper | 5.515 | 25.972 | +20.457 | 4.71x |
| yolo | 1.197 | 8.628 | +7.431 | 7.21x |
| **Mean** | **3.193** | **16.562** | **+13.369** | **5.19x** |

*(Source: `S36_TIER3_STEP5_REPORT.md`, "Cross-tier Wasserstein
comparison" section, and `outputs/phase4/validation/s36_tier3/validation_report.json`.)*

**Scale-artifact caveat (verbatim from Step 5's Wasserstein section):**
unlike VR (a ratio, scale-invariant), Wasserstein distance here is
computed in raw physical units per metric, then pooled across all 7
metrics into one mean (`mean_wasserstein_all` = flat average of every
individual per-replica, per-metric Wasserstein value, no per-metric
normalization). `Pod Memory` and `GPU Power` dominate the pooled mean
for every workload (tens to hundreds vs. sub-1 for CPU/PSI/Latency-type
metrics), simply because they're measured in larger absolute units.
Step 2's raw-mean audit already found Tier 3's
`gpu_memory_total`/`gpu_power_watts` running several times higher than
A16's in absolute terms (H100 vs A16 hardware) — a metric with
mechanically larger raw magnitude will mechanically produce a larger
raw-unit Wasserstein distance even at equivalent *relative* fit
quality. Whether the ~5x uniform inflation here is fully explained by
that scale effect or partly reflects real fit differences isn't
something this script's output can distinguish.

**Interpretation.** No single workload individually crosses a 10x
inflation, but all five sit in the same narrow 4.6x–7.6x band —
notably uniform given how differently the same workloads behave on
VR (whisper fails, bert overshoots, others pass cleanly). That
uniformity is more consistent with a shared raw-unit scale effect
(above) than five independent per-workload fidelity failures. Not
resolving the question here; treating the pooled Wasserstein numbers
as a secondary signal behind VR for pass/fail purposes, consistent
with the plan's original framing (VR is the threshold metric).

---

## Appendix: A16 vs Tier 3 real-data distributional distance

Produced by `scripts/analysis/a16_vs_tier3_wasserstein.py` (new
script, Step 7 of the retrain plan). Run once, read-only against
`data/processed/phase1_v3/*.npz` and `data/processed/tier3/*.npz`.
Outputs:
- `outputs/phase4/timegan_s36_tier3/dataset_distance_a16_tier3.json`
- `outputs/phase4/timegan_s36_tier3/dataset_distance_a16_tier3.md`

Traces were denormalized to raw physical units before computing
distance (each tier's own `{workload}_normalization.json` min/max) —
computing Wasserstein directly on independently-normalized [0,1] data
would not be physically meaningful, since a value of 0.5 in A16's
normalized space is a different real quantity than 0.5 in Tier 3's.
`metric_names` was read from each `.npz` file (not hardcoded); both
tiers' orderings were asserted identical before pairing by index —
confirmed identical for all 5 workloads.

**This is real-vs-real distance, NOT a model fidelity metric.** It
quantifies how much the raw distribution differs between hardware
platforms (A16 vs Tier 3), independent of any generator. It should not
be compared directly to Table 2's model-vs-real Wasserstein numbers
above — different quantity entirely.

### Full table (verbatim from the `.md` output)

| Workload | pod_cpu_usage | pod_memory_bytes | pod_psi_cpu | pod_latency_avg | pod_throughput | gpu_utilization | gpu_memory_used | gpu_memory_total | gpu_power_watts | gpu_temperature |
|---|---|---|---|---|---|---|---|---|---|---|
| bert | 0.02834 | 2.084e+08 | 1.884e-05 | 0.002583 | 0.1363 | 1.801 | 632 | 1.461e+04 | 15.88 | 1.986 |
| gpt2 | 0.1023 | 2.688e+08 | 3.663e-05 | 0.5483 | 0.1909 | 2.325 | 630.7 | 1.461e+04 | 15.74 | 2.214 |
| resnet152 | 0.01843 | 1.775e+09 | 1.766e-05 | 0.007736 | 0.09437 | 2.556 | 446.2 | 1.461e+04 | 15.49 | 1.946 |
| whisper | 1.239 | 3.412e+08 | 0.3466 | 0.7504 | 0.2753 | 7.592 | 709.2 | 1.461e+04 | 15.4 | 2.573 |
| yolo | 0.02896 | 1.315e+09 | 8.935e-06 | 0.009154 | 0.1319 | 1.138 | 286.9 | 1.461e+04 | 15.68 | 1.685 |

### Flagged outliers (>10x the cross-workload median for that metric)

| Metric | Workload | Value | Median (that metric, across 5 workloads) | Ratio |
|---|---|---|---|---|
| pod_cpu_usage | whisper | 1.239 | 0.0290 | 42.8x |
| pod_psi_cpu | whisper | 0.3466 | 1.884e-05 | 18,394x |
| pod_latency_avg | gpt2 | 0.5483 | 0.00915 | 59.9x |
| pod_latency_avg | whisper | 0.7504 | 0.00915 | 82.0x |

`gpu_memory_total` is identical (14,610) across all 5 workloads by
construction — it's a fixed hardware property (total VRAM), not
workload-dependent, so its A16↔Tier3 distance is just the constant gap
between the two GPUs' total memory, correctly uniform.

**Whisper dominates the outlier list**, and by a wide margin on
`pod_psi_cpu` in particular — though note the *absolute* PSI values on
both sides are tiny (median ~1.9e-5), so an 18,394x ratio is a
near-zero-to-small-nonzero jump, not evidence of a huge absolute
distributional shift on that one metric. `pod_cpu_usage` and
`pod_latency_avg` are more substantive in absolute terms. gpt2 also
shows an elevated `pod_latency_avg` distance, isolated to that one
metric.

**Interpretation.** Whisper's real A16-vs-Tier3 data is genuinely more
different than the other four workloads', concentrated in exactly the
metrics (CPU usage, latency) that Step 2's preprocessing audit already
flagged as having Tier 3's biggest front-gap/warmup pattern (13/50
experiments affected, whisper among the most-affected workloads). This
is consistent with — not proof of — the idea that part of whisper's
model-fit difficulty on Tier 3 (VR 0.759 at frozen hyperparameters)
reflects a genuinely larger real distribution shift for this workload,
not purely a hyperparameter mismatch. Not overreaching further than
that; both explanations (real distribution shift, and
hyperparameter-recipe mismatch) are supported by evidence in this
session and are not mutually exclusive.
