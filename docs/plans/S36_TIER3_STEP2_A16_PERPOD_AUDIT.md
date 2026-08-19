# A16 (Phase 1 v3) Per-Pod Front-Gap Re-Audit

Read-only. `preprocess_v3_10metric.py` was not modified, A16 raw data
was not re-preprocessed, `data/processed/phase1_v3/*.npz` is untouched.

## Method

Same per-(metric, pod) logic used to build the Tier 3 fix, applied to
`data/raw/phase1_v3/` (all 50 experiments, 5 pod-level metrics only —
`pod_cpu_usage`, `pod_memory_bytes`, `pod_psi_cpu`, `pod_latency_avg`,
`pod_throughput`; system GPU metrics have no per-pod dimension). For
each (metric, pod) pair, `front_gap = first_timestamp(metric, pod) -
min(first_timestamp across all (metric, pod) pairs in that
experiment)`, flagged if `> 5s`.

**Important scope caveat:** this method only catches a pod that has
*at least one row but starts late*. It cannot flag a pod that is
*entirely absent* from a metric (0 rows) — that case has no
`first_timestamp` to compare. A16 Whisper r=10's already-known missing
pod_9 (from Step 1/2) falls in that second, different category and is
not and cannot be caught by this front-gap check; it's a separate,
already-documented and already-handled mechanism (warn + zero-fill in
`create_pod_traces`).

## Result: 1 flagged series total (vs. Tier 3's 154)

```
whisper r=10 pod_cpu_usage pod=whisper-inference-796f65c69b-npt98 front_gap_s=325.0
```

**Aggregate stats:**
- Experiments with any per-pod front-gap: **1 / 50** (whisper r=10)
- (metric, pod) series flagged: **1**
- Max front_gap observed: **325.0s**
- (workload, r) combos: **whisper r=10** only

## Comparison to Tier 3

| | A16 (Phase 1 v3) | Tier 3 |
|---|---|---|
| Experiments affected | 1 / 50 | 13 / 50 |
| (metric, pod) series flagged | 1 | 154 |
| Max front_gap | 325.0s | 165.0s |
| Metric(s) involved | `pod_cpu_usage` only | `pod_latency_avg` + `pod_throughput` |
| Workloads involved | whisper only | bert, gpt2, whisper |

A16 has the pattern, but it is far narrower (1 series vs. 154) and hits
a **different metric** than Tier 3. Tier 3's front-gaps are
concentrated in `pod_latency_avg`/`pod_throughput` — metrics computed
from *completed inference requests*, which plausibly take longer to
appear under contention. A16's one instance is in `pod_cpu_usage`, a
cAdvisor resource-usage gauge, not a request-derived metric.

## What actually happened to this pod in the tracked, committed data

Traced it through: this is the **same physical pod** already known to
be missing entirely from `pod_latency_avg`/`pod_throughput` (Step 1/2
finding — "Missing pod_9" warning, 9/10 pods in those two metrics).
For the other two cAdvisor-based pod metrics it's present but with
different behavior:

```
pod_cpu_usage      pod_first=14:29:17.807  exp_min=14:23:52.807  gap_s=325.0
pod_memory_bytes   pod_first=14:23:52.807  exp_min=14:23:52.807  gap_s=0.0
pod_psi_cpu        pod_first=14:23:52.807  exp_min=14:23:52.807  gap_s=0.0
```

Only `pod_cpu_usage` is late for this pod — `pod_memory_bytes` and
`pod_psi_cpu` start on time. Checked what the recovered preprocessor
actually did with this pod's leading 65 timesteps (325s / 5s interval)
in the **already-committed** `data/processed/phase1_v3/whisper_traces.npz`
(pod_index=5 within the r=10 group, global trace index 50):

```
raw pod_cpu_usage, first 80 timesteps:
[0, 0, 0, ..., 0 (70 zeros total in first 80), 0.5627, 0.7192, 0.8757, 0.6769, ...]
```

70 leading zeros (matches the 65-step gap plus a few pivot artifacts),
then real values kick in once the pod's cAdvisor CPU metric starts
reporting. This confirms the original `align_timestamps`
intersection logic never dropped these timesteps (the pooled
per-metric set still had 715 unique timestamps, since 9 other pods
supplied early rows) — it silently produced NaN for this one pod's
missing early cells during the pivot, and `np.nan_to_num(..., nan=0.0)`
at the end of `process_workload` zero-filled them. **This has been in
the tracked A16 data all along**, is not a new artifact, and is not
something this session changed.

## Why this doesn't necessarily need the same left-fill treatment

Structurally this is the same "front-gap" shape as the Tier 3 pattern,
but the two cases differ in whether zero-fill is *defensible* as an
approximation of ground truth:

- **Tier 3** (`pod_latency_avg`/`pod_throughput`): these metrics are
  only computable from completed inference requests. Zero would claim
  "0ms latency" / "0 req/s throughput" during a window when the real
  answer is "no data yet" — actively misleading, which is why the
  left-fill fix was appropriate.
- **A16** (`pod_cpu_usage` for this one late-starting pod): before a
  pod is actually running, CPU usage genuinely is ~0 — there's no
  process consuming CPU yet. Zero-fill here is a plausible
  approximation of physical reality, not a misrepresentation the way
  it would be for latency/throughput.

So while the paper's methodology section should honestly disclose that
A16 has one instance of this same front-gap pattern (previously
undocumented — the earlier per-metric-pooled diagnosis couldn't see
it), it does not obviously warrant re-running A16 through the same
left-fill logic. That's a judgment call, not something resolved by
this audit alone — flagging it for your decision rather than silently
leaving it out of the methodology write-up or silently "fixing" it.

## Bottom line for the paper's methodology section

- A16 is **not** free of this pattern — 1 experiment, 1 series, and
  arguably *worse* in magnitude (325s vs. Tier 3's max 165s) — but it
  is far narrower in scope (1 vs. 154 series) and in a metric where
  zero-fill is defensible rather than misleading.
- The original claim "A16 never exhibits this pattern" (from the
  earlier `S36_TIER3_STEP2_DIAGNOSIS.md`, which only checked pooled
  per-metric intersection) was **incomplete, not wrong for its own
  metric** — it's true that A16 never showed a *pooled* front-gap, but
  a per-pod check reveals one real instance underneath. Recommend the
  methodology section state the per-pod-corrected numbers (1/50 vs.
  13/50), not the earlier pooled numbers (0/50 vs. 3/50).
