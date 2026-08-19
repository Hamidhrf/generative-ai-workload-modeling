# S36 Tier 3 Retrain Plan — Step 2 Report

## 1. Adapted script
`scripts/phase4/preprocess_tier3.py` created by copying the recovered
`preprocess_v3_10metric.py` and changing exactly 2 lines (plus the
requested 2-line header comment):

```diff
+# Adapted from preprocess_v3_10metric.py for Tier 3 H100
+# data. Only RAW_DATA_DIR and OUTPUT_DIR changed.
-RAW_DATA_DIR = Path("data/raw/phase1_v3")
-OUTPUT_DIR = Path("data/processed")
+RAW_DATA_DIR = Path("data/raw/extension_tier3")
+OUTPUT_DIR = Path("data/processed/tier3")
```

`OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` preserved unchanged.
No other edits. Not committed.

## 2. Raw structure check — no drift
`data/raw/extension_tier3/` has the expected 50 `{workload}_r{n}`
directories. CSV filenames follow `{workload}_r{n}_{metric}_{timestamp}.csv`,
matching what the loader expects. All 10 metrics the script consumes
(5 pod + 5 system GPU) are present under the exact same names as
Phase 1 v3. Tier 3 raw dirs contain extra metric files the script
doesn't touch (`app_latency_p50/p95/p99`, `app_throughput`, `node_*`,
`pod_psi_io`, `pod_psi_memory`) — no naming conflict, nothing to fix.

## 3. Run result — CRASHED on GPT-2 r=10, did not invent a fix

```
python scripts/phase4/preprocess_tier3.py
```

**BERT completed successfully** (all 10 replicas, 715 timesteps each,
`data/processed/tier3/bert_traces.npz` + `bert_normalization.json`
written).

**GPT-2 crashed** on r=10:

```
Loading gpt2 r=10...
  Aligning to 711 timestamps
  Creating: 10 pods, 711 timesteps
Traceback (most recent call last):
  ...
  File "scripts/phase4/preprocess_tier3.py", line 123, in process_workload
    all_traces = np.nan_to_num(np.concatenate(all_traces, axis=0), nan=0.0)
ValueError: all the input array dimensions except for the concatenation
axis must match exactly, but along dimension 1, the array at index 0
has size 715 and the array at index 9 has size 711
```

Script exits on unhandled exception — **`resnet152`, `whisper`, and
`yolo` were never processed.** No partial/corrupt file was written for
gpt2 (crash happens before `np.savez_compressed`), so
`data/processed/tier3/` only contains valid `bert_*` files right now.

**So the plan's headline question — does Tier 3 Whisper r=10 hit the
same "Missing pod_9" path as A16, or is it complete? — is still
unanswered.** The pipeline never reached Whisper.

### Root cause of the GPT-2 r=10 crash (diagnosed, not fixed)
All 10 pods are present for GPT-2 r=10 (this is *not* the Whisper-style
missing-pod issue). Instead, two of the five per-pod metric CSVs start
recording later than the rest:

| file | unique timestamps | first timestamp |
|---|---|---|
| pod_cpu_usage, pod_memory_bytes, pod_psi_cpu, gpu_* (5 files) | 715 | 2026-08-18 05:17:50.790 |
| pod_latency_avg | 711 | 2026-08-18 05:18:10.790 |
| pod_throughput | 711 | 2026-08-18 05:18:10.790 |

`pod_latency_avg`/`pod_throughput` are missing the first 4 samples (20s)
present in every other metric for this experiment — plausibly a real
warmup gap (no completed inference requests yet to compute
latency/throughput from), not a naming or schema issue. `align_timestamps`
intersects timestamps across *all* metrics, so the whole GPT-2 r=10
experiment truncates to 711 timesteps while every other replica in the
workload stays at 715, and `create_pod_traces`/`process_workload` has no
handling for per-replica timestep-count variation — it assumes uniform
715 across a workload for the final `np.concatenate`. This is Risk R2
from the plan, just a different mechanism (partial-metric warmup gap,
not the `application`→`app` label rename).

I did not patch the script or drop/pad timestamps — stopping here per
your instruction not to invent a fix.

## 4. Verification — BERT only (the one workload that succeeded)

```
shape: (55, 715, 10)
replica_counts sorted set: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
counter: {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}
nan: 0   inf: 0
```
Clean — matches all expected invariants.

## 5. Sanity comparison — BERT, Tier 3 vs A16 (raw, un-normalized means)

| metric | A16 raw range | A16 raw mean | Tier3 raw range | Tier3 raw mean |
|---|---|---|---|---|
| pod_cpu_usage | [0.01, 0.20] | 0.096 | [0.00, 0.20] | 0.068 |
| pod_memory_bytes | [7.9e8, 1.12e9] | 8.36e8 | [3.4e8, 2.11e9] | 1.04e9 |
| pod_psi_cpu | [0.00, 0.00] | 0.000 | [0.00, 0.00] | 0.000 |
| pod_latency_avg | [0.01, 0.01] | 0.008 | [0.00, 0.01] | 0.006 |
| pod_throughput | [0.31, 4.24] | 2.304 | [0.00, 4.45] | 2.370 |
| gpu_utilization | [0.00, 12.25] | 2.696 | [0.00, 4.00] | 0.895 |
| gpu_memory_used | [554.9, 569.0] | 556.0 | [119.3, 1192.0] | 1186.4 |
| gpu_memory_total | [1496.2, 14962.0] | 2720.3 | [9531.8, 95318.0] | 17330.6 |
| gpu_power_watts | [3.29, 41.54] | 7.26 | [11.52, 123.09] | 23.14 |
| gpu_temperature | [6.40, 66.00] | 12.26 | [4.50, 56.00] | 10.28 |

No red flags — differences look like real, expected hardware deltas
between A16 and H100, not a preprocessing bug:
- `gpu_memory_total` ~6.4x higher on Tier 3 (H100 NVL ~94GB vs A16 ~15GB
  reported here after replica-count division — consistent with known
  H100 vs A16 VRAM).
- `gpu_power_watts` ~3x higher on Tier 3 (H100 TDP far exceeds A16's).
- `gpu_utilization` is *lower* on Tier 3 (0.895 vs 2.696 mean) — H100
  needs less utilization to do the same BERT inference work, the
  opposite direction of a "5x higher, something's wrong" red flag.
- `pod_memory_bytes`/`gpu_temperature`/`pod_throughput` are same order
  of magnitude, plausible spread.

Could not run this comparison for gpt2/resnet152/whisper/yolo since
only BERT's Tier 3 output exists.

## Bottom line
Step 2 is **blocked** on the GPT-2 r=10 timestamp-alignment crash.
`preprocess_tier3.py` is correct as a direct line-for-line adaptation —
this is a genuine Tier 3 data characteristic, not a bug introduced by
the adaptation. Need a decision on how `create_pod_traces`/
`process_workload` should handle a replica whose common-timestamp
window is shorter than the rest of its workload (e.g. truncate every
replica in the workload to the shortest common length, drop just that
replica, or something else) before re-running to reach GPT-2 r=10 and
beyond, including the still-unanswered Whisper r=10 question.

---

## Step 2 rerun after fix

### Diff summary

Only `scripts/phase4/preprocess_tier3.py` was touched.
`preprocess_v3_10metric.py` (the recovered A16 script) was **not**
modified — it stays as historical reference, per instructions.

`align_timestamps()` was rewritten (intersection → union + per-(metric,
pod) left-fill, ~50 new lines including the policy comment block) and
`process_workload()`'s call site was updated to pass `workload`/`r`
through for logging (1 line). Two new module-level items were added:
`MAX_FRONT_GAP_S = 180` (with rationale comment) and three globals for
end-of-run bookkeeping (`LEFT_FILL_EXPERIMENTS`,
`LEFT_FILL_SERIES_COUNT`, `MAX_FRONT_GAP_OBSERVED`). A 3-line SUMMARY
print block was added after the main loop. `create_pod_traces` was
**not** touched — it already handled variable pod counts correctly;
the crash was entirely in `align_timestamps`.

Net diff vs. `preprocess_v3_10metric.py`: 138 → 208 lines (+78/-8 vs.
the unpatched `preprocess_tier3.py`, which itself was a 2-line diff
from the A16 script).

### Rerun result

```
python scripts/phase4/preprocess_tier3.py
```

**All 5/5 workloads processed without exception**, output written for
bert/gpt2/resnet152/whisper/yolo. Full log: `logs/preprocess_tier3_v2.log`.

### ⚠️ Surprise: 13 experiments needed left-fill, not 3

The task's expectation (from `S36_TIER3_STEP2_DIAGNOSIS.md`) was 3
flagged experiments: gpt2 r=10, whisper r=9, whisper r=10. The actual
result is **13 experiments, 154 individual (metric, pod) series**:

```
bert r=8, bert r=9, bert r=10,
gpt2 r=7, gpt2 r=8, gpt2 r=9, gpt2 r=10,
whisper r=4, whisper r=6, whisper r=7, whisper r=8, whisper r=9, whisper r=10
```

**Why the earlier diagnosis undercounted:** Part A1/A2 of the diagnosis
computed `intersection_length` from the *pooled* per-metric timestamp
set — i.e. `set(df['timestamp'])` on the long-format CSV across all
pods at once. If even one pod in a replica starts on time, that pod's
early timestamps land in the pooled set and mask every other pod's
individual lag. The diagnosis only caught the case where *every* pod
in a metric lagged by the same amount. The new per-pod check (required
by this fix, since the fill itself has to be per-(metric, pod)) is
strictly more sensitive and surfaces every partial lag, not just
uniform ones. This is a real gap in the earlier diagnosis, not a bug
in the fix — flagging it because the task explicitly called out
"anything else is a surprise."

`resnet152` and `yolo` never appear — consistent with the diagnosis's
finding that only latency/throughput-sensitive workloads under
contention exhibit this.

No `MAX_FRONT_GAP_S` (180s) threshold breaches occurred — max observed
gap was 165.0s (several whisper r=10 pods), 15s under the tripwire.
Given the diagnosis's per-pooled-metric numbers undercounted actual
per-pod gaps, this margin is closer than the original 65s-vs-180s
comparison suggested; worth keeping in mind if Tier 2 shows a similar
pattern at higher contention.

### Full LEFT_FILL log (154 lines)

```
LEFT_FILL: bert r=8 pod_latency_avg pod=pod_2 fill_n=1 first_val=0.0079 gap_s=5.0
LEFT_FILL: bert r=8 pod_latency_avg pod=pod_7 fill_n=2 first_val=0.0057 gap_s=10.0
LEFT_FILL: bert r=8 pod_throughput pod=pod_2 fill_n=1 first_val=0.0515 gap_s=5.0
LEFT_FILL: bert r=8 pod_throughput pod=pod_7 fill_n=2 first_val=0.0363 gap_s=10.0
LEFT_FILL: bert r=9 pod_latency_avg pod=pod_0 fill_n=4 first_val=0.0062 gap_s=20.0
LEFT_FILL: bert r=9 pod_latency_avg pod=pod_2 fill_n=6 first_val=0.0077 gap_s=30.0
LEFT_FILL: bert r=9 pod_latency_avg pod=pod_5 fill_n=7 first_val=0.0056 gap_s=35.0
LEFT_FILL: bert r=9 pod_latency_avg pod=pod_7 fill_n=6 first_val=0.0062 gap_s=30.0
LEFT_FILL: bert r=9 pod_latency_avg pod=pod_8 fill_n=6 first_val=0.0062 gap_s=30.0
LEFT_FILL: bert r=9 pod_throughput pod=pod_0 fill_n=4 first_val=0.0413 gap_s=20.0
LEFT_FILL: bert r=9 pod_throughput pod=pod_2 fill_n=6 first_val=0.0593 gap_s=30.0
LEFT_FILL: bert r=9 pod_throughput pod=pod_5 fill_n=7 first_val=0.0408 gap_s=35.0
LEFT_FILL: bert r=9 pod_throughput pod=pod_7 fill_n=6 first_val=0.0375 gap_s=30.0
LEFT_FILL: bert r=9 pod_throughput pod=pod_8 fill_n=6 first_val=0.0337 gap_s=30.0
LEFT_FILL: bert r=10 pod_latency_avg pod=pod_1 fill_n=5 first_val=0.0074 gap_s=25.0
LEFT_FILL: bert r=10 pod_latency_avg pod=pod_2 fill_n=5 first_val=0.0060 gap_s=25.0
LEFT_FILL: bert r=10 pod_latency_avg pod=pod_3 fill_n=11 first_val=0.0069 gap_s=55.0
LEFT_FILL: bert r=10 pod_latency_avg pod=pod_4 fill_n=12 first_val=0.0062 gap_s=60.0
LEFT_FILL: bert r=10 pod_latency_avg pod=pod_5 fill_n=5 first_val=0.0068 gap_s=25.0
LEFT_FILL: bert r=10 pod_latency_avg pod=pod_6 fill_n=13 first_val=0.0058 gap_s=65.0
LEFT_FILL: bert r=10 pod_latency_avg pod=pod_7 fill_n=14 first_val=0.0059 gap_s=70.0
LEFT_FILL: bert r=10 pod_latency_avg pod=pod_8 fill_n=10 first_val=0.0068 gap_s=50.0
LEFT_FILL: bert r=10 pod_latency_avg pod=pod_9 fill_n=12 first_val=0.0085 gap_s=60.0
LEFT_FILL: bert r=10 pod_throughput pod=pod_1 fill_n=5 first_val=0.0320 gap_s=25.0
LEFT_FILL: bert r=10 pod_throughput pod=pod_2 fill_n=5 first_val=0.0669 gap_s=25.0
LEFT_FILL: bert r=10 pod_throughput pod=pod_3 fill_n=11 first_val=0.0766 gap_s=55.0
LEFT_FILL: bert r=10 pod_throughput pod=pod_4 fill_n=12 first_val=0.0253 gap_s=60.0
LEFT_FILL: bert r=10 pod_throughput pod=pod_5 fill_n=5 first_val=0.0408 gap_s=25.0
LEFT_FILL: bert r=10 pod_throughput pod=pod_6 fill_n=13 first_val=0.0558 gap_s=65.0
LEFT_FILL: bert r=10 pod_throughput pod=pod_7 fill_n=14 first_val=0.0627 gap_s=70.0
LEFT_FILL: bert r=10 pod_throughput pod=pod_8 fill_n=10 first_val=0.0669 gap_s=50.0
LEFT_FILL: bert r=10 pod_throughput pod=pod_9 fill_n=12 first_val=0.0771 gap_s=60.0
LEFT_FILL: gpt2 r=7 pod_latency_avg pod=pod_1 fill_n=2 first_val=0.4744 gap_s=10.0
LEFT_FILL: gpt2 r=7 pod_latency_avg pod=pod_2 fill_n=1 first_val=0.2497 gap_s=5.0
LEFT_FILL: gpt2 r=7 pod_throughput pod=pod_1 fill_n=2 first_val=0.0471 gap_s=10.0
LEFT_FILL: gpt2 r=7 pod_throughput pod=pod_2 fill_n=1 first_val=0.0399 gap_s=5.0
LEFT_FILL: gpt2 r=8 pod_latency_avg pod=pod_0 fill_n=9 first_val=0.5773 gap_s=45.0
LEFT_FILL: gpt2 r=8 pod_latency_avg pod=pod_2 fill_n=3 first_val=0.5077 gap_s=15.0
LEFT_FILL: gpt2 r=8 pod_latency_avg pod=pod_3 fill_n=7 first_val=0.4882 gap_s=35.0
LEFT_FILL: gpt2 r=8 pod_latency_avg pod=pod_4 fill_n=9 first_val=0.4441 gap_s=45.0
LEFT_FILL: gpt2 r=8 pod_latency_avg pod=pod_5 fill_n=3 first_val=0.4613 gap_s=15.0
LEFT_FILL: gpt2 r=8 pod_latency_avg pod=pod_7 fill_n=3 first_val=0.6250 gap_s=15.0
LEFT_FILL: gpt2 r=8 pod_throughput pod=pod_0 fill_n=9 first_val=0.0271 gap_s=45.0
LEFT_FILL: gpt2 r=8 pod_throughput pod=pod_2 fill_n=3 first_val=0.0415 gap_s=15.0
LEFT_FILL: gpt2 r=8 pod_throughput pod=pod_3 fill_n=7 first_val=0.0303 gap_s=35.0
LEFT_FILL: gpt2 r=8 pod_throughput pod=pod_4 fill_n=9 first_val=0.0445 gap_s=45.0
LEFT_FILL: gpt2 r=8 pod_throughput pod=pod_5 fill_n=3 first_val=0.0392 gap_s=15.0
LEFT_FILL: gpt2 r=8 pod_throughput pod=pod_7 fill_n=3 first_val=0.0283 gap_s=15.0
LEFT_FILL: gpt2 r=9 pod_latency_avg pod=pod_0 fill_n=18 first_val=0.5929 gap_s=90.0
LEFT_FILL: gpt2 r=9 pod_latency_avg pod=pod_2 fill_n=16 first_val=0.2753 gap_s=80.0
LEFT_FILL: gpt2 r=9 pod_latency_avg pod=pod_3 fill_n=12 first_val=0.6603 gap_s=60.0
LEFT_FILL: gpt2 r=9 pod_latency_avg pod=pod_4 fill_n=17 first_val=0.5795 gap_s=85.0
LEFT_FILL: gpt2 r=9 pod_latency_avg pod=pod_5 fill_n=14 first_val=0.5231 gap_s=70.0
LEFT_FILL: gpt2 r=9 pod_latency_avg pod=pod_6 fill_n=5 first_val=0.5262 gap_s=25.0
LEFT_FILL: gpt2 r=9 pod_latency_avg pod=pod_7 fill_n=6 first_val=0.4572 gap_s=30.0
LEFT_FILL: gpt2 r=9 pod_throughput pod=pod_0 fill_n=18 first_val=0.0303 gap_s=90.0
LEFT_FILL: gpt2 r=9 pod_throughput pod=pod_2 fill_n=16 first_val=0.0426 gap_s=80.0
LEFT_FILL: gpt2 r=9 pod_throughput pod=pod_3 fill_n=12 first_val=0.0406 gap_s=60.0
LEFT_FILL: gpt2 r=9 pod_throughput pod=pod_4 fill_n=17 first_val=0.0585 gap_s=85.0
LEFT_FILL: gpt2 r=9 pod_throughput pod=pod_5 fill_n=14 first_val=0.0200 gap_s=70.0
LEFT_FILL: gpt2 r=9 pod_throughput pod=pod_6 fill_n=5 first_val=0.0561 gap_s=25.0
LEFT_FILL: gpt2 r=9 pod_throughput pod=pod_7 fill_n=6 first_val=0.0379 gap_s=30.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_0 fill_n=26 first_val=0.3985 gap_s=130.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_1 fill_n=5 first_val=0.3694 gap_s=25.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_2 fill_n=4 first_val=0.3965 gap_s=20.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_3 fill_n=19 first_val=0.6067 gap_s=95.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_4 fill_n=22 first_val=0.2914 gap_s=110.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_5 fill_n=22 first_val=0.5155 gap_s=110.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_6 fill_n=26 first_val=0.4400 gap_s=130.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_7 fill_n=20 first_val=0.3706 gap_s=100.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_8 fill_n=24 first_val=0.5702 gap_s=120.0
LEFT_FILL: gpt2 r=10 pod_latency_avg pod=pod_9 fill_n=25 first_val=0.5961 gap_s=125.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_0 fill_n=26 first_val=0.0469 gap_s=130.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_1 fill_n=5 first_val=0.0396 gap_s=25.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_2 fill_n=4 first_val=0.0488 gap_s=20.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_3 fill_n=19 first_val=0.0561 gap_s=95.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_4 fill_n=22 first_val=0.0438 gap_s=110.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_5 fill_n=22 first_val=0.0172 gap_s=110.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_6 fill_n=26 first_val=0.0828 gap_s=130.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_7 fill_n=20 first_val=0.0411 gap_s=100.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_8 fill_n=24 first_val=0.0493 gap_s=120.0
LEFT_FILL: gpt2 r=10 pod_throughput pod=pod_9 fill_n=25 first_val=0.0180 gap_s=125.0
LEFT_FILL: whisper r=4 pod_latency_avg pod=pod_3 fill_n=3 first_val=0.1253 gap_s=15.0
LEFT_FILL: whisper r=4 pod_throughput pod=pod_3 fill_n=3 first_val=0.0367 gap_s=15.0
LEFT_FILL: whisper r=6 pod_latency_avg pod=pod_0 fill_n=5 first_val=0.0718 gap_s=25.0
LEFT_FILL: whisper r=6 pod_latency_avg pod=pod_1 fill_n=2 first_val=0.1992 gap_s=10.0
LEFT_FILL: whisper r=6 pod_latency_avg pod=pod_2 fill_n=3 first_val=0.1132 gap_s=15.0
LEFT_FILL: whisper r=6 pod_latency_avg pod=pod_4 fill_n=6 first_val=0.4061 gap_s=30.0
LEFT_FILL: whisper r=6 pod_throughput pod=pod_0 fill_n=5 first_val=0.0356 gap_s=25.0
LEFT_FILL: whisper r=6 pod_throughput pod=pod_1 fill_n=2 first_val=0.0373 gap_s=10.0
LEFT_FILL: whisper r=6 pod_throughput pod=pod_2 fill_n=3 first_val=0.0518 gap_s=15.0
LEFT_FILL: whisper r=6 pod_throughput pod=pod_4 fill_n=6 first_val=0.0642 gap_s=30.0
LEFT_FILL: whisper r=7 pod_latency_avg pod=pod_0 fill_n=8 first_val=1.6361 gap_s=40.0
LEFT_FILL: whisper r=7 pod_latency_avg pod=pod_1 fill_n=7 first_val=0.5262 gap_s=35.0
LEFT_FILL: whisper r=7 pod_latency_avg pod=pod_2 fill_n=9 first_val=0.1150 gap_s=45.0
LEFT_FILL: whisper r=7 pod_latency_avg pod=pod_3 fill_n=7 first_val=0.3820 gap_s=35.0
LEFT_FILL: whisper r=7 pod_latency_avg pod=pod_4 fill_n=7 first_val=0.2042 gap_s=35.0
LEFT_FILL: whisper r=7 pod_latency_avg pod=pod_5 fill_n=5 first_val=0.6480 gap_s=25.0
LEFT_FILL: whisper r=7 pod_throughput pod=pod_0 fill_n=8 first_val=0.0278 gap_s=40.0
LEFT_FILL: whisper r=7 pod_throughput pod=pod_1 fill_n=7 first_val=0.0368 gap_s=35.0
LEFT_FILL: whisper r=7 pod_throughput pod=pod_2 fill_n=9 first_val=0.0712 gap_s=45.0
LEFT_FILL: whisper r=7 pod_throughput pod=pod_3 fill_n=7 first_val=0.0336 gap_s=35.0
LEFT_FILL: whisper r=7 pod_throughput pod=pod_4 fill_n=7 first_val=0.0441 gap_s=35.0
LEFT_FILL: whisper r=7 pod_throughput pod=pod_5 fill_n=5 first_val=0.0662 gap_s=25.0
LEFT_FILL: whisper r=8 pod_latency_avg pod=pod_2 fill_n=15 first_val=0.1117 gap_s=75.0
LEFT_FILL: whisper r=8 pod_latency_avg pod=pod_3 fill_n=12 first_val=0.9634 gap_s=60.0
LEFT_FILL: whisper r=8 pod_latency_avg pod=pod_4 fill_n=15 first_val=0.1257 gap_s=75.0
LEFT_FILL: whisper r=8 pod_latency_avg pod=pod_5 fill_n=14 first_val=0.0887 gap_s=70.0
LEFT_FILL: whisper r=8 pod_latency_avg pod=pod_6 fill_n=14 first_val=0.1280 gap_s=70.0
LEFT_FILL: whisper r=8 pod_latency_avg pod=pod_7 fill_n=14 first_val=0.9274 gap_s=70.0
LEFT_FILL: whisper r=8 pod_throughput pod=pod_2 fill_n=15 first_val=0.0713 gap_s=75.0
LEFT_FILL: whisper r=8 pod_throughput pod=pod_3 fill_n=12 first_val=0.0340 gap_s=60.0
LEFT_FILL: whisper r=8 pod_throughput pod=pod_4 fill_n=15 first_val=0.0753 gap_s=75.0
LEFT_FILL: whisper r=8 pod_throughput pod=pod_5 fill_n=14 first_val=0.0340 gap_s=70.0
LEFT_FILL: whisper r=8 pod_throughput pod=pod_6 fill_n=14 first_val=0.0557 gap_s=70.0
LEFT_FILL: whisper r=8 pod_throughput pod=pod_7 fill_n=14 first_val=0.0664 gap_s=70.0
LEFT_FILL: whisper r=9 pod_latency_avg pod=pod_0 fill_n=20 first_val=0.1113 gap_s=100.0
LEFT_FILL: whisper r=9 pod_latency_avg pod=pod_1 fill_n=19 first_val=1.7568 gap_s=95.0
LEFT_FILL: whisper r=9 pod_latency_avg pod=pod_2 fill_n=16 first_val=0.4366 gap_s=80.0
LEFT_FILL: whisper r=9 pod_latency_avg pod=pod_3 fill_n=14 first_val=0.4507 gap_s=70.0
LEFT_FILL: whisper r=9 pod_latency_avg pod=pod_4 fill_n=20 first_val=0.3078 gap_s=100.0
LEFT_FILL: whisper r=9 pod_latency_avg pod=pod_5 fill_n=20 first_val=0.6445 gap_s=100.0
LEFT_FILL: whisper r=9 pod_latency_avg pod=pod_6 fill_n=13 first_val=0.1873 gap_s=65.0
LEFT_FILL: whisper r=9 pod_latency_avg pod=pod_7 fill_n=18 first_val=0.5469 gap_s=90.0
LEFT_FILL: whisper r=9 pod_latency_avg pod=pod_8 fill_n=19 first_val=0.3281 gap_s=95.0
LEFT_FILL: whisper r=9 pod_throughput pod=pod_0 fill_n=20 first_val=0.0659 gap_s=100.0
LEFT_FILL: whisper r=9 pod_throughput pod=pod_1 fill_n=19 first_val=0.0234 gap_s=95.0
LEFT_FILL: whisper r=9 pod_throughput pod=pod_2 fill_n=16 first_val=0.0397 gap_s=80.0
LEFT_FILL: whisper r=9 pod_throughput pod=pod_3 fill_n=14 first_val=0.0573 gap_s=70.0
LEFT_FILL: whisper r=9 pod_throughput pod=pod_4 fill_n=20 first_val=0.0486 gap_s=100.0
LEFT_FILL: whisper r=9 pod_throughput pod=pod_5 fill_n=20 first_val=0.0665 gap_s=100.0
LEFT_FILL: whisper r=9 pod_throughput pod=pod_6 fill_n=13 first_val=0.0407 gap_s=65.0
LEFT_FILL: whisper r=9 pod_throughput pod=pod_7 fill_n=18 first_val=0.0335 gap_s=90.0
LEFT_FILL: whisper r=9 pod_throughput pod=pod_8 fill_n=19 first_val=0.0461 gap_s=95.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_0 fill_n=29 first_val=0.1716 gap_s=145.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_1 fill_n=17 first_val=0.4969 gap_s=85.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_2 fill_n=22 first_val=0.5801 gap_s=110.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_3 fill_n=9 first_val=0.1439 gap_s=45.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_4 fill_n=31 first_val=0.9297 gap_s=155.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_5 fill_n=33 first_val=0.1143 gap_s=165.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_6 fill_n=20 first_val=0.1790 gap_s=100.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_7 fill_n=33 first_val=0.6744 gap_s=165.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_8 fill_n=31 first_val=1.0659 gap_s=155.0
LEFT_FILL: whisper r=10 pod_latency_avg pod=pod_9 fill_n=33 first_val=0.1507 gap_s=165.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_0 fill_n=29 first_val=0.0598 gap_s=145.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_1 fill_n=17 first_val=0.0547 gap_s=85.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_2 fill_n=22 first_val=0.0410 gap_s=110.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_3 fill_n=9 first_val=0.0348 gap_s=45.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_4 fill_n=31 first_val=0.0339 gap_s=155.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_5 fill_n=33 first_val=0.0651 gap_s=165.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_6 fill_n=20 first_val=0.0601 gap_s=100.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_7 fill_n=33 first_val=0.0517 gap_s=165.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_8 fill_n=31 first_val=0.0556 gap_s=155.0
LEFT_FILL: whisper r=10 pod_throughput pod=pod_9 fill_n=33 first_val=0.0817 gap_s=165.0
```

No `WARNING` lines were printed anywhere in the run (0 threshold
breaches, 0 unresolved trailing NaN) — confirmed via
`grep -c WARNING logs/preprocess_tier3_v2.log` → `0`.

### SUMMARY block (verbatim)

```
SUMMARY: 13 experiments required left-fill: ['bert r=8', 'bert r=9', 'bert r=10', 'gpt2 r=7', 'gpt2 r=8', 'gpt2 r=9', 'gpt2 r=10', 'whisper r=4', 'whisper r=6', 'whisper r=7', 'whisper r=8', 'whisper r=9', 'whisper r=10']
SUMMARY: total (metric, pod) series left-filled: 154
SUMMARY: max front_gap observed: 165.0 seconds
```

### Verification table

| workload | shape | replica_counts set | counter == {1:1..10:10} | NaN | Inf |
|---|---|---|---|---|---|
| bert | (55, 715, 10) | [1..10] | True | 0 | 0 |
| gpt2 | (55, 715, 10) | [1..10] | True | 0 | 0 |
| resnet152 | (55, 715, 10) | [1..10] | True | 0 | 0 |
| whisper | (55, 715, 10) | [1..10] | True | 0 | 0 |
| yolo | (55, 715, 10) | [1..10] | True | 0 | 0 |

All 5 workloads: correct shape, full 1–10 replica coverage, exactly
one experiment per replica count, zero NaN, zero Inf.

### A16 vs Tier 3 — raw per-metric means, all 5 workloads

Raw (un-normalized) means, reconstructed from each tier's own
normalization JSON (`raw = norm_mean * (max - min) + min`):

**BERT**

| metric | A16 raw mean | Tier3 raw mean |
|---|---|---|
| pod_cpu_usage | 0.0962 | 0.0679 |
| pod_memory_bytes | 8.362e8 | 1.043e9 |
| pod_psi_cpu | 0.0001 | 0.0001 |
| pod_latency_avg | 0.0083 | 0.0057 |
| pod_throughput | 2.3045 | 2.3705 |
| gpu_utilization | 2.6959 | 0.8952 |
| gpu_memory_used | 556.02 | 1186.39 |
| gpu_memory_total | 2720.31 | 17330.64 |
| gpu_power_watts | 7.26 | 23.14 |
| gpu_temperature | 12.26 | 10.28 |

**GPT-2**

| metric | A16 raw mean | Tier3 raw mean |
|---|---|---|
| pod_cpu_usage | 0.6683 | 0.5660 |
| pod_memory_bytes | 1.113e9 | 1.374e9 |
| pod_psi_cpu | 0.0003 | 0.0003 |
| pod_latency_avg | 1.3632 | 0.8150 |
| pod_throughput | 0.5439 | 0.7208 |
| gpu_utilization | 14.4854 | 12.1606 |
| gpu_memory_used | 634.52 | 1258.81 |
| gpu_memory_total | 2720.31 | 17330.64 |
| gpu_power_watts | 9.79 | 25.53 |
| gpu_temperature | 13.00 | 10.78 |

**ResNet-152**

| metric | A16 raw mean | Tier3 raw mean |
|---|---|---|
| pod_cpu_usage | 0.0430 | 0.0249 |
| pod_memory_bytes | 2.615e9 | 8.400e8 |
| pod_psi_cpu | 0.0001 | 0.0001 |
| pod_latency_avg | 0.0169 | 0.0092 |
| pod_throughput | 2.5024 | 2.5459 |
| gpu_utilization | 3.5139 | 1.0156 |
| gpu_memory_used | 548.02 | 994.20 |
| gpu_memory_total | 2720.31 | 17330.64 |
| gpu_power_watts | 7.16 | 22.65 |
| gpu_temperature | 12.11 | 10.16 |

**Whisper**

| metric | A16 raw mean | Tier3 raw mean |
|---|---|---|
| pod_cpu_usage | 2.4097 | 1.1722 |
| pod_memory_bytes | 1.326e9 | 1.646e9 |
| pod_psi_cpu | 0.4450 | 0.0984 |
| pod_latency_avg | 1.1285 | 0.3820 |
| pod_throughput | 1.1437 | 1.1258 |
| gpu_utilization | 10.9592 | 3.3674 |
| gpu_memory_used | 1555.86 | 2225.73 |
| gpu_memory_total | 2720.31 | 17330.64 |
| gpu_power_watts | 9.25 | 24.66 |
| gpu_temperature | 13.17 | 10.60 |

**YOLO**

| metric | A16 raw mean | Tier3 raw mean |
|---|---|---|
| pod_cpu_usage | 0.0696 | 0.0407 |
| pod_memory_bytes | 2.551e9 | 1.236e9 |
| pod_psi_cpu | 0.0001 | 0.0001 |
| pod_latency_avg | 0.0229 | 0.0138 |
| pod_throughput | 2.3843 | 2.5162 |
| gpu_utilization | 1.6250 | 0.4867 |
| gpu_memory_used | 418.02 | 704.96 |
| gpu_memory_total | 2720.31 | 17330.64 |
| gpu_power_watts | 6.54 | 22.21 |
| gpu_temperature | 11.80 | 10.11 |

**No red flags anywhere.** The same consistent, hardware-explained
pattern holds across all 5 workloads: `gpu_memory_total`/`gpu_memory_used`/
`gpu_power_watts` are all higher on Tier 3 (H100 has far more VRAM and
much higher TDP than A16), `gpu_utilization` is consistently *lower* on
Tier 3 (H100 needs less utilization per unit of work), and
`gpu_temperature` is consistently a bit lower on Tier 3. `pod_psi_cpu`
for Whisper is notably lower on Tier 3 (0.098 vs A16's 0.445) —
consistent with the plan's earlier note that A16 Whisper crashed at
PSI 0.63 while Tier 3 completed cleanly with much lower PSI.

### Status

Step 2 is now **complete**. `data/processed/tier3/` contains all 5
workloads' `.npz` + normalization JSON, ready for Step 3 (the
unifier). Nothing has been committed.
