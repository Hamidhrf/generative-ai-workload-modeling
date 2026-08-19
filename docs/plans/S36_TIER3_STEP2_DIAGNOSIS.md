# S36 Tier 3 Retrain Plan — Timestamp-Misalignment Diagnosis

Read-only audit. No scripts modified, nothing run except a throwaway
audit script in `/tmp`, no commits. Scope: quantify the pattern behind
the GPT-2 r=10 crash reported in `S36_TIER3_STEP2_REPORT.md`.

`intersection_length` below is computed with the exact same logic as
`align_timestamps()` in the preprocessor: the size of the set
intersection of unique timestamp values across all 10 metric CSVs for
that experiment (not just first/last range — the true intersection).

## Part A1 — Tier 3 audit (50 experiments)

| workload | r | intersection_length | front_gap_s | back_gap_s | late_metrics |
|---|---|---|---|---|---|
| **whisper** | 9 | **702** | 65.0 | 0.0 | pod_latency_avg, pod_throughput |
| **whisper** | 10 | **706** | 45.0 | 0.0 | pod_latency_avg, pod_throughput |
| **gpt2** | 10 | **711** | 20.0 | 0.0 | pod_latency_avg, pod_throughput |
| bert | 1–10 | 715 | 0.0 | 0.0 | — |
| gpt2 | 1–9 | 715 | 0.0 | 0.0 | — |
| resnet152 | 1–10 | 715 | 0.0 | 0.0 | — |
| whisper | 1–8 | 715 | 0.0 | 0.0 | — |
| yolo | 1–10 | 715 | 0.0 | 0.0 | — |

(Full 50-row table collapsed above for the 47 unaffected rows — every
one of them is exactly `715 / 0.0 / 0.0 / —`.)

**Tier 3 aggregate stats:**
- Experiments with intersection_length < 715: **3 / 50**
- Experiments with front_gap_seconds > 5: **3 / 50** (identical set)
- Metrics appearing in late_metrics: `pod_latency_avg` (3), `pod_throughput` (3) — always together, never alone, never any other metric
- (workload, r) combos affected: **gpt2 r=10, whisper r=9, whisper r=10**
- back_gap_seconds is 0.0 in every case — only the *start* lags, the end is clean

## Part A2 — Phase 1 v3 (A16) audit (50 experiments)

| workload | r | intersection_length | front_gap_s | back_gap_s | late_metrics |
|---|---|---|---|---|---|
| all 50 rows | — | 715 | 0.0 | 0.0 | — |

**Phase 1 v3 aggregate stats:**
- Experiments with intersection_length < 715: **0 / 50**
- Experiments with front_gap_seconds > 5: **0 / 50**
- No metric ever appears in late_metrics — no front-gap anywhere
- No (workload, r) combos show the pattern

**A16 never exhibits the timestamp-lag pattern at all**, including
Whisper r=10 (see Part A4 — that experiment has a different problem:
a missing pod, not a timestamp lag).

## Part A3 — cross-check against processed files

Part A2's condition ("any Phase 1 v3 experiment with intersection <
715") is **never triggered** — 0/50. So there's nothing to reconcile:
the recovered preprocessor never had to silently truncate any A16
workload, and Step 1 already confirmed byte-for-byte reproduction of
the tracked `.npz` files. Confirmed directly anyway:

| workload | # short-intersection experiments in raw | processed traces.shape | all replicas same length? |
|---|---|---|---|
| bert | 0 | (55, 715, 10) | yes |
| gpt2 | 0 | (55, 715, 10) | yes |
| resnet152 | 0 | (55, 715, 10) | yes |
| whisper | 0 | (55, 715, 10) | yes |
| yolo | 0 | (55, 715, 10) | yes |

A16 processed data has never carried inconsistent trace lengths. Clean.

## Part A4 — Whisper r=10, both tiers

**Tier 3 whisper r=10:**
- All 10 pods present across every pod-level metric, including
  `pod_latency_avg` and `pod_throughput` — no missing pod.
- `pod_cpu_usage` / `pod_memory_bytes` / `pod_psi_cpu`: 715 unique
  timestamps, `2026-08-18 16:17:52.143` → `17:17:22.143`.
- `pod_latency_avg` / `pod_throughput`: only 706 unique timestamps,
  starting **45s later** (`16:18:37.143`), same end time.
- intersection_length = **706**, front_gap_s = 45.0, back_gap_s = 0.0.
- Failure mode: **temporal** — all 10 pods eventually report, but
  latency/throughput collection doesn't start until 45s into the run.

**Phase 1 v3 (A16) whisper r=10:**
- `pod_cpu_usage` / `pod_memory_bytes` / `pod_psi_cpu`: 10 pods, 715
  timestamps.
- `pod_latency_avg` / `pod_throughput`: only **9 pods** (one pod,
  presumably pod_9, never reports latency/throughput for the entire
  run) — but the 9 pods that do report cover all 715 timestamps.
- intersection_length = **715** (full) — because `align_timestamps`
  intersects on *timestamp values*, not on which pods reported them.
  A missing pod doesn't shrink the timestamp intersection; it just
  leaves that pod's column absent, which `create_pod_traces` already
  handles with a `WARNING: Missing pod_9` print and a zero-filled
  column (confirmed in Step 1's byte-for-byte reproduction run).
- Failure mode: **spatial** — one pod is silently absent from two
  metrics for the whole run, not delayed.

**This is directly the A16-vs-Tier3 asymmetry NEXT_STEPS was pointing
at**: A16 Whisper r=10 loses a *pod* under contention; Tier 3 Whisper
r=10 (and r=9, and GPT-2 r=10) loses *leading timestamps* under
contention instead. Different mechanism, same underlying cause
(system under heavy load takes longer before the first inference
completes, so latency/throughput — which are computed from completed
requests — have nothing to report yet).

## Findings

1. **The pattern is real but narrow: 3 of 50 Tier 3 experiments (6%)**,
   all at high replica counts (whisper r=9, whisper r=10, gpt2 r=10),
   and in every case it's specifically `pod_latency_avg` and
   `pod_throughput` that lag — the two metrics computed from completed
   inference requests. The 8 other metrics (resource/GPU telemetry)
   never lag; they start recording immediately regardless of load.
2. **A16 (Phase 1 v3) never shows this failure mode** — 0/50 experiments
   have any timestamp-intersection shortfall or front-gap. A16's only
   irregularity anywhere in the dataset is Whisper r=10's missing
   pod_9, which is a completely different (and already-handled)
   mechanism.
3. **Tier 3 and A16 fail differently on the *same* nominal experiment**
   (Whisper r=10): A16 loses a pod (spatial), Tier 3 loses leading
   timestamps (temporal). Both plausibly stem from the same root cause
   — heavy contention delays the first successful inference — but they
   manifest as different data shapes and need different handling in
   the preprocessor. The existing code already tolerates the A16
   failure mode (warn + zero-fill); it has no handling at all for the
   Tier 3 failure mode (hard crash on concatenate).
4. **The gap is always at the front, never the back** (back_gap_s = 0.0
   in all 3 flagged cases) — consistent with a warmup/ramp-up
   explanation rather than e.g. a mid-run collector hiccup or an early
   termination.
5. **Severity is not strictly proportional to replica count**: whisper
   r=9 has a *larger* front-gap (65s) than whisper r=10 (45s), even
   though r=10 has more contention. So replica count alone doesn't
   fully explain the magnitude — worth keeping in mind if a fix ends
   up being replica-count-conditional.
