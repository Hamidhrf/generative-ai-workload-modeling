<!-- DRAFT PATCH - local only, NOT applied to TIER3_NOTES.md -->
<!-- Proposed addition to TIER3_NOTES.md, Section 7, whisper row -->

| workload | r=7 latency mean-ratio (A16) | r=7 latency mean-ratio (Tier2) | r=7 latency mean-ratio (Tier3) | r=7 latency peak-ratio (Tier2) | r=7 latency peak-ratio (Tier3) | verdict (mean) | verdict (peak) |
|---|---|---|---|---|---|---|---|
| whisper | 8.4713 | 2.8643 | 2.9345 | 5.3415 | 5.3479 | COLLAPSES (mean-ratio) | COLLAPSES (peak-ratio) |

Full detail: COLLAPSES (mean-ratio): tier3_r7=2.934 is close to tier2_r7=2.864 - Tier2 and Tier3 share the same 16 devlab vCPUs and produce similar mean-ratio ratios at r=7, so 'MIG preserves CPU contention' cannot explain a Tier2-vs-Tier3 difference in mean-ratio.

Full detail: COLLAPSES (peak-ratio): tier3_r7=5.348 is close to tier2_r7=5.342 - Tier2 and Tier3 share the same 16 devlab vCPUs and produce similar peak-ratio ratios at r=7, so 'MIG preserves CPU contention' cannot explain a Tier2-vs-Tier3 difference in peak-ratio.

Source: `scripts/analysis/whisper_cross_tier_cpu.py` -> `outputs/analysis/whisper_cpu/whisper_cross_tier_cpu.{json,md,png}` (E13).
