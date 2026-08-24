# Tier 2 Notes — H100 MIG per-slice collection

**Status:** Complete
**Batch dates:** 2026-08-14T15:59:47Z to 2026-08-16T06:29:15Z (1 day 14h 29m)
**Cluster:** devLab (172.22.174.66), NVIDIA H100 NVL 94 GB, MIG all-1g.12gb (7 instances)
**Branch:** extension-h100
**Commit range:** 451b955 (Tier 2 manifests) through 78ff774 (yolo r=7)
**Runner:** tools/run_experiment_v4.py
**Batch orchestrator:** tools/run_tier2_batch.sh
**Data location:** data/raw/extension_tier2/<workload>_r<n>/

---

## 1. Executive summary

Tier 2 collected 35 experiments (5 workloads x r=1..7) on H100 with MIG in single-strategy mode, one 1g.12gb slice per pod. Every experiment ran the same 60-minute Business Day load pattern as Phase 1 v3 and Tier 1. Zero failures. Whisper r=7 (the highest-risk cell for CPU contention) completed without halting the batch. Mean per-experiment runtime 65 minutes 59 seconds.

The methodological headline: Tier 2 is the first tier of this research where per-pod GPU attribution is natively available. On A16 Phase 1 v3, GPU time-slicing forced dividing whole-GPU values by replica count. On H100 Tier 1, per-pod trivially equals whole-GPU at r=1. On Tier 2, DCGM per-instance labels attach the pod name directly to per-slice metrics, so per-pod GPU usage is a direct measurement rather than an inference.

35 output directories, 34 files each (33 CSV + 1 timestamps.txt), all commits pushed to origin/extension-h100.

---

## 2. Cluster state during Tier 2

Node: devlab, Ubuntu 26.04, kernel 7.0.0-29.
Kubernetes: 1.34.0 with CRI-O 1.31.5, Calico CNI.
GPU Operator: v26.3.3 (driver.enabled=false; host driver 580.173.02).
MIG configuration: `nvidia.com/mig.config=all-1g.12gb`, state `success`, strategy `single`.
Node capacity: `nvidia.com/gpu: 7` (single strategy collapses per-slice resources under the generic name).
Monitoring stack: Prometheus at `http://172.22.174.66:30090`, DCGM Exporter via GPU Operator's `nvidia-dcgm-exporter` in the `gpu-operator` namespace, node-exporter for host metrics, workload Prometheus endpoints on port 8000. Grafana and kube-state-metrics intentionally skipped.
Workload containers: v4 tag on DockerHub (`hamidhrf/<workload>-inference:v4`), unified CUDA 12.1 base.

---

## 3. Runner (run_experiment_v4.py)

v4 is a copy-and-diverge from v3. v3 remains untouched and backward-compatible with A16 Phase 1 v3 and H100 Tier 1. Divergences from v3 are limited to the GPU query layer and the writer.

**Env vars (identical shape to v3):**
- `PROMETHEUS_URL` — default `http://172.22.174.66:30090`
- `DATA_OUTPUT_DIR` — default `data/raw/extension_tier2`
- `EXPERIMENT_AUTO_CONFIRM` — `1` skips prompts

**CLI:** `python tools/run_experiment_v4.py <workload> <replicas>`

**Constants:**
- Sample interval: 5 s
- Experiment duration: 3600 s (720 ticks)
- Expected rows per tick-based file: 700–720 (tolerance for scrape jitter)
- Pod ready timeout: 180 s default, 300 s Whisper

**Execution flow:**
1. Parse args, resolve env, echo run configuration.
2. Interactive confirm unless `EXPERIMENT_AUTO_CONFIRM=1`.
3. Apply `k8s/workloads/tier2/<workload>-mig.yaml`, scale to r.
4. Wait for pods Ready within timeout. Halt if not reached.
5. Sample every 5 s for 720 ticks, buffer in memory.
6. Write all CSV files to output directory.
7. Delete deployment, wait ≤60 s for pod termination. Halt on stragglers.
8. Print `EXPERIMENT COMPLETE: <workload> r=<n>`. Exit 0.

**Halt conditions inside v4:**
- Pod not Running within timeout (exit 2)
- Deployment cleanup timeout (exit 3)
- Prometheus HTTP error on 3 consecutive ticks (exit 4)
- Pod count drops to zero mid-experiment (exit 5)

**Non-halting conditions:**
- Pod crash with recovery to Running before minute 60: logged, batch continues.
- Prometheus gap within 3 consecutive ticks: null cell written, continue.

---

## 4. Batch orchestrator (run_tier2_batch.sh)

Unattended orchestration of all 35 experiments under `tmux` + `nohup`. Halt-on-first-failure. Resume via `.tier2_state.json` (gitignored).

**Config:** WORKLOADS=(bert gpt2 resnet152 whisper yolo), REPLICAS=(1..7), 35 (workload, r) pairs total.

**Startup preconditions:**
- Assert running under tmux; warn if not.
- Assert current branch is `extension-h100` (guard added after dry-run push side effect; see section 12).
- Verify MIG label `nvidia.com/mig.config=all-1g.12gb` and state `success`.
- Verify node capacity `nvidia.com/gpu: 7`.
- Load `.tier2_state.json`; halt if any entry is `in_progress` (defensive against half-completed writes).

**Per-experiment sub-flow:**
1. Mark `in_progress` in state file.
2. `pre_experiment_checklist.sh` + `clear_system_cache.sh`. Halt on nonzero.
3. Run v4 with `PYTHONUNBUFFERED=1` and env vars.
4. Validate: exit code 0, success marker in log, output dir exists, exactly 33 CSVs + 1 timestamps.txt, row counts in [700,720] for tick-based files (scaled by replicas for per-pod files, by 7 for per-slice files), no null cells in `value` column of aggregated files.
5. Wait 10 s, verify no straggler pods.
6. Re-verify MIG state unchanged.
7. Mark `success` in state file (before git commit, so a resumed batch cannot misread a completed run as crashed).
8. `git add`, `git commit`, `git push origin extension-h100`.
9. Append one line to `EXTENSION_JOURNAL.md`.

**Commit granularity:** one commit per experiment, message format `data: H100 Tier 2 <workload> r=<n> collected on devLab (MIG 1g.12gb)`. 35 data commits total.

**State file schema** (`.tier2_state.json`, gitignored, repo root):
```json
{
  "index": 0,
  "workload": "bert",
  "replicas": 1,
  "status": "success | pending | in_progress | failed | skipped",
  "csv_paths": [...],
  "started_at": "ISO 8601 UTC",
  "finished_at": "ISO 8601 UTC",
  "error_message": null
}
```

---

## 5. Step 0 — MIG DCGM discovery findings (critical for methodology section)

Before v4 was designed, a diagnostic pass on the H100 VM enabled MIG and probed which DCGM metrics survive MIG partitioning. Findings:

**5.1 MIG resource surface.** GPU Operator single-strategy exposes 7 slices under the generic name `nvidia.com/gpu: 7` (not `nvidia.com/mig-1g.12gb: 7`). Consequence: Tier 2 workload manifests are byte-for-byte copies of Tier 1 manifests — no resource-name change needed. Kernel guarantees each pod that requests `nvidia.com/gpu: 1` lands on one 1g.12gb slice.

**5.2 Legacy GPU_UTIL is not emitted per MIG instance on this build.** DCGM version 580.173.02 under GPU Operator v26.3.3 does not expose `DCGM_FI_DEV_GPU_UTIL` per compute instance on this H100. This is a known DCGM limitation: legacy SM-occupancy counters are not MIG-instance-aware. NVIDIA's GPM profiling API replaces them since Ampere.

**5.3 Substitute counters (per-instance, populated).** DCGM emits the following per-instance metrics with `GPU_I_ID`, `GPU_I_PROFILE`, `pod`, `namespace` labels:
- `DCGM_FI_PROF_GR_ENGINE_ACTIVE` (0.0–1.0) — graphics/compute engine active fraction. Closest analog to legacy `GPU_UTIL`.
- `DCGM_FI_PROF_DRAM_ACTIVE` (0.0–1.0) — memory bandwidth utilization.
- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (0.0–1.0) — tensor core active fraction.
- `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE` (MiB) — per-slice memory.
- `DCGM_FI_DEV_POWER_USAGE` (W), `DCGM_FI_DEV_GPU_TEMP` (C), `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` (mJ) — physical-GPU quantities replicated on every per-slice series.

**5.4 FB_TOTAL is not emitted per MIG instance.** Substitute: `gpu_memory_total` is written as a constant 11007 MiB (per-slice usable memory, verified by `FB_USED + FB_FREE` at idle).

**5.5 Pod-to-slice attribution is native.** DCGM per-instance series carry `pod=<pod-name>` and `namespace=<ns>` labels when a pod is running on that compute instance. Idle slices carry empty pod label. No node-annotation cross-reference needed.

**5.6 Slice memory reports as 1g.11gb.** DCGM reports `GPU_I_PROFILE=1g.11gb` (11 GB usable) even though the MIG config label is `all-1g.12gb`. NVIDIA reserves ~1 GB per instance for the CUDA context; DCGM reports usable memory.

---

## 6. Schema decisions (with rationale, for direct paste into methodology)

**6.1 File output shape.** 34 files per experiment: 33 CSVs plus one plain-text timestamps file.
- 17 non-GPU files (identical to Tier 1 and Phase 1 v3)
- 5 aggregated GPU files (Tier-1-compatible names and column layout)
- 3 aggregated Tier-2-new files (dram_active, pipe_tensor_active, total_energy_consumption)
- 5 per-slice files for the Tier-1 GPU metrics (Tier 2 addition)
- 3 per-slice files for the Tier-2-new metrics (Tier 2 addition)
- 1 timestamps.txt

**6.2 Aggregation rules per GPU metric.**
| Metric | Aggregation rule | Rationale |
|---|---|---|
| `gpu_utilization` | `sum(GR_ENGINE_ACTIVE across active slices) * 100` | Each slice's engine-active is genuinely independent; sum is honest. Max ~700 at r=7 fully busy. |
| `gpu_dram_active` | sum across active slices | Same reasoning. |
| `gpu_pipe_tensor_active` | sum across active slices | Same reasoning. |
| `gpu_memory_used` | sum across active slices (pod label non-empty) | "Memory used by this experiment's pods". Filtering out idle slices avoids ~14 MiB × 7 baseline noise. |
| `gpu_memory_total` | constant 11007 MiB | Single-slice constant. Preserves per-pod resource-envelope semantic consistent with A16 (16 GB) and Tier 1 (94 GB). Not summed, not physical. |
| `gpu_power_watts` | first-slice value (`GPU_I_ID=7`) | H100 power sensor is physical-GPU-scoped; DCGM replicates the value on each per-slice series. Sum would over-count 7x. |
| `gpu_temperature` | first-slice value | Same physical-GPU replication. |
| `gpu_total_energy_consumption` | first-slice value | Same. |

**6.3 Aggregated GPU CSV column layout.** Header order matches Tier 1 exactly:
`timestamp,value,DCGM_FI_DRIVER_VERSION,Hostname,UUID,[__name__,]container,device,gpu,instance,job,modelName,namespace,pci_bus_id,pod`

At r>=1, the `pod`, `container`, `namespace`, `pci_bus_id` cells are written as empty strings. Rationale: at r>1 multiple pods contribute to the aggregated row and no single value is meaningful. Empty cells preserve column order so any loader that reads by name works on both Tier 1 and Tier 2 without changes. See section 12 for the load_experiment.py normalisation.

**6.4 Per-slice CSV column layout.** Long-format:
`timestamp,value,GPU_I_ID,GPU_I_PROFILE,pod,namespace,container,DCGM_FI_DRIVER_VERSION,UUID`

One row per (timestamp, GPU_I_ID) pair. 720 timestamps × 7 slices = 5040 rows per file. Idle slices (pod label empty) are included with their raw values — this is intentional; downstream analysis decides whether to filter.

**6.5 Null policy.** Per-slice files legitimately contain nulls when a pod crashes and Prometheus has no sample for it at timestamp t (write-with-nulls, per design). Aggregated files have no nulls in the `value` column under normal operation. Both files preserve the (timestamp, pod) or (timestamp, slice) grid rather than dropping rows.

---

## 7. Metric semantics and cross-tier comparability (critical for paper)

The `gpu_utilization` column has the same name and same 0-based scale across all three tiers, but the underlying DCGM counter differs on Tier 2. This is a substitution, not a bug, and must be documented in any comparative analysis.

| Tier | Underlying counter | Scope | Scale |
|---|---|---|---|
| A16 Phase 1 v3 | `DCGM_FI_DEV_GPU_UTIL` (legacy SM occupancy) | whole GPU, time-sliced | 0–100 |
| H100 Tier 1 | `DCGM_FI_DEV_GPU_UTIL` (legacy SM occupancy) | whole GPU at r=1 | 0–100 |
| H100 Tier 2 | `DCGM_FI_PROF_GR_ENGINE_ACTIVE × 100` (GPM engine active) | per MIG instance, summed | 0–700 (r=7 fully busy) |
| H100 Tier 3 (planned) | back to legacy `GPU_UTIL` (whole GPU) | whole GPU, time-sliced | 0–100 |

Numerically comparable within limits. Semantically, GPM engine-active is a stricter definition than SM occupancy — it counts time when at least one SM is doing work, not the SM occupancy percentage. For low-utilization workloads (single-digit-percent range), the two counters read similarly. At high utilization they may diverge by up to ~10 percentage points. This is a documented NVIDIA-side caveat, not our measurement error.

**Recommendation for comparative plots:** treat Tier 2 `gpu_utilization` as a close proxy, not an exact match. Where possible, use `gpu_dram_active` and `gpu_pipe_tensor_active` as workload-specific auxiliary signals rather than folding them into a single "utilization" number.

---

## 8. Schema differences across tiers (for load_experiment.py aliasing)

Three column-level differences worth listing explicitly.

**8.1 GPU CSV extra columns.** Tier 1 and Tier 2 aggregated GPU CSVs carry four additional columns from GPU Operator's DCGM integration: `container`, `namespace`, `pci_bus_id`, `pod`. Phase 1 v3 does not. Values unchanged.

**8.2 `application` → `app` rename.** `pod_latency_avg` and `pod_throughput` renamed the workload label from `application` (Phase 1 v3) to `app` (Tier 1 and Tier 2, inherited from v4 container rebuild). Permanent split across phases.

**8.3 `pod` cell empty vs populated in aggregated GPU CSVs.** Tier 1 at r=1 writes the actual pod name in the `pod` column of aggregated GPU CSVs. Tier 2 at r>=1 writes empty. This is by design (section 6.3).

Downstream loader normalises all three differences into one canonical schema. See section 12.

---

## 9. Per-workload observations

**BERT.** GPU utilization at r=7 averages 1.83% with cross-pod spread of 0.04 — negligible contention, consistent with BERT's low compute footprint. PSI CPU ~0 throughout. Latency r=7/r=1 ratio 1.00 — no degradation under full replica load. Frozen S36 VR: 1.142 (pass, mild overshoot). None of the 4 ablation variants improves on frozen; best-of-5 remains frozen itself.

**GPT-2.** GPU utilization at r=7 averages 15.79% with spread of 0.58 — the largest absolute cross-pod variance of any workload (2.18), reflecting autoregressive per-inference variance that per-slice measurement preserves rather than averages away. PSI CPU ~0. Latency r=7/r=1 ratio 1.07. Frozen S36 VR: 0.612 (fail). Best ablation variant is fm_low at 0.616 — effectively unchanged; local hyperparameter search does not recover the frozen recipe.

**ResNet-152.** GPU utilization at r=7 averages 1.96% with spread of 0.07. PSI CPU ~0. Latency r=7/r=1 ratio 0.92. Frozen S36 VR: 0.369 — the worst of any workload across the full three-tier study. Best ablation variant is fm_high at 0.432, still well short of the 0.8 pass threshold. The local ablation neighborhood (vr ±0.2, fm ×0.5/×2.0) is insufficient to reach passing.

**Whisper.** GPU utilization at r=7 averages 5.71% with spread of 0.61. PSI CPU 0.32 — the only workload with non-negligible PSI, consistent with section 12.7's CPU-contention finding. Latency r=7/r=1 ratio 2.86 — the largest degradation of any workload; host CPU sharing under MIG (16 vCPUs across 7 pods, unlike GPU which is isolated per-slice) preserves substantial contention and consequently substantial per-pod variance. Frozen S36 VR: 1.106 (pass) — flips from Tier 3's failure to a clean Tier 2 pass. Best ablation variant is fm_low at 1.294.

**YOLO.** GPU utilization at r=7 averages 0.94% with spread of 0.03. PSI CPU ~0. Latency r=7/r=1 ratio 0.97. Frozen S36 VR: 0.774 — a marginal fail. Best ablation variant is fm_low at 0.957, a 23% improvement that crosses the pass threshold. Cleanest ablation-recovery story in the extension.

---

## 10. File inventory per experiment

Directory: `data/raw/extension_tier2/<workload>_r<n>/`

**17 non-GPU files (identical to Tier 1):**
- `app_latency_p50`, `app_latency_p95`, `app_latency_p99`, `app_throughput`
- `node_cpu_usage`, `node_memory_available_bytes`, `node_memory_used_percent`
- `node_psi_cpu`, `node_psi_io`, `node_psi_memory`
- `pod_cpu_usage`, `pod_latency_avg`, `pod_memory_bytes`
- `pod_psi_cpu`, `pod_psi_io`, `pod_psi_memory`, `pod_throughput`
- `<workload>_r<n>_<ts>_timestamps.txt` (18th non-CSV file)

**5 aggregated GPU files (Tier-1-compatible):**
- `gpu_utilization`, `gpu_memory_used`, `gpu_memory_total`, `gpu_power_watts`, `gpu_temperature`

**3 aggregated Tier-2-new files:**
- `gpu_dram_active`, `gpu_pipe_tensor_active`, `gpu_total_energy_consumption`

**5 per-slice files (Tier-1 metrics):**
- `gpu_utilization_per_slice`, `gpu_memory_used_per_slice`, `gpu_memory_total_per_slice`, `gpu_power_watts_per_slice`, `gpu_temperature_per_slice`

**3 per-slice files (Tier-2-new metrics):**
- `gpu_dram_active_per_slice`, `gpu_pipe_tensor_active_per_slice`, `gpu_total_energy_consumption_per_slice`

Total: 33 CSV + 1 timestamps.txt = 34 files per experiment. Naming convention: `<workload>_r<n>_<metric>_<YYYYMMDD_HHMMSS>.csv`.

---

## 11. Reproduction

**Single experiment:**
```bash
cd ~/generative-ai-workload-modeling
git checkout extension-h100
# verify MIG state
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl get node $NODE -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config}'
# should print: all-1g.12gb
# invoke runner directly
PROMETHEUS_URL=http://172.22.174.66:30090 \
DATA_OUTPUT_DIR=data/raw/extension_tier2 \
EXPERIMENT_AUTO_CONFIRM=1 \
~/miniconda3/envs/tracegen/bin/python tools/run_experiment_v4.py bert 1
```

**Full batch:**
```bash
cd ~/generative-ai-workload-modeling
git checkout extension-h100
tmux new -s tier2-batch
nohup bash tools/run_tier2_batch.sh > /tmp/tier2_batch.log 2>&1 &
# detach: Ctrl-B D
```

**Enable MIG (one-time cluster setup):**
```bash
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl label node $NODE nvidia.com/mig.config=all-1g.12gb --overwrite
# poll until state=success (typically ~30-40 s)
```

**Disable MIG (to run whole-GPU work again):**
```bash
kubectl label node $NODE nvidia.com/mig.config=all-disabled --overwrite
```

---

## 12. Incidents and lessons learned

**12.1 Dry-run push side effect.** During dry-run testing on a throwaway branch (`tier2-dryrun`), `git push origin extension-h100` in the batch script pushed the local `extension-h100` branch, not the currently-checked-out branch. Effect: the Tier 2 manifest commit (451b955) and .gitignore commit (1a10936) landed on remote a step earlier than the sanctioned final push. No harmful content pushed. Mitigation: added a branch guard to the batch script that halts if not on `extension-h100`. Guard commit: c71486c.

**12.2 Python stdout buffering.** First launch attempt had no visibility into per-experiment runner log until the experiment finished, because Python buffers stdout when piped to a file. Fix: added `PYTHONUNBUFFERED=1` to the runner env in the batch script. Same commit c71486c.

**12.3 File count validation off-by-one.** Spec said "33 files per experiment" (CSV only). Batch validation initially counted all files including timestamps.txt = 34, would have halted every experiment. Fix caught pre-launch: validation scoped to `*.csv` glob. No experiment lost.

**12.4 Row count scaling for per-pod and per-slice files.** Flat 700–720 row check would have halted every r>1 experiment (per-pod files legitimately have replicas × 720 rows) and every experiment with per-slice files (7 × 720 rows). Fix caught pre-launch: row bounds scale by file category.

**12.5 Aggregated GPU intentional empty cells.** Blanket null-cell check would have flagged the intentional empty `pod`/`container`/`namespace`/`pci_bus_id` cells at r>1. Fix: null-check scoped to `value` column only.

**12.6 mark_status ordering.** Original spec marked `success` after the git commit. If commit succeeded but push failed, state file would say `success` while remote didn't have the commit — a resumed batch would skip it. Fix caught pre-implementation: `mark_status(success)` moved before git commit; git failures halt the batch and the operator diagnoses.

**12.7 Whisper r=7 survival.** Highest-risk cell for CPU contention. On A16 Phase 1 v3 at r=10, whisper crashed with PSI=0.63. On MIG each pod gets its own slice so GPU memory is comfortable, but CPU is unchanged (still 16 vCPUs shared across 7 pods). Whisper r=7 completed the full 60-minute recording without triggering the batch halt. Post-hoc analysis of `whisper_r7_pod_psi_cpu` and `whisper_r7_pod_latency_avg` should show the degradation pattern.

---

## 13. Known limitations (for methodology section)

**13.1 Per-slice power and temperature are not truly per-slice.** H100 physical GPU has one power sensor and one temperature sensor. DCGM replicates these values on every per-slice series. Our per-slice CSVs preserve the raw DCGM output; aggregation takes first-slice only. Any comparative analysis of per-slice power/temperature should treat these as physical-GPU-scoped, not per-instance.

**13.2 `gpu_memory_total` is a hardcoded constant.** DCGM does not emit `FB_TOTAL` per MIG instance on this build. Constant 11007 MiB (per-slice usable) is written for every timestamp. Not queried from Prometheus.

**13.3 `gpu_utilization` semantics differ from Tier 1.** See section 7. GPM engine-active counter substitutes for legacy SM occupancy counter. Column name and scale preserved; underlying measurement changed.

**13.4 Pod-level latency percentiles remain unavailable.** Same limitation as Phase 1 v3 and Tier 1: only `pod_latency_avg` is collected. Prometheus data expired (15-day retention) precludes recovery of per-pod p50/p95/p99. Documented thesis limitation.

**13.5 Slice-to-pod mapping depends on GPU Operator DCGM version.** The `pod` label on per-instance DCGM series comes from GPU Operator v26.3.3's DCGM integration. Earlier GPU Operator versions may not populate this label; downstream reproducibility on other clusters may require the fallback path (query pod annotations for `nvidia.com/mig-*` resource claims).

**13.6 Business Day rate unchanged from Tier 1 / Phase 1 v3.** Per methodological answer during v4 design, workload inference rate was not scaled per-tier or per-workload. Apples-to-apples fidelity matters more than avoiding degradation at high r. Whisper r=7 CPU contention is a legitimate finding, not a measurement artefact.

**13.7 S36 retrain uses per-slice gpu_utilization column.** The Tier 2 retrain trains on `gpu_utilization_per_slice_*.csv`, not the aggregated `gpu_utilization_*.csv` (section 10). This preserves per-pod attribution, consistent with section 1's methodological headline and section 6.2's aggregation rules — the aggregated file serves whole-experiment reporting, not per-pod training. The alternative of dividing the aggregated file by MIG's fixed 7-slice count was considered and rejected: it would manually destroy the per-pod variance Tier 2 was designed to capture. See `S36_TIER2_EXTENSION_REFERENCE.md` section 2.1 for full justification and section 4.6 for the diagnostic quantifying what /7 would have destroyed.

---

## 14. Cross-tier dataset landscape (for paper's dataset section)

| Tier | Cluster | GPU config | Workloads | r range | Experiments | Files/exp | Per-pod GPU? |
|---|---|---|---|---|---|---|---|
| Phase 1 v3 | A16 VM | whole GPU, 10 time-slices | 5 | 1..10 (all int) | 50 | 22 CSV + timestamps | inferred (whole ÷ r) |
| Tier 1 | H100 devLab | whole GPU | 5 | 1 | 5 | 22 CSV + timestamps | trivial at r=1 |
| Tier 2 | H100 devLab | MIG 1g.12gb, 7 slices | 5 | 1..7 | 35 | 33 CSV + timestamps | direct per-slice |
| Tier 3 | H100 devLab | whole GPU + time-slice / Kostya's scheduler | 5 | TBD | TBD | 22 CSV + timestamps (expected) | inferred (whole ÷ r) |

Total dataset once Tier 3 complete: ~95 (workload, r) cells across three GPU sharing modes on two hardware generations.

---

## 15. Open questions for downstream work

**15.1 Tier 3 approach.** Kostya Smirnov's custom time-slicing scheduler is a candidate. Fallback is standard NVIDIA time-slicing on whole H100 for a clean comparison to A16 Phase 1 v3. Decision blocked on technical assessment of Kostya's scheduler.

**15.2 `load_experiment.py` aliasing details.** Canonical schema decision points documented in section 8. Implementation is the next deliverable after this doc.

**15.3 S36 retrain on Tier 2 vs Tier 1 vs Phase 1 v3.** Once loader is in, retrain S36 per-workload on each tier's data separately and cross-evaluate. The comparison is the core of the extension paper.

**15.4 Jan Hagemann's fidelity analysis.** Uses S36 generator trained on Phase 1 v3 to score how well synthetic traces match Tier 1 and Tier 2 real traces. Distinct workstream, blocked on loader.

**15.5 Kwok trace generation at r=10, r=50, r=100.** Deferred from Phase 4. Uses S36 trained on the extended dataset once available.

---

## 16. Commit index (for archival reference)

Infrastructure commits:
- `451b955` — Tier 2 workload manifests (byte-identical copies of Tier 1)
- `1a10936` — gitignore exception for extension_tier2 data and state file
- `77391b2` — feat: tools/run_experiment_v4.py and run_tier2_batch.sh for MIG Tier 2
- `c71486c` — fix: unbuffer python stdout in tier2 batch runner for live log visibility

Data commits: 35 in total, one per experiment, first `data: H100 Tier 2 bert r=1` at ~2026-08-14T17:05Z, last `data: H100 Tier 2 yolo r=7` at ~2026-08-16T06:29Z. See `git log --oneline --grep="Tier 2" origin/extension-h100`.

---

**End of TIER2_NOTES.md**