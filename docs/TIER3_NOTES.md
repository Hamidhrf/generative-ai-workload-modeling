# Tier 3 Notes — H100 whole-GPU with NVIDIA time-slicing

**Status:** Complete
**Batch dates:** 2026-08-16T10:19:05Z to 2026-08-18T17:47:00Z (~2 days 7h)
**Cluster:** devLab (172.22.174.66), NVIDIA H100 NVL 94 GB, time-slicing at 10 replicas per physical GPU
**Branch:** extension-h100
**Commit range:** 8ede589 (Tier 3 manifests) through the 50th data commit
**Runner:** tools/run_experiment_v3.py (unchanged from Phase 1 v3, backward-compatible)
**Batch orchestrator:** tools/run_tier3_batch.sh
**Data location:** data/raw/extension_tier3/<workload>_r<n>/

---

## 1. Executive summary

Tier 3 collected 50 experiments (5 workloads × r=1..10) on H100 with standard NVIDIA time-slicing at 10 virtual replicas per physical GPU. This is the apples-to-apples H100 counterpart to A16 Phase 1 v3: same replica range, same GPU sharing mode, same 60-minute Business Day load pattern, same 5-second sampling. Zero failures, zero row-count warnings, ~2 days 7h total wall clock, mean 1h 6m per experiment.

The methodological purpose: isolate the hardware axis by holding sharing mode constant. Any difference between Phase 1 v3 and Tier 3 attributes to A16 → H100 hardware generation, since everything else is held equal. This is the third arm of the sharing-mode-as-axis story that the extension paper tells (Phase 1 v3 = time-slicing on A16, Tier 1 = no sharing on H100, Tier 2 = MIG hard partitioning on H100, Tier 3 = time-slicing on H100).

50 output directories, 23 files each (22 CSV + 1 timestamps.txt), all commits pushed to origin/extension-h100.

---

## 2. Cluster state during Tier 3

Node: devlab, Ubuntu 26.04, kernel 7.0.0-29.
Kubernetes: 1.34.0 with CRI-O 1.31.5, Calico CNI.
GPU Operator: v26.3.3 (driver.enabled=false; host driver 580.173.02).
MIG configuration: `nvidia.com/mig.config=all-disabled` (disabled before Tier 3).
Time-slicing configuration: ConfigMap `time-slicing-config` in `gpu-operator` namespace, referenced by ClusterPolicy `cluster-policy` via `spec.devicePlugin.config.name=time-slicing-config` and `.config.default=any`. Config spec:

sharing:
timeSlicing:
resources:
- name: nvidia.com/gpu
replicas: 10

Node capacity: `nvidia.com/gpu: 10` (10 time-slice replicas of the single physical H100).
Node labels: `nvidia.com/gpu.product=NVIDIA-H100-NVL-SHARED`, `nvidia.com/gpu.replicas=10`, `nvidia.com/gpu.sharing-strategy=time-slicing`.
Monitoring stack: unchanged from Tier 1 and Tier 2 (Prometheus, DCGM Exporter via GPU Operator, node-exporter, workload endpoints on port 8000).
Workload containers: v4 tag on DockerHub (`hamidhrf/<workload>-inference:v4`), same containers as Tier 1 and Tier 2.

---

## 3. Runner (run_experiment_v3.py, unchanged)

Tier 3 reuses v3 as-is. No new runner. v3's original design — 60-minute Business Day pattern, 5-second sampling, final `query_range` call, 22 per-metric CSV output — is exactly what apples-to-apples with Phase 1 v3 requires.

**Env vars (identical to Phase 1 v3):**
- `PROMETHEUS_URL` — default `http://172.22.174.66:30090`
- `DATA_OUTPUT_DIR` — set to `data/raw/extension_tier3` by batch
- `EXPERIMENT_AUTO_CONFIRM=1` — bypass interactive prompts

**CLI:** `python tools/run_experiment_v3.py <workload> <replicas>`

**Constants:**
- Sample interval: 5 s (implicit in query_range step)
- Experiment duration: 3600 s
- Expected rows per file: 700–720
- Pod ready timeout: 180 s default, 300 s Whisper

**Execution flow (identical to Phase 1 v3):**
1. Apply workload manifest, scale to r replicas.
2. Wait for pods Ready.
3. Sleep in 5-minute chunks for 60 minutes (no live tick sampling).
4. One `query_range` call per metric at the end.
5. Write 22 CSV + 1 timestamps.txt.
6. Delete deployment.

**Architectural difference from v4:** no live pod-health monitoring during the recording window. A crashed-and-restarted pod appears in the final query as an additional pod identity, fragmenting `sum by (pod)` output. See section 5 for the validation logic that handles this.

---

## 4. Batch orchestrator (run_tier3_batch.sh)

Structurally near-identical to `run_tier2_batch.sh` with specific adjustments for the v3 runner and time-slicing precondition.

**Config:** WORKLOADS=(bert yolo resnet152 gpt2 whisper), REPLICAS=(1..10), 50 (workload, r) pairs total.

**Workload order rationale:** low-risk workloads first. bert (light) and yolo (very light) come first so any cluster or runner regression surfaces in the first ~11 hours. Whisper (highest crash risk from A16 Phase 1 v3 at r=10) runs last, so a Whisper halt still preserves 40 completed experiments.

**Precondition function `check_timeslicing_state`:** verifies THREE signals in one query to defeat the race condition observed during Phase A of the cluster reconfiguration (where the mig.config.state label read "success" before the capacity had actually updated):
- `nvidia.com/mig.config == "all-disabled"`
- Node capacity `nvidia.com/gpu == 10`
- Node label `nvidia.com/gpu.replicas == 10`

All three must agree before the precondition passes.

**Per-experiment sub-flow:** same as Tier 2 except:
- Runner is v3 (not v4).
- Expected file count: 22 CSV + 1 timestamps.txt (not 33 CSV).
- Row-count validation: universal relaxed check on per-pod files (see section 5).
- `check_timeslicing_state` replaces `check_mig_state` at startup and after each experiment.
- Defensive git add verification (see section 5.3).
- Commit message format: `data: H100 Tier 3 <workload> r=<n> collected on devLab (time-slicing 10)`.

**State file:** `.tier3_state.json` at repo root, gitignored, JSON list, same schema as Tier 2's `.tier2_state.json`.

**.gitignore updates:**

!data/raw/extension_tier3/
!data/raw/extension_tier3/**
.tier3_state.json


---

## 5. Design decisions specific to Tier 3

**5.1 Universal relaxed per-pod row validation.** v3's `query_range` architecture returns rows for both the crashed and the restarted pod when a Deployment auto-restarts a mid-experiment failure. A tight 700–720 × r row band would false-positive-halt on legitimate pod churn. Rule applied universally across all 50 experiments:

- Extract `distinct_pod_count` from the file's `pod` column (last column).
- Compute `rows_per_pod = total_rows / distinct_pod_count`.
- If `rows_per_pod < 500`: HALT (something genuinely broken).
- If `rows_per_pod > 1080` (=720 × 1.5): LOG WARNING to state file's `error_message`, do NOT halt. Cell counts as success.
- Otherwise: pass silently.

**Outcome:** 50 experiments, 0 halts, 0 warnings. Every per-pod file landed inside the 500–1080 band. No pod restarts occurred (or if they did, they finished within the tolerance).

Non-per-pod files (app_*, node_*, gpu_*) kept the standard 700–720 row check unchanged.

**5.2 Race-condition-resistant precondition check.** During Phase A cluster reconfiguration (MIG disable), Sonnet observed the `mig.config.state` label reading "success" for a moment before the node capacity had actually updated to reflect the new geometry. `check_timeslicing_state` reads all three signals (mig label, capacity, replicas label) in one shot; all three must agree. Prevents false-positive precondition passes.

**5.3 Defensive git add verification.** After `git add "$DIR" "$JOURNAL"` and before `git commit`, the script verifies that at least one file from `$DIR` is staged. If none, halt with `git add staged 0 files from $DIR (path may be gitignored)`. Rationale: caught during the Tier 3 dry-run — when `DATA_DIR` was pointed at an unexempted gitignore path, `git add` silently skipped 23 files and committed only the journal line, then pushed. On a real 50-experiment batch this could produce 50 data-less commits with no halt. Fix is small and defensive.

---

## 6. Schema and metric semantics

Tier 3 file set is identical to Tier 1 and Phase 1 v3: 22 per-metric CSVs + 1 timestamps.txt per experiment.

**GPU CSV column layout:** matches Tier 1 exactly (14 columns including GPU Operator's `container`, `namespace`, `pci_bus_id`, `pod` labels). This is a difference from Phase 1 v3 (11 columns, no GPU Operator fields). See TIER2_NOTES.md section 8 for the full cross-tier schema comparison table — Tier 3 follows Tier 1's layout.

**`gpu_utilization` counter:** back to legacy `DCGM_FI_DEV_GPU_UTIL` (0–100 percent), same as A16 Phase 1 v3 and Tier 1. On whole-GPU + time-slicing, the per-instance limitation that forced Tier 2 to substitute GPM `GR_ENGINE_ACTIVE` does not apply. This restores direct comparability with A16 Phase 1 v3 for the paper's core comparison.

**Per-pod GPU attribution:** same limitation as A16 Phase 1 v3. Time-slicing hides pod ownership from DCGM. Whole-GPU value is the only signal; per-pod estimates require dividing by replica count. Documented as a limitation, not a defect. Tier 2 remains the only tier with native per-pod GPU attribution.

**`application` → `app` rename:** inherited from the v4 container rebuild. Same as Tier 1 and Tier 2. `load_experiment.py` aliases both to a single canonical schema.

---

## 7. Per-workload observations (populated post-hoc after data loading)

Placeholder for after cross-tier analysis begins. Structure:

- BERT r=1..10: [pending] — expect linear GPU scaling, stable latency, matching A16 pattern.
- GPT-2 r=1..10: [pending] — expect non-linear GPU saturation, likely reaches ceiling earlier on H100 due to raw compute.
- ResNet-152 r=1..10: [pending] — expect balanced scaling, matches A16 pattern.
- Whisper r=1..10: [pending — priority] — A16 crashed at r=10 with PSI=0.63. H100 has more compute; expect similar CPU-bound pattern but possibly milder crash signature. 50/50 success with 0 warnings suggests no fatal crash occurred, but crash-and-recover within tolerance is possible.
- YOLO r=1..10: [pending] — expect very light resource usage.

Each entry: peak GPU utilization, peak GPU memory used, peak pod CPU, peak PSI CPU, peak per-pod latency, crash pattern if any, hardware-attributable difference vs A16.

---

## 8. Cross-tier dataset landscape (updated with Tier 3)

| Tier | Cluster | GPU config | Workloads | r range | Experiments | Files/exp | Per-pod GPU? |
|---|---|---|---|---|---|---|---|
| Phase 1 v3 | A16 VM | whole GPU, 10 time-slices | 5 | 1..10 (all int) | 50 | 22 CSV + timestamps | inferred (whole ÷ r) |
| Tier 1 | H100 devLab | whole GPU | 5 | 1 | 5 | 22 CSV + timestamps | trivial at r=1 |
| Tier 2 | H100 devLab | MIG 1g.12gb, 7 slices | 5 | 1..7 | 35 | 33 CSV + timestamps | direct per-slice |
| Tier 3 | H100 devLab | whole GPU, time-slicing 10 | 5 | 1..10 | 50 | 22 CSV + timestamps | inferred (whole ÷ r) |
| Tier 4 (planned) | H100 devLab | whole GPU + Kostya's custom scheduler | 5 | TBD | TBD | 22 CSV + timestamps (expected) | scheduler-dependent |

**Total dataset:** 140 experiments across four GPU sharing modes on two hardware generations. Tier 4 (Kostya's scheduler) is the fifth arm if it materialises.

---

## 9. Reproduction

**Single Tier 3 experiment:**
```bash
cd ~/generative-ai-workload-modeling
git checkout extension-h100
# verify time-slicing state
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl get node $NODE -o jsonpath='{.status.capacity.nvidia\.com/gpu}'
# should print: 10
PROMETHEUS_URL=http://172.22.174.66:30090 \
DATA_OUTPUT_DIR=data/raw/extension_tier3 \
EXPERIMENT_AUTO_CONFIRM=1 \
~/miniconda3/envs/tracegen/bin/python tools/run_experiment_v3.py bert 1
```

**Full batch:**
```bash
cd ~/generative-ai-workload-modeling
git checkout extension-h100
nohup bash tools/run_tier3_batch.sh > /tmp/tier3_batch.log 2>&1 &
```

**Enable time-slicing (one-time cluster setup, from MIG-disabled or fresh state):**
```bash
# 1. Disable MIG (if enabled)
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl label node $NODE nvidia.com/mig.config=all-disabled --overwrite
# wait for state=success

# 2. Create ConfigMap
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 10
EOF

# 3. Patch ClusterPolicy
kubectl patch clusterpolicy cluster-policy \
  --type merge \
  -p '{"spec":{"devicePlugin":{"config":{"name":"time-slicing-config","default":"any"}}}}'

# 4. Wait for device plugin restart and capacity update to 10
```

**Disable time-slicing (to run whole-GPU or MIG work again):**
```bash
kubectl patch clusterpolicy cluster-policy \
  --type json \
  -p '[{"op":"remove","path":"/spec/devicePlugin/config"}]'
```

---

## 10. Incidents and lessons learned

**10.1 Race-condition-resistant precondition — carry forward.** During Phase A cluster reconfiguration, the `mig.config.state` label read "success" for a moment before the node capacity had actually updated. This informed `check_timeslicing_state`'s three-signal design. Applies equally to any future tier's precondition check.

**10.2 Defensive git add verification — carry forward.** Dry run caught that `git add` silently skips gitignored paths. Any future batch script should include the `STAGED_FROM_DIR > 0` check between `git add` and `git commit`.

**10.3 Held-back commits pushed as side effect during dry run.** Same pattern as Tier 2 dry run: `git push origin extension-h100` pushes the local `extension-h100` ref regardless of currently-checked-out branch. Two commits (manifests, gitignore) that were queued locally with "do not push yet" instruction went to remote as a side effect of the dry run's push. Content was pre-reviewed so no harm, but lesson: do not hold unpushed commits on `extension-h100` when running a dry run on another branch.

**10.4 Zero warnings on the batch is unusual.** Tier 2 had 0 halts but the per-pod row validation was tight (rows == 720 × r ± jitter). Tier 3's relaxed validation gives more room, and 50 experiments landed comfortably inside the 500–1080 band with 0 warnings. Suggests time-slicing at 10 replicas on H100 handled all 5 workloads at every replica count without pod restarts — including Whisper r=10, the highest-risk cell.

---

## 11. Known limitations (for methodology section)

**11.1 Per-pod GPU attribution unavailable.** Same limitation as A16 Phase 1 v3. Time-slicing hides pod ownership from DCGM. Whole-GPU value is the only per-timestamp GPU signal; per-pod estimates come from dividing by replica count. Tier 2 remains the only tier with native per-pod GPU attribution.

**11.2 Pod-level latency percentiles remain unavailable.** Same limitation as Phase 1 v3, Tier 1, and Tier 2: only `pod_latency_avg` is collected. Prometheus retention (15 days) precludes recovery of per-pod p50/p95/p99 for expired data.

**11.3 Business Day rate unchanged.** Per methodological consistency, workload inference rate was not scaled per-tier. Whisper r=10 on Tier 3 is directly comparable to Whisper r=10 on A16 Phase 1 v3 — same rate, same replica count, same 60-min pattern.

**11.4 Hardware confounding on Phase 1 v3 vs Tier 3 comparison.** The intended isolated comparison is A16 vs H100 with time-slicing held constant. However, the H100 has ~7× more compute, ~6× more memory, and a newer CUDA/driver stack than the A16. Attributing behavioural differences purely to hardware generation is defensible for coarse effects (throughput, latency), but fine-grained microarchitectural claims should be avoided in the paper.

---

## 12. Commit index (for archival reference)

Infrastructure commits:
- `8ede589` — Tier 3 workload manifests (byte-identical copies of Tier 2 mig)
- `9a70b08` — gitignore exception for extension_tier3 data and state file
- `09c4026` — feat: tools/run_tier3_batch.sh for H100 Tier 3 time-slicing batch

Data commits: 50 in total, one per experiment, message format `data: H100 Tier 3 <workload> r=<n> collected on devLab (time-slicing 10)`. First `bert r=1` at ~2026-08-16T11:25Z, last `whisper r=10` at ~2026-08-18T17:47Z. See `git log --oneline --grep="Tier 3 " origin/extension-h100`.

---

**End of TIER3_NOTES.md**