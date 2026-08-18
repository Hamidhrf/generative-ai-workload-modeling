# H100 Extension Journal

Known issue: tools/clear_system_cache.sh and tools/pre_experiment_checklist.sh use stale phase 1 v1 workload labels (resnet50, distilbert, whisper) — will miss current 5-workload pods. Deferred to H100 tooling task.

Aug 2 2026: Docker-side cleanup — removed 26 containerlab containers (clab-century-*), containerlab images (abdullahmuzlim279/k3s-serf-node, frrouting/frr), obsolete phase 1 v1 images (resnet50, distilbert). Docker builder prune reclaimed 41.32GB build cache. Total freed: 137GB (225GB → 88GB used, 96% → 38%). DiskPressure taint initially remained after Docker cleanup because CRI-O's imagefs is a separate store. Cleared after `sudo crictl rmi --prune` — the prune itself didn't free significant space (CRI-O storage held at 22GB, all in active use), but the earlier 137GB Docker reclaim was enough once kubelet re-evaluated. Final state: 96GB used / 139GB free (41%). bert-inference deployment applied at :v4, image already in CRI-O, smoke test ready to proceed.

Aug 2 2026: v4 container rebuild complete. All 5 workloads smoke-tested on A16 GPU with startup log confirming NVIDIA A16, compute_capability=(8, 6). Results:

  workload    | image                              | counter after 60s
  ------------|------------------------------------|-------------------
  bert        | hamidhrf/bert-inference:v4         | 13655 (pre-warmed)
  gpt2        | hamidhrf/gpt2-inference:v4         | 18
  resnet152   | hamidhrf/resnet152-inference:v4    | 54
  whisper     | hamidhrf/whisper-inference:v4      | 22
  yolo        | hamidhrf/yolo-inference:v4         | 23

Base image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime (all 5).
H100 compatibility: CUDA 12.1 runtime supports sm_90; startup log will confirm compute_capability=(9, 0) on first H100 run.


## Aug 10-11 2026: H100 VM provisioned and cluster brought up

VM: `devLab`, IP 172.22.174.66, Ubuntu 26.04, kernel 7.0.0-29, same subnet as A16 (172.22.174.0/24). NVIDIA H100 NVL 94GB PCIe passthrough confirmed via lspci.

**Host driver**: NVIDIA driver 580.173.02 installed via `nvidia-driver-580-server` (Ubuntu package). Kept intentionally on 580 major to match A16 driver line, minimizing driver as a variable in H100 vs A16 comparisons. Driver 580.65.06+ also required to avoid the known 570.x mixed-MIG-Pending scheduling bug.

**Kubernetes stack**: kubeadm/kubelet/kubectl v1.34.0 (held), CRI-O 1.31.5, crictl v1.31.1 — versions identical to A16. Single-node cluster, control-plane taint removed. Calico CNI v3.29.1, pod CIDR 10.244.0.0/16.

**GPU Operator**: v26.3.3 via Helm, installed with `driver.enabled=false` to use pre-installed host driver. v26.3.0+ introduces runtime MIG profile discovery via NVML — required for H100 NVL support (94GB variant uses 12gb slice naming, not the 10gb naming of the 80GB variant). All 11 operator pods healthy on first install.

**Node labels confirmed**:
- `nvidia.com/gpu.product=NVIDIA-H100-NVL`
- `nvidia.com/gpu.family=hopper`
- `nvidia.com/gpu.compute.major=9`, `minor=0`
- `nvidia.com/gpu.memory=95830` (94GB)
- `nvidia.com/mig.capable=true`
- `nvidia.com/mig.strategy=single`
- `nvidia.com/mig.config=all-disabled` (correct for Tier 1)
- `nvidia.com/gpu: 1` allocatable

**Tier 2 MIG profile correction**: original plan referenced `all-1g.10gb`, but that only exists on H100 80GB variant. H100 NVL 94GB uses `all-1g.12gb` (7 instances, ~10.75GB usable each). Same 7-slice single strategy, different profile name. Updated Tier 2 plan accordingly.

**Known cosmetic bug**: DCGM Exporter reports MIG profile as `1g.11gb` instead of `1g.12gb` on H100 NVL. Cosmetic label issue, does not affect metric values. Upstream tracked at NVIDIA/dcgm-exporter#544. Grafana dashboards using GPU_I_PROFILE label will need to accept `1g.11gb` string.

**FH network gotcha**: TLS interception on baltocdn.com prevents Helm apt repo. Worked around by installing Helm from GitHub tarball (v3.16.4). Same interception may hit other less-common domains during later work; k8s.io and github.com passed cleanly.

Cluster is ready for workload deployment. All 5 v4 images already on Docker Hub from the A16 rebuild task.

Aug 11 2026: H100 smoke test bert passed. Transient ErrImagePull on first pull attempt (hit quay.io mirror, unauthorized), auto-retried successfully against correct registry in 1.1s. Pod reached 1/1 Running, [startup] log confirmed device=NVIDIA H100 NVL, compute_capability=(9, 0). bert_inference_total counter=25.0 after warm-up. Scaled back to 0.

Aug 11 2026: H100 smoke test gpt2 passed. Pod reached 1/1 Running cleanly, no image pull issues. [startup] log confirmed device=NVIDIA H100 NVL, compute_capability=(9, 0). gpt2_inference_total counter=7.0 after warm-up. Scaled back to 0.

Aug 11 2026: H100 smoke test resnet152 passed. Pod reached 1/1 Running cleanly, no image pull issues. [startup] log confirmed device=NVIDIA H100 NVL, compute_capability=(9, 0). resnet152_inference_total counter=40.0 after warm-up. Scaled back to 0.

Aug 11 2026: H100 smoke test whisper passed. Used bumped 300s readiness timeout per protocol (heavier model load, ~460MB); pod reached 1/1 Running within that window. [startup] log confirmed device=NVIDIA H100 NVL, compute_capability=(9, 0). whisper_inference_total counter=12.0 after warm-up. Scaled back to 0.

Aug 11 2026: H100 smoke test yolo passed. Pod reached 1/1 Running cleanly, no image pull issues. [startup] log confirmed device=NVIDIA H100 NVL, compute_capability=(9, 0). yolo_inference_total counter=100.0 after warm-up. Scaled back to 0. All 5 v4 workloads (bert, gpt2, resnet152, whisper, yolo) now smoke-tested and passing on H100 NVL.

Aug 11 2026: monitoring stack deployed on H100. Selective subset of k8s/monitoring/: namespace, prometheus-config (with DCGM ns patched to gpu-operator), prometheus-deployment, node-exporter. Skipped grafana (unused), kube-state-metrics (unused), kepler (unused + eBPF risk on kernel 7.0), and both dcgm-exporter YAMLs (GPU Operator already provides one). All Prometheus targets UP: prometheus, node-exporter, dcgm-exporter (via gpu-operator ns), kubelet-cadvisor. ai-inference-apps target is empty pending workload deploys during experiments. Runner prometheus_url now env-configurable; A16 default preserved.

Repo bug found and fixed: prometheus-config.yaml was committed as raw Prometheus config (global:, scrape_configs:) rather than a Kubernetes ConfigMap manifest. Deploys silently relied on a manual kubectl create configmap step that was never committed. Wrapped the file as a proper ConfigMap so future kubectl apply -f works standalone.

local-path-provisioner v0.0.24 installed on H100 and set as default storageclass; A16's deploy-monitoring-stack.sh assumed it was already present but H100 was a fresh cluster.

Target list note: ai-inference-apps and kube-state-metrics scrape jobs use role: endpoints — when no matching Endpoints exist, no target entry is emitted at all (not shown as 'down'). ai-inference-apps will appear during experiments when workload pods are deployed; kube-state-metrics stays absent by design.

Aug 12 2026: H100 Tier 1 bert r=1 collected. Runtime ~65 min (matches estimate: 60-min recording + setup/teardown). 22/22 metrics collected, 715 rows each, zero nulls across all CSVs. All 6 Business Day load phases ran (NIGHT/RAMP/MORNING/LUNCH/PEAK/EVENING); phase boundaries are not persisted as a CSV column and must be reconstructed from timestamps, consistent with A16. gpu_utilization peaked at 4% (mean 0.87%) — vs A16's r=1 baseline of 2%, confirming H100 is even more over-provisioned for light workloads like BERT-base than A16 was, a key thesis comparison point. gpu_memory_used flat/constant for the full run — expected, same as A16 (PyTorch caching allocator claims once and holds). Runner's cleanup() does a full `kubectl delete deployment`, not scale-to-0 — different from the smoke-test workflow, so the deployment YAML needs a fresh `kubectl apply` before each subsequent Tier 1 workload run.

Aug 13 2026: H100 Tier 1 batch (gpt2, resnet152, whisper, yolo, r=1) collected via autopilot. All 4 exited cleanly with [OK] Experiment successful, 22/22 metrics each, ~715 rows, zero nulls. Runtimes: gpt2 ~68 min, resnet152 ~67 min, whisper ~61 min, yolo ~67 min. Runner's internal recording phase was exactly 3600s for all runs — variation is from setup/teardown.

gpu_utilization peak across Tier 1 (all r=1): bert 4%, gpt2 24%, resnet152 4%, whisper 26%, yolo 2%. Transformer/attention models (gpt2, whisper) show higher peak utilization than classification/embedding models (bert, resnet152, yolo), all under 30% on H100 at r=1. Whisper's peak drop from 65% (A16) to 26% (H100) is notable — on A16 whisper was CPU-bound (PSI=0.63 at r=10, pod crashes); on H100's faster CPU the workload profile changes and GPU becomes the sole bound resource but is under-utilized. Hardware upgrades reclassify resource bounds, not just accelerate.

Batch wall-clock: ~6h43m for 4 runs vs. naive ~4h20m estimate. ~2h23m accumulated in inter-step scheduling overhead (most notably ~26 min before yolo Step 1, ~95 min between yolo exit and final push). Runner and cluster behavior was uniform; overhead is on the agent scheduling side. Relevant for Tier 2 batch estimation (35 runs).

## Tier 2 batch started 2026-08-14T15:59:47Z — 35 experiments (index 0 onward)
- 2026-08-14T17:05:43Z Tier 2 bert r=1: 33 csv files, mean rows=1767
- 2026-08-14T18:11:42Z Tier 2 bert r=2: 33 csv files, mean rows=1920
- 2026-08-14T19:17:40Z Tier 2 bert r=3: 33 csv files, mean rows=2072
- 2026-08-14T20:23:39Z Tier 2 bert r=4: 33 csv files, mean rows=2225
- 2026-08-14T21:29:38Z Tier 2 bert r=5: 33 csv files, mean rows=2378
- 2026-08-14T22:35:38Z Tier 2 bert r=6: 33 csv files, mean rows=2530
- 2026-08-14T23:41:37Z Tier 2 bert r=7: 33 csv files, mean rows=2683
- 2026-08-15T00:47:36Z Tier 2 gpt2 r=1: 33 csv files, mean rows=1767
- 2026-08-15T01:53:35Z Tier 2 gpt2 r=2: 33 csv files, mean rows=1920
- 2026-08-15T02:59:33Z Tier 2 gpt2 r=3: 33 csv files, mean rows=2072
- 2026-08-15T04:05:32Z Tier 2 gpt2 r=4: 33 csv files, mean rows=2225
- 2026-08-15T05:11:32Z Tier 2 gpt2 r=5: 33 csv files, mean rows=2378
- 2026-08-15T06:17:31Z Tier 2 gpt2 r=6: 33 csv files, mean rows=2530
- 2026-08-15T07:23:31Z Tier 2 gpt2 r=7: 33 csv files, mean rows=2683
- 2026-08-15T08:29:31Z Tier 2 resnet152 r=1: 33 csv files, mean rows=1767
- 2026-08-15T09:35:30Z Tier 2 resnet152 r=2: 33 csv files, mean rows=1920
- 2026-08-15T10:41:28Z Tier 2 resnet152 r=3: 33 csv files, mean rows=2072
- 2026-08-15T11:47:29Z Tier 2 resnet152 r=4: 33 csv files, mean rows=2225
- 2026-08-15T12:53:28Z Tier 2 resnet152 r=5: 33 csv files, mean rows=2378
- 2026-08-15T13:59:26Z Tier 2 resnet152 r=6: 33 csv files, mean rows=2530
- 2026-08-15T15:05:25Z Tier 2 resnet152 r=7: 33 csv files, mean rows=2683
- 2026-08-15T16:11:24Z Tier 2 whisper r=1: 33 csv files, mean rows=1767
- 2026-08-15T17:17:23Z Tier 2 whisper r=2: 33 csv files, mean rows=1920
- 2026-08-15T18:23:22Z Tier 2 whisper r=3: 33 csv files, mean rows=2072
- 2026-08-15T19:29:21Z Tier 2 whisper r=4: 33 csv files, mean rows=2225
- 2026-08-15T20:35:21Z Tier 2 whisper r=5: 33 csv files, mean rows=2378
- 2026-08-15T21:41:21Z Tier 2 whisper r=6: 33 csv files, mean rows=2530
- 2026-08-15T22:47:21Z Tier 2 whisper r=7: 33 csv files, mean rows=2683
- 2026-08-15T23:53:20Z Tier 2 yolo r=1: 33 csv files, mean rows=1767
- 2026-08-16T00:59:19Z Tier 2 yolo r=2: 33 csv files, mean rows=1920
- 2026-08-16T02:05:18Z Tier 2 yolo r=3: 33 csv files, mean rows=2072
- 2026-08-16T03:11:17Z Tier 2 yolo r=4: 33 csv files, mean rows=2225
- 2026-08-16T04:17:16Z Tier 2 yolo r=5: 33 csv files, mean rows=2378
- 2026-08-16T05:23:15Z Tier 2 yolo r=6: 33 csv files, mean rows=2530
- 2026-08-16T06:29:15Z Tier 2 yolo r=7: 33 csv files, mean rows=2683

## Tier 2 batch complete — 2026-08-16

- 35/35 experiments successful (5 workloads x r=1..7)
- Zero failures, whisper r=7 survived (CPU-bound but did not crash mid-run)
- Total elapsed: 1 day 14h 29m; mean 65m 59s per experiment
- Output: data/raw/extension_tier2/<workload>_r<n>/ (34 files each: 33 CSV + 1 timestamps.txt)
- Per-slice GPU attribution recovered natively via DCGM per-instance labels
- MIG config stable throughout at all-1g.12gb (7 slices)

## Tier 3 batch started 2026-08-16T10:19:05Z — 50 experiments (index 0 onward)
- 2026-08-16T11:25:01Z Tier 3 bert r=1: 22 CSV files, mean rows=715
- 2026-08-16T12:31:00Z Tier 3 bert r=2: 22 CSV files, mean rows=942
- 2026-08-16T13:36:59Z Tier 3 bert r=3: 22 CSV files, mean rows=1170
- 2026-08-16T14:42:58Z Tier 3 bert r=4: 22 CSV files, mean rows=1397
- 2026-08-16T15:48:56Z Tier 3 bert r=5: 22 CSV files, mean rows=1625
- 2026-08-16T16:54:55Z Tier 3 bert r=6: 22 CSV files, mean rows=1852
- 2026-08-16T18:00:54Z Tier 3 bert r=7: 22 CSV files, mean rows=2080
- 2026-08-16T19:06:54Z Tier 3 bert r=8: 22 CSV files, mean rows=2307
- 2026-08-16T20:12:54Z Tier 3 bert r=9: 22 CSV files, mean rows=2532
- 2026-08-16T21:18:53Z Tier 3 bert r=10: 22 CSV files, mean rows=2754
- 2026-08-16T22:24:52Z Tier 3 yolo r=1: 22 CSV files, mean rows=715
- 2026-08-16T23:30:50Z Tier 3 yolo r=2: 22 CSV files, mean rows=942
- 2026-08-17T00:36:49Z Tier 3 yolo r=3: 22 CSV files, mean rows=1170
- 2026-08-17T01:42:48Z Tier 3 yolo r=4: 22 CSV files, mean rows=1397
- 2026-08-17T02:48:46Z Tier 3 yolo r=5: 22 CSV files, mean rows=1625
- 2026-08-17T03:54:46Z Tier 3 yolo r=6: 22 CSV files, mean rows=1852
- 2026-08-17T05:00:45Z Tier 3 yolo r=7: 22 CSV files, mean rows=2080
- 2026-08-17T06:06:44Z Tier 3 yolo r=8: 22 CSV files, mean rows=2307
- 2026-08-17T07:12:44Z Tier 3 yolo r=9: 22 CSV files, mean rows=2535
- 2026-08-17T08:18:44Z Tier 3 yolo r=10: 22 CSV files, mean rows=2762
- 2026-08-17T09:24:43Z Tier 3 resnet152 r=1: 22 CSV files, mean rows=715
- 2026-08-17T10:30:42Z Tier 3 resnet152 r=2: 22 CSV files, mean rows=942
- 2026-08-17T11:36:41Z Tier 3 resnet152 r=3: 22 CSV files, mean rows=1170
- 2026-08-17T12:42:39Z Tier 3 resnet152 r=4: 22 CSV files, mean rows=1397
- 2026-08-17T13:48:38Z Tier 3 resnet152 r=5: 22 CSV files, mean rows=1625
- 2026-08-17T14:54:36Z Tier 3 resnet152 r=6: 22 CSV files, mean rows=1852
- 2026-08-17T16:00:35Z Tier 3 resnet152 r=7: 22 CSV files, mean rows=2080
- 2026-08-17T17:06:35Z Tier 3 resnet152 r=8: 22 CSV files, mean rows=2307
- 2026-08-17T18:12:34Z Tier 3 resnet152 r=9: 22 CSV files, mean rows=2535
- 2026-08-17T19:18:34Z Tier 3 resnet152 r=10: 22 CSV files, mean rows=2762
- 2026-08-17T20:24:34Z Tier 3 gpt2 r=1: 22 CSV files, mean rows=715
- 2026-08-17T21:30:33Z Tier 3 gpt2 r=2: 22 CSV files, mean rows=942
- 2026-08-17T22:36:32Z Tier 3 gpt2 r=3: 22 CSV files, mean rows=1170
- 2026-08-17T23:42:31Z Tier 3 gpt2 r=4: 22 CSV files, mean rows=1397
- 2026-08-18T00:48:31Z Tier 3 gpt2 r=5: 22 CSV files, mean rows=1625
- 2026-08-18T01:54:31Z Tier 3 gpt2 r=6: 22 CSV files, mean rows=1852
- 2026-08-18T03:00:31Z Tier 3 gpt2 r=7: 22 CSV files, mean rows=2079
- 2026-08-18T04:06:31Z Tier 3 gpt2 r=8: 22 CSV files, mean rows=2304
- 2026-08-18T05:12:33Z Tier 3 gpt2 r=9: 22 CSV files, mean rows=2527
- 2026-08-18T06:18:33Z Tier 3 gpt2 r=10: 22 CSV files, mean rows=2744
