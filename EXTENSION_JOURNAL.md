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
