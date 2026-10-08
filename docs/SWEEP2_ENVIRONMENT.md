# Sweep 2 Environment Record

Captured: 2026-10-08 on devLab. Read-only preflight for the second
data-collection sweeps. No cluster state was changed to produce this record.

## Provenance notes: sweep 2 vs. August

- **Driver:** sweep 2 runs on NVIDIA driver **580.178.04**. The August Tier 2 and
  Tier 3 data were collected on **580.173.02** (section 1).
- **kube-prometheus-stack:** present since 2026-09-16 (namespace `monitoring`,
  Prometheus v3.11.3 at a 30 s scrape interval, plus a second node-exporter,
  Grafana and kube-state-metrics). It is **not** a data source. The sweep
  queries the August Prometheus at `172.22.174.66:30090` (v2.48.0, created
  2026-08-11, 5 s scrape, same jobs as August). It adds a small background load
  that the August sweep did not have.
- **gpu-gatekeeper:** the `gpu-gatekeeper` DaemonSet (added 2026-09-16) is taken
  off the node for **both** sweeps with Kostya's approval (Checklist A step 2a),
  and restored afterwards; Kostya re-enables it himself. The companion
  `gpu-scheduler` stays running but is disabled and has no jobs (step 2a).

## 1. Hardware / driver

`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`

```
name, memory.total [MiB], driver_version
NVIDIA H100 NVL, 95830 MiB, 580.178.04
```

Driver delta since August: all GPU CSVs under `data/raw/extension_tier2/`
(560 files) and `data/raw/extension_tier3/` (250 files) carry a single
`DCGM_FI_DRIVER_VERSION = 580.173.02`. The node now reports **580.178.04**, so
sweep 2 will be collected on a newer driver than the August sweep. Record this
in the sweep 2 provenance.

## 2. Kubernetes

`kubectl version`

```
Client Version: v1.34.0
Kustomize Version: v5.7.1
Server Version: v1.34.0
```

Node (`kubectl get nodes -o wide`):

| Field | Value |
|-------|-------|
| Name | `devlab` |
| Roles | control-plane |
| Internal IP | 172.22.174.66 |
| OS image | Ubuntu 26.04 LTS |
| Kernel | 7.0.0-31-generic |
| Container runtime | cri-o://1.31.5 |
| kubelet | v1.34.0 |

## 3. GPU operator stack (namespace `gpu-operator`)

GPU Operator version: **v26.3.3** (Helm release `gpu-operator-1786444857`;
ClusterPolicy `cluster-policy`, `app.kubernetes.io/version=v26.3.3`).

Images, from `kubectl get daemonset -n gpu-operator -o wide`:

| Component | Image |
|-----------|-------|
| gpu-operator | `nvcr.io/nvidia/gpu-operator:v26.3.3` |
| nvidia-dcgm-exporter | `nvcr.io/nvidia/k8s/dcgm-exporter:4.5.3-4.8.2-distroless` |
| nvidia-device-plugin-daemonset | `nvcr.io/nvidia/k8s-device-plugin:v0.19.3` |
| gpu-feature-discovery | `nvcr.io/nvidia/k8s-device-plugin:v0.19.3` |
| nvidia-container-toolkit-daemonset | `nvcr.io/nvidia/k8s/container-toolkit:v1.19.1` |
| nvidia-mig-manager | `nvcr.io/nvidia/cloud-native/k8s-mig-manager:v0.14.2` |
| nvidia-node-status-exporter | `nvcr.io/nvidia/gpu-operator:v26.3.3` |
| nvidia-operator-validator | `nvcr.io/nvidia/gpu-operator:v26.3.3` |
| node-feature-discovery | `registry.k8s.io/nfd/node-feature-discovery:v0.18.3` |

Running pods of interest:
- `nvidia-dcgm-exporter-wzxc8` (1/1 Running, 22d)
- `nvidia-device-plugin-daemonset-vhksl` (2/2 Running, 22d)
- `nvidia-mig-manager-ghkbb` (1/1 Running)
- `gpu-operator-c5f9d98f7-phfg4` (1/1 Running)

The MPS control daemon (`nvidia-device-plugin-mps-control-daemon`) is scaled to
0 (MPS not in use; `nvidia.com/mps.capable=false`).

## 4. Container images (CRI-O)

`sudo crictl images --digests | grep inference`

Captured 2026-10-08 16:03 UTC by the operator in an interactive terminal (sudo
cannot authenticate from the agent session), saved to `~/sweep2_images.txt`
and pasted verbatim below. The file holds two listings: the first 10 lines show
v3/v3.1 and v4 tags with truncated digests, the last 5 show only v4 with full
digests. The two agree on every v4 digest prefix and image ID.

```
docker.io/hamidhrf/bert-inference                        v3                        472c4cd03f7d0       664411de1798f       6.07GB
docker.io/hamidhrf/bert-inference                        v4                        5c0df8cf3ffd7       098ae7afbffa1       6.07GB
docker.io/hamidhrf/gpt2-inference                        v3.1                      6ab8b92780f46       47324a7c6de9e       6.07GB
docker.io/hamidhrf/gpt2-inference                        v4                        76870613e8567       3f8d5e2680243       6.07GB
docker.io/hamidhrf/resnet152-inference                   v3                        e4608d5743025       59b6671e6f261       6.55GB
docker.io/hamidhrf/resnet152-inference                   v4                        33eb057c5020e       26cfb9af99ae7       5.96GB
docker.io/hamidhrf/whisper-inference                     v3                        98231fe597001       365cb49919f2a       8.25GB
docker.io/hamidhrf/whisper-inference                     v4                        2d3c7b1771619       4a49aab3dcb27       6.73GB
docker.io/hamidhrf/yolo-inference                        v3.1                      e3b5596816b6b       31918cff427fa       7.27GB
docker.io/hamidhrf/yolo-inference                        v4                        b991d9be31b8b       9772440b4ba14       6.72GB
docker.io/hamidhrf/bert-inference                        v4                        sha256:5c0df8cf3ffd7cfc605111f00a5f2f4e0714b6acdf53e9aeb547ee2370e88b40   098ae7afbffa1ed3491c85551372f4f03160c0e116b899b67b37cd567f6f0f67   6.07GB
docker.io/hamidhrf/gpt2-inference                        v4                        sha256:76870613e856701807368bc046f2fd1feca95d65c71a62a0fd8daa30c8f78c55   3f8d5e2680243c4ebb0331cd86da74ee91f8a27a7c470b573a1279f2bd225c2b   6.07GB
docker.io/hamidhrf/resnet152-inference                   v4                        sha256:33eb057c5020e00c4fcfe9415e3cb07765da3892f644117f278a944c54279c92   26cfb9af99ae71a48b799bb8de6389c17537f40054650e822c71d4707568cc3f   5.96GB
docker.io/hamidhrf/whisper-inference                     v4                        sha256:2d3c7b17716198050e8f74c36754d5f9efad2e90e0cadb85bf91c06e4da6c5d1   4a49aab3dcb270a97faf9fccb2e6fc44fcba7f11c9a8da92761a11cff3aa5b8c   6.73GB
docker.io/hamidhrf/yolo-inference                        v4                        sha256:b991d9be31b8bc0961638feebd65cae25fe4fba6d32886162a979e28e4fd531e   9772440b4ba148cdf9bed672984e945a041b427b3a1ca71ac9cf5e9b5c196d7d   6.72GB
```

Check: all five sweep images `hamidhrf/<wl>-inference:v4` are cached (bert,
gpt2, resnet152, whisper, yolo). PASS. No image was pulled or built.

(Workload images are tagged `<workload>-inference` per the Tier 2/Tier 3
manifests, e.g. `bert-inference`.)

## 5. Current node GPU labels and allocatable

MIG / sharing labels on node `devlab` (`kubectl get node devlab -o jsonpath=...`):

```
nvidia.com/mig.capable         = true
nvidia.com/mig.config          = all-disabled
nvidia.com/mig.config.state    = success
nvidia.com/gpu.product         = NVIDIA-H100-NVL-SHARED
nvidia.com/gpu.replicas        = 5
nvidia.com/gpu.sharing-strategy= time-slicing
nvidia.com/gpu.count           = 1
nvidia.com/gpu.memory          = 95830
```

GPU allocatable / capacity:

```
status.allocatable.nvidia.com/gpu = 5
status.capacity.nvidia.com/gpu    = 5
```

ClusterPolicy device-plugin binding:
`spec.devicePlugin.config = {"default":"any","name":"time-slicing-config"}`

Live `time-slicing-config` ConfigMap (`gpu-operator` ns), `data.any`:
```
version: v1
flags:
  migStrategy: none
sharing:
  timeSlicing:
    renameByDefault: false
    failRequestsGreaterThanOne: true
    resources:
      - name: nvidia.com/gpu
        replicas: 5
```

### IMPORTANT — current state matches NEITHER sweep precondition

The node is currently in **time-slicing at 5 replicas** (capacity 5). This is a
third configuration, distinct from both documented sweep states:

- Tier 2 MIG sweep requires `mig.config=all-1g.12gb`, `state=success`,
  capacity **7**. (`check_mig_state` in `run_tier2_batch.sh`.)
- Tier 3 time-slicing sweep requires `mig.config=all-disabled`, capacity **10**,
  `gpu.replicas=10`. (`check_timeslicing_state` in `run_tier3_batch.sh`.)

Both batch scripts will HALT at their precondition check if launched now.
Reconfiguration (section 6 or 7) is required before either sweep. Note the live
ConfigMap pins `replicas: 5`, not the `10` documented in `TIER3_NOTES.md` — to
reach the Tier 3 state the ConfigMap must be re-applied with `replicas: 10`
(Checklist B step 3).

### Time-slicing ConfigMap: current state vs. sweep 2

- The **live** `time-slicing-config` ConfigMap is `replicas: 5` — this is
  Kostya's current state, not the sweep configuration. Leave it as found until
  the sweep operator deliberately reconfigures.
- **Sweep 2 time-slicing uses `replicas: 10`**, identical to the August Tier 3
  sweep (applied via Checklist B step 3).
- The backup in `~/sweep2_backup/` (`time-slicing-config.*.yaml`) captures the
  **replicas-5** state. After sweep 2 completes, re-apply that backup to restore
  Kostya's replicas-5 configuration. Strip the server-managed fields first:
  the saved `resourceVersion` is stale once Tier 3 has changed the ConfigMap,
  and a plain `kubectl apply -f` of the backup is then rejected with a conflict.
  ```bash
  grep -v -E '^\s+(resourceVersion|uid|creationTimestamp):' \
    ~/sweep2_backup/time-slicing-config.20261008_151744.yaml | kubectl apply -f -
  kubectl patch clusterpolicy cluster-policy --type merge \
    -p '{"spec":{"devicePlugin":{"config":{"name":"time-slicing-config","default":"any"}}}}'
  # sweep 2 set mig.strategy=single (Checklist A step 2b); Kostya's state is none
  kubectl patch clusterpolicy cluster-policy --type merge -p '{"spec":{"mig":{"strategy":"none"}}}'
  kubectl get node devlab -o jsonpath='cap={.status.capacity.nvidia\.com/gpu} replicas={.metadata.labels.nvidia\.com/gpu\.replicas} strategy={.metadata.labels.nvidia\.com/gpu\.sharing-strategy} mig={.metadata.labels.nvidia\.com/mig\.config}{"\n"}'
  # expect cap=5 replicas=5 strategy=time-slicing mig=all-disabled
  ```

---

## 6. Checklist A — switch time-slicing to MIG (for the Tier 2 sweep)

Source: `docs/TIER2_NOTES.md` section 11 (Enable/Disable MIG) and
`docs/TIER3_NOTES.md` section 9 (Disable time-slicing). Exact August commands.
DO NOT RUN during preflight — recorded for the sweep operator.

1. Set the node handle and confirm the starting (time-slicing) state:
   ```bash
   NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
   kubectl get node $NODE -o jsonpath='{.status.capacity.nvidia\.com/gpu}'   # time-slicing value
   ```
2. Clear the time-slicing ConfigMap binding from the ClusterPolicy
   (ConfigMap clearing — removes `spec.devicePlugin.config`):
   ```bash
   kubectl patch clusterpolicy cluster-policy \
     --type json \
     -p '[{"op":"remove","path":"/spec/devicePlugin/config"}]'
   ```
2a. **DEVIATION from the August procedure (added 2026-10-08).** Take the
   `gpu-gatekeeper` DaemonSet off the node before any MIG label change.

   Why: the `gpu-gatekeeper` DaemonSet (namespace `gpu-gatekeeper`, image
   `localhost/gpu-gatekeeper:v0.3.0`, created 2026-09-16, after the August
   sweep) runs privileged with `runtimeClassName: nvidia` and
   `NVIDIA_VISIBLE_DEVICES=all`. It gets the GPU without requesting
   `nvidia.com/gpu`, and its node selector (`nvidia.com/gpu.present=true`) is
   not one of the `nvidia.com/gpu.deploy.*` labels the mig-manager pauses. On
   the first sweep 2 attempt (2026-10-08 16:12:08 UTC) the mig-manager enabled
   MIG mode but failed to create the 1g.12gb instances with
   `Error creating GPU instance for '1g.12gb': ERROR_IN_USE`. It set
   `mig.config.state=failed` and left `kubelet.service` stopped (the
   mig-manager stops the kubelet before applying and only restarts it on
   success). The gatekeeper is the most likely GPU holder; this was not proven
   with a root process scan.

   Taking it off the node: add a nodeSelector key that no node carries, wait for
   the pod to go, and confirm the GPU is idle:
   ```bash
   kubectl patch daemonset gpu-gatekeeper -n gpu-gatekeeper --type merge \
     -p '{"spec":{"template":{"spec":{"nodeSelector":{"sweep2-gatekeeper-paused":"true"}}}}}'
   kubectl wait --for=delete pod -l app=gpu-gatekeeper -n gpu-gatekeeper --timeout=120s
   kubectl get daemonset gpu-gatekeeper -n gpu-gatekeeper   # expect DESIRED 0, CURRENT 0
   nvidia-smi                                               # expect "No running processes found"
   ```
   Restore it after **both** sweeps (Kostya re-enables it himself) by removing
   only that key (the rest of the spec is unchanged; backup in
   `~/sweep2_backup/daemonset-gpu-gatekeeper.20261008_162214.yaml`):
   ```bash
   kubectl patch daemonset gpu-gatekeeper -n gpu-gatekeeper --type json \
     -p '[{"op":"remove","path":"/spec/template/spec/nodeSelector/sweep2-gatekeeper-paused"}]'
   kubectl rollout status daemonset/gpu-gatekeeper -n gpu-gatekeeper --timeout=180s
   ```
   The companion `gpu-scheduler` Deployment (namespace `gpu-scheduler`) is left
   running. It is disabled (`scheduler_enabled: false`), has no active jobs,
   only creates pods in `odm-gpu`, and has no RBAC on the `default` namespace
   that our inference pods use.

   If a MIG label change has already failed (state `failed`, kubelet stopped):
   start the kubelet again (`sudo systemctl start kubelet`), do this step, then
   re-trigger the mig-manager. It only reacts to a label *change*, so set
   `nvidia.com/mig.config=all-disabled`, wait for `success`, then continue with
   step 3.

2b. **DEVIATION from the August procedure (added 2026-10-08).** Set the
   ClusterPolicy MIG strategy to `single`:
   ```bash
   kubectl patch clusterpolicy cluster-policy --type merge -p '{"spec":{"mig":{"strategy":"single"}}}'
   kubectl get node $NODE -o jsonpath='{.metadata.labels.nvidia\.com/mig\.strategy}'   # expect: single
   ```
   Why: in August the strategy was already `single` from the Helm install
   (`TIER1_NOTES.md`, `TIER2_NOTES.md` section 2), so the August checklist had
   no strategy step. The pre-change backup
   (`~/sweep2_backup/clusterpolicy.20261008_151744.yaml`) shows
   `spec.mig.strategy: none`. With `none`, the device plugin ignores the MIG
   slices and advertises the whole GPU (`nvidia.com/gpu: 1`), so the
   precondition check in `run_tier2_batch.sh` fails. On the 2026-10-08 switch
   the node reached `mig.config.state=success` with 7 MIG devices in
   `nvidia-smi -L` but allocatable 1 until this step. Keep `single` for both
   sweeps; restore `none` at handback (section 5 restore commands).

3. Label the node onto the MIG profile:
   ```bash
   kubectl label node $NODE nvidia.com/mig.config=all-1g.12gb --overwrite
   ```
4. Poll until MIG reconfiguration reports success (typically ~30-40 s):
   ```bash
   kubectl get node $NODE -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config\.state}'
   # wait for: success
   kubectl get node $NODE -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config}'
   # should print: all-1g.12gb
   ```
5. Device-plugin / capacity verification — confirm the new geometry is live
   (capacity must read 7 before launching; `check_mig_state` enforces this):
   ```bash
   kubectl get node $NODE -o jsonpath='{.status.capacity.nvidia\.com/gpu}'    # expect 7
   kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset \
     -c nvidia-device-plugin --tail=50
   ```
   NOTE: no explicit device-plugin log-check command is recorded in the Tier 2 /
   Tier 3 August notes — those notes verify via the label + capacity poll above.
   The only `kubectl logs` for the device plugin in the repo
   (`scripts/gpu-setup/README.md:154`) targets `-n kube-system -l
   name=nvidia-device-plugin-ds`, which is stale for this operator-managed
   cluster (the plugin runs in `gpu-operator` as
   `app=nvidia-device-plugin-daemonset`). The corrected command is shown above.

## 7. Checklist B — switch MIG back to time-slicing (for the Tier 3 sweep)

Source: `docs/TIER2_NOTES.md` section 11 (Disable MIG) and `docs/TIER3_NOTES.md`
section 9 (Enable time-slicing). Exact August commands. DO NOT RUN during
preflight.

1. Set the node handle:
   ```bash
   NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
   ```
2. Disable MIG and wait for success:
   ```bash
   kubectl label node $NODE nvidia.com/mig.config=all-disabled --overwrite
   kubectl get node $NODE -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config\.state}'
   # wait for: success
   ```
3. Apply the time-slicing ConfigMap (ConfigMap clearing/replacement — this is
   the canonical `replicas: 10` form from `TIER3_NOTES.md`; the currently-live
   ConfigMap pins `replicas: 5`, so re-applying this is required to reach
   capacity 10):
   ```bash
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
   ```
4. Patch the ClusterPolicy to reference the ConfigMap:
   ```bash
   kubectl patch clusterpolicy cluster-policy \
     --type merge \
     -p '{"spec":{"devicePlugin":{"config":{"name":"time-slicing-config","default":"any"}}}}'
   ```
5. Wait for the device plugin to restart and capacity to update to 10, then
   verify (capacity 10 and `gpu.replicas=10` before launching; three-signal
   check is what `check_timeslicing_state` enforces):
   ```bash
   kubectl get node $NODE -o jsonpath='{.status.capacity.nvidia\.com/gpu}'           # expect 10
   kubectl get node $NODE -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.replicas}' # expect 10
   kubectl get node $NODE -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config}'   # expect all-disabled
   kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset \
     -c nvidia-device-plugin --tail=50
   ```
   Same note as Checklist A step 5: the log command is the corrected form; the
   August notes themselves verified via the label + capacity poll, not logs.

## 8. Backups taken

Saved under `~/sweep2_backup/` (timestamp 20261008_151744):
- `time-slicing-config.20261008_151744.yaml` — current time-slicing ConfigMap
- `clusterpolicy.20261008_151744.yaml` — current ClusterPolicy `cluster-policy`

Second set, before taking the gatekeeper off the node (timestamp 20261008_162214):
- `gpu-gatekeeper-all`, `daemonset-gpu-gatekeeper`, `ns-gpu-gatekeeper`: the
  `gpu-gatekeeper` namespace (DaemonSet, ControllerRevision, Service,
  ServiceAccount, ConfigMap, Pod, endpoints)
- `gpu-scheduler-all`, `ns-gpu-scheduler-odm-gpu`, `odm-gpu-rbac`: the
  `gpu-scheduler` namespace (Deployment, ReplicaSet, Service, ServiceAccounts,
  ConfigMap, Pod, endpoints) and its Role/RoleBinding in `odm-gpu`
- `runtimeclasses` (`nvidia`, `nvidia-cdi`, `nvidia-legacy`), `node-devlab`
- `clusterpolicy.20261008_162214.yaml`: taken *after* Checklist A step 2 (no
  `devicePlugin.config`). The pre-change ClusterPolicy is the 151744 file.

## 9. MIG switch record (2026-10-08)

All times UTC. Changes were run by the operator; checks by the agent.

| Time | Event |
|------|-------|
| 16:04:28 | Checklist A step 1: node `devlab`, capacity 5 (time-slicing, replicas 5) |
| 16:09:12 | Pre-check: `nvidia-smi` no processes, 4 MiB used; no pod requests `nvidia.com/gpu` |
| ~16:10 | Step 2: `devicePlugin.config` removed from ClusterPolicy; capacity 5 -> 1 (16:10:46) |
| ~16:11:34 | Step 3: label `all-1g.12gb`; mig-manager stopped kubelet 16:11:38 |
| 16:12:08 | **Failed**: MIG mode enabled, `Error creating GPU instance for '1g.12gb': ERROR_IN_USE`; state `failed`, kubelet left stopped |
| ~16:15 | Operator restarted kubelet; node Ready again |
| ~16:28 | Step 2a: gatekeeper DaemonSet given `sweep2-gatekeeper-paused=true` nodeSelector (Kostya approved); pod gone by 16:28:41, `nvidia-smi` no processes |
| ~16:29 | Re-trigger: label `all-disabled`; `success` 16:30:11, MIG mode Disabled, kubelet active |
| ~16:31:50 | Label `all-1g.12gb`; `success` 16:32:59, kubelet active |
| 16:33:50 | Step 5 check: 7 MIG devices but allocatable **1** (`mig.strategy: none`) |
| ~16:38 | Step 2b: ClusterPolicy `mig.strategy=single`; device plugin and GFD restarted 16:38:09 with `MIG_STRATEGY=single` |
| **16:38:55** | **allocatable 7**: switch complete |

Step 5 checks at 2026-10-08T16:39:07Z:

```
kubelet                      active
mig.config=all-1g.12gb state=success strategy=single
capacity=7 allocatable=7
gpu.deploy.* not true:       none
nvidia.com/gpu.product = NVIDIA-H100-NVL-MIG-1g.12gb
nvidia.com/gpu.count = 7
nvidia.com/gpu.memory = 11008
nvidia.com/gpu.replicas = 1
nvidia.com/gpu.sharing-strategy = none
GPU 0: NVIDIA H100 NVL (UUID: GPU-d59cb1c4-ad97-d91a-6731-e7b5f38951a6)
  MIG 1g.12gb     Device  0: (UUID: MIG-101340d5-6e71-5cc3-8b15-386f99ddda2e)
  MIG 1g.12gb     Device  1: (UUID: MIG-868ee273-5d41-581d-ae51-96419be44cd7)
  MIG 1g.12gb     Device  2: (UUID: MIG-2b51ae36-9d49-5176-aa85-8dea4fa5a110)
  MIG 1g.12gb     Device  3: (UUID: MIG-a83ec6c0-c1ef-5553-8e6a-ee3879217e0e)
  MIG 1g.12gb     Device  4: (UUID: MIG-9e08e7df-5f86-54b3-b9c4-3cb8a1d3c47e)
  MIG 1g.12gb     Device  5: (UUID: MIG-8ba8ba69-b483-543f-9559-fcf5556e4b67)
  MIG 1g.12gb     Device  6: (UUID: MIG-9a9c14dd-644c-57e6-9459-603c8e320d36)
gpu-operator pods            all Running/Completed
```

Device-plugin log (`nvidia-device-plugin-daemonset-jgd5z`): `"migStrategy":
"single"`, `Registered device plugin for 'nvidia.com/gpu' with Kubelet` at
16:38:47. No errors; the only warnings are missing optional files (vulkan
layers, fabricmanager, MPS, imex, sandboxutils), the same as on the
`migStrategy: none` start at 16:33:38.

These satisfy `check_mig_state` in `run_tier2_batch.sh` (mig.config,
state, capacity 7).
