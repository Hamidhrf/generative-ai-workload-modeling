# Sweep 2 Environment Record

Captured: 2026-10-08 on devLab. Read-only preflight for the second
data-collection sweeps. No cluster state was changed to produce this record.

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

NOT CAPTURED during this preflight: `sudo` requires interactive
authentication in this session, so `crictl` could not be run non-interactively.
Run manually and paste the output here before the sweep:

```bash
sudo crictl images --digests | grep inference
```

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
  Kostya's replicas-5 configuration:
  `kubectl apply -f ~/sweep2_backup/time-slicing-config.20261008_151744.yaml`

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
