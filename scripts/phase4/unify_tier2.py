#!/usr/bin/env python3
# Adapted from unify_tier3.py. Path retarget only: data/processed/tier3 ->
# data/processed/tier2, output -> data/processed/tier2/unified/. Split
# mechanism (rng.shuffle seed=42, ~11% val fraction) unchanged; only the
# input count changes (140 total vs 275), which the same round() formula
# naturally resolves to 125 train / 15 val.
#
# combined_normalization.json is NOT produced by unify_tier3.py itself --
# Tier 3's version of this file (data/processed/tier3/unified/
# combined_normalization.json) is a workload-keyed copy of each of the 5
# already-computed data/processed/tier3/{workload}_normalization.json
# files (method=minmax, params={min,max} per metric, fit at preprocess
# time on all pods of that workload -- not a fresh fit on the unified
# train partition). Reproduced identically here from the 5 existing
# data/processed/tier2/{workload}_normalization.json files.
import json
import numpy as np
from pathlib import Path

WORKLOADS = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']  # match A16 id ordering
IN_DIR = Path('data/processed/tier2')
OUT_DIR = Path('data/processed/tier2/unified')
OUT_DIR.mkdir(parents=True, exist_ok=True)

traces_all, replicas_all, workload_ids_all, metadata_all = [], [], [], []
metric_names = None
combined_norm = {}

for wid, wl in enumerate(WORKLOADS):
    d = np.load(IN_DIR / f'{wl}_traces.npz', allow_pickle=True)
    traces_all.append(d['traces'])
    replicas_all.append(d['replica_counts'])
    workload_ids_all.append(np.full(len(d['traces']), wid, dtype=np.int32))
    metadata_all.append(d['metadata'])
    if metric_names is None:
        metric_names = d['metric_names']
    else:
        assert list(d['metric_names']) == list(metric_names), \
            f'metric_names mismatch: {wl}'

    with open(IN_DIR / f'{wl}_normalization.json') as f:
        combined_norm[wl] = json.load(f)

traces = np.concatenate(traces_all, axis=0)
replicas = np.concatenate(replicas_all, axis=0)
workload_ids = np.concatenate(workload_ids_all, axis=0)
metadata = np.concatenate(metadata_all, axis=0)

# Train/val split: match Tier 3 policy exactly (rng.shuffle seed=42,
# ~89/11 split via round()). Unchanged mechanism; count follows from len(idx).
rng = np.random.default_rng(42)
idx = np.arange(len(traces))
rng.shuffle(idx)
n_val = int(round(len(idx) * 0.11))  # matches 245/30 A16 / Tier3 split ratio
val_idx = np.sort(idx[:n_val])
train_idx = np.sort(idx[n_val:])

np.savez(
    OUT_DIR / 'combined_dataset.npz',
    traces=traces,
    replica_counts=replicas,
    workload_ids=workload_ids,
    workload_names=np.array(WORKLOADS),
    metric_names=metric_names,
    metadata=metadata,
    train_idx=train_idx,
    val_idx=val_idx,
)
print(f'wrote {OUT_DIR / "combined_dataset.npz"}: {traces.shape}')

with open(OUT_DIR / 'combined_normalization.json', 'w') as f:
    json.dump(combined_norm, f, indent=2)
print(f'wrote {OUT_DIR / "combined_normalization.json"}')
