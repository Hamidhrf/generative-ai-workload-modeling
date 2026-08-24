#!/usr/bin/env python3
# E1 matched-size control (NOT committed). Adapted from unify_tier2.py
# (which itself documents byte-identical split mechanism to unify_tier3.py:
# rng.shuffle seed=42, n_val=round(0.11*N)). Path retarget only:
# data/processed/tier3_matched -> .../unified. N=140 here (28 traces x 5
# workloads), same total N as Tier 2, so this reproduces Tier 2's exact
# split ratio (round(0.11*140)=15 val / 125 train) via the identical
# mechanism -- not just the same ratio in the abstract.
import json
import numpy as np
from pathlib import Path

WORKLOADS = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']  # match A16 id ordering
IN_DIR = Path('data/processed/tier3_matched')
OUT_DIR = Path('data/processed/tier3_matched/unified')
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

# Train/val split: match Tier 2 / Tier 3 policy exactly (rng.shuffle
# seed=42, ~89/11 split via round()).
rng = np.random.default_rng(42)
idx = np.arange(len(traces))
rng.shuffle(idx)
n_val = int(round(len(idx) * 0.11))
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
print(f'split: train={len(train_idx)} val={len(val_idx)} (n_val = round(0.11 * {len(idx)}) = {n_val})')

with open(OUT_DIR / 'combined_normalization.json', 'w') as f:
    json.dump(combined_norm, f, indent=2)
print(f'wrote {OUT_DIR / "combined_normalization.json"}')
