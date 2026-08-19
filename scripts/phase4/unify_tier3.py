#!/usr/bin/env python3
import numpy as np
from pathlib import Path

WORKLOADS = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']  # match A16 id ordering
IN_DIR = Path('data/processed/tier3')
OUT_DIR = Path('data/processed/tier3/unified')
OUT_DIR.mkdir(parents=True, exist_ok=True)

traces_all, replicas_all, workload_ids_all, metadata_all = [], [], [], []
metric_names = None

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

traces = np.concatenate(traces_all, axis=0)
replicas = np.concatenate(replicas_all, axis=0)
workload_ids = np.concatenate(workload_ids_all, axis=0)
metadata = np.concatenate(metadata_all, axis=0)

# Train/val split: match A16 policy. Original used random_state=42,
# ~89/11 split. Reproduce that ratio; document seed.
rng = np.random.default_rng(42)
idx = np.arange(len(traces))
rng.shuffle(idx)
n_val = int(round(len(idx) * 0.11))  # matches 245/30 A16 split ratio
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
