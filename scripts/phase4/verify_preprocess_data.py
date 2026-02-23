#!/usr/bin/env python3
"""
Verify Preprocessed Data (10 Metrics)

Quick validation script to check the preprocessed .npz files.

Usage:
    python scripts/phase4/verify_preprocessed.py
"""

import numpy as np
import json
from pathlib import Path

DATA_DIR = Path('data/processed')
WORKLOADS = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']
EXPECTED_METRICS = 10
EXPECTED_TIMESTEPS = 715

print("="*70)
print("VERIFYING PREPROCESSED DATA")
print("="*70)

all_good = True

for workload in WORKLOADS:
    print(f"\n{workload.upper()}:")
    
    # Load .npz file
    npz_file = DATA_DIR / f'{workload}_traces.npz'
    if not npz_file.exists():
        print(f"  ERROR: File not found: {npz_file}")
        all_good = False
        continue
    
    data = np.load(npz_file, allow_pickle=True)
    
    # Check shape
    traces = data['traces']
    expected_shape = (55, EXPECTED_TIMESTEPS, EXPECTED_METRICS)
    if traces.shape != expected_shape:
        print(f"  ERROR: Shape {traces.shape}, expected {expected_shape}")
        all_good = False
    else:
        print(f"  Shape: {traces.shape} OK")
    
    # Check normalization
    if traces.min() < 0 or traces.max() > 1:
        print(f"  ERROR: Values outside [0,1]: min={traces.min():.3f}, max={traces.max():.3f}")
        all_good = False
    else:
        print(f"  Normalization: [0, 1] OK (min={traces.min():.3f}, max={traces.max():.3f})")
    
    # Check for NaN/Inf
    if np.isnan(traces).any():
        print(f"  ERROR: Contains NaN values")
        all_good = False
    if np.isinf(traces).any():
        print(f"  ERROR: Contains Inf values")
        all_good = False
    
    # Check splits
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    total_samples = len(traces)
    
    if len(train_idx) + len(val_idx) != total_samples:
        print(f"  ERROR: Split mismatch: train={len(train_idx)}, val={len(val_idx)}, total={total_samples}")
        all_good = False
    else:
        print(f"  Splits: train={len(train_idx)}, val={len(val_idx)} OK")
    
    # Check replica counts
    replica_counts = data['replica_counts']
    unique_r = np.unique(replica_counts)
    if not np.array_equal(unique_r, np.arange(1, 11)):
        print(f"  ERROR: Missing replica counts: {unique_r}")
        all_good = False
    else:
        print(f"  Replica counts: r=1-10 OK")
    
    # Check metric names
    metric_names = data['metric_names']
    if len(metric_names) != EXPECTED_METRICS:
        print(f"  ERROR: Expected {EXPECTED_METRICS} metrics, got {len(metric_names)}")
        all_good = False
    else:
        print(f"  Metrics: {EXPECTED_METRICS} OK")
    
    # Verify no zero-variance metrics
    variances = traces.var(axis=(0, 1))  # Variance across pods and time
    zero_var = (variances < 1e-10).sum()
    if zero_var > 0:
        print(f"  WARNING: {zero_var} metrics with near-zero variance")
        for i, (name, var) in enumerate(zip(metric_names, variances)):
            if var < 1e-10:
                print(f"    - {name}: var={var:.10f}")
    
    # Load normalization params
    json_file = DATA_DIR / f'{workload}_normalization.json'
    if not json_file.exists():
        print(f"  ERROR: Normalization file not found: {json_file}")
        all_good = False
    else:
        with open(json_file) as f:
            norm_params = json.load(f)
        if len(norm_params['params']) != EXPECTED_METRICS:
            print(f"  ERROR: Norm params has {len(norm_params['params'])} metrics, expected {EXPECTED_METRICS}")
            all_good = False

print("\n" + "="*70)
if all_good:
    print("VERIFICATION PASSED!")
    print("="*70)
    print("\nAll checks passed. Data is ready for training.")
else:
    print("VERIFICATION FAILED!")
    print("="*70)
    print("\nPlease fix errors before proceeding.")

print("\nDataset Summary:")
print(f"  Total workloads: {len(WORKLOADS)}")
print(f"  Total pods: {len(WORKLOADS) * 55}")
print(f"  Metrics per trace: {EXPECTED_METRICS}")
print(f"  Timesteps per trace: {EXPECTED_TIMESTEPS}")
print(f"  Shape per workload: (55, {EXPECTED_TIMESTEPS}, {EXPECTED_METRICS})")