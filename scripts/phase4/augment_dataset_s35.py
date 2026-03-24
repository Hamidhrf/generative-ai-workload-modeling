"""
S35 Data Augmentation Script
=============================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

ABLATION STUDY: S35 = S34 + Data Augmentation (4x)
===================================================

AUGMENTATION STRATEGY:
----------------------
Original dataset: 176 pod traces
1. Sliding Window (3 overlapping segments): +352 traces
2. Noise Injection (Gaussian, sigma=0.02): +176 traces
Total: 176 + 352 + 176 = 704 traces (4x augmentation)

SLIDING WINDOW:
- Take 3 overlapping segments from each trace
- Segment 1: timesteps 0-480 (67% of trace)
- Segment 2: timesteps 120-600 (67% of trace, shifted)
- Segment 3: timesteps 235-715 (67% of trace, shifted more)
- Each segment resampled to 715 timesteps

NOISE INJECTION:
- Add Gaussian noise: N(0, 0.02 * std(trace))
- Per-metric noise (respects metric scale)
- Clipped to [0, 1] range (normalized space)

OUTPUT:
-------
data/processed/phase4/unified/combined_dataset_s35_augmented.npz
data/processed/phase4/unified/combined_normalization_s35_augmented.json

USAGE:
------
python augment_dataset_s35.py
"""

import json
import shutil
from pathlib import Path
import numpy as np

# Paths
DATA_ORIG = Path("data/processed/phase4/unified/combined_dataset.npz")
NORM_ORIG = Path("data/processed/phase4/unified/combined_normalization.json")
DATA_AUG = Path("data/processed/phase4/unified/combined_dataset_s35_augmented.npz")
NORM_AUG = Path("data/processed/phase4/unified/combined_normalization_s35_augmented.json")

# Augmentation config
NOISE_SIGMA = 0.02  # 2% of trace std
WINDOW_SEGMENTS = [
    (0, 480),      # First 67% of trace
    (120, 600),    # Middle 67%
    (235, 715),    # Last 67%
]
TARGET_LEN = 715


def resample_segment(segment, target_len):
    """Resample segment to target length using linear interpolation."""
    src_len, n_metrics = segment.shape
    if src_len == target_len:
        return segment
    
    src_t = np.linspace(0, 1, src_len)
    dst_t = np.linspace(0, 1, target_len)
    
    resampled = np.zeros((target_len, n_metrics), dtype=np.float32)
    for m in range(n_metrics):
        resampled[:, m] = np.interp(dst_t, src_t, segment[:, m])
    
    return resampled


def sliding_window_augmentation(traces):
    """Generate augmented traces using sliding windows."""
    N, T, M = traces.shape
    augmented = []
    
    print(f"  Applying sliding window augmentation...")
    print(f"  Original trace length: {T} timesteps")
    
    for trace_idx in range(N):
        trace = traces[trace_idx]
        
        for start, end in WINDOW_SEGMENTS:
            segment = trace[start:end, :]
            resampled = resample_segment(segment, TARGET_LEN)
            augmented.append(resampled)
    
    augmented_array = np.array(augmented, dtype=np.float32)
    print(f"  Generated {len(augmented)} windowed traces from {N} originals")
    return augmented_array


def noise_injection_augmentation(traces, noise_sigma):
    """Generate augmented traces by adding Gaussian noise."""
    N, T, M = traces.shape
    augmented = []
    
    print(f"  Applying noise injection (sigma={noise_sigma})...")
    
    for trace_idx in range(N):
        trace = traces[trace_idx]
        
        # Compute per-metric std
        trace_std = trace.std(axis=0, keepdims=True)
        
        # Generate noise scaled by trace std
        noise = np.random.normal(0, noise_sigma, size=(T, M)).astype(np.float32)
        noise = noise * trace_std
        
        # Add noise and clip to valid range
        noisy_trace = trace + noise
        noisy_trace = np.clip(noisy_trace, 0.0, 1.0)
        
        augmented.append(noisy_trace)
    
    augmented_array = np.array(augmented, dtype=np.float32)
    print(f"  Generated {len(augmented)} noisy traces from {N} originals")
    return augmented_array


def augment_metadata(orig_metadata, n_original, n_window, n_noise):
    """Create metadata for augmented traces."""
    augmented_meta = []
    
    # Copy original metadata
    for i, meta in enumerate(orig_metadata):
        new_meta = meta.copy()
        new_meta['augmentation'] = 'original'
        new_meta['aug_source_idx'] = i
        augmented_meta.append(new_meta)
    
    # Add windowed trace metadata
    for i, meta in enumerate(orig_metadata):
        for seg_idx, (start, end) in enumerate(WINDOW_SEGMENTS):
            new_meta = meta.copy()
            new_meta['augmentation'] = f'window_seg{seg_idx}'
            new_meta['aug_source_idx'] = i
            new_meta['window_range'] = f'{start}-{end}'
            augmented_meta.append(new_meta)
    
    # Add noisy trace metadata
    for i, meta in enumerate(orig_metadata):
        new_meta = meta.copy()
        new_meta['augmentation'] = 'noise_injection'
        new_meta['aug_source_idx'] = i
        new_meta['noise_sigma'] = NOISE_SIGMA
        augmented_meta.append(new_meta)
    
    return augmented_meta


def main():
    print("="*80)
    print("S35 DATA AUGMENTATION")
    print("="*80)
    print(f"Original dataset: {DATA_ORIG}")
    print(f"Output dataset: {DATA_AUG}")
    print()
    
    # Load original dataset
    print("Loading original dataset...")
    data = np.load(DATA_ORIG, allow_pickle=True)
    
    traces_orig = data['traces']
    rc_orig = data['replica_counts']
    wl_orig = data['workload_ids']
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    metadata_orig = data['metadata']
    
    N, T, M = traces_orig.shape
    print(f"Original: {N} traces, {T} timesteps, {M} metrics")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print()
    
    # Augmentation 1: Sliding Windows
    print("[1/3] Sliding Window Augmentation")
    traces_window = sliding_window_augmentation(traces_orig)
    rc_window = np.repeat(rc_orig, len(WINDOW_SEGMENTS))
    wl_window = np.repeat(wl_orig, len(WINDOW_SEGMENTS))
    print()
    
    # Augmentation 2: Noise Injection
    print("[2/3] Noise Injection Augmentation")
    traces_noise = noise_injection_augmentation(traces_orig, NOISE_SIGMA)
    rc_noise = rc_orig.copy()
    wl_noise = wl_orig.copy()
    print()
    
    # Combine all augmented data
    print("[3/3] Combining augmented datasets")
    traces_combined = np.concatenate([traces_orig, traces_window, traces_noise], axis=0)
    rc_combined = np.concatenate([rc_orig, rc_window, rc_noise], axis=0)
    wl_combined = np.concatenate([wl_orig, wl_window, wl_noise], axis=0)
    
    print(f"Combined: {len(traces_combined)} traces")
    print(f"  - Original: {len(traces_orig)}")
    print(f"  - Windowed: {len(traces_window)}")
    print(f"  - Noisy: {len(traces_noise)}")
    print()
    
    # Update train/val indices
    # Original indices stay the same
    # Windowed indices: original_idx + N, + 2*N, + 3*N for each window segment
    # Noisy indices: original_idx + N + len(window)
    
    offset_window = N
    offset_noise = N + len(traces_window)
    
    train_idx_augmented = np.concatenate([
        train_idx,  # Original
        train_idx + offset_window,  # Window seg 0
        train_idx + offset_window + N,  # Window seg 1
        train_idx + offset_window + 2*N,  # Window seg 2
        train_idx + offset_noise,  # Noise
    ])
    
    val_idx_augmented = np.concatenate([
        val_idx,  # Original
        val_idx + offset_window,  # Window seg 0
        val_idx + offset_window + N,  # Window seg 1
        val_idx + offset_window + 2*N,  # Window seg 2
        val_idx + offset_noise,  # Noise
    ])
    
    print(f"Augmented indices:")
    print(f"  Train: {len(train_idx_augmented)} (was {len(train_idx)})")
    print(f"  Val: {len(val_idx_augmented)} (was {len(val_idx)})")
    print()
    
    # Create augmented metadata
    metadata_augmented = augment_metadata(
        metadata_orig.tolist(),
        len(traces_orig),
        len(traces_window),
        len(traces_noise)
    )
    
    # Save augmented dataset
    print(f"Saving augmented dataset to {DATA_AUG}...")
    DATA_AUG.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        DATA_AUG,
        traces=traces_combined,
        replica_counts=rc_combined,
        workload_ids=wl_combined,
        train_idx=train_idx_augmented,
        val_idx=val_idx_augmented,
        metadata=np.array(metadata_augmented, dtype=object)
    )
    
    print(f"Saved: {DATA_AUG}")
    print(f"Size: {DATA_AUG.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    
    # Copy normalization file (unchanged)
    print(f"Copying normalization file to {NORM_AUG}...")
    shutil.copy(NORM_ORIG, NORM_AUG)
    print(f"Saved: {NORM_AUG}")
    print()
    
    # Summary
    print("="*80)
    print("AUGMENTATION COMPLETE")
    print("="*80)
    print(f"Original dataset: {N} traces")
    print(f"Augmented dataset: {len(traces_combined)} traces (4x)")
    print()
    print("Breakdown:")
    print(f"  Original:         {len(traces_orig):3d} traces")
    print(f"  Sliding windows:  {len(traces_window):3d} traces ({len(WINDOW_SEGMENTS)}x windowing)")
    print(f"  Noise injection:  {len(traces_noise):3d} traces (sigma={NOISE_SIGMA})")
    print()
    print("Next step: Train S35 using augmented dataset")
    print("  python scripts/phase4/timegan_s35.py")
    print("="*80)


if __name__ == "__main__":
    main()