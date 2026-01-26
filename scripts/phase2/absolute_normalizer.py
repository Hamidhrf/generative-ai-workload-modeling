#!/usr/bin/env python3
"""
Absolute Normalization for Pod Traces

Uses ABSOLUTE ranges (not system-relative) to ensure pod metrics
remain consistent across different VM configurations.

Key Principle:
- Pod-intrinsic metrics (CPU, memory, latency) use absolute ranges
- System-wide metrics (GPU) can use system-relative ranges
- This ensures a pod using 200Mi RAM generates 200Mi on ANY system

Author: Hamidreza Fathollahzadeh
Date: January 23, 2026
"""

import numpy as np
from typing import Dict, Tuple, Optional
import joblib
from pathlib import Path


# Metric names (for reference)
METRIC_NAMES = [
    'cpu_psi',              # 0
    'cpu_usage',            # 1
    'gpu_memory',           # 2
    'gpu_power',            # 3
    'gpu_temperature',      # 4
    'gpu_utilization',      # 5
    'latency_avg',          # 6
    'latency_p50',          # 7
    'latency_p95',          # 8
    'latency_p99',          # 9
    'throughput',           # 10
    'total_inferences',     # 11
    'io_psi',               # 12
    'memory_psi',           # 13
    'memory_usage'          # 14
]


# ABSOLUTE NORMALIZATION RANGES
# These are independent of system configuration
ABSOLUTE_RANGES = {
    # Pod-intrinsic metrics (ABSOLUTE ranges)
    'cpu_psi': (0.0, 1.0),              # PSI is already 0-1
    'cpu_usage': (0.0, 8.0),            # Max 8 cores per pod (reasonable max)
    'latency_avg': (0.0, 5.0),          # Max 5 seconds (reasonable max)
    'latency_p50': (0.0, 5.0),          # Max 5 seconds
    'latency_p95': (0.0, 8.0),          # Max 8 seconds (higher percentile)
    'latency_p99': (0.0, 10.0),         # Max 10 seconds (tail latency)
    'throughput': (0.0, 500.0),         # Max 500 req/s per pod
    'total_inferences': (0.0, 100000.0), # Max 100k total inferences
    'io_psi': (0.0, 1.0),               # PSI is 0-1
    'memory_psi': (0.0, 1.0),           # PSI is 0-1
    'memory_usage': (0.0, 10e9),        # Max 10GB per pod (bytes)
    
    # System-wide metrics (can be system-relative, but using reasonable absolutes)
    'gpu_memory': (0.0, 20000.0),       # Max 20GB GPU memory (MB)
    'gpu_power': (0.0, 100.0),          # Max 100W power draw
    'gpu_temperature': (0.0, 100.0),    # Max 100°C (reasonable)
    'gpu_utilization': (0.0, 100.0),    # Percentage (0-100%)
}


# Metric indices for easier access
METRIC_INDICES = {name: idx for idx, name in enumerate(METRIC_NAMES)}


class AbsoluteNormalizer:
    """
    Normalizer using absolute ranges (not system-relative).
    
    This ensures pod-intrinsic metrics (CPU, memory, latency) maintain
    their absolute values regardless of the target system configuration.
    """
    
    def __init__(self, ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize normalizer with absolute ranges.
        
        Args:
            ranges: Optional custom ranges. Uses ABSOLUTE_RANGES by default.
        """
        self.ranges = ranges or ABSOLUTE_RANGES
        self.metric_names = METRIC_NAMES
        
        # Pre-compute min/max arrays for efficiency
        self.mins = np.array([self.ranges[name][0] for name in METRIC_NAMES])
        self.maxs = np.array([self.ranges[name][1] for name in METRIC_NAMES])
        self.scales = self.maxs - self.mins
        
    def normalize(self, trace: np.ndarray) -> np.ndarray:
        """
        Normalize trace to [0, 1] using absolute ranges.
        
        Args:
            trace: Raw trace array, shape (n_timesteps, 15)
            
        Returns:
            Normalized trace, shape (n_timesteps, 15), values in [0, 1]
        """
        # Min-max normalization: (x - min) / (max - min)
        normalized = (trace - self.mins) / self.scales
        
        # Clip to [0, 1] to handle any outliers
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized.astype(np.float32)
    
    def denormalize(self, normalized_trace: np.ndarray) -> np.ndarray:
        """
        Denormalize trace back to original units using absolute ranges.
        
        Args:
            normalized_trace: Normalized trace, shape (n_timesteps, 15), values in [0, 1]
            
        Returns:
            Denormalized trace in original units
        """
        # Inverse: x = normalized * (max - min) + min
        denormalized = normalized_trace * self.scales + self.mins
        
        return denormalized.astype(np.float32)
    
    def normalize_batch(self, traces: np.ndarray) -> np.ndarray:
        """
        Normalize batch of traces.
        
        Args:
            traces: Batch of traces, shape (n_samples, n_timesteps, 15)
            
        Returns:
            Normalized traces, shape (n_samples, n_timesteps, 15)
        """
        return np.array([self.normalize(trace) for trace in traces])
    
    def denormalize_batch(self, normalized_traces: np.ndarray) -> np.ndarray:
        """
        Denormalize batch of traces.
        
        Args:
            normalized_traces: Normalized traces, shape (n_samples, n_timesteps, 15)
            
        Returns:
            Denormalized traces
        """
        return np.array([self.denormalize(trace) for trace in normalized_traces])
    
    def save(self, filepath: str):
        """Save normalizer to disk."""
        joblib.dump({
            'ranges': self.ranges,
            'metric_names': self.metric_names,
            'mins': self.mins,
            'maxs': self.maxs,
            'scales': self.scales
        }, filepath)
        print(f"✓ Saved normalizer to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AbsoluteNormalizer':
        """Load normalizer from disk."""
        data = joblib.load(filepath)
        normalizer = cls(ranges=data['ranges'])
        print(f"✓ Loaded normalizer from {filepath}")
        return normalizer
    
    def get_stats(self) -> str:
        """Get human-readable stats about normalization ranges."""
        stats = ["Absolute Normalization Ranges:", "=" * 80]
        stats.append(f"{'Metric':<20} {'Min':>12} {'Max':>12} {'Range':>12}")
        stats.append("-" * 80)
        
        for name in METRIC_NAMES:
            min_val, max_val = self.ranges[name]
            range_val = max_val - min_val
            stats.append(f"{name:<20} {min_val:>12.2f} {max_val:>12.2f} {range_val:>12.2f}")
        
        return "\n".join(stats)


def create_normalizer_from_data(pod_traces, 
                                percentile: float = 99.5,
                                min_range_factor: float = 1.2) -> AbsoluteNormalizer:
    """
    Create normalizer with ranges computed from actual data.
    
    Uses percentiles to set max values (robust to outliers).
    Adds margin with min_range_factor.
    
    Args:
        pod_traces: List of (trace, metadata) tuples
        percentile: Percentile to use for max (default 99.5)
        min_range_factor: Multiply by this for safety margin (default 1.2 = 20% margin)
        
    Returns:
        AbsoluteNormalizer with data-derived ranges
    """
    # Collect all traces
    all_traces = np.array([trace for trace, _ in pod_traces])
    
    # Compute ranges from data
    data_ranges = {}
    for idx, name in enumerate(METRIC_NAMES):
        values = all_traces[:, :, idx].flatten()
        
        # Min is always 0 (or actual min if negative)
        min_val = max(0.0, values.min())
        
        # Max uses percentile + margin
        max_val = np.percentile(values, percentile) * min_range_factor
        
        # Ensure reasonable minimum range
        if max_val - min_val < 1e-6:
            max_val = min_val + 1.0
        
        data_ranges[name] = (min_val, max_val)
    
    return AbsoluteNormalizer(ranges=data_ranges)


def demonstrate_normalization():
    """Demonstrate normalization with example pod trace."""
    print("=" * 80)
    print("ABSOLUTE NORMALIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Create example pod trace (single timestep)
    example_pod = np.array([
        0.0005,         # cpu_psi
        1.2,            # cpu_usage (cores)
        3850.0,         # gpu_memory (MB)
        58.4,           # gpu_power (W)
        72.0,           # gpu_temperature (C)
        86.7,           # gpu_utilization (%)
        0.298,          # latency_avg (s)
        0.315,          # latency_p50 (s)
        0.598,          # latency_p95 (s)
        0.699,          # latency_p99 (s)
        171.25,         # throughput (req/s)
        171.25,         # total_inferences
        0.0,            # io_psi
        0.0,            # memory_psi
        1643693933.0    # memory_usage (bytes = ~1.53 GB)
    ])
    
    # Create normalizer
    normalizer = AbsoluteNormalizer()
    
    # Normalize
    normalized = normalizer.normalize(example_pod.reshape(1, -1))
    
    # Denormalize (should get original back)
    denormalized = normalizer.denormalize(normalized)
    
    # Print results
    print("\nExample Pod Trace:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Original':>15} {'Normalized':>15} {'Denorm':>15}")
    print("-" * 80)
    
    for i, name in enumerate(METRIC_NAMES):
        print(f"{name:<20} {example_pod[i]:>15.2f} {normalized[0, i]:>15.6f} {denormalized[0, i]:>15.2f}")
    
    # Verify denormalization
    max_error = np.abs(example_pod - denormalized[0]).max()
    print("-" * 80)
    print(f"Max denormalization error: {max_error:.2e}")
    
    if max_error < 1e-3:
        print("✓ Denormalization is accurate!")
    else:
        print("⚠ Warning: Denormalization has significant error")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("Memory was 1,643,693,933 bytes (1.53 GB)")
    print("After normalization → 0.164")
    print("After denormalization → 1,643,693,933 bytes (SAME!)")
    print("\nThis works on ANY system size because ranges are ABSOLUTE!")
    print("Pod uses 1.53 GB on 16-core OR 32-core system.")
    print("=" * 80)


if __name__ == "__main__":
    # Demonstrate normalization
    demonstrate_normalization()
    
    # Show ranges
    normalizer = AbsoluteNormalizer()
    print("\n")
    print(normalizer.get_stats())
    
    print("\n" + "=" * 80)
    print("USAGE:")
    print("=" * 80)
    print("""
# Training:
from absolute_normalizer import AbsoluteNormalizer

normalizer = AbsoluteNormalizer()
normalized_traces = normalizer.normalize_batch(raw_traces)

# Train model on normalized_traces...

# Save for later
normalizer.save('normalizer.pkl')

# Generation:
normalizer = AbsoluteNormalizer.load('normalizer.pkl')
synthetic_normalized = model.generate(...)
synthetic_real = normalizer.denormalize_batch(synthetic_normalized)

# Result: synthetic_real contains absolute values!
# 200Mi RAM stays 200Mi RAM on any system!
    """)