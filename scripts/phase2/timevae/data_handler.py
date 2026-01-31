"""
Data Handler for TimeVAE Training
Handles absolute normalization, denormalization, and VM config

CRITICAL: Normalization ranges MUST match LSTM baseline for fair comparison!

"""

import numpy as np
import yaml
from pathlib import Path


# Metric names in order (matches LSTM baseline)
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


class DataNormalizer:
    """
    Handles absolute normalization/denormalization for pod traces
    
    CRITICAL: Data in pod_traces.npy should already be normalized from Phase 1
    This class provides denormalization for generation and validation
    """
    
    def __init__(self):
        """
        Define absolute normalization ranges from Phase 1
        CRITICAL: These MUST match LSTM baseline for fair comparison!
        Copied from lstm_baseline.py ABSOLUTE_RANGES
        """
        self.ranges = {
            # Index: (metric_name, min, max)
            0: ('cpu_psi', 0.0, 1.0),
            1: ('cpu_usage', 0.0, 8.0),
            2: ('gpu_memory', 0.0, 20000.0),        # MB (20GB GPU - matches LSTM)
            3: ('gpu_power', 0.0, 100.0),
            4: ('gpu_temperature', 0.0, 100.0),
            5: ('gpu_utilization', 0.0, 100.0),
            6: ('latency_avg', 0.0, 5.0),
            7: ('latency_p50', 0.0, 5.0),
            8: ('latency_p95', 0.0, 8.0),
            9: ('latency_p99', 0.0, 10.0),
            10: ('throughput', 0.0, 500.0),
            11: ('total_inferences', 0.0, 500.0),
            12: ('io_psi', 0.0, 1.0),
            13: ('memory_psi', 0.0, 1.0),
            14: ('memory_usage', 0.0, 10e9)         # bytes (10GB - matches LSTM)
        }
        
        self.metric_names = [self.ranges[i][0] for i in range(15)]
    
    def denormalize(self, normalized_data):
        """
        Denormalize data back to original ranges
        
        Args:
            normalized_data: (N, 715, 15) or (715, 15) - normalized values
        
        Returns:
            denormalized_data: same shape, original scale
        """
        denorm = normalized_data.copy()
        
        # Handle both batched and single sample
        if denorm.ndim == 3:
            for i in range(15):
                _, min_val, max_val = self.ranges[i]
                denorm[:, :, i] = denorm[:, :, i] * (max_val - min_val) + min_val
        else:  # 2D
            for i in range(15):
                _, min_val, max_val = self.ranges[i]
                denorm[:, i] = denorm[:, i] * (max_val - min_val) + min_val
        
        return denorm
    
    def normalize(self, data):
        """
        Normalize data to [0, 1] range
        
        Args:
            data: (N, 715, 15) or (715, 15) - original scale
        
        Returns:
            normalized_data: same shape, [0, 1] range (clipped)
        """
        norm = data.copy()
        
        if norm.ndim == 3:
            for i in range(15):
                _, min_val, max_val = self.ranges[i]
                norm[:, :, i] = (norm[:, :, i] - min_val) / (max_val - min_val)
                # Clip to [0, 1] to handle outliers (matches LSTM baseline)
                norm[:, :, i] = np.clip(norm[:, :, i], 0.0, 1.0)
        else:
            for i in range(15):
                _, min_val, max_val = self.ranges[i]
                norm[:, i] = (norm[:, i] - min_val) / (max_val - min_val)
                # Clip to [0, 1] to handle outliers (matches LSTM baseline)
                norm[:, i] = np.clip(norm[:, i], 0.0, 1.0)
        
        return norm.astype(np.float32)  # Match LSTM dtype
    
    def clip_to_valid_range(self, data):
        """
        Clip denormalized data to valid physical ranges
        Sometimes generation produces slightly out-of-range values
        
        Args:
            data: denormalized data
        
        Returns:
            clipped data
        """
        clipped = data.copy()
        
        if clipped.ndim == 3:
            for i in range(15):
                _, min_val, max_val = self.ranges[i]
                clipped[:, :, i] = np.clip(clipped[:, :, i], min_val, max_val)
        else:
            for i in range(15):
                _, min_val, max_val = self.ranges[i]
                clipped[:, i] = np.clip(clipped[:, i], min_val, max_val)
        
        return clipped


class VMConfigHandler:
    """
    Handles VM configuration loading and normalization
    Integrates with current_vm_config.yaml
    """
    
    def __init__(self, config_path=None):
        """
        Args:
            config_path: Path to current_vm_config.yaml
                        If None, uses default location
        """
        if config_path is None:
            config_path = Path.home() / 'generative-ai-workload-modeling/outputs/current_vm_config.yaml'
        
        self.config_path = Path(config_path)
        
        # Check if config exists
        if not self.config_path.exists():
            print(f"WARNING: VM config not found at {self.config_path}")
            print("Using default values (current hardware)")
            self._create_default_config()
        
        # Load config
        self.vm_config = self._load_config()
        
        # Normalization ranges (for conditioning)
        self.norm_ranges = {
            'cpu_cores': 64.0,         # Max cores to consider
            'gpu_memory_gb': 40.0,     # Max GPU memory (A100 = 40GB)
            'total_memory_gb': 256.0   # Max RAM (typical max)
        }
    
    def _load_config(self):
        """Load VM config from YAML"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded VM config from {self.config_path}")
        print(f"  CPU cores: {config['cpu_cores']}")
        print(f"  GPU memory: {config['gpu_memory_gb']} GB")
        print(f"  Total memory: {config['total_memory_gb']} GB")
        
        return config
    
    def _create_default_config(self):
        """Create default config if missing"""
        default_config = {
            'cpu_cores': 16,
            'gpu_memory_gb': 15,
            'total_memory_gb': 61,
            'cpu_model': 'AMD EPYC 7643',
            'gpu_model': 'NVIDIA A16'
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default VM config at {self.config_path}")
    
    def get_normalized_config(self):
        """
        Get normalized VM config for conditioning
        
        Returns:
            dict with normalized values (0-1 range)
        """
        return {
            'cpu_cores_norm': self.vm_config['cpu_cores'] / self.norm_ranges['cpu_cores'],
            'gpu_memory_norm': self.vm_config['gpu_memory_gb'] / self.norm_ranges['gpu_memory_gb'],
            'total_memory_norm': self.vm_config['total_memory_gb'] / self.norm_ranges['total_memory_gb']
        }
    
    def get_conditioning_vector(self):
        """
        Get 3D conditioning vector for VM config
        
        Returns:
            numpy array [cpu_norm, gpu_norm, ram_norm]
        """
        norm = self.get_normalized_config()
        return np.array([
            norm['cpu_cores_norm'],
            norm['gpu_memory_norm'],
            norm['total_memory_norm']
        ])
    
    def create_custom_config(self, cpu_cores, gpu_memory_gb, total_memory_gb):
        """
        Create a custom VM config for capacity planning
        
        Args:
            cpu_cores: int
            gpu_memory_gb: float
            total_memory_gb: float
        
        Returns:
            normalized 3D vector
        """
        return np.array([
            cpu_cores / self.norm_ranges['cpu_cores'],
            gpu_memory_gb / self.norm_ranges['gpu_memory_gb'],
            total_memory_gb / self.norm_ranges['total_memory_gb']
        ])


def load_pod_traces(data_dir=None):
    """
    Load pod traces and metadata
    
    Args:
        data_dir: Path to processed data directory
                 If None, uses default location
    
    Returns:
        traces: (N, 715, 15) - NORMALIZED to [0, 1]
        metadata: list of dicts with workload and replica_count
    """
    if data_dir is None:
        data_dir = Path.home() / 'generative-ai-workload-modeling/data/processed/phase1'
    
    data_dir = Path(data_dir)
    
    # Load traces
    traces = np.load(data_dir / 'pod_traces.npy')
    metadata = np.load(data_dir / 'pod_metadata.npy', allow_pickle=True)
    
    print(f"\nLoaded pod traces:")
    print(f"  Shape: {traces.shape}")
    print(f"  Samples: {len(metadata)}")
    
    # Check if normalization is needed
    data_min = traces.min()
    data_max = traces.max()
    
    if data_max > 1.01:
        print(f"  Data NOT normalized (max={data_max:.2f})")
        print(f"  Applying absolute normalization...")
        
        # Apply normalization using DataNormalizer
        normalizer = DataNormalizer()
        traces = normalizer.normalize(traces)
        
        print(f"  After normalization: min={traces.min():.4f}, max={traces.max():.4f}")
        print(f"  Normalization: COMPLETE")
        
        # Save normalized version for future use
        normalized_path = data_dir / 'pod_traces_normalized.npy'
        np.save(normalized_path, traces)
        print(f"  Saved normalized data to: {normalized_path}")
    else:
        print(f"  Normalization: Data already in [0, 1] range")
        print(f"  Validation: OK (min={data_min:.4f}, max={data_max:.4f})")
    
    return traces, metadata


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Data Handler Test")
    print("=" * 60)
    
    # Test normalizer
    print("\n1. Testing DataNormalizer:")
    normalizer = DataNormalizer()
    
    # Test with sample data
    sample_trace = np.random.rand(715, 15)  # Normalized [0, 1]
    denorm_trace = normalizer.denormalize(sample_trace)
    
    print(f"  Normalized CPU range: [{sample_trace[:, 0].min():.3f}, {sample_trace[:, 0].max():.3f}]")
    print(f"  Denormalized CPU range: [{denorm_trace[:, 0].min():.3f}, {denorm_trace[:, 0].max():.3f}]")
    print(f"  Expected: [0, 8] cores")
    
    # Test VM config handler
    print("\n2. Testing VMConfigHandler:")
    vm_handler = VMConfigHandler()
    
    vm_vector = vm_handler.get_conditioning_vector()
    print(f"  VM conditioning vector: {vm_vector}")
    print(f"  Dimensions: {len(vm_vector)}")
    
    # Test custom config
    print("\n3. Testing custom VM config:")
    custom_vector = vm_handler.create_custom_config(
        cpu_cores=32,
        gpu_memory_gb=40,
        total_memory_gb=128
    )
    print(f"  Custom VM (32 cores, 40GB GPU, 128GB RAM)")
    print(f"  Normalized vector: {custom_vector}")
    
    # Test data loading
    print("\n4. Testing data loading:")
    try:
        traces, metadata = load_pod_traces()
        print(f"  Successfully loaded {len(traces)} traces")
    except FileNotFoundError as e:
        print(f"  Data not found: {e}")
        print(f"  This is expected if running outside project directory")
    
    print("\n" + "=" * 60)
    print("Data handler test complete!")
    print("=" * 60)