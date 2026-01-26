#!/usr/bin/env python3
"""
VM Configuration Handler

Manages VM hardware configuration for model conditioning.
Provides normalized features for TimeVAE/TimeGAN input.


"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class VMConfig:
    """VM hardware configuration."""
    cpu_cores: int
    gpu_memory_gb: float
    total_memory_gb: float
    gpu_model: str = "NVIDIA A16"
    cpu_model: str = "AMD EPYC 7643"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'cpu_cores': self.cpu_cores,
            'gpu_memory_gb': self.gpu_memory_gb,
            'total_memory_gb': self.total_memory_gb,
            'gpu_model': self.gpu_model,
            'cpu_model': self.cpu_model
        }
    
    def to_normalized_vector(self, 
                            max_cpu: int = 64,
                            max_gpu_mem: float = 64.0,
                            max_total_mem: float = 512.0) -> np.ndarray:
        """
        Convert to normalized feature vector for model input.
        
        Args:
            max_cpu: Maximum CPU cores to normalize against
            max_gpu_mem: Maximum GPU memory (GB) to normalize against
            max_total_mem: Maximum total memory (GB) to normalize against
            
        Returns:
            Normalized vector (3,) with values in [0,1]
        """
        return np.array([
            self.cpu_cores / max_cpu,
            self.gpu_memory_gb / max_gpu_mem,
            self.total_memory_gb / max_total_mem
        ], dtype=np.float32)
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'VMConfig':
        """Create VMConfig from dictionary."""
        return cls(
            cpu_cores=config['cpu_cores'],
            gpu_memory_gb=config['gpu_memory_gb'],
            total_memory_gb=config['total_memory_gb'],
            gpu_model=config.get('gpu_model', 'NVIDIA A16'),
            cpu_model=config.get('cpu_model', 'AMD EPYC 7643')
        )


# Current VM (Phase 1 experiments)
CURRENT_VM = VMConfig(
    cpu_cores=16,
    gpu_memory_gb=15.0,
    total_memory_gb=61.0,
    gpu_model="NVIDIA A16",
    cpu_model="AMD EPYC 7643"
)


# Example cloud instance configurations for future use
CLOUD_INSTANCES = {
    'aws_p3_2xlarge': VMConfig(
        cpu_cores=8,
        gpu_memory_gb=16.0,
        total_memory_gb=61.0,
        gpu_model="NVIDIA V100",
        cpu_model="Intel Xeon E5-2686 v4"
    ),
    
    'aws_p3_8xlarge': VMConfig(
        cpu_cores=32,
        gpu_memory_gb=64.0,  # 4x V100 16GB
        total_memory_gb=244.0,
        gpu_model="NVIDIA V100",
        cpu_model="Intel Xeon E5-2686 v4"
    ),
    
    'azure_nc6s_v3': VMConfig(
        cpu_cores=6,
        gpu_memory_gb=16.0,
        total_memory_gb=112.0,
        gpu_model="NVIDIA V100",
        cpu_model="Intel Xeon E5-2690 v4"
    ),
    
    'azure_nc24s_v3': VMConfig(
        cpu_cores=24,
        gpu_memory_gb=64.0,  # 4x V100 16GB
        total_memory_gb=448.0,
        gpu_model="NVIDIA V100",
        cpu_model="Intel Xeon E5-2690 v4"
    ),
    
    'gcp_n1_standard_8_t4': VMConfig(
        cpu_cores=8,
        gpu_memory_gb=16.0,
        total_memory_gb=30.0,
        gpu_model="NVIDIA T4",
        cpu_model="Intel Skylake"
    ),
}


def get_vm_config(name: str = 'current') -> VMConfig:
    """
    Get VM configuration by name.
    
    Args:
        name: Configuration name ('current', 'aws_p3_2xlarge', etc.)
        
    Returns:
        VMConfig object
    """
    if name == 'current':
        return CURRENT_VM
    
    if name in CLOUD_INSTANCES:
        return CLOUD_INSTANCES[name]
    
    raise ValueError(f"Unknown VM config: {name}. Available: {list(CLOUD_INSTANCES.keys())}")


def create_vm_embedding(vm_config: VMConfig,
                       max_cpu: int = 64,
                       max_gpu_mem: float = 64.0,
                       max_total_mem: float = 512.0) -> np.ndarray:
    """
    Create normalized embedding for VM configuration.
    
    This is used as conditioning input to TimeVAE/TimeGAN.
    
    Args:
        vm_config: VM configuration
        max_cpu: Maximum CPU cores for normalization
        max_gpu_mem: Maximum GPU memory for normalization
        max_total_mem: Maximum total memory for normalization
        
    Returns:
        Normalized vector (3,) for model input
    """
    return vm_config.to_normalized_vector(max_cpu, max_gpu_mem, max_total_mem)


def compare_vm_configs(vm1: VMConfig, vm2: VMConfig) -> Dict[str, float]:
    """
    Compare two VM configurations.
    
    Returns:
        Dictionary with ratio comparisons
    """
    return {
        'cpu_ratio': vm2.cpu_cores / vm1.cpu_cores,
        'gpu_memory_ratio': vm2.gpu_memory_gb / vm1.gpu_memory_gb,
        'total_memory_ratio': vm2.total_memory_gb / vm1.total_memory_gb
    }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("VM CONFIGURATION HANDLER")
    print("="*80)
    
    # Current VM
    print("\nCurrent VM (Phase 1 experiments):")
    print(f"  CPU: {CURRENT_VM.cpu_cores} cores ({CURRENT_VM.cpu_model})")
    print(f"  GPU: {CURRENT_VM.gpu_memory_gb} GB ({CURRENT_VM.gpu_model})")
    print(f"  Memory: {CURRENT_VM.total_memory_gb} GB")
    print(f"  Normalized: {CURRENT_VM.to_normalized_vector()}")
    
    # Cloud instances
    print("\n" + "="*80)
    print("Example Cloud Instance Configs:")
    print("="*80)
    
    for name, config in CLOUD_INSTANCES.items():
        print(f"\n{name}:")
        print(f"  CPU: {config.cpu_cores} cores")
        print(f"  GPU: {config.gpu_memory_gb} GB ({config.gpu_model})")
        print(f"  Memory: {config.total_memory_gb} GB")
        print(f"  Normalized: {config.to_normalized_vector()}")
        
        # Compare to current
        ratios = compare_vm_configs(CURRENT_VM, config)
        print(f"  vs Current: {ratios['cpu_ratio']:.2f}x CPU, {ratios['gpu_memory_ratio']:.2f}x GPU")
    
    print("\n" + "="*80)
    print("Usage in Model Training:")
    print("="*80)
    print("""
# During training:
vm_embedding = create_vm_embedding(CURRENT_VM)

training_sample = {
    'trace': trace_normalized,      # (715, 15)
    'replica_count': 3,
    'workload': 'resnet50',
    'vm_config': vm_embedding       # (3,) ← VM conditioning
}

# During generation for different hardware:
target_vm = get_vm_config('aws_p3_8xlarge')
target_embedding = create_vm_embedding(target_vm)

synthetic = model.generate(
    replica_count=50,
    workload='resnet50',
    vm_config=target_embedding  # Different hardware!
)
    """)