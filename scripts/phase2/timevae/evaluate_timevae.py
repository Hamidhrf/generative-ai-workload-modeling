"""
TimeVAE Evaluation and Generation Script - Organized Structure
For: Master's Thesis - Generative AI Workload Modeling
Location: scripts/phase2/timevae/evaluate_timevae.py

"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

from timevae_architecture import ConditionalTimeVAE, TimeVAEConfig
from data_handler import (
    load_pod_traces,
    VMConfigHandler,
    DataNormalizer
)


class TimeVAEEvaluator:
    """Evaluation and generation utilities for TimeVAE"""
    
    def __init__(self, model_path, config=None):
        """
        Args:
            model_path: Path to saved model checkpoint
            config: TimeVAEConfig (if None, loads from checkpoint)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        if config is None:
            config = TimeVAEConfig()
            config.__dict__.update(checkpoint['config'])
        self.config = config
        
        # Load model
        self.model = ConditionalTimeVAE(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize handlers
        self.vm_handler = VMConfigHandler()
        self.normalizer = DataNormalizer()
        
        # Metrics
        self.metrics = self.normalizer.metric_names
        
        print(f"Loaded model from {model_path}")
        print(f"Best validation MSE: {checkpoint['metrics']['recon']:.6f}")
    
    def evaluate_reconstruction(self, traces, conditions):
        """
        Evaluate reconstruction quality on test data
        
        Args:
            traces: (N, 715, 15) - test traces
            conditions: (N, 7) - conditioning vectors
        
        Returns:
            dict with metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            traces_t = torch.FloatTensor(traces).to(self.device)
            conditions_t = torch.FloatTensor(conditions).to(self.device)
            
            # Reconstruct
            recon, mu, logvar, components = self.model(traces_t, conditions_t)
            
            # Convert back to numpy
            recon_np = recon.cpu().numpy()
            traces_np = traces
            
            # Compute metrics
            mse = mean_squared_error(traces_np.flatten(), recon_np.flatten())
            mae = mean_absolute_error(traces_np.flatten(), recon_np.flatten())
            
            # Per-metric MSE
            per_metric_mse = []
            for i in range(15):
                metric_mse = mean_squared_error(
                    traces_np[:, :, i].flatten(),
                    recon_np[:, :, i].flatten()
                )
                per_metric_mse.append(metric_mse)
            
            results = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'per_metric_mse': per_metric_mse
            }
            
            return results, recon_np
    
    def generate_traces(self, workload, replica_count, vm_handler=None, 
                       num_samples=10, denormalize=False):
        """
        Generate synthetic traces for given conditions
        
        Args:
            workload: 'distilbert', 'resnet50', or 'whisper'
            replica_count: int (1-10 for interpolation, >10 for extrapolation)
            vm_handler: VMConfigHandler (if None, uses self.vm_handler)
            num_samples: number of synthetic traces to generate
            denormalize: if True, return traces in original scale
        
        Returns:
            synthetic_traces: (num_samples, 715, 15)
                            normalized [0,1] if denormalize=False
                            original scale if denormalize=True
        """
        self.model.eval()
        
        if vm_handler is None:
            vm_handler = self.vm_handler
        
        # Build conditioning vector
        condition = self._build_condition(workload, replica_count, vm_handler)
        condition = condition.unsqueeze(0).repeat(num_samples, 1).to(self.device)
        
        # Generate
        with torch.no_grad():
            synthetic = self.model.generate(condition, num_samples)
        
        synthetic_np = synthetic.cpu().numpy()
        
        # Denormalize if requested
        if denormalize:
            synthetic_np = self.normalizer.denormalize(synthetic_np)
            synthetic_np = self.normalizer.clip_to_valid_range(synthetic_np)
        
        return synthetic_np
    
    def _build_condition(self, workload, replica_count, vm_handler=None):
        """
        Build 7D conditioning vector
        
        Args:
            workload: str
            replica_count: int
            vm_handler: VMConfigHandler (if None, uses self.vm_handler)
        """
        if vm_handler is None:
            vm_handler = self.vm_handler
        
        # Workload encoding
        workload_to_idx = {
            'distilbert': 0,
            'resnet50': 1,
            'whisper': 2
        }
        
        # Replica count (normalized)
        r_norm = replica_count / 10.0
        
        # Workload one-hot
        workload_onehot = torch.zeros(3)
        workload_onehot[workload_to_idx[workload]] = 1.0
        
        # VM config (normalized) from handler
        vm_vector = vm_handler.get_conditioning_vector()
        vm_tensor = torch.from_numpy(vm_vector).float()
        
        # Combine
        condition = torch.cat([
            torch.tensor([r_norm]),
            workload_onehot,
            vm_tensor
        ])
        
        return condition
    
    def compare_with_baseline(self, test_traces, test_conditions, baseline_mse=0.006197):
        """
        Compare TimeVAE with LSTM baseline
        
        Args:
            test_traces: Test data
            test_conditions: Conditioning vectors
            baseline_mse: LSTM baseline MSE
        """
        print("\n" + "=" * 60)
        print("TimeVAE vs LSTM Baseline Comparison")
        print("=" * 60)
        
        # Evaluate TimeVAE
        results, recon = self.evaluate_reconstruction(test_traces, test_conditions)
        
        # Print results
        print(f"\nOverall Metrics:")
        print(f"  LSTM Baseline MSE: {baseline_mse:.6f}")
        print(f"  TimeVAE MSE:       {results['mse']:.6f}")
        print(f"  TimeVAE MAE:       {results['mae']:.6f}")
        print(f"  TimeVAE RMSE:      {results['rmse']:.6f}")
        
        if results['mse'] < baseline_mse:
            improvement = (baseline_mse - results['mse']) / baseline_mse * 100
            print(f"\n  IMPROVEMENT: {improvement:.1f}% better than baseline")
            
            if results['mse'] < 0.005:
                print("  GOAL REACHED: MSE < 0.005")
        else:
            worse = (results['mse'] - baseline_mse) / baseline_mse * 100
            print(f"\n  WORSE: {worse:.1f}% worse than baseline")
            print("  Consider hyperparameter tuning")
        
        # Per-metric analysis
        print("\nPer-Metric MSE:")
        for i, (metric, mse) in enumerate(zip(self.metrics, results['per_metric_mse'])):
            print(f"  {metric:25s}: {mse:.6f}")
        
        # Identify best/worst metrics
        best_idx = np.argmin(results['per_metric_mse'])
        worst_idx = np.argmax(results['per_metric_mse'])
        
        print(f"\nBest metric:  {self.metrics[best_idx]} (MSE: {results['per_metric_mse'][best_idx]:.6f})")
        print(f"Worst metric: {self.metrics[worst_idx]} (MSE: {results['per_metric_mse'][worst_idx]:.6f})")
        
        return results
    
    def visualize_reconstruction(self, traces, conditions, num_samples=3, output_dir='outputs/timevae'):
        """
        Visualize reconstruction quality
        
        Args:
            traces: (N, 715, 15) - real traces
            conditions: (N, 7) - conditioning vectors
            num_samples: number of samples to visualize
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get reconstructions
        _, recon = self.evaluate_reconstruction(traces, conditions)
        
        # Select random samples
        indices = np.random.choice(len(traces), size=num_samples, replace=False)
        
        for idx in indices:
            real = traces[idx]
            pred = recon[idx]
            
            # Plot 5 key metrics
            key_metrics = [0, 2, 4, 5, 7]  # cpu, gpu, latency_p50, latency_p95, throughput
            
            fig, axes = plt.subplots(len(key_metrics), 1, figsize=(12, 10))
            
            for i, metric_idx in enumerate(key_metrics):
                ax = axes[i]
                
                # Plot
                ax.plot(real[:, metric_idx], label='Real', alpha=0.7, linewidth=1.5)
                ax.plot(pred[:, metric_idx], label='TimeVAE', alpha=0.7, linewidth=1.5)
                
                # Styling
                ax.set_ylabel(self.metrics[metric_idx], fontsize=10)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    ax.set_title(f'Sample {idx} - Real vs TimeVAE Reconstruction', fontsize=12)
                if i == len(key_metrics) - 1:
                    ax.set_xlabel('Time step', fontsize=10)
            
            plt.tight_layout()
            
            save_path = output_dir / f'reconstruction_sample_{idx}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved reconstruction visualization: {save_path}")
    
    def visualize_generation(self, workload, replica_counts, output_dir='outputs/timevae'):
        """
        Visualize generated traces for different replica counts
        
        Args:
            workload: 'distilbert', 'resnet50', or 'whisper'
            replica_counts: list of replica counts to generate
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Key metrics to visualize
        key_metrics = [0, 2, 4, 7]  # cpu, gpu, latency_p50, throughput
        
        fig, axes = plt.subplots(len(key_metrics), 1, figsize=(12, 10))
        
        for r in replica_counts:
            # Generate traces
            synthetic = self.generate_traces(workload, r, num_samples=5)
            
            # Plot average
            for i, metric_idx in enumerate(key_metrics):
                ax = axes[i]
                
                # Average across samples
                avg_trace = synthetic[:, :, metric_idx].mean(axis=0)
                
                ax.plot(avg_trace, label=f'r={r}', alpha=0.7, linewidth=1.5)
        
        # Styling
        for i, metric_idx in enumerate(key_metrics):
            ax = axes[i]
            ax.set_ylabel(self.metrics[metric_idx], fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_title(f'{workload.capitalize()} - Generated Traces for Different Replica Counts', fontsize=12)
            if i == len(key_metrics) - 1:
                ax.set_xlabel('Time step', fontsize=10)
        
        plt.tight_layout()
        
        save_path = output_dir / f'generation_{workload}_scaling.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved generation visualization: {save_path}")
    
    def statistical_comparison(self, real_traces, synthetic_traces):
        """
        Statistical comparison between real and synthetic traces
        
        Args:
            real_traces: (N, 715, 15)
            synthetic_traces: (M, 715, 15)
        
        Returns:
            dict with statistical metrics
        """
        results = {}
        
        for i, metric in enumerate(self.metrics):
            real_data = real_traces[:, :, i].flatten()
            synth_data = synthetic_traces[:, :, i].flatten()
            
            # Distribution comparison (KS test)
            ks_stat, ks_pval = stats.ks_2samp(real_data, synth_data)
            
            # Mean/std comparison
            real_mean, real_std = real_data.mean(), real_data.std()
            synth_mean, synth_std = synth_data.mean(), synth_data.std()
            
            results[metric] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'real_mean': real_mean,
                'real_std': real_std,
                'synth_mean': synth_mean,
                'synth_std': synth_std,
                'mean_diff_pct': abs(synth_mean - real_mean) / (real_mean + 1e-8) * 100,
                'std_diff_pct': abs(synth_std - real_std) / (real_std + 1e-8) * 100
            }
        
        return results


def main():
    """Main evaluation script"""
    print("=" * 60)
    print("TimeVAE Evaluation - Phase 2")
    print("=" * 60)
    
    # Load model
    model_path = Path('outputs/phase2_timevae/best_model.pt')
    evaluator = TimeVAEEvaluator(model_path)
    
    from sklearn.model_selection import train_test_split

    traces, metadata = load_pod_traces()

    # Use SAME split as training (80/20 with seed 42)
    train_traces, test_traces, train_meta, test_metadata = train_test_split(
        traces, metadata,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    print(f"\nTest set: {len(test_traces)} samples (same split as training)")
    
    # Build test conditions
    test_conditions = []
    for meta in test_metadata:
        cond = evaluator._build_condition(
            meta['workload'],
            meta['replica_count']
        )
        test_conditions.append(cond.numpy())
    test_conditions = np.array(test_conditions)
    
    # 1. Compare with baseline
    evaluator.compare_with_baseline(test_traces, test_conditions)
    
    # 2. Visualize reconstructions
    print("\nGenerating reconstruction visualizations...")
    evaluator.visualize_reconstruction(test_traces, test_conditions, num_samples=3)
    
    # 3. Generate synthetic traces for different scenarios
    print("\nGenerating synthetic traces...")
    
    for workload in ['distilbert', 'resnet50', 'whisper']:
        # Interpolation (within training range)
        replica_counts = [1, 3, 5, 8, 10]
        evaluator.visualize_generation(workload, replica_counts)
        
        # Extrapolation (beyond training range) - DENORMALIZED
        print(f"\n{workload.capitalize()} - Extrapolation to r=50:")
        synthetic_50 = evaluator.generate_traces(
            workload, 50, 
            num_samples=50, 
            denormalize=True  # Get original scale
        )
        print(f"  Generated {len(synthetic_50)} pod traces")
        print(f"  Shape: {synthetic_50.shape}")
        print(f"  Mean CPU (cores): {synthetic_50[:, :, 1].mean():.3f}")          # cpu_usage (index 1)
        print(f"  Mean GPU utilization (%): {synthetic_50[:, :, 5].mean():.3f}") # gpu_utilization (index 5)
        print(f"  Mean latency p50 (ms): {synthetic_50[:, :, 7].mean():.3f}")    # latency_p50 (index 7)
        
        # Save denormalized traces
        save_path = Path('outputs/timevae') / f'synthetic_{workload}_r50_denorm.npy'
        np.save(save_path, synthetic_50)
        print(f"  Saved to: {save_path}")
    
    print("\nEvaluation complete!")
    print("Check outputs/timevae/ for visualizations")


if __name__ == "__main__":
    main()