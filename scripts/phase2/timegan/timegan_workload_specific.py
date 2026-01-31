#!/usr/bin/env python3
"""
Workload-Specific TimeGAN
=========================

Trains SEPARATE TimeGAN models for each workload type (ResNet50, DistilBERT, Whisper).

This addresses the mode collapse problem caused by mixing heterogeneous workloads.

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
CONFIG = {
    'data_path': '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_traces_normalized.npy',
    'metadata_path': '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_metadata.json',
    'output_dir': '/home/hamid/generative-ai-workload-modeling/outputs/timegan_workload_specific',
    
    # Sliding window (smaller for less data per workload)
    'window_size': 64,
    'stride': 8,  # More overlap = more samples
    
    # Model (smaller for less data)
    'input_dim': 15,
    'hidden_dim': 32,
    'latent_dim': 16,
    'num_layers': 2,
    
    # Training
    'batch_size': 16,
    'epochs_embed': 100,
    'epochs_supervisor': 100,
    'epochs_joint': 150,
    'learning_rate': 0.0005,
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42
}

METRICS_NAMES = ['cpu_psi', 'cpu_usage', 'cpu_num', 'cpu_temperature', 
                 'gpu_power', 'gpu_utilization', 'gpu_temperature',
                 'memory_usage', 'memory_psi', 'latency', 'latency_p50',
                 'latency_p95', 'latency_p99', 'success_rate', 'throughput']


def create_sliding_windows(data, window_size, stride):
    """Convert traces to overlapping windows"""
    windows = []
    for trace in data:
        seq_len = trace.shape[0]
        for start in range(0, seq_len - window_size + 1, stride):
            windows.append(trace[start:start + window_size])
    return np.array(windows)


class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.sigmoid(self.fc(h))


class Recovery(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return torch.sigmoid(self.fc(out))


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z):
        h, _ = self.rnn(z)
        return torch.sigmoid(self.fc(h))


class Supervisor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return torch.sigmoid(self.fc(out))


class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return self.fc(out)


class WorkloadTimeGAN:
    """TimeGAN for a single workload type"""
    
    def __init__(self, config, workload_name):
        self.config = config
        self.workload_name = workload_name
        self.device = torch.device(config['device'])
        
        # Initialize networks
        self.embedder = Embedder(
            config['input_dim'], config['hidden_dim'],
            config['latent_dim'], config['num_layers']
        ).to(self.device)
        
        self.recovery = Recovery(
            config['latent_dim'], config['hidden_dim'],
            config['input_dim'], config['num_layers']
        ).to(self.device)
        
        self.generator = Generator(
            config['latent_dim'], config['hidden_dim'],
            config['num_layers']
        ).to(self.device)
        
        self.supervisor = Supervisor(
            config['latent_dim'], config['hidden_dim'],
            config['num_layers']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            config['latent_dim'], config['hidden_dim'],
            config['num_layers']
        ).to(self.device)
        
        # Optimizers
        self.opt_embed = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=config['learning_rate']
        )
        self.opt_supervisor = torch.optim.Adam(
            self.supervisor.parameters(), lr=config['learning_rate']
        )
        self.opt_generator = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=config['learning_rate']
        )
        self.opt_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=config['learning_rate']
        )
        
        self.history = {'embed': [], 'supervisor': [], 'g_loss': [], 'd_loss': []}
    
    def train_embedder(self, dataloader, epochs):
        """Phase 1: Train autoencoder"""
        print(f"\n  Phase 1: Training Embedder ({self.workload_name})")
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                self.opt_embed.zero_grad()
                h = self.embedder(x)
                x_recon = self.recovery(h)
                loss = criterion(x_recon, x)
                loss.backward()
                self.opt_embed.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['embed'].append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
    
    def train_supervisor(self, dataloader, epochs):
        """Phase 2: Train supervisor"""
        print(f"\n  Phase 2: Training Supervisor ({self.workload_name})")
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                self.opt_supervisor.zero_grad()
                
                with torch.no_grad():
                    h = self.embedder(x)
                
                h_supervised = self.supervisor(h)
                loss = criterion(h_supervised[:, :-1, :], h[:, 1:, :])
                loss.backward()
                self.opt_supervisor.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['supervisor'].append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
    
    def train_joint(self, dataloader, epochs):
        """Phase 3: Joint adversarial training"""
        print(f"\n  Phase 3: Joint Training ({self.workload_name})")
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_mse = nn.MSELoss()
        
        for epoch in range(epochs):
            g_losses, d_losses = [], []
            
            for batch in dataloader:
                x = batch[0].to(self.device)
                batch_size = x.size(0)
                seq_len = x.size(1)
                
                z = torch.randn(batch_size, seq_len, self.config['latent_dim']).to(self.device)
                
                with torch.no_grad():
                    h_real = self.embedder(x)
                
                h_fake_raw = self.generator(z)
                h_fake = self.supervisor(h_fake_raw)
                
                # Train Discriminator
                self.opt_discriminator.zero_grad()
                d_real = self.discriminator(h_real)
                d_fake = self.discriminator(h_fake.detach())
                d_loss = criterion_bce(d_real, torch.ones_like(d_real)) + \
                         criterion_bce(d_fake, torch.zeros_like(d_fake))
                d_loss.backward()
                self.opt_discriminator.step()
                
                # Train Generator
                self.opt_generator.zero_grad()
                h_fake_raw = self.generator(z)
                h_fake = self.supervisor(h_fake_raw)
                d_fake = self.discriminator(h_fake)
                
                g_loss_adv = criterion_bce(d_fake, torch.ones_like(d_fake))
                h_supervised = self.supervisor(h_real)
                g_loss_sup = criterion_mse(h_supervised[:, :-1, :], h_real[:, 1:, :])
                
                x_fake = self.recovery(h_fake)
                h_fake_embed = self.embedder(x_fake)
                g_loss_fm = criterion_mse(h_fake, h_fake_embed)
                
                g_loss = g_loss_adv + 10 * g_loss_sup + 10 * g_loss_fm
                g_loss.backward()
                self.opt_generator.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            self.history['g_loss'].append(np.mean(g_losses))
            self.history['d_loss'].append(np.mean(d_losses))
            
            if (epoch + 1) % 30 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | G: {np.mean(g_losses):.4f} | D: {np.mean(d_losses):.4f}")
    
    def generate(self, n_samples=1):
        """Generate synthetic windows"""
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        
        with torch.no_grad():
            z = torch.randn(n_samples, self.config['window_size'], 
                           self.config['latent_dim']).to(self.device)
            h = self.generator(z)
            h = self.supervisor(h)
            x = self.recovery(h)
        
        return x.cpu().numpy()


def split_by_workload(data, metadata):
    """Split data by workload type"""
    workload_data = {'resnet50': [], 'distilbert': [], 'whisper': []}
    
    for i, meta in enumerate(metadata):
        workload = meta.get('workload', 'unknown').lower()
        if 'resnet' in workload:
            workload_data['resnet50'].append(data[i])
        elif 'distilbert' in workload or 'bert' in workload:
            workload_data['distilbert'].append(data[i])
        elif 'whisper' in workload:
            workload_data['whisper'].append(data[i])
    
    return {k: np.array(v) for k, v in workload_data.items() if len(v) > 0}


def evaluate_model(model, test_windows, workload_name):
    """Evaluate a single workload model"""
    n_test = min(len(test_windows), 100)
    
    # Generate samples
    generated = model.generate(n_test)
    test_subset = test_windows[:n_test]
    
    results = {'workload': workload_name, 'per_metric': {}}
    
    for i, name in enumerate(METRICS_NAMES):
        real_vals = test_subset[:, :, i].flatten()
        gen_vals = generated[:, :, i].flatten()
        
        real_std = np.std(real_vals)
        gen_std = np.std(gen_vals)
        
        if real_std > 1e-6 and gen_std > 1e-6:
            corr = np.corrcoef(real_vals, gen_vals)[0, 1]
        else:
            corr = 0.0
        
        results['per_metric'][name] = {
            'real_std': float(real_std),
            'gen_std': float(gen_std),
            'std_ratio': float(gen_std / real_std) if real_std > 0 else 0,
            'correlation': float(corr) if not np.isnan(corr) else 0.0
        }
    
    # Overall metrics
    real_var = np.var(test_subset)
    gen_var = np.var(generated)
    
    results['overall'] = {
        'real_variance': float(real_var),
        'generated_variance': float(gen_var),
        'variance_ratio': float(gen_var / real_var) if real_var > 0 else 0,
        'temporal_variance_ratio': compute_temporal_variance_ratio(test_subset, generated)
    }
    
    return results, generated


def compute_temporal_variance_ratio(real, generated):
    """Compute how well temporal variance is preserved"""
    # Per-sample temporal variance
    real_temporal_var = np.mean([np.var(real[i], axis=0).mean() for i in range(len(real))])
    gen_temporal_var = np.mean([np.var(generated[i], axis=0).mean() for i in range(len(generated))])
    
    return float(gen_temporal_var / real_temporal_var) if real_temporal_var > 0 else 0


def plot_workload_comparison(test_data, generated, workload_name, output_dir):
    """Plot comparison for a single workload"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics_idx = [0, 1, 4, 5, 6, 7]
    names = ['cpu_psi', 'cpu_usage', 'gpu_power', 'gpu_utilization', 'gpu_temperature', 'memory_usage']
    
    for ax, idx, name in zip(axes.flatten(), metrics_idx, names):
        # Plot real samples
        for i in range(min(3, len(test_data))):
            ax.plot(test_data[i, :, idx], 'b-', alpha=0.4, label='Real' if i == 0 else '')
        
        # Plot generated samples
        for i in range(min(3, len(generated))):
            ax.plot(generated[i, :, idx], 'r--', alpha=0.4, label='Generated' if i == 0 else '')
        
        ax.set_title(f'{name}')
        ax.set_xlabel('Time Step')
        ax.legend()
    
    plt.suptitle(f'{workload_name.upper()} - Window Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / f'{workload_name}_comparison.png', dpi=150)
    plt.close()


def main():
    print("="*70)
    print("WORKLOAD-SPECIFIC TIMEGAN")
    print("="*70)
    
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1] Loading data...")
    data = np.load(CONFIG['data_path'])
    
    # Try to load metadata, or create default based on data structure
    try:
        with open(CONFIG['metadata_path'], 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("    Metadata not found, inferring from data structure...")
        # Assume data is organized: ResNet50 (22), DistilBERT (19), Whisper (19)
        metadata = []
        for i in range(22):
            metadata.append({'workload': 'resnet50', 'replica_count': i % 5 + 1})
        for i in range(19):
            metadata.append({'workload': 'distilbert', 'replica_count': i % 4 + 1})
        for i in range(19):
            metadata.append({'workload': 'whisper', 'replica_count': i % 5 + 1})
    
    print(f"    Total traces: {len(data)}")
    
    # Split by workload
    print("\n[2] Splitting by workload...")
    workload_data = split_by_workload(data, metadata)
    
    for wl, wl_data in workload_data.items():
        print(f"    {wl}: {len(wl_data)} traces")
    
    # Train and evaluate each workload
    all_results = {}
    models = {}
    
    for workload_name, wl_data in workload_data.items():
        print(f"\n{'='*70}")
        print(f"TRAINING: {workload_name.upper()}")
        print(f"{'='*70}")
        
        # Create windows
        windows = create_sliding_windows(wl_data, CONFIG['window_size'], CONFIG['stride'])
        print(f"  Windows created: {len(windows)}")
        
        # Train/test split
        n_train = int(len(windows) * 0.8)
        indices = np.random.permutation(len(windows))
        train_windows = windows[indices[:n_train]]
        test_windows = windows[indices[n_train:]]
        
        print(f"  Train: {len(train_windows)}, Test: {len(test_windows)}")
        
        # Create dataloader
        train_tensor = torch.FloatTensor(train_windows)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=CONFIG['batch_size'],
            shuffle=True
        )
        
        # Initialize and train model
        model = WorkloadTimeGAN(CONFIG, workload_name)
        model.train_embedder(train_loader, CONFIG['epochs_embed'])
        model.train_supervisor(train_loader, CONFIG['epochs_supervisor'])
        model.train_joint(train_loader, CONFIG['epochs_joint'])
        
        models[workload_name] = model
        
        # Evaluate
        print(f"\n  Evaluating {workload_name}...")
        results, generated = evaluate_model(model, test_windows, workload_name)
        all_results[workload_name] = results
        
        # Plot
        plot_workload_comparison(test_windows, generated, workload_name, output_dir)
        
        # Print key metrics
        print(f"\n  Results for {workload_name}:")
        print(f"    Variance ratio: {results['overall']['variance_ratio']:.4f}")
        print(f"    Temporal variance ratio: {results['overall']['temporal_variance_ratio']:.4f}")
        
        # Per-metric std ratios
        print(f"    Per-metric std ratios:")
        for name in ['cpu_psi', 'cpu_usage', 'gpu_utilization', 'memory_usage']:
            r = results['per_metric'][name]
            print(f"      {name}: {r['std_ratio']:.4f} (real={r['real_std']:.4f}, gen={r['gen_std']:.4f})")
    
    # Save all results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: WORKLOAD-SPECIFIC RESULTS")
    print("="*70)
    print(f"\n{'Workload':<15} | {'Var Ratio':>10} | {'Temporal Var Ratio':>18} | {'Status':<10}")
    print("-"*70)
    
    for wl, res in all_results.items():
        var_ratio = res['overall']['variance_ratio']
        temp_var_ratio = res['overall']['temporal_variance_ratio']
        
        if temp_var_ratio > 0.5:
            status = "GOOD"
        elif temp_var_ratio > 0.1:
            status = "PARTIAL"
        else:
            status = "POOR"
        
        print(f"{wl:<15} | {var_ratio:>10.4f} | {temp_var_ratio:>18.4f} | {status:<10}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()