#!/usr/bin/env python3
"""
IMPROVED CONDITIONAL VAE FOR ALL WORKLOAD TYPES
================================================

Addresses the flat output problem with multiple improvements:
1. Per-window normalization (amplifies local variations)
2. Spectral loss (forces frequency pattern learning)
3. Beta-VAE scheduling (prevents posterior collapse)
4. Residual connections (preserves fine details)
5. Multi-scale reconstruction loss

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
CONFIG = {
    'data_path': '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_traces_normalized.npy',
    'metadata_path': '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_metadata.json',
    'output_dir': '/home/hamid/generative-ai-workload-modeling/outputs/improved_conditional_vae',
    
    # Metrics
    'core_metrics': ['cpu_usage', 'gpu_utilization', 'memory_usage', 'latency'],
    'core_metric_indices': [1, 5, 7, 9],
    
    # Windowing
    'window_size': 64,
    'stride': 8,
    
    # Normalization strategy: 'global', 'per_window', 'none'
    'normalization': 'per_window',
    
    # Model architecture
    'hidden_dim': 128,
    'latent_dim': 32,
    'num_layers': 2,
    'condition_dim': 16,
    
    # Training
    'batch_size': 32,
    'epochs': 300,
    'learning_rate': 0.001,
    
    # VAE specific
    'beta_start': 0.0,      # Start with pure reconstruction
    'beta_end': 0.1,        # End with small KL weight
    'beta_warmup': 100,     # Epochs to reach beta_end
    
    # Loss weights
    'spectral_weight': 0.5,  # Weight for frequency domain loss
    'temporal_weight': 1.0,  # Weight for temporal difference loss
    
    'max_replica_count': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42
}

ALL_METRICS = ['cpu_psi', 'cpu_usage', 'cpu_num', 'cpu_temperature', 
               'gpu_power', 'gpu_utilization', 'gpu_temperature',
               'memory_usage', 'memory_psi', 'latency', 'latency_p50',
               'latency_p95', 'latency_p99', 'success_rate', 'throughput']


# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================

def per_window_normalize(window):
    """Normalize each window independently to amplify local variations"""
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True) + 1e-8
    normalized = (window - mean) / std
    return normalized, mean, std

def per_window_denormalize(normalized, mean, std):
    """Reverse per-window normalization"""
    return normalized * std + mean


class WindowNormalizer:
    """Handles per-window normalization during training and generation"""
    def __init__(self):
        self.global_means = None
        self.global_stds = None
    
    def fit(self, windows):
        """Compute global statistics for generation phase"""
        # Compute mean and std across all windows for each metric
        self.global_means = windows.mean(axis=(0, 1))  # (n_metrics,)
        self.global_stds = windows.std(axis=(0, 1)) + 1e-8
    
    def normalize_batch(self, windows):
        """Normalize a batch of windows (each independently)"""
        # windows: (batch, seq_len, n_metrics)
        means = windows.mean(axis=1, keepdims=True)  # (batch, 1, n_metrics)
        stds = windows.std(axis=1, keepdims=True) + 1e-8
        normalized = (windows - means) / stds
        return normalized, means, stds
    
    def denormalize_batch(self, normalized, means, stds):
        """Denormalize using stored statistics"""
        return normalized * stds + means


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class ConditionEmbedding(nn.Module):
    """Embed replica count"""
    def __init__(self, condition_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(condition_dim, condition_dim),
            nn.Tanh()
        )
    
    def forward(self, r):
        return self.net(r.unsqueeze(-1))


class ResidualBlock(nn.Module):
    """Residual block for preserving fine details"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x + self.net(x))


class Encoder(nn.Module):
    """VAE Encoder with residual connections"""
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim, num_layers):
        super().__init__()
        self.cond_embed = ConditionEmbedding(condition_dim)
        
        # Bidirectional GRU for better temporal modeling
        self.rnn = nn.GRU(
            input_dim + condition_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        
        # Residual blocks
        self.residual = ResidualBlock(hidden_dim * 2)
        
        # Output mean and log_var
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, x, r):
        batch_size, seq_len, _ = x.shape
        
        cond = self.cond_embed(r).unsqueeze(1).expand(-1, seq_len, -1)
        x_cond = torch.cat([x, cond], dim=-1)
        
        h, _ = self.rnn(x_cond)
        h = self.residual(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder with residual connections"""
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim, num_layers):
        super().__init__()
        self.cond_embed = ConditionEmbedding(condition_dim)
        
        self.rnn = nn.GRU(
            latent_dim + condition_dim, hidden_dim, num_layers,
            batch_first=True
        )
        
        self.residual = ResidualBlock(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z, r):
        batch_size, seq_len, _ = z.shape
        
        cond = self.cond_embed(r).unsqueeze(1).expand(-1, seq_len, -1)
        z_cond = torch.cat([z, cond], dim=-1)
        
        h, _ = self.rnn(z_cond)
        h = self.residual(h)
        
        # No activation - output in normalized space
        return self.fc(h)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def spectral_loss(pred, target):
    """Loss in frequency domain to capture periodic patterns"""
    # FFT along time dimension
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    
    # Compare magnitude spectra
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    return F.mse_loss(pred_mag, target_mag)


def temporal_difference_loss(pred, target):
    """Loss on first differences to capture temporal changes"""
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    target_diff = target[:, 1:, :] - target[:, :-1, :]
    
    return F.mse_loss(pred_diff, target_diff)


def multi_scale_loss(pred, target):
    """Reconstruction loss at multiple time scales"""
    loss = F.mse_loss(pred, target)  # Original scale
    
    # Downsampled scales
    for scale in [2, 4]:
        pred_down = F.avg_pool1d(pred.transpose(1, 2), scale).transpose(1, 2)
        target_down = F.avg_pool1d(target.transpose(1, 2), scale).transpose(1, 2)
        loss = loss + F.mse_loss(pred_down, target_down)
    
    return loss / 3.0


# =============================================================================
# VAE MODEL
# =============================================================================

class ImprovedConditionalVAE:
    """Conditional VAE with improvements for low-variance data"""
    
    def __init__(self, config, input_dim, workload_name):
        self.config = config
        self.input_dim = input_dim
        self.workload_name = workload_name
        self.device = torch.device(config['device'])
        
        self.encoder = Encoder(
            input_dim, config['hidden_dim'], config['latent_dim'],
            config['condition_dim'], config['num_layers']
        ).to(self.device)
        
        self.decoder = Decoder(
            config['latent_dim'], config['hidden_dim'], input_dim,
            config['condition_dim'], config['num_layers']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config['learning_rate']
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=20, factor=0.5
        )
        
        self.normalizer = WindowNormalizer()
        self.history = {'loss': [], 'recon': [], 'kl': [], 'spectral': [], 'temporal': []}
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_beta(self, epoch):
        """Beta scheduling for KL weight"""
        if epoch < self.config['beta_warmup']:
            return self.config['beta_start'] + \
                   (self.config['beta_end'] - self.config['beta_start']) * \
                   (epoch / self.config['beta_warmup'])
        return self.config['beta_end']
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.encoder.train()
        self.decoder.train()
        
        beta = self.get_beta(epoch)
        total_loss, total_recon, total_kl, total_spec, total_temp = 0, 0, 0, 0, 0
        
        for batch in dataloader:
            x, r, means, stds = batch
            x = x.to(self.device)
            r = r.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Encode
            mu, logvar = self.encoder(x, r)
            
            # Reparameterize
            z = self.reparameterize(mu, logvar)
            
            # Decode
            x_recon = self.decoder(z, r)
            
            # Losses
            recon_loss = multi_scale_loss(x_recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            spec_loss = spectral_loss(x_recon, x)
            temp_loss = temporal_difference_loss(x_recon, x)
            
            # Total loss
            loss = recon_loss + \
                   beta * kl_loss + \
                   self.config['spectral_weight'] * spec_loss + \
                   self.config['temporal_weight'] * temp_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_spec += spec_loss.item()
            total_temp += temp_loss.item()
        
        n = len(dataloader)
        self.history['loss'].append(total_loss / n)
        self.history['recon'].append(total_recon / n)
        self.history['kl'].append(total_kl / n)
        self.history['spectral'].append(total_spec / n)
        self.history['temporal'].append(total_temp / n)
        
        self.scheduler.step(total_loss / n)
        
        return total_loss / n
    
    def train(self, train_windows, train_conditions):
        """Full training"""
        print(f"\n{'='*60}")
        print(f"Training {self.workload_name.upper()} Improved VAE")
        print(f"{'='*60}")
        
        # Fit normalizer
        self.normalizer.fit(train_windows)
        
        # Prepare data with per-window normalization
        if self.config['normalization'] == 'per_window':
            normalized, means, stds = self.normalizer.normalize_batch(train_windows)
        else:
            normalized = train_windows
            means = np.zeros((len(train_windows), 1, self.input_dim))
            stds = np.ones((len(train_windows), 1, self.input_dim))
        
        dataset = TensorDataset(
            torch.FloatTensor(normalized),
            torch.FloatTensor(train_conditions),
            torch.FloatTensor(means),
            torch.FloatTensor(stds)
        )
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        for epoch in range(self.config['epochs']):
            loss = self.train_epoch(dataloader, epoch)
            
            if (epoch + 1) % 50 == 0:
                beta = self.get_beta(epoch)
                print(f"  Epoch {epoch+1}/{self.config['epochs']} | "
                      f"Loss: {loss:.6f} | Beta: {beta:.4f}")
    
    def generate(self, replica_count, n_samples=10):
        """Generate traces for a specific replica count"""
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            r_norm = min(replica_count / self.config['max_replica_count'], 1.0)
            r = torch.full((n_samples,), r_norm).to(self.device)
            
            # Sample from prior
            z = torch.randn(n_samples, self.config['window_size'],
                           self.config['latent_dim']).to(self.device)
            
            # Decode
            x_norm = self.decoder(z, r)
            
            # Denormalize using global statistics
            if self.config['normalization'] == 'per_window':
                means = torch.FloatTensor(self.normalizer.global_means).to(self.device)
                stds = torch.FloatTensor(self.normalizer.global_stds).to(self.device)
                x = x_norm * stds + means
            else:
                x = x_norm
        
        return x.cpu().numpy()
    
    def reconstruct(self, windows, conditions):
        """Reconstruct windows (for evaluation)"""
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Normalize
            if self.config['normalization'] == 'per_window':
                normalized, means, stds = self.normalizer.normalize_batch(windows)
            else:
                normalized = windows
                means = np.zeros((len(windows), 1, self.input_dim))
                stds = np.ones((len(windows), 1, self.input_dim))
            
            x = torch.FloatTensor(normalized).to(self.device)
            r = torch.FloatTensor(conditions).to(self.device)
            
            # Encode and decode
            mu, logvar = self.encoder(x, r)
            z = mu  # Use mean for reconstruction
            x_recon_norm = self.decoder(z, r)
            
            # Denormalize
            if self.config['normalization'] == 'per_window':
                x_recon = x_recon_norm.cpu().numpy() * stds + means
            else:
                x_recon = x_recon_norm.cpu().numpy()
        
        return x_recon


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_split_by_workload(config):
    """Load data and split by workload"""
    data = np.load(config['data_path'])
    
    try:
        with open(config['metadata_path'], 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Creating metadata...")
        metadata = []
        for r, count in [(1,1), (2,2), (3,3), (6,6), (10,10)]:
            for _ in range(count):
                metadata.append({'workload': 'resnet50', 'replica_count': r})
        for r, count in [(1,1), (2,2), (6,6), (10,10)]:
            for _ in range(count):
                metadata.append({'workload': 'distilbert', 'replica_count': r})
        for r, count in [(1,1), (2,2), (3,3), (5,5), (8,8)]:
            for _ in range(count):
                metadata.append({'workload': 'whisper', 'replica_count': r})
    
    workload_data = {}
    for wl in ['resnet50', 'distilbert', 'whisper']:
        wl_traces = []
        wl_replica_counts = []
        
        for i, meta in enumerate(metadata):
            if wl in meta.get('workload', '').lower():
                wl_traces.append(data[i])
                wl_replica_counts.append(meta.get('replica_count', 1))
        
        if wl_traces:
            workload_data[wl] = {
                'traces': np.array(wl_traces),
                'replica_counts': wl_replica_counts
            }
    
    return workload_data


def create_windows(traces, replica_counts, config):
    """Create sliding windows"""
    windows = []
    conditions = []
    
    for trace, r in zip(traces, replica_counts):
        seq_len = trace.shape[0]
        
        for start in range(0, seq_len - config['window_size'] + 1, config['stride']):
            window = trace[start:start + config['window_size']]
            window = window[:, config['core_metric_indices']]
            
            windows.append(window)
            conditions.append(r / config['max_replica_count'])
    
    return np.array(windows), np.array(conditions)


def evaluate_model(model, test_windows, test_conditions, config):
    """Evaluate reconstruction and generation quality"""
    results = {}
    metric_names = config['core_metrics']
    
    unique_r = sorted(set(int(c * config['max_replica_count']) for c in test_conditions))
    
    for r in unique_r:
        r_norm = r / config['max_replica_count']
        mask = np.abs(test_conditions - r_norm) < 0.01
        real = test_windows[mask]
        conds = test_conditions[mask]
        
        if len(real) < 3:
            continue
        
        # Generate new samples
        gen = model.generate(r, n_samples=len(real))
        
        # Also get reconstructions
        recon = model.reconstruct(real, conds)
        
        results[r] = {'n_samples': len(real), 'metrics': {}, 'reconstruction': {}}
        
        for i, metric in enumerate(metric_names):
            real_vals = real[:, :, i]
            gen_vals = gen[:, :, i]
            recon_vals = recon[:, :, i]
            
            # Statistics
            real_std = np.std(real_vals)
            gen_std = np.std(gen_vals)
            recon_std = np.std(recon_vals)
            
            # Temporal variance (within-window variance)
            real_temp_var = np.mean([np.var(real_vals[j]) for j in range(len(real_vals))])
            gen_temp_var = np.mean([np.var(gen_vals[j]) for j in range(len(gen_vals))])
            recon_temp_var = np.mean([np.var(recon_vals[j]) for j in range(len(recon_vals))])
            
            results[r]['metrics'][metric] = {
                'real_std': float(real_std),
                'gen_std': float(gen_std),
                'recon_std': float(recon_std),
                'std_ratio_gen': float(gen_std / real_std) if real_std > 1e-6 else 0,
                'std_ratio_recon': float(recon_std / real_std) if real_std > 1e-6 else 0,
                'temp_var_ratio_gen': float(gen_temp_var / real_temp_var) if real_temp_var > 1e-6 else 0,
                'temp_var_ratio_recon': float(recon_temp_var / real_temp_var) if real_temp_var > 1e-6 else 0
            }
        
        # Overall
        real_temp = np.mean([np.var(real[j]) for j in range(len(real))])
        gen_temp = np.mean([np.var(gen[j]) for j in range(len(gen))])
        recon_temp = np.mean([np.var(recon[j]) for j in range(len(recon))])
        
        results[r]['overall_temp_var_ratio_gen'] = float(gen_temp / real_temp) if real_temp > 1e-6 else 0
        results[r]['overall_temp_var_ratio_recon'] = float(recon_temp / real_temp) if real_temp > 1e-6 else 0
    
    return results


def plot_comparison(model, test_windows, test_conditions, workload_name, config, output_dir):
    """Plot real vs generated vs reconstructed"""
    metric_names = config['core_metrics']
    unique_r = sorted(set(int(c * config['max_replica_count']) for c in test_conditions))
    
    fig, axes = plt.subplots(len(unique_r), len(metric_names), 
                            figsize=(4*len(metric_names), 3*len(unique_r)))
    
    if len(unique_r) == 1:
        axes = axes.reshape(1, -1)
    
    for row, r in enumerate(unique_r):
        r_norm = r / config['max_replica_count']
        mask = np.abs(test_conditions - r_norm) < 0.01
        real = test_windows[mask]
        conds = test_conditions[mask]
        
        if len(real) < 1:
            continue
        
        gen = model.generate(r, n_samples=min(3, len(real)))
        recon = model.reconstruct(real[:3], conds[:3])
        
        for col, metric in enumerate(metric_names):
            ax = axes[row, col]
            
            # Real
            for i in range(min(3, len(real))):
                ax.plot(real[i, :, col], 'b-', alpha=0.5, label='Real' if i == 0 else '')
            
            # Generated
            for i in range(len(gen)):
                ax.plot(gen[i, :, col], 'r--', alpha=0.5, label='Generated' if i == 0 else '')
            
            # Reconstructed
            for i in range(len(recon)):
                ax.plot(recon[i, :, col], 'g:', alpha=0.5, label='Reconstructed' if i == 0 else '')
            
            if row == 0:
                ax.set_title(metric)
            if col == 0:
                ax.set_ylabel(f'r={r}')
            ax.legend(fontsize=6)
    
    plt.suptitle(f'{workload_name.upper()} - VAE Results')
    plt.tight_layout()
    plt.savefig(output_dir / f'{workload_name}_comparison.png', dpi=150)
    plt.close()


def plot_training_curves(model, workload_name, output_dir):
    """Plot training history"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].plot(model.history['loss'])
    axes[0].set_title('Total Loss')
    
    axes[1].plot(model.history['recon'])
    axes[1].set_title('Reconstruction Loss')
    
    axes[2].plot(model.history['kl'])
    axes[2].set_title('KL Loss')
    
    axes[3].plot(model.history['spectral'])
    axes[3].set_title('Spectral Loss')
    
    axes[4].plot(model.history['temporal'])
    axes[4].set_title('Temporal Diff Loss')
    
    plt.suptitle(f'{workload_name.upper()} Training')
    plt.tight_layout()
    plt.savefig(output_dir / f'{workload_name}_training.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("IMPROVED CONDITIONAL VAE FOR ALL WORKLOADS")
    print("="*70)
    print(f"Normalization: {CONFIG['normalization']}")
    print(f"Spectral weight: {CONFIG['spectral_weight']}")
    print(f"Temporal weight: {CONFIG['temporal_weight']}")
    print(f"Beta schedule: {CONFIG['beta_start']} -> {CONFIG['beta_end']} over {CONFIG['beta_warmup']} epochs")
    
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1] Loading data...")
    workload_data = load_and_split_by_workload(CONFIG)
    
    for wl, data in workload_data.items():
        print(f"    {wl}: {len(data['traces'])} traces")
    
    all_results = {}
    
    for workload_name, wl_data in workload_data.items():
        print(f"\n{'='*70}")
        print(f"WORKLOAD: {workload_name.upper()}")
        print(f"{'='*70}")
        
        # Create windows
        windows, conditions = create_windows(
            wl_data['traces'], wl_data['replica_counts'], CONFIG
        )
        
        input_dim = windows.shape[2]
        print(f"  Windows: {len(windows)}, Input dim: {input_dim}")
        
        # Train/test split
        n_train = int(len(windows) * 0.8)
        indices = np.random.permutation(len(windows))
        
        train_windows = windows[indices[:n_train]]
        train_conditions = conditions[indices[:n_train]]
        test_windows = windows[indices[n_train:]]
        test_conditions = conditions[indices[n_train:]]
        
        print(f"  Train: {len(train_windows)}, Test: {len(test_windows)}")
        
        # Train model
        model = ImprovedConditionalVAE(CONFIG, input_dim, workload_name)
        model.train(train_windows, train_conditions)
        
        # Evaluate
        print(f"\n  Evaluating {workload_name}...")
        results = evaluate_model(model, test_windows, test_conditions, CONFIG)
        all_results[workload_name] = results
        
        # Print results
        print(f"\n  Results for {workload_name}:")
        for r in sorted(results.keys()):
            res = results[r]
            print(f"    r={r}: gen_temp_var={res['overall_temp_var_ratio_gen']:.4f}, "
                  f"recon_temp_var={res['overall_temp_var_ratio_recon']:.4f}")
        
        # Plot
        plot_comparison(model, test_windows, test_conditions, workload_name, CONFIG, output_dir)
        plot_training_curves(model, workload_name, output_dir)
        
        # Save model
        torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            'normalizer_means': model.normalizer.global_means,
            'normalizer_stds': model.normalizer.global_stds,
            'history': model.history
        }, output_dir / f'{workload_name}_model.pt')
    
    # Save results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for wl, results in all_results.items():
        gen_ratios = [r['overall_temp_var_ratio_gen'] for r in results.values()]
        recon_ratios = [r['overall_temp_var_ratio_recon'] for r in results.values()]
        
        print(f"\n{wl.upper()}:")
        print(f"  Avg generation temp var ratio: {np.mean(gen_ratios):.4f}")
        print(f"  Avg reconstruction temp var ratio: {np.mean(recon_ratios):.4f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()