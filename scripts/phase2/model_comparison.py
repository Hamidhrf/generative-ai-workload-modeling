#!/usr/bin/env python3
"""
Comprehensive Model Comparison for Workload Trace Generation
=============================================================

Tests multiple approaches systematically:
1. Normalization: Global vs Per-workload vs No normalization
2. Architecture: VAE vs GAN vs LSTM-based
3. All with: Workload-specific + Replica count conditioning + Windowing

Goal: Find what works for ALL workloads, not just Whisper

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

# Configuration
CONFIG = {
    'data_path': '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_traces_normalized.npy',
    'raw_data_path': '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_traces_raw.npy',  # If exists
    'metadata_path': '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_metadata.json',
    'output_dir': '/home/hamid/generative-ai-workload-modeling/outputs/model_comparison',
    
    # Metrics to use
    'core_metrics': ['cpu_usage', 'gpu_utilization', 'memory_usage', 'latency'],
    'core_metric_indices': [1, 5, 7, 9],
    
    # Window settings
    'window_size': 64,
    'stride': 8,
    
    # Model settings  
    'hidden_dim': 64,
    'latent_dim': 32,
    'num_layers': 2,
    'condition_dim': 16,
    
    # Training
    'batch_size': 32,
    'epochs': 200,
    'learning_rate': 0.001,
    
    'max_replica_count': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42
}


# =============================================================================
# DATA LOADING AND NORMALIZATION STRATEGIES
# =============================================================================

def load_raw_data():
    """Load data and metadata"""
    data = np.load(CONFIG['data_path'])
    
    # Try to load metadata
    try:
        with open(CONFIG['metadata_path'], 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Creating synthetic metadata...")
        metadata = []
        # ResNet50: r=1,2,3,6,10
        for r, count in [(1,1), (2,2), (3,3), (6,6), (10,10)]:
            for _ in range(count):
                metadata.append({'workload': 'resnet50', 'replica_count': r})
        # DistilBERT: r=1,2,6,10
        for r, count in [(1,1), (2,2), (6,6), (10,10)]:
            for _ in range(count):
                metadata.append({'workload': 'distilbert', 'replica_count': r})
        # Whisper: r=1,2,3,5,8
        for r, count in [(1,1), (2,2), (3,3), (5,5), (8,8)]:
            for _ in range(count):
                metadata.append({'workload': 'whisper', 'replica_count': r})
    
    return data, metadata


def analyze_data_characteristics(data, metadata):
    """Analyze the actual data to understand what we're working with"""
    print("\n" + "="*70)
    print("DATA CHARACTERISTICS ANALYSIS")
    print("="*70)
    
    workload_data = defaultdict(list)
    for i, meta in enumerate(metadata):
        wl = meta.get('workload', 'unknown').lower()
        if 'resnet' in wl:
            wl = 'resnet50'
        elif 'distil' in wl:
            wl = 'distilbert'
        elif 'whisper' in wl:
            wl = 'whisper'
        workload_data[wl].append(data[i])
    
    metric_names = CONFIG['core_metrics']
    metric_indices = CONFIG['core_metric_indices']
    
    stats = {}
    
    for wl in ['resnet50', 'distilbert', 'whisper']:
        if wl not in workload_data:
            continue
            
        traces = np.array(workload_data[wl])
        print(f"\n{wl.upper()}:")
        print(f"  Traces: {len(traces)}, Shape: {traces.shape}")
        
        stats[wl] = {}
        
        for name, idx in zip(metric_names, metric_indices):
            vals = traces[:, :, idx]
            
            # Global stats
            global_min = np.min(vals)
            global_max = np.max(vals)
            global_mean = np.mean(vals)
            global_std = np.std(vals)
            
            # Temporal variance (average variance within each trace)
            temporal_vars = [np.var(vals[i]) for i in range(len(vals))]
            avg_temporal_var = np.mean(temporal_vars)
            
            # Range as percentage of mean
            range_pct = (global_max - global_min) / (global_mean + 1e-8) * 100
            
            stats[wl][name] = {
                'min': global_min,
                'max': global_max,
                'mean': global_mean,
                'std': global_std,
                'temporal_var': avg_temporal_var,
                'range_pct': range_pct
            }
            
            print(f"  {name}:")
            print(f"    Range: [{global_min:.4f}, {global_max:.4f}]")
            print(f"    Mean: {global_mean:.4f}, Std: {global_std:.4f}")
            print(f"    Temporal Var: {avg_temporal_var:.6f}")
            print(f"    Range as % of mean: {range_pct:.1f}%")
    
    return stats


def normalize_per_workload(data, metadata):
    """
    Normalize each workload separately to maximize signal
    This should help with low-variance workloads
    """
    workload_indices = defaultdict(list)
    for i, meta in enumerate(metadata):
        wl = meta.get('workload', 'unknown').lower()
        if 'resnet' in wl:
            wl = 'resnet50'
        elif 'distil' in wl:
            wl = 'distilbert'
        elif 'whisper' in wl:
            wl = 'whisper'
        workload_indices[wl].append(i)
    
    normalized_data = np.copy(data)
    normalization_params = {}
    
    metric_indices = CONFIG['core_metric_indices']
    
    for wl, indices in workload_indices.items():
        wl_data = data[indices]
        normalization_params[wl] = {}
        
        for m_idx in metric_indices:
            vals = wl_data[:, :, m_idx]
            min_val = np.min(vals)
            max_val = np.max(vals)
            
            # Avoid division by zero
            range_val = max_val - min_val
            if range_val < 1e-6:
                range_val = 1.0
            
            # Normalize to [0.1, 0.9] to leave room for generation
            for i in indices:
                normalized_data[i, :, m_idx] = 0.1 + 0.8 * (data[i, :, m_idx] - min_val) / range_val
            
            normalization_params[wl][m_idx] = {'min': min_val, 'max': max_val}
    
    return normalized_data, normalization_params


# =============================================================================
# CONDITIONAL VAE MODEL
# =============================================================================

class ConditionalVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim):
        super().__init__()
        self.condition_embed = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        self.rnn = nn.GRU(input_dim + condition_dim, hidden_dim, 2, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, r):
        batch_size, seq_len, _ = x.shape
        cond = self.condition_embed(r.unsqueeze(-1))
        cond = cond.unsqueeze(1).expand(-1, seq_len, -1)
        x_cond = torch.cat([x, cond], dim=-1)
        h, _ = self.rnn(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConditionalVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim):
        super().__init__()
        self.condition_embed = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        self.rnn = nn.GRU(latent_dim + condition_dim, hidden_dim, 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z, r):
        batch_size, seq_len, _ = z.shape
        cond = self.condition_embed(r.unsqueeze(-1))
        cond = cond.unsqueeze(1).expand(-1, seq_len, -1)
        z_cond = torch.cat([z, cond], dim=-1)
        h, _ = self.rnn(z_cond)
        return torch.sigmoid(self.fc(h))


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim):
        super().__init__()
        self.encoder = ConditionalVAEEncoder(input_dim, hidden_dim, latent_dim, condition_dim)
        self.decoder = ConditionalVAEDecoder(latent_dim, hidden_dim, input_dim, condition_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, r):
        mu, logvar = self.encoder(x, r)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, r)
        return recon, mu, logvar
    
    def generate(self, r, seq_len, n_samples, device):
        """Generate new samples"""
        z = torch.randn(n_samples, seq_len, self.latent_dim).to(device)
        r_tensor = torch.full((n_samples,), r).to(device)
        with torch.no_grad():
            return self.decoder(z, r_tensor)


# =============================================================================
# CONDITIONAL LSTM AUTOENCODER (No adversarial, pure reconstruction)
# =============================================================================

class ConditionalLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim):
        super().__init__()
        self.condition_embed = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # Encoder
        self.encoder = nn.LSTM(input_dim + condition_dim, hidden_dim, 2, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, input_dim)
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x, r):
        batch_size, seq_len, _ = x.shape
        
        # Encode
        cond = self.condition_embed(r.unsqueeze(-1))
        cond_exp = cond.unsqueeze(1).expand(-1, seq_len, -1)
        x_cond = torch.cat([x, cond_exp], dim=-1)
        enc_out, _ = self.encoder(x_cond)
        latent = self.enc_fc(enc_out)
        
        # Add noise during training for regularization
        if self.training:
            latent = latent + torch.randn_like(latent) * 0.1
        
        # Decode
        latent_cond = torch.cat([latent, cond_exp], dim=-1)
        dec_input = torch.relu(self.dec_fc(latent_cond))
        dec_out, _ = self.decoder(dec_input)
        output = torch.sigmoid(self.output_fc(dec_out))
        
        return output, latent
    
    def generate(self, r, seq_len, n_samples, device):
        """Generate by sampling from learned latent space"""
        # Use random latent codes
        z = torch.randn(n_samples, seq_len, self.latent_dim).to(device)
        r_tensor = torch.full((n_samples,), r).to(device)
        
        cond = self.condition_embed(r_tensor.unsqueeze(-1))
        cond_exp = cond.unsqueeze(1).expand(-1, seq_len, -1)
        
        latent_cond = torch.cat([z, cond_exp], dim=-1)
        dec_input = torch.relu(self.dec_fc(latent_cond))
        dec_out, _ = self.decoder(dec_input)
        output = torch.sigmoid(self.output_fc(dec_out))
        
        return output


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_vae(model, dataloader, epochs, device):
    """Train VAE with weighted reconstruction + KL loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    history = []
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for x, r in dataloader:
            x, r = x.to(device), r.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(x, r)
            
            # Reconstruction loss (weighted higher)
            recon_loss = nn.MSELoss()(recon, x)
            
            # KL divergence (weighted lower to preserve variance)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Beta-VAE style: lower KL weight
            loss = recon_loss + 0.001 * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        history.append({
            'loss': total_loss / len(dataloader),
            'recon': total_recon / len(dataloader),
            'kl': total_kl / len(dataloader)
        })
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {history[-1]['loss']:.6f} | "
                  f"Recon: {history[-1]['recon']:.6f} | KL: {history[-1]['kl']:.4f}")
    
    return history


def train_lstm_ae(model, dataloader, epochs, device):
    """Train LSTM Autoencoder"""
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    history = []
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for x, r in dataloader:
            x, r = x.to(device), r.to(device)
            
            optimizer.zero_grad()
            recon, _ = model(x, r)
            
            loss = nn.MSELoss()(recon, x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        history.append(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
    
    return history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, model_type, test_windows, test_conditions, config, device):
    """Evaluate generated vs real data"""
    model.eval()
    results = {}
    
    unique_r = sorted(set(int(c * config['max_replica_count']) for c in test_conditions))
    
    for r in unique_r:
        r_norm = r / config['max_replica_count']
        mask = np.abs(test_conditions - r_norm) < 0.01
        real = test_windows[mask]
        
        if len(real) < 3:
            continue
        
        # Generate
        with torch.no_grad():
            if model_type == 'vae':
                gen = model.generate(r_norm, config['window_size'], len(real), device)
            else:  # lstm_ae
                gen = model.generate(r_norm, config['window_size'], len(real), device)
            gen = gen.cpu().numpy()
        
        results[r] = {'n_samples': len(real), 'metrics': {}}
        
        metric_names = config['core_metrics']
        
        for i, metric in enumerate(metric_names):
            real_vals = real[:, :, i]
            gen_vals = gen[:, :, i]
            
            real_std = np.std(real_vals)
            gen_std = np.std(gen_vals)
            
            real_temp_var = np.mean([np.var(real_vals[j]) for j in range(len(real_vals))])
            gen_temp_var = np.mean([np.var(gen_vals[j]) for j in range(len(gen_vals))])
            
            results[r]['metrics'][metric] = {
                'real_std': float(real_std),
                'gen_std': float(gen_std),
                'std_ratio': float(gen_std / real_std) if real_std > 1e-6 else 0,
                'real_temp_var': float(real_temp_var),
                'gen_temp_var': float(gen_temp_var),
                'temp_var_ratio': float(gen_temp_var / real_temp_var) if real_temp_var > 1e-6 else 0
            }
        
        # Overall
        real_temp = np.mean([np.var(real[j]) for j in range(len(real))])
        gen_temp = np.mean([np.var(gen[j]) for j in range(len(gen))])
        results[r]['overall_temp_var_ratio'] = float(gen_temp / real_temp) if real_temp > 1e-6 else 0
    
    return results


def plot_comparison(model, model_type, test_windows, test_conditions, workload, config, output_dir, device):
    """Plot real vs generated comparison"""
    model.eval()
    
    unique_r = sorted(set(int(c * config['max_replica_count']) for c in test_conditions))[:4]
    metric_names = config['core_metrics']
    
    fig, axes = plt.subplots(len(unique_r), len(metric_names), 
                            figsize=(4*len(metric_names), 3*len(unique_r)))
    
    if len(unique_r) == 1:
        axes = axes.reshape(1, -1)
    
    for row, r in enumerate(unique_r):
        r_norm = r / config['max_replica_count']
        mask = np.abs(test_conditions - r_norm) < 0.01
        real = test_windows[mask]
        
        if len(real) < 1:
            continue
        
        with torch.no_grad():
            gen = model.generate(r_norm, config['window_size'], min(3, len(real)), device)
            gen = gen.cpu().numpy()
        
        for col, metric in enumerate(metric_names):
            ax = axes[row, col]
            
            for i in range(min(3, len(real))):
                ax.plot(real[i, :, col], 'b-', alpha=0.5, label='Real' if i == 0 else '')
            for i in range(len(gen)):
                ax.plot(gen[i, :, col], 'r--', alpha=0.5, label='Gen' if i == 0 else '')
            
            if row == 0:
                ax.set_title(metric)
            if col == 0:
                ax.set_ylabel(f'r={r}')
            ax.legend(fontsize=6)
    
    plt.suptitle(f'{workload.upper()} - {model_type.upper()}')
    plt.tight_layout()
    plt.savefig(output_dir / f'{workload}_{model_type}_comparison.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(workload_name, traces, replica_counts, config, output_dir, device):
    """Run comparison experiment for one workload"""
    print(f"\n{'='*60}")
    print(f"WORKLOAD: {workload_name.upper()}")
    print(f"{'='*60}")
    
    # Create windows
    windows = []
    conditions = []
    
    for trace, r in zip(traces, replica_counts):
        seq_len = trace.shape[0]
        for start in range(0, seq_len - config['window_size'] + 1, config['stride']):
            window = trace[start:start + config['window_size']]
            window = window[:, config['core_metric_indices']]
            windows.append(window)
            conditions.append(r / config['max_replica_count'])
    
    windows = np.array(windows)
    conditions = np.array(conditions)
    
    print(f"  Windows: {len(windows)}, Shape: {windows.shape}")
    
    # Split
    n_train = int(len(windows) * 0.8)
    indices = np.random.permutation(len(windows))
    
    train_windows = windows[indices[:n_train]]
    train_conditions = conditions[indices[:n_train]]
    test_windows = windows[indices[n_train:]]
    test_conditions = conditions[indices[n_train:]]
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_windows), torch.FloatTensor(train_conditions)),
        batch_size=config['batch_size'], shuffle=True
    )
    
    input_dim = windows.shape[2]
    results = {}
    
    # Test 1: Conditional VAE
    print(f"\n  [1] Training Conditional VAE...")
    vae = ConditionalVAE(
        input_dim, config['hidden_dim'], config['latent_dim'], config['condition_dim']
    ).to(device)
    vae_history = train_vae(vae, train_loader, config['epochs'], device)
    vae_results = evaluate_model(vae, 'vae', test_windows, test_conditions, config, device)
    plot_comparison(vae, 'vae', test_windows, test_conditions, workload_name, config, output_dir, device)
    results['vae'] = vae_results
    
    # Test 2: Conditional LSTM Autoencoder
    print(f"\n  [2] Training Conditional LSTM-AE...")
    lstm_ae = ConditionalLSTMAutoencoder(
        input_dim, config['hidden_dim'], config['latent_dim'], config['condition_dim']
    ).to(device)
    lstm_history = train_lstm_ae(lstm_ae, train_loader, config['epochs'], device)
    lstm_results = evaluate_model(lstm_ae, 'lstm_ae', test_windows, test_conditions, config, device)
    plot_comparison(lstm_ae, 'lstm_ae', test_windows, test_conditions, workload_name, config, output_dir, device)
    results['lstm_ae'] = lstm_results
    
    # Summary
    print(f"\n  Results Summary for {workload_name}:")
    for model_name, model_results in results.items():
        avg_ratio = np.mean([r['overall_temp_var_ratio'] for r in model_results.values()])
        print(f"    {model_name}: avg_temporal_var_ratio = {avg_ratio:.4f}")
    
    return results


def main():
    print("="*70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")
    
    # Load data
    print("\n[1] Loading data...")
    data, metadata = load_raw_data()
    print(f"    Loaded {len(data)} traces")
    
    # Analyze data characteristics
    data_stats = analyze_data_characteristics(data, metadata)
    
    # Apply per-workload normalization
    print("\n[2] Applying per-workload normalization...")
    data_normalized, norm_params = normalize_per_workload(data, metadata)
    
    # Split by workload
    workload_data = defaultdict(lambda: {'traces': [], 'replica_counts': []})
    for i, meta in enumerate(metadata):
        wl = meta.get('workload', 'unknown').lower()
        if 'resnet' in wl:
            wl = 'resnet50'
        elif 'distil' in wl:
            wl = 'distilbert'
        elif 'whisper' in wl:
            wl = 'whisper'
        workload_data[wl]['traces'].append(data_normalized[i])
        workload_data[wl]['replica_counts'].append(meta.get('replica_count', 1))
    
    # Run experiments
    all_results = {}
    
    for workload_name in ['resnet50', 'distilbert', 'whisper']:
        if workload_name not in workload_data:
            continue
        
        wl_data = workload_data[workload_name]
        traces = np.array(wl_data['traces'])
        replica_counts = wl_data['replica_counts']
        
        results = run_experiment(
            workload_name, traces, replica_counts, CONFIG, output_dir, device
        )
        all_results[workload_name] = results
    
    # Save results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    print(f"\n{'Workload':<15} {'VAE':<20} {'LSTM-AE':<20}")
    print("-"*55)
    
    for wl, results in all_results.items():
        vae_avg = np.mean([r['overall_temp_var_ratio'] for r in results['vae'].values()])
        lstm_avg = np.mean([r['overall_temp_var_ratio'] for r in results['lstm_ae'].values()])
        print(f"{wl:<15} {vae_avg:<20.4f} {lstm_avg:<20.4f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()