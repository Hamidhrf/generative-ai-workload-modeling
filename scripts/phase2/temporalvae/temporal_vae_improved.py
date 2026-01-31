#!/usr/bin/env python3
"""
Improved Temporal VAE for AI Workload Trace Generation
Addresses posterior collapse with:
- Temporal modeling (Conv1D + LSTM)
- KL annealing
- Beta-VAE (beta < 1)
- Advanced evaluation metrics

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_dir': Path.home() / 'generative-ai-workload-modeling/data/processed/phase1',
    'output_dir': Path.home() / 'generative-ai-workload-modeling/outputs/temporal_vae_extreme',
    'model_name': 'temporal_vae_extreme',
    
    # Model architecture
    'input_dim': 15,
    'hidden_dim': 128,
    'latent_dim': 16,
    'conv_channels': [64, 128],
    
    # Training hyperparameters
    'beta': 0.01,  # EXTREME: Beta-VAE weight (< 1 to prevent collapse)
    'kl_warmup_epochs': 100,  # EXTREME: KL annealing warmup
    'recon_weight': 5.0,  # EXTREME: Reconstruction weight
    'learning_rate': 0.0005,
    'batch_size': 8,
    'epochs': 300,
    'patience': 50,
    
    # Data
    'seq_len': 715,
    'train_split': 0.8,
    'random_seed': 42,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

class TemporalVAE(nn.Module):
    """
    Temporal VAE with Conv1D + LSTM architecture
    Designed to prevent posterior collapse and preserve temporal dynamics
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, conv_channels, beta=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: Conv1D layers for temporal feature extraction
        encoder_layers = []
        in_channels = input_dim
        for out_channels in conv_channels:
            encoder_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_channels = out_channels
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # LSTM for sequential modeling
        self.encoder_rnn = nn.LSTM(
            conv_channels[-1], 
            hidden_dim, 
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Latent space projections
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: LSTM + ConvTranspose
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Reverse conv layers
        decoder_layers = []
        in_channels = hidden_dim
        for i in range(len(conv_channels)-1, 0, -1):
            out_channels = conv_channels[i-1]
            decoder_layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_channels = out_channels
        
        # Final layer to input_dim
        decoder_layers.append(
            nn.ConvTranspose1d(in_channels, input_dim, kernel_size=3, padding=1)
        )
        self.decoder_conv = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input sequence to latent distribution"""
        # x: [batch, seq_len, features]
        batch_size, seq_len, _ = x.shape
        
        # Conv expects [batch, features, seq_len]
        x_conv = x.permute(0, 2, 1)
        
        # Apply conv layers
        z_conv = self.encoder_conv(x_conv)
        
        # Back to [batch, seq_len, features]
        z_conv = z_conv.permute(0, 2, 1)
        
        # LSTM encoding
        _, (h_n, _) = self.encoder_rnn(z_conv)
        
        # Use last hidden state
        h = h_n[-1]  # [batch, hidden_dim]
        
        # Project to latent space
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z, seq_len):
        """Decode latent vector to sequence"""
        batch_size = z.size(0)
        
        # Project latent to hidden dim
        h = self.decoder_fc(z)  # [batch, hidden_dim]
        
        # Repeat for sequence length
        h_seq = h.unsqueeze(1).repeat(1, seq_len, 1)
        
        # LSTM decoding
        dec_out, _ = self.decoder_rnn(h_seq)
        
        # Reshape for conv: [batch, hidden_dim, seq_len]
        dec_conv_in = dec_out.permute(0, 2, 1)
        
        # Apply conv layers
        recon = self.decoder_conv(dec_conv_in)
        
        # Back to [batch, seq_len, features]
        recon = recon.permute(0, 2, 1)
        
        return recon
    
    def forward(self, x):
        """Full forward pass"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mean, logvar

class WorkloadDataset(Dataset):
    """Dataset for pod-level traces"""
    def __init__(self, data, scaler=None):
        self.data = torch.FloatTensor(data)
        self.scaler = scaler
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def kl_weight_schedule(epoch, warmup_epochs):
    """KL annealing schedule"""
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

def load_data(data_dir):
    """Load pod-level traces from Phase 1 (numpy format)"""
    data_dir = Path(data_dir).expanduser()
    
    print(f"Loading data from {data_dir}")
    
    # Load numpy files
    traces_file = data_dir / 'pod_traces.npy'
    normalized_file = data_dir / 'pod_traces_normalized.npy'
    metadata_file = data_dir / 'pod_metadata.npy'
    
    # Check if files exist
    if not traces_file.exists():
        raise FileNotFoundError(f"Could not find {traces_file}")
    
    # Load traces (prefer normalized if available)
    if normalized_file.exists():
        print(f"Loading normalized traces from {normalized_file}")
        data = np.load(normalized_file)
        print("Using pre-normalized data")
    else:
        print(f"Loading raw traces from {traces_file}")
        data = np.load(traces_file)
        print("Will normalize during training")
    
    # Load metadata if available
    metadata = []
    if metadata_file.exists():
        print(f"Loading metadata from {metadata_file}")
        metadata = np.load(metadata_file, allow_pickle=True)
        if len(metadata) > 0:
            print(f"Loaded metadata for {len(metadata)} pods")
    
    print(f"\nLoaded {len(data)} pod traces")
    print(f"Data shape: {data.shape}")
    
    if len(data.shape) == 3:
        print(f"  Number of samples: {data.shape[0]}")
        print(f"  Sequence length: {data.shape[1]}")
        print(f"  Number of features: {data.shape[2]}")
    
    return data, metadata

def compute_dtw_distance(seq1, seq2):
    """Compute Dynamic Time Warping distance"""
    n, m = len(seq1), len(seq2)
    dtw = np.zeros((n+1, m+1))
    dtw[0, 1:] = np.inf
    dtw[1:, 0] = np.inf
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.abs(seq1[i-1] - seq2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]

def compute_autocorrelation(seq, max_lag=50):
    """Compute autocorrelation function"""
    seq = seq - np.mean(seq)
    autocorr = np.correlate(seq, seq, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    return autocorr[:max_lag]

def compute_spectral_error(real, generated):
    """Compute spectral error between sequences"""
    real_fft = np.fft.fft(real)
    gen_fft = np.fft.fft(generated)
    
    real_power = np.abs(real_fft)**2
    gen_power = np.abs(gen_fft)**2
    
    return np.mean(np.abs(real_power - gen_power))

def evaluate_generation_quality(real_data, generated_data, metric_names):
    """Comprehensive evaluation metrics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*80)
    
    results = {
        'per_metric': {},
        'temporal': {},
        'overall': {}
    }
    
    # 1. Per-metric MSE, MAE
    print("\n1. PER-METRIC RECONSTRUCTION ERRORS:")
    print("-" * 80)
    
    for i, metric_name in enumerate(metric_names):
        real_metric = real_data[:, :, i].flatten()
        gen_metric = generated_data[:, :, i].flatten()
        
        mse = mean_squared_error(real_metric, gen_metric)
        mae = mean_absolute_error(real_metric, gen_metric)
        
        # Correlation
        if np.std(real_metric) > 0 and np.std(gen_metric) > 0:
            corr, _ = pearsonr(real_metric, gen_metric)
        else:
            corr = 0.0
        
        results['per_metric'][metric_name] = {
            'mse': float(mse),
            'mae': float(mae),
            'correlation': float(corr)
        }
        
        print(f"{metric_name:25s} | MSE: {mse:10.6f} | MAE: {mae:10.6f} | Corr: {corr:6.3f}")
    
    # 2. DTW distances
    print("\n2. DYNAMIC TIME WARPING (DTW) ANALYSIS:")
    print("-" * 80)
    
    dtw_distances = []
    for metric_idx in [0, 1, 4, 5]:  # Sample of key metrics
        metric_name = metric_names[metric_idx]
        dtw_vals = []
        
        for sample_idx in range(min(10, len(real_data))):
            real_seq = real_data[sample_idx, :, metric_idx]
            gen_seq = generated_data[sample_idx, :, metric_idx]
            dtw_dist = compute_dtw_distance(real_seq, gen_seq)
            dtw_vals.append(dtw_dist)
        
        avg_dtw = np.mean(dtw_vals)
        dtw_distances.append(avg_dtw)
        
        print(f"{metric_name:25s} | Avg DTW: {avg_dtw:10.4f}")
    
    results['temporal']['avg_dtw'] = float(np.mean(dtw_distances))
    
    # 3. Autocorrelation similarity
    print("\n3. AUTOCORRELATION SIMILARITY:")
    print("-" * 80)
    
    autocorr_errors = []
    for metric_idx in [0, 1, 4, 5]:
        metric_name = metric_names[metric_idx]
        acf_errors = []
        
        for sample_idx in range(min(10, len(real_data))):
            real_seq = real_data[sample_idx, :, metric_idx]
            gen_seq = generated_data[sample_idx, :, metric_idx]
            
            real_acf = compute_autocorrelation(real_seq)
            gen_acf = compute_autocorrelation(gen_seq)
            
            acf_error = np.mean(np.abs(real_acf - gen_acf))
            acf_errors.append(acf_error)
        
        avg_acf_error = np.mean(acf_errors)
        autocorr_errors.append(avg_acf_error)
        
        print(f"{metric_name:25s} | ACF Error: {avg_acf_error:10.6f}")
    
    results['temporal']['avg_acf_error'] = float(np.mean(autocorr_errors))
    
    # 4. Spectral analysis
    print("\n4. SPECTRAL ANALYSIS:")
    print("-" * 80)
    
    spectral_errors = []
    for metric_idx in [0, 1, 4, 5]:
        metric_name = metric_names[metric_idx]
        spec_errors = []
        
        for sample_idx in range(min(10, len(real_data))):
            real_seq = real_data[sample_idx, :, metric_idx]
            gen_seq = generated_data[sample_idx, :, metric_idx]
            
            spec_error = compute_spectral_error(real_seq, gen_seq)
            spec_errors.append(spec_error)
        
        avg_spec_error = np.mean(spec_errors)
        spectral_errors.append(avg_spec_error)
        
        print(f"{metric_name:25s} | Spectral Error: {avg_spec_error:10.4f}")
    
    results['temporal']['avg_spectral_error'] = float(np.mean(spectral_errors))
    
    # 5. Overall statistics
    print("\n5. OVERALL STATISTICS:")
    print("-" * 80)
    
    # Variance preservation
    real_var = np.var(real_data)
    gen_var = np.var(generated_data)
    var_ratio = gen_var / real_var if real_var > 0 else 0
    
    print(f"Real data variance:          {real_var:.6f}")
    print(f"Generated data variance:     {gen_var:.6f}")
    print(f"Variance ratio (gen/real):   {var_ratio:.6f}")
    
    # Range preservation
    real_range = np.max(real_data) - np.min(real_data)
    gen_range = np.max(generated_data) - np.min(generated_data)
    range_ratio = gen_range / real_range if real_range > 0 else 0
    
    print(f"Real data range:             {real_range:.6f}")
    print(f"Generated data range:        {gen_range:.6f}")
    print(f"Range ratio (gen/real):      {range_ratio:.6f}")
    
    results['overall'] = {
        'variance_ratio': float(var_ratio),
        'range_ratio': float(range_ratio),
        'real_variance': float(real_var),
        'generated_variance': float(gen_var)
    }
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("-" * 80)
    
    if var_ratio < 0.5:
        print("WARNING: Generated traces have much lower variance than real data.")
        print("         This suggests over-smoothing / posterior collapse.")
    elif var_ratio > 0.8 and var_ratio < 1.2:
        print("GOOD: Variance is well preserved.")
    
    if results['temporal']['avg_dtw'] > 100:
        print("WARNING: High DTW distances suggest poor temporal alignment.")
    else:
        print("GOOD: DTW distances are reasonable.")
    
    if results['temporal']['avg_acf_error'] > 0.3:
        print("WARNING: High ACF error suggests temporal dynamics not captured.")
    else:
        print("GOOD: Temporal dynamics preserved.")
    
    print("="*80 + "\n")
    
    return results

def visualize_reconstructions(model, test_loader, scaler, metric_names, output_dir, num_samples=3):
    """Generate reconstruction visualizations"""
    model.eval()
    
    with torch.no_grad():
        # Get test batch
        test_batch = next(iter(test_loader))
        test_batch = test_batch.to(CONFIG['device'])
        
        # Generate reconstructions
        recon, mean, logvar = model(test_batch)
        
        # Move to CPU and inverse transform
        real = test_batch.cpu().numpy()
        generated = recon.cpu().numpy()
        
        # Inverse transform
        real_inv = np.zeros_like(real)
        gen_inv = np.zeros_like(generated)
        
        for i in range(real.shape[0]):
            real_inv[i] = scaler.inverse_transform(real[i])
            gen_inv[i] = scaler.inverse_transform(generated[i])
        
        # Plot reconstructions for key metrics
        key_metrics = [0, 1, 4, 5, 6, 7]  # Sample of important metrics
        
        for sample_idx in range(min(num_samples, len(real_inv))):
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'Sample {sample_idx+1} - Reconstruction Comparison', fontsize=14, fontweight='bold')
            
            for plot_idx, metric_idx in enumerate(key_metrics):
                ax = axes[plot_idx // 2, plot_idx % 2]
                
                real_seq = real_inv[sample_idx, :, metric_idx]
                gen_seq = gen_inv[sample_idx, :, metric_idx]
                
                ax.plot(real_seq, label='Real', alpha=0.8, linewidth=1.5, color='blue')
                ax.plot(gen_seq, label='Generated', alpha=0.8, linewidth=1.5, color='red', linestyle='--')
                
                ax.set_title(metric_names[metric_idx], fontsize=10, fontweight='bold')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'reconstruction_sample_{sample_idx+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved reconstruction visualization for sample {sample_idx+1}")

def visualize_latent_space(model, test_loader, output_dir):
    """Visualize latent space using t-SNE or PCA"""
    from sklearn.manifold import TSNE
    
    model.eval()
    
    all_means = []
    all_logvars = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(CONFIG['device'])
            mean, logvar = model.encode(batch)
            all_means.append(mean.cpu().numpy())
            all_logvars.append(logvar.cpu().numpy())
    
    means = np.vstack(all_means)
    logvars = np.vstack(all_logvars)
    
    # t-SNE on means
    if len(means) > 2:
        # Use lower perplexity for small datasets (perplexity must be < n_samples)
        perplexity = min(5, len(means) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        means_2d = tsne.fit_transform(means)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(means_2d[:, 0], means_2d[:, 1], alpha=0.6, s=50)
        plt.title('Latent Space Visualization (t-SNE)', fontsize=14, fontweight='bold')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'latent_space_tsne.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Saved latent space visualization")
    
    # KL divergence distribution
    kl_divs = -0.5 * np.sum(1 + logvars - means**2 - np.exp(logvars), axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(kl_divs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(kl_divs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(kl_divs):.2f}')
    plt.title('KL Divergence Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('KL Divergence')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'kl_divergence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Average KL divergence: {np.mean(kl_divs):.4f}")
    print(f"Std KL divergence: {np.std(kl_divs):.4f}")

def train_model():
    """Main training function"""
    print("="*80)
    print("TEMPORAL VAE TRAINING - IMPROVED ARCHITECTURE")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        if key != 'data_dir' and key != 'output_dir':
            print(f"  {key}: {value}")
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir']).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data, metadata = load_data(CONFIG['data_dir'])
    print(f"\nData shape: {data.shape}")
    
    # Update CONFIG with actual data dimensions
    if len(data.shape) == 3:
        actual_seq_len = data.shape[1]
        actual_input_dim = data.shape[2]
        
        if actual_seq_len != CONFIG['seq_len']:
            print(f"\nWarning: CONFIG seq_len ({CONFIG['seq_len']}) != actual ({actual_seq_len})")
            print(f"Updating CONFIG['seq_len'] to {actual_seq_len}")
            CONFIG['seq_len'] = actual_seq_len
        
        if actual_input_dim != CONFIG['input_dim']:
            print(f"\nWarning: CONFIG input_dim ({CONFIG['input_dim']}) != actual ({actual_input_dim})")
            print(f"Updating CONFIG['input_dim'] to {actual_input_dim}")
            CONFIG['input_dim'] = actual_input_dim
    
    # Check if data is already normalized
    data_min, data_max = data.min(), data.max()
    print(f"Data range: [{data_min:.4f}, {data_max:.4f}]")
    
    # Normalize data if not already normalized
    if data_min < -0.1 or data_max > 1.1:
        print("\nNormalizing data...")
        scaler = StandardScaler()
        data_flat = data.reshape(-1, CONFIG['input_dim'])
        scaler.fit(data_flat)
        
        data_normalized = np.zeros_like(data)
        for i in range(len(data)):
            data_normalized[i] = scaler.transform(data[i])
    else:
        print("\nData already normalized, skipping normalization...")
        data_normalized = data
        # Create a dummy scaler for inverse transform
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(CONFIG['input_dim'])
        scaler.scale_ = np.ones(CONFIG['input_dim'])
        scaler.n_features_in_ = CONFIG['input_dim']
    
    # Split data
    n_samples = len(data_normalized)
    n_train = int(n_samples * CONFIG['train_split'])
    
    train_data = data_normalized[:n_train]
    test_data = data_normalized[n_train:]
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create datasets and dataloaders
    train_dataset = WorkloadDataset(train_data, scaler)
    test_dataset = WorkloadDataset(test_data, scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Initialize model
    print("\nInitializing Temporal VAE...")
    model = TemporalVAE(
        input_dim=CONFIG['input_dim'],  # Use updated value
        hidden_dim=CONFIG['hidden_dim'],
        latent_dim=CONFIG['latent_dim'],
        conv_channels=CONFIG['conv_channels'],
        beta=CONFIG['beta']
    ).to(CONFIG['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input dimensions: {CONFIG['input_dim']} features, {CONFIG['seq_len']} timesteps")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Metric names
    metric_names = [
        'cpu_psi', 'cpu_usage', 'cpu_num', 'cpu_temperature',
        'gpu_power', 'gpu_utilization', 'gpu_temperature',
        'memory_usage', 'memory_psi',
        'latency', 'latency_p50', 'latency_p95', 'latency_p99',
        'success_rate', 'throughput'
    ]
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'test_loss': [],
        'test_recon': [],
        'test_kl': [],
        'kl_weight': []
    }
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(CONFIG['epochs']):
        # KL weight schedule
        kl_weight = kl_weight_schedule(epoch, CONFIG['kl_warmup_epochs'])
        history['kl_weight'].append(kl_weight)
        
        # Training
        model.train()
        train_loss_total = 0
        train_recon_total = 0
        train_kl_total = 0
        
        for batch in train_loader:
            batch = batch.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, mean, logvar = model(batch)
            
            # Reconstruction loss (weighted)
            recon_loss = nn.MSELoss()(recon, batch) * CONFIG['recon_weight']
            
            # KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_div = kl_div / batch.size(0)  # Average over batch
            
            # Total loss with annealed KL
            loss = recon_loss + kl_weight * CONFIG['beta'] * kl_div
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            train_recon_total += recon_loss.item()
            train_kl_total += kl_div.item()
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_recon = train_recon_total / len(train_loader)
        avg_train_kl = train_kl_total / len(train_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['train_recon'].append(avg_train_recon)
        history['train_kl'].append(avg_train_kl)
        
        # Validation
        model.eval()
        test_loss_total = 0
        test_recon_total = 0
        test_kl_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(CONFIG['device'])
                
                recon, mean, logvar = model(batch)
                
                recon_loss = nn.MSELoss()(recon, batch) * CONFIG['recon_weight']
                kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                kl_div = kl_div / batch.size(0)
                
                loss = recon_loss + kl_weight * CONFIG['beta'] * kl_div
                
                test_loss_total += loss.item()
                test_recon_total += recon_loss.item()
                test_kl_total += kl_div.item()
        
        avg_test_loss = test_loss_total / len(test_loader)
        avg_test_recon = test_recon_total / len(test_loader)
        avg_test_kl = test_kl_total / len(test_loader)
        
        history['test_loss'].append(avg_test_loss)
        history['test_recon'].append(avg_test_recon)
        history['test_kl'].append(avg_test_kl)
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
            print(f"  KL Weight: {kl_weight:.4f}")
            print(f"  Train Loss: {avg_train_loss:.6f} | Recon: {avg_train_recon:.6f} | KL: {avg_train_kl:.6f}")
            print(f"  Test Loss:  {avg_test_loss:.6f} | Recon: {avg_test_recon:.6f} | KL: {avg_test_kl:.6f}")
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': avg_test_loss,
                'config': CONFIG
            }, output_dir / 'best_model.pt')
            
            print(f"  >>> Best model saved (test_loss: {avg_test_loss:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    print("\n" + "="*80)
    print("LOADING BEST MODEL FOR EVALUATION")
    print("="*80)
    
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with test loss {checkpoint['test_loss']:.6f}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(history['test_loss'], label='Test', alpha=0.7)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(history['train_recon'], label='Train', alpha=0.7)
    axes[0, 1].plot(history['test_recon'], label='Test', alpha=0.7)
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[1, 0].plot(history['train_kl'], label='Train', alpha=0.7)
    axes[1, 0].plot(history['test_kl'], label='Test', alpha=0.7)
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Div')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # KL weight schedule
    axes[1, 1].plot(history['kl_weight'], alpha=0.7, color='green')
    axes[1, 1].set_title('KL Weight Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved training curves")
    
    # Generate comprehensive evaluation
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    
    model.eval()
    all_real = []
    all_generated = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(CONFIG['device'])
            recon, _, _ = model(batch)
            
            all_real.append(batch.cpu().numpy())
            all_generated.append(recon.cpu().numpy())
    
    real_data = np.vstack(all_real)
    generated_data = np.vstack(all_generated)
    
    # Inverse transform
    real_inv = np.zeros_like(real_data)
    gen_inv = np.zeros_like(generated_data)
    
    for i in range(len(real_data)):
        real_inv[i] = scaler.inverse_transform(real_data[i])
        gen_inv[i] = scaler.inverse_transform(generated_data[i])
    
    # Comprehensive evaluation
    eval_results = evaluate_generation_quality(real_inv, gen_inv, metric_names)
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Generate visualizations
    visualize_reconstructions(model, test_loader, scaler, metric_names, output_dir)
    visualize_latent_space(model, test_loader, output_dir)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nBest test loss: {best_test_loss:.6f}")
    print(f"\nKey metrics:")
    print(f"  Variance ratio: {eval_results['overall']['variance_ratio']:.4f}")
    print(f"  Average DTW: {eval_results['temporal']['avg_dtw']:.4f}")
    print(f"  Average ACF error: {eval_results['temporal']['avg_acf_error']:.6f}")

if __name__ == "__main__":
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    train_model()