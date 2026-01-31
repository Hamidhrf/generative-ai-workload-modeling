#!/usr/bin/env python3
"""
TimeGAN for AI Workload Trace Generation
Based on SeriesGAN improvements for stability

Key features:
- Adversarial training (no posterior collapse)
- Explicit temporal supervision
- Early stopping mechanism
- Preserves sharp transitions

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_dir': Path.home() / 'generative-ai-workload-modeling/data/processed/phase1',
    'output_dir': Path.home() / 'generative-ai-workload-modeling/outputs/timegan',
    'model_name': 'timegan',
    
    # Model architecture
    'input_dim': 15,
    'hidden_dim': 128,
    'num_layers': 3,
    
    # Training hyperparameters
    'batch_size': 8,
    'epochs_embed': 100,      # Phase 1: Embedding training
    'epochs_supervisor': 100,  # Phase 2: Supervisor training
    'epochs_joint': 200,      # Phase 3: Joint training
    'learning_rate': 0.001,
    'gamma': 1.0,             # Adversarial loss weight
    
    # Data
    'seq_len': 715,
    'train_split': 0.8,
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Absolute ranges for denormalization (from data_handler.py)
ABSOLUTE_RANGES = {
    0: ('cpu_psi', 0.0, 1.0),
    1: ('cpu_usage', 0.0, 8.0),
    2: ('cpu_num', 0.0, 16.0),
    3: ('cpu_temperature', 0.0, 100.0),
    4: ('gpu_power', 0.0, 100.0),
    5: ('gpu_utilization', 0.0, 100.0),
    6: ('gpu_temperature', 0.0, 100.0),
    7: ('memory_usage', 0.0, 10e9),
    8: ('memory_psi', 0.0, 1.0),
    9: ('latency', 0.0, 5.0),
    10: ('latency_p50', 0.0, 5.0),
    11: ('latency_p95', 0.0, 8.0),
    12: ('latency_p99', 0.0, 10.0),
    13: ('success_rate', 0.0, 1.0),
    14: ('throughput', 0.0, 500.0)
}

METRIC_NAMES = [ABSOLUTE_RANGES[i][0] for i in range(15)]


class WorkloadDataset(Dataset):
    """Dataset for pre-normalized workload traces"""
    
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class EmbeddingNetwork(nn.Module):
    """Maps input sequences to latent representation"""
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        h, _ = self.rnn(x)
        h = torch.sigmoid(self.linear(h))
        return h


class RecoveryNetwork(nn.Module):
    """Maps latent representation back to original space"""
    
    def __init__(self, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h):
        x, _ = self.rnn(h)
        x = self.linear(x)
        return x


class Generator(nn.Module):
    """Generates latent sequences from random noise"""
    
    def __init__(self, noise_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, z):
        h, _ = self.rnn(z)
        h = torch.sigmoid(self.linear(h))
        return h


class Discriminator(nn.Module):
    """Distinguishes real from fake in latent space"""
    
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        h_out, _ = self.rnn(h)
        y = self.linear(h_out)
        return y


class Supervisor(nn.Module):
    """Provides one-step-ahead supervision in latent space"""
    
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers-1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h):
        h_out, _ = self.rnn(h)
        s = torch.sigmoid(self.linear(h_out))
        return s


class TimeGAN:
    """TimeGAN model with SeriesGAN improvements"""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        # Initialize networks
        self.embedder = EmbeddingNetwork(
            config['input_dim'], 
            config['hidden_dim'], 
            config['num_layers']
        ).to(self.device)
        
        self.recovery = RecoveryNetwork(
            config['hidden_dim'], 
            config['input_dim'], 
            config['num_layers']
        ).to(self.device)
        
        self.generator = Generator(
            config['input_dim'],  # Use same as input_dim for noise
            config['hidden_dim'], 
            config['num_layers']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            config['hidden_dim'], 
            config['num_layers']
        ).to(self.device)
        
        self.supervisor = Supervisor(
            config['hidden_dim'], 
            config['num_layers']
        ).to(self.device)
        
        # Optimizers (discriminator learns slower to prevent collapse)
        self.opt_embedder = optim.Adam(self.embedder.parameters(), lr=config['learning_rate'])
        self.opt_recovery = optim.Adam(self.recovery.parameters(), lr=config['learning_rate'])
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=config['learning_rate'])
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=config['learning_rate'] * 0.25)  # 4x slower
        self.opt_supervisor = optim.Adam(self.supervisor.parameters(), lr=config['learning_rate'])
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def train_embedder(self, train_loader, epochs):
        """Phase 1: Train embedder and recovery networks"""
        print("=" * 80)
        print("PHASE 1: TRAINING EMBEDDER AND RECOVERY")
        print("=" * 80)
        
        history = []
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            self.embedder.train()
            self.recovery.train()
            epoch_loss = 0
            
            for batch in train_loader:
                x = batch.to(self.device)
                
                # Forward pass
                h = self.embedder(x)
                x_recon = self.recovery(h)
                
                # Reconstruction loss
                loss = self.mse_loss(x_recon, x)
                
                # Backward pass
                self.opt_embedder.zero_grad()
                self.opt_recovery.zero_grad()
                loss.backward()
                self.opt_embedder.step()
                self.opt_recovery.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history.append({'epoch': epoch, 'loss': avg_loss})
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return pd.DataFrame(history)
    
    def train_supervisor(self, train_loader, epochs):
        """Phase 2: Train supervisor for temporal coherence"""
        print("\n" + "=" * 80)
        print("PHASE 2: TRAINING SUPERVISOR")
        print("=" * 80)
        
        history = []
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            self.supervisor.train()
            epoch_loss = 0
            
            for batch in train_loader:
                x = batch.to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    h = self.embedder(x)
                
                # Supervisor predicts next step
                h_supervised = self.supervisor(h)
                
                # Loss: predict next time step
                loss = self.mse_loss(h_supervised[:, :-1, :], h[:, 1:, :])
                
                # Backward pass
                self.opt_supervisor.zero_grad()
                loss.backward()
                self.opt_supervisor.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history.append({'epoch': epoch, 'loss': avg_loss})
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return pd.DataFrame(history)
    
    def train_joint(self, train_loader, epochs):
        """Phase 3: Joint adversarial training"""
        print("\n" + "=" * 80)
        print("PHASE 3: JOINT ADVERSARIAL TRAINING")
        print("=" * 80)
        
        history = []
        best_g_loss = float('inf')
        patience = 30
        patience_counter = 0
        
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            self.supervisor.train()
            
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                x = batch.to(self.device)
                batch_size = x.size(0)
                seq_len = x.size(1)
                
                # Generate random noise
                z = torch.randn(batch_size, seq_len, self.config['input_dim']).to(self.device)
                
                # Real embeddings
                h_real = self.embedder(x)
                
                # Fake embeddings
                h_fake = self.generator(z)
                h_fake_supervised = self.supervisor(h_fake)
                
                # --- Train Discriminator (every 2 steps to prevent collapse) ---
                if batch_idx % 2 == 0:
                    y_real = self.discriminator(h_real)
                    y_fake = self.discriminator(h_fake.detach())
                    
                    d_loss_real = self.bce_loss(y_real, torch.ones_like(y_real))
                    d_loss_fake = self.bce_loss(y_fake, torch.zeros_like(y_fake))
                    d_loss = d_loss_real + d_loss_fake
                    
                    self.opt_discriminator.zero_grad()
                    d_loss.backward()
                    self.opt_discriminator.step()
                else:
                    # Skip discriminator update, just compute for logging
                    with torch.no_grad():
                        y_real = self.discriminator(h_real)
                        y_fake = self.discriminator(h_fake.detach())
                        d_loss_real = self.bce_loss(y_real, torch.ones_like(y_real))
                        d_loss_fake = self.bce_loss(y_fake, torch.zeros_like(y_fake))
                        d_loss = d_loss_real + d_loss_fake
                
                # --- Train Generator ---
                # Recompute h_real for generator loss (avoid graph reuse)
                h_real_g = self.embedder(x)
                
                y_fake_g = self.discriminator(h_fake)
                g_loss_adv = self.bce_loss(y_fake_g, torch.ones_like(y_fake_g))
                
                # Supervised loss
                g_loss_sup = self.mse_loss(h_fake_supervised[:, :-1, :], h_fake[:, 1:, :])
                
                # Moment matching loss (use recomputed h_real_g)
                g_loss_moment = torch.mean(torch.abs(torch.mean(h_fake, dim=0) - torch.mean(h_real_g, dim=0)))
                
                # Total generator loss
                g_loss = g_loss_adv + 100 * torch.sqrt(g_loss_sup) + 100 * g_loss_moment
                
                self.opt_generator.zero_grad()
                self.opt_supervisor.zero_grad()
                g_loss.backward()
                self.opt_generator.step()
                self.opt_supervisor.step()
                
                # Embedding network update
                h_fake_new = self.generator(z)
                x_recon = self.recovery(h_fake_new)
                e_loss = self.mse_loss(x_recon, x)
                
                self.opt_embedder.zero_grad()
                self.opt_recovery.zero_grad()
                e_loss.backward()
                self.opt_embedder.step()
                self.opt_recovery.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
            
            avg_g_loss = epoch_g_loss / len(train_loader)
            avg_d_loss = epoch_d_loss / len(train_loader)
            
            history.append({
                'epoch': epoch, 
                'g_loss': avg_g_loss, 
                'd_loss': avg_d_loss
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
            
            # Early stopping on generator loss
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                patience_counter = 0
                # Save best model
                self.save_model(self.config['output_dir'] / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return pd.DataFrame(history)
    
    def generate(self, n_samples, seq_len):
        """Generate synthetic traces"""
        self.generator.eval()
        self.recovery.eval()
        self.supervisor.eval()
        
        with torch.no_grad():
            z = torch.randn(n_samples, seq_len, self.config['input_dim']).to(self.device)
            h = self.generator(z)
            h = self.supervisor(h)
            x_generated = self.recovery(h)
        
        return x_generated.cpu().numpy()
    
    def save_model(self, path):
        """Save all model components"""
        torch.save({
            'embedder': self.embedder.state_dict(),
            'recovery': self.recovery.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'supervisor': self.supervisor.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """Load all model components"""
        checkpoint = torch.load(path, weights_only=False)
        self.embedder.load_state_dict(checkpoint['embedder'])
        self.recovery.load_state_dict(checkpoint['recovery'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.supervisor.load_state_dict(checkpoint['supervisor'])


def compute_dtw(seq1, seq2):
    """Compute Dynamic Time Warping distance"""
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # For 1D sequences, compute absolute difference
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                          dtw_matrix[i, j-1],
                                          dtw_matrix[i-1, j-1])
    
    return dtw_matrix[n, m]


def compute_acf(data, max_lag=50):
    """Compute autocorrelation function"""
    mean = np.mean(data)
    var = np.var(data)
    normalized = data - mean
    
    acf = np.correlate(normalized, normalized, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf[:max_lag] / (var * len(data))
    
    return acf


def evaluate_timegan(model, test_data, output_dir):
    """Comprehensive evaluation of TimeGAN"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Generate synthetic data
    n_samples = len(test_data)
    seq_len = test_data.shape[1]
    generated_data = model.generate(n_samples, seq_len)
    
    results = {
        'per_metric': {},
        'temporal': {},
        'overall': {}
    }
    
    # Per-metric evaluation
    print("\n1. PER-METRIC RECONSTRUCTION ERRORS:")
    print("-" * 80)
    
    for i, metric_name in enumerate(METRIC_NAMES):
        real_metric = test_data[:, :, i].flatten()
        gen_metric = generated_data[:, :, i].flatten()
        
        mse = mean_squared_error(real_metric, gen_metric)
        mae = mean_absolute_error(real_metric, gen_metric)
        
        if np.std(real_metric) > 1e-6 and np.std(gen_metric) > 1e-6:
            corr, _ = pearsonr(real_metric, gen_metric)
        else:
            corr = 0.0
        
        results['per_metric'][metric_name] = {
            'mse': float(mse),
            'mae': float(mae),
            'correlation': float(corr)
        }
        
        print(f"{metric_name:25s} | MSE: {mse:10.6f} | MAE: {mae:10.6f} | Corr: {corr:6.3f}")
    
    # DTW analysis (on first 4 metrics for speed)
    print("\n2. DYNAMIC TIME WARPING (DTW) ANALYSIS:")
    print("-" * 80)
    
    key_metrics = ['cpu_psi', 'cpu_usage', 'gpu_power', 'gpu_utilization']
    dtw_distances = []
    
    for metric_name in key_metrics:
        idx = METRIC_NAMES.index(metric_name)
        dtw_sum = 0
        for i in range(min(5, n_samples)):
            dtw_dist = compute_dtw(test_data[i, :, idx], generated_data[i, :, idx])
            dtw_sum += dtw_dist
        avg_dtw = dtw_sum / min(5, n_samples)
        dtw_distances.append(avg_dtw)
        print(f"{metric_name:25s} | Avg DTW: {avg_dtw:10.4f}")
    
    results['temporal']['avg_dtw'] = float(np.mean(dtw_distances))
    
    # ACF analysis
    print("\n3. AUTOCORRELATION SIMILARITY:")
    print("-" * 80)
    
    acf_errors = []
    for metric_name in key_metrics:
        idx = METRIC_NAMES.index(metric_name)
        real_flat = test_data[:, :, idx].flatten()
        gen_flat = generated_data[:, :, idx].flatten()
        
        acf_real = compute_acf(real_flat)
        acf_gen = compute_acf(gen_flat)
        acf_error = np.mean(np.abs(acf_real - acf_gen))
        acf_errors.append(acf_error)
        
        print(f"{metric_name:25s} | ACF Error: {acf_error:10.6f}")
    
    results['temporal']['avg_acf_error'] = float(np.mean(acf_errors))
    
    # Spectral analysis
    print("\n4. SPECTRAL ANALYSIS:")
    print("-" * 80)
    
    spectral_errors = []
    for metric_name in key_metrics:
        idx = METRIC_NAMES.index(metric_name)
        real_flat = test_data[:, :, idx].flatten()
        gen_flat = generated_data[:, :, idx].flatten()
        
        fft_real = np.abs(np.fft.fft(real_flat))
        fft_gen = np.abs(np.fft.fft(gen_flat))
        spectral_error = np.mean(np.abs(fft_real - fft_gen))
        spectral_errors.append(spectral_error)
        
        print(f"{metric_name:25s} | Spectral Error: {spectral_error:10.4f}")
    
    results['temporal']['avg_spectral_error'] = float(np.mean(spectral_errors))
    
    # Overall statistics
    print("\n5. OVERALL STATISTICS:")
    print("-" * 80)
    
    real_var = np.var(test_data)
    gen_var = np.var(generated_data)
    var_ratio = gen_var / real_var
    
    real_range = np.max(test_data) - np.min(test_data)
    gen_range = np.max(generated_data) - np.min(generated_data)
    range_ratio = gen_range / real_range
    
    results['overall'] = {
        'variance_ratio': float(var_ratio),
        'range_ratio': float(range_ratio),
        'real_variance': float(real_var),
        'generated_variance': float(gen_var)
    }
    
    print(f"Real data variance:          {real_var:.6f}")
    print(f"Generated data variance:     {gen_var:.6f}")
    print(f"Variance ratio (gen/real):   {var_ratio:.6f}")
    print(f"Real data range:             {real_range:.6f}")
    print(f"Generated data range:        {gen_range:.6f}")
    print(f"Range ratio (gen/real):      {range_ratio:.6f}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("-" * 80)
    
    if var_ratio > 0.8:
        print("EXCELLENT: Variance is well preserved.")
    elif var_ratio > 0.6:
        print("GOOD: Variance is reasonably preserved.")
    else:
        print("WARNING: Generated data has lower variance than real data.")
    
    if results['temporal']['avg_dtw'] < 100:
        print("GOOD: Temporal alignment is reasonable.")
    else:
        print("WARNING: High DTW distances suggest poor temporal alignment.")
    
    if results['temporal']['avg_acf_error'] < 0.1:
        print("GOOD: Temporal dynamics preserved.")
    else:
        print("WARNING: Autocorrelation patterns differ significantly.")
    
    print("=" * 80)
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize reconstructions
    visualize_reconstructions(test_data, generated_data, output_dir)
    
    return results


def visualize_reconstructions(real_data, generated_data, output_dir):
    """Visualize sample reconstructions"""
    n_samples = min(3, len(real_data))
    key_metrics = ['cpu_psi', 'cpu_usage', 'gpu_power', 'gpu_utilization', 
                   'gpu_temperature', 'memory_usage']
    
    for sample_idx in range(n_samples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Sample {sample_idx+1} - Reconstruction Comparison', 
                    fontsize=14, fontweight='bold')
        
        for idx, metric_name in enumerate(key_metrics):
            ax = axes[idx // 3, idx % 3]
            metric_idx = METRIC_NAMES.index(metric_name)
            
            ax.plot(real_data[sample_idx, :, metric_idx], 
                   label='Real', color='blue', linewidth=1.5, alpha=0.7)
            ax.plot(generated_data[sample_idx, :, metric_idx], 
                   label='Generated', color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.set_title(metric_name, fontsize=10, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'reconstruction_sample_{sample_idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstruction visualization for sample {sample_idx+1}")


def train_timegan():
    """Main training function"""
    print("=" * 80)
    print("TIMEGAN TRAINING - IMPROVED ARCHITECTURE")
    print("=" * 80)
    print()
    
    # Print configuration
    print("Configuration:")
    for key, value in CONFIG.items():
        if key not in ['data_dir', 'output_dir']:
            print(f"  {key}: {value}")
    print()
    
    # Create output directory
    output_dir = CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    # Load data
    print(f"Loading data from {CONFIG['data_dir']}")
    data_path = CONFIG['data_dir'] / 'pod_traces_normalized.npy'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading normalized traces from {data_path}")
    data = np.load(data_path)
    print(f"Using pre-normalized data")
    
    print(f"\nLoaded {len(data)} pod traces")
    print(f"Data shape: {data.shape}")
    print(f"  Number of samples: {data.shape[0]}")
    print(f"  Sequence length: {data.shape[1]}")
    print(f"  Number of features: {data.shape[2]}")
    print()
    
    # Verify data range
    print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
    print("Data already normalized, skipping normalization...\n")
    
    # Train/test split
    n_train = int(len(data) * CONFIG['train_split'])
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}\n")
    
    # Create data loaders
    train_dataset = WorkloadDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Initialize TimeGAN
    print("Initializing TimeGAN...")
    model = TimeGAN(CONFIG)
    
    # Count parameters
    total_params = sum(p.numel() for net in [model.embedder, model.recovery, model.generator, 
                                             model.discriminator, model.supervisor] 
                      for p in net.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Phase 1: Train embedder
    history_embed = model.train_embedder(train_loader, CONFIG['epochs_embed'])
    
    # Phase 2: Train supervisor
    history_supervisor = model.train_supervisor(train_loader, CONFIG['epochs_supervisor'])
    
    # Phase 3: Joint training
    history_joint = model.train_joint(train_loader, CONFIG['epochs_joint'])
    
    # Save training history
    history_embed.to_csv(output_dir / 'history_embed.csv', index=False)
    history_supervisor.to_csv(output_dir / 'history_supervisor.csv', index=False)
    history_joint.to_csv(output_dir / 'history_joint.csv', index=False)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history_embed['loss'])
    axes[0].set_title('Phase 1: Embedding Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    axes[1].plot(history_supervisor['loss'])
    axes[1].set_title('Phase 2: Supervisor Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    axes[2].plot(history_joint['g_loss'], label='Generator')
    axes[2].plot(history_joint['d_loss'], label='Discriminator')
    axes[2].set_title('Phase 3: Joint Training')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved training curves")
    
    # Load best model
    print("\nLoading best model for evaluation...")
    model.load_model(output_dir / 'best_model.pt')
    
    # Evaluate
    results = evaluate_timegan(model, test_data, output_dir)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey metric: Variance ratio = {results['overall']['variance_ratio']:.4f}")
    print("Target: > 0.8 for success")
    print()


if __name__ == '__main__':
    train_timegan()