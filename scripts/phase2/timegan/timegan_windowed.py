#!/usr/bin/env python3
"""
Sliding Window TimeGAN Implementation
=====================================

This script addresses the flat output problem by:
1. Converting long sequences (715) into many short windows (64)
2. Training TimeGAN on manageable sequence lengths
3. Generating full traces by stitching windows

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Configuration
CONFIG = {
    # Data
    'data_path': '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_traces_normalized.npy',
    'output_dir': '/home/hamid/generative-ai-workload-modeling/outputs/timegan_windowed',
    
    # Sliding window
    'window_size': 64,      # Short enough for GAN to learn
    'stride': 16,           # Overlap for more samples
    
    # Model architecture
    'input_dim': 15,
    'hidden_dim': 64,       # Smaller for shorter sequences
    'latent_dim': 32,
    'num_layers': 2,
    
    # Training
    'batch_size': 32,       # Larger batches possible now
    'epochs_embed': 50,
    'epochs_supervisor': 50,
    'epochs_joint': 100,
    'learning_rate': 0.001,
    'train_split': 0.8,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42
}


def create_sliding_windows(data, window_size, stride):
    """
    Convert full traces into overlapping windows.
    
    Args:
        data: (n_samples, seq_len, n_features) - e.g., (60, 715, 15)
        window_size: length of each window
        stride: step between consecutive windows
    
    Returns:
        windows: (n_windows, window_size, n_features)
        metadata: list of dicts with trace_idx, start, end
    """
    windows = []
    metadata = []
    
    n_samples, seq_len, n_features = data.shape
    
    for trace_idx in range(n_samples):
        trace = data[trace_idx]
        
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window = trace[start:end]
            windows.append(window)
            metadata.append({
                'trace_idx': trace_idx,
                'start': start,
                'end': end
            })
    
    return np.array(windows), metadata


# -----------------------------------------------------------------------------
# TimeGAN Components (simplified for shorter sequences)
# -----------------------------------------------------------------------------

class Embedder(nn.Module):
    """Embeds features into latent space"""
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.sigmoid(self.fc(h))


class Recovery(nn.Module):
    """Recovers features from latent space"""
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return torch.sigmoid(self.fc(out))


class Generator(nn.Module):
    """Generates latent sequences from noise"""
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z):
        h, _ = self.rnn(z)
        return torch.sigmoid(self.fc(h))


class Supervisor(nn.Module):
    """Supervises temporal dynamics"""
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return torch.sigmoid(self.fc(out))


class Discriminator(nn.Module):
    """Discriminates real vs fake in latent space"""
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return self.fc(out)


class WindowedTimeGAN:
    """TimeGAN with sliding window support"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Initialize components
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
        
        # Training history
        self.history = {
            'embed_loss': [],
            'supervisor_loss': [],
            'g_loss': [],
            'd_loss': []
        }
    
    def train_embedder(self, dataloader, epochs):
        """Phase 1: Train autoencoder"""
        print("\n" + "="*60)
        print("PHASE 1: Training Embedder and Recovery")
        print("="*60)
        
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                self.opt_embed.zero_grad()
                
                # Forward pass
                h = self.embedder(x)
                x_recon = self.recovery(h)
                
                # Loss
                loss = criterion(x_recon, x)
                loss.backward()
                self.opt_embed.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['embed_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
        return self.history['embed_loss']
    
    def train_supervisor(self, dataloader, epochs):
        """Phase 2: Train supervisor"""
        print("\n" + "="*60)
        print("PHASE 2: Training Supervisor")
        print("="*60)
        
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                self.opt_supervisor.zero_grad()
                
                # Get embeddings (detached)
                with torch.no_grad():
                    h = self.embedder(x)
                
                # Supervisor predicts next step
                h_supervised = self.supervisor(h)
                
                # Loss: predict h[t+1] from h[t]
                loss = criterion(h_supervised[:, :-1, :], h[:, 1:, :])
                loss.backward()
                self.opt_supervisor.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['supervisor_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
        return self.history['supervisor_loss']
    
    def train_joint(self, dataloader, epochs):
        """Phase 3: Joint adversarial training"""
        print("\n" + "="*60)
        print("PHASE 3: Joint Adversarial Training")
        print("="*60)
        
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_mse = nn.MSELoss()
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch in dataloader:
                x = batch[0].to(self.device)
                batch_size = x.size(0)
                seq_len = x.size(1)
                
                # Generate noise
                z = torch.randn(batch_size, seq_len, self.config['latent_dim']).to(self.device)
                
                # Get real embeddings
                with torch.no_grad():
                    h_real = self.embedder(x)
                
                # Generate fake embeddings
                h_fake_raw = self.generator(z)
                h_fake = self.supervisor(h_fake_raw)
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                self.opt_discriminator.zero_grad()
                
                d_real = self.discriminator(h_real)
                d_fake = self.discriminator(h_fake.detach())
                
                d_loss_real = criterion_bce(d_real, torch.ones_like(d_real))
                d_loss_fake = criterion_bce(d_fake, torch.zeros_like(d_fake))
                d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                self.opt_discriminator.step()
                
                # ---------------------
                # Train Generator
                # ---------------------
                self.opt_generator.zero_grad()
                
                # Regenerate (need fresh computation graph)
                h_fake_raw = self.generator(z)
                h_fake = self.supervisor(h_fake_raw)
                
                # Adversarial loss
                d_fake = self.discriminator(h_fake)
                g_loss_adv = criterion_bce(d_fake, torch.ones_like(d_fake))
                
                # Supervised loss (temporal coherence)
                h_supervised = self.supervisor(h_real)
                g_loss_sup = criterion_mse(h_supervised[:, :-1, :], h_real[:, 1:, :])
                
                # Feature matching loss
                x_fake = self.recovery(h_fake)
                h_fake_embed = self.embedder(x_fake)
                g_loss_fm = criterion_mse(h_fake, h_fake_embed)
                
                # Total generator loss
                g_loss = g_loss_adv + 10 * g_loss_sup + 10 * g_loss_fm
                
                g_loss.backward()
                self.opt_generator.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
        
        return self.history['g_loss'], self.history['d_loss']
    
    def generate_window(self, n_samples=1):
        """Generate a single window"""
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
    
    def generate_full_trace(self, target_length=715, blend_overlap=True):
        """
        Generate full trace by stitching windows.
        
        Uses overlapping windows with linear blending for smooth transitions.
        """
        window_size = self.config['window_size']
        stride = window_size // 2  # 50% overlap for blending
        
        generated = None
        
        while generated is None or len(generated) < target_length:
            # Generate new window
            window = self.generate_window(1)[0]  # (window_size, n_features)
            
            if generated is None:
                generated = window
            elif blend_overlap:
                # Blend overlapping region
                overlap = window_size - stride
                
                # Create blended region
                for i in range(overlap):
                    alpha = i / overlap
                    idx = len(generated) - overlap + i
                    generated[idx] = (1 - alpha) * generated[idx] + alpha * window[i]
                
                # Append non-overlapping part
                generated = np.vstack([generated, window[overlap:]])
            else:
                # Simple concatenation
                generated = np.vstack([generated, window])
        
        return generated[:target_length]


def evaluate_windows(model, test_data, output_dir):
    """Evaluate generation quality on windowed data"""
    
    n_test = len(test_data)
    
    # Generate same number of samples
    generated_windows = []
    for _ in range(n_test):
        window = model.generate_window(1)[0]
        generated_windows.append(window)
    generated_windows = np.array(generated_windows)
    
    # Compute metrics
    results = {}
    
    # Per-metric MSE and correlation
    metrics_names = ['cpu_psi', 'cpu_usage', 'cpu_num', 'cpu_temperature', 
                     'gpu_power', 'gpu_utilization', 'gpu_temperature',
                     'memory_usage', 'memory_psi', 'latency', 'latency_p50',
                     'latency_p95', 'latency_p99', 'success_rate', 'throughput']
    
    results['per_metric'] = {}
    
    for i, name in enumerate(metrics_names):
        real_metric = test_data[:, :, i].flatten()
        gen_metric = generated_windows[:, :, i].flatten()
        
        mse = np.mean((real_metric - gen_metric) ** 2)
        corr = np.corrcoef(real_metric, gen_metric)[0, 1] if np.std(gen_metric) > 0 else 0
        
        results['per_metric'][name] = {
            'mse': float(mse),
            'correlation': float(corr),
            'real_std': float(np.std(real_metric)),
            'gen_std': float(np.std(gen_metric))
        }
    
    # Overall variance comparison
    real_var = np.var(test_data)
    gen_var = np.var(generated_windows)
    
    results['overall'] = {
        'real_variance': float(real_var),
        'generated_variance': float(gen_var),
        'variance_ratio': float(gen_var / real_var) if real_var > 0 else 0
    }
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, generated_windows


def plot_comparison(test_data, generated_windows, output_dir):
    """Plot comparison between real and generated windows"""
    
    metrics_names = ['cpu_psi', 'cpu_usage', 'gpu_power', 
                     'gpu_utilization', 'gpu_temperature', 'memory_usage']
    metrics_idx = [0, 1, 4, 5, 6, 7]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for ax, name, idx in zip(axes, metrics_names, metrics_idx):
        # Plot a few real samples
        for i in range(min(3, len(test_data))):
            ax.plot(test_data[i, :, idx], 'b-', alpha=0.3, label='Real' if i == 0 else '')
        
        # Plot a few generated samples
        for i in range(min(3, len(generated_windows))):
            ax.plot(generated_windows[i, :, idx], 'r--', alpha=0.3, label='Generated' if i == 0 else '')
        
        ax.set_title(name)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'window_comparison.png', dpi=150)
    plt.close()


def plot_full_trace_comparison(model, original_trace, output_dir):
    """Generate and compare full trace"""
    
    generated_trace = model.generate_full_trace(len(original_trace))
    
    metrics_names = ['cpu_psi', 'cpu_usage', 'gpu_power', 
                     'gpu_utilization', 'gpu_temperature', 'memory_usage']
    metrics_idx = [0, 1, 4, 5, 6, 7]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for ax, name, idx in zip(axes, metrics_names, metrics_idx):
        ax.plot(original_trace[:, idx], 'b-', label='Real', alpha=0.7)
        ax.plot(generated_trace[:, idx], 'r--', label='Generated', alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.suptitle('Full Trace Comparison (Generated via Window Stitching)')
    plt.tight_layout()
    plt.savefig(output_dir / 'full_trace_comparison.png', dpi=150)
    plt.close()
    
    return generated_trace


def main():
    print("="*70)
    print("WINDOWED TIMEGAN - Addressing Flat Output Problem")
    print("="*70)
    
    # Set random seeds
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1] Loading data...")
    data = np.load(CONFIG['data_path'])
    print(f"    Original data shape: {data.shape}")
    
    # Create sliding windows
    print("\n[2] Creating sliding windows...")
    windows, metadata = create_sliding_windows(
        data, 
        CONFIG['window_size'], 
        CONFIG['stride']
    )
    print(f"    Window size: {CONFIG['window_size']}")
    print(f"    Stride: {CONFIG['stride']}")
    print(f"    Number of windows: {len(windows)}")
    print(f"    Windows shape: {windows.shape}")
    
    # Train/test split
    n_train = int(len(windows) * CONFIG['train_split'])
    indices = np.random.permutation(len(windows))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_data = windows[train_idx]
    test_data = windows[test_idx]
    
    print(f"\n    Train windows: {len(train_data)}")
    print(f"    Test windows: {len(test_data)}")
    
    # Create dataloader
    train_tensor = torch.FloatTensor(train_data)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True
    )
    
    # Initialize model
    print("\n[3] Initializing Windowed TimeGAN...")
    model = WindowedTimeGAN(CONFIG)
    
    # Training
    print("\n[4] Training...")
    
    # Phase 1: Embedder
    model.train_embedder(train_loader, CONFIG['epochs_embed'])
    
    # Phase 2: Supervisor
    model.train_supervisor(train_loader, CONFIG['epochs_supervisor'])
    
    # Phase 3: Joint
    model.train_joint(train_loader, CONFIG['epochs_joint'])
    
    # Evaluation
    print("\n[5] Evaluating...")
    results, generated_windows = evaluate_windows(model, test_data, output_dir)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\nPer-Metric Correlations:")
    for name, metrics in results['per_metric'].items():
        print(f"  {name:20s} | Corr: {metrics['correlation']:7.4f} | "
              f"Real std: {metrics['real_std']:.4f} | Gen std: {metrics['gen_std']:.4f}")
    
    print(f"\nOverall Variance Ratio: {results['overall']['variance_ratio']:.4f}")
    print(f"  Real variance:      {results['overall']['real_variance']:.6f}")
    print(f"  Generated variance: {results['overall']['generated_variance']:.6f}")
    
    # Plot comparisons
    print("\n[6] Creating visualizations...")
    plot_comparison(test_data, generated_windows, output_dir)
    
    # Generate and compare full trace
    original_trace = data[0]  # First full trace
    generated_full = plot_full_trace_comparison(model, original_trace, output_dir)
    
    # Save training curves
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].plot(model.history['embed_loss'])
    axes[0].set_title('Embedding Loss')
    axes[0].set_xlabel('Epoch')
    
    axes[1].plot(model.history['supervisor_loss'])
    axes[1].set_title('Supervisor Loss')
    axes[1].set_xlabel('Epoch')
    
    axes[2].plot(model.history['g_loss'], label='Generator')
    axes[2].set_title('Generator Loss')
    axes[2].set_xlabel('Epoch')
    
    axes[3].plot(model.history['d_loss'], label='Discriminator')
    axes[3].set_title('Discriminator Loss')
    axes[3].set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    print(f"\n[7] Results saved to: {output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()