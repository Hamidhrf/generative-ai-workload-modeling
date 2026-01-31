#!/usr/bin/env python3
"""
Conditional TimeGAN with Replica Count Conditioning
====================================================

This addresses the core issue: the model needs to know the replica count
to learn the relationship between contention level and resource patterns.

Key insight from PROJECT_DECISIONS.md:
  "Condition on replica count (r) during training"
  "Model learns: How does a pod behave when there are r total pods?"

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
    'output_dir': '/home/hamid/generative-ai-workload-modeling/outputs/conditional_timegan',
    
    # Use fewer metrics initially to prove concept
    'use_reduced_metrics': True,
    'core_metrics': ['cpu_usage', 'gpu_utilization', 'memory_usage', 'latency'],
    'core_metric_indices': [1, 5, 7, 9],  # Indices in the 15-metric array
    
    # Sliding window
    'window_size': 64,
    'stride': 16,
    
    # Model architecture
    'hidden_dim': 64,
    'latent_dim': 32,
    'num_layers': 2,
    'condition_dim': 8,  # Embedding dimension for replica count
    
    # Training
    'batch_size': 32,
    'epochs_embed': 100,
    'epochs_supervisor': 100,
    'epochs_joint': 200,
    'learning_rate': 0.001,
    
    # Replica count normalization (for conditioning)
    'max_replica_count': 20,  # Normalize replica counts to [0, 1]
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42
}

ALL_METRICS = ['cpu_psi', 'cpu_usage', 'cpu_num', 'cpu_temperature', 
               'gpu_power', 'gpu_utilization', 'gpu_temperature',
               'memory_usage', 'memory_psi', 'latency', 'latency_p50',
               'latency_p95', 'latency_p99', 'success_rate', 'throughput']


class ConditionEmbedding(nn.Module):
    """Embed replica count into a learned representation"""
    def __init__(self, condition_dim):
        super().__init__()
        # Simple MLP to embed normalized replica count
        self.net = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.Tanh()
        )
    
    def forward(self, replica_count):
        # replica_count: (batch,) normalized to [0, 1]
        x = replica_count.unsqueeze(-1)  # (batch, 1)
        return self.net(x)  # (batch, condition_dim)


class ConditionalEmbedder(nn.Module):
    """Embedder that takes condition into account"""
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim, num_layers):
        super().__init__()
        self.condition_embed = ConditionEmbedding(condition_dim)
        # Input: features + condition at each timestep
        self.rnn = nn.GRU(input_dim + condition_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, condition):
        # x: (batch, seq_len, input_dim)
        # condition: (batch,) normalized replica count
        batch_size, seq_len, _ = x.shape
        
        # Embed condition and repeat for each timestep
        cond_emb = self.condition_embed(condition)  # (batch, condition_dim)
        cond_emb = cond_emb.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, condition_dim)
        
        # Concatenate input with condition
        x_cond = torch.cat([x, cond_emb], dim=-1)  # (batch, seq_len, input_dim + condition_dim)
        
        h, _ = self.rnn(x_cond)
        return torch.sigmoid(self.fc(h))


class ConditionalRecovery(nn.Module):
    """Recovery network that uses condition"""
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim, num_layers):
        super().__init__()
        self.condition_embed = ConditionEmbedding(condition_dim)
        self.rnn = nn.GRU(latent_dim + condition_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h, condition):
        batch_size, seq_len, _ = h.shape
        
        cond_emb = self.condition_embed(condition)
        cond_emb = cond_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        h_cond = torch.cat([h, cond_emb], dim=-1)
        out, _ = self.rnn(h_cond)
        return torch.sigmoid(self.fc(out))


class ConditionalGenerator(nn.Module):
    """Generator that is conditioned on replica count"""
    def __init__(self, latent_dim, hidden_dim, condition_dim, num_layers):
        super().__init__()
        self.condition_embed = ConditionEmbedding(condition_dim)
        self.rnn = nn.GRU(latent_dim + condition_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z, condition):
        batch_size, seq_len, _ = z.shape
        
        cond_emb = self.condition_embed(condition)
        cond_emb = cond_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        z_cond = torch.cat([z, cond_emb], dim=-1)
        h, _ = self.rnn(z_cond)
        return torch.sigmoid(self.fc(h))


class ConditionalSupervisor(nn.Module):
    """Supervisor with condition"""
    def __init__(self, latent_dim, hidden_dim, condition_dim, num_layers):
        super().__init__()
        self.condition_embed = ConditionEmbedding(condition_dim)
        self.rnn = nn.GRU(latent_dim + condition_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, h, condition):
        batch_size, seq_len, _ = h.shape
        
        cond_emb = self.condition_embed(condition)
        cond_emb = cond_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        h_cond = torch.cat([h, cond_emb], dim=-1)
        out, _ = self.rnn(h_cond)
        return torch.sigmoid(self.fc(out))


class ConditionalDiscriminator(nn.Module):
    """Discriminator that knows the condition"""
    def __init__(self, latent_dim, hidden_dim, condition_dim, num_layers):
        super().__init__()
        self.condition_embed = ConditionEmbedding(condition_dim)
        self.rnn = nn.GRU(latent_dim + condition_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, h, condition):
        batch_size, seq_len, _ = h.shape
        
        cond_emb = self.condition_embed(condition)
        cond_emb = cond_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        h_cond = torch.cat([h, cond_emb], dim=-1)
        out, _ = self.rnn(h_cond)
        return self.fc(out)


class ConditionalTimeGAN:
    """
    TimeGAN with replica count conditioning.
    
    This model learns: P(trace | replica_count)
    Can generate: trace for any replica_count (including extrapolation)
    """
    
    def __init__(self, config, input_dim):
        self.config = config
        self.input_dim = input_dim
        self.device = torch.device(config['device'])
        
        # Initialize networks with conditioning
        self.embedder = ConditionalEmbedder(
            input_dim, config['hidden_dim'], config['latent_dim'],
            config['condition_dim'], config['num_layers']
        ).to(self.device)
        
        self.recovery = ConditionalRecovery(
            config['latent_dim'], config['hidden_dim'], input_dim,
            config['condition_dim'], config['num_layers']
        ).to(self.device)
        
        self.generator = ConditionalGenerator(
            config['latent_dim'], config['hidden_dim'],
            config['condition_dim'], config['num_layers']
        ).to(self.device)
        
        self.supervisor = ConditionalSupervisor(
            config['latent_dim'], config['hidden_dim'],
            config['condition_dim'], config['num_layers']
        ).to(self.device)
        
        self.discriminator = ConditionalDiscriminator(
            config['latent_dim'], config['hidden_dim'],
            config['condition_dim'], config['num_layers']
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
        """Phase 1: Train conditional autoencoder"""
        print("\nPhase 1: Training Conditional Embedder")
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x, condition = batch[0].to(self.device), batch[1].to(self.device)
                
                self.opt_embed.zero_grad()
                h = self.embedder(x, condition)
                x_recon = self.recovery(h, condition)
                loss = criterion(x_recon, x)
                loss.backward()
                self.opt_embed.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['embed'].append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
    
    def train_supervisor(self, dataloader, epochs):
        """Phase 2: Train conditional supervisor"""
        print("\nPhase 2: Training Conditional Supervisor")
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x, condition = batch[0].to(self.device), batch[1].to(self.device)
                
                self.opt_supervisor.zero_grad()
                with torch.no_grad():
                    h = self.embedder(x, condition)
                
                h_supervised = self.supervisor(h, condition)
                loss = criterion(h_supervised[:, :-1, :], h[:, 1:, :])
                loss.backward()
                self.opt_supervisor.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['supervisor'].append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
    
    def train_joint(self, dataloader, epochs):
        """Phase 3: Joint adversarial training with condition"""
        print("\nPhase 3: Joint Adversarial Training")
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_mse = nn.MSELoss()
        
        for epoch in range(epochs):
            g_losses, d_losses = [], []
            
            for batch in dataloader:
                x, condition = batch[0].to(self.device), batch[1].to(self.device)
                batch_size = x.size(0)
                seq_len = x.size(1)
                
                # Random noise
                z = torch.randn(batch_size, seq_len, self.config['latent_dim']).to(self.device)
                
                # Get real embeddings
                with torch.no_grad():
                    h_real = self.embedder(x, condition)
                
                # Generate fake embeddings (conditioned on same replica counts)
                h_fake_raw = self.generator(z, condition)
                h_fake = self.supervisor(h_fake_raw, condition)
                
                # Train Discriminator
                self.opt_discriminator.zero_grad()
                d_real = self.discriminator(h_real, condition)
                d_fake = self.discriminator(h_fake.detach(), condition)
                
                # Label smoothing for stability
                real_labels = torch.ones_like(d_real) * 0.9
                fake_labels = torch.zeros_like(d_fake) + 0.1
                
                d_loss = criterion_bce(d_real, real_labels) + criterion_bce(d_fake, fake_labels)
                d_loss.backward()
                self.opt_discriminator.step()
                
                # Train Generator
                self.opt_generator.zero_grad()
                h_fake_raw = self.generator(z, condition)
                h_fake = self.supervisor(h_fake_raw, condition)
                d_fake = self.discriminator(h_fake, condition)
                
                # Adversarial loss
                g_loss_adv = criterion_bce(d_fake, torch.ones_like(d_fake))
                
                # Supervised loss
                h_supervised = self.supervisor(h_real, condition)
                g_loss_sup = criterion_mse(h_supervised[:, :-1, :], h_real[:, 1:, :])
                
                # Feature matching loss
                x_fake = self.recovery(h_fake, condition)
                h_fake_embed = self.embedder(x_fake, condition)
                g_loss_fm = criterion_mse(h_fake, h_fake_embed)
                
                g_loss = g_loss_adv + 10 * g_loss_sup + 10 * g_loss_fm
                g_loss.backward()
                self.opt_generator.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            self.history['g_loss'].append(np.mean(g_losses))
            self.history['d_loss'].append(np.mean(d_losses))
            
            if (epoch + 1) % 40 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | G: {np.mean(g_losses):.4f} | D: {np.mean(d_losses):.4f}")
    
    def generate(self, replica_counts, n_per_count=1):
        """
        Generate synthetic traces for specified replica counts.
        
        Args:
            replica_counts: list of replica counts to generate for
            n_per_count: number of traces per replica count
        
        Returns:
            dict: {replica_count: array of traces}
        """
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        
        results = {}
        
        with torch.no_grad():
            for r in replica_counts:
                # Normalize replica count
                r_norm = min(r / self.config['max_replica_count'], 1.0)
                condition = torch.full((n_per_count,), r_norm).to(self.device)
                
                # Generate
                z = torch.randn(n_per_count, self.config['window_size'], 
                               self.config['latent_dim']).to(self.device)
                h = self.generator(z, condition)
                h = self.supervisor(h, condition)
                x = self.recovery(h, condition)
                
                results[r] = x.cpu().numpy()
        
        return results


def load_data_with_metadata(config):
    """Load data and extract replica counts from metadata"""
    data = np.load(config['data_path'])
    
    # Try to load metadata
    try:
        with open(config['metadata_path'], 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("WARNING: Metadata not found. Creating synthetic metadata.")
        # Create metadata based on expected structure
        # ResNet50: r=1(1) + r=2(2) + r=3(3) + r=6(6) + r=10(10) = 22
        # DistilBERT: r=1(1) + r=2(2) + r=6(6) + r=10(10) = 19
        # Whisper: r=1(1) + r=2(2) + r=3(3) + r=5(5) + r=8(8) = 19
        metadata = []
        
        # ResNet50
        for r, count in [(1,1), (2,2), (3,3), (6,6), (10,10)]:
            for _ in range(count):
                metadata.append({'workload': 'resnet50', 'replica_count': r})
        
        # DistilBERT
        for r, count in [(1,1), (2,2), (6,6), (10,10)]:
            for _ in range(count):
                metadata.append({'workload': 'distilbert', 'replica_count': r})
        
        # Whisper
        for r, count in [(1,1), (2,2), (3,3), (5,5), (8,8)]:
            for _ in range(count):
                metadata.append({'workload': 'whisper', 'replica_count': r})
    
    return data, metadata


def create_windows_with_conditions(data, metadata, config):
    """Create sliding windows with associated replica counts"""
    windows = []
    conditions = []
    window_metadata = []
    
    for i, (trace, meta) in enumerate(zip(data, metadata)):
        r = meta.get('replica_count', 1)
        workload = meta.get('workload', 'unknown')
        
        seq_len = trace.shape[0]
        for start in range(0, seq_len - config['window_size'] + 1, config['stride']):
            window = trace[start:start + config['window_size']]
            
            # Select only core metrics if configured
            if config['use_reduced_metrics']:
                window = window[:, config['core_metric_indices']]
            
            windows.append(window)
            conditions.append(r / config['max_replica_count'])  # Normalize
            window_metadata.append({'workload': workload, 'replica_count': r})
    
    return np.array(windows), np.array(conditions), window_metadata


def evaluate_by_replica_count(model, test_windows, test_conditions, test_metadata, config):
    """Evaluate generation quality per replica count"""
    # Group test data by replica count
    replica_counts = sorted(set(int(c * config['max_replica_count']) for c in test_conditions))
    
    results = {}
    
    for r in replica_counts:
        # Get test windows for this replica count
        r_norm = r / config['max_replica_count']
        mask = np.abs(test_conditions - r_norm) < 0.01
        real_windows = test_windows[mask]
        
        if len(real_windows) < 5:
            continue
        
        # Generate same number of windows for this replica count
        generated = model.generate([r], n_per_count=len(real_windows))[r]
        
        # Compute metrics
        n_metrics = real_windows.shape[2]
        metric_names = config['core_metrics'] if config['use_reduced_metrics'] else ALL_METRICS
        
        results[r] = {'per_metric': {}}
        
        for m in range(n_metrics):
            real_vals = real_windows[:, :, m]
            gen_vals = generated[:, :, m]
            
            real_std = np.std(real_vals)
            gen_std = np.std(gen_vals)
            
            # Temporal variance (average variance within each window)
            real_temp_var = np.mean([np.var(real_vals[j]) for j in range(len(real_vals))])
            gen_temp_var = np.mean([np.var(gen_vals[j]) for j in range(len(gen_vals))])
            
            results[r]['per_metric'][metric_names[m]] = {
                'real_std': float(real_std),
                'gen_std': float(gen_std),
                'std_ratio': float(gen_std / real_std) if real_std > 0 else 0,
                'real_temporal_var': float(real_temp_var),
                'gen_temporal_var': float(gen_temp_var),
                'temporal_var_ratio': float(gen_temp_var / real_temp_var) if real_temp_var > 0 else 0
            }
        
        # Overall temporal variance ratio
        real_overall_temp = np.mean([np.var(real_windows[j]) for j in range(len(real_windows))])
        gen_overall_temp = np.mean([np.var(generated[j]) for j in range(len(generated))])
        
        results[r]['overall_temporal_var_ratio'] = float(gen_overall_temp / real_overall_temp) if real_overall_temp > 0 else 0
        results[r]['n_samples'] = len(real_windows)
    
    return results


def plot_by_replica_count(model, test_windows, test_conditions, config, output_dir):
    """Plot comparison for different replica counts"""
    replica_counts = sorted(set(int(c * config['max_replica_count']) for c in test_conditions))
    metric_names = config['core_metrics'] if config['use_reduced_metrics'] else ALL_METRICS
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(len(replica_counts), n_metrics, figsize=(4*n_metrics, 3*len(replica_counts)))
    
    if len(replica_counts) == 1:
        axes = axes.reshape(1, -1)
    
    for row, r in enumerate(replica_counts):
        r_norm = r / config['max_replica_count']
        mask = np.abs(test_conditions - r_norm) < 0.01
        real_windows = test_windows[mask]
        
        if len(real_windows) < 1:
            continue
        
        generated = model.generate([r], n_per_count=min(3, len(real_windows)))[r]
        
        for col, metric in enumerate(metric_names):
            ax = axes[row, col]
            
            # Plot real
            for i in range(min(3, len(real_windows))):
                ax.plot(real_windows[i, :, col], 'b-', alpha=0.5, 
                       label='Real' if i == 0 else '')
            
            # Plot generated
            for i in range(len(generated)):
                ax.plot(generated[i, :, col], 'r--', alpha=0.5,
                       label='Generated' if i == 0 else '')
            
            if row == 0:
                ax.set_title(metric)
            if col == 0:
                ax.set_ylabel(f'r={r}')
            if row == len(replica_counts) - 1:
                ax.set_xlabel('Time Step')
            
            ax.legend(fontsize=6)
    
    plt.suptitle('Conditional Generation by Replica Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'conditional_comparison.png', dpi=150)
    plt.close()


def plot_extrapolation(model, config, output_dir):
    """Plot extrapolation to unseen replica counts"""
    metric_names = config['core_metrics'] if config['use_reduced_metrics'] else ALL_METRICS
    
    # Generate for seen and unseen replica counts
    seen_counts = [1, 3, 6, 10]
    unseen_counts = [15, 20, 30, 50]
    all_counts = seen_counts + unseen_counts
    
    generated = model.generate(all_counts, n_per_count=3)
    
    fig, axes = plt.subplots(len(all_counts), len(metric_names), 
                            figsize=(4*len(metric_names), 2.5*len(all_counts)))
    
    for row, r in enumerate(all_counts):
        traces = generated[r]
        is_extrapolation = r in unseen_counts
        color = 'orange' if is_extrapolation else 'green'
        
        for col, metric in enumerate(metric_names):
            ax = axes[row, col]
            
            for i in range(len(traces)):
                ax.plot(traces[i, :, col], color=color, alpha=0.5)
            
            if row == 0:
                ax.set_title(metric)
            if col == 0:
                label = f'r={r}' + (' (extrap)' if is_extrapolation else ' (seen)')
                ax.set_ylabel(label)
    
    plt.suptitle('Extrapolation to Higher Replica Counts\n(Green=Training data, Orange=Extrapolation)')
    plt.tight_layout()
    plt.savefig(output_dir / 'extrapolation.png', dpi=150)
    plt.close()


def main():
    print("="*70)
    print("CONDITIONAL TIMEGAN WITH REPLICA COUNT")
    print("="*70)
    
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data with metadata
    print("\n[1] Loading data with replica count metadata...")
    data, metadata = load_data_with_metadata(CONFIG)
    print(f"    Total traces: {len(data)}")
    
    # Show replica count distribution
    replica_counts = [m.get('replica_count', 1) for m in metadata]
    unique_counts = sorted(set(replica_counts))
    print(f"    Replica counts in data: {unique_counts}")
    for r in unique_counts:
        count = sum(1 for rc in replica_counts if rc == r)
        print(f"      r={r}: {count} traces")
    
    # Create windows with conditions
    print("\n[2] Creating windows with replica count conditions...")
    windows, conditions, window_meta = create_windows_with_conditions(data, metadata, CONFIG)
    
    input_dim = windows.shape[2]
    print(f"    Windows: {len(windows)}")
    print(f"    Input dim: {input_dim} metrics")
    if CONFIG['use_reduced_metrics']:
        print(f"    Using core metrics: {CONFIG['core_metrics']}")
    
    # Train/test split
    n_train = int(len(windows) * 0.8)
    indices = np.random.permutation(len(windows))
    
    train_windows = windows[indices[:n_train]]
    train_conditions = conditions[indices[:n_train]]
    test_windows = windows[indices[n_train:]]
    test_conditions = conditions[indices[n_train:]]
    test_meta = [window_meta[i] for i in indices[n_train:]]
    
    print(f"    Train: {len(train_windows)}, Test: {len(test_windows)}")
    
    # Create dataloader
    train_tensor = torch.FloatTensor(train_windows)
    cond_tensor = torch.FloatTensor(train_conditions)
    train_loader = DataLoader(
        TensorDataset(train_tensor, cond_tensor),
        batch_size=CONFIG['batch_size'],
        shuffle=True
    )
    
    # Initialize model
    print("\n[3] Initializing Conditional TimeGAN...")
    model = ConditionalTimeGAN(CONFIG, input_dim)
    
    # Train
    print("\n[4] Training...")
    model.train_embedder(train_loader, CONFIG['epochs_embed'])
    model.train_supervisor(train_loader, CONFIG['epochs_supervisor'])
    model.train_joint(train_loader, CONFIG['epochs_joint'])
    
    # Evaluate
    print("\n[5] Evaluating by replica count...")
    results = evaluate_by_replica_count(model, test_windows, test_conditions, test_meta, CONFIG)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS BY REPLICA COUNT")
    print("="*70)
    
    for r in sorted(results.keys()):
        res = results[r]
        print(f"\nReplica Count r={r} (n={res['n_samples']} samples):")
        print(f"  Overall Temporal Var Ratio: {res['overall_temporal_var_ratio']:.4f}")
        print(f"  Per-metric:")
        for metric, m_res in res['per_metric'].items():
            print(f"    {metric}: std_ratio={m_res['std_ratio']:.4f}, temp_var_ratio={m_res['temporal_var_ratio']:.4f}")
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    print("\n[6] Creating visualizations...")
    plot_by_replica_count(model, test_windows, test_conditions, CONFIG, output_dir)
    plot_extrapolation(model, CONFIG, output_dir)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].plot(model.history['embed'])
    axes[0].set_title('Embedding Loss')
    axes[1].plot(model.history['supervisor'])
    axes[1].set_title('Supervisor Loss')
    axes[2].plot(model.history['g_loss'])
    axes[2].set_title('Generator Loss')
    axes[3].plot(model.history['d_loss'])
    axes[3].set_title('Discriminator Loss')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    print(f"\n[7] Results saved to: {output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()