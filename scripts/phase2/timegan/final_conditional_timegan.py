#!/usr/bin/env python3
"""
FINAL APPROACH: Workload-Specific Conditional Windowed TimeGAN
===============================================================

Combines all three solutions:
1. Workload-specific: Separate model per app (ResNet50, DistilBERT, Whisper)
2. Windowed: Sliding windows for more training samples
3. Conditional: Replica count conditioning for learning scaling behavior

This is the correct approach based on:
- Thesis requirement: "for each application scaled to 10x-100x replicas"
- Data structure: 3 workloads x multiple replica counts
- Training needs: Enough samples per condition

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
    'output_dir': '/home/hamid/generative-ai-workload-modeling/outputs/final_conditional_timegan',
    
    # Use 4 core metrics to start (can expand later)
    'use_reduced_metrics': True,
    'core_metrics': ['cpu_usage', 'gpu_utilization', 'memory_usage', 'latency'],
    'core_metric_indices': [1, 5, 7, 9],
    
    # Sliding window
    'window_size': 64,
    'stride': 8,  # More overlap = more samples
    
    # Model architecture (smaller for per-workload training)
    'hidden_dim': 64,
    'latent_dim': 32,
    'num_layers': 2,
    'condition_dim': 16,
    
    # Training
    'batch_size': 32,
    'epochs_embed': 150,
    'epochs_supervisor': 150,
    'epochs_joint': 300,
    'learning_rate': 0.0005,
    
    # Replica count range for normalization
    'max_replica_count': 15,  # Max in training data (for normalization)
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42
}

ALL_METRICS = ['cpu_psi', 'cpu_usage', 'cpu_num', 'cpu_temperature', 
               'gpu_power', 'gpu_utilization', 'gpu_temperature',
               'memory_usage', 'memory_psi', 'latency', 'latency_p50',
               'latency_p95', 'latency_p99', 'success_rate', 'throughput']


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class ReplicaCountEmbedding(nn.Module):
    """Learnable embedding for replica count"""
    def __init__(self, condition_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(condition_dim, condition_dim),
            nn.Tanh()
        )
    
    def forward(self, r):
        # r: (batch,) normalized replica count [0, 1]
        return self.net(r.unsqueeze(-1))


class ConditionalEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim, num_layers):
        super().__init__()
        self.cond_embed = ReplicaCountEmbedding(condition_dim)
        self.rnn = nn.GRU(input_dim + condition_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, r):
        batch_size, seq_len, _ = x.shape
        cond = self.cond_embed(r).unsqueeze(1).expand(-1, seq_len, -1)
        x_cond = torch.cat([x, cond], dim=-1)
        h, _ = self.rnn(x_cond)
        return torch.sigmoid(self.fc(h))


class ConditionalRecovery(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim, num_layers):
        super().__init__()
        self.cond_embed = ReplicaCountEmbedding(condition_dim)
        self.rnn = nn.GRU(latent_dim + condition_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h, r):
        batch_size, seq_len, _ = h.shape
        cond = self.cond_embed(r).unsqueeze(1).expand(-1, seq_len, -1)
        h_cond = torch.cat([h, cond], dim=-1)
        out, _ = self.rnn(h_cond)
        return torch.sigmoid(self.fc(out))


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, condition_dim, num_layers):
        super().__init__()
        self.cond_embed = ReplicaCountEmbedding(condition_dim)
        self.rnn = nn.GRU(latent_dim + condition_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z, r):
        batch_size, seq_len, _ = z.shape
        cond = self.cond_embed(r).unsqueeze(1).expand(-1, seq_len, -1)
        z_cond = torch.cat([z, cond], dim=-1)
        h, _ = self.rnn(z_cond)
        return torch.sigmoid(self.fc(h))


class ConditionalSupervisor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, condition_dim, num_layers):
        super().__init__()
        self.cond_embed = ReplicaCountEmbedding(condition_dim)
        self.rnn = nn.GRU(latent_dim + condition_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, h, r):
        batch_size, seq_len, _ = h.shape
        cond = self.cond_embed(r).unsqueeze(1).expand(-1, seq_len, -1)
        h_cond = torch.cat([h, cond], dim=-1)
        out, _ = self.rnn(h_cond)
        return torch.sigmoid(self.fc(out))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, condition_dim, num_layers):
        super().__init__()
        self.cond_embed = ReplicaCountEmbedding(condition_dim)
        self.rnn = nn.GRU(latent_dim + condition_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, h, r):
        batch_size, seq_len, _ = h.shape
        cond = self.cond_embed(r).unsqueeze(1).expand(-1, seq_len, -1)
        h_cond = torch.cat([h, cond], dim=-1)
        out, _ = self.rnn(h_cond)
        return self.fc(out)


# =============================================================================
# TIMEGAN MODEL
# =============================================================================

class WorkloadConditionalTimeGAN:
    """TimeGAN for a single workload, conditioned on replica count"""
    
    def __init__(self, config, input_dim, workload_name):
        self.config = config
        self.input_dim = input_dim
        self.workload_name = workload_name
        self.device = torch.device(config['device'])
        
        # Initialize networks
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
    
    def train(self, dataloader):
        """Full training pipeline"""
        print(f"\n{'='*60}")
        print(f"Training {self.workload_name.upper()} Conditional TimeGAN")
        print(f"{'='*60}")
        
        self._train_embedder(dataloader)
        self._train_supervisor(dataloader)
        self._train_joint(dataloader)
    
    def _train_embedder(self, dataloader):
        print(f"\n  Phase 1: Embedder ({self.config['epochs_embed']} epochs)")
        criterion = nn.MSELoss()
        
        for epoch in range(self.config['epochs_embed']):
            total_loss = 0
            for x, r in dataloader:
                x, r = x.to(self.device), r.to(self.device)
                
                self.opt_embed.zero_grad()
                h = self.embedder(x, r)
                x_recon = self.recovery(h, r)
                loss = criterion(x_recon, x)
                loss.backward()
                self.opt_embed.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['embed'].append(avg_loss)
            
            if (epoch + 1) % 30 == 0:
                print(f"    Epoch {epoch+1}/{self.config['epochs_embed']} | Loss: {avg_loss:.6f}")
    
    def _train_supervisor(self, dataloader):
        print(f"\n  Phase 2: Supervisor ({self.config['epochs_supervisor']} epochs)")
        criterion = nn.MSELoss()
        
        for epoch in range(self.config['epochs_supervisor']):
            total_loss = 0
            for x, r in dataloader:
                x, r = x.to(self.device), r.to(self.device)
                
                self.opt_supervisor.zero_grad()
                with torch.no_grad():
                    h = self.embedder(x, r)
                
                h_sup = self.supervisor(h, r)
                loss = criterion(h_sup[:, :-1, :], h[:, 1:, :])
                loss.backward()
                self.opt_supervisor.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['supervisor'].append(avg_loss)
            
            if (epoch + 1) % 30 == 0:
                print(f"    Epoch {epoch+1}/{self.config['epochs_supervisor']} | Loss: {avg_loss:.6f}")
    
    def _train_joint(self, dataloader):
        print(f"\n  Phase 3: Joint Training ({self.config['epochs_joint']} epochs)")
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_mse = nn.MSELoss()
        
        for epoch in range(self.config['epochs_joint']):
            g_losses, d_losses = [], []
            
            for x, r in dataloader:
                x, r = x.to(self.device), r.to(self.device)
                batch_size, seq_len = x.size(0), x.size(1)
                
                z = torch.randn(batch_size, seq_len, self.config['latent_dim']).to(self.device)
                
                with torch.no_grad():
                    h_real = self.embedder(x, r)
                
                h_fake_raw = self.generator(z, r)
                h_fake = self.supervisor(h_fake_raw, r)
                
                # Train Discriminator
                self.opt_discriminator.zero_grad()
                d_real = self.discriminator(h_real, r)
                d_fake = self.discriminator(h_fake.detach(), r)
                
                d_loss = criterion_bce(d_real, torch.ones_like(d_real) * 0.9) + \
                         criterion_bce(d_fake, torch.zeros_like(d_fake) + 0.1)
                d_loss.backward()
                self.opt_discriminator.step()
                
                # Train Generator
                self.opt_generator.zero_grad()
                h_fake_raw = self.generator(z, r)
                h_fake = self.supervisor(h_fake_raw, r)
                d_fake = self.discriminator(h_fake, r)
                
                g_loss_adv = criterion_bce(d_fake, torch.ones_like(d_fake))
                
                h_sup = self.supervisor(h_real, r)
                g_loss_sup = criterion_mse(h_sup[:, :-1, :], h_real[:, 1:, :])
                
                x_fake = self.recovery(h_fake, r)
                h_fake_emb = self.embedder(x_fake, r)
                g_loss_fm = criterion_mse(h_fake, h_fake_emb)
                
                g_loss = g_loss_adv + 10 * g_loss_sup + 10 * g_loss_fm
                g_loss.backward()
                self.opt_generator.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            self.history['g_loss'].append(np.mean(g_losses))
            self.history['d_loss'].append(np.mean(d_losses))
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{self.config['epochs_joint']} | "
                      f"G: {np.mean(g_losses):.4f} | D: {np.mean(d_losses):.4f}")
    
    def generate(self, replica_count, n_samples=10):
        """Generate traces for a specific replica count"""
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        
        with torch.no_grad():
            r_norm = min(replica_count / self.config['max_replica_count'], 1.0)
            r = torch.full((n_samples,), r_norm).to(self.device)
            
            z = torch.randn(n_samples, self.config['window_size'],
                           self.config['latent_dim']).to(self.device)
            
            h = self.generator(z, r)
            h = self.supervisor(h, r)
            x = self.recovery(h, r)
        
        return x.cpu().numpy()


# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_and_split_by_workload(config):
    """Load data and split by workload"""
    data = np.load(config['data_path'])
    
    # Load or create metadata
    try:
        with open(config['metadata_path'], 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Creating metadata based on expected structure...")
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
    
    # Split by workload
    workload_data = {}
    for wl in ['resnet50', 'distilbert', 'whisper']:
        wl_traces = []
        wl_replica_counts = []
        
        for i, meta in enumerate(metadata):
            if meta.get('workload', '').lower() == wl or wl in meta.get('workload', '').lower():
                wl_traces.append(data[i])
                wl_replica_counts.append(meta.get('replica_count', 1))
        
        if wl_traces:
            workload_data[wl] = {
                'traces': np.array(wl_traces),
                'replica_counts': wl_replica_counts
            }
    
    return workload_data


def create_windows(traces, replica_counts, config):
    """Create sliding windows with replica count labels"""
    windows = []
    conditions = []
    
    for trace, r in zip(traces, replica_counts):
        seq_len = trace.shape[0]
        
        for start in range(0, seq_len - config['window_size'] + 1, config['stride']):
            window = trace[start:start + config['window_size']]
            
            if config['use_reduced_metrics']:
                window = window[:, config['core_metric_indices']]
            
            windows.append(window)
            conditions.append(r / config['max_replica_count'])
    
    return np.array(windows), np.array(conditions)


def evaluate_model(model, test_windows, test_conditions, config):
    """Evaluate model performance per replica count"""
    results = {}
    metric_names = config['core_metrics'] if config['use_reduced_metrics'] else ALL_METRICS
    
    # Get unique replica counts
    unique_r = sorted(set(int(c * config['max_replica_count']) for c in test_conditions))
    
    for r in unique_r:
        r_norm = r / config['max_replica_count']
        mask = np.abs(test_conditions - r_norm) < 0.01
        real = test_windows[mask]
        
        if len(real) < 3:
            continue
        
        gen = model.generate(r, n_samples=len(real))
        
        results[r] = {'n_samples': len(real), 'metrics': {}}
        
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
                'temporal_var_ratio': float(gen_temp_var / real_temp_var) if real_temp_var > 1e-6 else 0
            }
        
        # Overall
        real_temp = np.mean([np.var(real[j]) for j in range(len(real))])
        gen_temp = np.mean([np.var(gen[j]) for j in range(len(gen))])
        results[r]['overall_temporal_var_ratio'] = float(gen_temp / real_temp) if real_temp > 1e-6 else 0
    
    return results


def plot_results(model, test_windows, test_conditions, workload_name, config, output_dir):
    """Create comparison plots"""
    metric_names = config['core_metrics'] if config['use_reduced_metrics'] else ALL_METRICS
    unique_r = sorted(set(int(c * config['max_replica_count']) for c in test_conditions))
    
    # Plot 1: Comparison by replica count
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
        
        gen = model.generate(r, n_samples=min(3, len(real)))
        
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
    
    plt.suptitle(f'{workload_name.upper()} - Conditional Generation')
    plt.tight_layout()
    plt.savefig(output_dir / f'{workload_name}_comparison.png', dpi=150)
    plt.close()
    
    # Plot 2: Extrapolation
    fig, axes = plt.subplots(2, len(metric_names), figsize=(4*len(metric_names), 6))
    
    seen_r = [max(unique_r)]  # Highest seen
    extrap_r = [20, 50]  # Extrapolation targets
    
    for row, r_list in enumerate([seen_r, extrap_r]):
        for r in r_list:
            gen = model.generate(r, n_samples=3)
            color = 'green' if r in unique_r else 'orange'
            
            for col, metric in enumerate(metric_names):
                ax = axes[row, col]
                for i in range(len(gen)):
                    label = f'r={r}' if i == 0 else ''
                    ax.plot(gen[i, :, col], color=color, alpha=0.5, label=label)
                
                if row == 0:
                    ax.set_title(metric)
                ax.legend(fontsize=6)
        
        axes[row, 0].set_ylabel('Seen' if row == 0 else 'Extrapolated')
    
    plt.suptitle(f'{workload_name.upper()} - Extrapolation (Green=Seen, Orange=Extrapolated)')
    plt.tight_layout()
    plt.savefig(output_dir / f'{workload_name}_extrapolation.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("FINAL: Workload-Specific Conditional Windowed TimeGAN")
    print("="*70)
    
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1] Loading and splitting data by workload...")
    workload_data = load_and_split_by_workload(CONFIG)
    
    for wl, data in workload_data.items():
        print(f"    {wl}: {len(data['traces'])} traces, replica counts: {sorted(set(data['replica_counts']))}")
    
    # Process each workload
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
        print(f"  Replica count distribution:")
        for r in sorted(set(wl_data['replica_counts'])):
            r_norm = r / CONFIG['max_replica_count']
            count = np.sum(np.abs(conditions - r_norm) < 0.01)
            print(f"    r={r}: {count} windows")
        
        # Train/test split
        n_train = int(len(windows) * 0.8)
        indices = np.random.permutation(len(windows))
        
        train_windows = windows[indices[:n_train]]
        train_conditions = conditions[indices[:n_train]]
        test_windows = windows[indices[n_train:]]
        test_conditions = conditions[indices[n_train:]]
        
        print(f"  Train: {len(train_windows)}, Test: {len(test_windows)}")
        
        # Create dataloader
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(train_windows),
                torch.FloatTensor(train_conditions)
            ),
            batch_size=CONFIG['batch_size'],
            shuffle=True
        )
        
        # Train model
        model = WorkloadConditionalTimeGAN(CONFIG, input_dim, workload_name)
        model.train(train_loader)
        
        # Evaluate
        print(f"\n  Evaluating {workload_name}...")
        results = evaluate_model(model, test_windows, test_conditions, CONFIG)
        all_results[workload_name] = results
        
        # Print results
        print(f"\n  Results for {workload_name}:")
        for r in sorted(results.keys()):
            res = results[r]
            print(f"    r={r} (n={res['n_samples']}): temporal_var_ratio = {res['overall_temporal_var_ratio']:.4f}")
        
        # Plot
        plot_results(model, test_windows, test_conditions, workload_name, CONFIG, output_dir)
        
        # Save model
        torch.save({
            'embedder': model.embedder.state_dict(),
            'recovery': model.recovery.state_dict(),
            'generator': model.generator.state_dict(),
            'supervisor': model.supervisor.state_dict(),
            'discriminator': model.discriminator.state_dict(),
            'history': model.history
        }, output_dir / f'{workload_name}_model.pt')
    
    # Save all results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for wl, results in all_results.items():
        print(f"\n{wl.upper()}:")
        avg_ratio = np.mean([r['overall_temporal_var_ratio'] for r in results.values()])
        print(f"  Average temporal variance ratio: {avg_ratio:.4f}")
        
        if avg_ratio > 0.3:
            print(f"  Status: GOOD - Model captures temporal dynamics")
        elif avg_ratio > 0.1:
            print(f"  Status: PARTIAL - Some temporal dynamics captured")
        else:
            print(f"  Status: POOR - Flat output persists")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()