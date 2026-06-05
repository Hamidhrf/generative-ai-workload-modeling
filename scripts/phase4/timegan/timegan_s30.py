#!/usr/bin/env python3
"""
S30: Unified TimeGAN for GPU-Bound Workloads Only
Key Innovation: Excludes Whisper (CPU-bound) to focus on GPU-saturated pattern space
Hypothesis: Cleaner embedding space without conflicting CPU-bound patterns

Dataset: 206 pods (BERT: 55, GPT2: 55, ResNet152: 55, YOLO: 41)
Architecture: Same as S29 (seg_len=120, 6 phases, workload embedding)
Expected: YOLO VR > 0.8, maintain/improve BERT/GPT2/ResNet152
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("S30: GPU-BOUND UNIFIED TIMEGAN TRAINING")
print("=" * 80)
print(f"Excluding: Whisper (CPU-bound workload)")
print(f"Including: BERT, GPT2, ResNet152, YOLO (GPU-bound workloads)")
print("=" * 80)

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# S30 specific configuration
EXCLUDE_WORKLOADS = ['whisper']  # KEY CHANGE: Exclude Whisper
VALID_WORKLOADS = ['bert', 'gpt2', 'resnet152', 'yolo']  # Only GPU-bound
N_WORKLOADS = 4  # 4 workload classes

# Model architecture (same as S29)
SEG_LEN = 120
N_PHASES = 6
PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
HIDDEN_DIM = 256
LATENT_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.1

# Training hyperparameters (same as S29 - these worked well)
BATCH_SIZE = 32
N_EPOCHS = 150
LR_G = 1e-4
LR_D = 1e-4
LAMBDA_VR = 0.4      # Variance regression
LAMBDA_FM_STAT = 1.2 # Frequency-moment statistics
LAMBDA_AE = 1.5      # Autoencoder reconstruction

# Paths
DATA_PATH = project_root / 'data' / 'processed' / 'phase1_v3_pod_level.npz'
OUTPUT_DIR = project_root / 'outputs' / 'phase4' / 'timegan_s30'
MODEL_DIR = project_root / 'models' / 'phase4' / 'timegan_s30'
RUN_TAG = 's30_gpu_bound_vr04_fm12_ae150'

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
run_output_dir = OUTPUT_DIR / RUN_TAG
run_model_dir = MODEL_DIR / RUN_TAG
run_output_dir.mkdir(exist_ok=True)
run_model_dir.mkdir(exist_ok=True)

print(f"\nData path: {DATA_PATH}")
print(f"Output dir: {run_output_dir}")
print(f"Model dir: {run_model_dir}")

# Load data
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

data = np.load(DATA_PATH, allow_pickle=True)
all_traces = data['traces']  # (N, 715, 10)
all_replica_counts = data['replica_counts']
all_workloads = data['workload_labels']
metadata = data['metadata']

print(f"Original dataset: {all_traces.shape[0]} pods")
print(f"Trace shape: {all_traces.shape[1:]} (time x features)")

# Filter out Whisper
print("\nFiltering dataset...")
print(f"Excluding: {EXCLUDE_WORKLOADS}")
print(f"Including: {VALID_WORKLOADS}")

# Create filter mask
valid_mask = np.array([meta['workload'] in VALID_WORKLOADS for meta in metadata])

# Apply filter
traces = all_traces[valid_mask]
replica_counts = all_replica_counts[valid_mask]
workload_labels = all_workloads[valid_mask]
filtered_metadata = [meta for meta, valid in zip(metadata, valid_mask) if valid]

print(f"\nFiltered dataset: {traces.shape[0]} pods ({traces.shape[0] / all_traces.shape[0] * 100:.1f}%)")
print(f"Excluded: {all_traces.shape[0] - traces.shape[0]} Whisper pods")

# Count pods per workload
workload_counts = {}
for meta in filtered_metadata:
    workload_counts[meta['workload']] = workload_counts.get(meta['workload'], 0) + 1

print("\nDataset composition:")
for wl in VALID_WORKLOADS:
    count = workload_counts.get(wl, 0)
    print(f"  {wl.upper():12s}: {count:3d} pods ({count / len(filtered_metadata) * 100:5.1f}%)")
print(f"  {'TOTAL':12s}: {len(filtered_metadata):3d} pods")

# Trim to 715 timesteps (if needed)
if traces.shape[1] == 720:
    traces = traces[:, :715, :]
    print(f"\nTrimmed traces to {traces.shape[1]} timesteps")

N_PODS, SEQ_LEN, N_FEATURES = traces.shape
print(f"\nFinal data shape: {traces.shape}")
print(f"Features: {N_FEATURES}")
print(f"Sequence length: {SEQ_LEN}")

# Normalize data (z-score per feature)
print("\nNormalizing data...")
mean = traces.mean(axis=(0, 1))
std = traces.std(axis=(0, 1))
std[std < 1e-8] = 1.0  # Prevent division by zero for constant features
traces_normalized = (traces - mean) / std

print(f"Mean per feature: min={mean.min():.4f}, max={mean.max():.4f}")
print(f"Std per feature: min={std.min():.4f}, max={std.max():.4f}")

# Convert workload labels to integer indices
print("\nEncoding workload labels...")
workload_to_idx = {wl: idx for idx, wl in enumerate(VALID_WORKLOADS)}
workload_indices = np.array([workload_to_idx[meta['workload']] for meta in filtered_metadata])

print(f"Workload encoding:")
for wl, idx in workload_to_idx.items():
    count = (workload_indices == idx).sum()
    print(f"  {idx}: {wl.upper():12s} ({count} pods)")

# Create dataset
class PodTraceDataset(Dataset):
    def __init__(self, traces, replica_counts, workload_indices):
        self.traces = torch.FloatTensor(traces)
        self.replica_counts = torch.FloatTensor(replica_counts).unsqueeze(1) / 10.0  # Normalize to [0,1]
        self.workload_indices = torch.LongTensor(workload_indices)
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return self.traces[idx], self.replica_counts[idx], self.workload_indices[idx]

# Train/val split (80/20)
print("\nCreating train/val split...")
n_train = int(0.8 * N_PODS)
indices = np.random.permutation(N_PODS)
train_indices = indices[:n_train]
val_indices = indices[n_train:]

train_dataset = PodTraceDataset(
    traces_normalized[train_indices],
    replica_counts[train_indices],
    workload_indices[train_indices]
)

val_dataset = PodTraceDataset(
    traces_normalized[val_indices],
    replica_counts[val_indices],
    workload_indices[val_indices]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train set: {len(train_dataset)} pods ({len(train_dataset) / N_PODS * 100:.1f}%)")
print(f"Val set: {len(val_dataset)} pods ({len(val_dataset) / N_PODS * 100:.1f}%)")
print(f"Batch size: {BATCH_SIZE}")
print(f"Train batches per epoch: {len(train_loader)}")

# Model architecture (same as S29)
print("\n" + "=" * 80)
print("MODEL ARCHITECTURE")
print("=" * 80)

class WorkloadEmbedding(nn.Module):
    """Embedding for 4 workload types"""
    def __init__(self, n_workloads, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_workloads, embed_dim)
    
    def forward(self, workload_indices):
        return self.embedding(workload_indices)

class ConditionEncoder(nn.Module):
    """Encodes replica count + workload embedding"""
    def __init__(self, n_workloads, embed_dim=32, output_dim=64):
        super().__init__()
        self.workload_embed = WorkloadEmbedding(n_workloads, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(1 + embed_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, replica_count, workload_idx):
        workload_emb = self.workload_embed(workload_idx)
        condition = torch.cat([replica_count, workload_emb], dim=1)
        return self.fc(condition)

class GeneratorSeg(nn.Module):
    """Segment-based generator with phase boundaries"""
    def __init__(self, latent_dim, hidden_dim, n_features, condition_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        
        # Phase index embedding
        self.phase_embed = nn.Embedding(N_PHASES, 16)
        
        # Input projection: latent + condition + phase
        input_dim = latent_dim + condition_dim + 16
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
            nn.Tanh()
        )
    
    def forward(self, z, condition, phase_idx):
        batch_size = z.size(0)
        
        # Embed phase
        phase_emb = self.phase_embed(phase_idx).unsqueeze(1).expand(batch_size, SEG_LEN, -1)
        
        # Expand latent and condition
        z_expanded = z.unsqueeze(1).expand(batch_size, SEG_LEN, -1)
        condition_expanded = condition.unsqueeze(1).expand(batch_size, SEG_LEN, -1)
        
        # Concatenate inputs
        x = torch.cat([z_expanded, condition_expanded, phase_emb], dim=2)
        x = self.input_proj(x)
        
        # LSTM
        out, _ = self.lstm(x)
        
        # Output projection
        out = self.output_proj(out)
        
        return out

class Discriminator(nn.Module):
    """Conditional discriminator"""
    def __init__(self, n_features, hidden_dim, condition_dim, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Input projection: features + condition
        input_dim = n_features + condition_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, condition):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Expand condition
        condition_expanded = condition.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Concatenate inputs
        x = torch.cat([x, condition_expanded], dim=2)
        x = self.input_proj(x)
        
        # LSTM
        out, _ = self.lstm(x)
        
        # Use last timestep
        out = out[:, -1, :]
        
        # Output projection
        out = self.output_proj(out)
        
        return out

# Initialize models
print("Initializing models...")
condition_encoder = ConditionEncoder(N_WORKLOADS, embed_dim=32, output_dim=64).to(DEVICE)
generator = GeneratorSeg(LATENT_DIM, HIDDEN_DIM, N_FEATURES, condition_dim=64, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
discriminator = Discriminator(N_FEATURES, HIDDEN_DIM, condition_dim=64, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)

print(f"\nModel summary:")
print(f"  Condition Encoder: {sum(p.numel() for p in condition_encoder.parameters()):,} params")
print(f"  Generator: {sum(p.numel() for p in generator.parameters()):,} params")
print(f"  Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params")
print(f"  Total: {sum(p.numel() for p in condition_encoder.parameters()) + sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters()):,} params")

# Optimizers
optimizer_G = optim.Adam(list(condition_encoder.parameters()) + list(generator.parameters()), lr=LR_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))

print(f"\nOptimizers:")
print(f"  Generator LR: {LR_G}")
print(f"  Discriminator LR: {LR_D}")

# Loss functions
def variance_loss(fake, real):
    """Variance regression loss"""
    fake_var = fake.var(dim=1, unbiased=False)
    real_var = real.var(dim=1, unbiased=False)
    return nn.functional.mse_loss(fake_var, real_var)

def frequency_moment_loss(fake, real):
    """Frequency-domain and moment matching"""
    # Mean
    loss_mean = nn.functional.mse_loss(fake.mean(dim=1), real.mean(dim=1))
    
    # Std
    loss_std = nn.functional.mse_loss(fake.std(dim=1), real.std(dim=1))
    
    # Autocorrelation (lag=1)
    fake_ac = (fake[:, :-1] * fake[:, 1:]).mean(dim=1)
    real_ac = (real[:, :-1] * real[:, 1:]).mean(dim=1)
    loss_ac = nn.functional.mse_loss(fake_ac, real_ac)
    
    return loss_mean + loss_std + loss_ac

def autoencoder_loss(fake, real):
    """Reconstruction loss"""
    return nn.functional.mse_loss(fake, real)

# Training loop
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)
print(f"Epochs: {N_EPOCHS}")
print(f"Loss weights:")
print(f"  Variance regression: {LAMBDA_VR}")
print(f"  Frequency-moment: {LAMBDA_FM_STAT}")
print(f"  Autoencoder: {LAMBDA_AE}")
print("=" * 80)

history = {
    'train_loss_G': [],
    'train_loss_D': [],
    'val_loss_G': [],
    'val_loss_D': []
}

best_val_loss = float('inf')
start_time = time.time()

for epoch in range(N_EPOCHS):
    epoch_start = time.time()
    
    # Training
    condition_encoder.train()
    generator.train()
    discriminator.train()
    
    train_loss_G_epoch = 0
    train_loss_D_epoch = 0
    
    for batch_idx, (real_traces, replica_count, workload_idx) in enumerate(train_loader):
        real_traces = real_traces.to(DEVICE)
        replica_count = replica_count.to(DEVICE)
        workload_idx = workload_idx.to(DEVICE)
        
        batch_size = real_traces.size(0)
        
        # Encode condition
        condition = condition_encoder(replica_count, workload_idx)
        
        # Generate fake traces (6 segments)
        fake_segments = []
        for phase_idx in range(N_PHASES):
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            phase_tensor = torch.full((batch_size,), phase_idx, dtype=torch.long).to(DEVICE)
            fake_seg = generator(z, condition, phase_tensor)
            fake_segments.append(fake_seg)
        
        fake_traces = torch.cat(fake_segments, dim=1)
        
        # Trim to 715 timesteps
        fake_traces = fake_traces[:, :SEQ_LEN, :]
        
        # Train discriminator
        optimizer_D.zero_grad()
        
        real_pred = discriminator(real_traces, condition)
        fake_pred = discriminator(fake_traces.detach(), condition)
        
        loss_D_real = nn.functional.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        loss_D_fake = nn.functional.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        loss_D = (loss_D_real + loss_D_fake) / 2
        
        loss_D.backward()
        optimizer_D.step()
        
        # Train generator
        optimizer_G.zero_grad()
        
        fake_pred = discriminator(fake_traces, condition)
        loss_adv = nn.functional.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        
        loss_vr = variance_loss(fake_traces, real_traces)
        loss_fm = frequency_moment_loss(fake_traces, real_traces)
        loss_ae = autoencoder_loss(fake_traces, real_traces)
        
        loss_G = loss_adv + LAMBDA_VR * loss_vr + LAMBDA_FM_STAT * loss_fm + LAMBDA_AE * loss_ae
        
        loss_G.backward()
        optimizer_G.step()
        
        train_loss_G_epoch += loss_G.item()
        train_loss_D_epoch += loss_D.item()
    
    train_loss_G_epoch /= len(train_loader)
    train_loss_D_epoch /= len(train_loader)
    
    # Validation
    condition_encoder.eval()
    generator.eval()
    discriminator.eval()
    
    val_loss_G_epoch = 0
    val_loss_D_epoch = 0
    
    with torch.no_grad():
        for real_traces, replica_count, workload_idx in val_loader:
            real_traces = real_traces.to(DEVICE)
            replica_count = replica_count.to(DEVICE)
            workload_idx = workload_idx.to(DEVICE)
            
            batch_size = real_traces.size(0)
            
            condition = condition_encoder(replica_count, workload_idx)
            
            # Generate fake traces
            fake_segments = []
            for phase_idx in range(N_PHASES):
                z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
                phase_tensor = torch.full((batch_size,), phase_idx, dtype=torch.long).to(DEVICE)
                fake_seg = generator(z, condition, phase_tensor)
                fake_segments.append(fake_seg)
            
            fake_traces = torch.cat(fake_segments, dim=1)
            fake_traces = fake_traces[:, :SEQ_LEN, :]
            
            # Discriminator loss
            real_pred = discriminator(real_traces, condition)
            fake_pred = discriminator(fake_traces, condition)
            loss_D = (nn.functional.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) + 
                     nn.functional.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))) / 2
            
            # Generator loss
            fake_pred = discriminator(fake_traces, condition)
            loss_adv = nn.functional.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
            loss_vr = variance_loss(fake_traces, real_traces)
            loss_fm = frequency_moment_loss(fake_traces, real_traces)
            loss_ae = autoencoder_loss(fake_traces, real_traces)
            loss_G = loss_adv + LAMBDA_VR * loss_vr + LAMBDA_FM_STAT * loss_fm + LAMBDA_AE * loss_ae
            
            val_loss_G_epoch += loss_G.item()
            val_loss_D_epoch += loss_D.item()
    
    val_loss_G_epoch /= len(val_loader)
    val_loss_D_epoch /= len(val_loader)
    
    # Save history
    history['train_loss_G'].append(train_loss_G_epoch)
    history['train_loss_D'].append(train_loss_D_epoch)
    history['val_loss_G'].append(val_loss_G_epoch)
    history['val_loss_D'].append(val_loss_D_epoch)
    
    epoch_time = time.time() - epoch_start
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{N_EPOCHS}] | "
              f"Train G: {train_loss_G_epoch:.4f} D: {train_loss_D_epoch:.4f} | "
              f"Val G: {val_loss_G_epoch:.4f} D: {val_loss_D_epoch:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    # Save best model
    if val_loss_G_epoch < best_val_loss:
        best_val_loss = val_loss_G_epoch
        torch.save({
            'epoch': epoch,
            'condition_encoder_state_dict': condition_encoder.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'val_loss_G': val_loss_G_epoch,
            'val_loss_D': val_loss_D_epoch,
        }, run_model_dir / 'best_model.pt')

total_time = time.time() - start_time
print(f"\nTraining completed in {total_time / 3600:.2f} hours")
print(f"Best validation loss G: {best_val_loss:.4f}")

# Save final models
torch.save(condition_encoder.state_dict(), run_model_dir / 'condition_encoder.pt')
torch.save(generator.state_dict(), run_model_dir / 'generator.pt')
torch.save(discriminator.state_dict(), run_model_dir / 'discriminator.pt')

# Save training history
with open(run_output_dir / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

# Save normalization parameters
with open(run_model_dir / 'normalization_params.json', 'w') as f:
    json.dump({
        'mean': mean.tolist(),
        'std': std.tolist()
    }, f, indent=2)

# Save configuration
config = {
    'model': 'S30_GPU_Bound_Unified',
    'excluded_workloads': EXCLUDE_WORKLOADS,
    'valid_workloads': VALID_WORKLOADS,
    'n_workloads': N_WORKLOADS,
    'dataset_size': N_PODS,
    'train_size': len(train_dataset),
    'val_size': len(val_dataset),
    'architecture': {
        'seg_len': SEG_LEN,
        'n_phases': N_PHASES,
        'hidden_dim': HIDDEN_DIM,
        'latent_dim': LATENT_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    },
    'hyperparameters': {
        'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
        'lr_g': LR_G,
        'lr_d': LR_D,
        'lambda_vr': LAMBDA_VR,
        'lambda_fm_stat': LAMBDA_FM_STAT,
        'lambda_ae': LAMBDA_AE
    },
    'training_time_hours': total_time / 3600,
    'best_val_loss_G': best_val_loss
}

with open(run_output_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\nModel saved to: {run_model_dir}")
print(f"Results saved to: {run_output_dir}")
print("\n" + "=" * 80)
print("S30 TRAINING COMPLETE")
print("=" * 80)
print(f"Next step: Run evaluation script (evaluate_s30.py)")
print("=" * 80)