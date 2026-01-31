#!/usr/bin/env python3
"""
LSTM Baseline Model for Pod-Level Trace Generation
Phase 2: Model Selection - Baseline Implementation

This establishes baseline performance for comparison with TimeVAE/TimeGAN.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    'data_dir': '/home/hamid/generative-ai-workload-modeling/data/processed/phase1',
    'output_dir': '/home/hamid/generative-ai-workload-modeling/outputs/phase2_baseline',
    
    # Model architecture
    'input_dim': 15,           # 15 metrics
    'hidden_dim': 128,         # LSTM hidden size
    'num_layers': 2,           # Stacked LSTM layers
    'dropout': 0.2,            # Dropout rate
    'condition_dim': 4,        # replica_count (1) + workload_onehot (3)
    
    # Training
    'batch_size': 8,           # Small batch (only 60 samples)
    'epochs': 200,             # Max epochs
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 20,
    
    # Data split
    'test_size': 0.15,         # 15% test (~9 traces)
    'val_size': 0.15,          # 15% validation (~9 traces)
    'random_seed': 42,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Absolute normalization ranges (from Phase 2 EDA)
ABSOLUTE_RANGES = {
    'cpu_psi': (0.0, 1.0),
    'cpu_usage': (0.0, 8.0),
    'gpu_memory': (0.0, 20000.0),
    'gpu_power': (0.0, 100.0),
    'gpu_temperature': (0.0, 100.0),
    'gpu_utilization': (0.0, 100.0),
    'latency_avg': (0.0, 5.0),
    'latency_p50': (0.0, 5.0),
    'latency_p95': (0.0, 8.0),
    'latency_p99': (0.0, 10.0),
    'throughput': (0.0, 500.0),
    'total_inferences': (0.0, 500.0),
    'io_psi': (0.0, 1.0),
    'memory_psi': (0.0, 1.0),
    'memory_usage': (0.0, 10e9),
}

METRIC_NAMES = list(ABSOLUTE_RANGES.keys())

# ============================================================================
# NORMALIZATION
# ============================================================================

class AbsoluteNormalizer:
    """Normalize using absolute ranges (not data-dependent)"""
    
    def __init__(self, ranges=ABSOLUTE_RANGES):
        self.ranges = ranges
        self.metric_names = list(ranges.keys())
    
    def normalize(self, data):
        """Normalize to [0, 1]"""
        normalized = np.zeros_like(data)
        for i, metric in enumerate(self.metric_names):
            min_val, max_val = self.ranges[metric]
            normalized[..., i] = (data[..., i] - min_val) / (max_val - min_val)
            # Clip to [0, 1]
            normalized[..., i] = np.clip(normalized[..., i], 0, 1)
        return normalized
    
    def denormalize(self, normalized_data):
        """Denormalize from [0, 1] to original scale"""
        denormalized = np.zeros_like(normalized_data)
        for i, metric in enumerate(self.metric_names):
            min_val, max_val = self.ranges[metric]
            denormalized[..., i] = normalized_data[..., i] * (max_val - min_val) + min_val
        return denormalized

# ============================================================================
# DATASET
# ============================================================================

class PodTraceDataset(Dataset):
    """Dataset for pod-level time series with conditioning"""
    
    def __init__(self, traces, metadata, normalizer):
        """
        Args:
            traces: (N, T, F) array of pod traces
            metadata: list of dicts with workload, replica_count
            normalizer: AbsoluteNormalizer instance
        """
        self.traces = normalizer.normalize(traces)
        self.metadata = metadata
        
        # Workload encoding
        self.workload_map = {'distilbert': 0, 'resnet50': 1, 'whisper': 2}
        
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = torch.FloatTensor(self.traces[idx])  # (T, F)
        
        # Conditioning vector
        workload = self.metadata[idx]['workload']
        replica_count = self.metadata[idx]['replica_count']
        
        # One-hot workload
        workload_onehot = torch.zeros(3)
        workload_onehot[self.workload_map[workload]] = 1.0
        
        # Normalize replica_count to [0, 1]
        replica_norm = replica_count / 10.0  # Max observed is 10
        
        # Condition vector: [replica_count, workload_onehot]
        condition = torch.cat([
            torch.tensor([replica_norm]),
            workload_onehot
        ])
        
        return trace, condition

# ============================================================================
# MODEL
# ============================================================================

class ConditionalLSTM(nn.Module):
    """Multi-output LSTM conditioned on replica count and workload"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, condition_dim, dropout=0.2):
        super(ConditionalLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # LSTM (input = features + condition embedding)
        self.lstm = nn.LSTM(
            input_size=input_dim + 64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x, condition):
        """
        Args:
            x: (B, T, F) input traces
            condition: (B, C) conditioning vector
        Returns:
            output: (B, T, F) predicted traces
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed condition
        cond_embed = self.condition_embed(condition)  # (B, 64)
        
        # Expand condition to all timesteps
        cond_expanded = cond_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, 64)
        
        # Concatenate input with condition
        lstm_input = torch.cat([x, cond_expanded], dim=-1)  # (B, T, F+64)
        
        # LSTM
        lstm_out, _ = self.lstm(lstm_input)  # (B, T, H)
        
        # Output layer
        output = self.fc(lstm_out)  # (B, T, F)
        
        return output

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for traces, conditions in dataloader:
        traces = traces.to(device)
        conditions = conditions.to(device)
        
        # Forward pass
        outputs = model(traces, conditions)
        loss = criterion(outputs, traces)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for traces, conditions in dataloader:
            traces = traces.to(device)
            conditions = conditions.to(device)
            
            outputs = model(traces, conditions)
            loss = criterion(outputs, traces)
            
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(traces.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, predictions, targets

def calculate_metrics(predictions, targets, normalizer):
    """Calculate detailed metrics per metric"""
    # Denormalize
    pred_denorm = normalizer.denormalize(predictions)
    target_denorm = normalizer.denormalize(targets)
    
    metrics = {}
    for i, metric_name in enumerate(METRIC_NAMES):
        pred_metric = pred_denorm[..., i]
        target_metric = target_denorm[..., i]
        
        mse = np.mean((pred_metric - target_metric) ** 2)
        mae = np.mean(np.abs(pred_metric - target_metric))
        rmse = np.sqrt(mse)
        
        # Relative metrics (avoid division by zero)
        target_mean = np.mean(target_metric)
        if target_mean > 1e-6:
            mape = np.mean(np.abs((pred_metric - target_metric) / (target_metric + 1e-8))) * 100
        else:
            mape = 0.0
        
        metrics[metric_name] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    
    # Overall metrics
    overall_mse = np.mean([m['mse'] for m in metrics.values()])
    overall_mae = np.mean([m['mae'] for m in metrics.values()])
    overall_rmse = np.mean([m['rmse'] for m in metrics.values()])
    
    metrics['overall'] = {
        'mse': float(overall_mse),
        'mae': float(overall_mae),
        'rmse': float(overall_rmse)
    }
    
    return metrics

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(train_losses, val_losses, output_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('LSTM Baseline: Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300)
    plt.close()
    print(f"✓ Saved training curves")

def plot_sample_predictions(predictions, targets, normalizer, output_dir, n_samples=3):
    """Plot sample predictions vs ground truth"""
    pred_denorm = normalizer.denormalize(predictions[:n_samples])
    target_denorm = normalizer.denormalize(targets[:n_samples])
    
    # Plot first 6 metrics
    metrics_to_plot = ['cpu_usage', 'gpu_utilization', 'latency_avg', 
                       'memory_usage', 'throughput', 'cpu_psi']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        metric_idx = METRIC_NAMES.index(metric)
        
        for i in range(n_samples):
            ax.plot(target_denorm[i, :, metric_idx], 
                   label=f'True {i+1}', alpha=0.7, linewidth=1.5)
            ax.plot(pred_denorm[i, :, metric_idx], 
                   '--', label=f'Pred {i+1}', alpha=0.7, linewidth=1.5)
        
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Time (5s intervals)')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('LSTM Baseline: Sample Predictions vs Ground Truth', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_predictions.png', dpi=300)
    plt.close()
    print(f"✓ Saved sample predictions")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("LSTM BASELINE TRAINING - Phase 2")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------
    print("[1/6] Loading data...")
    
    data_dir = Path(CONFIG['data_dir'])
    traces = np.load(data_dir / 'pod_traces.npy')
    metadata = np.load(data_dir / 'pod_metadata.npy', allow_pickle=True)
    
    print(f"  Loaded {len(traces)} pod traces")
    print(f"  Shape: {traces.shape}")  # (60, 715, 15)
    print()
    
    # -------------------------------------------------------------------------
    # NORMALIZE
    # -------------------------------------------------------------------------
    print("[2/6] Normalizing data...")
    normalizer = AbsoluteNormalizer(ABSOLUTE_RANGES)
    print(f"  Using absolute ranges for {len(METRIC_NAMES)} metrics")
    print()
    
    # -------------------------------------------------------------------------
    # SPLIT DATA
    # -------------------------------------------------------------------------
    print("[3/6] Splitting data...")
    
    # Train/temp split
    train_traces, temp_traces, train_meta, temp_meta = train_test_split(
        traces, metadata, 
        test_size=(CONFIG['test_size'] + CONFIG['val_size']),
        random_state=CONFIG['random_seed'],
        shuffle=True
    )
    
    # Val/test split
    val_traces, test_traces, val_meta, test_meta = train_test_split(
        temp_traces, temp_meta,
        test_size=CONFIG['test_size'] / (CONFIG['test_size'] + CONFIG['val_size']),
        random_state=CONFIG['random_seed'],
        shuffle=True
    )
    
    print(f"  Train: {len(train_traces)} traces")
    print(f"  Val:   {len(val_traces)} traces")
    print(f"  Test:  {len(test_traces)} traces")
    print()
    
    # Create datasets
    train_dataset = PodTraceDataset(train_traces, train_meta, normalizer)
    val_dataset = PodTraceDataset(val_traces, val_meta, normalizer)
    test_dataset = PodTraceDataset(test_traces, test_meta, normalizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # -------------------------------------------------------------------------
    # CREATE MODEL
    # -------------------------------------------------------------------------
    print("[4/6] Creating model...")
    
    model = ConditionalLSTM(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        condition_dim=CONFIG['condition_dim'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------
    print("[5/6] Training model...")
    print()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # -------------------------------------------------------------------------
    # EVALUATE
    # -------------------------------------------------------------------------
    print("[6/6] Evaluating model...")
    
    test_loss, test_predictions, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"  Test Loss (MSE): {test_loss:.6f}")
    print()
    
    # Calculate detailed metrics
    metrics = calculate_metrics(test_predictions, test_targets, normalizer)
    
    print("Per-Metric Performance:")
    print("-" * 60)
    for metric_name in METRIC_NAMES:
        m = metrics[metric_name]
        print(f"{metric_name:20s} | MSE: {m['mse']:10.4f} | "
              f"MAE: {m['mae']:10.4f} | RMSE: {m['rmse']:10.4f}")
    print("-" * 60)
    print(f"{'Overall':20s} | MSE: {metrics['overall']['mse']:10.4f} | "
          f"MAE: {metrics['overall']['mae']:10.4f} | "
          f"RMSE: {metrics['overall']['rmse']:10.4f}")
    print()
    
    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------
    print("Saving results...")
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'config': CONFIG,
        'metrics': metrics,
        'training_time': training_time,
        'best_val_loss': best_val_loss,
        'final_epoch': len(train_losses)
    }, output_dir / 'lstm_baseline_model.pt')
    print("  ✓ Saved model")
    
    # Save metrics as JSON
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  ✓ Saved metrics")
    
    # Save training history
    np.save(output_dir / 'train_losses.npy', train_losses)
    np.save(output_dir / 'val_losses.npy', val_losses)
    print("  ✓ Saved training history")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, output_dir)
    
    # Plot sample predictions
    plot_sample_predictions(test_predictions, test_targets, normalizer, output_dir)
    
    print()
    print("=" * 80)
    print("LSTM BASELINE TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Training time: {training_time/60:.2f} minutes")
    print()

if __name__ == '__main__':
    main()