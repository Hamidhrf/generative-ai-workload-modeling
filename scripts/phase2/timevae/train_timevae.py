"""
TimeVAE Training Script - Organized Structure
For: Master's Thesis - Generative AI Workload Modeling
Location: scripts/phase2/timevae/train_timevae.py

"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import from same directory
from timevae_architecture import (
    ConditionalTimeVAE,
    TimeVAEConfig,
    compute_loss
)
from data_handler import (
    load_pod_traces,
    VMConfigHandler,
    DataNormalizer
)


class PodTraceDataset(Dataset):
    """Dataset for pod-level traces with conditioning"""
    
    def __init__(self, traces, metadata, vm_handler):
        """
        Args:
            traces: (N, 715, 15) - pod traces (ALREADY NORMALIZED)
            metadata: list of dicts with 'workload' and 'replica_count'
            vm_handler: VMConfigHandler instance
        """
        self.traces = torch.FloatTensor(traces)
        self.metadata = metadata
        self.vm_handler = vm_handler
        
        # Workload encoding
        self.workload_to_idx = {
            'distilbert': 0,
            'resnet50': 1,
            'whisper': 2
        }
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        """
        Returns:
            trace: (715, 15) - normalized
            condition: (7,) [r, workload_3d, vm_3d]
        """
        trace = self.traces[idx]
        meta = self.metadata[idx]
        
        # Build conditioning vector
        condition = self._build_condition(meta)
        
        return trace, condition
    
    def _build_condition(self, meta):
        """
        Build 7D conditioning vector:
        [replica_count, workload_onehot(3), vm_config(3)]
        """
        # 1. Replica count (normalized)
        r_norm = meta['replica_count'] / 10.0
        
        # 2. Workload one-hot
        workload_onehot = torch.zeros(3)
        workload_idx = self.workload_to_idx[meta['workload']]
        workload_onehot[workload_idx] = 1.0
        
        # 3. VM config (normalized) from handler
        vm_vector = self.vm_handler.get_conditioning_vector()
        vm_tensor = torch.from_numpy(vm_vector).float()
        
        # Combine
        condition = torch.cat([
            torch.tensor([r_norm]),
            workload_onehot,
            vm_tensor
        ])
        
        return condition


class TimeVAETrainer:
    """Training manager for TimeVAE"""
    
    def __init__(self, config, output_dir=None):
        if output_dir is None:
            # Use organized output directory
            output_dir = project_root / 'outputs' / 'phase2_timevae'
        
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model
        self.model = ConditionalTimeVAE(config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=50
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        for batch_idx, (traces, conditions) in enumerate(dataloader):
            traces = traces.to(self.device)
            conditions = conditions.to(self.device)
            
            # Forward pass
            recon, mu, logvar, components = self.model(traces, conditions)
            
            # Compute loss
            loss, recon_loss, kl_loss = compute_loss(
                recon, traces, mu, logvar, self.config
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        # Average losses
        n_batches = len(dataloader)
        return {
            'loss': epoch_loss / n_batches,
            'recon': epoch_recon / n_batches,
            'kl': epoch_kl / n_batches
        }
    
    def validate(self, dataloader):
        """Validate on validation set"""
        self.model.eval()
        
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        
        with torch.no_grad():
            for traces, conditions in dataloader:
                traces = traces.to(self.device)
                conditions = conditions.to(self.device)
                
                # Forward pass
                recon, mu, logvar, components = self.model(traces, conditions)
                
                # Compute loss
                loss, recon_loss, kl_loss = compute_loss(
                    recon, traces, mu, logvar, self.config
                )
                
                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
        
        n_batches = len(dataloader)
        return {
            'loss': val_loss / n_batches,
            'recon': val_recon / n_batches,
            'kl': val_kl / n_batches
        }
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"\nStarting training for {self.config.max_epochs} epochs")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"Target: Beat LSTM baseline MSE 0.006197")
        print(f"Goal: < 0.005 MSE\n")
        
        for epoch in range(self.config.max_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['val_kl'].append(val_metrics['kl'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check for improvement
            if val_metrics['recon'] < self.best_val_loss:
                self.best_val_loss = val_metrics['recon']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                
                improvement = "IMPROVED"
                if val_metrics['recon'] < 0.005:
                    improvement = "GOAL REACHED"
            else:
                self.patience_counter += 1
                improvement = f"No improvement ({self.patience_counter}/{self.config.patience})"
            
            # Print progress every 10 epochs
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {train_metrics['loss']:.6f} | "
                      f"Val Recon: {val_metrics['recon']:.6f} | "
                      f"Best: {self.best_val_loss:.6f} | "
                      f"{improvement}")
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation MSE: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
                break
        
        # Final summary
        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation MSE: {self.best_val_loss:.6f}")
        print(f"LSTM baseline MSE: 0.006197")
        
        if self.best_val_loss < 0.006197:
            improvement_pct = (0.006197 - self.best_val_loss) / 0.006197 * 100
            print(f"IMPROVEMENT: {improvement_pct:.1f}% better than baseline")
            if self.best_val_loss < 0.005:
                print("GOAL REACHED: < 0.005 MSE")
        else:
            print("Did not beat baseline (consider hyperparameter tuning)")
        
        # Save final results
        self.save_training_history()
        self.plot_training_curves()
    
    def save_checkpoint(self, filename, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'history': self.history
        }
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
    
    def save_training_history(self):
        """Save training history as JSON"""
        history_path = self.output_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(self.history['train_loss'], label='Train', alpha=0.7)
        ax.plot(self.history['val_loss'], label='Validation', alpha=0.7)
        ax.axhline(y=0.006197, color='r', linestyle='--', label='LSTM Baseline')
        ax.axhline(y=0.005, color='g', linestyle='--', label='Target Goal')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reconstruction loss
        ax = axes[0, 1]
        ax.plot(self.history['train_recon'], label='Train', alpha=0.7)
        ax.plot(self.history['val_recon'], label='Validation', alpha=0.7)
        ax.axhline(y=0.006197, color='r', linestyle='--', label='LSTM Baseline')
        ax.axhline(y=0.005, color='g', linestyle='--', label='Target Goal')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction MSE')
        ax.set_title('Reconstruction Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # KL divergence
        ax = axes[1, 0]
        ax.plot(self.history['train_kl'], label='Train', alpha=0.7)
        ax.plot(self.history['val_kl'], label='Validation', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 1]
        ax.plot(self.history['lr'], alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        
        plt.close()


def create_data_loaders(traces, metadata, vm_handler, train_ratio=0.8, batch_size=8):
    """Create train/val data loaders"""
    # Train/val split
    n_samples = len(traces)
    n_train = int(n_samples * train_ratio)
    
    # Random shuffle
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_dataset = PodTraceDataset(
        traces[train_indices],
        [metadata[i] for i in train_indices],
        vm_handler
    )
    
    val_dataset = PodTraceDataset(
        traces[val_indices],
        [metadata[i] for i in val_indices],
        vm_handler
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nData split:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


def main():
    """Main training script"""
    print("=" * 60)
    print("TimeVAE Training - Phase 2")
    print("Master's Thesis: Generative AI Workload Modeling")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data (already normalized from Phase 1)
    traces, metadata = load_pod_traces()
    
    # Load VM config handler
    vm_handler = VMConfigHandler()
    
    # Create config
    config = TimeVAEConfig()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        traces, metadata, vm_handler,
        batch_size=config.batch_size
    )
    
    # Create trainer
    trainer = TimeVAETrainer(config)
    
    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\nTraining complete. Next steps:")
    print("1. Check outputs/phase2_timevae/training_curves.png")
    print("2. If MSE < 0.005: Proceed to generation")
    print("3. If MSE > 0.006: Tune hyperparameters")
    print("4. Run: python scripts/phase2/timevae/evaluate_timevae.py")


if __name__ == "__main__":
    main()