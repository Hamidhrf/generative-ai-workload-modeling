"""
TimeVAE Architecture Specification
For: Master's Thesis - Generative AI Workload Modeling
Date: January 26, 2026

Data Specifications:
- Input shape: (batch, 715, 15)  # timesteps, metrics
- Conditioning: 7D vector [replica_count, workload_3d, vm_config_3d]
- Target: Beat LSTM baseline MSE 0.006197

Architecture Design Rationale:
- Encoder: Captures temporal patterns in pod traces
- Latent space: Compressed representation conditioned on system config
- Decoder: Reconstructs traces with interpretable components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeVAEConfig:
    """Configuration for TimeVAE model adapted to thesis requirements"""
    
    # Data dimensions
    sequence_length = 715      # Timesteps per trace
    n_features = 15           # Metrics (CPU, GPU, memory, latency, etc.)
    
    # Conditioning dimensions
    conditioning_dim = 7      # [r, workload_3d, vm_3d]
    
    # Architecture parameters
    # Start conservative, tune if needed
    hidden_dim = 128          # RNN hidden dimension   
    latent_dim = 32           # Latent space dimension
    num_layers = 2            # Number of RNN layers.  
    
    # Component decomposition (TimeVAE's key feature)
    level_dim = 16            # Baseline resource usage
    trend_dim = 8             # Gradual changes (contention buildup)
    seasonality_dim = 8       # Periodic patterns (if any)
    
    # Training parameters
    learning_rate = 1e-3
    batch_size = 8            # Small due to limited data (60 samples)
    max_epochs = 2000         # Allow long training
    
    # Loss weights (tune these)
    reconstruction_weight = 3.0
    kl_weight = 1.0   # Lower = less smoothing
    
    # Regularization
    dropout = 0.2
    
    # Early stopping
    patience = 200
    min_delta = 1e-5
    
    @staticmethod
    def validate():
        """Sanity checks"""
        assert TimeVAEConfig.latent_dim == (
            TimeVAEConfig.level_dim + 
            TimeVAEConfig.trend_dim + 
            TimeVAEConfig.seasonality_dim
        ), "Latent dim must equal sum of component dims"


class ConditionalEncoder(nn.Module):
    """
    Encoder: Maps (time_series, condition) -> latent distribution
    
    Key insight: Conditioning helps encoder learn that pod behavior
    depends on replica count and VM specs.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Condition processing
        # Maps 7D condition to same dimension as hidden state
        self.condition_proj = nn.Sequential(
            nn.Linear(config.conditioning_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.hidden_dim)
        )
        
        # Main RNN encoder
        # Input: metrics (15) + projected condition (128)
        self.rnn = nn.GRU(
            input_size=config.n_features,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
    
    def forward(self, x, condition):
        """
        Args:
            x: (batch, 715, 15) - pod traces
            condition: (batch, 7) - [r, workload, vm_config]
        
        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        batch_size = x.size(0)
        
        # Process condition
        cond_emb = self.condition_proj(condition)  # (batch, hidden_dim)
        
        # Initialize hidden state with condition
        h0 = cond_emb.unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        
        # Encode sequence
        _, hidden = self.rnn(x, h0)  # hidden: (num_layers, batch, hidden_dim)
        
        # Use final hidden state
        h_final = hidden[-1]  # (batch, hidden_dim)
        
        # Map to latent distribution
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        
        return mu, logvar


class InterpretableDecoder(nn.Module):
    """
    Decoder: Maps (latent_code, condition) -> reconstructed time series
    
    Key feature: Decomposes into level + trend + seasonality
    This makes generated traces interpretable.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Condition processing
        self.condition_proj = nn.Sequential(
            nn.Linear(config.conditioning_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.hidden_dim)
        )
        
        # Component generators
        # Level: Baseline resource usage
        self.level_decoder = nn.Linear(
            config.level_dim + config.hidden_dim, 
            config.n_features
        )
        
        # Trend: How resources change over time
        self.trend_rnn = nn.GRU(
            input_size=config.trend_dim + config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        self.trend_proj = nn.Linear(config.hidden_dim // 2, config.n_features)
        
        # Seasonality: Periodic patterns (may be weak in your data)
        self.season_rnn = nn.GRU(
            input_size=config.seasonality_dim + config.hidden_dim,
            hidden_size=config.hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        self.season_proj = nn.Linear(config.hidden_dim // 2, config.n_features)
    
    def forward(self, z, condition):
        """
        Args:
            z: (batch, latent_dim) - latent code
            condition: (batch, 7) - conditioning vector
        
        Returns:
            recon: (batch, 715, 15) - reconstructed traces
            components: dict with level, trend, seasonality
        """
        batch_size = z.size(0)
        seq_len = self.config.sequence_length
        
        # Process condition
        cond_emb = self.condition_proj(condition)  # (batch, hidden_dim)
        
        # Split latent code into components
        level_z = z[:, :self.config.level_dim]
        trend_z = z[:, self.config.level_dim:self.config.level_dim + self.config.trend_dim]
        season_z = z[:, self.config.level_dim + self.config.trend_dim:]
        
        # 1. LEVEL: Constant baseline
        level_input = torch.cat([level_z, cond_emb], dim=1)
        level = self.level_decoder(level_input)  # (batch, n_features)
        level = level.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, 715, 15)
        
        # 2. TREND: Gradual changes
        trend_input = torch.cat([
            trend_z.unsqueeze(1).repeat(1, seq_len, 1),
            cond_emb.unsqueeze(1).repeat(1, seq_len, 1)
        ], dim=2)  # (batch, 715, trend_dim + hidden_dim)
        
        trend_hidden, _ = self.trend_rnn(trend_input)
        trend = self.trend_proj(trend_hidden)  # (batch, 715, n_features)
        
        # 3. SEASONALITY: Periodic patterns
        season_input = torch.cat([
            season_z.unsqueeze(1).repeat(1, seq_len, 1),
            cond_emb.unsqueeze(1).repeat(1, seq_len, 1)
        ], dim=2)
        
        season_hidden, _ = self.season_rnn(season_input)
        season = self.season_proj(season_hidden)  # (batch, 715, n_features)
        
        # Combine components
        recon = level + trend + season
        
        components = {
            'level': level,
            'trend': trend,
            'seasonality': season
        }
        
        return recon, components


class ConditionalTimeVAE(nn.Module):
    """
    Complete TimeVAE model with conditioning on replica count + workload + VM config
    
    This is your main model class.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = ConditionalEncoder(config)
        self.decoder = InterpretableDecoder(config)
    
    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, condition):
        """
        Forward pass
        
        Args:
            x: (batch, 715, 15) - input traces
            condition: (batch, 7) - conditioning vector
        
        Returns:
            recon: (batch, 715, 15) - reconstructed traces
            mu, logvar: latent distribution parameters
            components: interpretable decomposition
        """
        # Encode
        mu, logvar = self.encoder(x, condition)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon, components = self.decoder(z, condition)
        
        return recon, mu, logvar, components
    
    def generate(self, condition, num_samples=1):
        """
        Generate synthetic traces
        
        Args:
            condition: (num_samples, 7) or (7,)
            num_samples: int
        
        Returns:
            synthetic_traces: (num_samples, 715, 15)
        """
        self.eval()
        with torch.no_grad():
            # Handle single condition
            if condition.dim() == 1:
                condition = condition.unsqueeze(0).repeat(num_samples, 1)
            
            # Sample from prior
            z = torch.randn(num_samples, self.config.latent_dim).to(condition.device)
            
            # Decode
            synthetic, _ = self.decoder(z, condition)
            
            return synthetic


def compute_loss(recon, x, mu, logvar, config):
    """
    TimeVAE loss function
    
    Components:
    1. Reconstruction loss: How well we reconstruct input
    2. KL divergence: Regularization of latent space
    """
    recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')  
    recon_loss = recon_loss * config.reconstruction_weight
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  
    kl_loss = kl_loss * config.kl_weight
    
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss / config.reconstruction_weight, kl_loss / config.kl_weight


# Example usage
if __name__ == "__main__":
    # Validate config
    TimeVAEConfig.validate()
    
    # Create model
    config = TimeVAEConfig()
    model = ConditionalTimeVAE(config)
    
    print("TimeVAE Architecture:")
    print(f"  Input: (batch, {config.sequence_length}, {config.n_features})")
    print(f"  Conditioning: (batch, {config.conditioning_dim})")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Components: level={config.level_dim}, trend={config.trend_dim}, season={config.seasonality_dim}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, config.sequence_length, config.n_features)
    condition = torch.randn(batch_size, config.conditioning_dim)
    
    recon, mu, logvar, components = model(x, condition)
    print(f"\nForward pass test:")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Latent mu shape: {mu.shape}")
    print(f"  Level shape: {components['level'].shape}")
    print(f"  Trend shape: {components['trend'].shape}")
    print(f"  Seasonality shape: {components['seasonality'].shape}")
    
    # Test generation
    test_condition = torch.tensor([[
        0.5,  # replica_count = 5 (normalized)
        1.0, 0.0, 0.0,  # workload = distilbert
        0.25, 0.375, 0.238  # VM config
    ]])
    
    synthetic = model.generate(test_condition, num_samples=10)
    print(f"\nGeneration test:")
    print(f"  Synthetic traces shape: {synthetic.shape}")
    print(f"  Expected: (10, 715, 15)")