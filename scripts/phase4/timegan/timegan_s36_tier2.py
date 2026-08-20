"""
Adapted from timegan_s36_tier3.py for H100 Tier 2 MIG data. Only
DATA_COMBINED, NORM_COMBINED, and the two output path prefixes
(models/phase4/timegan_s36_tier2/, outputs/phase4/timegan_s36_tier2/)
changed. S27_HYPERPARAMS, architecture, seeds, and epoch counts
are untouched -- frozen per the S36 Tier 2 retrain plan.

TimeGAN S36 - S34 + Spectral Normalization
===========================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

S36 = S34 Architecture + Spectral Normalization
================================================

ABLATION STUDY: Does spectral normalization improve stability?
- S34 baseline: Mean VR = 1.205 (176 traces, no augmentation)
- S36: Same architecture + spectral norm in discriminator

SPECTRAL NORMALIZATION:
- Constrains discriminator Lipschitz constant
- Prevents discriminator from overpowering generator
- Reduces mode collapse, improves gradient flow
- Proven technique from Miyato et al. (2018)

RATIONALE:
- S35 (data augmentation) failed → try architectural improvement
- Small dataset (176 traces) needs stable training
- Spectral norm addresses stability, not data quantity

USAGE:
------
    python timegan_s36.py --workloads bert gpt2 resnet152 whisper yolo
    python timegan_s36.py --workloads whisper  # Train only Whisper

OUTPUT:
-------
    models/phase4/timegan_s36/s36_{workload}_vr{X}_fm{Y}/
    outputs/phase4/timegan_s36/s36_summary.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Data paths (Tier 2 H100 MIG)
DATA_COMBINED = Path("data/processed/tier2/unified/combined_dataset.npz")
NORM_COMBINED = Path("data/processed/tier2/unified/combined_normalization.json")

# ALL workloads (including Whisper)
WORKLOADS_ALL = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
VALID_WORKLOAD_IDS = [0, 1, 2, 3, 4]  # All 5 workloads
VALID_WORKLOAD_NAMES = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

ALL_METRICS = [
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
    "pod_latency_avg", "pod_throughput",
    "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
    "gpu_power_watts", "gpu_temperature",
]

# Drop config per workload - ALL 5 workloads have SAME 7 metrics
DROP_PER_WORKLOAD = {
    "bert":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "gpt2":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "resnet152": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "whisper":   {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "yolo":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
}

# S27 hyperparameters (proven optimal per workload) - IDENTICAL TO S34
S27_HYPERPARAMS = {
    "bert": {
        "lambda_var_reg": 0.5,
        "lambda_fm_stat": 1.5,
        "lambda_smooth": 0.05,
        "n_disc_steps": 2,
    },
    "gpt2": {
        "lambda_var_reg": 0.3,
        "lambda_fm_stat": 2.0,
        "lambda_smooth": 0.1,
        "n_disc_steps": 2,
    },
    "resnet152": {
        "lambda_var_reg": 0.4,
        "lambda_fm_stat": 1.0,
        "lambda_smooth": 0.1,
        "n_disc_steps": 2,
    },
    "whisper": {
        "lambda_var_reg": 0.5,
        "lambda_fm_stat": 0.5,
        "lambda_smooth": 0.05,
        "n_disc_steps": 1,
    },
    "yolo": {
        "lambda_var_reg": 0.3,
        "lambda_fm_stat": 1.2,
        "lambda_smooth": 0.1,
        "n_disc_steps": 2,
    },
}

# Phase configuration - IDENTICAL TO S34
DEFAULT_PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
N_PHASES = 6
SEGMENT_LEN = 120

BASE_CONFIG = {
    "adversarial": "wgan",
    "lambda_adv_warmup": 0.0,
    "lambda_adv_post": 1.0,
    "lambda_fm": 0.01,
    "lambda_gp": 10.0,
}

GEN_CFG = {
    "hidden_dim": 128,
    "num_layers": 2,
    "latent_dim": 64,
    "replica_embed_dim": 16,
    "phase_embed_dim": 8,
    "dropout": 0.1,
    "grad_clip": 1.0,
    "seed": 42,
}

DISC_CFG = {
    "hidden_dim": 32,
    "num_layers": 1,
    "dropout": 0.0,
}


def load_workload_data(workload_name):
    """Load data for a single workload."""
    data = np.load(DATA_COMBINED, allow_pickle=True)
    with open(NORM_COMBINED) as f:
        norm = json.load(f)
    
    # Get workload ID
    workload_id = WORKLOADS_ALL.index(workload_name)
    
    # Filter for this workload only
    all_traces = data["traces"]
    all_rc = data["replica_counts"]
    all_wl = data["workload_ids"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    
    wl_mask = (all_wl == workload_id)
    traces = all_traces[wl_mask]
    rc = all_rc[wl_mask]
    
    # Get kept metrics for this workload
    drop = DROP_PER_WORKLOAD.get(workload_name, set())
    kept_idx = [i for i, m in enumerate(ALL_METRICS) if m not in drop]
    kept_names = [ALL_METRICS[i] for i in kept_idx]
    
    traces = traces[:, :, kept_idx].astype(np.float32)
    
    # Remap train/val indices
    orig_to_new = {}
    new_idx = 0
    for old_idx in range(len(all_traces)):
        if wl_mask[old_idx]:
            orig_to_new[old_idx] = new_idx
            new_idx += 1
    
    new_train_idx = np.array([orig_to_new[i] for i in train_idx if i in orig_to_new])
    new_val_idx = np.array([orig_to_new[i] for i in val_idx if i in orig_to_new])
    
    return traces, rc, new_train_idx, new_val_idx, norm[workload_name], kept_names


def segment_traces(traces, rc, boundaries, n_phases, seg_len):
    """Segment traces into phases."""
    N, T, M = traces.shape
    bounds = list(boundaries) + [T]
    segments = []
    seg_rc = []
    seg_ph = []
    seg_ti = []
    
    for i in range(N):
        for ph in range(n_phases):
            t_start = bounds[ph]
            t_end = bounds[ph + 1]
            raw_seg = traces[i, t_start:t_end, :]
            span = t_end - t_start
            
            if span == seg_len:
                resampled = raw_seg
            else:
                src_t = np.linspace(0, 1, span)
                dst_t = np.linspace(0, 1, seg_len)
                resampled = np.stack(
                    [np.interp(dst_t, src_t, raw_seg[:, m]) for m in range(M)],
                    axis=1
                ).astype(np.float32)
            
            segments.append(resampled)
            seg_rc.append(int(rc[i]))
            seg_ph.append(ph)
            seg_ti.append(i)
    
    return (np.array(segments), np.array(seg_rc, dtype=np.int32),
            np.array(seg_ph, dtype=np.int32), np.array(seg_ti, dtype=np.int32))


class SegmentDataset(Dataset):
    def __init__(self, segments, seg_rc, seg_ph, indices, jitter=0):
        self.segs = segments
        self.rc = seg_rc
        self.ph = seg_ph
        self.indices = indices
        self.jitter = jitter
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        seg = torch.tensor(self.segs[idx], dtype=torch.float32)
        r_val = int(self.rc[idx])
        ph = int(self.ph[idx])
        
        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            seg = torch.roll(seg, shift, dims=0)
        
        r_norm = torch.tensor((r_val - 1.0) / 9.0, dtype=torch.float32)
        phase_idx = torch.tensor(ph, dtype=torch.long)
        
        return seg, torch.tensor(r_val, dtype=torch.long), r_norm, phase_idx


class GeneratorSeg(nn.Module):
    """S27 generator architecture - IDENTICAL TO S34."""
    def __init__(self, seg_len, n_metrics, cfg):
        super().__init__()
        self.seg_len = seg_len
        self.n_metrics = n_metrics
        hidden = cfg["hidden_dim"]
        n_layers = cfg["num_layers"]
        latent_dim = cfg["latent_dim"]
        r_emb_dim = cfg["replica_embed_dim"]
        ph_emb_dim = cfg["phase_embed_dim"]
        dropout = cfg["dropout"] if n_layers > 1 else 0.0
        
        self.r_embed = nn.Sequential(nn.Linear(1, r_emb_dim), nn.Tanh())
        self.ph_embed = nn.Embedding(N_PHASES + 1, ph_emb_dim)
        
        init_in = latent_dim + r_emb_dim + ph_emb_dim
        self.h_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())
        self.c_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())
        
        dec_in_dim = latent_dim + r_emb_dim + ph_emb_dim
        self.dec_rnn = nn.LSTM(dec_in_dim, hidden, n_layers,
                               batch_first=True, dropout=dropout)
        
        self.out_fc = nn.Linear(hidden, n_metrics)
        self.out_act = nn.Sigmoid()
        
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden
    
    def forward(self, r_norm, phase_idx, z=None):
        B = r_norm.shape[0]
        T = self.seg_len
        device = r_norm.device
        
        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)
        
        r_emb = self.r_embed(r_norm.unsqueeze(-1))
        ph_emb = self.ph_embed(phase_idx)
        
        zrp = torch.cat([z, r_emb, ph_emb], dim=-1)
        h0 = self.h_init(zrp)
        c0 = self.c_init(zrp)
        h0 = h0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = c0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        
        z_exp = z.unsqueeze(1).expand(-1, T, -1)
        r_exp = r_emb.unsqueeze(1).expand(-1, T, -1)
        ph_exp = ph_emb.unsqueeze(1).expand(-1, T, -1)
        dec_in = torch.cat([z_exp, r_exp, ph_exp], dim=-1)
        
        dec_out, _ = self.dec_rnn(dec_in, (h0, c0))
        return self.out_act(self.out_fc(dec_out))
    
    @torch.no_grad()
    def generate_trace(self, r_norm_val, device, n_samples=1):
        """Generate full trace."""
        self.eval()
        segments = []
        for ph in range(N_PHASES):
            r_norm = torch.full((n_samples,), r_norm_val, dtype=torch.float32, device=device)
            ph_idx = torch.full((n_samples,), ph, dtype=torch.long, device=device)
            seg = self(r_norm, ph_idx)
            segments.append(seg.cpu().numpy())
        full_trace = np.concatenate(segments, axis=1)
        return full_trace[:, :715, :]


class DiscriminatorSpectral(nn.Module):
    """S36 discriminator with spectral normalization - NEW!"""
    def __init__(self, n_metrics, cfg):
        super().__init__()
        hidden = cfg["hidden_dim"]
        n_layers = cfg["num_layers"]
        dropout = cfg["dropout"] if n_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(n_metrics, hidden, n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Apply spectral norm to classifier layers using PyTorch's built-in
        fc1 = nn.utils.spectral_norm(nn.Linear(hidden * 2 + 1, max(hidden, 16)))
        fc2 = nn.utils.spectral_norm(nn.Linear(max(hidden, 16), 1))
        
        self.classifier = nn.Sequential(
            fc1,
            nn.LeakyReLU(0.2),
            fc2
        )
    
    def forward(self, x, r_norm=None):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        if r_norm is not None:
            h = torch.cat([h, r_norm.unsqueeze(1)], dim=1)
        else:
            h = torch.cat([h, torch.zeros(h.shape[0], 1, device=h.device)], dim=1)
        
        return self.classifier(h)
    
    def get_features(self, x):
        _, (h_n, _) = self.lstm(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)


def gradient_penalty(disc, real, fake, r_norm, device, lam=10.0):
    B = real.shape[0]
    eps = torch.rand(B, 1, 1, device=device)
    mid = (eps * real + (1 - eps) * fake).requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        d = disc(mid, r_norm)
    grad = torch.autograd.grad(d.sum(), mid, create_graph=True)[0]
    return lam * ((grad.norm(2, dim=(1, 2)) - 1) ** 2).mean()


def precompute_fm_targets(segments, seg_rc, seg_ph, device):
    """Precompute FM targets per (phase, replica)."""
    targets = {}
    for ph_val in set(seg_ph.tolist()):
        for r_val in set(seg_rc.tolist()):
            mask = (seg_rc == r_val) & (seg_ph == ph_val)
            if mask.sum() < 2:
                continue
            group = segments[mask]
            targets[(int(ph_val), int(r_val))] = {
                'mean': torch.tensor(group.mean(axis=(0, 1)), dtype=torch.float32).to(device),
                'std': torch.tensor(group.std(axis=(0, 1)), dtype=torch.float32).to(device) + 1e-8,
            }
    return targets


def compute_fm_stat(fake, r_int, ph_idx, fm_targets, device):
    """Compute FM loss."""
    loss = torch.tensor(0.0, device=device)
    count = 0
    
    for ph_val in torch.unique(ph_idx):
        for r_val in torch.unique(r_int):
            key = (int(ph_val.item()), int(r_val.item()))
            if key not in fm_targets:
                continue
            mask = (r_int == r_val) & (ph_idx == ph_val)
            if mask.sum() < 1:
                continue
            
            fake_seg = fake[mask]
            t_mean = fm_targets[key]['mean']
            t_std = fm_targets[key]['std']
            
            fake_mean = fake_seg.mean(dim=(0, 1))
            scale = (t_mean.abs() + t_std + 1e-6).detach()
            loss += ((fake_mean - t_mean.detach()) / scale).pow(2).mean()
            
            fake_std = fake_seg.std(dim=(0, 1)) + 1e-8
            log_std = torch.log(fake_std / t_std.detach())
            loss += (log_std ** 2).mean()
            
            count += 1
    
    return loss / max(count, 1)


def train_workload(workload_name, hyperparams, device):
    """Train S36 model for single workload."""
    print(f"\n{'='*80}")
    print(f"S36 TRAINING: {workload_name.upper()}")
    print(f"{'='*80}")
    
    # Load data
    traces, rc, train_idx, val_idx, norm, kept_names = load_workload_data(workload_name)
    n_metrics = len(kept_names)
    
    print(f"Data: {len(traces)} pods, {len(train_idx)} train, {len(val_idx)} val")
    print(f"Metrics ({n_metrics}): {kept_names}")
    
    # Segment
    all_segs, all_rc, all_ph, all_ti = segment_traces(
        traces, rc, DEFAULT_PHASE_BOUNDARIES, N_PHASES, SEGMENT_LEN)
    
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    seg_train = np.where(np.isin(all_ti, list(train_set)))[0]
    seg_val = np.where(np.isin(all_ti, list(val_set)))[0]
    
    print(f"Segments: train={len(seg_train)}, val={len(seg_val)}")
    
    # Datasets
    ds_train = SegmentDataset(all_segs[seg_train], all_rc[seg_train],
                               all_ph[seg_train], np.arange(len(seg_train)), jitter=4)
    ds_val = SegmentDataset(all_segs[seg_val], all_rc[seg_val],
                             all_ph[seg_val], np.arange(len(seg_val)), jitter=0)
    
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)
    
    # Models - S36 uses spectral norm discriminator
    generator = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    discriminator = DiscriminatorSpectral(n_metrics, DISC_CFG).to(device)
    
    print(f"Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"Discriminator: Spectral Normalization ENABLED")
    print(f"Hyperparams: {hyperparams}")
    
    # FM targets
    fm_targets = precompute_fm_targets(all_segs[seg_train], all_rc[seg_train],
                                        all_ph[seg_train], device)
    
    # Train
    opt_G = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.0, 0.9), weight_decay=1e-5)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, 0.9), weight_decay=1e-5)
    
    warmup = 20
    adv = 150
    total = warmup + adv
    best_val = float("inf")
    best_G = None
    
    for epoch in range(1, total + 1):
        in_warmup = (epoch <= warmup)
        n_disc = 0 if in_warmup else hyperparams["n_disc_steps"]
        
        generator.train()
        discriminator.train()
        
        for target, r_int, r_norm, ph_idx in dl_train:
            target = target.to(device)
            r_norm = r_norm.to(device)
            r_int = r_int.to(device)
            ph_idx = ph_idx.to(device)
            
            # Discriminator
            if n_disc > 0:
                with torch.no_grad():
                    fake_d = generator(r_norm, ph_idx)
                for _ in range(n_disc):
                    opt_D.zero_grad()
                    sr = discriminator(target, r_norm)
                    sf = discriminator(fake_d, r_norm)
                    wd = sr.mean() - sf.mean()
                    ld = -wd + gradient_penalty(discriminator, target, fake_d, r_norm, device)
                    ld.backward()
                    nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
                    opt_D.step()
            
            # Generator
            opt_G.zero_grad()
            fake = generator(r_norm, ph_idx)
            loss_G = torch.tensor(0.0, device=device)
            
            if not in_warmup:
                loss_G += -discriminator(fake, r_norm).mean()
            
            if hyperparams["lambda_var_reg"] > 0:
                std_real = target.std(dim=[0, 1]).detach()
                std_fake = fake.std(dim=[0, 1])
                log_ratio = torch.log((std_fake + 1e-8) / (std_real + 1e-8))
                loss_G += hyperparams["lambda_var_reg"] * (log_ratio ** 2).mean()
            
            if hyperparams["lambda_fm_stat"] > 0:
                loss_G += hyperparams["lambda_fm_stat"] * compute_fm_stat(
                    fake, r_int, ph_idx, fm_targets, device)
            
            if hyperparams["lambda_smooth"] > 0 and not in_warmup:
                diff = fake[:, 1:, :] - fake[:, :-1, :]
                loss_G += hyperparams["lambda_smooth"] * (diff ** 2).mean()
            
            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), GEN_CFG["grad_clip"])
            opt_G.step()
        
        # Validation
        generator.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for target_v, r_int_v, r_norm_v, ph_idx_v in dl_val:
                target_v = target_v.to(device)
                r_int_v = r_int_v.to(device)
                r_norm_v = r_norm_v.to(device)
                ph_idx_v = ph_idx_v.to(device)
                fake_v = generator(r_norm_v, ph_idx_v)
                val_loss += compute_fm_stat(fake_v, r_int_v, ph_idx_v, fm_targets, device).item()
                n_val += 1
        val_loss /= max(n_val, 1)
        
        if val_loss < best_val:
            best_val = val_loss
            best_G = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
        
        if epoch % 10 == 1 or epoch == total:
            print(f"  Epoch {epoch:3d} val={val_loss:.5f}")
    
    generator.load_state_dict(best_G)
    return generator, best_val


def main():
    parser = argparse.ArgumentParser(description="S36 - S34 + Spectral Normalization")
    parser.add_argument("--workloads", nargs='+', type=str,
                        default=VALID_WORKLOAD_NAMES,
                        help="Workloads to train (default: all 5)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("S36: S34 + SPECTRAL NORMALIZATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Workloads: {args.workloads}")
    print(f"Architecture: S34 + Spectral Norm in Discriminator")
    print(f"Dataset: Original 176 traces (NO augmentation)")
    print(f"Improvement: Stabilized discriminator training")
    print("="*80)
    
    results = {}
    
    for wl in args.workloads:
        if wl not in VALID_WORKLOAD_NAMES:
            print(f"Skipping {wl} (not in valid list)")
            continue
        
        hyperparams = S27_HYPERPARAMS[wl]
        
        t0 = time.time()
        generator, best_val = train_workload(wl, hyperparams, device)
        elapsed = time.time() - t0
        
        # Save model
        tag = f"s36_{wl}_vr{int(hyperparams['lambda_var_reg']*10):02d}_fm{int(hyperparams['lambda_fm_stat']*10):02d}"
        model_dir = Path(f"models/phase4/timegan_s36_tier2/{tag}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(generator.state_dict(), model_dir / "generator.pt")
        
        results[wl] = {
            "best_val": float(best_val),
            "elapsed_s": elapsed,
            "model_path": str(model_dir),
        }
        
        print(f"\n{wl.upper()} complete: {elapsed:.1f}s, val={best_val:.5f}")
        print(f"Saved: {model_dir}")
    
    # Save summary
    out_dir = Path("outputs/phase4/timegan_s36_tier2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "s36_summary.json", "w") as f:
        json.dump({
            "stage": "s36",
            "description": "S34 architecture + Spectral Normalization in discriminator",
            "workloads": args.workloads,
            "improvement": "spectral_normalization",
            "baseline": "s34",
            "results": results,
        }, f, indent=2)
    
    print("\n"+"="*80)
    print("S36 TRAINING COMPLETE")
    print("="*80)
    print(f"Summary: {out_dir / 's36_summary.json'}")
    print("\nNext: Evaluate with eval_s36.py to compare vs S34")


if __name__ == "__main__":
    main()