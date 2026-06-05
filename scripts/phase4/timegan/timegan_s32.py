"""
TimeGAN S32 - GPU-Bound Unified Model with Boundary Smoothing
==============================================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

S32 = S29 (Unified) + Whisper Exclusion + Boundary Smoothing
=============================================================

KEY INNOVATIONS:
1. GPU-bound workloads only (BERT, GPT2, ResNet152, YOLO)
2. Boundary smoothing loss: eliminates phase discontinuities
3. Proven S29 architecture + hyperparameters

RATIONALE:
- S29 mean VR = 1.474 (proven quality)
- S31 mean VR = 3.634 (too high, excessive variance)
- S32 targets: 1.5-1.7 mean VR + YOLO > 0.8 + smooth boundaries

ARCHITECTURE:
- GeneratorSegUnified (same as S29)
- 4 workloads (not 5) with remapped IDs
- Boundary smoothing loss added to generator training

BOUNDARY SMOOTHING:
- Penalizes discontinuities at phase transitions
- Loss = MSE(end_of_phase_i, start_of_phase_i+1)
- Applied during training (not post-processing)

EXPECTED RESULTS:
- Mean VR: 1.5-1.7 (realistic, not excessive)
- YOLO VR: 0.85-0.95 (crosses 0.8 threshold)
- Smooth traces at phase boundaries
- Better than S29, realistic unlike S31

USAGE:
------
    python timegan_s32.py
    python timegan_s32.py --boundary-weight 0.2

OUTPUT:
-------
    models/phase4/timegan_s32/s32_gpu_bound_smooth02/
    outputs/phase4/timegan_s32/s32_gpu_bound_smooth02/results.json
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Combined dataset path (same as S29)
DATA_COMBINED = Path("data/processed/phase4/unified/combined_dataset.npz")
NORM_COMBINED = Path("data/processed/phase4/unified/combined_normalization.json")

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
VALID_WORKLOAD_IDS = [0, 1, 2, 4]  # Exclude whisper (ID=3)
VALID_WORKLOAD_NAMES = ["bert", "gpt2", "resnet152", "yolo"]
N_WORKLOADS = 4  # GPU-bound only

ALL_METRICS = [
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
    "pod_latency_avg", "pod_throughput",
    "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
    "gpu_power_watts", "gpu_temperature",
]

# S32 uses S29's drop config (applied to GPU-bound workloads)
DROP_PER_WORKLOAD = {
    "bert":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "gpt2":      {"gpu_memory_total", "gpu_temperature"},
    "resnet152": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "yolo":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
}

# Phase configuration (same as S29)
DEFAULT_PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
N_PHASES = 6
SEGMENT_LEN = 120
FULL_SEQ_LEN = 720

BASE_CONFIG = {
    "adversarial": "wgan",
    "n_disc_steps": 2,
    "lambda_adv_warmup": 0.0,
    "lambda_adv_post": 1.0,
    "lambda_fm": 0.01,
    "lambda_gp": 10.0,
    "preprocessing": "raw_normalized",
}

GEN_CFG = {
    "hidden_dim": 128,
    "num_layers": 2,
    "latent_dim": 64,
    "replica_embed_dim": 16,
    "workload_embed_dim": 16,
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

# S32 hyperparameters (same as S29 proven values)
S32_HYPERPARAMS = {
    "lambda_smooth": 0.07,
    "n_disc_steps": 2,
    "lambda_fm_stat": 1.2,
    "lambda_var_reg": 0.4,
    "lambda_boundary": 0.2,  # NEW: boundary smoothing weight
}

REFERENCE_VR = {
    "bert": {"s29": 1.683, "s27": 0.886},
    "gpt2": {"s29": 2.204, "s27": 0.946},
    "resnet152": {"s29": 1.728, "s27": 0.851},
    "yolo": {"s29": 0.761, "s27": 0.997},
}


def load_combined_data():
    """Load unified dataset and filter GPU-bound workloads."""
    data = np.load(DATA_COMBINED, allow_pickle=True)
    with open(NORM_COMBINED) as f:
        norm = json.load(f)
    
    # Filter GPU-bound workloads
    raw_traces = data["traces"]
    replica_counts = data["replica_counts"]
    workload_ids = data["workload_ids"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    
    # Filter mask
    valid_mask = np.isin(workload_ids, VALID_WORKLOAD_IDS)
    filtered_traces = raw_traces[valid_mask]
    filtered_rc = replica_counts[valid_mask]
    filtered_wl = workload_ids[valid_mask]
    
    # Remap workload IDs: 0,1,2,4 -> 0,1,2,3
    wl_map = {0: 0, 1: 1, 2: 2, 4: 3}
    filtered_wl = np.array([wl_map[w] for w in filtered_wl])
    
    # Remap train/val indices
    original_to_new = {}
    new_idx = 0
    for old_idx in range(len(raw_traces)):
        if valid_mask[old_idx]:
            original_to_new[old_idx] = new_idx
            new_idx += 1
    
    new_train_idx = np.array([original_to_new[i] for i in train_idx if i in original_to_new])
    new_val_idx = np.array([original_to_new[i] for i in val_idx if i in original_to_new])
    
    filtered_data = {
        "traces": filtered_traces,
        "replica_counts": filtered_rc,
        "workload_ids": filtered_wl,
        "train_idx": new_train_idx,
        "val_idx": new_val_idx,
    }
    
    return filtered_data, norm


def get_unified_kept_metrics():
    """Determine kept metrics for GPU-bound workloads."""
    all_kept = set()
    for wl in VALID_WORKLOAD_NAMES:
        drop = DROP_PER_WORKLOAD.get(wl, set())
        for m in ALL_METRICS:
            if m not in drop:
                all_kept.add(m)
    
    kept_idx = []
    kept_names = []
    for i, m in enumerate(ALL_METRICS):
        if m in all_kept:
            kept_idx.append(i)
            kept_names.append(m)
    
    return kept_idx, kept_names


def segment_traces_unified(traces, replica_counts, workload_ids, phase_boundaries, n_phases, seg_len):
    """Segment traces with workload_id tracking."""
    N, T, M = traces.shape
    bounds = list(phase_boundaries) + [T]
    segments = []
    seg_rc = []
    seg_wl = []
    seg_phase_idx = []
    seg_trace_idx = []

    for trace_i in range(N):
        for ph in range(n_phases):
            t_start = bounds[ph]
            t_end = bounds[ph + 1]
            raw_seg = traces[trace_i, t_start:t_end, :]
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
            seg_rc.append(int(replica_counts[trace_i]))
            seg_wl.append(int(workload_ids[trace_i]))
            seg_phase_idx.append(ph)
            seg_trace_idx.append(trace_i)

    return (np.array(segments, dtype=np.float32),
            np.array(seg_rc, dtype=np.int32),
            np.array(seg_wl, dtype=np.int32),
            np.array(seg_phase_idx, dtype=np.int32),
            np.array(seg_trace_idx, dtype=np.int32))


class SegmentDatasetUnified(Dataset):
    """Dataset with workload conditioning."""
    def __init__(self, segments, seg_rc, seg_wl, seg_phase_idx, indices, jitter=0):
        self.segs = segments
        self.rc = seg_rc
        self.wl = seg_wl
        self.ph = seg_phase_idx
        self.indices = indices
        self.jitter = jitter

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        seg = torch.tensor(self.segs[idx], dtype=torch.float32)
        r_val = int(self.rc[idx])
        wl_val = int(self.wl[idx])
        ph = int(self.ph[idx])

        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            seg = torch.roll(seg, shift, dims=0)

        r_norm = torch.tensor((r_val - 1.0) / 9.0, dtype=torch.float32)
        wl_id = torch.tensor(wl_val, dtype=torch.long)
        phase_idx = torch.tensor(ph, dtype=torch.long)

        return seg, torch.tensor(r_val, dtype=torch.long), r_norm, wl_id, phase_idx


class GeneratorSegUnified(nn.Module):
    """Unified generator (same as S29)."""
    def __init__(self, seg_len, n_metrics, cfg):
        super().__init__()
        self.seg_len = seg_len
        self.n_metrics = n_metrics
        hidden = cfg["hidden_dim"]
        n_layers = cfg["num_layers"]
        latent_dim = cfg["latent_dim"]
        r_emb_dim = cfg["replica_embed_dim"]
        wl_emb_dim = cfg["workload_embed_dim"]
        ph_emb_dim = cfg["phase_embed_dim"]
        dropout = cfg["dropout"] if n_layers > 1 else 0.0

        self.r_embed = nn.Sequential(nn.Linear(1, r_emb_dim), nn.Tanh())
        self.wl_embed = nn.Embedding(N_WORKLOADS, wl_emb_dim)
        self.ph_embed = nn.Embedding(N_PHASES + 1, ph_emb_dim)

        init_in = latent_dim + r_emb_dim + wl_emb_dim + ph_emb_dim
        self.h_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())
        self.c_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())

        dec_in_dim = latent_dim + r_emb_dim + wl_emb_dim + ph_emb_dim
        self.dec_rnn = nn.LSTM(dec_in_dim, hidden, n_layers,
                               batch_first=True, dropout=dropout)

        self.out_fc = nn.Linear(hidden, n_metrics)
        self.out_act = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden

    def forward(self, r_norm, workload_id, phase_idx, z=None):
        B = r_norm.shape[0]
        T = self.seg_len
        device = r_norm.device

        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)

        r_emb = self.r_embed(r_norm.unsqueeze(-1))
        wl_emb = self.wl_embed(workload_id)
        ph_emb = self.ph_embed(phase_idx)

        zrwp = torch.cat([z, r_emb, wl_emb, ph_emb], dim=-1)
        h0 = self.h_init(zrwp)
        c0 = self.c_init(zrwp)
        h0 = h0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = c0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        z_exp = z.unsqueeze(1).expand(-1, T, -1)
        r_exp = r_emb.unsqueeze(1).expand(-1, T, -1)
        wl_exp = wl_emb.unsqueeze(1).expand(-1, T, -1)
        ph_exp = ph_emb.unsqueeze(1).expand(-1, T, -1)
        dec_in = torch.cat([z_exp, r_exp, wl_exp, ph_exp], dim=-1)

        dec_out, _ = self.dec_rnn(dec_in, (h0, c0))
        return self.out_act(self.out_fc(dec_out))

    @torch.no_grad()
    def generate_full_trace(self, r_norm_val, workload_id_val, device, n_samples=1):
        """Generate full trace and return all phase segments."""
        self.eval()
        segments = []
        for ph in range(N_PHASES):
            r_norm = torch.full((n_samples,), r_norm_val, dtype=torch.float32, device=device)
            wl_id = torch.full((n_samples,), workload_id_val, dtype=torch.long, device=device)
            ph_idx = torch.full((n_samples,), ph, dtype=torch.long, device=device)
            seg = self(r_norm, wl_id, ph_idx)
            segments.append(seg)
        return segments  # List of 6 tensors


class DiscriminatorUnified(nn.Module):
    """Discriminator with workload conditioning (same as S29)."""
    def __init__(self, n_metrics, cfg):
        super().__init__()
        hidden = cfg["hidden_dim"]
        n_layers = cfg["num_layers"]
        dropout = cfg["dropout"] if n_layers > 1 else 0.0
        
        self.wl_embed = nn.Embedding(N_WORKLOADS, 8)
        
        self.lstm = nn.LSTM(n_metrics, hidden, n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2 + 1 + 8, max(hidden, 16)),
            nn.LeakyReLU(0.2),
            nn.Linear(max(hidden, 16), 1)
        )

    def forward(self, x, r_norm=None, workload_id=None):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        if r_norm is not None:
            h = torch.cat([h, r_norm.unsqueeze(1)], dim=1)
        else:
            h = torch.cat([h, torch.zeros(h.shape[0], 1, device=h.device)], dim=1)
        
        if workload_id is not None:
            wl_emb = self.wl_embed(workload_id)
            h = torch.cat([h, wl_emb], dim=1)
        else:
            h = torch.cat([h, torch.zeros(h.shape[0], 8, device=h.device)], dim=1)
        
        return self.classifier(h)

    def get_features(self, x):
        _, (h_n, _) = self.lstm(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)


def compute_boundary_smoothness_loss(segments):
    """
    Compute smoothness loss at phase boundaries.
    
    Args:
        segments: List of 6 phase tensors (B, 120, M)
    
    Returns:
        Loss penalizing discontinuities
    """
    loss = torch.tensor(0.0, device=segments[0].device)
    
    for i in range(len(segments) - 1):
        # Last timestep of phase i
        end_phase_i = segments[i][:, -1, :]  # (B, M)
        # First timestep of phase i+1
        start_phase_next = segments[i+1][:, 0, :]  # (B, M)
        
        # MSE between boundary points
        loss += F.mse_loss(end_phase_i, start_phase_next)
    
    # Average over 5 boundaries
    return loss / (len(segments) - 1)


def gradient_penalty(disc, real, fake, r_norm, workload_id, device, lam=10.0):
    B = real.shape[0]
    eps = torch.rand(B, 1, 1, device=device)
    mid = (eps * real + (1 - eps) * fake).requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        d = disc(mid, r_norm, workload_id)
    grad = torch.autograd.grad(d.sum(), mid, create_graph=True)[0]
    return lam * ((grad.norm(2, dim=(1, 2)) - 1) ** 2).mean()


def precompute_fm_targets_unified(segments, seg_rc, seg_wl, seg_phase_idx, device):
    """Precompute FM targets per (workload, phase, replica)."""
    targets = {}
    for wl_val in range(N_WORKLOADS):
        for ph_val in set(seg_phase_idx.tolist()):
            for r_val in set(seg_rc.tolist()):
                mask = (seg_rc == r_val) & (seg_phase_idx == ph_val) & (seg_wl == wl_val)
                if mask.sum() < 2:
                    continue
                group = segments[mask]
                targets[(int(wl_val), int(ph_val), int(r_val))] = {
                    'mean': torch.tensor(group.mean(axis=(0, 1)), dtype=torch.float32).to(device),
                    'std': torch.tensor(group.std(axis=(0, 1)), dtype=torch.float32).to(device) + 1e-8,
                }
    return targets


def compute_segment_fm_stat_unified(fake, r_int, workload_id_batch, phase_idx_batch, fm_targets, device):
    """Compute FM loss for unified model."""
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    unique_wl = torch.unique(workload_id_batch)
    unique_r = torch.unique(r_int)
    unique_ph = torch.unique(phase_idx_batch)

    for wl_val in unique_wl:
        for ph_val in unique_ph:
            for r_val in unique_r:
                key = (int(wl_val.item()), int(ph_val.item()), int(r_val.item()))
                if key not in fm_targets:
                    continue
                mask = (r_int == r_val) & (phase_idx_batch == ph_val) & (workload_id_batch == wl_val)
                if mask.sum() < 1:
                    continue

                fake_seg = fake[mask]
                t_mean = fm_targets[key]['mean']
                t_std = fm_targets[key]['std']

                fake_mean = fake_seg.mean(dim=(0, 1))
                scale = (t_mean.abs() + t_std + 1e-6).detach()
                total_loss = total_loss + ((fake_mean - t_mean.detach()) / scale).pow(2).mean()

                fake_std = fake_seg.std(dim=(0, 1)) + 1e-8
                log_std = torch.log(fake_std / t_std.detach())
                total_loss = total_loss + (log_std ** 2).mean()

                count += 1

    return total_loss / max(count, 1)


def train_s32(generator, discriminator, train_dl, val_dl,
              base_cfg, stat_warmup, adv_epochs, gen_cfg,
              lambda_var_reg, lambda_fm_stat, lambda_smooth, lambda_boundary,
              n_disc_steps, device, fm_targets):
    """Training loop with boundary smoothing."""
    
    use_fm = base_cfg.get("lambda_fm", 0.0) > 0
    lam_fm = base_cfg.get("lambda_fm", 0.0)
    lam_gp = base_cfg.get("lambda_gp", 10.0)
    lam_adv = base_cfg.get("lambda_adv_post", 1.0)
    betas = (0.0, 0.9)

    opt_G = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=betas, weight_decay=1e-5)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=betas, weight_decay=1e-5)
    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=0.5, patience=7)

    best_val = float("inf")
    best_G = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    patience_c = 0
    warmup_done = False
    total_epochs = stat_warmup + adv_epochs
    history = []

    print(f"Warmup: {stat_warmup}, Adversarial: {adv_epochs}")
    print(f"lambda_var_reg={lambda_var_reg}, lambda_fm_stat={lambda_fm_stat}")
    print(f"lambda_smooth={lambda_smooth}, lambda_boundary={lambda_boundary} (NEW)")
    print("=" * 80)

    for epoch in range(1, total_epochs + 1):
        in_warmup = (epoch <= stat_warmup)
        n_disc = 0 if in_warmup else n_disc_steps

        if not in_warmup and not warmup_done:
            warmup_done = True

        generator.train()
        discriminator.train()
        ep_loss_g = ep_loss_d = ep_wdist = ep_vreg = ep_sm = ep_fmstat = ep_bound = 0.0
        n_batches = 0

        for target, r_int, r_norm, wl_id, phase_idx in train_dl:
            target = target.to(device)
            r_norm = r_norm.to(device)
            r_int = r_int.to(device)
            wl_id = wl_id.to(device)
            phase_idx = phase_idx.to(device)

            # Train discriminator
            if n_disc > 0:
                with torch.no_grad():
                    fake_d = generator(r_norm, wl_id, phase_idx)
                for _ in range(n_disc):
                    opt_D.zero_grad()
                    sr = discriminator(target, r_norm, wl_id)
                    sf = discriminator(fake_d, r_norm, wl_id)
                    wd = sr.mean() - sf.mean()
                    ld = -wd + gradient_penalty(discriminator, target, fake_d, r_norm, wl_id, device, lam_gp)
                    ld.backward()
                    nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                    opt_D.step()
                ep_loss_d += ld.item()
                ep_wdist += wd.item()

            # Train generator
            opt_G.zero_grad()
            fake = generator(r_norm, wl_id, phase_idx)
            loss_G = torch.tensor(0.0, device=device)

            # Adversarial loss
            if not in_warmup:
                loss_adv = -discriminator(fake, r_norm, wl_id).mean()
                loss_G = loss_G + lam_adv * loss_adv
                ep_loss_g += loss_adv.item()

                if use_fm:
                    fr = discriminator.get_features(target).detach()
                    ff = discriminator.get_features(fake)
                    loss_G = loss_G + lam_fm * F.mse_loss(ff.mean(0), fr.mean(0))

            # Variance regularization
            if lambda_var_reg > 0:
                std_real = target.std(dim=[0, 1]).detach()
                std_fake = fake.std(dim=[0, 1])
                log_ratio = torch.log((std_fake + 1e-8) / (std_real + 1e-8))
                var_reg = (log_ratio ** 2).mean()
                loss_G = loss_G + lambda_var_reg * var_reg
                ep_vreg += var_reg.item()

            # FM stat loss
            if lambda_fm_stat > 0:
                fm_stat_loss = compute_segment_fm_stat_unified(
                    fake, r_int, wl_id, phase_idx, fm_targets, device)
                loss_G = loss_G + lambda_fm_stat * fm_stat_loss
                ep_fmstat += fm_stat_loss.item()

            # Temporal smoothing
            if lambda_smooth > 0 and not in_warmup:
                diff = fake[:, 1:, :] - fake[:, :-1, :]
                sm = (diff ** 2).mean()
                loss_G = loss_G + lambda_smooth * sm
                ep_sm += sm.item()

            # BOUNDARY SMOOTHING (NEW for S32)
            if lambda_boundary > 0 and not in_warmup:
                # Generate full trace for a subset of batch WITH GRADIENTS
                B = min(4, r_norm.shape[0])  # Small subset to save memory
                unique_r = torch.unique(r_norm[:B])
                unique_wl = torch.unique(wl_id[:B])
                
                if len(unique_r) > 0 and len(unique_wl) > 0:
                    r_val = unique_r[0]
                    wl_val = unique_wl[0]
                    
                    # Generate all 6 phases WITH gradients (don't use @torch.no_grad)
                    segments = []
                    for ph in range(N_PHASES):
                        r_batch = torch.full((B,), r_val.item(), dtype=torch.float32, device=device)
                        wl_batch = torch.full((B,), wl_val.item(), dtype=torch.long, device=device)
                        ph_batch = torch.full((B,), ph, dtype=torch.long, device=device)
                        seg = generator(r_batch, wl_batch, ph_batch)
                        segments.append(seg)
                    
                    bound_loss = compute_boundary_smoothness_loss(segments)
                    loss_G = loss_G + lambda_boundary * bound_loss
                    ep_bound += bound_loss.item()

        # Validation
        generator.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for target_v, r_int_v, r_norm_v, wl_id_v, phase_idx_v in val_dl:
                target_v = target_v.to(device)
                r_norm_v = r_norm_v.to(device)
                r_int_v = r_int_v.to(device)
                wl_id_v = wl_id_v.to(device)
                phase_idx_v = phase_idx_v.to(device)
                fake_v = generator(r_norm_v, wl_id_v, phase_idx_v)
                vs = compute_segment_fm_stat_unified(
                    fake_v, r_int_v, wl_id_v, phase_idx_v, fm_targets, device)
                val_loss += vs.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        if n_batches > 0:
            ep_loss_g /= n_batches
            ep_loss_d /= n_batches
            ep_wdist /= n_batches
            ep_vreg /= n_batches
            ep_sm /= n_batches
            ep_fmstat /= n_batches
            ep_bound /= n_batches

        sched_G.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_G = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1

        history.append({
            "epoch": epoch, "val": val_loss,
            "g": ep_loss_g, "d": ep_loss_d, "wd": ep_wdist,
            "vr": ep_vreg, "fm": ep_fmstat, "sm": ep_sm, "bd": ep_bound,
        })

        if epoch % 10 == 1 or epoch == total_epochs:
            phase = "warmup" if in_warmup else "adv"
            print(f"Epoch {epoch:3d} [{phase}] G={ep_loss_g:.4f} D={ep_loss_d:.4f} "
                  f"W={ep_wdist:.4f} val={val_loss:.5f}")

    generator.load_state_dict(best_G)
    print(f"\nBest val loss: {best_val:.5f}")
    
    return generator, best_val, history


def main():
    parser = argparse.ArgumentParser(description="TimeGAN S32 - GPU-Bound + Boundary Smoothing")
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--var-reg", type=float, default=0.4)
    parser.add_argument("--fm-stat", type=float, default=1.2)
    parser.add_argument("--smooth", type=float, default=0.07)
    parser.add_argument("--adv-epochs", type=int, default=150)
    parser.add_argument("--stat-warmup", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(GEN_CFG["seed"])
    np.random.seed(GEN_CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag = f"s32_gpu_bound_smooth{int(args.boundary_weight * 10):02d}"
    out_dir = Path(f"outputs/phase4/timegan_s32/{run_tag}")
    model_dir = Path(f"models/phase4/timegan_s32/{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("S32: GPU-BOUND UNIFIED MODEL + BOUNDARY SMOOTHING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Innovation: S29 architecture + boundary smoothing loss")
    print(f"Workloads: {VALID_WORKLOAD_NAMES}")
    print(f"Output: {out_dir}")
    print("=" * 80)

    # Load data
    data, norm = load_combined_data()
    raw_traces = data["traces"]
    replica_counts = data["replica_counts"]
    workload_ids = data["workload_ids"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    
    print(f"\nFiltered dataset (GPU-bound only): {len(raw_traces)} pods")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    for i, wl_name in enumerate(VALID_WORKLOAD_NAMES):
        count = (workload_ids == i).sum()
        print(f"  {wl_name.upper():<10}: {count} pods")
    
    # Get kept metrics
    kept_idx, kept_names = get_unified_kept_metrics()
    n_metrics = len(kept_idx)
    raw_kept = raw_traces[:, :, kept_idx].astype(np.float32)
    
    print(f"Metrics: {n_metrics}")
    
    # Segment
    (all_segs, all_seg_rc, all_seg_wl,
     all_seg_ph, all_seg_ti) = segment_traces_unified(
        raw_kept, replica_counts, workload_ids,
        DEFAULT_PHASE_BOUNDARIES, N_PHASES, SEGMENT_LEN)
    
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    seg_train_idx = np.where(np.isin(all_seg_ti, list(train_set)))[0]
    seg_val_idx = np.where(np.isin(all_seg_ti, list(val_set)))[0]
    
    print(f"Segmenting traces...")
    print(f"Segments: train={len(seg_train_idx)}, val={len(seg_val_idx)}")
    
    # Datasets
    ds_train = SegmentDatasetUnified(
        all_segs[seg_train_idx], all_seg_rc[seg_train_idx],
        all_seg_wl[seg_train_idx], all_seg_ph[seg_train_idx],
        np.arange(len(seg_train_idx)), jitter=4)
    ds_val = SegmentDatasetUnified(
        all_segs[seg_val_idx], all_seg_rc[seg_val_idx],
        all_seg_wl[seg_val_idx], all_seg_ph[seg_val_idx],
        np.arange(len(seg_val_idx)), jitter=0)
    
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)
    
    # Models
    generator = GeneratorSegUnified(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    discriminator = DiscriminatorUnified(n_metrics, DISC_CFG).to(device)
    
    print("=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(f"Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params")
    
    # FM targets
    fm_targets = precompute_fm_targets_unified(
        all_segs[seg_train_idx], all_seg_rc[seg_train_idx],
        all_seg_wl[seg_train_idx], all_seg_ph[seg_train_idx], device)
    
    # Train
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    t0 = time.time()
    generator, best_val, history = train_s32(
        generator, discriminator, dl_train, dl_val,
        BASE_CONFIG, args.stat_warmup, args.adv_epochs, GEN_CFG,
        args.var_reg, args.fm_stat, args.smooth, args.boundary_weight,
        S32_HYPERPARAMS["n_disc_steps"], device, fm_targets)
    elapsed = time.time() - t0
    
    print(f"Training complete: {elapsed:.1f}s")
    print(f"Best val loss: {best_val:.5f}")
    
    # Save
    torch.save(generator.state_dict(), model_dir / "generator.pt")
    torch.save(discriminator.state_dict(), model_dir / "discriminator.pt")
    
    with open(model_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    result = {
        "stage": "s32",
        "innovation": "GPU-bound unified + boundary smoothing",
        "excluded_workloads": ["whisper"],
        "valid_workloads": VALID_WORKLOAD_NAMES,
        "n_workloads": N_WORKLOADS,
        "train_pods": len(train_idx),
        "val_pods": len(val_idx),
        "n_metrics": n_metrics,
        "elapsed_s": elapsed,
        "best_val_loss": float(best_val),
    }
    
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved to: {model_dir}")
    print("=" * 80)
    print("S32 TRAINING COMPLETE")
    print("=" * 80)
    print("Next: Evaluate S32 with eval_s32.py")
    print("Expected: Mean VR ~1.5-1.7, YOLO > 0.8, smooth boundaries")
    print("=" * 80)


if __name__ == "__main__":
    main()