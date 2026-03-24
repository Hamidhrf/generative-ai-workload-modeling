"""
TimeGAN S29 - Unified Cross-Workload Model
===========================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

S29 = S27 Architecture + Workload Conditioning
===============================================

KEY INNOVATION:
- Single model trained on ALL 5 workloads (245 pods vs 49 per-workload)
- Workload-conditional generation via learnable embeddings
- Addresses data scarcity by pooling cross-workload patterns

ARCHITECTURE:
- GeneratorSegUnified: S27 + workload_embed (16-dim)
- DiscriminatorUnified: S27 + workload conditioning
- Conditioning: replica_count + workload_id + phase_idx

TRAINING DATA:
- Combined: 275 pods (245 train, 30 val)
- BERT, GPT2, ResNet152, Whisper, YOLO
- 5x more data per replica count vs S27

HYPOTHESIS:
Unified model learns:
1. Shared patterns (pod lifecycle, GPU time-slicing, PSI curves)
2. Workload-specific differences (GPU 2% vs 98%, CPU-bound vs GPU-bound)
3. Better generalization through cross-workload regularization

EXPECTED RESULTS:
- Mean VR > S27 (1.062) due to more training data
- Metrics with shared patterns improve most (latency, PSI, memory)
- Workload-specific metrics maintain fidelity (gpu_utilization)

USAGE:
------
    python timegan_s29.py
    python timegan_s29.py --adv-epochs 150
    python timegan_s29.py --var-reg 0.4

OUTPUT:
-------
    outputs/phase4/timegan_s29/{run_tag}/plots/
    models/phase4/timegan_s29/{run_tag}/generator.pt  # SINGLE unified model
    outputs/phase4/timegan_s29/{run_tag}/results.json
"""

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Combined dataset path
DATA_COMBINED = Path("data/processed/phase4/unified/combined_dataset.npz")
NORM_COMBINED = Path("data/processed/phase4/unified/combined_normalization.json")

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
N_WORKLOADS = 5

ALL_METRICS = [
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
    "pod_latency_avg", "pod_throughput",
    "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
    "gpu_power_watts", "gpu_temperature",
]

# S29 DROP CONFIG: Per-workload drops (same as S27)
DROP_PER_WORKLOAD = {
    "bert":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "gpt2":      {"gpu_memory_total", "gpu_temperature"},
    "resnet152": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "whisper":   {"gpu_memory_total", "gpu_memory_used", "pod_throughput", "gpu_temperature"},
    "yolo":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
}

# Segment constants
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
    "workload_embed_dim": 16,  # NEW
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

# S29 uses averaged hyperparams across workloads
UNIFIED_HYPERPARAMS = {
    "lambda_smooth": 0.07,  # avg(0.05, 0.1, 0.1, 0.05, 0.1)
    "n_disc_steps": 2,      # majority value
    "lambda_fm_stat": 1.2,  # avg(1.5, 2.0, 1.0, 0.5, 1.2)
    "lambda_var_reg": 0.4,  # avg(0.5, 0.3, 0.4, 0.5, 0.3)
}

REFERENCE_VR = {
    "bert": {"s27": 0.886, "s21": 0.672},
    "gpt2": {"s27": 0.946, "s21": 0.996},
    "resnet152": {"s27": 0.851, "s21": 0.960},
    "whisper": {"s27": 1.630, "s21": 1.722},
    "yolo": {"s27": 0.997, "s21": 0.661},
}


def load_combined_data():
    """Load unified dataset (275 pods)."""
    data = np.load(DATA_COMBINED, allow_pickle=True)
    with open(NORM_COMBINED) as f:
        norm = json.load(f)
    return dict(data), norm


def get_unified_kept_metrics():
    """
    Determine which metrics to keep for unified model.
    Strategy: Keep metric if ANY workload trains it.
    """
    all_kept = set()
    for wl in WORKLOADS:
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


def denormalize_unified(traces, workload_ids, kept_names, norm_params):
    """
    Denormalize using per-workload normalization params.
    
    Args:
        traces: (N, T, M) normalized [0,1]
        workload_ids: (N,) workload indices
        kept_names: list of metric names
        norm_params: dict[workload_name] -> normalization params
    """
    N, T, M = traces.shape
    out = np.zeros_like(traces, dtype=np.float64)
    
    for i in range(N):
        wl_id = int(workload_ids[i])
        wl_name = WORKLOADS[wl_id]
        wl_norm = norm_params[wl_name]["params"]
        
        for j, mname in enumerate(kept_names):
            if mname not in wl_norm:
                continue
            mn = wl_norm[mname].get("min", 0.0)
            mx = wl_norm[mname].get("max", 1.0)
            out[i, :, j] = traces[i, :, j] * (mx - mn) + mn
    
    return out


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
    """
    Unified generator with workload conditioning.
    
    Conditioning: replica_count + workload_id + phase_idx
    """
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

        self.r_embed = nn.Sequential(
            nn.Linear(1, r_emb_dim),
            nn.Tanh(),
        )
        self.wl_embed = nn.Embedding(N_WORKLOADS, wl_emb_dim)
        self.ph_embed = nn.Embedding(N_PHASES + 1, ph_emb_dim)

        init_in = latent_dim + r_emb_dim + wl_emb_dim + ph_emb_dim
        self.h_init = nn.Sequential(
            nn.Linear(init_in, hidden * n_layers),
            nn.Tanh(),
        )
        self.c_init = nn.Sequential(
            nn.Linear(init_in, hidden * n_layers),
            nn.Tanh(),
        )

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
    def generate_trace(self, r_norm_val, workload_id_val, device, n_samples=1):
        """Generate full trace for given replica count + workload."""
        self.eval()
        segments = []
        for ph in range(N_PHASES):
            r_norm = torch.full((n_samples,), r_norm_val, dtype=torch.float32, device=device)
            wl_id = torch.full((n_samples,), workload_id_val, dtype=torch.long, device=device)
            ph_idx = torch.full((n_samples,), ph, dtype=torch.long, device=device)
            seg = self(r_norm, wl_id, ph_idx)
            segments.append(seg.cpu().numpy())
        full_trace = np.concatenate(segments, axis=1)
        return full_trace


class DiscriminatorUnified(nn.Module):
    """Discriminator with workload conditioning."""
    def __init__(self, n_metrics, cfg):
        super().__init__()
        hidden = cfg["hidden_dim"]
        n_layers = cfg["num_layers"]
        dropout = cfg["dropout"] if n_layers > 1 else 0.0
        
        self.wl_embed = nn.Embedding(N_WORKLOADS, 8)
        
        self.lstm = nn.LSTM(n_metrics, hidden, n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Conditioning: r_norm + workload_embed
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


def compute_variance_ratio(real, synthetic, cap=5.0):
    var_r = real.reshape(-1, real.shape[-1]).var(axis=0)
    var_s = synthetic.reshape(-1, synthetic.shape[-1]).var(axis=0)
    ratio = np.where(var_r > 1e-10,
                     np.clip(var_s / var_r, 0, cap),
                     np.ones_like(var_r))
    return ratio, float(ratio.mean())


def train_unified(generator, discriminator, train_dl, val_dl,
                  base_cfg, stat_warmup, adv_epochs, gen_cfg,
                  lambda_var_reg, lambda_fm_stat, lambda_smooth,
                  n_disc_steps, device, fm_targets):
    """Training loop for unified model."""
    
    use_fm = base_cfg.get("lambda_fm", 0.0) > 0
    lam_fm = base_cfg.get("lambda_fm", 0.0)
    lam_gp = base_cfg.get("lambda_gp", 10.0)
    lam_adv = base_cfg.get("lambda_adv_post", 1.0)
    betas = (0.0, 0.9)

    opt_G = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=betas, weight_decay=1e-5)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=betas, weight_decay=1e-5)
    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=0.5, patience=7)

    best_wdist = float("-inf")
    best_G = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_D = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
    best_stat = float("inf")
    best_G_stat = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    patience_c = 0
    warmup_done = False
    actual_warmup = stat_warmup
    total_epochs = stat_warmup + adv_epochs
    history = []

    print(f"    Statistical warmup: {stat_warmup} epochs")
    print(f"    Adversarial: {adv_epochs} epochs")
    print(f"    lambda_var_reg={lambda_var_reg}, lambda_fm_stat={lambda_fm_stat}, "
          f"lambda_smooth={lambda_smooth}, n_disc={n_disc_steps}")

    for epoch in range(1, total_epochs + 1):
        in_warmup = (epoch <= actual_warmup)
        n_disc_steps_ep = 0 if in_warmup else n_disc_steps

        if not in_warmup and not warmup_done:
            warmup_done = True
            print(f"    [epoch {epoch}] Adversarial start.")

        generator.train()
        discriminator.train()
        ep_adv_g = ep_loss_d = ep_wdist = ep_vreg = ep_sm = ep_fmstat = 0.0
        n_batches = 0

        for target, r_int, r_norm, wl_id, phase_idx in train_dl:
            target = target.to(device)
            r_norm = r_norm.to(device)
            r_int = r_int.to(device)
            wl_id = wl_id.to(device)
            phase_idx = phase_idx.to(device)

            if n_disc_steps_ep > 0:
                with torch.no_grad():
                    fake_d = generator(r_norm, wl_id, phase_idx)
                for _ in range(n_disc_steps_ep):
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

            opt_G.zero_grad()
            fake = generator(r_norm, wl_id, phase_idx)

            loss_G = torch.tensor(0.0, device=device)

            if not in_warmup:
                loss_adv = -discriminator(fake, r_norm, wl_id).mean()
                loss_G = loss_G + lam_adv * loss_adv
                ep_adv_g += loss_adv.item()

                if use_fm:
                    fr = discriminator.get_features(target).detach()
                    ff = discriminator.get_features(fake)
                    loss_G = loss_G + lam_fm * F.mse_loss(ff.mean(0), fr.mean(0))

            if lambda_var_reg > 0:
                std_real = target.std(dim=[0, 1]).detach()
                std_fake = fake.std(dim=[0, 1])
                log_ratio = torch.log((std_fake + 1e-8) / (std_real + 1e-8))
                var_reg = (log_ratio ** 2).mean()
                loss_G = loss_G + lambda_var_reg * var_reg
                ep_vreg += var_reg.item()

            if lambda_fm_stat > 0:
                fm_stat_loss = compute_segment_fm_stat_unified(
                    fake, r_int, wl_id, phase_idx, fm_targets, device)
                loss_G = loss_G + lambda_fm_stat * fm_stat_loss
                ep_fmstat += fm_stat_loss.item()

            if lambda_smooth > 0 and not in_warmup:
                diff = fake[:, 1:, :] - fake[:, :-1, :]
                sm = (diff ** 2).mean()
                loss_G = loss_G + lambda_smooth * sm
                ep_sm += sm.item()

            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=gen_cfg["grad_clip"])
            opt_G.step()
            n_batches += 1

        generator.eval()
        val_stat_sum = 0.0
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
                val_stat_sum += vs.item()
                n_val += 1
        val_stat = val_stat_sum / max(n_val, 1)

        if n_batches > 0:
            ep_adv_g /= n_batches
            ep_loss_d /= n_batches
            ep_wdist /= n_batches
            ep_vreg /= n_batches
            ep_sm /= n_batches
            ep_fmstat /= n_batches

        sched_G.step(val_stat)

        if warmup_done and ep_wdist > best_wdist:
            best_wdist = ep_wdist
            best_G = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            best_D = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}

        if val_stat < best_stat - 1e-4:
            best_stat = val_stat
            best_G_stat = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1

        if in_warmup and patience_c >= 15:
            actual_warmup = epoch
            print(f"    [warmup] Patience at epoch {epoch}. Starting adv early.")

        history.append({
            "epoch": epoch, "val_stat": val_stat,
            "adv_g": ep_adv_g, "disc": ep_loss_d, "wdist": ep_wdist,
            "var_reg": ep_vreg, "fm_stat": ep_fmstat, "smooth": ep_sm,
        })

        if epoch % 10 == 1 or epoch == total_epochs:
            phase_str = "warmup" if in_warmup else "adv"
            print(f"    ep {epoch:4d} [{phase_str}] val_stat={val_stat:.5f} "
                  f"G={ep_adv_g:.4f} D={ep_loss_d:.4f} W={ep_wdist:.4f} "
                  f"vr={ep_vreg:.4f} fm={ep_fmstat:.4f} sm={ep_sm:.4f}")

    if warmup_done and best_wdist > float("-inf"):
        generator.load_state_dict(best_G)
        discriminator.load_state_dict(best_D)
        print(f"    Final: best W-dist checkpoint (W={best_wdist:.4f})")
    else:
        generator.load_state_dict(best_G_stat)
        print(f"    Final: best stat checkpoint (stat={best_stat:.5f})")

    return generator, discriminator, best_stat, total_epochs, history


def main():
    parser = argparse.ArgumentParser(description="TimeGAN S29 - Unified Cross-Workload Model")
    parser.add_argument("--var-reg", type=float, default=UNIFIED_HYPERPARAMS["lambda_var_reg"])
    parser.add_argument("--fm-stat", type=float, default=UNIFIED_HYPERPARAMS["lambda_fm_stat"])
    parser.add_argument("--smooth", type=float, default=UNIFIED_HYPERPARAMS["lambda_smooth"])
    parser.add_argument("--adv-epochs", type=int, default=150)
    parser.add_argument("--stat-warmup", type=int, default=20)
    parser.add_argument("--jitter", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(GEN_CFG["seed"])
    np.random.seed(GEN_CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag = (f"s29_unified_vr{int(args.var_reg * 10):02d}"
               f"_fm{int(args.fm_stat * 10):02d}"
               f"_ae{args.adv_epochs}")
    out_dir = Path(f"outputs/phase4/timegan_s29/{run_tag}")
    model_dir = Path(f"models/phase4/timegan_s29/{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("TimeGAN S29 - Unified Cross-Workload Model")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Innovation: Single model for ALL 5 workloads")
    print(f"Training data: 245 pods (vs S27: 49 per workload)")
    print(f"Conditioning: replica_count + workload_id + phase_idx")
    print(f"Output: {out_dir}")
    print("=" * 72)

    # Load combined dataset
    data, norm = load_combined_data()
    raw_traces = data["traces"]  # (275, 715, 10)
    replica_counts = data["replica_counts"]  # (275,)
    workload_ids = data["workload_ids"]  # (275,)
    train_idx = data["train_idx"]  # (245,)
    val_idx = data["val_idx"]  # (30,)
    
    print(f"\nDataset loaded:")
    print(f"  Total pods: {len(raw_traces)}")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val: {len(val_idx)}")
    
    # Get unified kept metrics
    kept_idx, kept_names = get_unified_kept_metrics()
    n_metrics = len(kept_idx)
    raw_kept = raw_traces[:, :, kept_idx].astype(np.float32)
    
    print(f"  Unified metrics ({n_metrics}): {kept_names}")
    
    # Segment traces
    boundaries = DEFAULT_PHASE_BOUNDARIES
    (all_segs, all_seg_rc, all_seg_wl,
     all_seg_ph, all_seg_ti) = segment_traces_unified(
        raw_kept, replica_counts, workload_ids,
        boundaries, N_PHASES, SEGMENT_LEN)
    
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    seg_train_idx = np.where(np.isin(all_seg_ti, list(train_set)))[0]
    seg_val_idx = np.where(np.isin(all_seg_ti, list(val_set)))[0]
    
    print(f"  Segments: train={len(seg_train_idx)}, val={len(seg_val_idx)}")
    
    # Create datasets
    ds_train = SegmentDatasetUnified(
        all_segs[seg_train_idx],
        all_seg_rc[seg_train_idx],
        all_seg_wl[seg_train_idx],
        all_seg_ph[seg_train_idx],
        np.arange(len(seg_train_idx)),
        jitter=args.jitter
    )
    ds_val = SegmentDatasetUnified(
        all_segs[seg_val_idx],
        all_seg_rc[seg_val_idx],
        all_seg_wl[seg_val_idx],
        all_seg_ph[seg_val_idx],
        np.arange(len(seg_val_idx)),
        jitter=0
    )
    
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False)
    
    # Create models
    generator = GeneratorSegUnified(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    discriminator = DiscriminatorUnified(n_metrics, DISC_CFG).to(device)
    
    n_params_G = sum(p.numel() for p in generator.parameters())
    n_params_D = sum(p.numel() for p in discriminator.parameters())
    print(f"  Generator params: {n_params_G:,}")
    print(f"  Discriminator params: {n_params_D:,}")
    
    # Precompute FM targets
    fm_targets = precompute_fm_targets_unified(
        all_segs[seg_train_idx],
        all_seg_rc[seg_train_idx],
        all_seg_wl[seg_train_idx],
        all_seg_ph[seg_train_idx],
        device
    )
    print(f"  FM targets: {len(fm_targets)} (workload, phase, r) groups")
    
    # Train
    t0 = time.time()
    generator, discriminator, best_stat, n_epochs, history = train_unified(
        generator, discriminator, dl_train, dl_val,
        BASE_CONFIG, args.stat_warmup, args.adv_epochs, GEN_CFG,
        args.var_reg, args.fm_stat, args.smooth,
        UNIFIED_HYPERPARAMS["n_disc_steps"], device, fm_targets
    )
    elapsed = time.time() - t0
    
    print(f"\n Training complete: {elapsed:.1f}s, {n_epochs} epochs")
    print(f"  Best stat: {best_stat:.5f}")
    
    # Save models
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), model_dir / "generator.pt")
    torch.save(discriminator.state_dict(), model_dir / "discriminator.pt")
    
    with open(model_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    with open(model_dir / "config.json", "w") as f:
        json.dump({
            "stage": "s29",
            "architecture": "GeneratorSegUnified",
            "n_workloads": N_WORKLOADS,
            "train_pods": len(train_idx),
            "val_pods": len(val_idx),
            "unified_metrics": kept_names,
            **GEN_CFG,
        }, f, indent=2)
    
    result = {
        "stage": "s29",
        "architecture": "GeneratorSegUnified",
        "n_workloads": N_WORKLOADS,
        "n_metrics": n_metrics,
        "train_pods": len(train_idx),
        "val_pods": len(val_idx),
        "best_stat": float(best_stat),
        "n_epochs": n_epochs,
        "elapsed_s": elapsed,
        "unified_metrics": kept_names,
    }
    
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n Saved: {model_dir}")
    print(f" Results: {out_dir / 'results.json'}")
    print("\nS29 training complete!")
    print("\nNEXT STEP: Evaluate per-workload with evaluate_s29.py")


if __name__ == "__main__":
    main()