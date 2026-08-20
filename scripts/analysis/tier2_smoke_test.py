#!/usr/bin/env python3
"""Tier 2 S36 checkpoint smoke test (S36 Tier 2 retrain plan, Step 4 Phase 2).

Load-and-generate sanity check only -- not evaluation (Step 5 does VR/
Wasserstein eval properly). Confirms each of the 5 trained generators
loads and produces non-degenerate output.

Generator architecture, TRAINED_INDICES/TRAINED_NAMES, and the
r_norm=(replica_count-1)/9.0 conditioning formula are copied verbatim
from scripts/phase4/postprocess_s36_tier3.py -- the r_norm denominator
of 9.0 (not 6.0) is intentional: it's the same formula baked into
SegmentDataset during training (untouched, frozen architecture), so at
Tier 2's max r=7 the model only ever saw r_norm up to (7-1)/9.0=0.667,
never 1.0. Mirrored here exactly rather than "corrected" for Tier 2's
r=1..7 range, since generation must match what the model was actually
trained on.

Run from repo root:
    python scripts/analysis/tier2_smoke_test.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
DATA_DIR = Path("data/processed/tier2/unified")

TAGS = {
    "bert": "vr05_fm15",
    "gpt2": "vr03_fm20",
    "resnet152": "vr04_fm10",
    "whisper": "vr05_fm05",
    "yolo": "vr03_fm12",
}

TRAINED_INDICES = [0, 1, 2, 3, 4, 5, 8]
TRAINED_NAMES = [
    'pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu',
    'pod_latency_avg', 'pod_throughput', 'gpu_utilization', 'gpu_power_watts'
]

N_PHASES = 6
SEGMENT_LEN = 120

GEN_CFG = {
    "hidden_dim": 128,
    "num_layers": 2,
    "latent_dim": 64,
    "replica_embed_dim": 16,
    "phase_embed_dim": 8,
    "dropout": 0.1,
}


class GeneratorSeg(nn.Module):
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


def load_generator(workload, device):
    ckpt_path = Path(f"models/phase4/timegan_s36_tier2/s36_{workload}_{TAGS[workload]}/generator.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"S36 Tier 2 checkpoint not found: {ckpt_path}")
    gen = GeneratorSeg(SEGMENT_LEN, len(TRAINED_INDICES), GEN_CFG).to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()
    return gen


def generate_raw_traces(gen, replica_count, n_samples, device):
    """Generate n_samples raw (715, 7) normalized traces, batched."""
    r_norm = torch.full((n_samples,), (replica_count - 1) / 9.0,
                        dtype=torch.float32, device=device)
    phases = []
    for ph in range(N_PHASES):
        phase_idx = torch.full((n_samples,), ph, dtype=torch.long, device=device)
        with torch.no_grad():
            seg = gen(r_norm, phase_idx)  # (n_samples, SEGMENT_LEN, 7)
        phases.append(seg.cpu().numpy())
        if ph == 0:
            print(f"    generator input shapes: r_norm={tuple(r_norm.shape)} "
                  f"phase_idx={tuple(phase_idx.shape)}  output shape per phase: {tuple(seg.shape)}")
    full = np.concatenate(phases, axis=1)  # (n_samples, N_PHASES*SEGMENT_LEN, 7)
    return full[:, :715, :]


def load_normalization_params(workload):
    with open(DATA_DIR / "combined_normalization.json") as f:
        norm = json.load(f)
    return norm[workload]['params']


def denormalize(traces_norm, norm_params, metric_names):
    """traces_norm: (n_samples, 715, 7) -> denormalized in place-shape."""
    out = np.zeros_like(traces_norm)
    for i, name in enumerate(metric_names):
        p = norm_params[name]
        out[:, :, i] = traces_norm[:, :, i] * (p['max'] - p['min']) + p['min']
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    assert (DATA_DIR / "combined_dataset.npz").exists()
    assert (DATA_DIR / "combined_normalization.json").exists()
    np.load(DATA_DIR / "combined_dataset.npz", allow_pickle=True)
    with open(DATA_DIR / "combined_normalization.json") as f:
        json.load(f)
    print(f"Loaded {DATA_DIR / 'combined_dataset.npz'} and combined_normalization.json OK\n")

    results = {}
    any_fail = False

    for wl in WORKLOADS:
        print(f"=== {wl.upper()} ===")
        gen = load_generator(wl, device)
        raw = generate_raw_traces(gen, replica_count=1, n_samples=4, device=device)
        print(f"    post-concat/truncate output shape: {raw.shape}")

        norm_params = load_normalization_params(wl)
        denorm = denormalize(raw, norm_params, TRAINED_NAMES)

        assert denorm.shape == (4, 715, 7), f"{wl}: unexpected shape {denorm.shape}"

        has_nan = bool(np.isnan(denorm).any())
        has_inf = bool(np.isinf(denorm).any())
        all_zero_per_metric = [bool(np.all(denorm[:, :, i] == 0)) for i in range(7)]
        any_all_zero = any(all_zero_per_metric)

        cpu_idx = TRAINED_NAMES.index('pod_cpu_usage')
        gpu_idx = TRAINED_NAMES.index('gpu_utilization')
        cpu_cross_sample_var = float(denorm[:, :, cpu_idx].mean(axis=1).var())
        gpu_cross_sample_var = float(denorm[:, :, gpu_idx].mean(axis=1).var())

        degenerate = has_nan or has_inf or any_all_zero or cpu_cross_sample_var <= 1e-6 or gpu_cross_sample_var <= 1e-6

        print(f"    NaN: {has_nan}  Inf: {has_inf}  any_all_zero_metric: {any_all_zero}")
        print(f"    cross-sample var: pod_cpu_usage={cpu_cross_sample_var:.6g}  "
              f"gpu_utilization={gpu_cross_sample_var:.6g}")

        print(f"    per-metric min/max/mean (sample 0):")
        sample0_stats = {}
        for i, name in enumerate(TRAINED_NAMES):
            col = denorm[0, :, i]
            lo, hi, mean = float(col.min()), float(col.max()), float(col.mean())
            sample0_stats[name] = {'min': lo, 'max': hi, 'mean': mean}
            print(f"      {name:20s} min={lo:12.4f} max={hi:12.4f} mean={mean:12.4f}")

        if degenerate:
            any_fail = True
            print(f"    DEGENERATE OUTPUT DETECTED for {wl}")

        results[wl] = {
            'has_nan': has_nan, 'has_inf': has_inf,
            'any_all_zero_metric': any_all_zero,
            'cpu_cross_sample_var': cpu_cross_sample_var,
            'gpu_cross_sample_var': gpu_cross_sample_var,
            'sample0_stats': sample0_stats,
            'degenerate': degenerate,
        }
        print()

    print("=" * 60)
    if any_fail:
        print("SMOKE TEST: FAILED -- degenerate output detected, see above")
    else:
        print("SMOKE TEST: ALL 5 WORKLOADS PASSED")
    print("=" * 60)

    return results, any_fail


if __name__ == "__main__":
    main()
