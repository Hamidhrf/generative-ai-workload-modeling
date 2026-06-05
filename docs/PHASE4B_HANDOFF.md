# Phase 4b Handoff: LSTM Baseline Training

**Date:** February 23, 2026  
**From:** Phase 4a (Preprocessing + EDA)  
**To:** Phase 4b (LSTM Baseline Training)  
**Status:** Phase 4a COMPLETE → Ready for Phase 4b


## Critical Context from Phase 4a

### 1. Dataset Specifications

**Final Dataset:**
- **Total pods:** 275 (55 per workload)
- **Training samples:** 245 (49 per workload)
- **Validation samples:** 30 (6 per workload)
- **Timesteps:** 715 (60 minutes at 5s intervals)
- **Metrics:** 10 (removed pod_psi_memory, pod_psi_io)
- **Normalization:** MinMax [0, 1] per-workload
- **Replica counts:** Complete r=1-10 coverage

**Shape:** `(55, 715, 10)` per workload

### 2. Metrics (10)

```python
METRICS = [
    'pod_cpu_usage',       # 0
    'pod_memory_bytes',    # 1
    'pod_psi_cpu',         # 2
    'pod_latency_avg',     # 3
    'pod_throughput',      # 4
    'gpu_utilization',     # 5
    'gpu_memory_used',     # 6
    'gpu_memory_total',    # 7
    'gpu_power_watts',     # 8
    'gpu_temperature'      # 9
]
```

**Removed (zero variance):**
-  AI inference has no memory pressure
-  AI inference has no I/O pressure

### 3. Workload Characteristics

**GPT2 - Heavy GPU Saturation:**
- GPU: 0% → 50%, latency: 0.46s → 2.47s
- Pattern: Non-linear GPU saturation, exponential latency growth
- Challenge: Most GPU-intensive workload

**Whisper - Extreme CPU Bottleneck:**
- CPU: 0 → 6.18 cores, PSI: 0 → 0.62
- Pattern: CPU saturation prevents GPU utilization
- Challenge: r=10 pod_9 has partial data (see below)

**BERT - Balanced:**
- CPU: 0.007 → 0.20, GPU: 0% → 12.25%
- Pattern: Gradual scaling, stable latency
- Challenge: None, well-behaved

**ResNet152 - Light GPU:**
- CPU: 0.004 → 0.10, GPU: 0% → 43.5%
- Pattern: Linear scaling, consistent performance
- Challenge: None, efficient

**YOLO - Lightest:**
- CPU: 0.002 → 0.24, GPU: 0% → 7%
- Pattern: Very light usage, excellent scalability
- Challenge: None, minimal resource use

### 4. Whisper r=10 Pod_9 Decision

**Issue:**
Pod_9 at r=10 has:
-  CPU data (mean=0.267)
-  Memory data (mean=0.184)
-  PSI_CPU data (mean=0.839) ← EXTREMELY HIGH
-  Latency = 0 (missing)
-  Throughput = 0 (missing)

**Interpretation:**
- Pod was under extreme CPU stress but not serving requests
- Realistic degradation pattern (pod exists but unresponsive)

**Decision:**
 **KEPT in dataset** because:
1. Shows realistic failure mode
2. PSI_CPU=0.839 is valuable extreme data
3. Zero throughput is meaningful (not noise)

**Mitigation:**
- Monitor during LSTM training
- If causes issues (mode collapse, NaN loss), can remove easily
- Defer decision until we see actual training behavior

**Impact on LSTM:**
- Whisper model will learn: "at r=10, some pods have zero throughput"
- This is a FEATURE, not a bug - it's saturation signature

### 5. Critical Training Decisions

**Per-Workload vs Global Normalization:**
-  **Decision:** Per-workload normalization
- **Reason:** Preserves workload-specific characteristics
- **Implication:** Train 5 separate LSTM models (one per workload)

**Train/Val/Test Split:**
-  **Decision:** 90/10 train/val, NO test set
- **Reason:** Small dataset, generative model evaluation differs from discriminative
- **Implication:** Use validation for early stopping only
- **Evaluation:** Generation quality metrics (MSE, variance ratio, visual inspection)

**Conditioning Strategy:**
-  **Decision:** Condition on replica_count (1-10)
- **Reason:** Model needs to learn scaling behavior
- **Implication:** LSTM input = [trace features + replica_count embedding]

**Model Strategy:**
-  **Decision:** 5 separate models (NOT one multi-workload model)
- **Reason:** Different workload characteristics, per-workload normalization
- **Implication:** Train BERT model, GPT2 model, ResNet152 model, Whisper model, YOLO model independently

### 6. File Locations

**Data:**
```
data/processed/phase1_v3/
├── bert_traces.npz
├── bert_normalization.json
├── gpt2_traces.npz
├── gpt2_normalization.json
├── resnet152_traces.npz
├── resnet152_normalization.json
├── whisper_traces.npz
├── whisper_normalization.json
├── yolo_traces.npz
└── yolo_normalization.json
```

**Backup (12-metric version):**
```
data/processed_backup_12metrics/
└── ... (old 12-metric files)
```

**Reports:**
```
reports/phase4_eda_10m/
├── plots/ (27 PNG files)
├── dataset_summary.csv
├── normalization_check.csv
├── quality_checks.csv
├── split_coverage.txt
├── eda_summary.txt
└── normalization_params.txt
```

### 7. Loading Data Example

```python
import numpy as np
from pathlib import Path

# Load one workload
data = np.load('data/processed/phase1_v3/bert_traces.npz', allow_pickle=True)

# Access data
traces = data['traces']              # (55, 715, 10) - normalized [0,1]
replica_counts = data['replica_counts']  # (55,) - values 1-10
train_idx = data['train_idx']       # (49,) - training indices
val_idx = data['val_idx']           # (6,) - validation indices
metric_names = data['metric_names'] # (10,) - metric names
metadata = data['metadata']         # (55,) - pod info dicts

# Split data
X_train = traces[train_idx]         # (49, 715, 10)
X_val = traces[val_idx]             # (6, 715, 10)
r_train = replica_counts[train_idx] # (49,)
r_val = replica_counts[val_idx]     # (6,)
```

---

## LSTM Baseline Requirements

### Objective

Establish a **performance floor** for TimeGAN comparison by training simple LSTM autoencoders.

### Architecture Requirements

**Encoder-Decoder with Conditioning:**
```
Input: (batch, 715, 10) + replica_count
       ↓
Embedding: replica_count → (embedding_dim,)
       ↓
Concat: [trace_features, replica_count_embedding] at each timestep
       ↓
Encoder LSTM: → latent representation
       ↓
Decoder LSTM: → reconstructed (batch, 715, 10)
       ↓
Output: Reconstruction of input trace
```

**Key Points:**
- Reconstruction task (autoencoder)
- Conditioning on replica_count via embedding
- Per-workload models (train 5 separate)
- Simple architecture (baseline, not SOTA)

### Training Configuration

```python
HYPERPARAMETERS = {
    'latent_dim': 64,           # Latent representation size
    'hidden_dim': 128,          # LSTM hidden units
    'embedding_dim': 16,        # Replica count embedding
    'num_layers': 2,            # LSTM layers
    'dropout': 0.2,             # Dropout rate
    'batch_size': 16,           # Small dataset
    'learning_rate': 0.001,     # Adam optimizer
    'max_epochs': 200,          # With early stopping
    'patience': 20,             # Early stopping patience
    'min_delta': 0.0001        # Minimum improvement
}
```

### Success Criteria

**Baseline Performance:**
- MSE < 0.05 (normalized space)
- Variance ratio > 0.6 (captures some variation)
- Temporal coherence preserved (visual inspection)
- No NaN/Inf in reconstructions
- All 5 models converge

**Expected Limitations:**
- Smooth reconstructions (LSTM tendency)
- May lose high-frequency details
- Perfect reconstruction not expected (that's okay, it's a baseline)

### Deliverables

1. **Training Script:**
   - `scripts/phase4/train_lstm_baseline.py`
   - Handles all 5 workloads
   - Saves models and metrics

2. **Trained Models:**
   - `models/lstm_baseline/bert_lstm.pth`
   - `models/lstm_baseline/gpt2_lstm.pth`
   - `models/lstm_baseline/resnet152_lstm.pth`
   - `models/lstm_baseline/whisper_lstm.pth`
   - `models/lstm_baseline/yolo_lstm.pth`

3. **Evaluation Report:**
   - `reports/lstm_baseline/metrics.csv`
   - `reports/lstm_baseline/reconstructions/` (plots)
   - MSE, variance ratio, visual comparisons

4. **Baseline Summary:**
   - Performance metrics per workload
   - Comparison framework for TimeGAN
   - Identified limitations

---

## Things to Watch For

### During Training

1. **Loss Convergence:**
   - Should decrease steadily
   - If plateaus early → increase model capacity
   - If oscillates → reduce learning rate

2. **Whisper Model:**
   - Watch for NaN loss (due to pod_9 zeros)
   - If happens → easy to remove pod_9 and retrain
   - Expected: Should handle zeros fine (they're valid data)

3. **Overfitting:**
   - Val loss should track train loss
   - If diverges → reduce model capacity or add dropout
   - Small dataset = overfitting risk

### After Training

1. **Reconstruction Quality:**
   - Visual inspection of val set reconstructions
   - Should preserve general trends
   - May lose sharp transitions (expected)

2. **Metric Coverage:**
   - Check all 10 metrics reconstructed reasonably
   - Some metrics easier than others (GPU total should be constant)

3. **Workload Differences:**
   - Compare performance across workloads
   - Identify which are harder to model (likely Whisper)

---

## Fallback Plans

### If Whisper Training Fails (NaN loss)

**Option 1:** Remove pod_9 and retrain
```python
# Quick fix
mask = ~((replica_counts == 10) & (metadata['pod_index'] == 9))
traces_clean = traces[mask]
```

**Option 2:** Clip extreme values
```python
traces_clipped = np.clip(traces, 0.001, 0.999)
```

**Option 3:** Weight loss to reduce impact of zeros
```python
weights = (traces.sum(axis=(1,2)) > 0).astype(float) + 0.1
loss = (reconstruction_loss * weights).mean()
```

### If Training is Too Slow

**Option 1:** Reduce sequence length
```python
# Use every 2nd timestep → 715 → 358
traces_downsampled = traces[:, ::2, :]
```

**Option 2:** Reduce model size
```python
hidden_dim = 64  # instead of 128
num_layers = 1   # instead of 2
```

### If Validation Performance Poor

**Option 1:** More training data
```python
# Use 95/5 split instead of 90/10
train_size = 0.95
```

**Option 2:** Data augmentation
```python
# Add noise, time warping, etc.
```

**Option 3:** Accept baseline limitations
- LSTM is not meant to be perfect
- Just needs to be reasonable floor for TimeGAN comparison

---

## Repository State

**Current Branch:** `phase4-model-training`

**Recent Commits:**
- Phase 4a preprocessing (10 metrics)
- Phase 4a EDA (27 plots)
- Verification scripts

**Ready for Phase 4b:** 
