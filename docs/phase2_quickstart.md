# Phase 2 Quick-Start Guide
## Immediate Actions for This Week

**Date**: January 21, 2026  
**Goal**: Get started with model selection and baseline implementation

---

## Today's Tasks (2-3 hours)

### 1. Review the Framework Document 
Read the comprehensive framework: `phase2_model_selection_framework.md`

Key sections to focus on:
- Section 2: Model Architecture Landscape
- Section 5: Preliminary Recommendation
- Section 6: Next Steps for Implementation

### 2. Verify Your Data Structure

Run this quick check:
```bash
cd ~/generative-ai-workload-modeling/data/raw/phase1/
ls -lh
```

Expected files:
- 13 CSV files (one per experiment)
- Each with 720 rows (60 min × 12 samples/min = 720 samples)
- Each with 15 metrics + timestamp

### 3. Set Up Your Python Environment

```bash
conda activate tracegen

# Install additional libraries for Phase 2
pip install ydata-synthetic
pip install dtaidistance  # For DTW evaluation
pip install scikit-learn  # Latest version for metrics

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
```

---

## Tomorrow's Tasks (Day 2)

### 1. Exploratory Data Analysis

Use the provided script: `phase2_eda.py`

```bash
cd ~/generative-ai-workload-modeling
python scripts/phase2_eda.py
```

This will generate:
- Correlation matrices for each workload
- Temporal pattern visualizations
- Statistical summaries
- Data quality checks

### 2. Read Core Papers

**Priority 1** (Read today):
- TimeVAE paper: https://arxiv.org/abs/2111.08095
- Read sections: Introduction, Methodology, Results

**Priority 2** (Read tomorrow):
- TimeGAN paper: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
- Focus on: Architecture, Training procedure

**Priority 3** (Read this week):
- SeriesGAN improvements: https://arxiv.org/abs/2410.21203
- Survey paper: "Survey of Time Series Data Generation in IoT"

### 3. Start Literature Review Document

Create: `docs/phase2_literature_review.md`

Structure:
```markdown
# Phase 2 Literature Review

## 1. Introduction
- Problem statement
- Research questions

## 2. Background
- Time series generation overview
- Key concepts

## 3. State of the Art
### 3.1 GAN-based Approaches
- TimeGAN
- SeriesGAN
- TTS-GAN

### 3.2 VAE-based Approaches
- TimeVAE
- VAE-LSTM

### 3.3 Baseline Approaches
- LSTM/GRU
- Transformer models

## 4. Comparison & Selection
[To be filled after reading papers]

## 5. References
```

---

## This Week's Milestones

### Day 3-4: Baseline LSTM Implementation

**Goal**: Get a working prediction model as baseline

Key files to create:
- `scripts/models/lstm_baseline.py` - Model definition
- `scripts/models/train_baseline.py` - Training script
- `scripts/models/evaluate_baseline.py` - Evaluation

Expected output:
- Trained LSTM model
- Prediction accuracy metrics
- Baseline performance report

### Day 5-7: Data Pipeline & Evaluation Framework

**Goal**: Prepare for generative model training

Key components:
1. **Data loader** for 13 experiments
2. **Normalization** strategy
3. **Train/validation/test** split
4. **Evaluation metrics** implementation:
   - Discriminative score
   - Predictive score
   - Statistical metrics (KL div, Wasserstein)
   - Visual comparison tools

---

## Week 2 Preview: TimeVAE Implementation

### Preparation Steps

1. **Study TimeVAE code**:
   ```bash
   git clone https://github.com/wangyz1999/timeVAE-pytorch.git
   cd timeVAE-pytorch
   # Review: src/vae/timevae.py
   ```

2. **Adapt to your data**:
   - Input: 15 features (your metrics)
   - Sequence length: 720 timesteps
   - Batch size: 4-8 (small due to 13 experiments)

3. **Design experiments**:
   - Hyperparameter grid
   - Training strategies
   - Validation approach

---

## Key Decisions to Make This Week

### Decision 1: Data Split Strategy

**Option A**: Leave-one-workload-out
- Train on ResNet50 + DistilBERT, test on Whisper
- Tests generalization across workload types

**Option B**: Leave-one-replica-out
- Train on r=1,2,3,6,8, test on r=10
- Tests scalability prediction

**Option C**: Random 80/20 split
- Standard approach
- May not test key properties

**Recommendation**: Option B (aligns with thesis goal of extrapolation)

### Decision 2: Normalization Approach

**Option A**: Global normalization (all experiments together)
- Pro: Preserves relative scales
- Con: May dominate training with high-value metrics

**Option B**: Per-metric normalization
- Pro: Equal importance to all metrics
- Con: Loses scale relationships

**Option C**: Per-workload normalization
- Pro: Workload-specific patterns
- Con: Harder to compare across workloads

**Recommendation**: Option B for initial experiments, then compare

### Decision 3: Sequence Windowing

**Question**: Use full 720-step sequences or shorter windows?

**Full sequences** (720 steps):
- Pro: Captures complete experiment
- Con: Memory intensive, harder to train

**Windowed approach** (e.g., 120 steps):
- Pro: More training samples (sliding window)
- Con: May miss long-term patterns

**Recommendation**: Start with full sequences (you have GPU), window if needed

---

## Helpful Commands Reference

### Git Workflow
```bash
# Start Phase 2 work
cd ~/generative-ai-workload-modeling
git pull origin main
git checkout -b phase2-model-selection

# Regular commits
git add .
git commit -m "feat: initial LSTM baseline implementation"
git push origin phase2-model-selection
```

### Jupyter Notebook for Exploration
```bash
cd ~/generative-ai-workload-modeling
jupyter notebook notebooks/phase2_exploration.ipynb
```

### Quick Data Check
```python
import pandas as pd
import glob

# Load all experiment files
files = glob.glob('data/raw/phase1/*.csv')
print(f"Found {len(files)} experiment files")

# Check first file
df = pd.read_csv(files[0])
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

---

## Success Criteria for Week 1

By end of Week 1, you should have:

 Literature review structure created  
 5-7 core papers read and summarized  
 Phase 1 data explored and visualized  
 LSTM baseline implemented  
 Baseline results documented  
 Data pipeline ready for generative models  
 Evaluation framework designed  

---

## Getting Help

If you get stuck:

1. **Check the main framework document** - Most questions answered there
2. **Review example implementations**:
   - TimeVAE: https://github.com/wangyz1999/timeVAE-pytorch
   - TimeGAN: https://github.com/jsyoon0823/TimeGAN
3. **Test with toy data first** - Verify your pipeline with simple synthetic data
4. **Document issues** - Keep a running log of problems and solutions

---

## Tips for Success

**1. Start Simple**
- Get LSTM baseline working first
- Then add complexity

**2. Validate Early**
- Check data shapes at every step
- Visualize outputs frequently
- Unit test your pipeline

**3. Document as You Go**
- Code comments
- Markdown notes
- Experiments log

**4. Version Control**
- Commit after each working feature
- Descriptive commit messages
- Branch per major feature

**5. Time Management**
- 2-3 hours daily is better than 10 hours once a week
- Set specific daily goals
- Track progress

---

## Quick Reference: Your Data Structure

```
Workload: ResNet50
Experiments: r=1, r=2, r=3, r=6, r=10 (5 experiments)

Workload: DistilBERT  
Experiments: r=1, r=2, r=6, r=10 (4 experiments)

Workload: Whisper
Experiments: r=1, r=2, r=3, r=8 (4 experiments)

Total: 13 experiments
Per experiment: 720 samples × 15 metrics = 10,800 data points
Total dataset: 140,400 data points
```

Metrics (15 total):
1. CPU utilization (%)
2. Memory usage (MB/GB)
3. GPU utilization (%)
4. GPU memory (MB)
5. Latency p50 (ms)
6. Latency p95 (ms)
7. Latency p99 (ms)
8. Throughput (requests/sec)
9. PSI CPU some (%)
10. PSI CPU full (%)
11. PSI memory some (%)
12. PSI memory full (%)
13. PSI IO some (%)
14. PSI IO full (%)
15. [Additional metric if applicable]

---

## Resource Allocation

**GPU Usage**:
- NVIDIA A16 with 10 virtual slices
- Should be sufficient for training
- Monitor with `nvidia-smi` during training

**Storage**:
- Phase 1 data: ~50MB
- Models: ~100-500MB each
- Generated traces: ~500MB-1GB
- Total estimate: ~2GB for Phase 2

**Time Estimate**:
- LSTM baseline: 1-2 hours training
- TimeVAE: 4-8 hours training
- TimeGAN: 8-16 hours training
- Total for Phase 2: ~30-40 hours over 6-7 weeks

---

Good luck with Phase 2! You have excellent Phase 1 results to build on.

Remember: **Start simple, validate early, document everything**.

---

*Quick-Start Guide Version: 1.0*  
*Created: January 21, 2026*
