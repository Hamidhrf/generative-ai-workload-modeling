# Phase 2: Model Selection Framework for Synthetic Trace Generation
## Master's Thesis: Generative AI Workload Modeling

**Author**: Hamidreza Fathollahzadeh  
**Date**: January 21, 2026  
**Phase**: 2 - Literature Review & Model Selection  
**Status**: In Progress

---

## Executive Summary

**Phase 1 Achievements:**
- 13 experiments completed across 3 AI workloads (ResNet50, DistilBERT, Whisper)
- 140,400 data points collected (15 metrics × 720 samples × 13 experiments)
- Data quality: 9.5/10
- Three distinct workload patterns identified:
  - Gradual degradation (DistilBERT)
  - Immediate contention (ResNet50, Whisper)
  - Steep resource transitions

**Phase 2 Objective:**
Select and justify a generative model architecture that can:
1. Learn from small-scale experiments (r=1-10 replicas)
2. Generate synthetic traces for large-scale scenarios (r=50-100 replicas)
3. Preserve temporal dependencies (5-second resolution, 60-minute traces)
4. Model 15 metrics jointly (CPU, GPU, memory, latency, throughput, PSI metrics)

---

## 1. Literature Review Framework

### 1.1 Research Questions

**RQ1**: Which generative model architectures are best suited for multivariate time-series generation with limited training data?

**RQ2**: How do different architectures handle temporal coherence across multiple correlated metrics?

**RQ3**: What evaluation metrics best assess the quality of synthetic workload traces?

**RQ4**: Can generative models extrapolate from small-scale to large-scale replica counts?

### 1.2 Literature Search Strategy

**Databases**: IEEE Xplore, ACM Digital Library, arXiv, Springer, ScienceDirect  
**Keywords**: 
- Primary: "time series generation", "synthetic trace generation", "workload modeling"
- Secondary: "GAN", "VAE", "LSTM", "multivariate forecasting", "resource prediction"
- Domain: "cloud computing", "kubernetes", "performance modeling"

**Inclusion Criteria**:
- Published 2019-2025 (emphasis on 2023-2025)
- Focus on time-series generation or workload prediction
- Multivariate approaches
- Evaluation with real-world data

**Exclusion Criteria**:
- Single-variate only
- Purely theoretical without implementation
- Non-temporal data

---

## 2. Model Architecture Landscape

### 2.1 TimeGAN (Time-series Generative Adversarial Network)

**Core Paper**: Yoon et al., NeurIPS 2019  
**Architecture**: Combines unsupervised GAN with supervised autoregressive model  

**Key Components**:
1. **Embedding Network**: Maps high-dimensional features to latent space
2. **Generator**: Creates synthetic latent sequences from random noise
3. **Discriminator**: Distinguishes real vs synthetic in latent space
4. **Recovery Network**: Maps latent sequences back to original feature space

**Strengths**:
- Explicitly preserves temporal dynamics through supervised loss
- Handles multivariate time series naturally
- State-of-the-art performance on various benchmarks
- Flexible backbone (RNN, GRU, LSTM, or Transformer)

**Weaknesses**:
- Training instability (common GAN issue)
- Requires careful hyperparameter tuning
- Longer training times vs VAE approaches
- Mode collapse risk

**Applicability to Your Work**:
- ✅ Designed for multivariate sequences
- ✅ Temporal coherence explicitly modeled
- ✅ Successfully used for workload modeling (IoT survey 2023)
- ⚠️ Stability concerns may require early stopping strategies
- ⚠️ Limited training data (13 experiments) may affect quality

**Recent Improvements**:
- **SeriesGAN** (2024): Addresses stability through early stopping + improved architecture
  - 34% better discriminative score vs TimeGAN
  - More consistent results across training runs
- **TTS-GAN** (2022): Transformer backbone for longer sequences
  - Better handling of irregular temporal relations
  - Arbitrary length generation

**Implementation Resources**:
- Original: https://github.com/jsyoon0823/TimeGAN
- ydata-synthetic: Easy-to-use Python library

---

### 2.2 TimeVAE (Variational Auto-Encoder for Time Series)

**Core Paper**: Desai et al., 2021  
**Architecture**: VAE with interpretable temporal components  

**Key Components**:
1. **Encoder**: RNN-based encoder maps input to latent distribution (μ, σ)
2. **Latent Space**: Probabilistic representation of time series
3. **Decoder**: Structured decoder with level + trend + seasonality components
4. **Reconstruction Loss**: KL divergence + reconstruction error

**Strengths**:
- Interpretable components (level, trend, seasonality)
- More stable training than GANs
- Reduced training time vs TimeGAN
- Naturally handles uncertainty through latent distribution
- Better performance with limited data

**Weaknesses**:
- May blur temporal fine details due to KL regularization
- Less sharp transitions compared to GANs
- Fixed decoder structure may not capture all patterns

**Applicability to Your Work**:
- ✅ Explicit temporal structure (trend/seasonality)
- ✅ Stable training (critical with limited data)
- ✅ Interpretability (important for thesis explanation)
- ✅ Naturally handles multivariate data
- ✅ Domain knowledge can be encoded

**Implementation Resources**:
- TensorFlow: https://github.com/abudesai/timeVAE
- PyTorch: https://github.com/wangyz1999/timeVAE-pytorch

---

### 2.3 LSTM/GRU-based Approaches

**Common Architectures**:
1. **Vanilla LSTM**: Basic sequence-to-sequence
2. **Stacked LSTM**: Multiple LSTM layers for hierarchical features
3. **Bidirectional LSTM**: Forward + backward temporal context
4. **LSTM Encoder-Decoder**: Sequence-to-sequence with bottleneck
5. **CNN-LSTM Hybrid**: CNN for feature extraction + LSTM for temporal

**Strengths**:
- Simple architecture
- Well-understood training dynamics
- Fast inference
- Proven effectiveness for workload prediction
- Extensive literature and implementations

**Weaknesses**:
- Deterministic (no diversity in generation without noise injection)
- Primarily designed for prediction, not generation
- May require separate model for each metric (unless multivariate LSTM)
- Limited ability to capture complex distributions

**Applicability to Your Work**:
- ✅ Excellent for baseline comparison
- ✅ Fast training and inference
- ⚠️ Not primarily a generative model
- ⚠️ Would need modifications for synthetic trace generation
- ✅ Strong performance in workload prediction literature

**Key Papers**:
- Google Cloud traces with CNN-LSTM (Yazdanian & Sharifian)
- Attention-based LSTM encoder-decoder (Zhu et al., 2019)
- esDNN with GRU for multivariate workloads (Xu et al., 2022)

---

### 2.4 Hybrid Approaches

**1. LSTM-Autoencoder + TimeGAN**:
- Use autoencoder for dimensionality reduction
- Train TimeGAN in compressed space
- Relevant: "Resource-Based Prediction Using LSTM with Autoencoders" (2023)

**2. VAE-LSTM Hybrid**:
- VAE for data augmentation
- LSTM for forecasting
- Relevant: "Time Series generation with VAE LSTM" (2025)

**3. Transformer-VAE (T-VAE)**:
- Transformer encoder for long-range dependencies
- VAE framework for generation
- Relevant: "T-VAE for Multivariate Time Series" (2025)

---

## 3. Evaluation Framework

### 3.1 Quality Metrics for Synthetic Traces

**1. Statistical Similarity**:
- **Discriminative Score**: Train classifier to distinguish real vs synthetic (lower is better)
- **Predictive Score**: Train forecaster on synthetic, test on real (higher is better)
- **Distribution Metrics**: 
  - Kullback-Leibler divergence
  - Jensen-Shannon divergence
  - Wasserstein distance

**2. Temporal Coherence**:
- **Autocorrelation Function (ACF)**: Compare real vs synthetic
- **Partial Autocorrelation (PACF)**: Higher-order dependencies
- **Dynamic Time Warping (DTW)**: Sequence similarity

**3. Multivariate Correlation**:
- **Cross-correlation matrices**: Compare metric relationships
- **Copula-based metrics**: Capture joint distributions
- **Mutual Information**: Non-linear dependencies

**4. Visual Inspection**:
- **t-SNE/PCA plots**: Latent space structure
- **Time series overlay**: Visual comparison
- **Distribution plots**: Histogram/KDE comparison

### 3.2 Workload-Specific Metrics

**1. Resource Utilization Patterns**:
- Peak-to-mean ratio preservation
- Idle vs active period distribution
- Burstiness metrics (Hurst parameter)

**2. QoS Metrics**:
- Latency distribution preservation (p50, p95, p99)
- Throughput characteristics
- SLA violation patterns

**3. Scalability Assessment**:
- Does synthetic r=10 → r=50 preserve scaling behavior?
- Are replica-level relationships maintained?
- PSI metric realism (critical for contention modeling)

---

## 4. Model Selection Criteria

### 4.1 Technical Requirements

| Criterion | Weight | TimeGAN | TimeVAE | LSTM | Hybrid |
|-----------|--------|---------|---------|------|--------|
| Multivariate support | HIGH | ✅ Excellent | ✅ Excellent | ✅ Good | ✅ Excellent |
| Temporal coherence | HIGH | ✅ Excellent | ✅ Good | ⚠️ Fair | ✅ Excellent |
| Training stability | HIGH | ⚠️ Moderate | ✅ Excellent | ✅ Excellent | ⚠️ Moderate |
| Limited data performance | HIGH | ⚠️ Moderate | ✅ Good | ✅ Excellent | ✅ Good |
| Generation diversity | MEDIUM | ✅ Excellent | ✅ Good | ❌ Poor | ✅ Good |
| Interpretability | MEDIUM | ⚠️ Moderate | ✅ Excellent | ✅ Good | ⚠️ Moderate |
| Implementation complexity | MEDIUM | ⚠️ High | ✅ Moderate | ✅ Low | ❌ High |
| Training time | LOW | ⚠️ Slow | ✅ Fast | ✅ Fast | ⚠️ Slow |

### 4.2 Thesis-Specific Considerations

**Data Constraints**:
- ✅ Limited training data (13 experiments) → **Favor VAE or LSTM**
- ✅ High quality data (9.5/10) → **All architectures viable**
- ⚠️ Need to extrapolate scale (r=10 → r=100) → **Favor generative models**

**Academic Requirements**:
- ✅ Need theoretical justification → **TimeVAE (interpretable) or TimeGAN (proven)**
- ✅ Comparison with baselines → **Include LSTM baseline**
- ✅ Clear evaluation metrics → **Use discriminative + predictive scores**

**Time Constraints**:
- ✅ Implementation deadline → **Use existing frameworks**
- ✅ Training resources → **Consider GPU availability for GANs**
- ✅ Experimentation time → **Start with simpler baseline**

---

## 5. Preliminary Recommendation

### 5.1 Primary Approach: **TimeVAE** (with TimeGAN comparison)

**Justification**:

1. **Training Stability**: With only 13 experiments, stable training is critical. VAEs converge more reliably than GANs.

2. **Interpretability**: Thesis requires clear explanation of model behavior. TimeVAE's level+trend+seasonality decomposition provides this.

3. **Multivariate Support**: Native handling of 15 metrics simultaneously.

4. **Temporal Coherence**: Structured decoder ensures temporal patterns are preserved.

5. **Limited Data Performance**: Literature shows VAEs perform better than GANs with smaller datasets.

6. **Implementation Resources**: Well-documented PyTorch/TensorFlow implementations available.

**Alternative**: TimeGAN with SeriesGAN improvements for comparison
- Provides different generation mechanism
- May capture sharper transitions (ResNet50, Whisper patterns)
- Strengthens thesis by comparing two state-of-the-art approaches

### 5.2 Baseline: **Multi-output LSTM**

**Justification**:
- Establishes lower bound for performance
- Well-established in workload prediction literature
- Fast to implement and train
- Provides comparison point for generation quality

### 5.3 Proposed Architecture Pipeline

```
Phase 2A: Baseline Development (Week 1-2)
├── Multi-output LSTM
├── Train on 80% of experiments
└── Evaluate on 20% holdout

Phase 2B: TimeVAE Implementation (Week 3-4)
├── Architecture design (15 input features)
├── Hyperparameter tuning
├── Train on full dataset
└── Generate synthetic traces

Phase 2C: TimeGAN Implementation (Week 5-6)
├── SeriesGAN-style improvements
├── Early stopping mechanism
├── Compare with TimeVAE
└── Select best performer

Phase 2D: Evaluation (Week 7)
├── Statistical metrics
├── Temporal coherence
├── Scalability tests
└── Visual analysis
```

---

## 6. Next Steps for Implementation

### 6.1 Immediate Actions (This Week)

**1. Data Preparation** ✅
```python
# Structure your data for training
# Expected shape: (n_samples, sequence_length, n_features)
# Your case: (13, 720, 15)
```

**2. Environment Setup**
```bash
conda activate tracegen
pip install tensorflow torch ydata-synthetic
```

**3. Exploratory Analysis**
- Load your 13 experiments
- Visualize each workload pattern
- Compute correlation matrices
- Identify temporal patterns

### 6.2 Week 1 Tasks

**1. Literature Review Documentation**
- Create annotated bibliography
- Summarize key papers (5-7 core papers)
- Document evaluation metrics from literature

**2. Baseline LSTM Implementation**
```python
# Multi-output LSTM for 15 metrics
# Input: (batch, 720, 15)
# Output: (batch, 720, 15)
```

**3. Data Pipeline**
- Train/validation/test split strategy
- Normalization approach (MinMax vs StandardScaler)
- Sequence windowing if needed

### 6.3 Week 2-3 Tasks

**1. TimeVAE Implementation**
- Adapt architecture to your 15 metrics
- Define interpretable components:
  - Level: baseline resource usage
  - Trend: gradual changes over time
  - Seasonality: periodic patterns (if any)

**2. Training Strategy**
```python
# Hyperparameters to tune:
- latent_dim: 8-16
- hidden_layers: [50, 100, 200]
- batch_size: 4-8 (small due to limited data)
- reconstruction_weight: 2.0-5.0
- epochs: 1000-2000
```

**3. Evaluation Framework**
- Implement discriminative score
- Implement predictive score
- Statistical comparison metrics

### 6.4 Deliverables for Phase 2

1. **Literature Review Section** (10-15 pages)
   - Taxonomy of approaches
   - Comparison table
   - Justification for selected approach

2. **Model Architecture Description** (5-8 pages)
   - Detailed architecture diagrams
   - Hyperparameter justification
   - Training procedure

3. **Preliminary Results** (5-10 pages)
   - Baseline LSTM results
   - TimeVAE training curves
   - Initial quality metrics

---

## 7. Risk Assessment & Mitigation

### 7.1 Identified Risks

**Risk 1**: Insufficient training data (13 experiments)
- **Mitigation**: Use VAE (more data-efficient), data augmentation, careful validation

**Risk 2**: Model doesn't extrapolate well to r=100
- **Mitigation**: Test intermediate scales (r=20, r=30), use conditioning if needed

**Risk 3**: Training instability (especially for TimeGAN)
- **Mitigation**: Use SeriesGAN improvements, early stopping, multiple training runs

**Risk 4**: Generated traces lack realism
- **Mitigation**: Comprehensive evaluation metrics, expert validation, visual inspection

### 7.2 Fallback Options

**Option 1**: If generative models fail → Statistical augmentation
- Bootstrap resampling
- ARIMA-based synthesis
- Still valid for thesis (document why)

**Option 2**: If extrapolation fails → Interpolation only
- Generate traces for r=10-20 instead of r=100
- Still valuable for capacity planning

**Option 3**: If multivariate models fail → Per-metric generation
- Model each metric separately
- Post-hoc correlation adjustment
- Less elegant but defensible

---

## 8. References & Resources

### 8.1 Core Papers

**TimeGAN**:
- Yoon et al., "Time-series Generative Adversarial Networks," NeurIPS 2019
- Official code: https://github.com/jsyoon0823/TimeGAN
- Tutorial: https://github.com/jsyoon0823/TimeGAN/blob/master/tutorial_timegan.ipynb

**SeriesGAN** (TimeGAN improvement):
- "SeriesGAN: Time Series Generation via Adversarial and Autoregressive Learning," Oct 2024
- arXiv: https://arxiv.org/abs/2410.21203

**TimeVAE**:
- Desai et al., "TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation," 2021
- PyTorch code: https://github.com/wangyz1999/timeVAE-pytorch
- TensorFlow code: https://github.com/abudesai/timeVAE

**Workload Prediction**:
- Xu et al., "esDNN: Deep Neural Network Based Multivariate Workload Prediction," ACM TOIT 2022
- Zhu et al., "A novel approach to workload prediction using attention-based LSTM," EURASIP 2019

**Survey Papers**:
- "Survey of Time Series Data Generation in IoT," PMC 2023
- "Generative Adversarial Networks in Time Series: A Systematic Literature Review," ACM Computing Surveys

### 8.2 Implementation Libraries

**Python Libraries**:
```python
# Generative models
ydata-synthetic  # TimeGAN implementation
tensorflow/pytorch  # Base frameworks

# Evaluation
sklearn  # Metrics, preprocessing
dtaidistance  # Dynamic Time Warping
scipy  # Statistical tests

# Visualization
matplotlib
seaborn
plotly
```

**Kubernetes Metrics**:
- Your existing Prometheus data
- Data structure: timestamp, metric_name, value, labels

---

## 9. Academic Writing Guidelines

### 9.1 Literature Review Structure

**Introduction** (1 page)
- Problem statement
- Research questions
- Search methodology

**Background** (2-3 pages)
- Time series generation overview
- GAN vs VAE approaches
- Workload modeling context

**State of the Art** (6-8 pages)
- TimeGAN and variants
- VAE approaches
- LSTM baselines
- Hybrid methods
- Comparison table

**Gap Analysis** (1-2 pages)
- What's missing in current approaches
- How your work addresses gaps

**Selected Approach** (2-3 pages)
- Justification for TimeVAE + TimeGAN
- Expected advantages
- Potential limitations

### 9.2 Evaluation Section Structure

**Methodology** (2-3 pages)
- Train/test split strategy
- Hyperparameter tuning approach
- Evaluation metrics definition

**Baseline Results** (2-3 pages)
- LSTM performance
- Establishes lower bound

**Generative Model Results** (4-6 pages)
- TimeVAE performance
- TimeGAN performance
- Comparison analysis

**Discussion** (2-3 pages)
- Which model performs best and why
- Limitations observed
- Practical implications

---

## 10. Timeline Recommendation

### Week-by-Week Breakdown

**Week 1** (Jan 21-27): Literature Review + Baseline
- Complete literature review
- Implement LSTM baseline
- Document results

**Week 2** (Jan 28-Feb 3): TimeVAE Implementation
- Adapt TimeVAE to your data
- Initial training runs
- Hyperparameter exploration

**Week 3** (Feb 4-10): TimeVAE Refinement
- Optimize hyperparameters
- Generate synthetic traces
- Initial evaluation

**Week 4** (Feb 11-17): TimeGAN Implementation
- Implement TimeGAN with SeriesGAN improvements
- Training and generation
- Stability assessment

**Week 5** (Feb 18-24): Comprehensive Evaluation
- All evaluation metrics
- Statistical tests
- Visual comparisons

**Week 6** (Feb 25-Mar 3): Analysis & Writing
- Select best model
- Document methodology
- Write Phase 2 section

**Week 7** (Mar 4-10): Buffer & Integration
- Address any issues
- Integrate with Phase 1 write-up
- Prepare for Phase 3

---

## Conclusion

Phase 2 represents the theoretical and methodological core of your thesis. The recommendation to pursue **TimeVAE as primary approach with TimeGAN comparison** is based on:

1. Your limited but high-quality training data
2. Need for interpretable results
3. Multivariate requirements (15 metrics)
4. Academic rigor (comparing state-of-the-art approaches)

The baseline LSTM provides a solid foundation for comparison and establishes that your generative approaches actually add value beyond simple prediction.

This framework provides a clear roadmap for the next 6-7 weeks of work. Each week builds on the previous, allowing for iterative refinement and risk mitigation.

**Next immediate step**: Let's start with exploratory data analysis of your Phase 1 results to inform model design.

---

*Document Version: 1.0*  
*Last Updated: January 21, 2026*  
*Author: Research Framework for Hamidreza Fathollahzadeh*
