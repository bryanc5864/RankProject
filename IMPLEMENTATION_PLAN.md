# Rank-Order Learning for MPRA Data - Implementation Plan

## Overview
Train models on MPRA data using rank-order learning approaches to improve robustness to experimental noise and generalization to downstream tasks (CAGI5 saturation mutagenesis).

## Phase 0: Data Acquisition & Environment Setup

### 0.1 Data Sources
- [ ] **DREAM-RNN lentiMPRA data**: https://github.com/trchristensen-99/dream_rnn_lentimpra
- [ ] **CAGI5 Saturation Mutagenesis**: http://www.genomeinterpretation.org/cagi5-regulation-saturation.html

### 0.2 Environment
- [ ] Create conda/venv environment
- [ ] Install dependencies: PyTorch, torchsort, scikit-learn, pandas, numpy, scipy
- [ ] Set up project structure

### 0.3 Project Structure
```
RankProject/
├── data/
│   ├── raw/                    # Downloaded data
│   ├── processed/              # Preprocessed tensors
│   └── cagi5/                  # CAGI5 evaluation data
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # MPRA dataset class
│   │   ├── preprocessing.py    # Sequence encoding, normalization
│   │   └── curriculum.py       # Curriculum sampling
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py         # CNN/Transformer encoder
│   │   ├── heads.py            # Regression/ranking heads
│   │   └── dream_rnn.py        # DREAM-RNN replication
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── plackett_luce.py
│   │   ├── softsort.py
│   │   ├── ranknet.py
│   │   └── combined.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── curriculum_scheduler.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          # Spearman, Kendall, NDCG
│       └── cagi5_eval.py
├── configs/
│   └── experiments/            # YAML configs for each experiment
├── scripts/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate_cagi5.py
├── notebooks/
│   └── exploration.ipynb
└── results/
    └── experiments/
```

---

## Phase 1: Baseline Replication (DREAM-RNN)

### Goal
Replicate DREAM-RNN training on lentiMPRA data and verify test performance matches DEGU-Supplementary Figure 2c-d.

### Tasks
1. [ ] Clone/download DREAM-RNN code and data from GitHub
2. [ ] Understand data format (sequence encoding, activity labels, train/val/test splits)
3. [ ] Implement or adapt DREAM-RNN architecture
4. [ ] Train with MSE loss (standard regression)
5. [ ] Evaluate and compare to published metrics:
   - Pearson correlation
   - Spearman correlation
   - Per-cell-type performance

### Success Criteria
- Test Pearson/Spearman within range of published results
- Training curves match expected behavior

### Key References
- DREAM challenge paper: https://www.nature.com/articles/s41587-024-02414-w
- DREAM-RNN repo: https://github.com/trchristensen-99/dream_rnn_lentimpra

---

## Phase 2: Ranking Loss Implementations

### 2.1 Baselines
- [ ] **B1**: MSE regression (from Phase 1)
- [ ] **B2**: Soft classification (bin into 5-10 ordinal categories, cross-entropy)

### 2.2 Plackett-Luce (Primary)
- [ ] Implement listwise log-likelihood loss
- [ ] Temperature scaling hyperparameter (τ)
- [ ] List size configuration

```python
def plackett_luce_loss(scores, relevance):
    sorted_indices = relevance.argsort(descending=True)
    sorted_scores = scores.gather(-1, sorted_indices)
    cumsums = sorted_scores.flip(-1).logcumsumexp(-1).flip(-1)
    log_likelihood = (sorted_scores - cumsums).sum(-1)
    return -log_likelihood.mean()
```

### 2.3 SoftSort
- [ ] Integrate torchsort library
- [ ] Implement differentiable ranking loss
- [ ] Tune regularization strength

### 2.4 RankNet (Pairwise)
- [ ] Implement Bradley-Terry pairwise loss
- [ ] Margin-aware variant (larger activity gaps → larger score gaps)

### 2.5 Combined Losses
- [ ] Single head: `α * MSE + (1-α) * ranking_loss`
- [ ] Dual head: separate regression and ranking heads with shared backbone

---

## Phase 3: Curriculum Learning

### 3.1 Tier Assignment
- [ ] Compute per-sequence "extremeness" score (distance from median)
- [ ] Assign tiers: Tier 1 (extreme), Tier 2 (moderate), Tier 3 (close to median)

```python
def assign_tiers(y, q1=0.33, q2=0.66):
    median = y.median()
    distances = (y - median).abs()
    thresholds = distances.quantile([q1, q2])
    tiers = torch.ones_like(y, dtype=torch.long) * 3
    tiers[distances > thresholds[1]] = 1  # Most extreme
    tiers[(distances > thresholds[0]) & (distances <= thresholds[1])] = 2
    return tiers
```

### 3.2 Sampling Strategies
- [ ] **Tier-based**: Weight sampling by tier, shift weights over epochs
- [ ] **Self-paced**: Weight by loss magnitude, increase threshold over training
- [ ] **Batch composition**: Control tier proportions per batch

### 3.3 Schedules
- Linear transition
- Stepped phases
- Exponential decay

---

## Phase 4: Advanced Extensions (Optional)

### 4.1 Domain-Adversarial Training
- [ ] Experiment discriminator (predict which experiment batch)
- [ ] Gradient reversal layer
- [ ] Learn experiment-invariant features

### 4.2 Bias Factorization
- [ ] Train bias model on background sequences
- [ ] Adaptive bias correction with learned gating

### 4.3 Uncertainty-Weighted Ranking
- [ ] Weight pairs by confidence in label difference
- [ ] Heteroscedastic model (predict uncertainty)

---

## Phase 5: Evaluation on CAGI5

### 5.1 Data Preparation
- [ ] Download CAGI5 saturation mutagenesis data
- [ ] Process variant sequences for 11 enhancers + 10 promoters
- [ ] ~17,500 SNVs and small indels total

### 5.2 Metrics
**Primary (Rank-Based)**:
- Spearman ρ
- Kendall τ
- NDCG@k (k=10, 50)

**Secondary**:
- Pearson r
- AUC for over/under-expression classification
- Precision@k

### 5.3 Evaluation Protocol
- Zero-shot evaluation (no fine-tuning on CAGI5)
- Per-element breakdown (each enhancer/promoter separately)
- Aggregate statistics

---

## Experiment Matrix

| Exp | Loss | Curriculum | Domain Adv. | Bias Factor. | Notes |
|-----|------|------------|-------------|--------------|-------|
| B1 | MSE | None | No | No | Baseline |
| B2 | Soft class | None | No | No | Discretization |
| R1 | Plackett-Luce | None | No | No | Core ranking |
| R2 | SoftSort | None | No | No | Differentiable sort |
| R3 | RankNet | None | No | No | Pairwise |
| R4 | MSE + PL | None | No | No | Combined |
| C1 | Plackett-Luce | Tier-based | No | No | + Curriculum |
| C2 | Plackett-Luce | Self-paced | No | No | Adaptive |
| C3 | MSE + PL | Tier-based | No | No | Full combination |
| D1 | Plackett-Luce | Tier-based | Yes | No | + Domain adv. |
| D2 | Plackett-Luce | Tier-based | No | Yes | + Bias subtraction |
| D3 | Plackett-Luce | Tier-based | Yes | Yes | Kitchen sink |

---

## Immediate Next Steps

### This Week
1. [ ] Set up project structure and environment
2. [ ] Download DREAM-RNN data from GitHub
3. [ ] Download CAGI5 evaluation data
4. [ ] Implement basic data loading and sequence encoding

### Next Week
5. [ ] Replicate DREAM-RNN architecture
6. [ ] Train baseline MSE model
7. [ ] Verify performance matches published results

### Following Weeks
8. [ ] Implement ranking losses (Plackett-Luce first)
9. [ ] Add curriculum learning
10. [ ] Run experiment matrix
11. [ ] Evaluate on CAGI5
