# DISENTANGLE: Learning Biology, Not Noise
## A Framework for Noise-Resistant Sequence-to-Function Modeling

### Complete Research Plan — Agent-Executable Specification

---

# TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Background and Motivation](#2-background-and-motivation)
3. [Research Questions](#3-research-questions)
4. [Phase 1: Environment Setup and Data Acquisition](#4-phase-1-environment-setup-and-data-acquisition)
5. [Phase 2: Noise Characterization](#5-phase-2-noise-characterization)
6. [Phase 3: Training Framework Development](#6-phase-3-training-framework-development)
7. [Phase 4: Evaluation Framework](#7-phase-4-evaluation-framework)
8. [Phase 5: Full Experimental Campaign](#8-phase-5-full-experimental-campaign)
9. [Phase 6: Interpretability Analysis](#9-phase-6-interpretability-analysis)
10. [Phase 7: Paper Writing](#10-phase-7-paper-writing)
11. [Appendix A: Complete Code Architecture](#appendix-a-complete-code-architecture)
12. [Appendix B: Statistical Analysis Protocols](#appendix-b-statistical-analysis-protocols)
13. [Appendix C: Figure Specifications](#appendix-c-figure-specifications)
14. [Appendix D: Troubleshooting Guide](#appendix-d-troubleshooting-guide)

---

# 1. PROJECT OVERVIEW

## 1.1 One-Paragraph Summary

Current sequence-to-function models achieve high accuracy by learning everything that predicts training labels—including systematic experimental noise that is reproducible within an experiment but not across experiments. This is invisible under standard evaluation because train/test splits share the same noise structure. We propose DISENTANGLE, a framework that (1) characterizes what noise genomic models actually learn, (2) provides training strategies that force models to learn experiment-invariant biology instead, and (3) introduces cross-experiment transfer evaluation as the proper test of whether a model has learned regulatory grammar or experimental artifacts. We demonstrate that models trained with DISENTANGLE show modestly reduced within-experiment accuracy (because they can no longer exploit noise) but dramatically improved cross-experiment transfer—revealing that what the field has been calling "accuracy" is partly an illusion created by evaluating in the same noise regime used for training.

## 1.2 Key Contributions

1. **Empirical characterization** of what noise sequence-to-activity models learn, using representation analysis of models trained on overlapping datasets
2. **Three complementary noise-resistance strategies**: cross-experiment consensus targets, noise-contrastive representation learning, and adaptive margin ranking
3. **A four-tier evaluation framework** that distinguishes genuine biological learning from noise exploitation
4. **Architecture-agnostic training protocol** that wraps around any existing sequence model
5. **Open-source benchmark** for cross-experiment transfer evaluation in regulatory genomics

## 1.3 Repository Structure

```
disentangle/
├── README.md
├── setup.py
├── requirements.txt
├── configs/
│   ├── data/
│   │   ├── encode_lentimpra.yaml
│   │   ├── dream_lentimpra.yaml
│   │   ├── whg_starrseq.yaml
│   │   ├── atac_starrseq.yaml
│   │   ├── klein_multi_design.yaml
│   │   └── cagi5.yaml
│   ├── models/
│   │   ├── cnn_basset.yaml
│   │   ├── dilated_cnn_basenji.yaml
│   │   ├── bilstm_dream.yaml
│   │   └── transformer_lite.yaml
│   ├── training/
│   │   ├── baseline_mse.yaml
│   │   ├── ranking_only.yaml
│   │   ├── contrastive_only.yaml
│   │   ├── consensus_only.yaml
│   │   ├── ranking_contrastive.yaml
│   │   └── full_disentangle.yaml
│   └── experiments/
│       ├── noise_characterization.yaml
│       ├── ablation_study.yaml
│       ├── transfer_evaluation.yaml
│       └── interpretability.yaml
├── data/
│   ├── download.py
│   ├── preprocess.py
│   ├── pair_sequences.py
│   └── splits.py
├── models/
│   ├── __init__.py
│   ├── encoders/
│   │   ├── cnn.py
│   │   ├── dilated_cnn.py
│   │   ├── bilstm.py
│   │   └── transformer.py
│   ├── heads.py
│   └── wrapper.py
├── training/
│   ├── __init__.py
│   ├── losses/
│   │   ├── mse.py
│   │   ├── ranking.py
│   │   ├── contrastive.py
│   │   ├── consensus.py
│   │   └── disentangle.py
│   ├── trainer.py
│   └── curriculum.py
├── evaluation/
│   ├── __init__.py
│   ├── tier1_within_experiment.py
│   ├── tier2_cross_experiment.py
│   ├── tier3_cross_assay.py
│   ├── tier4_representation_quality.py
│   └── metrics.py
├── analysis/
│   ├── noise_characterization.py
│   ├── representation_probing.py
│   ├── attribution_analysis.py
│   └── motif_validation.py
├── scripts/
│   ├── run_noise_characterization.sh
│   ├── run_training.sh
│   ├── run_ablation.sh
│   ├── run_evaluation.sh
│   └── run_interpretability.sh
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_noise_characterization.ipynb
│   ├── 03_training_results.ipynb
│   ├── 04_transfer_evaluation.ipynb
│   └── 05_interpretability.ipynb
└── paper/
    ├── main.tex
    ├── figures/
    ├── tables/
    └── supplementary/
```

## 1.4 Timeline

| Month | Phase | Deliverable |
|-------|-------|-------------|
| 1 | Environment + Data | All datasets downloaded, preprocessed, paired sequences identified |
| 2 | Noise Characterization | Figures 1–3: empirical evidence models learn noise |
| 3–4 | Training Framework | All loss functions implemented, unit-tested, initial training runs |
| 5 | Evaluation Framework | All 4 tiers implemented, baseline models fully evaluated |
| 6–7 | Full Experimental Campaign | Complete ablation study across 4 architectures × 6 training conditions |
| 8 | Interpretability | Attribution analysis, motif validation, noise pattern discovery |
| 9 | Statistical Analysis | All significance tests, confidence intervals, effect sizes |
| 10 | Paper Draft | Complete manuscript with all figures |
| 11 | Revision | Internal review, additional experiments if needed |
| 12 | Submission | Final manuscript, code release, benchmark release |

---

# 2. BACKGROUND AND MOTIVATION

## 2.1 The Noise Problem in Genomics

### 2.1.1 What Is Experimental Noise in Functional Genomics?

When a functional assay (MPRA, STARR-seq, lentiMPRA) measures the regulatory activity of a DNA sequence, the measured value contains:

```
measured_activity = true_cis_regulatory_activity
                  + cell_type_context
                  + chromatin_environment
                  + assay_chemistry_bias
                  + library_preparation_bias
                  + sequencing_depth_effect
                  + PCR_amplification_bias
                  + barcode_specific_effect
                  + batch_effect
                  + stochastic_measurement_error
```

The first three components (true activity, cell type, chromatin) are **biological**. The remaining seven are **technical noise**. Current models are trained to predict the full measured_activity, meaning they learn all ten components. At inference on a new experiment, only the biological components transfer.

### 2.1.2 Why Noise Is Learnable (Not Just Random)

Critical distinction: most technical noise is **systematic and sequence-dependent**, meaning a model CAN learn it from sequence alone:

- **GC bias**: PCR amplification efficiency depends on GC content. GC-rich sequences get systematically different amplification in each library prep protocol. The model can learn "GC-rich → higher/lower activity" as if it were regulatory grammar.
- **Cloning bias**: Restriction enzyme efficiency varies by flanking sequence. Sequences near certain motifs get differentially represented in the plasmid library.
- **Barcode effects**: In MPRA, each sequence is tagged with barcodes. Barcode-specific RNA stability or processing introduces sequence-correlated noise.
- **Position effects**: In episomal assays, vector copy number and episomal chromatin state vary. In integrated assays, integration site effects create locus-dependent noise.
- **Library complexity**: Under-represented sequences have noisier estimates. Representation correlates with sequence properties (GC, length, secondary structure).

Because these are sequence-dependent, a sufficiently expressive model (like a deep neural network) will learn them. Within the training experiment, learning noise HELPS prediction accuracy—this is the fundamental problem.

### 2.1.3 Quantitative Evidence of Cross-Experiment Inconsistency

| Comparison | Metric | Value | Source |
|------------|--------|-------|--------|
| MPRA cross-lab (K562) | Pearson of log2(RNA/DNA) | 0.18–0.47 | Genome Biology 2025 |
| Basenji biological replicates | Pearson correlation | 0.479 | Kelley et al. 2018 |
| MAQC cross-platform (same RNA) | Gene list overlap | 74% across platforms vs 89% within | Nat Biotech 2006 |
| MAQC cross-platform (same RNA) | Rank correlation of log ratios | R ≥ 0.69 | Nat Biotech 2006 |
| ChIP-seq replicates (ENCODE) | Concordance rate | Only 2/3 concordant | Bailey et al. 2013 |
| Luciferase reporter replicates | Luminescence variation | Up to 10-fold | Methods Mol Biol 2019 |
| 9 MPRA designs, same enhancers | Design-dependent variation | Significant differences | Klein et al. 2020 |

**Key observation**: Absolute values show poor cross-experiment agreement (ρ = 0.18–0.47), but rank-order relationships are better preserved (R ≥ 0.69 across platforms in MAQC). This is the empirical foundation of our approach.

### 2.1.4 Evidence That Current Models Learn Noise

| Finding | Implication | Source |
|---------|------------|--------|
| Enformer, Basenji2, ExPecto, Xpresso often predict wrong direction of variant effects | Models learned experiment-specific scaling, not regulatory grammar | Nat Genet 2023 |
| 5 replicates of Basenji2 disagree on >50% of eQTL directions | Signal is not robustly encoded; noise dominates | Bajwa et al. |
| Cross-individual prediction is poor despite high cross-gene prediction | Models learn what distinguishes genes (promoter vs enhancer) but not what distinguishes alleles (variant effects) | Nat Genet 2023 |
| Performance drops sharply for distal enhancers | Long-range predictions are more noise-dependent | Karollus et al. 2023 |

## 2.2 Why This Problem Has Not Been Solved

1. **Evaluation hiding the problem**: Within-experiment random splits don't reveal noise learning. Papers report high accuracy, reviewers accept, nobody checks cross-experiment transfer systematically.

2. **No standardized cross-experiment benchmarks**: Unlike computer vision (ImageNet → domain adaptation benchmarks), genomics lacks curated benchmarks specifically designed to test cross-experiment generalization.

3. **Insufficient overlap**: Most MPRA studies use unique sequence libraries, making direct cross-experiment comparison impossible. We solve this by focusing on cell types (K562) where multiple studies have overlapping sequences.

4. **Computational cost**: Training large models multiple times under different conditions is expensive. Prior work has focused on getting one model to work well, not comparing training paradigms.

## 2.3 Our Approach in Context

| Existing Approach | What It Does | Limitation |
|---|---|---|
| **ComBat** | Statistical correction of known batch effects | Requires batch labels; can remove real biology; applies to expression matrices, not sequence models |
| **Domain adaptation** | Learn domain-invariant features for transfer | Applied in scRNA-seq cell type annotation, not in sequence-to-function modeling |
| **Data augmentation** | Add noise/perturbations during training | Adds random noise, doesn't address systematic sequence-dependent artifacts |
| **Ensemble methods** | Average predictions across models/batches | Reduces variance but doesn't remove systematic bias |
| **Rank-based evaluation** | Use Spearman instead of Pearson | Changes metric only, doesn't change what model learns |

**Our contribution**: We change what the model LEARNS (via loss functions and representation objectives), not just how we evaluate it. This is upstream of all existing approaches.

---

# 3. RESEARCH QUESTIONS

## Question 1: What noise do sequence-to-activity models learn?

**Sub-questions**:
- Q1a: Do models trained on different experiments for the same cell type learn different representations for the same sequences?
- Q1b: Can experiment/batch identity be predicted from learned representations?
- Q1c: Which representation dimensions are noise-predictive vs biology-predictive?
- Q1d: Do input attributions highlight known technical artifacts or known regulatory motifs?

**How we answer these**: Train identical models on different experiments, extract representations, perform probing analysis and attribution comparison. (Phase 2)

## Question 2: How can we make a model noise-aware?

**Sub-questions**:
- Q2a: Does training on rank-order consensus across experiments improve transfer?
- Q2b: Does noise-contrastive representation learning (same sequence, different experiment = positive pair) improve transfer?
- Q2c: Does adaptive margin ranking (down-weighting unreliable comparisons) improve transfer?
- Q2d: Which combination of strategies is most effective?
- Q2e: Are these strategies architecture-agnostic?

**How we answer these**: Implement three strategies, test each independently and in combination across four architectures. (Phase 3 + Phase 5)

## Question 3: How do we prove noise resistance?

**Sub-questions**:
- Q3a: Does DISENTANGLE reduce within-experiment performance (indicating noise was being exploited)?
- Q3b: Does DISENTANGLE improve cross-experiment transfer (Tier 2)?
- Q3c: Does DISENTANGLE improve cross-assay transfer (Tier 3)?
- Q3d: Are DISENTANGLE representations less batch-predictive (Tier 4)?
- Q3e: Is the characteristic "lower Tier 1, higher Tier 2" pattern consistent across architectures?

**How we answer these**: Four-tier evaluation framework with pre-registered hypotheses. (Phase 4 + Phase 5)

## Question 4: How does this change the field?

**Sub-questions**:
- Q4a: Can we release a training protocol that wraps any existing architecture?
- Q4b: Can we release a benchmark that the community adopts?
- Q4c: What recommendations can we make for experimental design to improve computational transferability?

**How we answer these**: Package framework, release benchmark with evaluation scripts, write field-facing recommendations. (Phase 7)

---

# 4. PHASE 1: ENVIRONMENT SETUP AND DATA ACQUISITION

**Duration**: Month 1
**Goal**: All datasets downloaded, preprocessed, quality-controlled, and paired sequences identified.

## 4.1 Computational Environment

### 4.1.1 Hardware Requirements

- **GPU**: Minimum 4× A100 (80GB) or equivalent. 8× preferred for parallel ablation runs.
- **CPU**: 64+ cores for data preprocessing
- **RAM**: 256 GB minimum (some datasets are large when loaded)
- **Storage**: 5 TB SSD (datasets ~1TB, checkpoints ~2TB, intermediate files ~1TB)

### 4.1.2 Software Environment Setup

```bash
# Step 1: Create conda environment
conda create -n disentangle python=3.10 -y
conda activate disentangle

# Step 2: Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Step 3: Install core dependencies
pip install \
    numpy==1.26.4 \
    pandas==2.2.0 \
    scipy==1.12.0 \
    scikit-learn==1.4.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.1 \
    plotly==5.18.0 \
    h5py==3.10.0 \
    pysam==0.22.0 \
    pybedtools==0.9.1 \
    biopython==1.83 \
    pyBigWig==0.3.22

# Step 4: Install ML dependencies
pip install \
    transformers==4.37.0 \
    einops==0.7.0 \
    wandb==0.16.2 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    lightning==2.1.3

# Step 5: Install genomics tools
conda install -c bioconda bedtools samtools tabix -y

# Step 6: Install analysis dependencies
pip install \
    captum==0.7.0 \
    umap-learn==0.5.5 \
    adjustText==0.8

# Step 7: Install development tools
pip install \
    pytest==7.4.4 \
    black==24.1.0 \
    isort==5.13.2 \
    mypy==1.8.0

# Step 8: Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### 4.1.3 Reference Genome Setup

```bash
# Download hg38 reference genome
mkdir -p data/reference
cd data/reference
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
samtools faidx hg38.fa

# Download hg38 chromosome sizes
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes

# Download hg19 reference (some datasets use hg19)
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz
samtools faidx hg19.fa

# Download chain file for liftOver
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
```

## 4.2 Dataset Acquisition

### 4.2.1 Dataset Overview

| ID | Dataset | Assay | Cell Type | Sequences | Role | Priority |
|----|---------|-------|-----------|-----------|------|----------|
| D1 | ENCODE lentiMPRA | Integrated reporter | K562 | ~680,000 | Training (primary) | CRITICAL |
| D2 | ENCODE lentiMPRA | Integrated reporter | HepG2 | ~680,000 | Cross-cell-type test | HIGH |
| D3 | DREAM Challenge | lentiMPRA | K562 | ~100,000 | Training + baseline | CRITICAL |
| D4 | WHG-STARR-seq (K562) | Episomal reporter | K562 | Genome-wide | Training + cross-assay | HIGH |
| D5 | ATAC-STARR-seq (K562) | Episomal reporter | K562 | Genome-wide | Held-out Tier 2 | CRITICAL |
| D6 | Klein et al. multi-design | 9 MPRA designs | K562/HepG2 | 2,440 enhancers | Cross-design analysis | MEDIUM |
| D7 | Kircher saturation mutagenesis | Saturation mutagenesis | Multiple | ~17,500 SNVs | Tier 3 evaluation | HIGH |
| D8 | CAGI5 regulatory saturation | Saturation mutagenesis | Multiple | ~17,500 SNVs | Zero-shot Tier 3 | CRITICAL |
| D9 | GTEx v8 eQTLs | In vivo | 48 tissues | ~3,259 genes | Variant direction test | MEDIUM |

### Phase 1 Completion Checklist

```
□ Conda environment created and verified (GPU check passes)
□ Reference genomes downloaded and indexed (hg38 + hg19)
□ All priority CRITICAL datasets downloaded (D1, D3, D5, D8)
□ All priority HIGH datasets downloaded (D2, D4, D7)
□ All datasets preprocessed to HDF5 format
□ QC checks passed for all datasets
□ Paired sequences identified (N > 500 minimum for contrastive learning)
□ Train/val/test splits created with experiment-awareness
□ All data files < 5 TB total storage
□ Data exploration notebook (01_data_exploration.ipynb) completed
```

---

# 5. PHASE 2: NOISE CHARACTERIZATION

**Duration**: Month 2
**Goal**: Empirically demonstrate that current models learn experiment-specific noise. Produce Figures 1–3 for the paper.

## 5.1 Experiment 2.1: Paired-Experiment Representation Divergence

### 5.1.1 Protocol

1. Train **three identical models** (same architecture, same hyperparameters) on three different K562 datasets independently:
   - Model_A: ENCODE lentiMPRA K562 (D1)
   - Model_B: DREAM lentiMPRA K562 (D3)
   - Model_C: WHG-STARR-seq K562 (D4)

2. Extract penultimate-layer representations for all **paired sequences** (sequences present in 2+ experiments).

3. Measure representational similarity using CKA (Centered Kernel Alignment).

4. Visualize with UMAP, coloring by experiment ID.

### 5.1.4 Expected Results and Interpretation

**Expected**: CKA between models trained on different experiments will be **low** (< 0.5), even though all models predict activity well within their own experiment. UMAP will show representations clustering by experiment, not by sequence identity.

**If CKA is high (> 0.8)**: Models DO learn similar representations regardless of training experiment. Our premise is partially wrong—noise may be less of a problem than hypothesized.

## 5.2 Experiment 2.2: Batch-Predictive Features (Linear Probing)

### 5.2.4 Expected Results

- **Experiment probe accuracy**: > 70% (well above chance of 33% for 3 experiments)
- **Activity probe R²**: > 0.5
- **Batch-activity feature overlap**: > 0.3
- **GC probe R²**: Will reveal if GC content is a major confound

## 5.3 Experiment 2.3: Attribution Analysis

### Phase 2 Completion Checklist

```
□ Three separate models trained (one per K562 experiment) for each architecture
□ Representations extracted for all paired sequences
□ CKA computed: expect < 0.5 across experiments
□ UMAP plotted: expect experiment-based clustering (Figure 1A)
□ Combined model trained (all experiments, MSE)
□ Linear probes run: experiment probe >> chance level
□ Batch-activity feature overlap computed: expect > 0.3
□ Attributions computed for paired sequences across model pairs
□ Attribution similarity measured: expect < 0.5 correlation
□ Figure 1 (representation divergence) complete
□ Figure 2 (probing results) complete
□ Figure 3 (attribution comparison) complete
□ Noise characterization notebook (02_noise_characterization.ipynb) complete
```

---

# 6. PHASE 3: TRAINING FRAMEWORK DEVELOPMENT

**Duration**: Months 3–4
**Goal**: Implement all three noise-resistance strategies. Unit test each. Run initial training.

## 6.1 Model Architectures

Four architectures all implementing the same BaseEncoder interface:
- CNN (Basset-style): 3 conv blocks → FC → prediction
- Dilated CNN (Basenji-style): dilated convolutions for growing receptive field
- BiLSTM (DREAM-RNN style): conv + bidirectional LSTM
- Transformer (Enformer-lite): conv reduction + transformer layers

## 6.2 Loss Functions

### 6.2.1 Strategy 1: Cross-Experiment Consensus Loss
Rank activities within each experiment, average ranks across experiments, use as training targets.

### 6.2.2 Strategy 2: Noise-Contrastive Representation Loss
InfoNCE loss with: Anchor (seq X in exp A), Positive (seq X in exp B), Negative (seq Y in exp A).

### 6.2.3 Strategy 3: Adaptive Margin Ranking Loss
Pairwise margin ranking that down-weights unreliable comparisons (small activity differences).

### 6.2.4 Combined DISENTANGLE Loss
Weighted combination of all three strategies.

## 6.3 Experiment-Conditional Architecture Wrapper
DisentangleWrapper with experiment-conditional batch normalization; denoised inference averages across normalizations.

### Phase 3 Completion Checklist

```
□ All 4 encoder architectures implemented and tested
□ ConsensusLoss implemented and unit-tested
□ NoiseContrastiveLoss implemented and unit-tested
□ AdaptiveMarginRankingLoss implemented and unit-tested
□ DisentangleLoss combining all three implemented and unit-tested
□ DisentangleWrapper implemented
□ Training loop with wandb logging functional
□ Paired sequence dataloader functional
□ Initial training run completes without errors
□ All unit tests pass: pytest tests/ -v
```

---

# 7. PHASE 4: EVALUATION FRAMEWORK

**Duration**: Month 5
**Goal**: Implement all four evaluation tiers. Fully evaluate baseline models.

## 7.1 Tier 1: Within-Experiment Evaluation
Standard random split evaluation. Sanity check.

## 7.2 Tier 2: Cross-Experiment Transfer
Train on {A, B, C}, evaluate on held-out D. **THE MAIN EVALUATION.**

## 7.3 Tier 3: Cross-Assay Transfer
CAGI5 zero-shot variant effect prediction, HepG2 cross-cell-type.

## 7.4 Tier 4: Representation Quality
Linear probing for experiment identity, activity, batch-activity overlap.

### Phase 4 Completion Checklist

```
□ Tier 1 evaluation implemented and tested
□ Tier 2 evaluation implemented
□ Tier 3 evaluation implemented (CAGI5)
□ Tier 4 evaluation implemented (probing)
□ NDCG@k and direction accuracy implemented
□ All baseline models evaluated on all 4 tiers
□ Baseline results documented
```

---

# 8. PHASE 5: FULL EXPERIMENTAL CAMPAIGN

**Duration**: Months 6–7
**Goal**: Complete ablation study. All models trained and evaluated.

## 8.1 Experimental Grid

Total: **4 architectures × 6 training conditions × 3 seeds = 72 runs**

| ID | Condition | Consensus | Contrastive | Ranking | MSE |
|----|-----------|:-:|:-:|:-:|:-:|
| C0 | Baseline (MSE only) | ✗ | ✗ | ✗ | ✓ |
| C1 | Ranking only | ✗ | ✗ | ✓ | ✗ |
| C2 | Contrastive only | ✗ | ✓ | ✗ | ✓ |
| C3 | Consensus only | ✓ | ✗ | ✗ | ✗ |
| C4 | Ranking + Contrastive | ✗ | ✓ | ✓ | ✗ |
| C5 | Full DISENTANGLE | ✓ | ✓ | ✓ | ✗ |

## 8.3 Pre-Registered Hypotheses

**H1**: Full DISENTANGLE (C5) achieves higher Tier 2 Spearman than baseline MSE (C0) across all 4 architectures.
**H2**: Full DISENTANGLE (C5) achieves lower Tier 1 Pearson than baseline MSE (C0) for at least 3/4 architectures.
**H3**: Tier 4 experiment probe accuracy is significantly lower for DISENTANGLE (C5) than baseline (C0).
**H4**: Contrastive loss (C2) provides the largest single-component improvement on Tier 2.
**H5**: The Tier 1 vs Tier 2 gap is smaller for DISENTANGLE than baseline.

### Phase 5 Completion Checklist

```
□ All 72 training runs completed
□ All 288 evaluation runs completed (72 × 4 tiers)
□ Results aggregated into single CSV
□ H1–H5 tested with statistical significance
□ All wandb runs organized
```

---

# 9. PHASE 6: INTERPRETABILITY ANALYSIS

**Duration**: Month 8

## 9.1 Attribution Comparison
Integrated gradients for baseline vs DISENTANGLE, JASPAR motif enrichment comparison.

## 9.2 Noise Pattern Discovery
Identify sequence features baseline learns that DISENTANGLE doesn't (GC runs, restriction sites, etc.).

---

# 10. PHASE 7: PAPER WRITING

**Duration**: Months 9–12
**Target venue**: Nature Methods or Genome Biology

### Figures

| Figure | Content |
|--------|---------|
| 1 | UMAP of representations + CKA matrix |
| 2 | Probing analysis (batch probe + activity probe) |
| 3 | DISENTANGLE framework overview |
| 4 | Main results: Tier 2 Spearman across all conditions |
| 5 | Ablation: component contributions |
| 6 | Attribution comparison: baseline vs DISENTANGLE |
| 7 | CAGI5 zero-shot results |

---

# APPENDIX B: STATISTICAL ANALYSIS PROTOCOLS

## B.1 Comparing Two Conditions
Paired t-test and Wilcoxon signed-rank (paired by architecture × seed), Cohen's d effect size.

## B.2 Multiple Comparison Correction
Benjamini-Hochberg FDR correction across 120 comparisons (6 conditions × 4 tiers × 5 metrics).

## B.3 Confidence Intervals
Bootstrap with 10,000 resamples, 95% CI.

---

# APPENDIX D: TROUBLESHOOTING GUIDE

- **Too few paired sequences (< 500)**: Use coordinate-based pairing; 200bp bins with 50% overlap
- **Contrastive loss collapses**: Add projection head, increase temperature, reduce LR
- **Consensus targets disagree**: Weight by number of agreeing experiments
- **Training diverges**: Staged training (ranking → contrastive → consensus)
- **CAGI5 format issues**: Convert to (ref_seq, variant_pos, alt_allele, effect_size) format
- **GPU memory**: Reduce sequence length, batch size, use gradient accumulation + mixed precision

---

# REVISION LOG

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-02 | 1.0 | Initial research plan |

---

*End of Research Plan*
