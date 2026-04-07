# DISENTANGLE: Learning Biology, Not Noise

A framework for noise-resistant sequence-to-function modeling. Developed alongside
the RankProject for rank-order learning on MPRA data.

## Overview

Current sequence-to-function models learn experimental noise that is reproducible
within an experiment but not across experiments. DISENTANGLE provides:

1. **Empirical characterization** of what noise models learn
2. **Three complementary noise-resistance strategies**: cross-experiment consensus
   targets, noise-contrastive representation learning, and adaptive margin ranking
3. **A four-tier evaluation framework** distinguishing genuine biological learning
   from noise exploitation
4. **Architecture-agnostic training protocol** wrapping any existing sequence model

## Structure

```
disentangle/
├── configs/          # YAML configurations for data, models, training, experiments
├── data/             # Data download, preprocessing, pairing, and splitting
├── models/           # Encoder architectures and DISENTANGLE wrapper
├── training/         # Loss functions and trainer
├── evaluation/       # Four-tier evaluation framework
├── analysis/         # Noise characterization, probing, attribution, motif analysis
├── scripts/          # Shell scripts for running experiments
├── tests/            # Unit tests
├── notebooks/        # Analysis notebooks
└── paper/            # Manuscript files
```

## Relationship to RankProject

This module extends the RankProject by focusing on cross-experiment generalization.
The existing RankProject models and data pipelines can be reused as baselines.

## Quick Start

```bash
# From RankProject root
cd disentangle

# Run noise characterization (Phase 2)
bash scripts/run_noise_characterization.sh

# Run full ablation study (Phase 5)
bash scripts/run_ablation.sh

# Run evaluation
bash scripts/run_evaluation.sh
```
