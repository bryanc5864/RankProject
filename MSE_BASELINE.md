# K562 lentiMPRA MSE Baseline

## Task

Predict lentiMPRA expression (log2 RNA/DNA) from 230bp DNA sequence in K562 cells.
Architecture: DREAM-RNN (Prix Fixe BHI BiLSTM). Protocol: 10-fold CV, 9 models per fold (90 total), ensemble predictions.

## Dataset

| File | Description |
|---|---|
| `data/raw/deboer_dream/human_mpra/K562_clean.tsv` | 226,253 sequences, columns: seq_id, seq (230bp), mean_value (log2 activity), fold (0–9) |

Forward and RC sequence pairs are grouped into the same fold. Fold 0 is the 1-fold test set (~24K sequences).

## Architecture (Prix Fixe DREAM-RNN)

| Component | File | Notes |
|---|---|---|
| First layers | `data/raw/deboer_dream/benchmarks/human/prixfixe/bhi/first_layers_block.py` | 2× Conv1D (k=9,15), 256ch each, concat → 512ch |
| Core layers | `data/raw/deboer_dream/benchmarks/human/prixfixe/bhi/coreblock.py` | BiLSTM (320 hidden each dir), then 2× Conv1D |
| Final layers | `data/raw/deboer_dream/benchmarks/human/prixfixe/autosome/final_layers_block.py` | Conv1D(256) → AdaptiveAvgPool → Linear(1) |
| Data processor | `data/raw/deboer_dream/benchmarks/human/prixfixe/autosome/dataprocessor.py` | Handles fold splits, `batch_per_epoch`, RC channel |
| Dataset | `data/raw/deboer_dream/benchmarks/human/prixfixe/autosome/dataset.py` | 5-channel input: 4 ACGT + 1 is_reverse |

Input: (batch, 5, 230). Parameters: ~4.2M.

## Training Scripts

| File | Description |
|---|---|
| `scripts/train_deboer_official.py` | Full 10-fold CV (90 models), MSE only, reference implementation |
| `scripts/train_deboer_rankloss.py` | 1-fold or full CV with custom losses (MSE, rank losses, combined) |
| `scripts/run_1fold_batch.sh` | Batch launcher; reads `RESULTS_DIR` env var for output location |

### Key Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| lr (max) | 0.005 | Paper: "max LR of 0.005 for most blocks"; non-attention only |
| batch_size | 32 | Paper: reduced from 1024 to match MPRAnn for human MPRA |
| batch_per_epoch | `n_train // 32` ≈ 5,600 | One full pass per epoch |
| epochs | 80 | Paper default |
| optimizer | AdamW, weight_decay=0.01 | Autosome trainer |
| scheduler | OneCycleLR, pct_start=0.3 | Autosome trainer |
| loss | MSE | Paper: KL-div replaced with MSE for human MPRA |
| seqsize | 230 | lentiMPRA insert length |
| in_channels | 5 | 4 ACGT + 1 is_reverse (is_singleton dropped for MPRA) |

## Results (Pearson r, fold 0 test set unless noted)

| Run | Models | lr | Fold | Pearson | Spearman | Notes |
|---|---|---|---|---|---|---|
| v1 | 9 | 0.005 | 0 | 0.786 | — | Wrong batch_per_epoch (=1000), Autosome default |
| v3 | 9 | 0.001 | 0 | **0.814** | 0.755 | Wrong lr (too low); batch_per_epoch = n_train//32 |
| official | 9 | 0.005 | 0 | 0.821 | 0.761 | `train_deboer_official.py`, fold 0 only |
| official | 90 | 0.005 | all | **0.825** | 0.770 | Full 10-fold CV mean; per-fold range 0.815–0.834 |
| v4 | 9 | 0.005 | 0 | *running* | — | Correct lr, batch_per_epoch = n_train//32 |

### Official 10-fold breakdown (90-model ensemble)

| Fold | Pearson | Spearman |
|---|---|---|
| 0 | 0.8213 | 0.7613 |
| 1 | 0.8235 | 0.7606 |
| 2 | 0.8154 | 0.7675 |
| 3 | 0.8263 | 0.7725 |
| 4 | 0.8344 | 0.7766 |
| 5 | 0.8221 | 0.7721 |
| 6 | 0.8223 | 0.7700 |
| 7 | 0.8277 | 0.7768 |
| 8 | 0.8301 | 0.7732 |
| 9 | 0.8258 | 0.7672 |
| **mean** | **0.8249** | **0.7698** |

## Paper Reference

Rafi et al. (2025) *Nature Biotechnology* 43:1373–1383. doi:10.1038/s41587-024-02414-w

Human MPRA benchmark protocol (Methods):
- 10-fold CV, forward+RC in same fold
- 9 models per fold (one val fold, eight train folds)
- batch_size=32 (reduced from 1024 to match MPRAnn)
- MSE loss (KL-div not applicable to MPRA)
- All other Autosome trainer settings unchanged (lr=0.005, AdamW, OneCycleLR, 80 epochs)

**Expected ceiling for this setup**: ~0.82–0.83 per fold (9 models), ~0.825 for 90-model ensemble.
The published DREAM-RNN K562 result (Fig. 3b) is consistent with our 90-model 0.8249.
