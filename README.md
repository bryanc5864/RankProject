# RankProject

Do rank-aware losses beat plain MSE for sequence-to-activity prediction on MPRA data,
and does any gain carry over to variant-effect prediction?

The setup is the published Prix Fixe DREAM-RNN architecture (Rafi et al., *Nat Biotechnol*
43:1373-1383, 2025) — a ~4.2M-parameter BiLSTM trained on 226,253 K562 lentiMPRA sequences
(230 bp, log2 RNA/DNA activity). Seventeen loss variants are trained under one identical
protocol (fold-0 held out, 9-model ensemble, 80 epochs each) and scored on two benchmarks:
the held-out K562 test fold, and CAGI5 saturation-mutagenesis variant effects.

The `disentangle/` subproject extends the same idea to cross-experiment noise resistance,
using experiment-conditional BatchNorm, contrastive terms, and consensus targets across
K562 + HepG2.

## Install

```bash
conda create -n rank python=3.10 && conda activate rank
pip install -r requirements.txt
python scripts/download_data.py     # clones the DREAM-RNN + Prix Fixe repos into data/raw
```

Raw data is not in the repo. `download_data.py` fetches the lentiMPRA activity tables and
the Prix Fixe architecture modules; CAGI5 saturation-mutagenesis files have to be pulled
from genomeinterpretation.org by hand.

## Running

One loss variant, 9 models, fold 0 held out:

```bash
python scripts/train_deboer_rankloss.py \
    --loss_type combined_ranknet --loss_alpha 0.3 \
    --gpu 0 --output_dir results/my_run --epochs 80 --n_test_folds 1
```

`scripts/run_1fold_batch.sh <gpu> <name:loss:alpha> ...` queues several variants on one GPU
and honours a `RESULTS_DIR` env var. `scripts/train_deboer_official.py` is the MSE-only
reference implementation that reproduces the paper's 90-model 10-fold ensemble.
CAGI5 scoring for a finished run directory:

```bash
python scripts/evaluate_cagi5_1fold.py --results_dir results/deboer_rankloss_1fold_v4
```

Loss implementations live in `src/losses/`, models in `src/models/`, metrics and CAGI5
evaluation in `src/evaluation/`.

## Results

On the held-out K562 fold the losses are nearly indistinguishable: the top five sit within
0.0006 Pearson of the MSE baseline (0.8233), and the 90-model reference ensemble reaches
0.8249 — so held-out Pearson does not discriminate between losses at all.

CAGI5 separates them. On the four K562-matched elements (GP1BB, HBB, HBG1, PKLR):

| Method | K562 Spearman | All-element Spearman |
|---|---|---|
| uniform top-5 ensemble | 0.4552 | 0.3156 |
| combined_ranknet (α=0.3) | 0.4494 | 0.3088 |
| combined_softsort (α=0.3) | 0.4449 | 0.3110 |
| MSE baseline | 0.4408 | 0.2993 |
| combined_spearman | 0.4234 | 0.2976 |

Four rank losses beat MSE, the α sweep is monotonic (more rank weight is better), and
averaging the top five methods beats any single one. Restricting to high-confidence
variants lifts the top-5 ensemble to 0.7261, which says most of the aggregate metric is
noise floor on small-effect variants. Directly optimising a soft Spearman surrogate wins
on held-out Pearson and loses on CAGI5 — the two benchmarks measure different things.

Cross-cell-type: `combined_ranknet` α=0.3 gains +0.006 Spearman over MSE on WTC11, where
there is more headroom than on K562.

Three follow-up bets (heteroscedastic NLL with log-variance supervision, large-Δ weighted
RankNet, small-Δ weighted RankNet) all failed to beat the top-5 ensemble. The useful
negative result: small-Δ weighting beats large-Δ weighting by +0.011, consistent with
variant effects being an inherently small-Δ task.

DISENTANGLE, on cross-experiment transfer to rank-averaged consensus targets, reaches
0.874 Spearman against 0.824 for the MSE baseline, at a cost of -8.4% within-experiment
Spearman. That trade is expected — within-experiment labels are the noisy ones.

Trained checkpoints and per-run metrics are under `results/`; `results/deboer_rankloss_1fold_v4/`
holds the 17 headline runs and `results/deboer_official/` the paper reproduction.
