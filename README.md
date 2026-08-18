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
python scripts/download_data.py     # clones the two upstream repos into data/raw and
                                    # prints the Zenodo commands for the data itself
```

Nothing under `data/` is in the repo (it is gitignored). Four inputs have to be fetched:

| Input | Where it goes | How to get it |
|---|---|---|
| `K562_clean.tsv`, `HepG2_clean.tsv`, `WTC11_clean.tsv` — the MPRA training tables | `data/raw/deboer_dream/human_mpra/` | `human_mpra_data.tar.gz` from Zenodo [10.5281/zenodo.10633252](https://zenodo.org/records/10633252) |
| Prix Fixe architecture modules (`prixfixe/{prixfixe,autosome,bhi}`) | `data/raw/deboer_dream/` | `git clone https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022` |
| `lentiMPRA_{K562,HepG2}_activity_and_aleatoric_data.h5` | `data/raw/dream_rnn_lentimpra/data/` | `data.tar.gz` from Zenodo [10.5281/zenodo.14145285](https://zenodo.org/records/14145285), as documented in the cloned `dream_rnn_lentimpra` README |
| CAGI5 saturation-mutagenesis `challenge_*.tsv` | `data/raw/dream_rnn_lentimpra/data/CAGI5/` | genomeinterpretation.org, by hand (registration required) |

`K562_clean.tsv` is 226,253 rows of `seq_id / seq / mean_value / fold`. The 10-fold split is
a `fold` column that ships inside the published TSV, and `prepare_fold_data()` is a plain
partition on it — so the fold dumps under `results/<run>/fold*_model*/{train,val,test}.txt`
are byte-reproducible and are not stored. Fold 0 (the held-out test fold everywhere below)
is 24,011 sequences.

The `disentangle/` results build on `disentangle/data/processed/*.h5`, produced from the
lentiMPRA HDF5 files by `python -m data.preprocess` (see `disentangle/data/preprocess.py`).
Those processed files and all trained weights are gitignored and live only on disk.

CAGI5 numbers can be re-derived without re-downloading anything: the per-element
predictions, ground truth and confidence values are stored in
`results/deboer_rankloss_1fold_v4/cagi5_predictions.pkl` and the two
`results/deboer_rankloss_1fold_v5/bet4*_cagi5_predictions.pkl` dumps.

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
python scripts/evaluate_cagi5_1fold.py --base_dir results/deboer_rankloss_1fold_v4
```

Loss implementations live in `src/losses/`, models in `src/models/`, metrics and CAGI5
evaluation in `src/evaluation/`.

## Results

On the held-out K562 fold the losses are nearly indistinguishable: the five methods that
edge past the MSE baseline (0.8233) span only 0.0012 Pearson, topping out at 0.8245 for
combined_spearman, and the 90-model reference ensemble reaches 0.8249 — so held-out Pearson
does not discriminate between losses at all.

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

Cross-cell-type: `combined_ranknet` α=0.3 gains +0.006 Spearman over MSE on WTC11
(0.6431 vs 0.6371), where there is more headroom than on K562. On HepG2 the same comparison
finished at 0.8025 vs 0.8010 — a +0.0015 gain, i.e. essentially flat, consistent with HepG2
already sitting near the architecture's ceiling.

Three follow-up bets (heteroscedastic NLL with log-variance supervision, large-Δ weighted
RankNet, small-Δ weighted RankNet) all failed to beat the top-5 ensemble. The useful
negative result: small-Δ weighting beats large-Δ weighting by +0.011, consistent with
variant effects being an inherently small-Δ task.

DISENTANGLE, on cross-experiment transfer to rank-averaged consensus targets, reaches
0.874 Spearman against 0.824 for the MSE baseline, at a cost of -8.4% within-experiment
Spearman. That trade is expected — within-experiment labels are the noisy ones.

Trained checkpoints and per-run metrics are under `results/`; `results/deboer_rankloss_1fold_v4/`
holds the 17 headline runs (9 `model_best.pth` each) and `results/deboer_official/` the paper
reproduction (90 `model_best.pth`, `cv_results.json`). Every number above is stored in a
committed results file: `*/cv_results.json` for held-out Pearson/Spearman,
`results/deboer_rankloss_1fold_v4/ensemble_results.csv` for the CAGI5 table, and
`disentangle/results/evaluation_final.csv` for the DISENTANGLE row.

Superseded v1 runs (`results/deboer_rankloss_1fold/`, `results/deboer_rankloss_1fold_v4_done/`,
`results/deboer_rankloss/combined_ranknet/`) keep their `cv_results.json` but their weights were
deleted, so those older numbers can no longer be recomputed. Optimizer states, per-epoch
checkpoints and `final_*` checkpoints were also deleted where a `best_*` sibling existed; the one
consequence is that `scripts/dump_cagi5_distributional.py` (Bet 1) can no longer be re-run against
the exact final-epoch weights it used — its output survives as
`results/noise_resistant/bet1_cagi5_predictions.pkl`.
