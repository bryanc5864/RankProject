#!/usr/bin/env python3
"""
Train DREAM-RNN using official Prix Fixe architecture but with rank-based losses.

Same protocol as train_deboer_official.py:
- 10-fold CV with RC-aware fold splitting
- 9 models per fold (rotating validation)
- Average predictions from all 9 models
- Report average Pearson/Spearman across 10 folds

But replaces MSE with combined MSE + ranking losses.
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# Prix Fixe code
PRIXFIXE_PARENT = '/home/bcheng/RankProject/data/raw/deboer_dream/benchmarks/human'
sys.path.insert(0, PRIXFIXE_PARENT)

from prixfixe.autosome import AutosomeFinalLayersBlock, AutosomeDataProcessor, AutosomeTrainer
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

# Our rank-based losses
sys.path.insert(0, '/home/bcheng/RankProject')
from src.losses.plackett_luce import plackett_luce_loss, weighted_plackett_luce_loss
from src.losses.ranknet import ranknet_loss, margin_ranknet_loss, lambda_ranknet_loss, sampled_ranknet_loss
from src.losses.softsort import softsort_loss, softsort_spearman_loss, differentiable_rank_mse
from src.losses.combined import combined_loss


class RankLossTrainer:
    """
    Custom trainer that uses the Prix Fixe model but with custom loss functions.
    Mirrors AutosomeTrainer behavior (AdamW + OneCycleLR) but overrides loss.
    """

    def __init__(self, model, device, model_dir, dataprocessor, num_epochs,
                 lr=0.001, loss_type='combined_pl', loss_alpha=0.5, **loss_kwargs):
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir
        self.dataprocessor = dataprocessor
        self.num_epochs = num_epochs
        self.loss_type = loss_type
        self.loss_alpha = loss_alpha
        self.loss_kwargs = loss_kwargs
        self.best_pearson = -np.inf

        os.makedirs(model_dir, exist_ok=True)

        # Same optimizer/scheduler as AutosomeTrainer
        weight_decay = 0.01
        max_lr = lr
        div_factor = 25.0
        min_lr = max_lr / div_factor

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=min_lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            div_factor=div_factor,
            steps_per_epoch=dataprocessor.train_epoch_size(),
            epochs=num_epochs,
            pct_start=0.3,
            three_phase=False
        )

        self._current_epoch = 0
        self.train_dataloader = dataprocessor.prepare_train_dataloader()
        self.valid_dataloader = dataprocessor.prepare_valid_dataloader()

    def compute_loss(self, predictions, targets):
        """Compute the configured loss."""
        pred = predictions.view(-1)
        tgt = targets.view(-1)

        if self.loss_type == 'combined_pl':
            return combined_loss(pred, tgt, alpha=self.loss_alpha,
                                 ranking_loss_fn='plackett_luce')
        elif self.loss_type == 'combined_ranknet':
            return combined_loss(pred, tgt, alpha=self.loss_alpha,
                                 ranking_loss_fn='ranknet')
        elif self.loss_type == 'combined_softsort':
            return combined_loss(pred, tgt, alpha=self.loss_alpha,
                                 ranking_loss_fn='softsort')
        elif self.loss_type == 'combined_margin_ranknet':
            mse = F.mse_loss(pred, tgt)
            rank = margin_ranknet_loss(pred, tgt)
            return self.loss_alpha * mse + (1 - self.loss_alpha) * rank
        elif self.loss_type == 'combined_lambda_ranknet':
            mse = F.mse_loss(pred, tgt)
            rank = lambda_ranknet_loss(pred, tgt)
            return self.loss_alpha * mse + (1 - self.loss_alpha) * rank
        elif self.loss_type == 'adaptive_softsort':
            # Adaptive: start MSE-heavy (alpha_start), end rank-heavy (alpha_end)
            # alpha_start and alpha_end passed via loss_kwargs
            alpha_start = self.loss_kwargs.get('alpha_start', 0.9)
            alpha_end = self.loss_kwargs.get('alpha_end', 0.3)
            warmup = self.loss_kwargs.get('warmup_epochs', 10)
            total = self.loss_kwargs.get('total_epochs', 80)
            epoch = self._current_epoch
            if epoch < warmup:
                alpha = alpha_start
            else:
                progress = min(1.0, (epoch - warmup) / (total - warmup))
                alpha = alpha_start + (alpha_end - alpha_start) * progress
            mse = F.mse_loss(pred, tgt)
            rank = softsort_loss(pred, tgt)
            return alpha * mse + (1 - alpha) * rank
        elif self.loss_type == 'combined_sampled_ranknet':
            mse = F.mse_loss(pred, tgt)
            rank = sampled_ranknet_loss(pred, tgt)
            return self.loss_alpha * mse + (1 - self.loss_alpha) * rank
        elif self.loss_type == 'combined_weighted_pl':
            mse = F.mse_loss(pred, tgt)
            rank = weighted_plackett_luce_loss(pred, tgt)
            return self.loss_alpha * mse + (1 - self.loss_alpha) * rank
        elif self.loss_type == 'combined_spearman':
            mse = F.mse_loss(pred, tgt)
            rank = differentiable_rank_mse(pred, tgt)
            return self.loss_alpha * mse + (1 - self.loss_alpha) * rank
        elif self.loss_type == 'plackett_luce':
            return plackett_luce_loss(pred, tgt)
        elif self.loss_type == 'ranknet':
            return ranknet_loss(pred, tgt)
        elif self.loss_type == 'mse':
            return F.mse_loss(pred, tgt)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def train_step(self, batch):
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device).float().squeeze(-1)

        pred = self.model(x).squeeze(-1)
        loss = self.compute_loss(pred, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return loss.item()

    def fit(self):
        for epoch in tqdm(range(1, self.num_epochs + 1)):
            self._current_epoch = epoch
            self.model.train()
            losses = []
            for batch in tqdm(self.train_dataloader,
                              total=self.dataprocessor.train_epoch_size(),
                              desc="Train epoch", leave=False):
                loss = self.train_step(batch)
                losses.append(loss)

            # Validate
            if self.valid_dataloader is not None:
                metrics = self.validate()
                if metrics['pearsonr'] > self.best_pearson:
                    self.best_pearson = metrics['pearsonr']
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_dir, 'model_best.pth'))

    def validate(self):
        self.model.eval()
        y_preds, y_trues = [], []
        with torch.inference_mode():
            for batch in self.valid_dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"].float().squeeze(-1)
                pred = self.model(x).squeeze(-1).cpu().numpy()
                y_preds.append(pred)
                y_trues.append(y.numpy())

        y_pred = np.concatenate(y_preds)
        y_true = np.concatenate(y_trues)
        metrics = {
            'MSE': float(np.mean((y_pred - y_true) ** 2)),
            'pearsonr': float(pearsonr(y_true, y_pred)[0]),
            'spearmanr': float(spearmanr(y_true, y_pred)[0])
        }
        print(metrics)
        return metrics


def prepare_fold_data(df, test_fold, val_fold, output_dir):
    """Write train/val/test TSV files for a given fold configuration."""
    os.makedirs(output_dir, exist_ok=True)

    train_folds = [f for f in range(10) if f != test_fold and f != val_fold]
    train_df = df[df['fold'].isin(train_folds)]
    val_df = df[df['fold'] == val_fold]
    test_df = df[df['fold'] == test_fold]

    train_path = os.path.join(output_dir, 'train.txt')
    val_path = os.path.join(output_dir, 'val.txt')
    test_path = os.path.join(output_dir, 'test.txt')

    train_df.to_csv(train_path, sep='\t', index=False)
    val_df.to_csv(val_path, sep='\t', index=False)
    test_df.to_csv(test_path, sep='\t', index=False)

    return train_path, val_path, test_path, len(train_df), len(val_df), len(test_df)


def build_model(generator):
    """Build DREAM-RNN model using official Prix Fixe components."""
    first = BHIFirstLayersBlock(
        in_channels=5, out_channels=320, seqsize=230,
        kernel_sizes=[9, 15], pool_size=1, dropout=0.2
    )
    core = BHICoreBlock(
        in_channels=first.out_channels, out_channels=320,
        seqsize=first.infer_outseqsize(), lstm_hidden_channels=320,
        kernel_sizes=[9, 15], pool_size=1, dropout1=0.2, dropout2=0.5
    )
    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)
    model = PrixFixeNet(first=first, core=core, final=final, generator=generator)
    return model


def predict_with_model(model, test_path, device, seqsize=230):
    """Generate predictions for test data using a trained model."""
    model.eval()
    test_df = pd.read_csv(test_path, sep='\t')
    test_df['rev'] = test_df['seq_id'].str.contains('_Reversed:').astype(int)

    mapping = {
        'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0],
        'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]
    }

    preds, targets = [], []
    batch_size = 256
    seqs = test_df['seq'].values
    revs = test_df['rev'].values
    ys = test_df['mean_value'].values

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            batch_revs = revs[i:i+batch_size]
            batch_ys = ys[i:i+batch_size]

            batch_x = []
            for seq, rev in zip(batch_seqs, batch_revs):
                encoded = np.array([mapping.get(b, [0,0,0,0]) for b in seq], dtype=np.float32)
                rev_channel = np.full((len(seq), 1), rev, dtype=np.float32)
                encoded = np.concatenate([encoded, rev_channel], axis=1)
                batch_x.append(encoded.T)

            batch_x = torch.tensor(np.array(batch_x), device=device, dtype=torch.float32)
            pred = model(batch_x)
            preds.extend(pred.cpu().numpy().flatten().tolist())
            targets.extend(batch_ys.tolist())

    return np.array(preds), np.array(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='/home/bcheng/RankProject/data/raw/deboer_dream/human_mpra/K562_clean.tsv')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_test_folds', type=int, default=10)
    parser.add_argument('--loss_type', type=str, default='combined_pl',
                        choices=['combined_pl', 'combined_ranknet', 'combined_softsort',
                                 'combined_margin_ranknet', 'combined_lambda_ranknet',
                                 'combined_sampled_ranknet', 'combined_weighted_pl',
                                 'combined_spearman', 'adaptive_softsort',
                                 'plackett_luce', 'ranknet', 'mse'])
    parser.add_argument('--loss_alpha', type=float, default=0.5,
                        help='Weight for MSE in combined loss (1-alpha for ranking)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loss type: {args.loss_type}, alpha: {args.loss_alpha}")

    # Load data
    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data, sep='\t')
    print(f"Total samples: {len(df)}")

    # Print model info
    generator = torch.Generator()
    generator.manual_seed(42)
    test_model = build_model(generator).to(device)
    n_params = sum(p.numel() for p in test_model.parameters())
    print(f"DREAM-RNN parameters: {n_params:,}")
    del test_model

    fold_results = []

    for test_fold in range(args.n_test_folds):
        print(f"\n{'='*70}")
        print(f"TEST FOLD {test_fold} (held-out)")
        print(f"{'='*70}")

        val_folds = [f for f in range(10) if f != test_fold]
        all_test_preds = []
        test_targets = None

        for model_idx, val_fold in enumerate(val_folds):
            print(f"\n  --- Model {model_idx+1}/9: val_fold={val_fold}, "
                  f"train_folds={[f for f in range(10) if f != test_fold and f != val_fold]} ---")

            fold_dir = os.path.join(args.output_dir, f'fold{test_fold}_model{model_idx}')
            train_path, val_path, test_path, n_train, n_val, n_test = \
                prepare_fold_data(df, test_fold, val_fold, fold_dir)
            print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

            generator = torch.Generator()
            generator.manual_seed(42 + test_fold * 100 + model_idx)

            model = build_model(generator).to(device)

            BATCH_SIZE = 1024
            BATCH_PER_EPOCH = 1000  # Official Prix Fixe default (oversamples ~6x per epoch)

            dataprocessor = AutosomeDataProcessor(
                path_to_training_data=train_path,
                path_to_validation_data=val_path,
                train_batch_size=BATCH_SIZE,
                batch_per_epoch=BATCH_PER_EPOCH,
                train_workers=4,
                valid_batch_size=4096,
                valid_workers=4,
                shuffle_train=True,
                shuffle_val=False,
                seqsize=230,
                generator=generator
            )

            model_dir = os.path.join(fold_dir, 'weights')
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            trainer = RankLossTrainer(
                model, device=device, model_dir=model_dir,
                dataprocessor=dataprocessor, num_epochs=args.epochs,
                lr=args.lr, loss_type=args.loss_type, loss_alpha=args.loss_alpha
            )

            trainer.fit()

            # Load best model
            best_path = os.path.join(model_dir, 'model_best.pth')
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device))
                print(f"  Loaded best model from {best_path}")
            else:
                print(f"  WARNING: No best model saved, using final model")

            test_preds, targets = predict_with_model(model, test_path, device)
            all_test_preds.append(test_preds)
            test_targets = targets

            sp = spearmanr(test_preds, targets)[0]
            pe = pearsonr(test_preds, targets)[0]
            print(f"  Model {model_idx+1} test: Spearman={sp:.4f}, Pearson={pe:.4f}")

            del model, trainer, dataprocessor
            torch.cuda.empty_cache()

        # Average predictions from all 9 models
        ensemble_preds = np.mean(all_test_preds, axis=0)
        sp = spearmanr(ensemble_preds, test_targets)[0]
        pe = pearsonr(ensemble_preds, test_targets)[0]

        print(f"\n  ENSEMBLE (9 models averaged) test fold {test_fold}:")
        print(f"    Spearman: {sp:.4f}")
        print(f"    Pearson:  {pe:.4f}")

        fold_results.append({
            'test_fold': test_fold,
            'ensemble_spearman': float(sp),
            'ensemble_pearson': float(pe),
            'individual_pearsons': [float(pearsonr(p, test_targets)[0]) for p in all_test_preds],
        })

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: {args.loss_type} (alpha={args.loss_alpha})")
    print(f"{'='*70}")

    for r in fold_results:
        print(f"  Fold {r['test_fold']}: Pearson={r['ensemble_pearson']:.4f}, "
              f"Spearman={r['ensemble_spearman']:.4f}")

    mean_pearson = np.mean([r['ensemble_pearson'] for r in fold_results])
    std_pearson = np.std([r['ensemble_pearson'] for r in fold_results])
    mean_spearman = np.mean([r['ensemble_spearman'] for r in fold_results])
    std_spearman = np.std([r['ensemble_spearman'] for r in fold_results])

    print(f"\n  Mean Pearson:  {mean_pearson:.4f} +/- {std_pearson:.4f}")
    print(f"  Mean Spearman: {mean_spearman:.4f} +/- {std_spearman:.4f}")

    summary = {
        'loss_type': args.loss_type,
        'loss_alpha': args.loss_alpha,
        'mean_pearson': float(mean_pearson),
        'std_pearson': float(std_pearson),
        'mean_spearman': float(mean_spearman),
        'std_spearman': float(std_spearman),
        'fold_results': fold_results
    }

    with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/cv_results.json")


if __name__ == '__main__':
    main()
