#!/usr/bin/env python3
"""
Train DREAM-RNN using the OFFICIAL de-Boer Prix Fixe code and methodology.

Protocol from the paper:
- 10-fold CV with RC-aware fold splitting (forward/reverse in same fold)
- For each held-out test fold, train 9 models (rotating validation fold)
- Average predictions from all 9 models for the test fold
- Report average Pearson across all 10 folds

Uses the actual Prix Fixe repo code from:
  github.com/de-Boer-Lab/random-promoter-dream-challenge-2022
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# Use the ACTUAL Prix Fixe code from their repo
# The package structure is benchmarks/human/prixfixe/{prixfixe,autosome,bhi,...}
PRIXFIXE_PARENT = '/home/bcheng/RankProject/data/raw/deboer_dream/benchmarks/human'
sys.path.insert(0, PRIXFIXE_PARENT)

from prixfixe.autosome import AutosomeFinalLayersBlock, AutosomeDataProcessor, AutosomeTrainer
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet


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
        in_channels=5,
        out_channels=320,
        seqsize=230,
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout=0.2
    )

    core = BHICoreBlock(
        in_channels=first.out_channels,
        out_channels=320,
        seqsize=first.infer_outseqsize(),
        lstm_hidden_channels=320,
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout1=0.2,
        dropout2=0.5
    )

    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

    model = PrixFixeNet(
        first=first,
        core=core,
        final=final,
        generator=generator
    )

    return model


def predict_with_model(model, test_path, device, seqsize=230):
    """Generate predictions for test data using a trained model."""
    model.eval()

    test_df = pd.read_csv(test_path, sep='\t')
    test_df['rev'] = test_df['seq_id'].str.contains('_Reversed:').astype(int)

    mapping = {
        'A': [1, 0, 0, 0],
        'G': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }

    preds = []
    targets = []

    # Process in batches for efficiency
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
                batch_x.append(encoded.T)  # (5, seqlen)

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
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results/deboer_official')
    parser.add_argument('--n_test_folds', type=int, default=10,
                        help='Number of test folds to evaluate (1-10)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data, sep='\t')
    print(f"Total samples: {len(df)}")
    print(f"Folds: {sorted(df['fold'].unique())}")

    # Print model info once
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

        # Get available validation folds (all except test)
        val_folds = [f for f in range(10) if f != test_fold]

        # Train 9 models with different validation folds
        all_test_preds = []
        test_targets = None

        for model_idx, val_fold in enumerate(val_folds):
            print(f"\n  --- Model {model_idx+1}/9: val_fold={val_fold}, "
                  f"train_folds={[f for f in range(10) if f != test_fold and f != val_fold]} ---")

            # Prepare fold data
            fold_dir = os.path.join(args.output_dir, f'fold{test_fold}_model{model_idx}')
            train_path, val_path, test_path, n_train, n_val, n_test = \
                prepare_fold_data(df, test_fold, val_fold, fold_dir)
            print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

            # Build model
            generator = torch.Generator()
            generator.manual_seed(42 + test_fold * 100 + model_idx)

            model = build_model(generator).to(device)

            # Set up data processor using their code
            BATCH_SIZE = 32
            batch_per_epoch = n_train // BATCH_SIZE

            dataprocessor = AutosomeDataProcessor(
                path_to_training_data=train_path,
                path_to_validation_data=val_path,
                train_batch_size=BATCH_SIZE,
                batch_per_epoch=batch_per_epoch,
                train_workers=4,
                valid_batch_size=BATCH_SIZE,
                valid_workers=4,
                shuffle_train=True,
                shuffle_val=False,
                seqsize=230,
                generator=generator
            )

            # Set up trainer using their code
            # Their trainer requires the dir doesn't already exist
            import shutil
            model_dir = os.path.join(fold_dir, 'weights')
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            trainer = AutosomeTrainer(
                model,
                device=device,
                model_dir=model_dir,
                dataprocessor=dataprocessor,
                num_epochs=args.epochs,
                lr=args.lr
            )

            # Train
            trainer.fit()

            # Load best model
            best_path = os.path.join(model_dir, 'model_best.pth')
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device))
                print(f"  Loaded best model from {best_path}")
            else:
                print(f"  WARNING: No best model saved, using final model")

            # Predict on test fold
            test_preds, targets = predict_with_model(model, test_path, device)
            all_test_preds.append(test_preds)
            test_targets = targets

            # Per-model metrics
            sp = spearmanr(test_preds, targets)[0]
            pe = pearsonr(test_preds, targets)[0]
            print(f"  Model {model_idx+1} test: Spearman={sp:.4f}, Pearson={pe:.4f}")

            # Clean up GPU memory
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
    print("FINAL RESULTS: 10-FOLD CV WITH 9-MODEL ENSEMBLE")
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
    print(f"\n  de-Boer reported K562 Pearson: 0.88")

    # Save results
    summary = {
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
