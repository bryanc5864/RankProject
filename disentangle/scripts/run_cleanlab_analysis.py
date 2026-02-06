#!/usr/bin/env python3
"""
Phase 5: Cleanlab Noise Detection.

Uses cross-validated residuals to identify likely mislabeled/noisy samples.
The approach:
1. 5-fold cross-validation with a simple model (dilated_cnn baseline)
2. Collect out-of-fold residuals for all samples
3. Flag samples where |residual| > 2*std AND replicate_std > median
4. Validate: flagged samples should have significantly higher replicate_std
5. Output per-sample quality scores for weighted training

Reference: Northcutt et al., "Confident Learning: Estimating Uncertainty
in Dataset Labels" (JAIR 2021)
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoders import build_encoder
from models.wrapper import DisentangleWrapper
from train import SequenceDataset


def train_fold(model, train_loader, val_loader, config, device, max_epochs=30):
    """Train a model for one fold with early stopping."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 1e-5)
    )

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        for batch in train_loader:
            seqs = batch["sequences"].to(device)
            acts = batch["activities"].to(device)

            optimizer.zero_grad()
            preds = model(seqs, return_variance=False)
            loss = F.mse_loss(preds, acts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                seqs = batch["sequences"].to(device)
                acts = batch["activities"].to(device)
                preds = model(seqs, return_variance=False)
                val_losses.append(F.mse_loss(preds, acts).item())

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


def get_predictions(model, loader, device):
    """Get predictions for a data loader."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            seqs = batch["sequences"].to(device)
            preds = model(seqs, return_variance=False)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Cleanlab noise detection")
    parser.add_argument("--data", nargs="+", required=True,
                        help="DISENTANGLE-format HDF5 files")
    parser.add_argument("--output_dir", default="data/processed",
                        help="Output directory for sample_weights.npy")
    parser.add_argument("--architecture", default="dilated_cnn",
                        choices=["cnn", "dilated_cnn", "bilstm"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residual_threshold", type=float, default=2.0,
                        help="Flag samples with |residual| > threshold * std")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model config
    model_config_map = {
        "cnn": "configs/models/cnn_basset.yaml",
        "dilated_cnn": "configs/models/dilated_cnn_basenji.yaml",
        "bilstm": "configs/models/bilstm_dream.yaml",
    }

    import yaml
    with open(model_config_map[args.architecture]) as f:
        config = yaml.safe_load(f)

    # Load full training data (train + val splits)
    print("Loading data...")
    train_dataset = SequenceDataset(args.data, split="train")
    val_dataset = SequenceDataset(args.data, split="val")

    # Combine train and val for cross-validation
    all_sequences = np.concatenate([train_dataset.sequences, val_dataset.sequences])
    all_activities = np.concatenate([train_dataset.activities, val_dataset.activities])
    all_replicate_stds = None
    if train_dataset.replicate_stds is not None and val_dataset.replicate_stds is not None:
        all_replicate_stds = np.concatenate([train_dataset.replicate_stds, val_dataset.replicate_stds])

    n_samples = len(all_sequences)
    print(f"Total samples for CV: {n_samples}")

    # Create a combined dataset for CV
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, sequences, activities):
            self.sequences = sequences
            self.activities = activities

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return {
                "sequences": torch.from_numpy(self.sequences[idx]),
                "activities": torch.tensor(self.activities[idx]),
            }

    full_dataset = SimpleDataset(all_sequences, all_activities)

    # Cross-validation
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof_predictions = np.zeros(n_samples)
    oof_mask = np.zeros(n_samples, dtype=bool)

    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_sequences)):
        print(f"\nFold {fold + 1}/{args.n_folds}")

        # Create data loaders
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False,
                                num_workers=4, pin_memory=True)

        # Build fresh model for each fold
        encoder = build_encoder(args.architecture, config)
        model = DisentangleWrapper(encoder, n_experiments=1, config=config)
        model = model.to(device)

        # Train
        model = train_fold(model, train_loader, val_loader, config, device)

        # Get out-of-fold predictions
        val_loader_ordered = DataLoader(val_subset, batch_size=128, shuffle=False,
                                        num_workers=4, pin_memory=True)
        fold_preds = get_predictions(model, val_loader_ordered, device)

        oof_predictions[val_idx] = fold_preds
        oof_mask[val_idx] = True

        # Report fold metrics
        fold_spearman = spearmanr(fold_preds, all_activities[val_idx])[0]
        print(f"  Fold {fold + 1} Spearman: {fold_spearman:.4f}")

    # Compute residuals
    residuals = all_activities - oof_predictions
    residual_std = np.std(residuals)
    abs_residuals = np.abs(residuals)

    print(f"\nResidual statistics:")
    print(f"  Mean: {np.mean(residuals):.4f}")
    print(f"  Std: {residual_std:.4f}")
    print(f"  Max |residual|: {np.max(abs_residuals):.4f}")

    # Flag noisy samples
    # Criterion 1: Large residual (model consistently wrong)
    large_residual_mask = abs_residuals > args.residual_threshold * residual_std

    # Criterion 2: High replicate_std (measurement is unreliable)
    if all_replicate_stds is not None:
        median_std = np.median(all_replicate_stds)
        high_noise_mask = all_replicate_stds > median_std
        flagged_mask = large_residual_mask & high_noise_mask
    else:
        print("Warning: No replicate_std available, using residual-only criterion")
        flagged_mask = large_residual_mask
        high_noise_mask = None

    n_flagged = flagged_mask.sum()
    print(f"\nFlagged samples: {n_flagged} ({100 * n_flagged / n_samples:.1f}%)")
    print(f"  Large residual (|r| > {args.residual_threshold}*std): {large_residual_mask.sum()}")
    if high_noise_mask is not None:
        print(f"  High replicate_std (> median): {high_noise_mask.sum()}")

    # Validate: flagged samples should have higher replicate_std
    if all_replicate_stds is not None and n_flagged > 0:
        flagged_stds = all_replicate_stds[flagged_mask]
        unflagged_stds = all_replicate_stds[~flagged_mask]

        stat, pval = mannwhitneyu(flagged_stds, unflagged_stds, alternative='greater')
        print(f"\nValidation (Mann-Whitney U test):")
        print(f"  Flagged replicate_std: {np.mean(flagged_stds):.4f} ± {np.std(flagged_stds):.4f}")
        print(f"  Unflagged replicate_std: {np.mean(unflagged_stds):.4f} ± {np.std(unflagged_stds):.4f}")
        print(f"  p-value (flagged > unflagged): {pval:.2e}")

    # Compute sample quality scores
    # Higher score = more reliable sample
    # Score = 1 / (1 + normalized_residual^2)
    normalized_residuals = abs_residuals / residual_std
    quality_scores = 1.0 / (1.0 + normalized_residuals ** 2)

    # Optionally incorporate replicate_std
    if all_replicate_stds is not None:
        # Normalize replicate_std
        normalized_stds = all_replicate_stds / np.median(all_replicate_stds)
        std_weights = 1.0 / (1.0 + normalized_stds)
        # Combine
        quality_scores = quality_scores * std_weights

    # Normalize to [0, 1]
    quality_scores = quality_scores / quality_scores.max()

    print(f"\nQuality score statistics:")
    print(f"  Mean: {np.mean(quality_scores):.4f}")
    print(f"  Min: {np.min(quality_scores):.4f}")
    print(f"  Median: {np.median(quality_scores):.4f}")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Save sample weights
    weights_path = os.path.join(args.output_dir, "sample_weights.npy")
    np.save(weights_path, quality_scores)
    print(f"\nSaved sample weights to {weights_path}")

    # Save flagged indices
    flagged_path = os.path.join(args.output_dir, "flagged_samples.npy")
    np.save(flagged_path, np.where(flagged_mask)[0])
    print(f"Saved flagged sample indices to {flagged_path}")

    # Save analysis summary
    summary = {
        "n_samples": int(n_samples),
        "n_flagged": int(n_flagged),
        "residual_std": float(residual_std),
        "residual_threshold": args.residual_threshold,
        "oof_spearman": float(spearmanr(oof_predictions, all_activities)[0]),
        "quality_score_mean": float(np.mean(quality_scores)),
        "quality_score_median": float(np.median(quality_scores)),
    }
    if all_replicate_stds is not None:
        summary["median_replicate_std"] = float(np.median(all_replicate_stds))

    summary_path = os.path.join(args.output_dir, "cleanlab_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved analysis summary to {summary_path}")


if __name__ == "__main__":
    main()
