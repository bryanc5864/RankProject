#!/usr/bin/env python3
"""
Training script for DISENTANGLE.

Supports all 6 training conditions:
  C0: baseline_mse       - Standard MSE on single/multi experiment data
  C1: ranking_only       - Adaptive margin ranking loss
  C2: contrastive_only   - MSE + contrastive loss (requires paired data)
  C3: consensus_only     - Consensus ranking loss (requires paired data)
  C4: ranking_contrastive - Ranking + contrastive (requires paired data)
  C5: full_disentangle   - Ranking + contrastive + consensus (requires paired data)

Usage:
    # C0: Baseline MSE on K562
    python train.py --architecture cnn --condition baseline_mse \
        --data data/processed/dream_K562.h5 --output_dir results/cnn_C0_seed42

    # C2: Contrastive only (needs paired data)
    python train.py --architecture cnn --condition contrastive_only \
        --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
        --paired_data data/processed/paired_K562_HepG2.h5 \
        --output_dir results/cnn_C2_seed42

    # C5: Full DISENTANGLE
    python train.py --architecture cnn --condition full_disentangle \
        --data data/processed/dream_K562.h5 data/processed/dream_HepG2.h5 \
        --paired_data data/processed/paired_K562_HepG2.h5 \
        --output_dir results/cnn_C5_seed42
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

import yaml

from models.encoders import build_encoder
from models.wrapper import DisentangleWrapper
from training.losses.ranking import AdaptiveMarginRankingLoss
from training.losses.consensus import ConsensusLoss
from training.losses.variant_contrastive import VariantContrastiveLoss
from training.losses.hierarchical_contrastive import HierarchicalContrastiveLoss
from training.losses.heteroscedastic import HeteroscedasticLoss, HeteroscedasticMSELoss
from training.augmentations import get_augmenter
from training.curriculum import create_noise_curriculum

# Conditions that require paired data
PAIRED_CONDITIONS = {"contrastive_only", "consensus_only", "ranking_contrastive", "full_disentangle"}


class SequenceDataset(Dataset):
    """Dataset loading from DISENTANGLE-format HDF5 files."""

    def __init__(self, data_files: list, split: str = "train",
                 quantile_normalize: bool = False,
                 noise_augmentation: str = None, noise_level: float = 0.1):
        split_code = {"train": 0, "val": 1, "test": 2}[split]

        all_seqs = []
        all_acts = []
        all_stds = []
        all_exp_ids = []

        for fpath in data_files:
            with h5py.File(fpath, "r") as f:
                splits = f["split"][:]
                mask = splits == split_code
                seqs = f["sequences"][:][mask]
                acts = f["activities"][:][mask]
                exp_ids = f["experiment_id"][:][mask]
                stds = f["replicate_std"][:][mask] if "replicate_std" in f else None

            all_seqs.append(seqs)
            all_acts.append(acts)
            all_exp_ids.append(exp_ids)
            if stds is not None:
                all_stds.append(stds)

            print(f"  {split}: loaded {mask.sum()} from {os.path.basename(fpath)} "
                  f"(exp_id={exp_ids[0] if len(exp_ids) > 0 else '?'})")

        self.sequences = np.concatenate(all_seqs, axis=0).astype(np.float32)
        self.activities = np.concatenate(all_acts, axis=0).astype(np.float32)
        self.experiment_ids = np.concatenate(all_exp_ids, axis=0)
        self.replicate_stds = (np.concatenate(all_stds, axis=0).astype(np.float32)
                               if all_stds else None)

        # C2: Quantile normalization of activity targets
        if quantile_normalize:
            from scipy.stats import rankdata
            ranks = rankdata(self.activities) / len(self.activities)
            from scipy.stats import norm as sp_norm
            self.activities = sp_norm.ppf(np.clip(ranks, 0.001, 0.999)).astype(np.float32)
            print(f"  Applied quantile normalization to activities")

        # E3: Noise augmentation (only for training)
        self.noise_augmentation = noise_augmentation if split == "train" else None
        self.noise_level = noise_level

        print(f"  {split} total: {len(self.sequences)} sequences, "
              f"activity range [{self.activities.min():.3f}, {self.activities.max():.3f}]")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "sequences": torch.from_numpy(self.sequences[idx]),
            "activities": torch.tensor(self.activities[idx]),
            "experiment_ids": torch.tensor(self.experiment_ids[idx], dtype=torch.long),
        }
        if self.replicate_stds is not None:
            item["replicate_std"] = torch.tensor(self.replicate_stds[idx])
        return item


class PairedSequenceDataset(Dataset):
    """Dataset for paired sequences measured in both K562 and HepG2."""

    def __init__(self, paired_file: str, split: str = "train"):
        split_code = {"train": 0, "val": 1, "test": 2}[split]

        with h5py.File(paired_file, "r") as f:
            splits = f["split"][:]
            mask = splits == split_code
            self.sequences = f["sequences"][:][mask].astype(np.float32)
            self.k562_activities = f["k562_activities"][:][mask].astype(np.float32)
            self.hepg2_activities = f["hepg2_activities"][:][mask].astype(np.float32)
            self.consensus_ranks = f["consensus_ranks"][:][mask].astype(np.float32)
            self.k562_stds = f["k562_stds"][:][mask].astype(np.float32)
            self.hepg2_stds = f["hepg2_stds"][:][mask].astype(np.float32)

        print(f"  {split} paired: {len(self.sequences)} sequences, "
              f"consensus_ranks [{self.consensus_ranks.min():.4f}, {self.consensus_ranks.max():.4f}]")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequences": torch.from_numpy(self.sequences[idx]),
            "k562_activities": torch.tensor(self.k562_activities[idx]),
            "hepg2_activities": torch.tensor(self.hepg2_activities[idx]),
            "consensus_ranks": torch.tensor(self.consensus_ranks[idx]),
            "k562_stds": torch.tensor(self.k562_stds[idx]),
            "hepg2_stds": torch.tensor(self.hepg2_stds[idx]),
        }


def compute_contrastive_loss(model, sequences, temperature=0.07):
    """
    In-batch contrastive loss using paired sequences.

    Same sequence projected through different experiment normalizations
    should be similar (positive pair), while different sequences should
    be dissimilar (in-batch negatives).
    """
    z_a = model.project(sequences, experiment_id=0)  # K562 norm
    z_b = model.project(sequences, experiment_id=1)  # HepG2 norm

    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    # Cosine similarity matrix: [B, B]
    sim = torch.mm(z_a, z_b.t()) / temperature

    # Positive pairs are on the diagonal
    labels = torch.arange(len(sequences), device=sequences.device)

    # Symmetric InfoNCE
    loss_ab = F.cross_entropy(sim, labels)
    loss_ba = F.cross_entropy(sim.t(), labels)

    return (loss_ab + loss_ba) / 2


def compute_sensitivity_loss(model, sequences, n_mutations=5):
    """
    Sensitivity loss for two-stage training (B1).

    Encourages model to be sensitive to single-nucleotide mutations
    by penalizing small prediction differences between ref and mutant.

    Uses a margin-based formulation: loss = max(0, margin - |f(ref) - f(mut)|)
    so it only pushes diffs above a threshold, preventing divergence.
    """
    B, L, C = sequences.shape
    margin = 0.01  # target minimum prediction difference per mutation

    # Generate random single-nt mutations
    mutants = sequences.repeat_interleave(n_mutations, dim=0)  # [B*n, L, 4]
    positions = torch.randint(0, L, (B * n_mutations,), device=sequences.device)
    nucleotides = torch.randint(0, C, (B * n_mutations,), device=sequences.device)

    idx = torch.arange(B * n_mutations, device=sequences.device)
    mutants[idx, positions, :] = 0.0
    mutants[idx, positions, nucleotides] = 1.0

    # Get predictions
    ref_preds = model.predict_denoised(sequences)  # [B]
    mut_preds = model.predict_denoised(mutants)  # [B*n]
    mut_preds = mut_preds.view(B, n_mutations)  # [B, n]

    # Margin-based: only penalize when diff is below the margin
    diffs = torch.abs(mut_preds - ref_preds.unsqueeze(1))  # [B, n]
    loss = torch.clamp(margin - diffs, min=0.0)  # [B, n]
    return loss.mean()


def validate(model, val_loader, device, use_consensus=False):
    """Validate using denoised predictions."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            seqs = batch["sequences"].to(device)
            # predict_denoised always returns mu only (not variance)
            preds = model.predict_denoised(seqs)
            # Handle case where model returns tuple (shouldn't happen with predict_denoised)
            if isinstance(preds, tuple):
                preds = preds[0]
            all_preds.append(preds.cpu().numpy())
            if use_consensus:
                all_targets.append(batch["consensus_ranks"].numpy())
            else:
                all_targets.append(batch["activities"].numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    pearson_r = float(pearsonr(preds, targets)[0])
    spearman_r = float(spearmanr(preds, targets)[0])
    mse = float(np.mean((preds - targets) ** 2))
    return {"pearson": pearson_r, "spearman": spearman_r, "mse": mse}


def train_epoch(model, main_loader, paired_loader, condition, config, optimizer, device,
                augmenter=None):
    """
    Train one epoch with support for all conditions C0-C5 plus new experiments.

    For C0/C1: uses only main_loader
    For C2-C5: uses main_loader + paired_loader
    New: B1 (sensitivity), B2 (variant-contrastive), A2 (hierarchical contrastive)
    Phase 3: augmenter for RC-Mixup / EvoAug-Lite
    """
    model.train()
    epoch_losses = []
    loss_components_accum = {}

    # Set up loss functions
    ranking_loss_fn = AdaptiveMarginRankingLoss(config) if condition in (
        "ranking_only", "ranking_contrastive", "full_disentangle"
    ) else None
    consensus_loss_fn = ConsensusLoss(config) if condition in (
        "consensus_only", "full_disentangle"
    ) else None

    # B2: Variant-contrastive loss
    w_variant_contrastive = config.get("w_variant_contrastive", 0.0)
    variant_contrastive_fn = VariantContrastiveLoss(config) if w_variant_contrastive > 0 else None

    # A2: Hierarchical contrastive
    use_hierarchical = config.get("use_hierarchical_contrastive", False)
    hierarchical_fn = HierarchicalContrastiveLoss(config) if use_hierarchical else None

    # B1: Sensitivity loss (for two-stage)
    w_sensitivity = config.get("w_sensitivity", 0.0)
    n_sensitivity_mutations = config.get("n_sensitivity_mutations", 5)

    # Heteroscedastic loss (Phase 2)
    use_heteroscedastic = config.get("use_heteroscedastic", False)
    heteroscedastic_fn = HeteroscedasticLoss(config) if use_heteroscedastic else None
    use_heteroscedastic_mse = config.get("use_heteroscedastic_mse", False)
    heteroscedastic_mse_fn = HeteroscedasticMSELoss(config) if use_heteroscedastic_mse else None

    # Loss weights
    w_mse = config.get("w_mse", 1.0 if condition in ("baseline_mse", "contrastive_only") else 0.0)
    w_ranking = config.get("w_ranking", 1.0 if condition in ("ranking_only",) else 0.0)
    w_contrastive = config.get("w_contrastive", 0.0)
    w_consensus = config.get("w_consensus", 0.0)
    temperature = config.get("contrastive_temperature", 0.07)

    # Paired data iterator (cycles)
    paired_iter = iter(paired_loader) if paired_loader else None

    for batch in main_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Apply augmentation before forward pass (Phase 3)
        if augmenter is not None:
            aug_seqs, aug_acts, aug_stds = augmenter(
                batch["sequences"],
                batch["activities"],
                batch.get("replicate_std")
            )
            batch["sequences"] = aug_seqs
            batch["activities"] = aug_acts
            if aug_stds is not None:
                batch["replicate_std"] = aug_stds

        optimizer.zero_grad()

        loss_parts = []
        components = {}

        # --- Main data losses ---
        predictions = None
        log_var = None

        # Forward pass - handle heteroscedastic (two-output) case
        if use_heteroscedastic:
            predictions, log_var = model(batch["sequences"], return_variance=True)
        elif w_mse > 0 or w_ranking > 0 or use_heteroscedastic_mse:
            predictions = model(batch["sequences"], return_variance=False)

        # Heteroscedastic loss (Beta-NLL) - replaces MSE when enabled
        if use_heteroscedastic and heteroscedastic_fn is not None and log_var is not None:
            het_loss = heteroscedastic_fn(
                predictions, log_var, batch["activities"],
                replicate_std=batch.get("replicate_std")
            )
            loss_parts.append(het_loss)
            components["heteroscedastic"] = het_loss.item()
        elif use_heteroscedastic_mse and heteroscedastic_mse_fn is not None:
            # Simplified: use replicate_std as weights, no variance prediction
            het_mse_loss = heteroscedastic_mse_fn(
                predictions, batch["activities"], batch.get("replicate_std")
            )
            loss_parts.append(het_mse_loss)
            components["heteroscedastic_mse"] = het_mse_loss.item()
        elif w_mse > 0:
            mse_loss = F.mse_loss(predictions, batch["activities"])
            loss_parts.append(w_mse * mse_loss)
            components["mse"] = mse_loss.item()

        if w_ranking > 0 and ranking_loss_fn is not None:
            # Ranking loss uses mu only (not log_var)
            rank_loss = ranking_loss_fn(
                predictions, batch["activities"], batch.get("replicate_std")
            )
            loss_parts.append(w_ranking * rank_loss)
            components["ranking"] = rank_loss.item()

        # B2: Variant-contrastive loss (uses main batch sequences)
        if w_variant_contrastive > 0 and variant_contrastive_fn is not None:
            vc_loss = variant_contrastive_fn(model, batch["sequences"])
            loss_parts.append(w_variant_contrastive * vc_loss)
            components["variant_contrastive"] = vc_loss.item()

        # B1: Sensitivity loss (two-stage)
        if w_sensitivity > 0:
            sens_loss = compute_sensitivity_loss(
                model, batch["sequences"], n_mutations=n_sensitivity_mutations
            )
            loss_parts.append(w_sensitivity * sens_loss)
            components["sensitivity"] = sens_loss.item()

        # --- Paired data losses ---
        if paired_iter and (w_contrastive > 0 or w_consensus > 0):
            try:
                paired_batch = next(paired_iter)
            except StopIteration:
                paired_iter = iter(paired_loader)
                paired_batch = next(paired_iter)
            paired_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in paired_batch.items()}

            paired_seqs = paired_batch["sequences"]

            # Contrastive loss (standard or hierarchical)
            if w_contrastive > 0:
                if use_hierarchical and hierarchical_fn is not None:
                    # A2: Hierarchical contrastive with activity weighting
                    c_loss = hierarchical_fn(
                        model, paired_seqs,
                        paired_batch["k562_activities"],
                        paired_batch["hepg2_activities"],
                    )
                else:
                    c_loss = compute_contrastive_loss(model, paired_seqs, temperature)
                loss_parts.append(w_contrastive * c_loss)
                components["contrastive"] = c_loss.item()

            # Consensus loss
            if w_consensus > 0 and consensus_loss_fn is not None:
                consensus_preds = model.predict_denoised(paired_seqs)
                cons_loss = consensus_loss_fn(
                    consensus_preds, paired_batch["consensus_ranks"]
                )
                loss_parts.append(w_consensus * cons_loss)
                components["consensus"] = cons_loss.item()

        if not loss_parts:
            continue
        total_loss = sum(loss_parts)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.get("gradient_clip_norm", 1.0),
        )
        optimizer.step()
        epoch_losses.append(total_loss.item())

        # Accumulate component losses for logging
        for k, v in components.items():
            loss_components_accum.setdefault(k, []).append(v)

    avg_loss = np.mean(epoch_losses)
    avg_components = {k: np.mean(v) for k, v in loss_components_accum.items()}
    return avg_loss, avg_components


def main():
    parser = argparse.ArgumentParser(description="DISENTANGLE training")
    parser.add_argument("--architecture", required=True,
                        choices=["cnn", "dilated_cnn", "bilstm", "transformer"])
    parser.add_argument("--condition", required=True,
                        choices=["baseline_mse", "ranking_only", "contrastive_only",
                                 "consensus_only", "ranking_contrastive",
                                 "full_disentangle", "two_stage",
                                 "heteroscedastic_mse", "heteroscedastic_ranking"])
    parser.add_argument("--data", nargs="+", required=True,
                        help="DISENTANGLE-format HDF5 files")
    parser.add_argument("--paired_data", default=None,
                        help="Paired sequences HDF5 (required for C2-C5)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_config", default=None,
                        help="Model config YAML (auto-selected if not given)")
    parser.add_argument("--train_config", default=None,
                        help="Training config YAML (auto-selected if not given)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max epochs from config")
    # B1: Two-stage training
    parser.add_argument("--two_stage", action="store_true",
                        help="Enable two-stage training (B1)")
    parser.add_argument("--stage1_checkpoint", default=None,
                        help="Path to stage 1 checkpoint (best_model.pt from full_disentangle)")
    parser.add_argument("--stage2_lr_factor", type=float, default=0.1,
                        help="LR multiplier for stage 2")
    parser.add_argument("--stage2_freeze_encoder", action="store_true",
                        help="Freeze encoder in stage 2")
    # Phase 2: Heteroscedastic training
    parser.add_argument("--predict_variance", action="store_true",
                        help="Enable variance prediction head for heteroscedastic loss")
    parser.add_argument("--heteroscedastic_beta", type=float, default=0.5,
                        help="Beta parameter for Beta-NLL loss (0=NLL, 0.5=balanced, 1=constant)")
    # Phase 3: Data augmentation
    parser.add_argument("--augmentation", type=str, default="none",
                        choices=["none", "rc_mixup", "evoaug", "both"],
                        help="Data augmentation strategy")
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                        help="Mixup alpha parameter for Beta distribution")
    # Phase 4: Noise curriculum
    parser.add_argument("--noise_curriculum", action="store_true",
                        help="Enable noise-aware curriculum (train on clean samples first)")
    # Phase 5: Cleanlab sample weights
    parser.add_argument("--sample_weights", type=str, default=None,
                        help="Path to sample_weights.npy from cleanlab analysis")
    args = parser.parse_args()

    # Validate paired data requirement
    if args.condition in PAIRED_CONDITIONS and args.paired_data is None:
        parser.error(f"Condition '{args.condition}' requires --paired_data")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Auto-select configs
    model_config_map = {
        "cnn": "configs/models/cnn_basset.yaml",
        "dilated_cnn": "configs/models/dilated_cnn_basenji.yaml",
        "bilstm": "configs/models/bilstm_dream.yaml",
        "transformer": "configs/models/transformer_lite.yaml",
    }
    train_config_map = {
        "baseline_mse": "configs/training/baseline_mse.yaml",
        "ranking_only": "configs/training/ranking_only.yaml",
        "contrastive_only": "configs/training/contrastive_only.yaml",
        "consensus_only": "configs/training/consensus_only.yaml",
        "ranking_contrastive": "configs/training/ranking_contrastive.yaml",
        "full_disentangle": "configs/training/full_disentangle.yaml",
        "two_stage": "configs/training/two_stage.yaml",
        "heteroscedastic_mse": "configs/training/heteroscedastic_mse.yaml",
        "heteroscedastic_ranking": "configs/training/heteroscedastic_ranking.yaml",
    }

    model_config_path = args.model_config or model_config_map[args.architecture]
    train_config_path = args.train_config or train_config_map[args.condition]

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    with open(train_config_path) as f:
        train_config = yaml.safe_load(f)

    config = {**model_config, **train_config}

    # CLI overrides
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.max_epochs:
        config["max_epochs"] = args.max_epochs
    if args.predict_variance:
        config["predict_variance"] = True
        config["use_heteroscedastic"] = True
    if args.heteroscedastic_beta != 0.5:
        config["heteroscedastic_beta"] = args.heteroscedastic_beta
    if args.augmentation != "none":
        config["augmentation"] = args.augmentation
    if args.mixup_alpha != 0.4:
        config["mixup_alpha"] = args.mixup_alpha
    if args.noise_curriculum:
        config["use_noise_curriculum"] = True
    if args.sample_weights:
        config["sample_weights_path"] = args.sample_weights

    config["output_dir"] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Save full config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({**config, "architecture": args.architecture,
                   "condition": args.condition, "seed": args.seed,
                   "data_files": args.data,
                   "paired_data": args.paired_data}, f, indent=2)

    # Load main experiment data
    print("Loading data...")
    quantile_norm = config.get("quantile_normalize", False)
    noise_aug = config.get("noise_augmentation", None)
    noise_level = config.get("noise_level", 0.1)
    train_dataset = SequenceDataset(args.data, split="train",
                                    quantile_normalize=quantile_norm,
                                    noise_augmentation=noise_aug,
                                    noise_level=noise_level)
    val_dataset = SequenceDataset(args.data, split="val")

    # Initialize noise curriculum if enabled (Phase 4)
    noise_curriculum = None
    if config.get("use_noise_curriculum", False):
        noise_curriculum = create_noise_curriculum(train_dataset, config)

    # Load sample weights if provided (Phase 5: Cleanlab)
    sample_weights = None
    if config.get("sample_weights_path"):
        sample_weights = np.load(config["sample_weights_path"])
        print(f"Loaded sample weights from {config['sample_weights_path']}")
        print(f"  Weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
        # Truncate to dataset size (weights may include val set)
        if len(sample_weights) > len(train_dataset):
            sample_weights = sample_weights[:len(train_dataset)]
            print(f"  Truncated to {len(sample_weights)} samples")

    # Create train_loader - may use weighted sampling or noise curriculum
    if noise_curriculum is not None:
        # Noise curriculum takes precedence (recreated each epoch)
        sampler = noise_curriculum.get_sampler(0)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  sampler=sampler, num_workers=4, pin_memory=True,
                                  drop_last=True)
    elif sample_weights is not None:
        # Use cleanlab sample weights
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  sampler=sampler, num_workers=4, pin_memory=True,
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  shuffle=True, num_workers=4, pin_memory=True,
                                  drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Load paired data if needed
    paired_train_loader = None
    paired_val_loader = None
    if args.paired_data:
        print("Loading paired data...")
        paired_train_dataset = PairedSequenceDataset(args.paired_data, split="train")
        paired_val_dataset = PairedSequenceDataset(args.paired_data, split="val")

        paired_train_loader = DataLoader(
            paired_train_dataset, batch_size=config["batch_size"],
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        paired_val_loader = DataLoader(
            paired_val_dataset, batch_size=config["batch_size"],
            shuffle=False, num_workers=4, pin_memory=True)

        print(f"Paired train: {len(paired_train_dataset)}, "
              f"Paired val: {len(paired_val_dataset)}")

    # Count experiments - need at least 2 for paired conditions
    n_experiments = max(len(args.data), 2 if args.paired_data else 1)
    # B1: Two-stage needs to match stage 1 checkpoint architecture
    if args.two_stage and args.stage1_checkpoint:
        stage1_state = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
        n_exp_norms = sum(1 for k in stage1_state if k.startswith("exp_norms.") and k.endswith(".weight"))
        if n_exp_norms > n_experiments:
            n_experiments = n_exp_norms
            print(f"Two-stage: inferred n_experiments={n_experiments} from checkpoint")
        del stage1_state
    print(f"N experiments: {n_experiments}")

    # Build model
    encoder = build_encoder(args.architecture, config)
    model = DisentangleWrapper(encoder, n_experiments=n_experiments, config=config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.architecture} with DisentangleWrapper, {n_params:,} parameters")

    # B1: Two-stage training - load stage 1 checkpoint
    if args.two_stage and args.stage1_checkpoint:
        print(f"Two-stage training: loading stage 1 checkpoint from {args.stage1_checkpoint}")
        state_dict = torch.load(args.stage1_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        config["learning_rate"] = config["learning_rate"] * args.stage2_lr_factor
        print(f"  Stage 2 LR: {config['learning_rate']:.6f}")
        if args.stage2_freeze_encoder:
            for param in model.base_model.parameters():
                param.requires_grad = False
            print("  Encoder frozen for stage 2")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["max_epochs"],
        eta_min=config["learning_rate"] * 0.01,
    )

    # Determine validation mode
    use_consensus_val = (args.condition == "consensus_only")

    # Training loop
    patience = config.get("early_stopping_patience", 15)
    best_spearman = -float("inf")
    patience_counter = 0
    history = []

    # Initialize augmenter (Phase 3)
    augmenter = get_augmenter(config)
    if augmenter is not None:
        print(f"Augmentation: {config.get('augmentation', 'none')}")

    print(f"\nStarting training: {args.architecture} / {args.condition} / seed={args.seed}")
    print(f"  LR={config['learning_rate']}, BS={config['batch_size']}, "
          f"max_epochs={config['max_epochs']}, patience={patience}")
    if args.paired_data:
        print(f"  Paired data: {os.path.basename(args.paired_data)}")
        w_c = config.get("w_contrastive", 0.0)
        w_cons = config.get("w_consensus", 0.0)
        print(f"  Loss weights: w_mse={config.get('w_mse', 0.0)}, "
              f"w_ranking={config.get('w_ranking', 0.0)}, "
              f"w_contrastive={w_c}, w_consensus={w_cons}")
    print("-" * 80)

    for epoch in range(config["max_epochs"]):
        t0 = time.time()

        # Update noise curriculum sampler if enabled (Phase 4)
        if noise_curriculum is not None:
            sampler = noise_curriculum.get_sampler(epoch)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                      sampler=sampler, num_workers=4, pin_memory=True,
                                      drop_last=True)
            if epoch == 0 or noise_curriculum.get_phase(epoch) != noise_curriculum.get_phase(epoch - 1):
                print(f"  Noise curriculum phase {noise_curriculum.get_phase(epoch)}")

        # Train
        avg_loss, avg_components = train_epoch(
            model, train_loader, paired_train_loader,
            args.condition, config, optimizer, device,
            augmenter=augmenter
        )
        scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, device, use_consensus=False)

        # Also validate on paired val set if available
        if paired_val_loader and use_consensus_val:
            paired_val_metrics = validate(
                model, paired_val_loader, device, use_consensus=True
            )
            val_spearman_for_stopping = paired_val_metrics["spearman"]
        else:
            val_spearman_for_stopping = val_metrics["spearman"]

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        comp_str = " ".join(f"{k}={v:.4f}" for k, v in avg_components.items())
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} [{comp_str}] | "
              f"val_sp={val_metrics['spearman']:.4f} | "
              f"val_mse={val_metrics['mse']:.4f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "train_components": avg_components,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": lr,
        })

        # Early stopping
        if val_spearman_for_stopping > best_spearman:
            best_spearman = val_spearman_for_stopping
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val_spearman={best_spearman:.4f})")
                break

    # Save final model and history
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "final_model.pt"))

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final evaluation on test set:")
    test_dataset = SequenceDataset(args.data, split="test")
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=4)

    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"),
                                     weights_only=True))
    test_metrics = validate(model, test_loader, device)
    print(f"  Test Pearson:  {test_metrics['pearson']:.4f}")
    print(f"  Test Spearman: {test_metrics['spearman']:.4f}")
    print(f"  Test MSE:      {test_metrics['mse']:.4f}")

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
