"""
Main training loop for DISENTANGLE.

Handles multi-experiment data loading, loss computation, logging, and checkpointing.

Usage:
    python -m training.trainer \
        --architecture cnn \
        --training_condition full_disentangle \
        --config configs/training/full_disentangle.yaml \
        --model_config configs/models/cnn_basset.yaml \
        --data_files data/processed/encode_lentimpra_K562.h5 data/processed/dream_K562.h5 \
        --output_dir results/ablation/cnn_full_disentangle_seed42 \
        --seed 42
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, kendalltau
from torch.utils.data import DataLoader, Dataset
import h5py

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.encoders import build_encoder
from models.wrapper import DisentangleWrapper
from training.losses.disentangle import DisentangleLoss
from training.losses.mse import MSELoss
from training.losses.ranking import AdaptiveMarginRankingLoss
from training.losses.contrastive import NoiseContrastiveLoss
from training.losses.consensus import ConsensusLoss


class MultiExperimentDataset(Dataset):
    """Dataset combining multiple experiments with experiment labels."""

    def __init__(self, data_files: list[str], splits_file: str, split: str = "train"):
        self.sequences = []
        self.activities = []
        self.experiment_ids = []
        self.replicate_stds = []

        with open(splits_file) as f:
            splits = json.load(f)

        for exp_id_str, split_info in splits.items():
            exp_id = int(exp_id_str)
            filepath = split_info["filepath"]
            indices = split_info[split]

            with h5py.File(filepath, "r") as h5f:
                seqs = h5f["sequences"][:]
                acts = h5f["activities"][:]
                has_std = "replicate_std" in h5f
                stds = h5f["replicate_std"][:] if has_std else None

            self.sequences.append(seqs[indices])
            self.activities.append(acts[indices])
            self.experiment_ids.append(np.full(len(indices), exp_id, dtype=np.int32))
            if stds is not None:
                self.replicate_stds.append(stds[indices])

        self.sequences = np.concatenate(self.sequences, axis=0)
        self.activities = np.concatenate(self.activities, axis=0)
        self.experiment_ids = np.concatenate(self.experiment_ids, axis=0)
        if self.replicate_stds:
            self.replicate_stds = np.concatenate(self.replicate_stds, axis=0)
        else:
            self.replicate_stds = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "sequences": torch.tensor(self.sequences[idx], dtype=torch.float32),
            "activities": torch.tensor(self.activities[idx], dtype=torch.float32),
            "experiment_ids": torch.tensor(self.experiment_ids[idx], dtype=torch.long),
        }
        if self.replicate_stds is not None:
            item["replicate_std"] = torch.tensor(
                self.replicate_stds[idx], dtype=torch.float32
            )
        return item


class DisentangleTrainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, config: dict):
        self.model = model.cuda()
        self.loss_fn = loss_fn
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-5),
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["max_epochs"],
            eta_min=config["learning_rate"] * 0.01,
        )

        self.max_epochs = config["max_epochs"]
        self.patience = config.get("early_stopping_patience", 15)
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            paired_loader: DataLoader = None):
        """Main training loop."""
        if HAS_WANDB and self.config.get("wandb_project"):
            wandb.init(
                project=self.config["wandb_project"],
                config=self.config,
                name=self.config.get("wandb_name",
                                     f"exp_{time.strftime('%Y%m%d_%H%M%S')}"),
            )

        best_val_metric = -float("inf")
        patience_counter = 0
        paired_iter = None

        for epoch in range(self.max_epochs):
            self.model.train()
            train_losses = []

            for batch in train_loader:
                batch = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Add contrastive pairs if available
                if paired_loader is not None:
                    try:
                        paired_batch = next(paired_iter)
                    except (StopIteration, TypeError):
                        paired_iter = iter(paired_loader)
                        paired_batch = next(paired_iter)

                    batch["anchor_sequences"] = paired_batch["anchor_sequences"].cuda()
                    batch["positive_sequences"] = paired_batch["positive_sequences"].cuda()
                    batch["negative_sequences"] = paired_batch["negative_sequences"].cuda()

                self.optimizer.zero_grad()

                if isinstance(self.loss_fn, DisentangleLoss):
                    total_loss, loss_components = self.loss_fn(self.model, batch)
                elif isinstance(self.loss_fn, MSELoss):
                    predictions = self.model(batch["sequences"])
                    total_loss = self.loss_fn(predictions, batch["activities"])
                    loss_components = {"total": total_loss.item(), "mse": total_loss.item()}
                else:
                    predictions = self.model(batch["sequences"])
                    total_loss = self.loss_fn(predictions, batch["activities"])
                    loss_components = {"total": total_loss.item()}

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get("gradient_clip_norm", 1.0),
                )
                self.optimizer.step()
                train_losses.append(loss_components)

            self.scheduler.step()

            # Validation
            val_metrics = self._validate(val_loader)

            # Logging
            avg_train = {}
            if train_losses:
                all_keys = set()
                for lc in train_losses:
                    all_keys.update(lc.keys())
                avg_train = {
                    k: np.mean([lc[k] for lc in train_losses if k in lc])
                    for k in all_keys
                }

            if HAS_WANDB and wandb.run is not None:
                log_dict = {f"train/{k}": v for k, v in avg_train.items()}
                log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                log_dict["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(log_dict, step=epoch)

            print(
                f"Epoch {epoch}: train_loss={avg_train.get('total', 0):.4f}, "
                f"val_spearman={val_metrics.get('spearman', 0):.4f}"
            )

            # Early stopping on validation Spearman
            val_metric = val_metrics.get("spearman", 0)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, "best.pt"),
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        torch.save(
            self.model.state_dict(),
            os.path.join(self.output_dir, "final.pt"),
        )

        if HAS_WANDB and wandb.run is not None:
            wandb.finish()

    def _validate(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                sequences = batch["sequences"].cuda()
                if hasattr(self.model, "predict_denoised"):
                    predictions = self.model.predict_denoised(sequences)
                else:
                    predictions = self.model(sequences)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch["activities"].numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        return {
            "pearson": float(pearsonr(preds, targets)[0]),
            "spearman": float(spearmanr(preds, targets)[0]),
            "kendall": float(kendalltau(preds, targets)[0]),
            "mse": float(np.mean((preds - targets) ** 2)),
        }


def build_loss_fn(condition: str, config: dict) -> nn.Module:
    """Build loss function from training condition name."""
    if condition == "baseline_mse":
        return MSELoss()
    elif condition == "ranking_only":
        return AdaptiveMarginRankingLoss(config)
    elif condition == "contrastive_only":
        return NoiseContrastiveLoss(config)
    elif condition == "consensus_only":
        return ConsensusLoss(config)
    elif condition in ("ranking_contrastive", "full_disentangle"):
        return DisentangleLoss(config)
    else:
        raise ValueError(f"Unknown condition: {condition}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DISENTANGLE training")
    parser.add_argument("--architecture", required=True,
                        choices=["cnn", "dilated_cnn", "bilstm", "transformer"])
    parser.add_argument("--training_condition", required=True)
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--model_config", required=True, help="Model config YAML")
    parser.add_argument("--data_files", nargs="+", required=True)
    parser.add_argument("--splits_file", required=True)
    parser.add_argument("--paired_sequences", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load configs
    import yaml
    with open(args.config) as f:
        train_config = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    config = {**model_config, **train_config}
    config["output_dir"] = args.output_dir
    config["wandb_project"] = args.wandb_project
    config["wandb_name"] = args.wandb_name

    # Build model
    encoder = build_encoder(args.architecture, config)
    n_experiments = len(args.data_files)
    model = DisentangleWrapper(encoder, n_experiments, config)

    # Build loss
    loss_fn = build_loss_fn(args.training_condition, config)

    # Build data loaders
    train_dataset = MultiExperimentDataset(args.data_files, args.splits_file, "train")
    val_dataset = MultiExperimentDataset(args.data_files, args.splits_file, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train
    trainer = DisentangleTrainer(model, loss_fn, config)
    trainer.fit(train_loader, val_loader)
