#!/usr/bin/env python3
"""
Enhanced Training Script for MPRA Models

Features:
- Multiple loss functions (MSE, Plackett-Luce, RankNet, SoftSort, Combined)
- Curriculum learning support
- Batch-level logging
- Per-epoch validation metrics history
- Training curves / checkpoints
- TensorBoard support
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
import h5py
import yaml
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    DREAM_RNN, DREAM_RNN_SingleOutput, DREAM_RNN_DualHead,
    DREAM_RNN_DomainAdversarial, DREAM_RNN_BiasFactorized, DREAM_RNN_FullAdvanced
)
from src.losses import (
    plackett_luce_loss, ranknet_loss, margin_ranknet_loss,
    combined_loss, CombinedLoss, AdaptiveCombinedLoss,
    SoftClassificationLoss,
    softsort_loss  # Has pure PyTorch fallback if torchsort not available
)
from src.evaluation import compute_all_metrics, spearman_correlation, pearson_correlation
from src.data import (
    assign_tiers, TierBasedCurriculumSampler, CurriculumScheduler,
    compute_batch_difficulty_metrics
)

# Optional TensorBoard - defer import to avoid TF/numpy conflicts
TENSORBOARD_AVAILABLE = False
SummaryWriter = None

def _try_import_tensorboard():
    global TENSORBOARD_AVAILABLE, SummaryWriter
    try:
        from torch.utils.tensorboard import SummaryWriter as SW
        SummaryWriter = SW
        TENSORBOARD_AVAILABLE = True
    except (ImportError, AttributeError, Exception) as e:
        print(f"TensorBoard not available: {e}")
        TENSORBOARD_AVAILABLE = False


class TrainingLogger:
    """Handles all logging: console, CSV, TensorBoard."""

    def __init__(self, output_dir: Path, use_tensorboard: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV logging
        self.batch_log_path = output_dir / "batch_metrics.csv"
        self.epoch_log_path = output_dir / "epoch_metrics.csv"
        self.batch_metrics = []
        self.epoch_metrics = []

        # TensorBoard - lazy import
        self.writer = None
        if use_tensorboard:
            _try_import_tensorboard()
            if TENSORBOARD_AVAILABLE and SummaryWriter is not None:
                try:
                    self.writer = SummaryWriter(log_dir=output_dir / "tensorboard")
                except Exception as e:
                    print(f"Could not initialize TensorBoard: {e}")
                    self.writer = None

        self.global_step = 0

    def log_batch(self, epoch: int, batch_idx: int, metrics: dict):
        """Log batch-level metrics."""
        self.global_step += 1

        record = {
            'epoch': epoch,
            'batch': batch_idx,
            'global_step': self.global_step,
            **metrics
        }
        self.batch_metrics.append(record)

        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'batch/{key}', value, self.global_step)

    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch-level metrics."""
        record = {
            'epoch': epoch,
            **metrics
        }
        self.epoch_metrics.append(record)

        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'epoch/{key}', value, epoch)

        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    def save(self):
        """Save all logs to CSV."""
        if self.batch_metrics:
            pd.DataFrame(self.batch_metrics).to_csv(self.batch_log_path, index=False)
        if self.epoch_metrics:
            pd.DataFrame(self.epoch_metrics).to_csv(self.epoch_log_path, index=False)
        if self.writer:
            self.writer.flush()

    def close(self):
        """Close all loggers."""
        self.save()
        if self.writer:
            self.writer.close()


class CheckpointManager:
    """Manages model checkpoints - tracks top 3 for Pearson, Spearman, and Loss."""

    def __init__(self, output_dir: Path, keep_n_best: int = 3,
                 save_every_n_epochs: int = 10):
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_best = keep_n_best
        self.save_every_n_epochs = save_every_n_epochs

        # Track top 3 for each metric: List of (metric_value, epoch, path)
        self.top_pearson = []   # Higher is better
        self.top_spearman = []  # Higher is better
        self.top_loss = []      # Lower is better

    def _update_top_list(self, top_list, value, epoch, checkpoint,
                         metric_name, higher_is_better=True):
        """Update a top-N list, saving/removing checkpoints as needed."""
        path = self.checkpoint_dir / f"best_{metric_name}_epoch_{epoch}.pth"

        # Check if this should be in top N
        if len(top_list) < self.keep_n_best:
            # Room available, just add
            torch.save(checkpoint, path)
            top_list.append((value, epoch, path))
        else:
            # Check if better than worst in list
            if higher_is_better:
                worst_idx = min(range(len(top_list)), key=lambda i: top_list[i][0])
                should_add = value > top_list[worst_idx][0]
            else:
                worst_idx = max(range(len(top_list)), key=lambda i: top_list[i][0])
                should_add = value < top_list[worst_idx][0]

            if should_add:
                # Remove worst checkpoint file
                _, _, old_path = top_list[worst_idx]
                if old_path.exists():
                    old_path.unlink()
                top_list.pop(worst_idx)

                # Add new one
                torch.save(checkpoint, path)
                top_list.append((value, epoch, path))

        # Sort for display (best first)
        if higher_is_better:
            top_list.sort(key=lambda x: x[0], reverse=True)
        else:
            top_list.sort(key=lambda x: x[0])

    def save_checkpoint(self, model, optimizer, scheduler, epoch: int,
                        metrics: dict):
        """Save checkpoint, updating top 3 for each metric."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
        }

        # Periodic checkpoint (every N epochs)
        if epoch % self.save_every_n_epochs == 0:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, path)

        # Update top 3 for each metric
        pearson = metrics.get('val_pearson', metrics.get('pearson', float('-inf')))
        spearman = metrics.get('val_spearman', metrics.get('spearman', float('-inf')))
        val_loss = metrics.get('val_loss', float('inf'))

        self._update_top_list(self.top_pearson, pearson, epoch, checkpoint,
                              'pearson', higher_is_better=True)
        self._update_top_list(self.top_spearman, spearman, epoch, checkpoint,
                              'spearman', higher_is_better=True)
        self._update_top_list(self.top_loss, val_loss, epoch, checkpoint,
                              'loss', higher_is_better=False)

        # Save pointer to overall best (by Spearman - most relevant for ranking)
        if self.top_spearman:
            best_spearman_path = self.top_spearman[0][2]
            best_checkpoint = torch.load(best_spearman_path, weights_only=False)
            torch.save(best_checkpoint, self.checkpoint_dir / "best_model.pth")

    def load_checkpoint(self, model, optimizer=None, scheduler=None,
                        checkpoint_path: str = None):
        """Load a checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint.get('metrics', {})

    def print_summary(self):
        """Print summary of best checkpoints."""
        print("\n" + "=" * 50)
        print("Best Checkpoints Summary")
        print("=" * 50)
        print("\nTop 3 by Pearson:")
        for val, ep, _ in self.top_pearson:
            print(f"  Epoch {ep}: {val:.4f}")
        print("\nTop 3 by Spearman:")
        for val, ep, _ in self.top_spearman:
            print(f"  Epoch {ep}: {val:.4f}")
        print("\nTop 3 by Loss (lowest):")
        for val, ep, _ in self.top_loss:
            print(f"  Epoch {ep}: {val:.4f}")


def get_loss_function(loss_type: str, **kwargs):
    """Get the appropriate loss function."""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'soft_classification':
        n_bins = kwargs.get('n_bins', 10)
        label_smoothing = kwargs.get('label_smoothing', 0.1)
        return SoftClassificationLoss(n_bins=n_bins, label_smoothing=label_smoothing)
    elif loss_type == 'plackett_luce':
        temperature = kwargs.get('temperature', 1.0)
        return lambda pred, target: plackett_luce_loss(pred, target, temperature=temperature)
    elif loss_type == 'ranknet':
        sigma = kwargs.get('sigma', 1.0)
        return lambda pred, target: ranknet_loss(pred, target, sigma=sigma)
    elif loss_type == 'margin_ranknet':
        margin = kwargs.get('margin', 0.1)
        return lambda pred, target: margin_ranknet_loss(pred, target, base_margin=margin)
    elif loss_type == 'softsort':
        # Uses pure PyTorch fallback if torchsort not available
        reg = kwargs.get('regularization', 1.0)
        return lambda pred, target: softsort_loss(pred, target, regularization=reg)
    elif loss_type == 'combined':
        alpha = kwargs.get('alpha', 0.5)
        ranking_loss = kwargs.get('ranking_loss', 'plackett_luce')
        return CombinedLoss(alpha=alpha, ranking_loss_fn=ranking_loss)
    elif loss_type == 'adaptive_combined':
        return AdaptiveCombinedLoss(
            alpha_start=kwargs.get('alpha_start', 0.9),
            alpha_end=kwargs.get('alpha_end', 0.3),
            warmup_epochs=kwargs.get('warmup_epochs', 10),
            total_epochs=kwargs.get('total_epochs', 80),
            ranking_loss_fn=kwargs.get('ranking_loss', 'plackett_luce')
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def load_data(data_path: str, downsample: float = 1.0):
    """Load lentiMPRA data from HDF5."""
    print(f"Loading data from {data_path}")

    with h5py.File(data_path, 'r') as f:
        X_train = f['Train/X'][:].astype(np.float32)
        y_train = f['Train/y'][:, :2].astype(np.float32)  # activity, aleatoric
        X_val = f['Val/X'][:].astype(np.float32)
        y_val = f['Val/y'][:, :2].astype(np.float32)
        X_test = f['Test/X'][:].astype(np.float32)
        y_test = f['Test/y'][:, :2].astype(np.float32)

    # Downsample if requested
    if downsample < 1.0:
        rng = np.random.default_rng(1234)
        n_samples = int(len(X_train) * downsample)
        indices = rng.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Downsampled training data to {n_samples} samples")

    # Transpose to (batch, channels, seq_len) for PyTorch
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    print(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_epoch(model, train_loader, optimizer, scheduler, criterion,
                device, epoch, logger, curriculum=None, log_every_n_batches=50,
                loss_type='mse'):
    """Train for one epoch with batch-level logging."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    is_soft_classification = loss_type == 'soft_classification'

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Extract activity (first column) for single-output models
        batch_y_activity = batch_y[:, 0] if batch_y.dim() > 1 else batch_y

        optimizer.zero_grad()
        outputs = model(batch_x)

        # Handle different output shapes and loss types
        if isinstance(outputs, tuple):
            # Domain adversarial or bias factorized model
            # First element is always activity prediction
            activity_pred = outputs[0]
            loss = criterion(activity_pred, batch_y_activity)
            pred_activity = activity_pred

            # Add domain adversarial loss if present (outputs[1] is domain logits)
            if len(outputs) >= 2 and outputs[1] is not None and hasattr(outputs[1], 'shape'):
                if len(outputs[1].shape) == 2:  # Domain logits [batch, n_domains]
                    # Note: domain labels would need to be provided for full adversarial training
                    # For now, we just use the activity loss
                    pass
        elif is_soft_classification:
            # Soft classification: outputs are logits [batch, n_bins]
            loss = criterion(outputs, batch_y_activity)
            # For metrics, convert bin predictions back to continuous scale
            pred_bins = outputs.argmax(dim=1).float()
            pred_activity = pred_bins / (outputs.shape[1] - 1)  # Normalize to [0, 1]
        elif outputs.dim() > 1 and outputs.shape[1] == 2:
            # Multi-output model (activity + aleatoric)
            loss = criterion(outputs, batch_y)
            pred_activity = outputs[:, 0]
        else:
            # Single output model
            outputs = outputs.squeeze()
            loss = criterion(outputs, batch_y_activity)
            pred_activity = outputs

        # Apply curriculum weighting if available
        if curriculum is not None:
            weights = curriculum.get_batch_weights(batch_y_activity)
            if weights is not None:
                # Recompute weighted loss
                pass  # TODO: implement weighted loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        all_preds.extend(pred_activity.detach().cpu().numpy())
        all_targets.extend(batch_y_activity.detach().cpu().numpy())

        # Batch-level logging
        if batch_idx % log_every_n_batches == 0:
            batch_metrics = {
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
            }

            # Add curriculum metrics if available
            if curriculum is not None:
                diff_metrics = compute_batch_difficulty_metrics(batch_y_activity)
                batch_metrics.update(diff_metrics)

            logger.log_batch(epoch, batch_idx, batch_metrics)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Epoch training metrics
    avg_loss = total_loss / len(train_loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    train_pearson = pearsonr(all_preds, all_targets)[0]
    train_spearman = spearmanr(all_preds, all_targets)[0]

    return {
        'train_loss': avg_loss,
        'train_pearson': train_pearson,
        'train_spearman': train_spearman,
    }


def validate(model, val_loader, criterion, device, loss_type='mse'):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    is_soft_classification = loss_type == 'soft_classification'

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_activity = batch_y[:, 0] if batch_y.dim() > 1 else batch_y

            outputs = model(batch_x)

            if isinstance(outputs, tuple):
                # Domain adversarial or bias factorized model
                activity_pred = outputs[0]
                loss = criterion(activity_pred, batch_y_activity)
                pred_activity = activity_pred
            elif is_soft_classification:
                loss = criterion(outputs, batch_y_activity)
                pred_bins = outputs.argmax(dim=1).float()
                pred_activity = pred_bins / (outputs.shape[1] - 1)
            elif outputs.dim() > 1 and outputs.shape[1] == 2:
                loss = criterion(outputs, batch_y)
                pred_activity = outputs[:, 0]
            else:
                outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y_activity)
                pred_activity = outputs

            total_loss += loss.item()
            all_preds.extend(pred_activity.cpu().numpy())
            all_targets.extend(batch_y_activity.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute all metrics
    metrics = compute_all_metrics(all_preds, all_targets, k_values=[10, 50, 100])
    metrics['val_loss'] = total_loss / len(val_loader)

    return metrics, all_preds, all_targets


def main(args):
    print("=" * 60)
    print("Enhanced MPRA Training Script")
    print("=" * 60)

    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out) / f"{args.experiment}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize logging
    logger = TrainingLogger(output_dir, use_tensorboard=args.tensorboard)
    if args.tensorboard and logger.writer is None:
        print("Warning: TensorBoard requested but not available. Continuing with CSV logging only.")
    checkpoint_mgr = CheckpointManager(output_dir, save_every_n_epochs=args.save_every)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        args.data, downsample=args.downsample
    )

    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    # Curriculum learning sampler if requested
    sampler = None
    if args.curriculum != 'none':
        print(f"Using curriculum learning: {args.curriculum}")
        tiers = assign_tiers(torch.FloatTensor(y_train[:, 0]))
        sampler = TierBasedCurriculumSampler(
            tiers, num_samples=len(train_dataset),
            total_epochs=args.epochs, schedule=args.curriculum
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    # For soft classification, we need n_bins outputs
    n_outputs = args.n_bins if args.loss == 'soft_classification' else 2

    if args.model == 'dream_rnn':
        model = DREAM_RNN(n_outputs=n_outputs)
    elif args.model == 'dream_rnn_single':
        model = DREAM_RNN_SingleOutput()
    elif args.model == 'dream_rnn_dual':
        model = DREAM_RNN_DualHead()
    elif args.model == 'dream_rnn_domain_adversarial':
        model = DREAM_RNN_DomainAdversarial(n_domains=args.n_domains)
    elif args.model == 'dream_rnn_bias_factorized':
        model = DREAM_RNN_BiasFactorized()
    elif args.model == 'dream_rnn_full_advanced':
        model = DREAM_RNN_FullAdvanced(n_domains=args.n_domains)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Warn if using soft_classification with wrong model
    if args.loss == 'soft_classification' and args.model != 'dream_rnn':
        print(f"Warning: soft_classification works best with --model dream_rnn")

    model = model.to(device)
    print(f"Model: {args.model} with {sum(p.numel() for p in model.parameters())} parameters")

    # Loss function
    loss_kwargs = {
        'temperature': args.temperature,
        'alpha': args.alpha,
        'total_epochs': args.epochs,
        'ranking_loss': args.ranking_loss,
        'n_bins': args.n_bins,
        'label_smoothing': 0.1,
    }
    criterion = get_loss_function(args.loss, **loss_kwargs)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                          steps_per_epoch=len(train_loader), epochs=args.epochs)

    # Curriculum scheduler
    curriculum = None
    if args.curriculum != 'none':
        curriculum = CurriculumScheduler(strategy='tier_based', total_epochs=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Update curriculum
        if sampler is not None:
            sampler.set_epoch(epoch)
        if curriculum is not None:
            curriculum.set_epoch(epoch)
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, logger, curriculum,
            log_every_n_batches=args.log_every,
            loss_type=args.loss
        )

        # Validate
        val_metrics, val_preds, val_targets = validate(model, val_loader, criterion, device, loss_type=args.loss)

        # Combine metrics
        epoch_metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items() if not k.startswith('val_')}}
        epoch_metrics['val_loss'] = val_metrics['val_loss']

        # Log epoch
        logger.log_epoch(epoch, epoch_metrics)

        # Checkpoint - saves top 3 for Pearson, Spearman, and Loss
        checkpoint_mgr.save_checkpoint(
            model, optimizer, scheduler, epoch, epoch_metrics
        )

        # Save logs after each epoch (so they're available even if training is interrupted)
        logger.save()

        # Print progress
        print(f"Epoch {epoch}/{args.epochs}: "
              f"Train Loss: {train_metrics['train_loss']:.4f}, "
              f"Val Loss: {val_metrics['val_loss']:.4f}, "
              f"Val Pearson: {val_metrics['pearson']:.4f}, "
              f"Val Spearman: {val_metrics['spearman']:.4f}")

    # Print checkpoint summary
    checkpoint_mgr.print_summary()

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model (best by Spearman)
    checkpoint_mgr.load_checkpoint(model)
    test_metrics, test_preds, test_targets = validate(model, test_loader, criterion, device, loss_type=args.loss)

    print("\nTest Results:")
    for key, value in sorted(test_metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Save final results
    results = {
        'test_metrics': test_metrics,
        'config': config,
    }
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)

    np.save(output_dir / "test_predictions.npy", test_preds)
    np.save(output_dir / "test_targets.npy", test_targets)

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pth")

    # Close logger
    logger.close()

    print(f"\nResults saved to {output_dir}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced MPRA Training Script")

    # Data
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 data file")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    parser.add_argument("--experiment", type=str, default="exp", help="Experiment name")
    parser.add_argument("--downsample", type=float, default=1.0, help="Downsample ratio")

    # Model
    parser.add_argument("--model", type=str, default="dream_rnn",
                       choices=["dream_rnn", "dream_rnn_single", "dream_rnn_dual",
                               "dream_rnn_domain_adversarial", "dream_rnn_bias_factorized",
                               "dream_rnn_full_advanced"],
                       help="Model architecture")
    parser.add_argument("--n_domains", type=int, default=10,
                       help="Number of domains for domain adversarial training")

    # Loss
    parser.add_argument("--loss", type=str, default="mse",
                       choices=["mse", "soft_classification", "plackett_luce", "ranknet",
                               "margin_ranknet", "softsort", "combined", "adaptive_combined"],
                       help="Loss function")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for MSE in combined loss")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for Plackett-Luce")
    parser.add_argument("--ranking_loss", type=str, default="plackett_luce",
                       help="Ranking loss for combined losses")
    parser.add_argument("--n_bins", type=int, default=10,
                       help="Number of bins for soft classification")

    # Training
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    # Curriculum
    parser.add_argument("--curriculum", type=str, default="none",
                       choices=["none", "linear", "stepped", "exponential"],
                       help="Curriculum learning schedule")

    # Logging
    parser.add_argument("--log_every", type=int, default=50,
                       help="Log batch metrics every N batches")
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Enable TensorBoard logging")

    args = parser.parse_args()
    main(args)
