#!/usr/bin/env python3
"""
Noise-Resistant Training Script for MPRA Models

Extended training script supporting:
- Novel noise-aware loss functions (rank stability, distributional, noise-gated)
- Distributional models that predict (μ, σ²)
- Factorized encoders with multi-scale decomposition
- Quantile-stratified and noise-aware sampling
- Comprehensive noise avoidance evaluation
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
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import h5py
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import base models
from src.models import (
    DREAM_RNN, DREAM_RNN_SingleOutput, DREAM_RNN_DualHead,
    DREAM_RNN_DomainAdversarial, DREAM_RNN_BiasFactorized, DREAM_RNN_FullAdvanced
)

# Import new distributional and factorized models
from src.models.distributional_head import (
    DREAM_RNN_Distributional, DREAM_RNN_DistributionalDualHead
)
from src.models.factorized_encoder import (
    FactorizedEncoder, FactorizedEncoderVIB, FactorizedEncoderGCAdv, FactorizedEncoderFull
)

# Import base losses
from src.losses import (
    plackett_luce_loss, ranknet_loss, margin_ranknet_loss,
    combined_loss, CombinedLoss, AdaptiveCombinedLoss,
    softsort_loss
)

# Import new noise-aware losses
from src.losses.rank_stability import RankStabilityRankNet, SampledRankStabilityRankNet
from src.losses.distributional import (
    DistributionalLoss, HeteroscedasticDistributionalLoss, VarianceWeightedMSE
)
from src.losses.noise_gated import (
    NoiseGatedRanking, AdaptiveNoiseGatedRanking, NoiseGatedMSERanking
)
from src.losses.contrastive_anchor import (
    ContrastiveNoiseAnchor, TripletNoiseAnchor, SoftContrastiveNoiseAnchor
)

# Import evaluation
from src.evaluation import compute_all_metrics
from src.evaluation.noise_avoidance import NoiseAvoidanceEvaluator

# Import samplers
from src.data.quantile_sampler import (
    QuantileStratifiedSampler, QuantileCurriculum, HardNegativeMiner, HardNegativeSampler
)
from src.data.curriculum import (
    TierBasedCurriculumSampler, CurriculumScheduler,
    QuantileResolutionCurriculum, NoiseCurriculum
)


class NoiseResistantLogger:
    """Extended logger for noise-resistant training."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_log_path = output_dir / "batch_metrics.csv"
        self.epoch_log_path = output_dir / "epoch_metrics.csv"
        self.noise_log_path = output_dir / "noise_metrics.csv"

        self.batch_metrics = []
        self.epoch_metrics = []
        self.noise_metrics = []
        self.global_step = 0

    def log_batch(self, epoch: int, batch_idx: int, metrics: dict):
        self.global_step += 1
        record = {'epoch': epoch, 'batch': batch_idx, 'global_step': self.global_step, **metrics}
        self.batch_metrics.append(record)

    def log_epoch(self, epoch: int, metrics: dict):
        record = {'epoch': epoch, **metrics}
        self.epoch_metrics.append(record)
        print(f"\nEpoch {epoch} Summary:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    def log_noise_metrics(self, epoch: int, metrics: dict):
        record = {'epoch': epoch, **metrics}
        self.noise_metrics.append(record)

    def save(self):
        if self.batch_metrics:
            pd.DataFrame(self.batch_metrics).to_csv(self.batch_log_path, index=False)
        if self.epoch_metrics:
            pd.DataFrame(self.epoch_metrics).to_csv(self.epoch_log_path, index=False)
        if self.noise_metrics:
            pd.DataFrame(self.noise_metrics).to_csv(self.noise_log_path, index=False)

    def close(self):
        self.save()


class CheckpointManager:
    """Manages model checkpoints."""

    def __init__(self, output_dir: Path, keep_n_best: int = 3, save_every_n_epochs: int = 10):
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_best = keep_n_best
        self.save_every_n_epochs = save_every_n_epochs
        self.top_spearman = []

    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, metrics: dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
        }

        if epoch % self.save_every_n_epochs == 0:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, path)

        spearman = metrics.get('val_spearman', float('-inf'))
        path = self.checkpoint_dir / f"best_spearman_epoch_{epoch}.pth"

        if len(self.top_spearman) < self.keep_n_best:
            torch.save(checkpoint, path)
            self.top_spearman.append((spearman, epoch, path))
        else:
            worst_idx = min(range(len(self.top_spearman)), key=lambda i: self.top_spearman[i][0])
            if spearman > self.top_spearman[worst_idx][0]:
                _, _, old_path = self.top_spearman[worst_idx]
                if old_path.exists():
                    old_path.unlink()
                self.top_spearman.pop(worst_idx)
                torch.save(checkpoint, path)
                self.top_spearman.append((spearman, epoch, path))

        self.top_spearman.sort(key=lambda x: x[0], reverse=True)

        if self.top_spearman:
            best_path = self.top_spearman[0][2]
            best_checkpoint = torch.load(best_path, weights_only=False)
            torch.save(best_checkpoint, self.checkpoint_dir / "best_model.pth")

    def load_checkpoint(self, model, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'], checkpoint.get('metrics', {})


def load_data_with_noise(data_path: str, downsample: float = 1.0):
    """Load lentiMPRA data with aleatoric uncertainty."""
    print(f"Loading data from {data_path}")

    with h5py.File(data_path, 'r') as f:
        X_train = f['Train/X'][:].astype(np.float32)
        y_raw = f['Train/y'][:].astype(np.float32)

        X_val = f['Val/X'][:].astype(np.float32)
        y_val_raw = f['Val/y'][:].astype(np.float32)

        X_test = f['Test/X'][:].astype(np.float32)
        y_test_raw = f['Test/y'][:].astype(np.float32)

    # Extract activity and aleatoric uncertainty
    # y[:, 0] = activity, y[:, 1] = aleatoric_uncertainty
    if y_raw.ndim > 1 and y_raw.shape[1] >= 2:
        y_train = y_raw[:, 0]
        noise_train = y_raw[:, 1]
        y_val = y_val_raw[:, 0]
        noise_val = y_val_raw[:, 1]
        y_test = y_test_raw[:, 0]
        noise_test = y_test_raw[:, 1]
    else:
        y_train = y_raw if y_raw.ndim == 1 else y_raw[:, 0]
        noise_train = np.zeros_like(y_train)
        y_val = y_val_raw if y_val_raw.ndim == 1 else y_val_raw[:, 0]
        noise_val = np.zeros_like(y_val)
        y_test = y_test_raw if y_test_raw.ndim == 1 else y_test_raw[:, 0]
        noise_test = np.zeros_like(y_test)

    # Downsample if requested
    if downsample < 1.0:
        rng = np.random.default_rng(1234)
        n_samples = int(len(X_train) * downsample)
        indices = rng.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        noise_train = noise_train[indices]
        print(f"Downsampled training data to {n_samples} samples")

    # Transpose to (batch, channels, seq_len) for PyTorch
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    print(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    print(f"Noise range: Train [{noise_train.min():.3f}, {noise_train.max():.3f}]")

    return (X_train, y_train, noise_train,
            X_val, y_val, noise_val,
            X_test, y_test, noise_test)


def get_model(model_name: str, **kwargs):
    """Get model by name."""
    models = {
        'dream_rnn': DREAM_RNN,
        'dream_rnn_single': DREAM_RNN_SingleOutput,
        'dream_rnn_dual': DREAM_RNN_DualHead,
        'dream_rnn_distributional': DREAM_RNN_Distributional,
        'dream_rnn_distributional_dual': DREAM_RNN_DistributionalDualHead,
        'factorized': FactorizedEncoder,
        'factorized_vib': FactorizedEncoderVIB,
        'factorized_gc_adv': FactorizedEncoderGCAdv,
        'factorized_full': FactorizedEncoderFull,
        'dream_rnn_domain_adversarial': DREAM_RNN_DomainAdversarial,
        'dream_rnn_bias_factorized': DREAM_RNN_BiasFactorized,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name](**kwargs)


def get_loss_function(loss_type: str, **kwargs):
    """Get loss function by name."""
    if loss_type == 'mse':
        return nn.MSELoss()

    elif loss_type == 'rank_stability':
        return RankStabilityRankNet(
            sigma=kwargs.get('sigma', 1.0),
            k=kwargs.get('noise_k', 1.0)
        )

    elif loss_type == 'distributional':
        return DistributionalLoss(lambda_var=kwargs.get('lambda_var', 1.0))

    elif loss_type == 'heteroscedastic_distributional':
        return HeteroscedasticDistributionalLoss(lambda_var=kwargs.get('lambda_var', 0.5))

    elif loss_type == 'noise_gated':
        return NoiseGatedRanking(
            alpha=kwargs.get('alpha', 0.3),
            beta=kwargs.get('beta', 0.1),
            rank_k=kwargs.get('noise_k', 1.0)
        )

    elif loss_type == 'adaptive_noise_gated':
        return AdaptiveNoiseGatedRanking(
            alpha_init=kwargs.get('alpha_init', 0.0),
            alpha_final=kwargs.get('alpha_final', 0.5),
            beta=kwargs.get('beta', 0.1),
            warmup_epochs=kwargs.get('warmup_epochs', 10),
            total_epochs=kwargs.get('total_epochs', 80)
        )

    elif loss_type == 'noise_gated_mse':
        return NoiseGatedMSERanking(
            alpha=kwargs.get('alpha', 0.3),
            rank_k=kwargs.get('noise_k', 1.0)
        )

    elif loss_type == 'contrastive_anchor':
        return ContrastiveNoiseAnchor(
            temperature=kwargs.get('temperature', 0.1),
            noise_threshold=kwargs.get('noise_threshold', 0.5)
        )

    elif loss_type == 'triplet_anchor':
        return TripletNoiseAnchor(
            margin=kwargs.get('margin', 0.5),
            noise_percentile=kwargs.get('noise_percentile', 0.5)
        )

    elif loss_type == 'plackett_luce':
        temperature = kwargs.get('temperature', 1.0)
        return lambda pred, target: plackett_luce_loss(pred, target, temperature=temperature)

    elif loss_type == 'ranknet':
        sigma = kwargs.get('sigma', 1.0)
        return lambda pred, target: ranknet_loss(pred, target, sigma=sigma)

    elif loss_type == 'combined':
        return CombinedLoss(
            alpha=kwargs.get('alpha', 0.5),
            ranking_loss_fn=kwargs.get('ranking_loss', 'plackett_luce')
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_sampler(sampler_type: str, targets, noise=None, num_samples=None, **kwargs):
    """Get sampler by name."""
    if sampler_type == 'none' or sampler_type is None:
        return None

    elif sampler_type == 'quantile_stratified':
        return QuantileStratifiedSampler(
            targets=targets,
            n_quantiles=kwargs.get('n_quantiles', 10),
            num_samples=num_samples,
            noise=noise,
            noise_weight=kwargs.get('noise_weight', 0.0)
        )

    elif sampler_type == 'quantile_noise_weighted':
        return QuantileStratifiedSampler(
            targets=targets,
            n_quantiles=kwargs.get('n_quantiles', 10),
            num_samples=num_samples,
            noise=noise,
            noise_weight=kwargs.get('noise_weight', 0.5)
        )

    elif sampler_type == 'hard_negative':
        return HardNegativeSampler(
            targets=targets,
            noise=noise,
            num_samples=num_samples,
            temperature=kwargs.get('temperature', 1.0)
        )

    elif sampler_type == 'tier_based':
        from src.data.curriculum import assign_tiers
        tiers = assign_tiers(targets)
        return TierBasedCurriculumSampler(
            tiers=tiers,
            num_samples=num_samples,
            total_epochs=kwargs.get('total_epochs', 80),
            schedule=kwargs.get('schedule', 'linear')
        )

    else:
        raise ValueError(f"Unknown sampler: {sampler_type}")


def train_epoch_noise_aware(model, train_loader, optimizer, scheduler, criterion,
                            device, epoch, logger, is_distributional=False,
                            log_every_n_batches=50):
    """Train one epoch with noise-aware losses."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    loss_components = {}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch (X, y, noise) or (X, y_with_noise)
        if len(batch) == 3:
            batch_x, batch_y, batch_noise = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_noise = batch_noise.to(device)
        else:
            batch_x, batch_y_full = batch
            batch_x = batch_x.to(device)
            batch_y_full = batch_y_full.to(device)
            if batch_y_full.dim() > 1 and batch_y_full.shape[1] >= 2:
                batch_y = batch_y_full[:, 0]
                batch_noise = batch_y_full[:, 1]
            else:
                batch_y = batch_y_full.view(-1)
                batch_noise = torch.zeros_like(batch_y)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)

        # Handle different model outputs and loss types
        if is_distributional:
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                mu, log_var = outputs[0], outputs[1]
                # For distributional losses
                if hasattr(criterion, 'forward') and callable(getattr(criterion, 'forward')):
                    result = criterion(mu, log_var, batch_y, batch_noise)
                    if isinstance(result, dict):
                        loss = result['loss']
                        for k, v in result.items():
                            if k != 'loss' and isinstance(v, (int, float)):
                                loss_components[k] = loss_components.get(k, 0) + v
                    else:
                        loss = result
                else:
                    loss = criterion(mu, log_var, batch_y, batch_noise)
                pred_activity = mu
            else:
                mu = outputs.squeeze()
                loss = nn.MSELoss()(mu, batch_y)
                pred_activity = mu
        elif isinstance(criterion, (RankStabilityRankNet, SampledRankStabilityRankNet)):
            outputs = outputs.squeeze() if not isinstance(outputs, tuple) else outputs[0].squeeze()
            loss = criterion(outputs, batch_y, batch_noise)
            pred_activity = outputs
        elif isinstance(criterion, (NoiseGatedMSERanking,)):
            outputs = outputs.squeeze() if not isinstance(outputs, tuple) else outputs[0].squeeze()
            result = criterion(outputs, batch_y, batch_noise)
            loss = result['loss'] if isinstance(result, dict) else result
            pred_activity = outputs
        elif isinstance(criterion, (ContrastiveNoiseAnchor, TripletNoiseAnchor, SoftContrastiveNoiseAnchor)):
            # For contrastive losses, need embeddings
            if hasattr(model, 'get_embeddings'):
                embeddings = model.get_embeddings(batch_x)
            else:
                embeddings = outputs if outputs.dim() > 1 else outputs.unsqueeze(-1)
            result = criterion(embeddings, batch_y, batch_noise)
            loss = result['loss'] if isinstance(result, dict) else result
            # Also compute MSE for predictions
            if isinstance(outputs, tuple):
                pred_activity = outputs[0].squeeze()
            else:
                pred_activity = outputs.squeeze()
            mse_loss = nn.MSELoss()(pred_activity, batch_y)
            loss = loss + mse_loss  # Combine contrastive + MSE
        else:
            # Standard loss
            if isinstance(outputs, tuple):
                pred_activity = outputs[0].squeeze()
            else:
                pred_activity = outputs.squeeze()
            loss = criterion(pred_activity, batch_y)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        all_preds.extend(pred_activity.detach().cpu().numpy())
        all_targets.extend(batch_y.detach().cpu().numpy())

        if batch_idx % log_every_n_batches == 0:
            batch_metrics = {
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
            }
            logger.log_batch(epoch, batch_idx, batch_metrics)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    train_pearson = pearsonr(all_preds, all_targets)[0]
    train_spearman = spearmanr(all_preds, all_targets)[0]

    result = {
        'train_loss': avg_loss,
        'train_pearson': train_pearson,
        'train_spearman': train_spearman,
    }

    # Add averaged loss components
    if loss_components:
        for k, v in loss_components.items():
            result[f'train_{k}'] = v / len(train_loader)

    return result


def validate_noise_aware(model, val_loader, criterion, device,
                         is_distributional=False, evaluator=None):
    """Validate with noise-aware metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_noise = []
    all_pred_var = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_noise = batch
            else:
                batch_x, batch_y_full = batch
                if batch_y_full.dim() > 1 and batch_y_full.shape[1] >= 2:
                    batch_y = batch_y_full[:, 0]
                    batch_noise = batch_y_full[:, 1]
                else:
                    batch_y = batch_y_full.view(-1)
                    batch_noise = torch.zeros_like(batch_y)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_noise = batch_noise.to(device)

            outputs = model(batch_x)

            if is_distributional:
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    mu, log_var = outputs[0], outputs[1]
                    pred_var = torch.exp(log_var)
                    all_pred_var.append(pred_var.cpu())
                    pred_activity = mu
                else:
                    pred_activity = outputs.squeeze()
            elif isinstance(outputs, tuple):
                pred_activity = outputs[0].squeeze()
            else:
                pred_activity = outputs.squeeze()

            # Compute loss (simplified for validation)
            loss = nn.MSELoss()(pred_activity, batch_y)

            total_loss += loss.item()
            all_preds.append(pred_activity.cpu())
            all_targets.append(batch_y.cpu())
            all_noise.append(batch_noise.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_noise = torch.cat(all_noise).numpy()

    # Compute standard metrics
    metrics = compute_all_metrics(all_preds, all_targets, k_values=[10, 50, 100])
    metrics['val_loss'] = total_loss / len(val_loader)

    # Compute noise-aware metrics if evaluator provided
    if evaluator is not None:
        noise_metrics = evaluator.residual_noise_correlation(
            torch.tensor(all_preds),
            torch.tensor(all_targets),
            torch.tensor(all_noise)
        )
        metrics['residual_noise_corr'] = noise_metrics['residual_noise_pearson']

        stratified = evaluator.stratified_performance(
            torch.tensor(all_preds),
            torch.tensor(all_targets),
            torch.tensor(all_noise)
        )
        if 'summary' in stratified:
            metrics['low_noise_spearman'] = stratified['summary']['low_noise_spearman']
            metrics['high_noise_spearman'] = stratified['summary']['high_noise_spearman']
            metrics['noise_performance_gap'] = stratified['summary']['performance_gap']

        if all_pred_var:
            all_pred_var = torch.cat(all_pred_var).numpy()
            var_metrics = evaluator.noise_prediction_accuracy(
                torch.tensor(all_pred_var),
                torch.tensor(all_noise ** 2)
            )
            metrics['variance_prediction_r'] = var_metrics['variance_spearman_r']

    return metrics, all_preds, all_targets


def main(args):
    print("=" * 60)
    print("Noise-Resistant MPRA Training Script")
    print("=" * 60)

    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out) / f"{args.experiment}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize logging
    logger = NoiseResistantLogger(output_dir)
    checkpoint_mgr = CheckpointManager(output_dir, save_every_n_epochs=args.save_every)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load data with noise
    (X_train, y_train, noise_train,
     X_val, y_val, noise_val,
     X_test, y_test, noise_test) = load_data_with_noise(args.data, downsample=args.downsample)

    # Create datasets (include noise as third tensor or stack with y)
    if args.include_noise_in_batch:
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(noise_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
            torch.FloatTensor(noise_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test),
            torch.FloatTensor(noise_test)
        )
    else:
        # Stack y and noise
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(np.stack([y_train, noise_train], axis=1))
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(np.stack([y_val, noise_val], axis=1))
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(np.stack([y_test, noise_test], axis=1))
        )

    # Create sampler
    sampler = get_sampler(
        args.sampler,
        targets=torch.FloatTensor(y_train),
        noise=torch.FloatTensor(noise_train),
        num_samples=len(train_dataset),
        n_quantiles=args.n_quantiles,
        noise_weight=args.noise_weight,
        total_epochs=args.epochs
    )

    if sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Determine if model is distributional
    is_distributional = args.model in ['dream_rnn_distributional', 'dream_rnn_distributional_dual']

    # Create model
    model_kwargs = {}
    if 'factorized' in args.model:
        model_kwargs['vib_beta'] = args.vib_beta
        model_kwargs['gc_bins'] = args.gc_bins
    if 'domain_adversarial' in args.model:
        model_kwargs['n_domains'] = args.n_domains

    model = get_model(args.model, **model_kwargs)
    model = model.to(device)
    print(f"Model: {args.model} with {sum(p.numel() for p in model.parameters())} parameters")

    # Create loss function
    loss_kwargs = {
        'alpha': args.alpha,
        'beta': args.beta,
        'sigma': args.sigma,
        'noise_k': args.noise_k,
        'lambda_var': args.lambda_var,
        'temperature': args.temperature,
        'total_epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'ranking_loss': args.ranking_loss,
    }
    criterion = get_loss_function(args.loss, **loss_kwargs)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                               steps_per_epoch=len(train_loader), epochs=args.epochs)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    else:
        scheduler = None

    # Noise evaluator
    evaluator = NoiseAvoidanceEvaluator(n_quantiles=5)

    # Quantile curriculum if enabled
    quantile_curriculum = None
    if args.quantile_curriculum:
        quantile_curriculum = QuantileResolutionCurriculum(total_epochs=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Update samplers/curriculums
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
        if quantile_curriculum is not None:
            quantile_curriculum.set_epoch(epoch)

        # Train
        train_metrics = train_epoch_noise_aware(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, logger, is_distributional=is_distributional,
            log_every_n_batches=args.log_every
        )

        # Validate
        val_metrics, val_preds, val_targets = validate_noise_aware(
            model, val_loader, criterion, device,
            is_distributional=is_distributional, evaluator=evaluator
        )

        # Combine metrics
        epoch_metrics = {**train_metrics}
        epoch_metrics['val_loss'] = val_metrics['val_loss']
        epoch_metrics['val_pearson'] = val_metrics['pearson']
        epoch_metrics['val_spearman'] = val_metrics['spearman']

        # Add noise-aware metrics
        if 'residual_noise_corr' in val_metrics:
            epoch_metrics['residual_noise_corr'] = val_metrics['residual_noise_corr']
        if 'noise_performance_gap' in val_metrics:
            epoch_metrics['noise_performance_gap'] = val_metrics['noise_performance_gap']

        # Log epoch
        logger.log_epoch(epoch, epoch_metrics)

        # Checkpoint
        checkpoint_mgr.save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics)

        # Save logs
        logger.save()

        print(f"Epoch {epoch}/{args.epochs}: "
              f"Train Loss: {train_metrics['train_loss']:.4f}, "
              f"Val Spearman: {val_metrics['spearman']:.4f}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    checkpoint_mgr.load_checkpoint(model)
    test_metrics, test_preds, test_targets = validate_noise_aware(
        model, test_loader, criterion, device,
        is_distributional=is_distributional, evaluator=evaluator
    )

    print("\nTest Results:")
    for key, value in sorted(test_metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Save results
    results = {'test_metrics': test_metrics, 'config': config}
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)

    np.save(output_dir / "test_predictions.npy", test_preds)
    np.save(output_dir / "test_targets.npy", test_targets)
    torch.save(model.state_dict(), output_dir / "final_model.pth")

    logger.close()
    print(f"\nResults saved to {output_dir}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise-Resistant MPRA Training")

    # Data
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 data file")
    parser.add_argument("--out", type=str, default="results/noise_resistant", help="Output directory")
    parser.add_argument("--experiment", type=str, default="NR", help="Experiment name")
    parser.add_argument("--downsample", type=float, default=1.0, help="Downsample ratio")

    # Model
    parser.add_argument("--model", type=str, default="dream_rnn_single",
                        choices=["dream_rnn", "dream_rnn_single", "dream_rnn_dual",
                                 "dream_rnn_distributional", "dream_rnn_distributional_dual",
                                 "factorized", "factorized_vib", "factorized_gc_adv", "factorized_full",
                                 "dream_rnn_domain_adversarial", "dream_rnn_bias_factorized"],
                        help="Model architecture")
    parser.add_argument("--n_domains", type=int, default=10, help="Domains for adversarial training")
    parser.add_argument("--vib_beta", type=float, default=0.01, help="VIB beta parameter")
    parser.add_argument("--gc_bins", type=int, default=10, help="GC adversary bins")

    # Loss
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse", "rank_stability", "distributional",
                                 "heteroscedastic_distributional", "noise_gated",
                                 "adaptive_noise_gated", "noise_gated_mse",
                                 "contrastive_anchor", "triplet_anchor",
                                 "plackett_luce", "ranknet", "combined"],
                        help="Loss function")
    parser.add_argument("--alpha", type=float, default=0.3, help="Ranking loss weight")
    parser.add_argument("--beta", type=float, default=0.1, help="Variance supervision weight")
    parser.add_argument("--sigma", type=float, default=1.0, help="RankNet sigma")
    parser.add_argument("--noise_k", type=float, default=1.0, help="Noise scaling factor")
    parser.add_argument("--lambda_var", type=float, default=0.5, help="Variance loss weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--ranking_loss", type=str, default="plackett_luce", help="Ranking loss type")

    # Sampler
    parser.add_argument("--sampler", type=str, default="none",
                        choices=["none", "quantile_stratified", "quantile_noise_weighted",
                                 "hard_negative", "tier_based"],
                        help="Sampling strategy")
    parser.add_argument("--n_quantiles", type=int, default=10, help="Number of quantiles")
    parser.add_argument("--noise_weight", type=float, default=0.0, help="Noise-based sampling weight")
    parser.add_argument("--quantile_curriculum", action="store_true", help="Use quantile resolution curriculum")

    # Training
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--scheduler", type=str, default="onecycle",
                        choices=["onecycle", "cosine", "none"], help="LR scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    # Data handling
    parser.add_argument("--include_noise_in_batch", action="store_true",
                        help="Include noise as separate tensor in batch")

    # Logging
    parser.add_argument("--log_every", type=int, default=50, help="Log every N batches")
    parser.add_argument("--save_every", type=int, default=10, help="Save every N epochs")

    args = parser.parse_args()
    main(args)
