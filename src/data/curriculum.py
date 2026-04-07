"""
Curriculum Learning for MPRA Data

Train on "easy" samples first (sequences with extreme high/low activity)
then progressively introduce "harder" samples (sequences with similar activities).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, WeightedRandomSampler
import numpy as np
from typing import Optional, Dict, List, Tuple


def assign_tiers(y: torch.Tensor, q1: float = 0.33, q2: float = 0.66) -> torch.Tensor:
    """
    Assign difficulty tiers based on distance from median.

    Tier 1: Most extreme values (easiest to distinguish)
    Tier 2: Moderate values
    Tier 3: Close to median (hardest to distinguish)

    Args:
        y: Activity values [n_samples] or [n_samples, 1]
        q1: Lower quantile threshold (default: bottom 33% of distances)
        q2: Upper quantile threshold (default: top 33% of distances)

    Returns:
        Tier assignments [n_samples], values in {1, 2, 3}
    """
    y = y.view(-1)
    median = y.median()
    distances = (y - median).abs()

    # Get quantile thresholds for distances
    thresholds = torch.quantile(distances, torch.tensor([q1, q2], device=y.device))

    tiers = torch.ones_like(y, dtype=torch.long) * 3  # Default: tier 3 (hardest)
    tiers[distances > thresholds[1]] = 1  # Most extreme -> tier 1 (easiest)
    tiers[(distances > thresholds[0]) & (distances <= thresholds[1])] = 2

    return tiers


def compute_extremeness_scores(y: torch.Tensor) -> torch.Tensor:
    """
    Compute continuous extremeness scores for each sample.

    Higher score = more extreme = easier to rank correctly.

    Args:
        y: Activity values [n_samples]

    Returns:
        Extremeness scores [n_samples], normalized to [0, 1]
    """
    y = y.view(-1)
    median = y.median()
    distances = (y - median).abs()

    # Normalize to [0, 1]
    min_dist = distances.min()
    max_dist = distances.max()

    if max_dist - min_dist < 1e-8:
        return torch.ones_like(distances)

    return (distances - min_dist) / (max_dist - min_dist)


class TierBasedCurriculumSampler(Sampler):
    """
    Sampler that adjusts tier probabilities over training.

    Early: Sample mostly from tier 1 (extreme values)
    Late: Sample uniformly across all tiers
    """

    def __init__(self, tiers: torch.Tensor, num_samples: int,
                 total_epochs: int, schedule: str = 'linear'):
        """
        Args:
            tiers: Tier assignments for each sample [n_samples]
            num_samples: Number of samples per epoch
            total_epochs: Total training epochs
            schedule: How to transition weights ('linear', 'stepped', 'exponential')
        """
        self.tiers = tiers.cpu().numpy() if isinstance(tiers, torch.Tensor) else tiers
        self.num_samples = num_samples
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.current_epoch = 0

        # Pre-compute indices for each tier
        self.tier_indices = {
            1: np.where(self.tiers == 1)[0],
            2: np.where(self.tiers == 2)[0],
            3: np.where(self.tiers == 3)[0],
        }

    def get_tier_weights(self) -> Dict[int, float]:
        """Get current tier sampling weights based on training progress."""
        progress = self.current_epoch / max(1, self.total_epochs - 1)
        progress = min(1.0, max(0.0, progress))

        if self.schedule == 'linear':
            # Linear transition from tier 1 heavy to uniform
            easy_weight = 0.7 * (1 - progress) + 0.33 * progress
            return {
                1: easy_weight,
                2: (1 - easy_weight) * 0.6,
                3: (1 - easy_weight) * 0.4
            }

        elif self.schedule == 'stepped':
            # Discrete phases
            if progress < 0.33:
                return {1: 0.8, 2: 0.15, 3: 0.05}
            elif progress < 0.66:
                return {1: 0.4, 2: 0.4, 3: 0.2}
            else:
                return {1: 0.33, 2: 0.33, 3: 0.34}

        elif self.schedule == 'exponential':
            # Smooth exponential transition
            tau = 3.0  # Temperature
            difficulties = np.array([1., 2., 3.])
            threshold = 1 + 2 * progress
            logits = -tau * np.maximum(0, difficulties - threshold)
            probs = np.exp(logits) / np.exp(logits).sum()
            return {i + 1: probs[i] for i in range(3)}

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def set_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch

    def __iter__(self):
        weights = self.get_tier_weights()

        # Build sample weights based on tier
        sample_weights = np.zeros(len(self.tiers))
        for tier, tier_weight in weights.items():
            tier_mask = self.tiers == tier
            n_tier = tier_mask.sum()
            if n_tier > 0:
                # Weight per sample in this tier
                sample_weights[tier_mask] = tier_weight / n_tier

        # Normalize
        sample_weights = sample_weights / sample_weights.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.tiers),
            size=self.num_samples,
            replace=True,
            p=sample_weights
        )

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


class SelfPacedCurriculum:
    """
    Self-paced learning that weights samples based on current model's loss.

    Easy samples (low loss) weighted higher early, harder samples later.
    """

    def __init__(self, lambda_init: float = 0.1, lambda_final: float = 1.0,
                 growth: str = 'linear'):
        """
        Args:
            lambda_init: Initial threshold (only include easy samples)
            lambda_final: Final threshold (include all samples)
            growth: How to grow lambda ('linear', 'exponential')
        """
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.growth = growth
        self.current_lambda = lambda_init

    def get_sample_weights(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute sample weights based on loss values.

        Args:
            losses: Per-sample loss values [batch_size]

        Returns:
            Normalized sample weights [batch_size]
        """
        # Self-paced weighting: v_i = max(0, 1 - loss_i / λ)
        weights = F.relu(1 - losses / self.current_lambda)

        # Avoid all-zero weights
        if weights.sum() < 1e-8:
            weights = torch.ones_like(weights)

        return weights / weights.sum()

    def step(self, progress: float):
        """
        Update lambda based on training progress.

        Args:
            progress: Training progress in [0, 1]
        """
        progress = min(1.0, max(0.0, progress))

        if self.growth == 'linear':
            self.current_lambda = self.lambda_init + (self.lambda_final - self.lambda_init) * progress
        elif self.growth == 'exponential':
            self.current_lambda = self.lambda_init * (self.lambda_final / self.lambda_init) ** progress
        else:
            raise ValueError(f"Unknown growth: {self.growth}")


class CurriculumScheduler:
    """
    Unified curriculum scheduler that supports multiple strategies.
    """

    def __init__(self, strategy: str = 'tier_based', total_epochs: int = 80, **kwargs):
        """
        Args:
            strategy: Curriculum strategy ('tier_based', 'self_paced', 'batch_composition')
            total_epochs: Total training epochs
            **kwargs: Strategy-specific arguments
        """
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.kwargs = kwargs

        if strategy == 'self_paced':
            self.self_paced = SelfPacedCurriculum(
                lambda_init=kwargs.get('lambda_init', 0.1),
                lambda_final=kwargs.get('lambda_final', 1.0),
                growth=kwargs.get('growth', 'linear')
            )

    def set_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch

        if self.strategy == 'self_paced':
            progress = epoch / max(1, self.total_epochs - 1)
            self.self_paced.step(progress)

    def get_batch_weights(self, y_batch: torch.Tensor,
                          losses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get sample weights for a batch.

        Args:
            y_batch: Target values for the batch
            losses: Per-sample losses (required for self_paced)

        Returns:
            Sample weights [batch_size]
        """
        batch_size = y_batch.shape[0]
        progress = self.current_epoch / max(1, self.total_epochs - 1)

        if self.strategy == 'tier_based':
            tiers = assign_tiers(y_batch)
            weights = self._get_tier_weights_for_batch(tiers, progress)

        elif self.strategy == 'self_paced':
            if losses is None:
                return torch.ones(batch_size, device=y_batch.device) / batch_size
            weights = self.self_paced.get_sample_weights(losses)

        elif self.strategy == 'none':
            weights = torch.ones(batch_size, device=y_batch.device) / batch_size

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return weights

    def _get_tier_weights_for_batch(self, tiers: torch.Tensor, progress: float) -> torch.Tensor:
        """Compute tier-based weights for a batch."""
        # Linear transition from easy-focused to uniform
        easy_weight = 0.7 * (1 - progress) + 0.33 * progress
        tier_weights = {
            1: easy_weight,
            2: (1 - easy_weight) * 0.6,
            3: (1 - easy_weight) * 0.4
        }

        weights = torch.zeros_like(tiers, dtype=torch.float)
        for tier, w in tier_weights.items():
            mask = tiers == tier
            n_tier = mask.sum()
            if n_tier > 0:
                weights[mask] = w / n_tier.float()

        return weights / weights.sum()


def compute_batch_difficulty_metrics(y_batch: torch.Tensor) -> Dict[str, float]:
    """
    Compute difficulty metrics for logging/monitoring.

    Args:
        y_batch: Target values [batch_size]

    Returns:
        Dictionary of metrics
    """
    y = y_batch.view(-1)

    tiers = assign_tiers(y)

    return {
        'batch_range': (y.max() - y.min()).item(),
        'batch_std': y.std().item(),
        'batch_mean_pairwise_diff': (y.unsqueeze(0) - y.unsqueeze(1)).abs().mean().item(),
        'tier1_fraction': (tiers == 1).float().mean().item(),
        'tier2_fraction': (tiers == 2).float().mean().item(),
        'tier3_fraction': (tiers == 3).float().mean().item(),
    }


class QuantileResolutionCurriculum:
    """
    Curriculum that progressively increases quantile resolution during training.

    Early training: Coarse quantiles (e.g., Q=3) for easy ranking distinctions
    Late training: Fine quantiles (e.g., Q=20) for discriminating subtle differences

    This helps the model first learn coarse structure, then refine.
    """

    def __init__(self, q_schedule: List[Tuple[int, int]] = None,
                 total_epochs: int = 80):
        """
        Args:
            q_schedule: List of (epoch, n_quantiles) pairs defining schedule
                        Default: [(0, 3), (10, 5), (20, 10), (30, 20)]
            total_epochs: Total training epochs
        """
        if q_schedule is None:
            q_schedule = [
                (0, 3),    # Epochs 0-9: 3 quantiles (coarse)
                (10, 5),   # Epochs 10-19: 5 quantiles
                (20, 10),  # Epochs 20-29: 10 quantiles
                (30, 20)   # Epochs 30+: 20 quantiles (fine)
            ]
        self.q_schedule = sorted(q_schedule, key=lambda x: x[0])
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch

    def get_n_quantiles(self) -> int:
        """Get current number of quantiles based on schedule."""
        n_quantiles = self.q_schedule[0][1]  # Default to first
        for epoch_threshold, q in self.q_schedule:
            if self.current_epoch >= epoch_threshold:
                n_quantiles = q
        return n_quantiles

    def get_quantile_edges(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile edges for current resolution.

        Args:
            y: Activity values [n_samples]

        Returns:
            Quantile edges [n_quantiles + 1]
        """
        y = y.view(-1)
        n_q = self.get_n_quantiles()
        percentiles = torch.linspace(0, 100, n_q + 1, device=y.device)
        edges = torch.tensor([
            torch.quantile(y, p / 100.0) for p in percentiles
        ], device=y.device)
        return edges

    def assign_quantile_labels(self, y: torch.Tensor) -> torch.Tensor:
        """
        Assign samples to quantile bins.

        Args:
            y: Activity values [n_samples]

        Returns:
            Quantile labels [n_samples], values in {0, ..., n_quantiles-1}
        """
        y = y.view(-1)
        edges = self.get_quantile_edges(y)
        n_q = self.get_n_quantiles()

        labels = torch.zeros_like(y, dtype=torch.long)
        for q in range(n_q - 1, -1, -1):
            labels[y >= edges[q]] = q

        return labels

    def get_batch_weights_by_quantile(self, y_batch: torch.Tensor) -> torch.Tensor:
        """
        Get uniform weights across current quantiles.

        Ensures balanced representation of all activity levels.

        Args:
            y_batch: Target values [batch_size]

        Returns:
            Sample weights [batch_size]
        """
        labels = self.assign_quantile_labels(y_batch)
        n_q = self.get_n_quantiles()

        weights = torch.zeros_like(y_batch, dtype=torch.float)

        for q in range(n_q):
            mask = labels == q
            n_in_q = mask.sum()
            if n_in_q > 0:
                # Equal weight per quantile, divided among samples
                weights[mask] = 1.0 / (n_q * n_in_q.float())

        # Normalize
        return weights / weights.sum()

    def get_status(self) -> Dict[str, any]:
        """Get current curriculum status."""
        return {
            'current_epoch': self.current_epoch,
            'n_quantiles': self.get_n_quantiles(),
            'schedule': self.q_schedule
        }


class NoiseCurriculum:
    """
    Curriculum that transitions from low-noise to all samples.

    Early training: Focus on clean (low aleatoric uncertainty) samples
    Late training: Include all samples uniformly
    """

    def __init__(self, noise_percentile_start: float = 0.3,
                 noise_percentile_end: float = 1.0,
                 warmup_epochs: int = 10, total_epochs: int = 80):
        """
        Args:
            noise_percentile_start: Initial noise percentile to include (0.3 = bottom 30%)
            noise_percentile_end: Final noise percentile to include (1.0 = all)
            warmup_epochs: Epochs before starting to expand noise range
            total_epochs: Total training epochs
        """
        self.noise_percentile_start = noise_percentile_start
        self.noise_percentile_end = noise_percentile_end
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch

    def get_noise_threshold_percentile(self) -> float:
        """Get current noise threshold percentile."""
        if self.current_epoch < self.warmup_epochs:
            return self.noise_percentile_start

        progress = (self.current_epoch - self.warmup_epochs) / \
                   max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)

        return self.noise_percentile_start + \
               progress * (self.noise_percentile_end - self.noise_percentile_start)

    def get_sample_weights(self, aleatoric_uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Compute sample weights based on noise levels and curriculum.

        Args:
            aleatoric_uncertainty: Noise values [n_samples]

        Returns:
            Sample weights [n_samples]
        """
        noise = aleatoric_uncertainty.view(-1)
        percentile = self.get_noise_threshold_percentile()

        # Compute threshold
        threshold = torch.quantile(noise, percentile)

        # Samples below threshold get weight 1, above get decaying weight
        weights = torch.ones_like(noise)
        above_threshold = noise > threshold

        if above_threshold.any():
            # Exponential decay for samples above threshold
            excess_noise = noise[above_threshold] - threshold
            max_excess = (noise.max() - threshold).clamp(min=1e-8)
            decay = torch.exp(-3.0 * excess_noise / max_excess)
            weights[above_threshold] = decay

        return weights / weights.sum()

    def get_status(self) -> Dict[str, float]:
        """Get current curriculum status."""
        return {
            'current_epoch': self.current_epoch,
            'noise_percentile': self.get_noise_threshold_percentile()
        }
