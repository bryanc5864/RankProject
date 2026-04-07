"""
Quantile-Stratified Sampling and Hard Negative Mining

Activity-stratified batch construction to ensure:
1. Balanced representation across activity levels
2. Informative pairwise comparisons
3. Noise-aware sampling within strata
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Optional, Dict, List, Tuple


class QuantileStratifiedSampler(Sampler):
    """
    Sampler that ensures uniform representation across activity quantiles.

    Each batch contains approximately equal samples from each quantile,
    with optional noise-based weighting within each stratum.
    """

    def __init__(self, targets: torch.Tensor, n_quantiles: int = 10,
                 num_samples: int = None, noise: Optional[torch.Tensor] = None,
                 noise_weight: float = 0.0):
        """
        Args:
            targets: Activity values [n_samples]
            n_quantiles: Number of quantiles/strata
            num_samples: Samples per epoch (default: len(targets))
            noise: Aleatoric uncertainty [n_samples], for noise-based weighting
            noise_weight: Weight for noise-based sampling (0=ignore, 1=full inverse)
        """
        self.targets = targets.view(-1).cpu().numpy() if isinstance(targets, torch.Tensor) else targets
        self.n_quantiles = n_quantiles
        self.num_samples = num_samples or len(self.targets)
        self.noise_weight = noise_weight

        if noise is not None:
            self.noise = noise.view(-1).cpu().numpy() if isinstance(noise, torch.Tensor) else noise
        else:
            self.noise = None

        # Compute quantile assignments
        self.quantile_edges = np.percentile(
            self.targets,
            np.linspace(0, 100, n_quantiles + 1)
        )
        self.quantile_assignments = np.digitize(self.targets, self.quantile_edges[1:-1])

        # Pre-compute indices for each quantile
        self.quantile_indices = {}
        for q in range(n_quantiles):
            self.quantile_indices[q] = np.where(self.quantile_assignments == q)[0]

        # Compute within-quantile weights if noise provided
        self.within_quantile_weights = {}
        if self.noise is not None and self.noise_weight > 0:
            for q in range(n_quantiles):
                q_indices = self.quantile_indices[q]
                if len(q_indices) == 0:
                    continue
                q_noise = self.noise[q_indices]
                # Inverse noise weighting: cleaner samples get higher weight
                inv_noise = 1.0 / (1.0 + q_noise)
                # Blend with uniform: (1-w)*uniform + w*inv_noise
                uniform = np.ones_like(inv_noise) / len(inv_noise)
                inv_noise = inv_noise / inv_noise.sum()
                self.within_quantile_weights[q] = (1 - noise_weight) * uniform + noise_weight * inv_noise
        else:
            for q in range(n_quantiles):
                if len(self.quantile_indices[q]) > 0:
                    self.within_quantile_weights[q] = np.ones(len(self.quantile_indices[q])) / len(self.quantile_indices[q])

    def __iter__(self):
        # Sample equal number from each quantile
        samples_per_quantile = self.num_samples // self.n_quantiles

        all_indices = []

        for q in range(self.n_quantiles):
            q_indices = self.quantile_indices[q]
            if len(q_indices) == 0:
                continue

            weights = self.within_quantile_weights.get(q)
            n_to_sample = samples_per_quantile

            # Handle last quantile: add remainder
            if q == self.n_quantiles - 1:
                n_to_sample += self.num_samples - samples_per_quantile * self.n_quantiles

            sampled = np.random.choice(
                q_indices,
                size=min(n_to_sample, len(q_indices) * 2),  # Allow replacement
                replace=True,
                p=weights
            )
            all_indices.extend(sampled.tolist())

        # Shuffle the combined samples
        np.random.shuffle(all_indices)

        return iter(all_indices[:self.num_samples])

    def __len__(self):
        return self.num_samples

    def get_quantile_statistics(self) -> Dict[str, any]:
        """Get statistics about quantile distribution."""
        stats = {
            'n_quantiles': self.n_quantiles,
            'quantile_edges': self.quantile_edges.tolist(),
            'samples_per_quantile': {
                q: len(self.quantile_indices[q]) for q in range(self.n_quantiles)
            }
        }
        return stats


class QuantileCurriculum:
    """
    Curriculum that progressively increases quantile resolution.

    Early training: coarse quantiles (easy ranking)
    Late training: fine quantiles (discriminating subtle differences)
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
        self._cached_sampler = None
        self._cached_n_quantiles = None

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

    def get_sampler(self, targets: torch.Tensor, num_samples: int = None,
                    noise: Optional[torch.Tensor] = None,
                    noise_weight: float = 0.0) -> QuantileStratifiedSampler:
        """
        Get sampler for current epoch.

        Returns cached sampler if n_quantiles hasn't changed.
        """
        n_quantiles = self.get_n_quantiles()

        if self._cached_n_quantiles != n_quantiles or self._cached_sampler is None:
            self._cached_sampler = QuantileStratifiedSampler(
                targets=targets,
                n_quantiles=n_quantiles,
                num_samples=num_samples,
                noise=noise,
                noise_weight=noise_weight
            )
            self._cached_n_quantiles = n_quantiles

        return self._cached_sampler

    def get_status(self) -> Dict[str, any]:
        """Get current curriculum status."""
        return {
            'current_epoch': self.current_epoch,
            'n_quantiles': self.get_n_quantiles(),
            'schedule': self.q_schedule
        }


class HardNegativeMiner:
    """
    Mines informative pairs for ranking:
    - Small activity difference (hard to distinguish)
    - Low noise (reliable comparison)

    These pairs provide the most learning signal.
    """

    def __init__(self, activity_threshold: float = 0.1,
                 noise_threshold: float = 0.5,
                 n_pairs_per_anchor: int = 5):
        """
        Args:
            activity_threshold: Max activity difference for "similar" pairs (quantile)
            noise_threshold: Max combined noise for "reliable" pairs (quantile)
            n_pairs_per_anchor: Number of pairs to mine per anchor sample
        """
        self.activity_threshold = activity_threshold
        self.noise_threshold = noise_threshold
        self.n_pairs_per_anchor = n_pairs_per_anchor

    def mine_pairs(self, targets: torch.Tensor,
                   noise: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Mine informative pairs from a batch.

        Args:
            targets: Activity values [batch_size]
            noise: Aleatoric uncertainty [batch_size]

        Returns:
            List of (i, j) index pairs
        """
        targets = targets.view(-1)
        noise = noise.view(-1)
        n = len(targets)

        if n < 2:
            return []

        # Compute pairwise differences
        activity_diff = (targets.unsqueeze(1) - targets.unsqueeze(0)).abs()
        combined_noise = noise.unsqueeze(1) + noise.unsqueeze(0)

        # Thresholds
        activity_thresh = torch.quantile(activity_diff[activity_diff > 0],
                                          self.activity_threshold)
        noise_thresh = torch.quantile(combined_noise, self.noise_threshold)

        # Find informative pairs
        informative = (activity_diff > 0) & \
                      (activity_diff < activity_thresh) & \
                      (combined_noise < noise_thresh)

        # Convert to list of pairs
        pairs = []
        for i in range(n):
            valid_j = informative[i].nonzero(as_tuple=True)[0]
            if len(valid_j) > 0:
                # Select top-k by informativeness (lowest combined noise)
                j_noise = combined_noise[i, valid_j]
                _, sorted_idx = j_noise.sort()
                top_k = min(self.n_pairs_per_anchor, len(valid_j))
                for k in range(top_k):
                    j = valid_j[sorted_idx[k]].item()
                    pairs.append((i, j))

        return pairs

    def get_pair_weights(self, targets: torch.Tensor,
                         noise: torch.Tensor) -> torch.Tensor:
        """
        Compute informativeness weights for all pairs.

        Args:
            targets: Activity values [batch_size]
            noise: Aleatoric uncertainty [batch_size]

        Returns:
            Weight matrix [batch_size, batch_size]
        """
        targets = targets.view(-1)
        noise = noise.view(-1)

        activity_diff = (targets.unsqueeze(1) - targets.unsqueeze(0)).abs()
        combined_noise = noise.unsqueeze(1) + noise.unsqueeze(0)

        # Informativeness: high when activity_diff is small but non-zero,
        # and combined_noise is low
        # weight = exp(-α * activity_diff) * exp(-β * combined_noise)
        activity_score = torch.exp(-5.0 * activity_diff)  # Small diff = high score
        noise_score = torch.exp(-2.0 * combined_noise)    # Low noise = high score

        # Zero out diagonal
        mask = ~torch.eye(len(targets), device=targets.device, dtype=torch.bool)
        weights = activity_score * noise_score * mask.float()

        return weights


class HardNegativeSampler(Sampler):
    """
    Sampler that prioritizes hard negative pairs.

    Samples indices such that batches contain informative pairs:
    samples with similar activity but low noise.
    """

    def __init__(self, targets: torch.Tensor, noise: torch.Tensor,
                 num_samples: int = None, temperature: float = 1.0):
        """
        Args:
            targets: Activity values [n_samples]
            noise: Aleatoric uncertainty [n_samples]
            num_samples: Samples per epoch
            temperature: Softmax temperature for sampling weights
        """
        self.targets = targets.view(-1).cpu().numpy()
        self.noise = noise.view(-1).cpu().numpy()
        self.num_samples = num_samples or len(self.targets)
        self.temperature = temperature

        # Pre-compute sample scores
        # Samples that are more likely to form informative pairs get higher scores
        self._compute_sample_scores()

    def _compute_sample_scores(self):
        """Compute per-sample informativeness scores."""
        n = len(self.targets)

        # Activity scores: samples near the median form harder pairs
        median = np.median(self.targets)
        distance_from_median = np.abs(self.targets - median)
        # Invert: closer to median = higher score
        activity_score = 1.0 / (1.0 + distance_from_median)

        # Noise scores: lower noise = higher score
        noise_score = 1.0 / (1.0 + self.noise)

        # Combined score
        self.sample_scores = activity_score * noise_score

        # Convert to probabilities
        self.sample_probs = np.exp(self.sample_scores / self.temperature)
        self.sample_probs = self.sample_probs / self.sample_probs.sum()

    def __iter__(self):
        indices = np.random.choice(
            len(self.targets),
            size=self.num_samples,
            replace=True,
            p=self.sample_probs
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


class AdaptiveQuantileSampler(Sampler):
    """
    Adaptive sampler that adjusts quantile weights based on validation performance.

    If model performs poorly on certain quantiles, increase their sampling rate.
    """

    def __init__(self, targets: torch.Tensor, n_quantiles: int = 10,
                 num_samples: int = None, adaptation_rate: float = 0.1):
        """
        Args:
            targets: Activity values [n_samples]
            n_quantiles: Number of quantiles
            num_samples: Samples per epoch
            adaptation_rate: How quickly to adapt weights (0=no adaptation, 1=full)
        """
        self.targets = targets.view(-1).cpu().numpy()
        self.n_quantiles = n_quantiles
        self.num_samples = num_samples or len(self.targets)
        self.adaptation_rate = adaptation_rate

        # Compute quantile assignments
        self.quantile_edges = np.percentile(
            self.targets,
            np.linspace(0, 100, n_quantiles + 1)
        )
        self.quantile_assignments = np.digitize(self.targets, self.quantile_edges[1:-1])

        # Pre-compute indices for each quantile
        self.quantile_indices = {}
        for q in range(n_quantiles):
            self.quantile_indices[q] = np.where(self.quantile_assignments == q)[0]

        # Initialize uniform weights
        self.quantile_weights = np.ones(n_quantiles) / n_quantiles

    def update_weights(self, quantile_performance: Dict[int, float]):
        """
        Update quantile weights based on performance.

        Args:
            quantile_performance: Dict mapping quantile -> Spearman correlation
        """
        # Lower performance = higher weight (sample more from weak quantiles)
        new_weights = np.zeros(self.n_quantiles)
        for q in range(self.n_quantiles):
            perf = quantile_performance.get(q, 1.0)
            new_weights[q] = 1.0 / (perf + 0.1)  # Inverse performance

        new_weights = new_weights / new_weights.sum()

        # Blend with current weights
        self.quantile_weights = (1 - self.adaptation_rate) * self.quantile_weights + \
                                 self.adaptation_rate * new_weights

    def __iter__(self):
        all_indices = []

        for q in range(self.n_quantiles):
            q_indices = self.quantile_indices[q]
            if len(q_indices) == 0:
                continue

            n_to_sample = int(self.num_samples * self.quantile_weights[q])
            sampled = np.random.choice(
                q_indices,
                size=n_to_sample,
                replace=True
            )
            all_indices.extend(sampled.tolist())

        np.random.shuffle(all_indices)
        return iter(all_indices[:self.num_samples])

    def __len__(self):
        return self.num_samples
