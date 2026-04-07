"""
Rank-Stability Weighted RankNet Loss

Weights pairwise comparisons by noise-based reliability.
High-noise pairs (high aleatoric uncertainty) get low weight,
allowing clean pairs to dominate learning.

Key insight: For pair (i,j), the reliability of the comparison
depends on the combined noise of both samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rank_stability_weight(sigma_sq_i: torch.Tensor, sigma_sq_j: torch.Tensor,
                          k: float = 1.0) -> torch.Tensor:
    """
    Compute reliability weight for a pair based on their noise levels.

    weight_ij = sigmoid(-k * (σ²_i + σ²_j))

    High combined noise → low weight (unreliable comparison)
    Low combined noise → high weight (reliable comparison)

    Args:
        sigma_sq_i: Aleatoric variance for sample i
        sigma_sq_j: Aleatoric variance for sample j
        k: Scaling factor (higher = more aggressive downweighting)

    Returns:
        Reliability weight in [0, 1]
    """
    combined_noise = sigma_sq_i + sigma_sq_j
    return torch.sigmoid(-k * combined_noise)


class RankStabilityRankNet(nn.Module):
    """
    RankNet loss weighted by rank stability.

    Pairs with high combined aleatoric uncertainty are downweighted,
    focusing learning on reliable comparisons.
    """

    def __init__(self, sigma: float = 1.0, k: float = 1.0,
                 min_weight: float = 0.01):
        """
        Args:
            sigma: Scaling factor for score differences in BCE
            k: Scaling factor for noise-based weight (higher = more aggressive)
            min_weight: Minimum weight to avoid completely ignoring pairs
        """
        super().__init__()
        self.sigma = sigma
        self.k = k
        self.min_weight = min_weight

    def forward(self, scores: torch.Tensor, relevance: torch.Tensor,
                aleatoric_uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Compute rank-stability weighted RankNet loss.

        Args:
            scores: Model predictions [batch_size] or [batch_size, 1]
            relevance: Ground truth activity values [batch_size]
            aleatoric_uncertainty: Noise proxy (replicate variance) [batch_size]

        Returns:
            Weighted pairwise ranking loss (scalar)
        """
        # Flatten to 1D
        scores = scores.view(-1)
        relevance = relevance.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)
        n = scores.shape[0]

        if n < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Compute all pairwise differences
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # [n, n]
        relevance_diff = relevance.unsqueeze(1) - relevance.unsqueeze(0)  # [n, n]

        # Target: 1 if relevance_i > relevance_j, 0 otherwise
        positive_pairs = relevance_diff > 0
        negative_pairs = relevance_diff < 0
        valid_pairs = positive_pairs | negative_pairs

        if not valid_pairs.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Compute noise-based weights for all pairs
        # σ²_i + σ²_j for each pair
        sigma_sq = aleatoric_uncertainty ** 2
        combined_noise = sigma_sq.unsqueeze(1) + sigma_sq.unsqueeze(0)  # [n, n]

        # weight_ij = sigmoid(-k * combined_noise)
        weights = torch.sigmoid(-self.k * combined_noise)
        weights = torch.clamp(weights, min=self.min_weight)  # Floor to avoid zeros

        # Targets: 1 for positive pairs, 0 for negative pairs
        targets = positive_pairs.float()

        # Extract valid pairs
        valid_score_diff = score_diff[valid_pairs]
        valid_targets = targets[valid_pairs]
        valid_weights = weights[valid_pairs]

        # Normalize weights to sum to number of pairs (preserves loss scale)
        valid_weights = valid_weights / valid_weights.mean()

        # Weighted BCE loss
        loss = F.binary_cross_entropy_with_logits(
            self.sigma * valid_score_diff,
            valid_targets,
            weight=valid_weights,
            reduction='mean'
        )

        return loss

    def get_pair_weights(self, aleatoric_uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Compute weight matrix for visualization/analysis.

        Returns:
            Weight matrix [n, n] for all pairs
        """
        sigma_sq = aleatoric_uncertainty.view(-1) ** 2
        combined_noise = sigma_sq.unsqueeze(1) + sigma_sq.unsqueeze(0)
        weights = torch.sigmoid(-self.k * combined_noise)
        return torch.clamp(weights, min=self.min_weight)


class SampledRankStabilityRankNet(nn.Module):
    """
    Efficient sampled version for large batches.

    Instead of O(n²) pairs, samples a fixed number weighted by reliability.
    """

    def __init__(self, sigma: float = 1.0, k: float = 1.0,
                 n_pairs: int = None, importance_sample: bool = True):
        """
        Args:
            sigma: Scaling factor for BCE
            k: Noise scaling factor
            n_pairs: Number of pairs to sample (default: 2*n)
            importance_sample: If True, sample proportional to weights
        """
        super().__init__()
        self.sigma = sigma
        self.k = k
        self.n_pairs = n_pairs
        self.importance_sample = importance_sample

    def forward(self, scores: torch.Tensor, relevance: torch.Tensor,
                aleatoric_uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Compute sampled rank-stability weighted loss.
        """
        scores = scores.view(-1)
        relevance = relevance.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)
        n = scores.shape[0]

        if n < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        n_pairs = self.n_pairs if self.n_pairs else min(2 * n, n * (n - 1) // 2)

        # Sample random pairs
        idx_i = torch.randint(0, n, (n_pairs,), device=scores.device)
        idx_j = torch.randint(0, n, (n_pairs,), device=scores.device)

        # Ensure i != j
        same_mask = idx_i == idx_j
        idx_j[same_mask] = (idx_j[same_mask] + 1) % n

        score_diff = scores[idx_i] - scores[idx_j]
        relevance_diff = relevance[idx_i] - relevance[idx_j]

        # Compute weights for sampled pairs
        sigma_sq_i = aleatoric_uncertainty[idx_i] ** 2
        sigma_sq_j = aleatoric_uncertainty[idx_j] ** 2
        weights = torch.sigmoid(-self.k * (sigma_sq_i + sigma_sq_j))

        # Only use pairs where relevance differs
        valid = relevance_diff != 0
        if not valid.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        targets = (relevance_diff[valid] > 0).float()
        valid_weights = weights[valid]
        valid_weights = valid_weights / valid_weights.mean()  # Normalize

        loss = F.binary_cross_entropy_with_logits(
            self.sigma * score_diff[valid],
            targets,
            weight=valid_weights,
            reduction='mean'
        )

        return loss


def rank_stability_ranknet_loss(scores: torch.Tensor, relevance: torch.Tensor,
                                 aleatoric_uncertainty: torch.Tensor,
                                 sigma: float = 1.0, k: float = 1.0) -> torch.Tensor:
    """
    Functional interface for rank-stability weighted RankNet loss.

    Args:
        scores: Model predictions [batch_size]
        relevance: Ground truth activity values [batch_size]
        aleatoric_uncertainty: Noise proxy [batch_size]
        sigma: BCE scaling factor
        k: Noise weight scaling factor

    Returns:
        Weighted pairwise ranking loss
    """
    loss_fn = RankStabilityRankNet(sigma=sigma, k=k)
    return loss_fn(scores, relevance, aleatoric_uncertainty)
