"""
RankNet Pairwise Ranking Loss (Bradley-Terry Model)

For each pair of items (i, j) where y_i > y_j, the model should predict
P(s_i > s_j) ≈ 1.

Bradley-Terry model: P(i beats j) = σ(s_i - s_j)

Reference: https://www.emergentmind.com/topics/bradley-terry-ranking-system
"""

import torch
import torch.nn.functional as F


def ranknet_loss(scores: torch.Tensor, relevance: torch.Tensor,
                 sigma: float = 1.0) -> torch.Tensor:
    """
    Standard RankNet pairwise ranking loss.

    For all pairs where relevance_i != relevance_j, compute BCE loss
    on the probability that the model ranks them correctly.

    Args:
        scores: Model predictions [batch_size] or [batch_size, 1]
        relevance: Ground truth activity values [batch_size] or [batch_size, 1]
        sigma: Scaling factor for score differences

    Returns:
        Pairwise ranking loss (scalar)
    """
    # Flatten to 1D
    scores = scores.view(-1)
    relevance = relevance.view(-1)
    n = scores.shape[0]

    if n < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # Compute all pairwise differences
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # [n, n]
    relevance_diff = relevance.unsqueeze(1) - relevance.unsqueeze(0)  # [n, n]

    # Target: 1 if relevance_i > relevance_j, 0 if relevance_i < relevance_j
    # Only consider pairs where relevance differs
    positive_pairs = relevance_diff > 0
    negative_pairs = relevance_diff < 0
    valid_pairs = positive_pairs | negative_pairs

    if not valid_pairs.any():
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # Targets: 1 for positive pairs, 0 for negative pairs
    targets = positive_pairs.float()

    # Binary cross-entropy on pairwise comparisons
    loss = F.binary_cross_entropy_with_logits(
        sigma * score_diff[valid_pairs],
        targets[valid_pairs],
        reduction='mean'
    )

    return loss


def margin_ranknet_loss(scores: torch.Tensor, relevance: torch.Tensor,
                        base_margin: float = 0.1, sigma: float = 1.0) -> torch.Tensor:
    """
    Margin-aware RankNet loss.

    Larger activity differences should produce larger score gaps.
    Uses hinge loss with adaptive margin based on relevance difference.

    Args:
        scores: Model predictions [batch_size]
        relevance: Ground truth activity values [batch_size]
        base_margin: Base margin multiplied by relevance difference
        sigma: Scaling factor

    Returns:
        Margin-aware ranking loss (scalar)
    """
    scores = scores.view(-1)
    relevance = relevance.view(-1)
    n = scores.shape[0]

    if n < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    relevance_diff = relevance.unsqueeze(1) - relevance.unsqueeze(0)

    # Only consider pairs where i should be ranked higher than j
    positive_pairs = relevance_diff > 0

    if not positive_pairs.any():
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # Adaptive margin based on relevance gap
    margin = base_margin * relevance_diff.abs()

    # Hinge loss: penalize if score_diff < margin when relevance_i > relevance_j
    # We want score_i - score_j >= margin
    hinge_loss = F.relu(margin - score_diff)

    return hinge_loss[positive_pairs].mean()


def lambda_ranknet_loss(scores: torch.Tensor, relevance: torch.Tensor,
                        sigma: float = 1.0, ndcg_weight: bool = True) -> torch.Tensor:
    """
    LambdaRank-style loss with NDCG weighting.

    Weights pairwise gradients by the change in NDCG that would result
    from swapping the pair.

    Args:
        scores: Model predictions [batch_size]
        relevance: Ground truth activity values [batch_size]
        sigma: Scaling factor
        ndcg_weight: Whether to apply NDCG-based weighting

    Returns:
        Lambda-weighted ranking loss (scalar)
    """
    scores = scores.view(-1)
    relevance = relevance.view(-1)
    n = scores.shape[0]

    if n < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # Get current ranking by scores
    score_ranks = scores.argsort(descending=True).argsort() + 1  # 1-indexed ranks

    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    relevance_diff = relevance.unsqueeze(1) - relevance.unsqueeze(0)

    positive_pairs = relevance_diff > 0

    if not positive_pairs.any():
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    if ndcg_weight:
        # Compute NDCG delta for swapping each pair
        # |1/log2(rank_i+1) - 1/log2(rank_j+1)| * |2^rel_i - 2^rel_j|
        rank_i = score_ranks.unsqueeze(1).float()
        rank_j = score_ranks.unsqueeze(0).float()

        dcg_diff = (1.0 / torch.log2(rank_i + 1) - 1.0 / torch.log2(rank_j + 1)).abs()
        gain_diff = (torch.pow(2.0, relevance.unsqueeze(1)) - torch.pow(2.0, relevance.unsqueeze(0))).abs()

        # Clamp to avoid extreme weights
        lambda_weights = (dcg_diff * gain_diff).clamp(min=1e-6, max=10.0)
    else:
        lambda_weights = torch.ones_like(score_diff)

    # Weighted BCE loss
    targets = positive_pairs.float()
    valid_pairs = positive_pairs | (relevance_diff < 0)

    loss = F.binary_cross_entropy_with_logits(
        sigma * score_diff[valid_pairs],
        targets[valid_pairs],
        weight=lambda_weights[valid_pairs],
        reduction='mean'
    )

    return loss


def sampled_ranknet_loss(scores: torch.Tensor, relevance: torch.Tensor,
                         n_pairs: int = None, sigma: float = 1.0) -> torch.Tensor:
    """
    RankNet with sampled pairs for efficiency on large batches.

    Instead of computing all O(n²) pairs, sample a subset.

    Args:
        scores: Model predictions [batch_size]
        relevance: Ground truth activity values [batch_size]
        n_pairs: Number of pairs to sample (default: 2*n)
        sigma: Scaling factor

    Returns:
        Sampled pairwise ranking loss (scalar)
    """
    scores = scores.view(-1)
    relevance = relevance.view(-1)
    n = scores.shape[0]

    if n < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    if n_pairs is None:
        n_pairs = min(2 * n, n * (n - 1) // 2)

    # Sample random pairs
    idx_i = torch.randint(0, n, (n_pairs,), device=scores.device)
    idx_j = torch.randint(0, n, (n_pairs,), device=scores.device)

    # Ensure i != j
    same_mask = idx_i == idx_j
    idx_j[same_mask] = (idx_j[same_mask] + 1) % n

    score_diff = scores[idx_i] - scores[idx_j]
    relevance_diff = relevance[idx_i] - relevance[idx_j]

    # Only use pairs where relevance differs
    valid = relevance_diff != 0
    if not valid.any():
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    targets = (relevance_diff[valid] > 0).float()

    loss = F.binary_cross_entropy_with_logits(
        sigma * score_diff[valid],
        targets,
        reduction='mean'
    )

    return loss
