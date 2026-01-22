"""
SoftSort Differentiable Sorting Loss

Approximates the sorting/ranking operation as a smooth, differentiable
function so you can backpropagate through the ranks.

Uses torchsort library: https://github.com/teddykoker/torchsort
"""

import torch
import torch.nn.functional as F

try:
    import torchsort
    TORCHSORT_AVAILABLE = True
except ImportError:
    TORCHSORT_AVAILABLE = False


def softsort_loss(scores: torch.Tensor, relevance: torch.Tensor,
                  regularization: float = 1.0) -> torch.Tensor:
    """
    SoftSort loss using differentiable soft ranks.

    Directly optimizes predicted ranks to match ground truth ranks.
    Uses torchsort if available, otherwise falls back to pure PyTorch implementation.

    Args:
        scores: Model predictions [batch_size] or [batch_size, list_size]
        relevance: Ground truth activity values, same shape as scores
        regularization: Regularization strength for soft ranking
                       (lower = sharper/more accurate, higher = smoother gradients)

    Returns:
        MSE loss between predicted and ground truth soft ranks
    """
    # Ensure 2D
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)

    if TORCHSORT_AVAILABLE:
        # Use torchsort if available
        pred_ranks = torchsort.soft_rank(scores, regularization_strength=regularization)
        true_ranks = torchsort.soft_rank(relevance, regularization_strength=regularization)
    else:
        # Pure PyTorch fallback using sigmoid-based soft ranking
        pred_ranks = _soft_rank_pytorch(scores, temperature=regularization)
        true_ranks = _soft_rank_pytorch(relevance, temperature=regularization)

    # MSE between rank positions
    return F.mse_loss(pred_ranks, true_ranks)


def _soft_rank_pytorch(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Pure PyTorch implementation of differentiable soft ranking.

    For each element, rank = 1 + sum of sigmoids of pairwise differences.
    Higher values get lower ranks (rank 1 = highest).

    Args:
        x: Input tensor [batch_size, n] or [n]
        temperature: Controls sharpness (lower = sharper ranking)

    Returns:
        Soft ranks [batch_size, n] or [n]
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, n = x.shape

    # Pairwise differences: x_j - x_i for all pairs
    # Shape: [batch_size, n, n]
    x_diff = x.unsqueeze(2) - x.unsqueeze(1)

    # Soft comparison: probability that x_j > x_i
    # Using sigmoid: P(x_j > x_i) = sigmoid((x_j - x_i) / temperature)
    soft_comparisons = torch.sigmoid(x_diff / temperature)

    # Rank_i = 1 + sum_j P(x_j > x_i) = 1 + sum_j sigmoid((x_j - x_i) / temp)
    # This gives rank 1 to the highest value
    ranks = 1.0 + soft_comparisons.sum(dim=2)

    if squeeze_output:
        ranks = ranks.squeeze(0)

    return ranks


def softsort_spearman_loss(scores: torch.Tensor, relevance: torch.Tensor,
                           regularization: float = 1.0) -> torch.Tensor:
    """
    Differentiable Spearman correlation loss using soft ranks.

    Maximizes Spearman correlation by minimizing 1 - ρ.

    Args:
        scores: Model predictions [batch_size] or [batch_size, list_size]
        relevance: Ground truth activity values
        regularization: Regularization strength for soft ranking

    Returns:
        1 - Spearman correlation (so minimizing this maximizes correlation)
    """
    if not TORCHSORT_AVAILABLE:
        raise ImportError("torchsort is required. Install with: pip install torchsort")

    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)

    pred_ranks = torchsort.soft_rank(scores, regularization_strength=regularization)
    true_ranks = torchsort.soft_rank(relevance, regularization_strength=regularization)

    # Compute Spearman correlation
    # ρ = 1 - (6 * Σd²) / (n * (n² - 1))
    # But we'll use the Pearson correlation of ranks for numerical stability

    pred_centered = pred_ranks - pred_ranks.mean(dim=-1, keepdim=True)
    true_centered = true_ranks - true_ranks.mean(dim=-1, keepdim=True)

    pred_norm = pred_centered / (pred_centered.std(dim=-1, keepdim=True) + 1e-8)
    true_norm = true_centered / (true_centered.std(dim=-1, keepdim=True) + 1e-8)

    correlation = (pred_norm * true_norm).mean(dim=-1)

    # Return 1 - correlation so we minimize
    return (1 - correlation).mean()


def soft_ndcg_loss(scores: torch.Tensor, relevance: torch.Tensor,
                   regularization: float = 1.0, k: int = None) -> torch.Tensor:
    """
    Differentiable NDCG loss using soft sorting.

    Approximates NDCG using soft permutation matrices.

    Args:
        scores: Model predictions [batch_size, list_size]
        relevance: Ground truth activity values [batch_size, list_size]
        regularization: Regularization strength
        k: Cutoff for NDCG@k (None = use full list)

    Returns:
        1 - NDCG (so minimizing this maximizes NDCG)
    """
    if not TORCHSORT_AVAILABLE:
        raise ImportError("torchsort is required. Install with: pip install torchsort")

    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)

    batch_size, list_size = scores.shape

    if k is None:
        k = list_size
    k = min(k, list_size)

    # Get soft ranks (1 = highest score)
    pred_ranks = torchsort.soft_rank(-scores, regularization_strength=regularization)

    # Compute DCG weights: 1/log2(rank+1)
    # For soft ranks, we use the soft rank values directly
    discount = 1.0 / torch.log2(pred_ranks + 1)

    # Gain: 2^relevance - 1 (standard NDCG gain)
    gain = torch.pow(2.0, relevance) - 1

    # Soft DCG: sum of gain * discount
    # Apply cutoff by weighting positions
    position_weight = (pred_ranks <= k).float()
    dcg = (gain * discount * position_weight).sum(dim=-1)

    # Ideal DCG: sort by relevance
    ideal_ranks = torchsort.soft_rank(-relevance, regularization_strength=regularization)
    ideal_discount = 1.0 / torch.log2(ideal_ranks + 1)
    ideal_position_weight = (ideal_ranks <= k).float()
    idcg = (gain * ideal_discount * ideal_position_weight).sum(dim=-1)

    # NDCG = DCG / IDCG
    ndcg = dcg / (idcg + 1e-8)

    return (1 - ndcg).mean()


# Fallback implementations without torchsort

def differentiable_rank_mse(scores: torch.Tensor, relevance: torch.Tensor,
                            temperature: float = 1.0) -> torch.Tensor:
    """
    Approximate rank MSE without torchsort using softmax-based ranking.

    Uses attention-style soft assignment to approximate ranks.

    Args:
        scores: Model predictions [batch_size] or [batch_size, list_size]
        relevance: Ground truth activity values
        temperature: Temperature for softmax (lower = sharper)

    Returns:
        Approximate rank MSE loss
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)

    batch_size, n = scores.shape

    # Compute pairwise comparisons for soft ranking
    # rank_i ≈ sum_j sigmoid((s_j - s_i) / temp) + 1
    score_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)  # [B, n, n]
    soft_comparisons = torch.sigmoid(score_diff / temperature)
    pred_ranks = soft_comparisons.sum(dim=-1) + 1  # [B, n]

    relevance_diff = relevance.unsqueeze(-1) - relevance.unsqueeze(-2)
    soft_comparisons_true = torch.sigmoid(relevance_diff / temperature)
    true_ranks = soft_comparisons_true.sum(dim=-1) + 1

    return F.mse_loss(pred_ranks, true_ranks)
