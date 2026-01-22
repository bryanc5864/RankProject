"""
Plackett-Luce Ranking Loss

Listwise ranking loss that models the probability of observing a particular
ranking as a product of sequential "choice" probabilities.

Reference: https://gist.github.com/crowsonkb/7df88ec63ea19ac335aa8b6c8f530769
"""

import torch
import torch.nn.functional as F


def plackett_luce_loss(scores: torch.Tensor, relevance: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
    """
    Compute Plackett-Luce ranking loss.

    Given scores s₁, s₂, ..., sₙ and ground-truth ranking π:
    P(π) = ∏ᵢ exp(s_π(i)) / Σⱼ≥ᵢ exp(s_π(j))

    Args:
        scores: Model predictions [batch_size, list_size] or [list_size]
        relevance: Ground truth activity values [batch_size, list_size] or [list_size]
        temperature: Temperature scaling (lower = sharper distinctions)

    Returns:
        Negative log-likelihood loss (scalar)
    """
    # Handle both batched and unbatched inputs
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)

    # Apply temperature scaling
    scores = scores / temperature

    # Sort by ground truth relevance (descending) to get target ranking
    sorted_indices = relevance.argsort(dim=-1, descending=True)
    sorted_scores = scores.gather(-1, sorted_indices)

    # Compute log-likelihood of correct ranking
    # For each position i, compute log P(selecting item at position i from remaining items)
    # log P = s_i - log(sum_{j>=i} exp(s_j))
    # The cumsum from the right gives us the log-sum-exp of remaining items
    cumsums = sorted_scores.flip(-1).logcumsumexp(-1).flip(-1)
    log_likelihood = (sorted_scores - cumsums).sum(-1)

    return -log_likelihood.mean()


def plackett_luce_loss_with_ties(scores: torch.Tensor, relevance: torch.Tensor,
                                  temperature: float = 1.0,
                                  tie_threshold: float = 0.01) -> torch.Tensor:
    """
    Plackett-Luce loss that handles ties in relevance values.

    When relevance values are within tie_threshold, they are considered
    equivalent and contribute equally to the loss.

    Args:
        scores: Model predictions [batch_size, list_size]
        relevance: Ground truth activity values [batch_size, list_size]
        temperature: Temperature scaling
        tie_threshold: Values within this threshold are considered tied

    Returns:
        Loss value (scalar)
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)

    scores = scores / temperature

    # Add small noise to break ties deterministically
    noise = torch.randn_like(relevance) * tie_threshold * 0.1
    relevance_noisy = relevance + noise

    sorted_indices = relevance_noisy.argsort(dim=-1, descending=True)
    sorted_scores = scores.gather(-1, sorted_indices)

    cumsums = sorted_scores.flip(-1).logcumsumexp(-1).flip(-1)
    log_likelihood = (sorted_scores - cumsums).sum(-1)

    return -log_likelihood.mean()


def weighted_plackett_luce_loss(scores: torch.Tensor, relevance: torch.Tensor,
                                 weights: torch.Tensor = None,
                                 temperature: float = 1.0) -> torch.Tensor:
    """
    Plackett-Luce loss with position-dependent weights.

    Allows emphasizing correct ranking of top items (e.g., NDCG-style weighting).

    Args:
        scores: Model predictions [batch_size, list_size]
        relevance: Ground truth activity values [batch_size, list_size]
        weights: Position weights [list_size], higher weight = more important
        temperature: Temperature scaling

    Returns:
        Weighted loss value (scalar)
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)

    scores = scores / temperature
    list_size = scores.shape[-1]

    # Default: DCG-style weights (1/log2(rank+1))
    if weights is None:
        positions = torch.arange(1, list_size + 1, device=scores.device, dtype=scores.dtype)
        weights = 1.0 / torch.log2(positions + 1)

    sorted_indices = relevance.argsort(dim=-1, descending=True)
    sorted_scores = scores.gather(-1, sorted_indices)

    cumsums = sorted_scores.flip(-1).logcumsumexp(-1).flip(-1)
    position_log_likelihoods = sorted_scores - cumsums

    # Apply position weights
    weighted_ll = (position_log_likelihoods * weights).sum(-1)

    return -weighted_ll.mean()
