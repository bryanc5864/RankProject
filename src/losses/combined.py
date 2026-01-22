"""
Combined Loss Functions

Combine regression (MSE) with ranking losses for multi-objective optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any

from .plackett_luce import plackett_luce_loss
from .ranknet import ranknet_loss, margin_ranknet_loss
from .softsort import softsort_loss, TORCHSORT_AVAILABLE


def combined_loss(scores: torch.Tensor, targets: torch.Tensor,
                  alpha: float = 0.5, ranking_loss_fn: str = 'plackett_luce',
                  **ranking_kwargs) -> torch.Tensor:
    """
    Combined MSE + Ranking loss.

    loss = α * MSE + (1-α) * ranking_loss

    Args:
        scores: Model predictions [batch_size] or [batch_size, 1]
        targets: Ground truth values [batch_size] or [batch_size, 1]
        alpha: Weight for MSE loss (1-alpha for ranking loss)
        ranking_loss_fn: Which ranking loss to use:
            - 'plackett_luce': Listwise Plackett-Luce
            - 'ranknet': Pairwise RankNet
            - 'margin_ranknet': Margin-aware RankNet
            - 'softsort': Differentiable sorting (requires torchsort)
        **ranking_kwargs: Additional arguments for the ranking loss

    Returns:
        Combined loss value
    """
    scores_flat = scores.view(-1)
    targets_flat = targets.view(-1)

    # MSE loss
    mse = F.mse_loss(scores_flat, targets_flat)

    # Ranking loss
    if ranking_loss_fn == 'plackett_luce':
        rank_loss = plackett_luce_loss(scores_flat, targets_flat, **ranking_kwargs)
    elif ranking_loss_fn == 'ranknet':
        rank_loss = ranknet_loss(scores_flat, targets_flat, **ranking_kwargs)
    elif ranking_loss_fn == 'margin_ranknet':
        rank_loss = margin_ranknet_loss(scores_flat, targets_flat, **ranking_kwargs)
    elif ranking_loss_fn == 'softsort':
        if not TORCHSORT_AVAILABLE:
            raise ImportError("torchsort required for softsort loss")
        rank_loss = softsort_loss(scores_flat, targets_flat, **ranking_kwargs)
    else:
        raise ValueError(f"Unknown ranking loss: {ranking_loss_fn}")

    return alpha * mse + (1 - alpha) * rank_loss


class CombinedLoss(nn.Module):
    """
    Combined loss as a PyTorch module for easier integration.
    """

    def __init__(self, alpha: float = 0.5, ranking_loss_fn: str = 'plackett_luce',
                 **ranking_kwargs):
        """
        Args:
            alpha: Weight for MSE loss
            ranking_loss_fn: Which ranking loss to use
            **ranking_kwargs: Arguments passed to ranking loss
        """
        super().__init__()
        self.alpha = alpha
        self.ranking_loss_fn = ranking_loss_fn
        self.ranking_kwargs = ranking_kwargs

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return combined_loss(
            scores, targets,
            alpha=self.alpha,
            ranking_loss_fn=self.ranking_loss_fn,
            **self.ranking_kwargs
        )


class AdaptiveCombinedLoss(nn.Module):
    """
    Combined loss with adaptive weighting that changes over training.

    Can schedule alpha to shift from regression-focused to ranking-focused.
    """

    def __init__(self, alpha_start: float = 0.9, alpha_end: float = 0.3,
                 warmup_epochs: int = 10, total_epochs: int = 80,
                 ranking_loss_fn: str = 'plackett_luce', **ranking_kwargs):
        """
        Args:
            alpha_start: Initial MSE weight (regression-focused)
            alpha_end: Final MSE weight (more ranking-focused)
            warmup_epochs: Epochs to keep alpha_start before transitioning
            total_epochs: Total training epochs
            ranking_loss_fn: Which ranking loss to use
        """
        super().__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.ranking_loss_fn = ranking_loss_fn
        self.ranking_kwargs = ranking_kwargs

        self.current_epoch = 0

    def get_alpha(self) -> float:
        """Get current alpha value based on training progress."""
        if self.current_epoch < self.warmup_epochs:
            return self.alpha_start

        progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))

        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress

    def set_epoch(self, epoch: int):
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.get_alpha()
        return combined_loss(
            scores, targets,
            alpha=alpha,
            ranking_loss_fn=self.ranking_loss_fn,
            **self.ranking_kwargs
        )


class MultiTaskRankingLoss(nn.Module):
    """
    Multi-task loss for models with separate regression and ranking heads.

    Useful for dual-head architectures where one head predicts absolute values
    and another predicts relative rankings.
    """

    def __init__(self, regression_weight: float = 0.5, ranking_weight: float = 0.5,
                 ranking_loss_fn: str = 'plackett_luce', **ranking_kwargs):
        super().__init__()
        self.regression_weight = regression_weight
        self.ranking_weight = ranking_weight
        self.ranking_loss_fn = ranking_loss_fn
        self.ranking_kwargs = ranking_kwargs

    def forward(self, reg_pred: torch.Tensor, rank_pred: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            reg_pred: Regression head predictions
            rank_pred: Ranking head predictions
            targets: Ground truth values

        Returns:
            Dictionary with 'total', 'regression', and 'ranking' losses
        """
        reg_loss = F.mse_loss(reg_pred.view(-1), targets.view(-1))

        if self.ranking_loss_fn == 'plackett_luce':
            rank_loss = plackett_luce_loss(rank_pred.view(-1), targets.view(-1),
                                           **self.ranking_kwargs)
        elif self.ranking_loss_fn == 'ranknet':
            rank_loss = ranknet_loss(rank_pred.view(-1), targets.view(-1),
                                     **self.ranking_kwargs)
        else:
            raise ValueError(f"Unknown ranking loss: {self.ranking_loss_fn}")

        total_loss = self.regression_weight * reg_loss + self.ranking_weight * rank_loss

        return {
            'total': total_loss,
            'regression': reg_loss,
            'ranking': rank_loss
        }


class UncertaintyWeightedLoss(nn.Module):
    """
    Learns task weights automatically using uncertainty weighting.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    """

    def __init__(self, ranking_loss_fn: str = 'plackett_luce', **ranking_kwargs):
        super().__init__()
        self.ranking_loss_fn = ranking_loss_fn
        self.ranking_kwargs = ranking_kwargs

        # Learnable log variances for each task
        self.log_var_mse = nn.Parameter(torch.zeros(1))
        self.log_var_rank = nn.Parameter(torch.zeros(1))

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores_flat = scores.view(-1)
        targets_flat = targets.view(-1)

        mse_loss = F.mse_loss(scores_flat, targets_flat)

        if self.ranking_loss_fn == 'plackett_luce':
            rank_loss = plackett_luce_loss(scores_flat, targets_flat, **self.ranking_kwargs)
        elif self.ranking_loss_fn == 'ranknet':
            rank_loss = ranknet_loss(scores_flat, targets_flat, **self.ranking_kwargs)
        else:
            raise ValueError(f"Unknown ranking loss: {self.ranking_loss_fn}")

        # Uncertainty weighting: L = L_i / (2 * σ²) + log(σ)
        # Using log variance for numerical stability
        precision_mse = torch.exp(-self.log_var_mse)
        precision_rank = torch.exp(-self.log_var_rank)

        weighted_mse = precision_mse * mse_loss + self.log_var_mse
        weighted_rank = precision_rank * rank_loss + self.log_var_rank

        total_loss = weighted_mse + weighted_rank

        return {
            'total': total_loss,
            'mse': mse_loss,
            'ranking': rank_loss,
            'mse_weight': precision_mse.item(),
            'rank_weight': precision_rank.item()
        }
