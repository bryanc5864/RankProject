"""
Noise-Gated Ranking Loss

Combined loss that integrates:
1. Heteroscedastic NLL (noise-aware regression)
2. Rank-stability weighted RankNet (noise-aware ranking)
3. Variance supervision (explicit uncertainty guidance)

This represents the full noise-resistant training objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rank_stability import RankStabilityRankNet


class NoiseGatedRanking(nn.Module):
    """
    Combined noise-gated ranking loss.

    L = L_heteroscedastic + α * L_rank_stability + β * L_variance_supervision

    All three components are noise-aware:
    - Heteroscedastic: downweights noisy samples in regression
    - Rank stability: downweights noisy pairs in ranking
    - Variance supervision: guides uncertainty estimation
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1,
                 rank_k: float = 1.0, rank_sigma: float = 1.0,
                 min_log_var: float = -10.0, max_log_var: float = 10.0):
        """
        Args:
            alpha: Weight for rank stability loss
            beta: Weight for variance supervision loss
            rank_k: Noise scaling factor for rank stability
            rank_sigma: BCE scaling for RankNet
            min_log_var: Lower bound on log variance
            max_log_var: Upper bound on log variance
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.rank_stability = RankStabilityRankNet(sigma=rank_sigma, k=rank_k)

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                targets: torch.Tensor, aleatoric_uncertainty: torch.Tensor) -> dict:
        """
        Compute noise-gated ranking loss.

        Args:
            mu: Predicted mean [batch_size]
            log_var: Predicted log variance [batch_size]
            targets: Ground truth activity [batch_size]
            aleatoric_uncertainty: True noise proxy [batch_size]

        Returns:
            Dict with 'loss' and all component losses
        """
        mu = mu.view(-1)
        log_var = log_var.view(-1)
        targets = targets.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)

        # Clamp log_var for stability
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        pred_var = torch.exp(log_var)

        # 1. Heteroscedastic NLL
        residual_sq = (targets - mu) ** 2
        nll = 0.5 * (log_var + residual_sq / pred_var)
        hetero_loss = nll.mean()

        # 2. Rank stability weighted RankNet
        rank_loss = self.rank_stability(mu, targets, aleatoric_uncertainty)

        # 3. Variance supervision
        true_var = aleatoric_uncertainty ** 2
        var_loss = F.mse_loss(pred_var, true_var)

        # Combined loss
        total_loss = hetero_loss + self.alpha * rank_loss + self.beta * var_loss

        return {
            'loss': total_loss,
            'hetero_loss': hetero_loss,
            'rank_loss': rank_loss,
            'var_loss': var_loss,
            'mean_pred_var': pred_var.mean().item(),
            'mean_true_var': true_var.mean().item()
        }


class AdaptiveNoiseGatedRanking(nn.Module):
    """
    Noise-gated ranking with adaptive weight scheduling.

    Schedules α (ranking weight) from low to high during training,
    starting with regression and gradually emphasizing ranking.
    """

    def __init__(self, alpha_init: float = 0.0, alpha_final: float = 0.5,
                 beta: float = 0.1, warmup_epochs: int = 10,
                 total_epochs: int = 80, schedule: str = 'linear',
                 rank_k: float = 1.0, rank_sigma: float = 1.0):
        """
        Args:
            alpha_init: Initial ranking weight
            alpha_final: Final ranking weight
            beta: Variance supervision weight (constant)
            warmup_epochs: Epochs before starting schedule
            total_epochs: Total training epochs
            schedule: 'linear', 'exponential', or 'cosine'
            rank_k: Noise scaling for rank stability
            rank_sigma: BCE scaling for RankNet
        """
        super().__init__()
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.current_epoch = 0

        self.rank_stability = RankStabilityRankNet(sigma=rank_sigma, k=rank_k)

    def set_epoch(self, epoch: int):
        """Update current epoch for scheduling."""
        self.current_epoch = epoch

    def get_alpha(self) -> float:
        """Compute current alpha based on schedule."""
        if self.current_epoch < self.warmup_epochs:
            return self.alpha_init

        progress = (self.current_epoch - self.warmup_epochs) / \
                   max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)

        if self.schedule == 'linear':
            alpha = self.alpha_init + progress * (self.alpha_final - self.alpha_init)
        elif self.schedule == 'exponential':
            # Exponential interpolation
            alpha = self.alpha_init * ((self.alpha_final / max(self.alpha_init, 1e-6)) ** progress)
        elif self.schedule == 'cosine':
            # Cosine annealing
            import math
            alpha = self.alpha_final - 0.5 * (self.alpha_final - self.alpha_init) * \
                    (1 + math.cos(math.pi * progress))
        else:
            alpha = self.alpha_final

        return alpha

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                targets: torch.Tensor, aleatoric_uncertainty: torch.Tensor) -> dict:
        """
        Compute adaptive noise-gated loss.
        """
        mu = mu.view(-1)
        log_var = log_var.view(-1)
        targets = targets.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)

        log_var = torch.clamp(log_var, -10.0, 10.0)
        pred_var = torch.exp(log_var)

        # Heteroscedastic NLL
        residual_sq = (targets - mu) ** 2
        nll = 0.5 * (log_var + residual_sq / pred_var)
        hetero_loss = nll.mean()

        # Rank stability loss
        rank_loss = self.rank_stability(mu, targets, aleatoric_uncertainty)

        # Variance supervision
        true_var = aleatoric_uncertainty ** 2
        var_loss = F.mse_loss(pred_var, true_var)

        # Current alpha
        alpha = self.get_alpha()
        total_loss = hetero_loss + alpha * rank_loss + self.beta * var_loss

        return {
            'loss': total_loss,
            'hetero_loss': hetero_loss,
            'rank_loss': rank_loss,
            'var_loss': var_loss,
            'alpha': alpha,
            'mean_pred_var': pred_var.mean().item()
        }


class NoiseGatedMSERanking(nn.Module):
    """
    Simplified noise-gated loss using MSE instead of heteroscedastic NLL.

    For models without explicit variance prediction, uses aleatoric_uncertainty
    directly for sample weighting.

    L = weighted_MSE + α * L_rank_stability

    Where weights = 1 / (1 + σ²_true)
    """

    def __init__(self, alpha: float = 0.3, rank_k: float = 1.0,
                 rank_sigma: float = 1.0, temperature: float = 1.0):
        """
        Args:
            alpha: Weight for ranking loss
            rank_k: Noise scaling for rank stability
            rank_sigma: BCE scaling for RankNet
            temperature: Softness of noise weighting
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.rank_stability = RankStabilityRankNet(sigma=rank_sigma, k=rank_k)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                aleatoric_uncertainty: torch.Tensor) -> dict:
        """
        Compute noise-gated MSE + ranking loss.

        Args:
            predictions: Model predictions [batch_size]
            targets: Ground truth [batch_size]
            aleatoric_uncertainty: Noise proxy [batch_size]

        Returns:
            Dict with loss components
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)

        # Noise-based sample weights
        true_var = aleatoric_uncertainty ** 2
        weights = 1.0 / (1.0 + true_var / self.temperature)
        weights = weights / weights.sum() * len(weights)

        # Weighted MSE
        residual_sq = (targets - predictions) ** 2
        mse_loss = (weights * residual_sq).mean()

        # Rank stability loss
        rank_loss = self.rank_stability(predictions, targets, aleatoric_uncertainty)

        total_loss = mse_loss + self.alpha * rank_loss

        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'rank_loss': rank_loss
        }


def noise_gated_ranking_loss(mu: torch.Tensor, log_var: torch.Tensor,
                              targets: torch.Tensor, aleatoric_uncertainty: torch.Tensor,
                              alpha: float = 0.3, beta: float = 0.1) -> torch.Tensor:
    """
    Functional interface for noise-gated ranking loss.
    """
    loss_fn = NoiseGatedRanking(alpha=alpha, beta=beta)
    return loss_fn(mu, log_var, targets, aleatoric_uncertainty)['loss']
