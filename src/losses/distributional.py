"""
Distributional Loss with Variance Supervision

Predicts both mean (μ) and variance (σ²) with explicit supervision
on the variance from aleatoric uncertainty.

Key insight: If the model learns to predict uncertainty accurately,
it can focus regression on low-noise samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionalLoss(nn.Module):
    """
    Combined loss for distributional predictions.

    Loss = MSE(μ, y) + λ_var * MSE(σ², aleatoric_uncertainty²)

    The variance supervision term encourages the model to
    accurately predict measurement noise.
    """

    def __init__(self, lambda_var: float = 1.0):
        """
        Args:
            lambda_var: Weight for variance supervision term
        """
        super().__init__()
        self.lambda_var = lambda_var

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                targets: torch.Tensor, aleatoric_uncertainty: torch.Tensor) -> dict:
        """
        Compute distributional loss.

        Args:
            mu: Predicted mean [batch_size]
            log_var: Predicted log variance [batch_size]
            targets: Ground truth activity values [batch_size]
            aleatoric_uncertainty: True noise proxy [batch_size]

        Returns:
            Dict with 'loss', 'mse_loss', 'var_loss' components
        """
        mu = mu.view(-1)
        log_var = log_var.view(-1)
        targets = targets.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)

        # MSE on mean prediction
        mse_loss = F.mse_loss(mu, targets)

        # Variance supervision: predict aleatoric_uncertainty²
        pred_var = torch.exp(log_var)
        true_var = aleatoric_uncertainty ** 2
        var_loss = F.mse_loss(pred_var, true_var)

        total_loss = mse_loss + self.lambda_var * var_loss

        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'var_loss': var_loss
        }


class HeteroscedasticDistributionalLoss(nn.Module):
    """
    Heteroscedastic NLL with explicit variance supervision.

    Loss = 0.5 * [log(σ²) + (y - μ)² / σ²] + λ * MSE(σ², true_var)

    Combines:
    1. Heteroscedastic NLL: downweights noisy samples automatically
    2. Variance supervision: explicit guidance on uncertainty
    """

    def __init__(self, lambda_var: float = 0.5, min_log_var: float = -10.0,
                 max_log_var: float = 10.0):
        """
        Args:
            lambda_var: Weight for variance supervision
            min_log_var: Lower bound on log variance (stability)
            max_log_var: Upper bound on log variance (stability)
        """
        super().__init__()
        self.lambda_var = lambda_var
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                targets: torch.Tensor, aleatoric_uncertainty: torch.Tensor) -> dict:
        """
        Compute heteroscedastic loss with variance supervision.
        """
        mu = mu.view(-1)
        log_var = log_var.view(-1)
        targets = targets.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)

        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        pred_var = torch.exp(log_var)

        # Heteroscedastic NLL
        # NLL = 0.5 * [log(σ²) + (y - μ)² / σ²]
        residual_sq = (targets - mu) ** 2
        nll = 0.5 * (log_var + residual_sq / pred_var)
        nll_loss = nll.mean()

        # Variance supervision
        true_var = aleatoric_uncertainty ** 2
        var_loss = F.mse_loss(pred_var, true_var)

        total_loss = nll_loss + self.lambda_var * var_loss

        return {
            'loss': total_loss,
            'nll_loss': nll_loss,
            'var_loss': var_loss,
            'mean_pred_var': pred_var.mean().item(),
            'mean_true_var': true_var.mean().item()
        }


class VarianceWeightedMSE(nn.Module):
    """
    MSE weighted inversely by predicted variance.

    Samples with high predicted uncertainty contribute less to loss.
    Uses variance supervision to guide uncertainty estimation.
    """

    def __init__(self, lambda_var: float = 1.0, temperature: float = 1.0):
        """
        Args:
            lambda_var: Weight for variance supervision
            temperature: Softness of inverse-variance weighting
        """
        super().__init__()
        self.lambda_var = lambda_var
        self.temperature = temperature

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                targets: torch.Tensor, aleatoric_uncertainty: torch.Tensor) -> dict:
        """
        Compute variance-weighted MSE with supervision.
        """
        mu = mu.view(-1)
        log_var = log_var.view(-1)
        targets = targets.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)

        pred_var = torch.exp(log_var)

        # Inverse variance weights (with temperature for smoothness)
        # weight_i = 1 / (1 + σ²_i / T)
        weights = 1.0 / (1.0 + pred_var / self.temperature)
        weights = weights / weights.sum() * len(weights)  # Normalize

        # Weighted MSE
        residual_sq = (targets - mu) ** 2
        weighted_mse = (weights * residual_sq).mean()

        # Variance supervision
        true_var = aleatoric_uncertainty ** 2
        var_loss = F.mse_loss(pred_var, true_var)

        total_loss = weighted_mse + self.lambda_var * var_loss

        return {
            'loss': total_loss,
            'weighted_mse': weighted_mse,
            'var_loss': var_loss
        }


def distributional_loss(mu: torch.Tensor, log_var: torch.Tensor,
                        targets: torch.Tensor, aleatoric_uncertainty: torch.Tensor,
                        lambda_var: float = 1.0) -> torch.Tensor:
    """
    Functional interface for distributional loss.

    Returns only the scalar loss (for simple use cases).
    """
    loss_fn = DistributionalLoss(lambda_var=lambda_var)
    return loss_fn(mu, log_var, targets, aleatoric_uncertainty)['loss']


def heteroscedastic_distributional_loss(mu: torch.Tensor, log_var: torch.Tensor,
                                         targets: torch.Tensor,
                                         aleatoric_uncertainty: torch.Tensor,
                                         lambda_var: float = 0.5) -> torch.Tensor:
    """
    Functional interface for heteroscedastic distributional loss.
    """
    loss_fn = HeteroscedasticDistributionalLoss(lambda_var=lambda_var)
    return loss_fn(mu, log_var, targets, aleatoric_uncertainty)['loss']
