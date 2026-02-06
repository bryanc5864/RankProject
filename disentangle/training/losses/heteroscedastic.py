"""
Heteroscedastic loss (Beta-NLL) for noise-aware training.

The model predicts both mean (mu) and log-variance (log_var).
Loss auto-weights by predicted noise - noisy samples contribute less to gradients.

Reference: Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation
with Probabilistic Neural Networks" (ICLR 2022)
"""

import torch
import torch.nn as nn


class HeteroscedasticLoss(nn.Module):
    """
    Beta-NLL heteroscedastic loss.

    L = weight * [log_var + (y - mu)^2 / exp(log_var)]
    weight = exp(log_var.detach())^beta

    Args:
        beta: Controls gradient weighting (default 0.5).
              beta=0: standard NLL (high-variance samples dominate gradients)
              beta=0.5: balanced (recommended)
              beta=1.0: constant weighting (ignores predicted variance)
        use_replicate_std_prior: If True, add prior encouraging
                                 log_var ≈ 2*log(replicate_std)
        prior_weight: Weight for the replicate_std prior term
    """

    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}
        self.beta = config.get("heteroscedastic_beta", 0.5)
        self.use_prior = config.get("use_replicate_std_prior", False)
        self.prior_weight = config.get("replicate_std_prior_weight", 0.1)

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                targets: torch.Tensor,
                replicate_std: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Beta-NLL loss.

        Args:
            mu: Predicted mean [B]
            log_var: Predicted log-variance [B]
            targets: Ground truth activity values [B]
            replicate_std: Optional replicate standard deviation [B]
                          Used for prior regularization if use_replicate_std_prior=True

        Returns:
            Scalar loss
        """
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10, max=10)

        # Squared residual
        sq_residual = (targets - mu) ** 2

        # NLL: log_var + residual^2 / exp(log_var)
        nll = log_var + sq_residual / (torch.exp(log_var) + 1e-8)

        # Beta weighting: detach to avoid double gradient
        if self.beta > 0:
            weight = torch.exp(log_var.detach()) ** self.beta
            nll = weight * nll

        loss = nll.mean()

        # Optional prior: encourage log_var ≈ 2*log(replicate_std)
        if self.use_prior and replicate_std is not None:
            # replicate_std is std, so variance = std^2, log_var_target = 2*log(std)
            log_var_target = 2 * torch.log(replicate_std + 1e-8)
            prior_loss = (log_var - log_var_target) ** 2
            loss = loss + self.prior_weight * prior_loss.mean()

        return loss


class HeteroscedasticMSELoss(nn.Module):
    """
    Simplified heteroscedastic loss that uses replicate_std directly as weights.

    Does NOT require model to predict variance - uses measured replicate_std
    to down-weight high-noise samples.

    L = (y - mu)^2 / (replicate_std^2 + epsilon)

    This is simpler than full Beta-NLL but still noise-aware.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}
        self.epsilon = config.get("heteroscedastic_epsilon", 0.1)
        self.normalize_weights = config.get("normalize_weights", True)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                replicate_std: torch.Tensor = None) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Args:
            predictions: Predicted values [B]
            targets: Ground truth [B]
            replicate_std: Replicate standard deviation [B]. If None, falls back to MSE.

        Returns:
            Scalar loss
        """
        sq_residual = (targets - predictions) ** 2

        if replicate_std is None:
            return sq_residual.mean()

        # Weight inversely by variance (std^2)
        weights = 1.0 / (replicate_std ** 2 + self.epsilon)

        if self.normalize_weights:
            weights = weights / weights.mean()

        return (weights * sq_residual).mean()
