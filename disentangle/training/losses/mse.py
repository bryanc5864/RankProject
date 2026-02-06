"""Standard MSE loss baseline."""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Baseline MSE loss for activity prediction."""

    def forward(self, predictions: torch.Tensor, activities: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(predictions, activities)
