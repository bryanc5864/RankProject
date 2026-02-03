"""
Prediction and projection heads for DISENTANGLE models.
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """Linear prediction head mapping representations to activity scores."""

    def __init__(self, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.

    Maps representations to a lower-dimensional space where the contrastive
    loss is computed. This prevents contrastive learning from distorting the
    main representation space.
    """

    def __init__(self, hidden_dim: int, projection_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
