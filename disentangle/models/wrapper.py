"""
DISENTANGLE wrapper that adds experiment-conditional normalization
to any base encoder.

Architecture-agnostic: wraps any model inheriting from BaseEncoder.
"""

import torch
import torch.nn as nn

from .encoders import BaseEncoder
from .heads import ProjectionHead


class DisentangleWrapper(nn.Module):
    """
    Wraps a base encoder with experiment-conditional batch normalization
    and a shared prediction head.

    During training: uses experiment-specific normalization.
    During inference: averages across all normalizations (denoised).
    """

    def __init__(self, base_model: BaseEncoder, n_experiments: int, config: dict):
        super().__init__()
        self.base_model = base_model
        self.n_experiments = n_experiments

        hidden_dim = base_model.hidden_dim

        # Experiment-conditional batch normalization
        self.exp_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_experiments)
        ])

        # Shared prediction head
        self.prediction_head = nn.Linear(hidden_dim, 1)

        # Optional projection head for contrastive learning
        projection_dim = config.get("projection_dim", 128)
        self.projection_head = ProjectionHead(hidden_dim, projection_dim)

    @property
    def hidden_dim(self) -> int:
        return self.base_model.hidden_dim

    def encode(self, x: torch.Tensor, experiment_id: int = None) -> torch.Tensor:
        """
        Get representations with optional experiment-specific normalization.

        Args:
            x: [B, L, 4] one-hot sequences
            experiment_id: int or None. If None, average across normalizations.
        """
        h = self.base_model.encode(x)

        if experiment_id is not None:
            h = self.exp_norms[experiment_id](h)
        else:
            # Denoised: average across experiment normalizations
            h_normalized = [norm(h) for norm in self.exp_norms]
            h = torch.mean(torch.stack(h_normalized), dim=0)

        return h

    def project(self, x: torch.Tensor, experiment_id: int = None) -> torch.Tensor:
        """Get projected representations for contrastive learning."""
        h = self.encode(x, experiment_id)
        return self.projection_head(h)

    def forward(self, x: torch.Tensor, experiment_id: int = None) -> torch.Tensor:
        h = self.encode(x, experiment_id)
        return self.prediction_head(h).squeeze(-1)

    def predict_denoised(self, x: torch.Tensor) -> torch.Tensor:
        """Denoised prediction: average across all experiment normalizations."""
        return self.forward(x, experiment_id=None)
