"""
Distributional Head Models

Models that predict both mean (μ) and variance (σ²) for uncertainty estimation.
Compatible with heteroscedastic and distributional losses.
"""

import torch
import torch.nn as nn
from .dream_rnn import BHIFirstLayersBlock, BHICoreBlock


class DistributionalHead(nn.Module):
    """
    Dual-output head that predicts mean and log-variance.

    Output: (μ, log_σ²) where:
    - μ: predicted activity
    - log_σ²: predicted log-variance (for numerical stability)

    Used with distributional losses that supervise variance prediction.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 64):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Mean prediction head
        self.mu_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Log-variance prediction head
        self.log_var_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> tuple:
        """
        Args:
            features: Input features [batch_size, input_dim]

        Returns:
            (mu, log_var): Both [batch_size]
        """
        mu = self.mu_head(features).squeeze(-1)
        log_var = self.log_var_head(features).squeeze(-1)
        return mu, log_var


class DREAM_RNN_Distributional(nn.Module):
    """
    DREAM-RNN with distributional output head.

    Predicts both mean activity and aleatoric uncertainty.
    """

    def __init__(self, in_channels: int = 4, seq_len: int = 230,
                 dropout: float = 0.2):
        super().__init__()

        self.first_block = BHIFirstLayersBlock(
            in_channels=in_channels,
            out_channels=512,
            kernel_sizes=[9, 15],
            dropout=dropout
        )

        self.core_block = BHICoreBlock(
            in_channels=512,
            out_channels=512,
            lstm_hidden=320,
            kernel_sizes=[9, 15],
            dropout1=dropout,
            dropout2=0.5
        )

        self.pointwise_conv = nn.Conv1d(512, 256, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.distributional_head = DistributionalHead(
            input_dim=256,
            hidden_dim=64
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: Input sequences [batch_size, 4, seq_len]
            return_features: If True, also return intermediate features

        Returns:
            (mu, log_var) or (mu, log_var, features) if return_features
        """
        x = self.first_block(x)
        x = self.core_block(x)
        x = self.pointwise_conv(x)
        x = self.global_avg_pool(x)
        features = x.squeeze(-1)

        mu, log_var = self.distributional_head(features)

        if return_features:
            return mu, log_var, features
        return mu, log_var

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate representations."""
        x = self.first_block(x)
        x = self.core_block(x)
        x = self.pointwise_conv(x)
        x = self.global_avg_pool(x)
        return x.squeeze(-1)

    def predict_with_uncertainty(self, x: torch.Tensor) -> dict:
        """
        Predict with uncertainty estimates.

        Returns:
            Dict with 'mu', 'sigma', 'sigma_sq' (all as tensors)
        """
        mu, log_var = self.forward(x)
        sigma_sq = torch.exp(log_var)
        sigma = torch.sqrt(sigma_sq)

        return {
            'mu': mu,
            'sigma': sigma,
            'sigma_sq': sigma_sq,
            'log_var': log_var
        }


class DREAM_RNN_DistributionalDualHead(nn.Module):
    """
    DREAM-RNN with separate regression and distributional heads.

    - Regression head: standard activity prediction
    - Distributional head: (μ, σ²) for uncertainty-aware ranking

    Useful for combining MSE with distributional ranking losses.
    """

    def __init__(self, in_channels: int = 4, seq_len: int = 230,
                 dropout: float = 0.2):
        super().__init__()

        self.first_block = BHIFirstLayersBlock(
            in_channels=in_channels,
            out_channels=512,
            kernel_sizes=[9, 15],
            dropout=dropout
        )

        self.core_block = BHICoreBlock(
            in_channels=512,
            out_channels=512,
            lstm_hidden=320,
            kernel_sizes=[9, 15],
            dropout1=dropout,
            dropout2=0.5
        )

        self.pointwise_conv = nn.Conv1d(512, 256, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Standard regression head
        self.regression_head = nn.Linear(256, 1)

        # Distributional head (mean + variance)
        self.distributional_head = DistributionalHead(
            input_dim=256,
            hidden_dim=64
        )

    def forward(self, x: torch.Tensor, mode: str = 'all'):
        """
        Args:
            x: Input sequences [batch_size, 4, seq_len]
            mode: 'all', 'regression', or 'distributional'

        Returns:
            Depends on mode:
            - 'all': (regression_output, mu, log_var, features)
            - 'regression': regression_output
            - 'distributional': (mu, log_var)
        """
        x = self.first_block(x)
        x = self.core_block(x)
        x = self.pointwise_conv(x)
        x = self.global_avg_pool(x)
        features = x.squeeze(-1)

        if mode == 'regression':
            return self.regression_head(features).squeeze(-1)

        if mode == 'distributional':
            return self.distributional_head(features)

        # mode == 'all'
        reg_out = self.regression_head(features).squeeze(-1)
        mu, log_var = self.distributional_head(features)
        return reg_out, mu, log_var, features


class SharedEncoderDistributional(nn.Module):
    """
    Shared encoder architecture where the mean head shares some parameters
    with the variance head, but has separate final layers.

    This encourages the model to learn representations useful for both
    activity prediction and uncertainty estimation.
    """

    def __init__(self, in_channels: int = 4, seq_len: int = 230,
                 dropout: float = 0.2, shared_dim: int = 128):
        super().__init__()

        self.first_block = BHIFirstLayersBlock(
            in_channels=in_channels,
            out_channels=512,
            kernel_sizes=[9, 15],
            dropout=dropout
        )

        self.core_block = BHICoreBlock(
            in_channels=512,
            out_channels=512,
            lstm_hidden=320,
            kernel_sizes=[9, 15],
            dropout1=dropout,
            dropout2=0.5
        )

        self.pointwise_conv = nn.Conv1d(512, 256, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Shared transformation
        self.shared_layer = nn.Sequential(
            nn.Linear(256, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Separate final layers
        self.mu_final = nn.Linear(shared_dim, 1)
        self.log_var_final = nn.Linear(shared_dim, 1)

    def forward(self, x: torch.Tensor, return_shared: bool = False):
        """
        Args:
            x: Input sequences
            return_shared: If True, return shared representation

        Returns:
            (mu, log_var) or (mu, log_var, shared_features)
        """
        x = self.first_block(x)
        x = self.core_block(x)
        x = self.pointwise_conv(x)
        x = self.global_avg_pool(x)
        features = x.squeeze(-1)

        shared = self.shared_layer(features)

        mu = self.mu_final(shared).squeeze(-1)
        log_var = self.log_var_final(shared).squeeze(-1)

        if return_shared:
            return mu, log_var, shared
        return mu, log_var
