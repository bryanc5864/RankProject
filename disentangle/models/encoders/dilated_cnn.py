"""
Basenji-style dilated CNN encoder.

Uses dilated convolutions for exponentially growing receptive field.
Input: [B, L, 4] one-hot DNA sequences
Output: [B] activity predictions
"""

import torch
import torch.nn as nn

from . import BaseEncoder


class DilatedConvBlock(nn.Module):
    """Residual dilated convolution block."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DilatedCNNEncoder(BaseEncoder):
    def __init__(self, config: dict):
        super().__init__(config)

        n_filters = config.get("n_filters_dilated", 256)
        n_dilated = config.get("n_dilated_layers", 6)
        dropout = config.get("dropout", 0.1)

        self.initial_conv = nn.Sequential(
            nn.Conv1d(4, n_filters, 15, padding=7),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        dilated_layers = []
        for i in range(n_dilated):
            dilation = 2 ** i
            dilated_layers.append(
                DilatedConvBlock(n_filters, dilation=dilation, dropout=dropout)
            )
        self.dilated_layers = nn.Sequential(*dilated_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_filters, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.prediction_head = nn.Linear(self.hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).float()  # [B, 4, L]
        x = self.initial_conv(x)
        x = self.dilated_layers(x)
        x = self.global_pool(x).squeeze(-1)  # [B, n_filters]
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.prediction_head(h).squeeze(-1)
