"""
Basset-style CNN encoder.

Architecture: 3 conv blocks (conv + BN + ReLU + pool) -> FC -> prediction
Input: [B, L, 4] one-hot DNA sequences
Output: [B] activity predictions
"""

import torch
import torch.nn as nn

from . import BaseEncoder


class CNNEncoder(BaseEncoder):
    def __init__(self, config: dict):
        super().__init__(config)

        n_filters = config.get("n_filters", [300, 200, 200])
        filter_sizes = config.get("filter_sizes", [19, 11, 7])
        pool_sizes = config.get("pool_sizes", [3, 4, 4])
        dropout = config.get("dropout", 0.1)

        layers = []
        in_channels = 4
        for n_filt, f_size, p_size in zip(n_filters, filter_sizes, pool_sizes):
            layers.extend([
                nn.Conv1d(in_channels, n_filt, f_size, padding=f_size // 2),
                nn.BatchNorm1d(n_filt),
                nn.ReLU(),
                nn.MaxPool1d(p_size),
                nn.Dropout(dropout),
            ])
            in_channels = n_filt

        self.conv_layers = nn.Sequential(*layers)

        # Compute output size after convolutions
        seq_len = config.get("sequence_length", 230)
        test_input = torch.zeros(1, 4, seq_len)
        with torch.no_grad():
            test_output = self.conv_layers(test_input)
        conv_output_size = test_output.shape[1] * test_output.shape[2]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.prediction_head = nn.Linear(self.hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 4] -> [B, 4, L] for conv1d
        x = x.transpose(1, 2).float()
        x = self.conv_layers(x)
        x = self.fc(x)
        return x  # [B, hidden_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.prediction_head(h).squeeze(-1)
