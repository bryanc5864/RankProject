"""
Lightweight Transformer encoder for sequences (Enformer-lite).

Uses initial convolutions to reduce sequence length, then transformer layers.
Input: [B, L, 4] one-hot DNA sequences
Output: [B] activity predictions
"""

import math

import torch
import torch.nn as nn

from . import BaseEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(BaseEncoder):
    def __init__(self, config: dict):
        super().__init__(config)

        n_heads = config.get("n_heads", 8)
        n_layers = config.get("n_transformer_layers", 4)
        dropout = config.get("dropout", 0.1)

        # Initial conv to reduce sequence length and create embeddings
        self.initial_conv = nn.Sequential(
            nn.Conv1d(4, self.hidden_dim, 15, padding=7),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.pos_encoding = PositionalEncoding(self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.prediction_head = nn.Linear(self.hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).float()    # [B, 4, L]
        x = self.initial_conv(x)          # [B, hidden_dim, L']
        x = x.transpose(1, 2)             # [B, L', hidden_dim]
        x = self.pos_encoding(x)
        x = self.transformer(x)           # [B, L', hidden_dim]
        x = x.transpose(1, 2)             # [B, hidden_dim, L']
        x = self.global_pool(x).squeeze(-1)  # [B, hidden_dim]
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.prediction_head(h).squeeze(-1)
