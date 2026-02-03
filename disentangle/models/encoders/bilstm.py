"""
Bidirectional LSTM encoder (inspired by DREAM Challenge RNN).

Input: [B, L, 4] one-hot DNA sequences
Output: [B] activity predictions
"""

import torch
import torch.nn as nn

from . import BaseEncoder


class BiLSTMEncoder(BaseEncoder):
    def __init__(self, config: dict):
        super().__init__(config)

        lstm_hidden = config.get("lstm_hidden", 128)
        n_layers = config.get("n_lstm_layers", 2)
        dropout = config.get("dropout", 0.1)

        # Optional initial conv for feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, self.hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.prediction_head = nn.Linear(self.hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).float()  # [B, 4, L]
        x = self.initial_conv(x)        # [B, 64, L]
        x = x.transpose(1, 2)           # [B, L, 64]

        _, (h_n, _) = self.lstm(x)

        # Use last hidden state from both directions
        h_forward = h_n[-2]   # [B, lstm_hidden]
        h_backward = h_n[-1]  # [B, lstm_hidden]
        h = torch.cat([h_forward, h_backward], dim=-1)  # [B, lstm_hidden*2]

        h = self.fc(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.prediction_head(h).squeeze(-1)
