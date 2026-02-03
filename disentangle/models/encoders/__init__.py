"""
Encoder architectures for sequence-to-function modeling.

All encoders inherit from BaseEncoder and implement:
- encode(sequences) -> representations [B, hidden_dim]
- forward(sequences) -> predictions [B]
"""

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Base class for all sequence encoders."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config["hidden_dim"]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer representations [B, hidden_dim]."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return activity predictions [B]."""
        h = self.encode(x)
        return self.prediction_head(h).squeeze(-1)


def build_encoder(architecture: str, config: dict) -> BaseEncoder:
    """Factory function to build encoder by name."""
    if architecture == "cnn":
        from .cnn import CNNEncoder
        return CNNEncoder(config)
    elif architecture == "dilated_cnn":
        from .dilated_cnn import DilatedCNNEncoder
        return DilatedCNNEncoder(config)
    elif architecture == "bilstm":
        from .bilstm import BiLSTMEncoder
        return BiLSTMEncoder(config)
    elif architecture == "transformer":
        from .transformer import TransformerEncoder
        return TransformerEncoder(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
