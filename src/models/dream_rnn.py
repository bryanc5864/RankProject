"""
DREAM-RNN Model Architecture

Based on the DREAM paper implementation for MPRA prediction.
Architecture: Conv -> BiLSTM -> Conv -> Global Pool -> Dense
"""

import torch
import torch.nn as nn


class BHIFirstLayersBlock(nn.Module):
    """
    First layer block: Two Conv1D layers with different kernel sizes.

    - Conv1D with kernel size 9, 256 channels
    - Conv1D with kernel size 15, 256 channels
    - Concatenate -> 512 channels total
    - BatchNorm, ReLU, Dropout
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 512,
                 kernel_sizes: list = [9, 15], dropout: float = 0.2):
        super().__init__()
        self.out_channels = out_channels

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels // 2, kernel_size=k, padding='same'),
                nn.BatchNorm1d(out_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outputs = [conv(x) for conv in self.conv_blocks]
        return torch.cat(conv_outputs, dim=1)


class BHICoreBlock(nn.Module):
    """
    Core layer block: BiLSTM followed by Conv layers.

    - Bi-LSTM with 320 hidden units per direction (640 total)
    - Two Conv1D layers similar to first block
    - BatchNorm, ReLU, Dropout
    """

    def __init__(self, in_channels: int = 512, out_channels: int = 512,
                 lstm_hidden: int = 320, kernel_sizes: list = [9, 15],
                 dropout1: float = 0.2, dropout2: float = 0.5):
        super().__init__()
        self.out_channels = out_channels

        self.lstm = nn.LSTM(in_channels, lstm_hidden,
                           bidirectional=True, batch_first=True)

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(lstm_hidden * 2, out_channels // 2, kernel_size=k, padding='same'),
                nn.BatchNorm1d(out_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout1)
            ) for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM expects (batch, seq, features), we have (batch, features, seq)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.transpose(1, 2)

        conv_outputs = [conv(lstm_out) for conv in self.conv_blocks]
        output = torch.cat(conv_outputs, dim=1)
        return self.dropout(output)


class FinalBlock(nn.Module):
    """
    Final layer block: Pointwise conv -> Global avg pool -> Dense.
    """

    def __init__(self, in_channels: int = 512, hidden_dim: int = 256,
                 n_outputs: int = 2):
        super().__init__()
        self.pointwise_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.final_dense = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise_conv(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        return self.final_dense(x)


class DREAM_RNN(nn.Module):
    """
    Full DREAM-RNN model for MPRA prediction.

    Architecture:
        Input: [batch, 4, seq_len] one-hot encoded DNA
        -> First Block (Conv + Conv concat)
        -> Core Block (BiLSTM + Conv)
        -> Final Block (Conv1x1 + Pool + Dense)
        Output: [batch, n_outputs]
    """

    def __init__(self, in_channels: int = 4, seq_len: int = 230,
                 n_outputs: int = 2, dropout: float = 0.2):
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

        self.final_block = FinalBlock(
            in_channels=512,
            hidden_dim=256,
            n_outputs=n_outputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_block(x)
        x = self.core_block(x)
        x = self.final_block(x)
        return x

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate representations before final dense layer."""
        x = self.first_block(x)
        x = self.core_block(x)
        x = self.final_block.pointwise_conv(x)
        x = self.final_block.global_avg_pool(x)
        return x.squeeze(-1)


class DREAM_RNN_SingleOutput(nn.Module):
    """
    DREAM-RNN variant with single output (activity only).

    For use with ranking losses that expect scalar outputs.
    """

    def __init__(self, in_channels: int = 4, seq_len: int = 230,
                 dropout: float = 0.2):
        super().__init__()
        self.backbone = DREAM_RNN(
            in_channels=in_channels,
            seq_len=seq_len,
            n_outputs=1,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


class DREAM_RNN_DualHead(nn.Module):
    """
    DREAM-RNN with separate regression and ranking heads.

    Useful for multi-task learning with combined losses.
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

        # Shared representation
        self.pointwise_conv = nn.Conv1d(512, 256, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Separate heads
        self.regression_head = nn.Linear(256, 1)
        self.ranking_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, return_both: bool = True):
        x = self.first_block(x)
        x = self.core_block(x)
        x = self.pointwise_conv(x)
        x = self.global_avg_pool(x)
        features = x.squeeze(-1)

        reg_out = self.regression_head(features).squeeze(-1)
        rank_out = self.ranking_head(features).squeeze(-1)

        if return_both:
            return reg_out, rank_out, features
        return reg_out

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_block(x)
        x = self.core_block(x)
        x = self.pointwise_conv(x)
        x = self.global_avg_pool(x)
        return x.squeeze(-1)
