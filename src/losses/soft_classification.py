"""
Soft Classification Loss

Bin continuous expression values into ordinal categories and use
cross-entropy loss with optional label smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SoftClassificationLoss(nn.Module):
    """
    Convert regression to ordinal classification with label smoothing.

    Bins continuous values into discrete categories and applies
    cross-entropy loss. Provides noise robustness through discretization.
    """

    def __init__(self, n_bins: int = 10, label_smoothing: float = 0.1,
                 bin_edges: Optional[torch.Tensor] = None):
        """
        Args:
            n_bins: Number of ordinal bins
            label_smoothing: Label smoothing factor (0 = no smoothing)
            bin_edges: Pre-computed bin edges [n_bins + 1]. If None, computed from data.
        """
        super().__init__()
        self.n_bins = n_bins
        self.label_smoothing = label_smoothing
        self.register_buffer('bin_edges', bin_edges)
        self._fitted = bin_edges is not None

    def fit(self, y: torch.Tensor):
        """Compute bin edges from training data using quantiles."""
        y_flat = y.view(-1)
        # Use quantiles for balanced bins
        quantiles = torch.linspace(0, 1, self.n_bins + 1, device=y.device)
        self.bin_edges = torch.quantile(y_flat, quantiles)
        # Ensure edges are strictly increasing
        self.bin_edges[-1] = self.bin_edges[-1] + 1e-6
        self._fitted = True

    def get_bin_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Convert continuous values to bin indices."""
        if not self._fitted:
            raise RuntimeError("Call fit() first or provide bin_edges")

        y_flat = y.view(-1)
        # bucketize returns indices in [0, n_bins]
        labels = torch.bucketize(y_flat, self.bin_edges[1:-1])
        return labels.clamp(0, self.n_bins - 1)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute soft classification loss.

        Args:
            logits: Model predictions [batch_size, n_bins]
            y: Ground truth continuous values [batch_size]

        Returns:
            Cross-entropy loss with label smoothing
        """
        if not self._fitted:
            self.fit(y)

        labels = self.get_bin_labels(y)

        return F.cross_entropy(
            logits, labels,
            label_smoothing=self.label_smoothing
        )


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss (cumulative link model).

    Instead of predicting a single class, predicts cumulative probabilities
    P(Y > k) for each threshold k. Better preserves ordinal structure.
    """

    def __init__(self, n_bins: int = 10, bin_edges: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer('bin_edges', bin_edges)
        self._fitted = bin_edges is not None

    def fit(self, y: torch.Tensor):
        """Compute bin edges from training data."""
        y_flat = y.view(-1)
        quantiles = torch.linspace(0, 1, self.n_bins + 1, device=y.device)
        self.bin_edges = torch.quantile(y_flat, quantiles)
        self.bin_edges[-1] = self.bin_edges[-1] + 1e-6
        self._fitted = True

    def get_bin_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Convert continuous values to bin indices."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        y_flat = y.view(-1)
        labels = torch.bucketize(y_flat, self.bin_edges[1:-1])
        return labels.clamp(0, self.n_bins - 1)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal regression loss.

        Args:
            logits: Cumulative logits [batch_size, n_bins - 1]
            y: Ground truth continuous values [batch_size]

        Returns:
            Binary cross-entropy loss for cumulative probabilities
        """
        if not self._fitted:
            self.fit(y)

        labels = self.get_bin_labels(y)
        batch_size = labels.shape[0]

        # Create cumulative targets: target[k] = 1 if label > k
        # Shape: [batch_size, n_bins - 1]
        thresholds = torch.arange(self.n_bins - 1, device=labels.device)
        cumulative_targets = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()

        # Binary cross-entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(logits, cumulative_targets)

        return loss

    def predict_class(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert cumulative logits to class predictions."""
        # P(Y > k) for each k
        cumulative_probs = torch.sigmoid(logits)
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        # Class is the first threshold we don't exceed
        predictions = (cumulative_probs > 0.5).sum(dim=1)
        return predictions


def soft_classification_loss(logits: torch.Tensor, y: torch.Tensor,
                             n_bins: int = 10, label_smoothing: float = 0.1) -> torch.Tensor:
    """
    Functional interface for soft classification loss.

    Note: This computes bin edges on-the-fly from the batch, which may
    not be ideal. For consistent bins, use SoftClassificationLoss class.
    """
    y_flat = y.view(-1)

    # Compute bin edges from this batch
    quantiles = torch.linspace(0, 1, n_bins + 1, device=y.device)
    bin_edges = torch.quantile(y_flat, quantiles)
    bin_edges[-1] = bin_edges[-1] + 1e-6

    # Get labels
    labels = torch.bucketize(y_flat, bin_edges[1:-1]).clamp(0, n_bins - 1)

    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
