"""
Adaptive Margin Ranking Loss.

Standard ranking loss that down-weights unreliable comparisons.
Sequences with small activity differences get lower weight,
because small differences may be noise rather than real biology.
"""

import torch
import torch.nn as nn


class AdaptiveMarginRankingLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.margin = config.get("ranking_margin", 1.0)
        self.noise_threshold = config.get("noise_threshold", 0.5)
        self.temperature = config.get("ranking_temperature", 0.5)
        self.n_pairs_per_sample = config.get("n_pairs_per_sample", 16)

    def forward(
        self,
        predictions: torch.Tensor,
        activities: torch.Tensor,
        replicate_std: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Weighted pairwise margin ranking loss.

        Args:
            predictions: [B] predicted activity scores
            activities: [B] measured activities
            replicate_std: [B] optional per-sequence replicate std
        """
        B = len(predictions)
        n_pairs = min(B * self.n_pairs_per_sample, B * (B - 1) // 2)

        idx_i = torch.randint(0, B, (n_pairs,), device=predictions.device)
        idx_j = torch.randint(0, B, (n_pairs,), device=predictions.device)

        # Remove self-pairs
        valid = idx_i != idx_j
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=predictions.device)

        act_diff = activities[idx_i] - activities[idx_j]
        pred_diff = predictions[idx_i] - predictions[idx_j]

        # Compute reliability weights
        if replicate_std is not None:
            pair_noise = torch.sqrt(replicate_std[idx_i] ** 2 + replicate_std[idx_j] ** 2)
            reliability = torch.sigmoid(
                (torch.abs(act_diff) - pair_noise) / self.temperature
            )
        else:
            reliability = torch.sigmoid(
                (torch.abs(act_diff) - self.noise_threshold) / self.temperature
            )

        # Margin ranking loss
        margin_loss = torch.clamp(
            self.margin - torch.sign(act_diff) * pred_diff, min=0
        )

        return (reliability * margin_loss).mean()
