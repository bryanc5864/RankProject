"""
Cross-Experiment Consensus Loss.

For sequences measured in multiple experiments:
1. Rank activities within each experiment (removes scale/offset noise)
2. Average ranks across experiments (reduces sequence-dependent noise)
3. Use averaged ranks as training targets
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import rankdata


class ConsensusLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.margin = config.get("consensus_margin", config.get("margin", 1.0))
        self.temperature = config.get("temperature", 0.1)

    @staticmethod
    def compute_consensus_targets(
        activities_by_experiment: dict[int, np.ndarray],
    ) -> torch.Tensor:
        """
        Compute rank-averaged consensus targets.

        Args:
            activities_by_experiment: {experiment_id: activity_array}
                where each array is indexed by a shared sequence index

        Returns:
            consensus_ranks: tensor of consensus rank-based targets [0, 1]
        """
        ranks = []
        for _, activities in activities_by_experiment.items():
            r = rankdata(activities) / len(activities)
            ranks.append(r)

        consensus = np.mean(ranks, axis=0)
        return torch.tensor(consensus, dtype=torch.float32)

    def forward(
        self, predictions: torch.Tensor, consensus_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Margin ranking loss using consensus targets.

        Weight by magnitude of consensus difference (larger diff = more reliable).
        """
        n = len(predictions)
        max_pairs = min(n * (n - 1) // 2, 10000)

        idx_i = torch.randint(0, n, (max_pairs,), device=predictions.device)
        idx_j = torch.randint(0, n, (max_pairs,), device=predictions.device)

        valid = idx_i != idx_j
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=predictions.device)

        diff = consensus_targets[idx_i] - consensus_targets[idx_j]
        pred_diff = predictions[idx_i] - predictions[idx_j]

        # Reliability weight: sigmoid of absolute consensus difference
        reliability = torch.sigmoid(
            (torch.abs(diff) - 0.1) / self.temperature
        )

        margin_loss = torch.clamp(
            self.margin - torch.sign(diff) * pred_diff, min=0
        )

        return (reliability * margin_loss).mean()
