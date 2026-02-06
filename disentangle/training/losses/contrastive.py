"""
Noise-Contrastive Representation Loss.

Uses paired sequences (same sequence in different experiments) to learn
experiment-invariant representations.

Anchor:   Sequence X measured in Experiment A
Positive: Sequence X measured in Experiment B (same biology, different noise)
Negative: Sequence Y measured in Experiment A (different biology, same noise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseContrastiveLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.temperature = config.get("contrastive_temperature", 0.07)

    def forward(
        self,
        anchor_reps: torch.Tensor,
        positive_reps: torch.Tensor,
        negative_reps: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss.

        Args:
            anchor_reps:   [B, D] anchor sequences (seq X in exp A)
            positive_reps: [B, D] positive sequences (seq X in exp B)
            negative_reps: [B, N_neg, D] negative sequences (different seqs in exp A)
        """
        anchor_reps = F.normalize(anchor_reps, dim=-1)
        positive_reps = F.normalize(positive_reps, dim=-1)
        negative_reps = F.normalize(negative_reps, dim=-1)

        # Positive similarity: same sequence, different experiment
        pos_sim = torch.sum(anchor_reps * positive_reps, dim=-1) / self.temperature

        # Negative similarity: different sequence, same experiment
        neg_sim = (
            torch.bmm(negative_reps, anchor_reps.unsqueeze(-1)).squeeze(-1)
            / self.temperature
        )

        # InfoNCE
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(len(anchor_reps), dtype=torch.long, device=anchor_reps.device)

        return F.cross_entropy(logits, labels)
