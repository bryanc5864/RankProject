"""
Combined DISENTANGLE loss integrating all three noise-resistance strategies.
"""

import torch
import torch.nn as nn

from .consensus import ConsensusLoss
from .contrastive import NoiseContrastiveLoss
from .ranking import AdaptiveMarginRankingLoss


class DisentangleLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.consensus_loss = ConsensusLoss(config)
        self.contrastive_loss = NoiseContrastiveLoss(config)
        self.ranking_loss = AdaptiveMarginRankingLoss(config)

        self.w_consensus = config.get("w_consensus", 1.0)
        self.w_contrastive = config.get("w_contrastive", 0.5)
        self.w_ranking = config.get("w_ranking", 0.5)

    def forward(self, model, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            model: DisentangleWrapper model
            batch: dict containing:
                'sequences': [B, L, 4]
                'activities': [B]
                'experiment_ids': [B] (int)
                'replicate_std': [B] (optional)
                'anchor_sequences': [B_paired, L, 4] (optional, for contrastive)
                'positive_sequences': [B_paired, L, 4] (optional)
                'negative_sequences': [B_paired, N_neg, L, 4] (optional)
                'consensus_targets': [B] (optional)

        Returns:
            (total_loss, loss_components_dict)
        """
        predictions = model(batch["sequences"])

        total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        loss_components = {}

        # Strategy 1: Consensus loss (for sequences with consensus targets)
        if batch.get("consensus_targets") is not None:
            L_consensus = self.consensus_loss(predictions, batch["consensus_targets"])
            total_loss = total_loss + self.w_consensus * L_consensus
            loss_components["consensus"] = L_consensus.item()

        # Strategy 2: Contrastive loss (for paired sequences)
        if batch.get("anchor_sequences") is not None:
            anchor_reps = model.project(batch["anchor_sequences"])
            positive_reps = model.project(batch["positive_sequences"])

            neg_seqs = batch["negative_sequences"]
            B_neg, N_neg = neg_seqs.shape[0], neg_seqs.shape[1]
            neg_flat = neg_seqs.reshape(B_neg * N_neg, *neg_seqs.shape[2:])
            neg_reps_flat = model.project(neg_flat)
            negative_reps = neg_reps_flat.reshape(B_neg, N_neg, -1)

            L_contrastive = self.contrastive_loss(
                anchor_reps, positive_reps, negative_reps
            )
            total_loss = total_loss + self.w_contrastive * L_contrastive
            loss_components["contrastive"] = L_contrastive.item()

        # Strategy 3: Ranking loss (for all sequences)
        L_ranking = self.ranking_loss(
            predictions,
            batch["activities"],
            batch.get("replicate_std"),
        )
        total_loss = total_loss + self.w_ranking * L_ranking
        loss_components["ranking"] = L_ranking.item()

        loss_components["total"] = total_loss.item()

        return total_loss, loss_components
