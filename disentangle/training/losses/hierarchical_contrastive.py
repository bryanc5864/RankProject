"""
Hierarchical (Activity-Weighted) Contrastive Loss (A2).

Extends the standard noise-contrastive loss with activity-concordance weighting.
Sequences with concordant K562/HepG2 activity get strong positive pull;
sequences with divergent activity get weak pull.

Rationale: concordant activity across cell types indicates true biological
signal, while divergent activity may reflect cell-type-specific regulation
that should not be fully denoised.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.temperature = config.get("contrastive_temperature", 0.07)
        self.concordance_scale = config.get("concordance_scale", 2.0)

    def forward(
        self,
        model,
        sequences: torch.Tensor,
        k562_activities: torch.Tensor,
        hepg2_activities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Activity-weighted contrastive loss for paired sequences.

        Args:
            model: DisentangleWrapper
            sequences: [B, L, 4] paired sequences
            k562_activities: [B] K562 activity measurements
            hepg2_activities: [B] HepG2 activity measurements

        Returns:
            Scalar contrastive loss with per-pair activity concordance weighting
        """
        # Project through different experiment normalizations
        z_a = model.project(sequences, experiment_id=0)  # K562
        z_b = model.project(sequences, experiment_id=1)  # HepG2

        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        B = len(sequences)

        # Compute rank concordance for weighting
        # Convert activities to ranks within batch
        k562_ranks = k562_activities.argsort().argsort().float() / max(B - 1, 1)
        hepg2_ranks = hepg2_activities.argsort().argsort().float() / max(B - 1, 1)

        # Concordance: how similar are the ranks? 1 = perfectly concordant
        rank_diff = torch.abs(k562_ranks - hepg2_ranks)
        concordance = 1.0 - rank_diff  # [B], range [0, 1]

        # Scale concordance to create weights for positive pairs
        # High concordance -> strong pull, low concordance -> weak pull
        pair_weights = torch.sigmoid(
            self.concordance_scale * (concordance - 0.5)
        )  # [B], range roughly [0.27, 0.73]

        # Cosine similarity matrix: [B, B]
        sim = torch.mm(z_a, z_b.t()) / self.temperature

        # Weighted InfoNCE: positive pairs weighted by concordance
        # Standard cross-entropy but with temperature-scaled positive logits
        labels = torch.arange(B, device=sequences.device)

        # Apply concordance weight to positive pair similarities
        # Weight the diagonal (positive pairs) by concordance
        positive_boost = torch.zeros_like(sim)
        positive_boost[labels, labels] = pair_weights * self.concordance_scale

        sim_weighted = sim + positive_boost

        loss_ab = F.cross_entropy(sim_weighted, labels)
        loss_ba = F.cross_entropy(sim_weighted.t(), labels)

        return (loss_ab + loss_ba) / 2
