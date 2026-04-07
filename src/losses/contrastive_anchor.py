"""
Contrastive Noise Anchoring Loss

Uses noise-based similarity for contrastive learning:
- Positive pairs: similar activity, both low noise (reliable matches)
- Negative pairs: similar activity, different noise levels (confounders)

Goal: Learn representations that distinguish signal from noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveNoiseAnchor(nn.Module):
    """
    Contrastive learning with noise-aware pair construction.

    Learns embeddings where:
    - Clean samples with similar activity are close
    - Noisy samples are pushed away even if activity is similar

    This helps the model learn that noisy measurements are unreliable.
    """

    def __init__(self, temperature: float = 0.1, noise_threshold: float = 0.5,
                 activity_threshold: float = 0.1, margin: float = 0.5):
        """
        Args:
            temperature: Softmax temperature for contrastive loss
            noise_threshold: Threshold for defining "low noise" (quantile)
            activity_threshold: Threshold for "similar activity"
            margin: Margin for triplet-style loss component
        """
        super().__init__()
        self.temperature = temperature
        self.noise_threshold = noise_threshold
        self.activity_threshold = activity_threshold
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor,
                aleatoric_uncertainty: torch.Tensor) -> dict:
        """
        Compute contrastive noise anchoring loss.

        Args:
            embeddings: Model embeddings [batch_size, embed_dim]
            targets: Activity values [batch_size]
            aleatoric_uncertainty: Noise levels [batch_size]

        Returns:
            Dict with 'loss' and diagnostic info
        """
        batch_size = embeddings.shape[0]
        if batch_size < 4:
            return {'loss': torch.tensor(0.0, device=embeddings.device, requires_grad=True)}

        targets = targets.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)

        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t())  # [B, B]

        # Define noise categories
        noise_threshold_value = torch.quantile(aleatoric_uncertainty, self.noise_threshold)
        is_low_noise = aleatoric_uncertainty < noise_threshold_value
        is_high_noise = ~is_low_noise

        # Activity similarity matrix
        activity_diff = (targets.unsqueeze(1) - targets.unsqueeze(0)).abs()
        activity_threshold_value = torch.quantile(activity_diff[activity_diff > 0],
                                                   self.activity_threshold)
        is_similar_activity = activity_diff < activity_threshold_value

        # Positive pairs: similar activity AND both low noise
        both_low_noise = is_low_noise.unsqueeze(1) & is_low_noise.unsqueeze(0)
        positive_mask = is_similar_activity & both_low_noise

        # Remove self-pairs
        eye_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        positive_mask = positive_mask & ~eye_mask

        # Negative pairs: similar activity BUT different noise levels
        # (one low, one high - these are confounding pairs)
        mixed_noise = (is_low_noise.unsqueeze(1) & is_high_noise.unsqueeze(0)) | \
                      (is_high_noise.unsqueeze(1) & is_low_noise.unsqueeze(0))
        hard_negative_mask = is_similar_activity & mixed_noise

        # InfoNCE-style loss
        # For each anchor, pull positive pairs close, push negative pairs away
        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        n_valid_anchors = 0

        for i in range(batch_size):
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = hard_negative_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Positive similarity
            pos_sim = similarity[i, pos_indices] / self.temperature

            # Negative similarity
            neg_sim = similarity[i, neg_indices] / self.temperature

            # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # For each positive, compute loss
            for pos_idx in range(len(pos_indices)):
                pos_score = pos_sim[pos_idx]
                # Combine with all negatives
                all_scores = torch.cat([pos_score.unsqueeze(0), neg_sim])
                log_prob = pos_score - torch.logsumexp(all_scores, dim=0)
                loss = loss - log_prob

            n_valid_anchors += len(pos_indices)

        if n_valid_anchors > 0:
            loss = loss / n_valid_anchors

        return {
            'loss': loss,
            'n_positive_pairs': positive_mask.sum().item(),
            'n_hard_negative_pairs': hard_negative_mask.sum().item(),
            'n_valid_anchors': n_valid_anchors
        }


class TripletNoiseAnchor(nn.Module):
    """
    Triplet loss variant of noise anchoring.

    For each anchor (low-noise sample):
    - Positive: another low-noise sample with similar activity
    - Negative: high-noise sample with similar activity

    This is more interpretable than full contrastive loss.
    """

    def __init__(self, margin: float = 0.5, noise_percentile: float = 0.5,
                 activity_percentile: float = 0.2):
        """
        Args:
            margin: Triplet margin
            noise_percentile: Percentile for noise threshold
            activity_percentile: Percentile for activity similarity threshold
        """
        super().__init__()
        self.margin = margin
        self.noise_percentile = noise_percentile
        self.activity_percentile = activity_percentile

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor,
                aleatoric_uncertainty: torch.Tensor) -> dict:
        """
        Compute triplet noise anchoring loss.
        """
        batch_size = embeddings.shape[0]
        if batch_size < 4:
            return {'loss': torch.tensor(0.0, device=embeddings.device, requires_grad=True)}

        targets = targets.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute distance matrix
        dist_matrix = torch.cdist(embeddings, embeddings)  # [B, B]

        # Noise threshold
        noise_thresh = torch.quantile(aleatoric_uncertainty, self.noise_percentile)
        is_low_noise = aleatoric_uncertainty < noise_thresh

        # Activity similarity threshold
        activity_diff = (targets.unsqueeze(1) - targets.unsqueeze(0)).abs()
        nonzero_diffs = activity_diff[activity_diff > 0]
        if len(nonzero_diffs) == 0:
            return {'loss': torch.tensor(0.0, device=embeddings.device, requires_grad=True)}

        activity_thresh = torch.quantile(nonzero_diffs, self.activity_percentile)

        # Build triplets
        triplet_losses = []

        for anchor_idx in range(batch_size):
            if not is_low_noise[anchor_idx]:
                continue  # Only use low-noise samples as anchors

            anchor_activity = targets[anchor_idx]

            # Find positive: low noise, similar activity, not self
            pos_candidates = []
            for j in range(batch_size):
                if j == anchor_idx:
                    continue
                if is_low_noise[j] and activity_diff[anchor_idx, j] < activity_thresh:
                    pos_candidates.append(j)

            # Find negative: high noise, similar activity
            neg_candidates = []
            for j in range(batch_size):
                if not is_low_noise[j] and activity_diff[anchor_idx, j] < activity_thresh:
                    neg_candidates.append(j)

            if not pos_candidates or not neg_candidates:
                continue

            # Use hardest positive (farthest) and hardest negative (closest)
            pos_dists = dist_matrix[anchor_idx, pos_candidates]
            neg_dists = dist_matrix[anchor_idx, neg_candidates]

            hardest_pos_dist = pos_dists.max()
            hardest_neg_dist = neg_dists.min()

            # Triplet loss: max(0, d_pos - d_neg + margin)
            triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
            triplet_losses.append(triplet_loss)

        if not triplet_losses:
            return {'loss': torch.tensor(0.0, device=embeddings.device, requires_grad=True)}

        loss = torch.stack(triplet_losses).mean()

        return {
            'loss': loss,
            'n_triplets': len(triplet_losses)
        }


class SoftContrastiveNoiseAnchor(nn.Module):
    """
    Soft contrastive loss using noise levels as continuous weights.

    Instead of hard thresholds, uses noise ratio to weight pairs:
    - Pair weight = exp(-k * (σ_i + σ_j)) for similar activity pairs
    - Pushes noisy pairs apart proportionally to their noise
    """

    def __init__(self, temperature: float = 0.1, noise_scale: float = 1.0,
                 activity_scale: float = 1.0):
        """
        Args:
            temperature: Softmax temperature
            noise_scale: Scale for noise-based weighting
            activity_scale: Scale for activity-based similarity
        """
        super().__init__()
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.activity_scale = activity_scale

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor,
                aleatoric_uncertainty: torch.Tensor) -> dict:
        """
        Compute soft contrastive noise loss.
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return {'loss': torch.tensor(0.0, device=embeddings.device, requires_grad=True)}

        targets = targets.view(-1)
        aleatoric_uncertainty = aleatoric_uncertainty.view(-1)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Activity similarity (higher = more similar)
        activity_diff = (targets.unsqueeze(1) - targets.unsqueeze(0)).abs()
        activity_sim = torch.exp(-self.activity_scale * activity_diff)

        # Noise similarity (higher = both cleaner)
        noise_sum = aleatoric_uncertainty.unsqueeze(1) + aleatoric_uncertainty.unsqueeze(0)
        noise_weight = torch.exp(-self.noise_scale * noise_sum)

        # Soft positive indicator: high activity similarity AND low noise
        soft_positive = activity_sim * noise_weight

        # Soft negative indicator: high activity similarity AND high noise
        soft_negative = activity_sim * (1 - noise_weight)

        # Remove diagonal
        mask = ~torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)

        # Weighted contrastive loss
        # Pull together soft positives, push apart soft negatives
        pos_loss = -(soft_positive[mask] * similarity[mask]).sum() / \
                    (soft_positive[mask].sum() + 1e-8)
        neg_loss = (soft_negative[mask] * F.relu(similarity[mask] + 0.5)).sum() / \
                   (soft_negative[mask].sum() + 1e-8)

        loss = pos_loss + neg_loss

        return {
            'loss': loss,
            'pos_loss': pos_loss.item(),
            'neg_loss': neg_loss.item()
        }


def contrastive_noise_anchor_loss(embeddings: torch.Tensor, targets: torch.Tensor,
                                   aleatoric_uncertainty: torch.Tensor,
                                   temperature: float = 0.1) -> torch.Tensor:
    """
    Functional interface for contrastive noise anchoring.
    """
    loss_fn = ContrastiveNoiseAnchor(temperature=temperature)
    return loss_fn(embeddings, targets, aleatoric_uncertainty)['loss']
