"""
Data augmentation strategies for noise-resistant MPRA training.

Two complementary approaches:
1. RCMixup: Reverse complement + Mixup for label-preserving augmentation
2. EvoAugLite: Random mutations and masking for robustness
"""

import torch
import torch.nn.functional as F
import numpy as np


class RCMixup:
    """
    Reverse Complement + Mixup augmentation.

    1. Reverse complement with probability p_rc
    2. Mixup: x' = λ*x_i + (1-λ)*x_j, y' = λ*y_i + (1-λ)*y_j
       λ ~ Beta(alpha, alpha)

    For MPRA data, RC should preserve activity (promoter is orientation-independent
    in most cases), and mixup creates soft interpolations between sequences.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.p_rc = config.get("rc_probability", 0.5)
        self.mixup_alpha = config.get("mixup_alpha", 0.4)
        self.apply_mixup = config.get("apply_mixup", True)

    def reverse_complement(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Reverse complement one-hot encoded sequences.

        For one-hot [L, 4] with order [A, C, G, T]:
        - Reverse along length dimension
        - Swap A<->T (indices 0<->3) and C<->G (indices 1<->2)

        Args:
            sequences: [B, L, 4] one-hot encoded sequences

        Returns:
            [B, L, 4] reverse complemented sequences
        """
        # Reverse along length dimension
        rc = torch.flip(sequences, dims=[1])
        # Complement: swap channels [A,C,G,T] -> [T,G,C,A] = indices [3,2,1,0]
        rc = rc[:, :, [3, 2, 1, 0]]
        return rc

    def __call__(self, sequences: torch.Tensor, activities: torch.Tensor,
                 replicate_std: torch.Tensor = None):
        """
        Apply RC-Mixup augmentation.

        Args:
            sequences: [B, L, 4] one-hot encoded sequences
            activities: [B] activity values
            replicate_std: [B] optional replicate standard deviations

        Returns:
            Augmented (sequences, activities, replicate_std)
        """
        B = sequences.shape[0]
        device = sequences.device

        # 1. Random reverse complement
        rc_mask = torch.rand(B, device=device) < self.p_rc
        sequences = sequences.clone()
        if rc_mask.any():
            sequences[rc_mask] = self.reverse_complement(sequences[rc_mask])

        # 2. Mixup
        if self.apply_mixup and B > 1:
            # Sample lambda from Beta distribution
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Random permutation for mixing partners
            perm = torch.randperm(B, device=device)

            # Mix sequences (soft one-hot)
            sequences = lam * sequences + (1 - lam) * sequences[perm]

            # Mix activities
            activities = lam * activities + (1 - lam) * activities[perm]

            # Propagate replicate_std: std' = sqrt(λ²*std_i² + (1-λ)²*std_j²)
            if replicate_std is not None:
                replicate_std = torch.sqrt(
                    lam**2 * replicate_std**2 +
                    (1 - lam)**2 * replicate_std[perm]**2
                )

        return sequences, activities, replicate_std


class EvoAugLite:
    """
    Lightweight evolutionary augmentation.

    Applies random single-nucleotide mutations and masking to sequences.
    Does NOT use insertions/deletions to avoid length changes.

    Inspired by EvoAug (Avsec et al.) but simplified for MPRA.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.mutation_prob = config.get("mutation_prob", 0.01)  # per position
        self.mask_prob = config.get("mask_prob", 0.05)  # per position
        self.mask_value = 0.25  # uniform distribution = [0.25, 0.25, 0.25, 0.25]

    def __call__(self, sequences: torch.Tensor, activities: torch.Tensor,
                 replicate_std: torch.Tensor = None):
        """
        Apply EvoAug-Lite augmentation.

        Args:
            sequences: [B, L, 4] one-hot encoded sequences
            activities: [B] activity values (unchanged)
            replicate_std: [B] optional replicate standard deviations (unchanged)

        Returns:
            Augmented (sequences, activities, replicate_std)
            Note: activities and replicate_std are returned unchanged since
            random mutations shouldn't systematically change activity.
        """
        B, L, C = sequences.shape
        device = sequences.device

        sequences = sequences.clone()

        # 1. Random single-nucleotide mutations
        mutation_mask = torch.rand(B, L, device=device) < self.mutation_prob
        if mutation_mask.any():
            # Random new nucleotides
            new_nucs = torch.randint(0, C, (B, L), device=device)
            # Create new one-hot
            new_onehot = F.one_hot(new_nucs, num_classes=C).float()
            # Apply mutations
            sequences = torch.where(
                mutation_mask.unsqueeze(-1).expand_as(sequences),
                new_onehot,
                sequences
            )

        # 2. Random masking to uniform distribution
        mask_mask = torch.rand(B, L, device=device) < self.mask_prob
        if mask_mask.any():
            uniform = torch.full((B, L, C), self.mask_value, device=device)
            sequences = torch.where(
                mask_mask.unsqueeze(-1).expand_as(sequences),
                uniform,
                sequences
            )

        return sequences, activities, replicate_std


class CombinedAugmentation:
    """Combines multiple augmentation strategies."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.augmenters = []

        if config.get("use_rc_mixup", False):
            self.augmenters.append(RCMixup(config))
        if config.get("use_evoaug", False):
            self.augmenters.append(EvoAugLite(config))

    def __call__(self, sequences: torch.Tensor, activities: torch.Tensor,
                 replicate_std: torch.Tensor = None):
        """Apply all augmentations in sequence."""
        for aug in self.augmenters:
            sequences, activities, replicate_std = aug(sequences, activities, replicate_std)
        return sequences, activities, replicate_std


def get_augmenter(config: dict):
    """Factory function to create augmenter from config."""
    aug_type = config.get("augmentation", "none")

    if aug_type == "none" or not aug_type:
        return None
    elif aug_type == "rc_mixup":
        return RCMixup(config)
    elif aug_type == "evoaug":
        return EvoAugLite(config)
    elif aug_type == "both":
        config["use_rc_mixup"] = True
        config["use_evoaug"] = True
        return CombinedAugmentation(config)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
