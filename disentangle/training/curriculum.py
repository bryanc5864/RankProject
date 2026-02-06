"""
Curriculum learning for DISENTANGLE training.

Two curriculum types:
1. CurriculumScheduler: Staged introduction of loss components
   - First N epochs: ranking loss only (stable foundation)
   - Next M epochs: add contrastive loss
   - Remaining epochs: add consensus loss (full DISENTANGLE)

2. NoiseCurriculum: Staged introduction of noisy samples
   - Early epochs: only low-noise samples (reliable measurements)
   - Later epochs: progressively include noisier samples
   - Based on replicate_std from the data
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler


class CurriculumScheduler:
    """Manages staged introduction of loss components."""

    def __init__(self, config: dict):
        self.ranking_start = config.get("curriculum_ranking_start", 0)
        self.contrastive_start = config.get("curriculum_contrastive_start", 10)
        self.consensus_start = config.get("curriculum_consensus_start", 20)

        self.w_ranking = config.get("w_ranking", 0.5)
        self.w_contrastive = config.get("w_contrastive", 0.5)
        self.w_consensus = config.get("w_consensus", 1.0)

    def get_weights(self, epoch: int) -> dict:
        """Return loss component weights for the given epoch."""
        weights = {
            "w_ranking": self.w_ranking if epoch >= self.ranking_start else 0.0,
            "w_contrastive": self.w_contrastive if epoch >= self.contrastive_start else 0.0,
            "w_consensus": self.w_consensus if epoch >= self.consensus_start else 0.0,
        }
        return weights

    def should_use_pairs(self, epoch: int) -> bool:
        """Whether contrastive pairs should be loaded this epoch."""
        return epoch >= self.contrastive_start


class NoiseCurriculum:
    """
    Noise-aware curriculum that progressively includes noisier samples.

    Based on replicate_std from the data:
    - Epoch 0-phase1_end: only samples with replicate_std < median
    - Epoch phase1_end-phase2_end: samples with replicate_std < 75th percentile
    - Epoch phase2_end+: all samples

    Uses WeightedRandomSampler with per-epoch weight updates.
    """

    def __init__(self, replicate_stds: np.ndarray, config: dict = None):
        """
        Initialize noise curriculum.

        Args:
            replicate_stds: Array of replicate standard deviations for all samples
            config: Configuration dict with curriculum parameters
        """
        config = config or {}
        self.replicate_stds = replicate_stds
        self.n_samples = len(replicate_stds)

        # Phase boundaries (epochs)
        self.phase1_end = config.get("noise_curriculum_phase1_end", 10)
        self.phase2_end = config.get("noise_curriculum_phase2_end", 25)

        # Compute percentile thresholds
        self.median_std = np.median(replicate_stds)
        self.p75_std = np.percentile(replicate_stds, 75)

        # Pre-compute masks for each phase
        self.phase1_mask = replicate_stds <= self.median_std
        self.phase2_mask = replicate_stds <= self.p75_std
        # phase3 = all samples (no mask needed)

        print(f"NoiseCurriculum initialized:")
        print(f"  Phase 1 (epochs 0-{self.phase1_end}): {self.phase1_mask.sum()} samples "
              f"(std <= {self.median_std:.4f})")
        print(f"  Phase 2 (epochs {self.phase1_end}-{self.phase2_end}): {self.phase2_mask.sum()} samples "
              f"(std <= {self.p75_std:.4f})")
        print(f"  Phase 3 (epochs {self.phase2_end}+): {self.n_samples} samples (all)")

    def get_sample_weights(self, epoch: int) -> np.ndarray:
        """
        Get per-sample weights for the given epoch.

        Returns:
            weights: Array of weights, where 0 means sample is excluded
        """
        if epoch < self.phase1_end:
            # Phase 1: only low-noise samples
            weights = self.phase1_mask.astype(np.float32)
        elif epoch < self.phase2_end:
            # Phase 2: low and medium noise samples
            weights = self.phase2_mask.astype(np.float32)
        else:
            # Phase 3: all samples
            weights = np.ones(self.n_samples, dtype=np.float32)

        return weights

    def get_sampler(self, epoch: int) -> WeightedRandomSampler:
        """
        Get a WeightedRandomSampler for the given epoch.

        Args:
            epoch: Current epoch number

        Returns:
            WeightedRandomSampler configured for this epoch's curriculum
        """
        weights = self.get_sample_weights(epoch)

        # Normalize weights so they sum to n_samples (for consistent batch counts)
        weights = weights / weights.sum() * self.n_samples

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=int(weights.sum()),
            replacement=True
        )

    def get_phase(self, epoch: int) -> int:
        """Return current phase (1, 2, or 3) for logging."""
        if epoch < self.phase1_end:
            return 1
        elif epoch < self.phase2_end:
            return 2
        else:
            return 3


def create_noise_curriculum(dataset, config: dict = None):
    """
    Factory function to create NoiseCurriculum from a dataset.

    Args:
        dataset: SequenceDataset with replicate_stds attribute
        config: Configuration dict

    Returns:
        NoiseCurriculum instance, or None if dataset lacks replicate_stds
    """
    if not hasattr(dataset, 'replicate_stds') or dataset.replicate_stds is None:
        print("Warning: Dataset lacks replicate_stds, noise curriculum disabled")
        return None

    return NoiseCurriculum(dataset.replicate_stds, config)
