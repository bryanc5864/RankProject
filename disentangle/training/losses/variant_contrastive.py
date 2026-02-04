"""
Variant-Contrastive Loss (B2).

Generates random single-nucleotide mutations in-batch and uses repulsive
InfoNCE to push apart reference and mutant representations. This encourages
the model to be sensitive to individual variant effects.

Key idea: if the model maps ref and mutant to the same representation,
it cannot distinguish variant effects for CAGI5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariantContrastiveLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_mutations = config.get("n_mutations_per_seq", 5)
        self.max_cosine = config.get("variant_max_cosine", 0.99)

    def _generate_mutations(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Generate random single-nucleotide mutations.

        Args:
            sequences: [B, L, 4] one-hot encoded DNA

        Returns:
            mutants: [B * n_mutations, L, 4] mutated sequences
        """
        B, L, C = sequences.shape
        n = self.n_mutations

        # Repeat each sequence n times
        mutants = sequences.repeat_interleave(n, dim=0)  # [B*n, L, 4]

        # Random positions and nucleotides
        positions = torch.randint(0, L, (B * n,), device=sequences.device)
        nucleotides = torch.randint(0, C, (B * n,), device=sequences.device)

        # Apply mutations
        idx = torch.arange(B * n, device=sequences.device)
        mutants[idx, positions, :] = 0.0
        mutants[idx, positions, nucleotides] = 1.0

        return mutants

    def forward(self, model, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute variant-contrastive repulsive loss.

        Args:
            model: DisentangleWrapper with encode() method
            sequences: [B, L, 4] one-hot sequences

        Returns:
            Scalar loss pushing ref and mutant representations apart
        """
        B = sequences.shape[0]
        n = self.n_mutations

        # Get reference representations (denoised)
        h_ref = model.encode(sequences, experiment_id=None)  # [B, D]
        h_ref = F.normalize(h_ref, dim=-1)

        # Generate and encode mutations
        mutants = self._generate_mutations(sequences)  # [B*n, L, 4]
        h_mut = model.encode(mutants, experiment_id=None)  # [B*n, D]
        h_mut = F.normalize(h_mut, dim=-1)

        # Reshape: [B, n, D]
        h_mut = h_mut.view(B, n, -1)

        # Cosine similarity between ref and each mutant: [B, n]
        h_ref_exp = h_ref.unsqueeze(1)  # [B, 1, D]
        cosine_sim = (h_ref_exp * h_mut).sum(dim=-1)  # [B, n]

        # Clamp to avoid log(0)
        cosine_sim = torch.clamp(cosine_sim, max=self.max_cosine)

        # Repulsive loss: -log(1 - sim)
        loss = -torch.log(1.0 - cosine_sim)  # [B, n]

        return loss.mean()
