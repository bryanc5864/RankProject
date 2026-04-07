"""
Motif Retention Analysis

Evaluates whether noise-resistant models preserve cell-type-specific
regulatory grammar (motif detection capabilities).

Methods:
1. In-silico mutagenesis (ISM) for attribution
2. Motif detection accuracy
3. Model comparison (ISM profile correlation)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings


class MotifRetentionAnalyzer:
    """
    Analyzes whether models retain ability to detect regulatory motifs.
    """

    def __init__(self, nucleotide_order: str = 'ACGT'):
        """
        Args:
            nucleotide_order: Order of nucleotides in one-hot encoding
        """
        self.nucleotide_order = nucleotide_order
        self.nuc_to_idx = {n: i for i, n in enumerate(nucleotide_order)}

    def in_silico_mutagenesis(self, model: nn.Module, sequences: torch.Tensor,
                               device: torch.device,
                               batch_size: int = 32) -> torch.Tensor:
        """
        Compute in-silico mutagenesis (ISM) attribution scores.

        For each position, mutate to all other nucleotides and measure
        the effect on model output.

        Args:
            model: Trained model
            sequences: One-hot sequences [n_seqs, 4, seq_len]
            device: Computation device
            batch_size: Batch size for efficiency

        Returns:
            ISM scores [n_seqs, 4, seq_len]
            Positive = mutation decreases activity (position is activating)
        """
        model.eval()
        n_seqs, n_channels, seq_len = sequences.shape

        # Get reference predictions
        ref_preds = []
        for i in range(0, n_seqs, batch_size):
            batch = sequences[i:i + batch_size].to(device)
            with torch.no_grad():
                pred = model(batch)
                if isinstance(pred, tuple):
                    pred = pred[0]  # For distributional models, use mean
                ref_preds.append(pred.cpu())
        ref_preds = torch.cat(ref_preds)  # [n_seqs]

        # ISM: for each position, compute effect of each mutation
        ism_scores = torch.zeros(n_seqs, n_channels, seq_len)

        for pos in range(seq_len):
            for mut_nuc in range(n_channels):
                # Create mutated sequences
                mutated = sequences.clone()
                mutated[:, :, pos] = 0
                mutated[:, mut_nuc, pos] = 1

                # Compute predictions for mutated sequences
                mut_preds = []
                for i in range(0, n_seqs, batch_size):
                    batch = mutated[i:i + batch_size].to(device)
                    with torch.no_grad():
                        pred = model(batch)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        mut_preds.append(pred.cpu())
                mut_preds = torch.cat(mut_preds)

                # ISM score = ref - mutant (positive = mutation hurts)
                ism_scores[:, mut_nuc, pos] = ref_preds - mut_preds

        return ism_scores

    def compute_attribution_profile(self, ism_scores: torch.Tensor,
                                     method: str = 'max_effect') -> torch.Tensor:
        """
        Summarize ISM scores into a single attribution profile.

        Args:
            ism_scores: ISM scores [n_seqs, 4, seq_len]
            method: Aggregation method
                   'max_effect': max absolute effect across mutations
                   'mean_effect': mean absolute effect
                   'ref_weighted': weight by reference nucleotide

        Returns:
            Attribution profile [n_seqs, seq_len]
        """
        if method == 'max_effect':
            # Maximum absolute effect across all mutations
            return ism_scores.abs().max(dim=1)[0]

        elif method == 'mean_effect':
            # Mean absolute effect
            return ism_scores.abs().mean(dim=1)

        elif method == 'ref_weighted':
            # Effect of mutating away from reference
            # (requires knowing which nucleotide is reference)
            # Use max for simplicity
            return ism_scores.abs().max(dim=1)[0]

        else:
            raise ValueError(f"Unknown method: {method}")

    def find_high_attribution_regions(self, attribution: torch.Tensor,
                                       threshold_quantile: float = 0.9,
                                       min_width: int = 6) -> List[Tuple[int, int]]:
        """
        Find contiguous regions with high attribution.

        Args:
            attribution: Attribution profile [seq_len]
            threshold_quantile: Quantile for defining "high" attribution
            min_width: Minimum width for a valid region

        Returns:
            List of (start, end) tuples for high-attribution regions
        """
        attribution = attribution.cpu().numpy()
        threshold = np.quantile(attribution, threshold_quantile)

        high_mask = attribution > threshold

        regions = []
        in_region = False
        start = 0

        for i, is_high in enumerate(high_mask):
            if is_high and not in_region:
                start = i
                in_region = True
            elif not is_high and in_region:
                if i - start >= min_width:
                    regions.append((start, i))
                in_region = False

        # Handle region at end
        if in_region and len(attribution) - start >= min_width:
            regions.append((start, len(attribution)))

        return regions

    def motif_detection_accuracy(self, model: nn.Module,
                                  sequences: torch.Tensor,
                                  motif_positions: List[Dict],
                                  device: torch.device,
                                  tolerance: int = 2) -> Dict[str, float]:
        """
        Evaluate motif detection accuracy using ISM.

        Checks if known motif positions have high attribution.

        Args:
            model: Trained model
            sequences: One-hot sequences [n_seqs, 4, seq_len]
            motif_positions: List of dicts with 'seq_idx', 'start', 'end', 'motif_name'
            device: Computation device
            tolerance: Position tolerance for matching

        Returns:
            Dict with detection metrics
        """
        # Compute ISM
        ism_scores = self.in_silico_mutagenesis(model, sequences, device)
        attribution = self.compute_attribution_profile(ism_scores)

        # Evaluate each motif
        detected = 0
        total = len(motif_positions)
        attribution_at_motifs = []

        for motif_info in motif_positions:
            seq_idx = motif_info['seq_idx']
            start = motif_info['start']
            end = motif_info['end']

            # Get attribution in motif region
            motif_attr = attribution[seq_idx, start:end].mean().item()
            attribution_at_motifs.append(motif_attr)

            # Get attribution in flanking regions (for comparison)
            flank_left = max(0, start - (end - start))
            flank_right = min(attribution.shape[1], end + (end - start))
            flank_attr = torch.cat([
                attribution[seq_idx, flank_left:start],
                attribution[seq_idx, end:flank_right]
            ]).mean().item()

            # Motif is "detected" if its attribution exceeds flanks
            if motif_attr > flank_attr * 1.5:  # 50% higher than flanks
                detected += 1

        detection_rate = detected / total if total > 0 else 0

        return {
            'detection_rate': detection_rate,
            'n_detected': detected,
            'n_total': total,
            'mean_motif_attribution': np.mean(attribution_at_motifs) if attribution_at_motifs else 0,
            'std_motif_attribution': np.std(attribution_at_motifs) if attribution_at_motifs else 0
        }

    def compare_models(self, model1: nn.Module, model2: nn.Module,
                       sequences: torch.Tensor, device: torch.device,
                       batch_size: int = 32) -> Dict[str, float]:
        """
        Compare ISM profiles between two models.

        High correlation indicates both models use similar sequence features.
        Used to verify noise-resistant models maintain regulatory grammar.

        Args:
            model1: First model (e.g., baseline)
            model2: Second model (e.g., noise-resistant)
            sequences: One-hot sequences [n_seqs, 4, seq_len]
            device: Computation device

        Returns:
            Dict with comparison metrics (target: profile_correlation > 0.9)
        """
        # Compute ISM for both models
        ism1 = self.in_silico_mutagenesis(model1, sequences, device, batch_size)
        ism2 = self.in_silico_mutagenesis(model2, sequences, device, batch_size)

        # Compute attribution profiles
        attr1 = self.compute_attribution_profile(ism1)
        attr2 = self.compute_attribution_profile(ism2)

        # Per-sequence correlations
        correlations = []
        for i in range(sequences.shape[0]):
            a1 = attr1[i].cpu().numpy()
            a2 = attr2[i].cpu().numpy()
            r, _ = stats.pearsonr(a1, a2)
            if not np.isnan(r):
                correlations.append(r)

        # Global correlation (all positions, all sequences)
        global_r, global_p = stats.pearsonr(
            attr1.view(-1).cpu().numpy(),
            attr2.view(-1).cpu().numpy()
        )

        # Spearman correlation (rank-based)
        global_spearman, _ = stats.spearmanr(
            attr1.view(-1).cpu().numpy(),
            attr2.view(-1).cpu().numpy()
        )

        return {
            'profile_correlation': float(global_r),
            'profile_correlation_p': float(global_p),
            'profile_spearman': float(global_spearman),
            'mean_per_seq_correlation': float(np.mean(correlations)) if correlations else 0,
            'std_per_seq_correlation': float(np.std(correlations)) if correlations else 0,
            'min_per_seq_correlation': float(np.min(correlations)) if correlations else 0,
            'interpretation': 'preserved' if global_r > 0.9 else 'moderate' if global_r > 0.7 else 'diverged'
        }

    def attribution_stability(self, model: nn.Module, sequences: torch.Tensor,
                               device: torch.device, n_samples: int = 5,
                               dropout_enabled: bool = True) -> Dict[str, float]:
        """
        Measure stability of attribution under model uncertainty.

        For models with dropout, compute ISM multiple times and measure
        consistency.

        Args:
            model: Model with dropout
            sequences: Input sequences
            device: Computation device
            n_samples: Number of forward passes
            dropout_enabled: Whether to keep dropout on

        Returns:
            Dict with stability metrics
        """
        if dropout_enabled:
            model.train()  # Enable dropout
        else:
            model.eval()

        attribution_samples = []

        for _ in range(n_samples):
            ism = self.in_silico_mutagenesis(model, sequences, device)
            attr = self.compute_attribution_profile(ism)
            attribution_samples.append(attr)

        model.eval()

        # Stack samples
        stacked = torch.stack(attribution_samples)  # [n_samples, n_seqs, seq_len]

        # Compute coefficient of variation at each position
        mean_attr = stacked.mean(dim=0)
        std_attr = stacked.std(dim=0)
        cv = std_attr / (mean_attr.abs() + 1e-8)

        # Mean stability (lower CV = more stable)
        mean_cv = cv.mean().item()

        # Pairwise correlations between samples
        pairwise_corrs = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                r, _ = stats.pearsonr(
                    stacked[i].view(-1).cpu().numpy(),
                    stacked[j].view(-1).cpu().numpy()
                )
                pairwise_corrs.append(r)

        return {
            'mean_cv': mean_cv,
            'mean_pairwise_correlation': np.mean(pairwise_corrs) if pairwise_corrs else 1.0,
            'min_pairwise_correlation': np.min(pairwise_corrs) if pairwise_corrs else 1.0,
            'stability_interpretation': 'stable' if mean_cv < 0.1 else 'moderate' if mean_cv < 0.3 else 'unstable'
        }

    def extract_important_kmers(self, attribution: torch.Tensor,
                                 sequences: torch.Tensor,
                                 k: int = 8, top_n: int = 20) -> List[Dict]:
        """
        Extract k-mers at positions with highest attribution.

        Args:
            attribution: Attribution profile [n_seqs, seq_len]
            sequences: One-hot sequences [n_seqs, 4, seq_len]
            k: k-mer length
            top_n: Number of top k-mers to return

        Returns:
            List of dicts with 'kmer', 'position', 'attribution', 'seq_idx'
        """
        n_seqs, seq_len = attribution.shape

        # Convert one-hot to sequences
        nuc_map = np.array(list(self.nucleotide_order))

        kmers = []

        for seq_idx in range(n_seqs):
            seq_onehot = sequences[seq_idx].cpu().numpy()  # [4, seq_len]
            seq_attr = attribution[seq_idx].cpu().numpy()  # [seq_len]

            # Decode sequence
            seq_indices = seq_onehot.argmax(axis=0)
            seq_str = ''.join(nuc_map[seq_indices])

            # Find k-mers at each position
            for pos in range(seq_len - k + 1):
                kmer = seq_str[pos:pos + k]
                kmer_attr = seq_attr[pos:pos + k].mean()

                kmers.append({
                    'kmer': kmer,
                    'position': pos,
                    'attribution': float(kmer_attr),
                    'seq_idx': seq_idx
                })

        # Sort by attribution and return top
        kmers.sort(key=lambda x: x['attribution'], reverse=True)
        return kmers[:top_n]


def compare_cagi5_performance(model, test_data: Dict,
                               device: torch.device) -> Dict[str, float]:
    """
    Verify model performance on CAGI5 elements.

    Ensures noise-resistant models maintain or improve CAGI5 performance.

    Args:
        model: Trained model
        test_data: Dict with element_name -> (sequences, activities)
        device: Computation device

    Returns:
        Dict with per-element and mean Spearman
    """
    model.eval()
    results = {}
    all_spearmans = []

    for element_name, (sequences, activities) in test_data.items():
        sequences = sequences.to(device)

        with torch.no_grad():
            preds = model(sequences)
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = preds.cpu().numpy()

        activities = activities.cpu().numpy()
        spearman_r, _ = stats.spearmanr(preds, activities)

        results[element_name] = float(spearman_r)
        all_spearmans.append(spearman_r)

    results['mean_spearman'] = float(np.mean(all_spearmans))
    results['std_spearman'] = float(np.std(all_spearmans))

    return results
