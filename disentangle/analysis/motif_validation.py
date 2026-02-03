"""
Motif enrichment analysis for attribution validation.

Checks whether high-attribution regions correspond to known TF binding motifs
(biology) or technical artifacts (noise).

Uses JASPAR motif database for motif scanning.
"""

import numpy as np


def compute_gc_attribution_correlation(
    attributions: np.ndarray, sequences: np.ndarray
) -> float:
    """
    Correlation between GC content and attribution magnitude.

    High correlation may indicate the model is using GC as a feature,
    which could reflect PCR amplification bias rather than regulatory logic.
    """
    from scipy.stats import pearsonr

    # Per-position attribution magnitude
    attr_magnitude = np.abs(attributions).sum(axis=2)  # [N, L]

    # Per-position GC indicator
    gc_indicator = sequences[:, :, 1] + sequences[:, :, 2]  # C + G channels

    # Flatten and correlate
    r, p = pearsonr(attr_magnitude.flatten(), gc_indicator.flatten())
    return float(r)


def extract_high_attribution_sequences(
    attributions: np.ndarray,
    sequences: np.ndarray,
    top_fraction: float = 0.1,
    window_size: int = 12,
) -> list[str]:
    """
    Extract sequence windows with highest attribution scores.

    Returns list of DNA strings for motif analysis.
    """
    attr_magnitude = np.abs(attributions).sum(axis=2)  # [N, L]

    base_map = {0: "A", 1: "C", 2: "G", 3: "T"}
    high_attr_seqs = []

    for i in range(len(sequences)):
        # Find top positions
        threshold = np.percentile(attr_magnitude[i], (1 - top_fraction) * 100)
        high_pos = np.where(attr_magnitude[i] >= threshold)[0]

        # Extract windows around high positions
        for pos in high_pos:
            start = max(0, pos - window_size // 2)
            end = min(sequences.shape[1], start + window_size)
            window = sequences[i, start:end]

            seq_str = ""
            for j in range(len(window)):
                idx = window[j].argmax()
                seq_str += base_map.get(idx, "N") if window[j].sum() > 0 else "N"

            if len(seq_str) == window_size:
                high_attr_seqs.append(seq_str)

    return high_attr_seqs
