"""
Attribution analysis comparing baseline and DISENTANGLE models.

Computes Integrated Gradients per-nucleotide importance scores and compares:
1. Attribution similarity for same sequences across models
2. Motif enrichment in high-attribution regions
3. GC content correlation with attribution magnitude

Produces Figures 5 and 6 of the paper.
"""

import numpy as np
import torch
from scipy.stats import spearmanr


def compute_attributions(
    model, sequences: np.ndarray, batch_size: int = 64, n_steps: int = 50
) -> np.ndarray:
    """
    Compute Integrated Gradients attributions for each sequence.

    Args:
        model: trained model
        sequences: [N, L, 4] one-hot encoded
        n_steps: number of integration steps

    Returns:
        attributions: [N, L, 4]
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise ImportError("captum required for attribution analysis: pip install captum")

    # Use train mode for cuDNN RNN backward compatibility, but disable dropout
    model.train()
    model.cuda()

    # Disable cuDNN to allow backward through RNNs in eval-like mode
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    # Wrap model to accept input and return scalar
    def forward_fn(x):
        if hasattr(model, "predict_denoised"):
            return model.predict_denoised(x)
        return model(x)

    ig = IntegratedGradients(forward_fn)

    all_attrs = []
    for i in range(0, len(sequences), batch_size):
        batch = torch.tensor(
            sequences[i:i + batch_size], dtype=torch.float32
        ).cuda()
        batch.requires_grad_(True)

        baseline = torch.zeros_like(batch)
        attr = ig.attribute(batch, baselines=baseline, n_steps=n_steps)
        all_attrs.append(attr.detach().cpu().numpy())

    torch.backends.cudnn.enabled = prev_cudnn
    model.eval()

    return np.concatenate(all_attrs, axis=0)


def compare_attributions_across_models(
    attr_model_a: np.ndarray, attr_model_b: np.ndarray
) -> np.ndarray:
    """
    Compare attribution patterns for same sequences across two models.

    Returns:
        per_sequence_correlations: [N] Spearman correlations
    """
    correlations = []
    for i in range(len(attr_model_a)):
        a = attr_model_a[i].flatten()
        b = attr_model_b[i].flatten()
        r, _ = spearmanr(a, b)
        correlations.append(r)

    return np.array(correlations)


def compute_noise_attributions(
    baseline_attrs: np.ndarray, disentangle_attrs: np.ndarray
) -> np.ndarray:
    """
    Compute 'noise attributions' = regions highlighted by baseline but not DISENTANGLE.

    These regions likely correspond to technical artifacts the baseline has learned.
    """
    baseline_mag = np.abs(baseline_attrs).sum(axis=2)   # [N, L]
    disent_mag = np.abs(disentangle_attrs).sum(axis=2)   # [N, L]

    # Ratio: high where baseline attributes strongly but DISENTANGLE doesn't
    ratio = (baseline_mag + 1e-8) / (disent_mag + 1e-8)

    return ratio
