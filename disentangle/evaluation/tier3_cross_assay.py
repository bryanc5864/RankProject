"""
Tier 3: Cross-Assay and Cross-Cell-Type Transfer Evaluation.

Tests whether the model learned general regulatory grammar.
Evaluation sets: CAGI5 saturation mutagenesis, HepG2 lentiMPRA.

Usage:
    python -m evaluation.tier3_cross_assay \
        --model_path results/ablation/cnn_full_disentangle_seed42/best.pt \
        --architecture cnn \
        --cagi5_data data/processed/cagi5.h5 \
        --output_file results/ablation/cnn_full_disentangle_seed42/tier3_results.json
"""

import argparse
import json

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau


def evaluate_cagi5(model, cagi5_data_path: str) -> dict:
    """
    Zero-shot CAGI5 variant effect prediction.

    For each variant:
    1. Predict activity for reference sequence
    2. Predict activity for alternate sequence
    3. Compute predicted_effect = alt - ref
    4. Compare to measured effects
    """
    with h5py.File(cagi5_data_path, "r") as f:
        ref_sequences = f["reference_sequences"][:]
        alt_sequences = f["alternate_sequences"][:]
        measured_effects = f["measured_effects"][:]

    model.eval()
    batch_size = 256

    def _predict_batch(sequences):
        preds = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = torch.tensor(
                    sequences[i:i + batch_size], dtype=torch.float32
                ).cuda()
                if hasattr(model, "predict_denoised"):
                    pred = model.predict_denoised(batch)
                else:
                    pred = model(batch)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    ref_preds = _predict_batch(ref_sequences)
    alt_preds = _predict_batch(alt_sequences)
    predicted_effects = alt_preds - ref_preds

    return {
        "tier": 3,
        "dataset": "cagi5",
        "spearman": float(spearmanr(predicted_effects, measured_effects)[0]),
        "kendall": float(kendalltau(predicted_effects, measured_effects)[0]),
        "direction_accuracy": float(
            np.mean(np.sign(predicted_effects) == np.sign(measured_effects))
        ),
        "n_variants": len(measured_effects),
    }


def evaluate_cross_cell_type(model, hepg2_data_path: str) -> dict:
    """Evaluate on HepG2 lentiMPRA (cross-cell-type transfer from K562)."""
    with h5py.File(hepg2_data_path, "r") as f:
        sequences = f["sequences"][:]
        activities = f["activities"][:]

    model.eval()
    predictions = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(
                sequences[i:i + batch_size], dtype=torch.float32
            ).cuda()
            if hasattr(model, "predict_denoised"):
                pred = model.predict_denoised(batch)
            else:
                pred = model(batch)
            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions)

    return {
        "tier": 3,
        "dataset": "hepg2_cross_cell_type",
        "spearman": float(spearmanr(predictions, activities)[0]),
        "kendall": float(kendalltau(predictions, activities)[0]),
        "n_sequences": len(activities),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--cagi5_data", default=None)
    parser.add_argument("--hepg2_data", default=None)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    print("Tier 3 evaluation CLI - implement model loading from checkpoint")
