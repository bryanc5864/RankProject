"""
Tier 2: Cross-Experiment Transfer Evaluation.

Train on experiments {A, B, C}, evaluate on held-out experiment D.
Same cell type, different assay/lab/batch.

THIS IS THE MAIN EVALUATION. DISENTANGLE should improve over baselines here.

Usage:
    python -m evaluation.tier2_cross_experiment \
        --model_path results/ablation/cnn_full_disentangle_seed42/best.pt \
        --architecture cnn \
        --held_out_data data/processed/atac_starrseq_K562.h5 \
        --output_file results/ablation/cnn_full_disentangle_seed42/tier2_results.json
"""

import argparse
import json

import h5py
import numpy as np
import torch

from .metrics import compute_all_metrics, compute_ndcg, compute_direction_accuracy


def evaluate_tier2(model, held_out_data_path: str) -> dict:
    """Evaluate on completely held-out experiment."""
    with h5py.File(held_out_data_path, "r") as f:
        sequences = f["sequences"][:]
        activities = f["activities"][:]

    model.eval()
    predictions = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(sequences[i:i + batch_size], dtype=torch.float32).cuda()
            if hasattr(model, "predict_denoised"):
                pred = model.predict_denoised(batch)
            else:
                pred = model(batch)
            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions)

    metrics = compute_all_metrics(predictions, activities)
    metrics["tier"] = 2

    # Additional Tier 2 metrics
    metrics["ndcg_10"] = compute_ndcg(activities, predictions, k=10)
    metrics["ndcg_50"] = compute_ndcg(activities, predictions, k=50)
    metrics["ndcg_100"] = compute_ndcg(activities, predictions, k=100)
    metrics["direction_accuracy"] = compute_direction_accuracy(
        predictions, activities, threshold=1.0
    )

    # Extreme-value Spearman (top/bottom 10%)
    from scipy.stats import spearmanr
    top_10 = np.percentile(activities, 90)
    bottom_10 = np.percentile(activities, 10)
    extreme_mask = (activities > top_10) | (activities < bottom_10)
    if extreme_mask.sum() > 10:
        metrics["extreme_spearman"] = float(
            spearmanr(predictions[extreme_mask], activities[extreme_mask])[0]
        )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--held_out_data", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    print("Tier 2 evaluation CLI - implement model loading from checkpoint")
