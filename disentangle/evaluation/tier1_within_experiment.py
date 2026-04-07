"""
Tier 1: Within-Experiment Evaluation.

Standard evaluation using random train/test split from the same experiment.
Sanity check: models should perform well here.
DISENTANGLE models may perform slightly worse (expected - no longer exploiting noise).

Usage:
    python -m evaluation.tier1_within_experiment \
        --model_path results/ablation/cnn_full_disentangle_seed42/best.pt \
        --architecture cnn \
        --data_files data/processed/encode_lentimpra_K562.h5 \
        --splits_file data/processed/splits.json \
        --output_file results/ablation/cnn_full_disentangle_seed42/tier1_results.json
"""

import argparse
import json

import h5py
import numpy as np
import torch

from .metrics import compute_all_metrics


def evaluate_tier1(model, data_file: str, splits_file: str, experiment_id: int) -> dict:
    """Evaluate on within-experiment test set."""
    with open(splits_file) as f:
        splits = json.load(f)

    split_info = splits[str(experiment_id)]
    test_indices = split_info["test"]

    with h5py.File(data_file, "r") as f:
        sequences = f["sequences"][:][test_indices]
        activities = f["activities"][:][test_indices]

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
    metrics["tier"] = 1
    metrics["experiment_id"] = experiment_id

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--data_files", nargs="+", required=True)
    parser.add_argument("--splits_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    # Load model
    import yaml
    from models.encoders import build_encoder
    from models.wrapper import DisentangleWrapper

    # TODO: Load config from saved checkpoint directory
    # model = ...
    # results = evaluate_tier1(model, ...)

    print("Tier 1 evaluation CLI - implement model loading from checkpoint")
