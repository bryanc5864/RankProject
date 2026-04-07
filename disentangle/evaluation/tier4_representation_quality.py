"""
Tier 4: Representation Quality Assessment.

Tests whether representations are experiment-invariant.

Metrics:
1. Experiment probe accuracy (should be near chance for DISENTANGLE)
2. Activity probe R^2 (should be maintained)
3. Batch-activity feature overlap (should be low for DISENTANGLE)

Usage:
    python -m evaluation.tier4_representation_quality \
        --model_path results/ablation/cnn_full_disentangle_seed42/best.pt \
        --architecture cnn \
        --data_files data/processed/encode_lentimpra_K562.h5 data/processed/dream_K562.h5 \
        --output_file results/ablation/cnn_full_disentangle_seed42/tier4_results.json
"""

import argparse
import json

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def extract_representations(model, sequences: np.ndarray) -> np.ndarray:
    """Extract penultimate-layer representations."""
    model.eval()
    representations = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(
                sequences[i:i + batch_size], dtype=torch.float32
            ).cuda()
            if hasattr(model, "encode"):
                reps = model.encode(batch)
            else:
                reps = model.base_model.encode(batch)
            representations.append(reps.cpu().numpy())

    return np.concatenate(representations, axis=0)


def run_probes(representations: np.ndarray, experiment_ids: np.ndarray,
               activities: np.ndarray) -> dict:
    """Run linear probing analysis on representations."""
    scaler = StandardScaler()
    X = scaler.fit_transform(representations)

    results = {}

    # Probe 1: Experiment ID prediction (classification)
    n_classes = len(np.unique(experiment_ids))
    if n_classes > 1:
        probe_exp = LogisticRegression(max_iter=1000, C=1.0)
        scores_exp = cross_val_score(probe_exp, X, experiment_ids, cv=5, scoring="accuracy")
        results["experiment_probe_accuracy"] = float(scores_exp.mean())
        results["experiment_probe_std"] = float(scores_exp.std())
        results["experiment_probe_chance"] = 1.0 / n_classes

    # Probe 2: Activity prediction (regression)
    probe_act = Ridge(alpha=1.0)
    scores_act = cross_val_score(probe_act, X, activities, cv=5, scoring="r2")
    results["activity_probe_r2"] = float(scores_act.mean())
    results["activity_probe_std"] = float(scores_act.std())

    # Probe 3: Batch-activity feature overlap
    if n_classes > 1:
        probe_exp_full = LogisticRegression(max_iter=1000, C=1.0)
        probe_exp_full.fit(X, experiment_ids)
        batch_importance = np.abs(probe_exp_full.coef_).mean(axis=0)

        probe_act_full = Ridge(alpha=1.0)
        probe_act_full.fit(X, activities)
        activity_importance = np.abs(probe_act_full.coef_)

        overlap = float(np.corrcoef(batch_importance, activity_importance)[0, 1])
        results["batch_activity_overlap"] = overlap

    results["tier"] = 4
    return results


def evaluate_tier4(model, data_files: list[str]) -> dict:
    """Full Tier 4 evaluation."""
    all_reps = []
    all_exp_ids = []
    all_activities = []

    for exp_id, filepath in enumerate(data_files):
        with h5py.File(filepath, "r") as f:
            sequences = f["sequences"][:]
            activities = f["activities"][:]

        reps = extract_representations(model, sequences)

        all_reps.append(reps)
        all_exp_ids.extend([exp_id] * len(reps))
        all_activities.extend(activities)

    all_reps = np.concatenate(all_reps)
    all_exp_ids = np.array(all_exp_ids)
    all_activities = np.array(all_activities)

    return run_probes(all_reps, all_exp_ids, all_activities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--data_files", nargs="+", required=True)
    parser.add_argument("--paired_sequences", default=None)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    print("Tier 4 evaluation CLI - implement model loading from checkpoint")
