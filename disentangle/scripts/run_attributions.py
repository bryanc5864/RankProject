#!/usr/bin/env python3
"""
Phase 6: Attribution Analysis.

Computes Integrated Gradients for baseline and DISENTANGLE models,
then compares attribution patterns to identify noise-learned features.

Usage:
    python scripts/run_attributions.py --baseline_dir results/bilstm_baseline_mse_seed42 \
        --disentangle_dir results/bilstm_full_disentangle_seed42 \
        --output_dir results/attributions
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoders import build_encoder
from models.wrapper import DisentangleWrapper
from analysis.attribution_analysis import (
    compute_attributions,
    compare_attributions_across_models,
    compute_noise_attributions,
)


def load_model(model_dir, device):
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "best_model.pt")

    with open(config_path) as f:
        config = json.load(f)

    architecture = config["architecture"]
    n_data_files = len(config.get("data_files", [1]))
    has_paired = config.get("paired_data") is not None
    n_experiments = max(n_data_files, 2 if has_paired else 1)

    encoder = build_encoder(architecture, config)
    model = DisentangleWrapper(encoder, n_experiments=n_experiments, config=config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Attribution analysis")
    parser.add_argument("--baseline_dir", required=True,
                        help="Path to baseline model results directory")
    parser.add_argument("--disentangle_dir", required=True,
                        help="Path to DISENTANGLE model results directory")
    parser.add_argument("--data", default="data/processed/dream_K562.h5",
                        help="Data file for attribution computation")
    parser.add_argument("--output_dir", default="results/attributions")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_sequences", type=int, default=500,
                        help="Number of test sequences for attributions")
    parser.add_argument("--n_steps", type=int, default=50,
                        help="Integration steps for IG")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test sequences
    print("Loading test sequences...")
    with h5py.File(args.data, "r") as f:
        splits = f["split"][:]
        test_mask = splits == 2
        sequences = f["sequences"][:][test_mask].astype(np.float32)
        activities = f["activities"][:][test_mask]

    # Select high-activity sequences (most informative for attribution)
    top_idx = np.argsort(activities)[-args.n_sequences:]
    sequences = sequences[top_idx]
    activities = activities[top_idx]
    print(f"Selected {len(sequences)} high-activity test sequences")

    # Load models
    print(f"\nLoading baseline model from {args.baseline_dir}")
    baseline_model, baseline_config = load_model(args.baseline_dir, device)

    print(f"Loading DISENTANGLE model from {args.disentangle_dir}")
    disentangle_model, disentangle_config = load_model(args.disentangle_dir, device)

    # Compute attributions
    print("\nComputing baseline attributions...")
    baseline_attrs = compute_attributions(
        baseline_model, sequences, batch_size=64, n_steps=args.n_steps
    )

    print("Computing DISENTANGLE attributions...")
    disentangle_attrs = compute_attributions(
        disentangle_model, sequences, batch_size=64, n_steps=args.n_steps
    )

    # Compare attributions
    print("\nComparing attribution patterns...")
    correlations = compare_attributions_across_models(baseline_attrs, disentangle_attrs)
    print(f"  Per-sequence attribution correlation: "
          f"mean={correlations.mean():.4f}, median={np.median(correlations):.4f}")

    # Compute noise attributions
    noise_ratio = compute_noise_attributions(baseline_attrs, disentangle_attrs)
    print(f"  Noise attribution ratio: mean={noise_ratio.mean():.4f}")

    # Attribution magnitude analysis
    baseline_mag = np.abs(baseline_attrs).sum(axis=2).mean(axis=0)  # [L]
    disentangle_mag = np.abs(disentangle_attrs).sum(axis=2).mean(axis=0)  # [L]

    # Save results
    results = {
        "baseline_model": os.path.basename(args.baseline_dir),
        "disentangle_model": os.path.basename(args.disentangle_dir),
        "n_sequences": len(sequences),
        "n_steps": args.n_steps,
        "mean_attribution_correlation": float(correlations.mean()),
        "median_attribution_correlation": float(np.median(correlations)),
        "std_attribution_correlation": float(correlations.std()),
        "mean_noise_ratio": float(noise_ratio.mean()),
        "baseline_mean_magnitude": float(np.abs(baseline_attrs).mean()),
        "disentangle_mean_magnitude": float(np.abs(disentangle_attrs).mean()),
    }

    with open(os.path.join(args.output_dir, "attribution_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save raw data for visualization
    np.savez_compressed(
        os.path.join(args.output_dir, "attributions.npz"),
        baseline_attrs=baseline_attrs,
        disentangle_attrs=disentangle_attrs,
        correlations=correlations,
        noise_ratio=noise_ratio,
        baseline_positional_mag=baseline_mag,
        disentangle_positional_mag=disentangle_mag,
        activities=activities,
    )

    print(f"\nResults saved to {args.output_dir}/")
    print("\nSummary:")
    print(f"  Attribution correlation: {correlations.mean():.4f} +/- {correlations.std():.4f}")
    print(f"  Noise ratio: {noise_ratio.mean():.4f}")
    print(f"  Baseline magnitude: {np.abs(baseline_attrs).mean():.6f}")
    print(f"  DISENTANGLE magnitude: {np.abs(disentangle_attrs).mean():.6f}")


if __name__ == "__main__":
    main()
