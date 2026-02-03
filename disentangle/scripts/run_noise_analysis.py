#!/usr/bin/env python3
"""
Phase 2: Noise Characterization Analysis.

For models with 2 experiment normalizations (multi-experiment or C2-C5),
compares representations obtained through K562 norm vs HepG2 norm on
the same paired sequences.

Key analyses:
  1. CKA between experiment-specific representations
  2. UMAP colored by experiment norm path
  3. Experiment probe accuracy (can we predict which norm from reps?)

A good DISENTANGLE model should have:
  - High CKA (representations are similar regardless of norm)
  - UMAP clusters by biology, not by norm
  - Low experiment probe accuracy (representations are experiment-invariant)
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
from analysis.noise_characterization import compute_cka


def load_model(model_dir, device):
    """Load a trained model."""
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return None, None

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


def extract_reps_with_exp_id(model, sequences, experiment_id, device, batch_size=512):
    """Extract representations using a specific experiment normalization."""
    all_reps = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[i:i+batch_size]).to(device)
            reps = model.encode(batch, experiment_id=experiment_id)
            all_reps.append(reps.cpu().numpy())
    return np.concatenate(all_reps)


def main():
    parser = argparse.ArgumentParser(description="Noise characterization analysis")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--paired_data", default="data/processed/paired_K562_HepG2.h5")
    parser.add_argument("--output_dir", default="results/noise_analysis")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max samples for CKA (memory-limited)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load paired test sequences
    print("Loading paired test sequences...")
    with h5py.File(args.paired_data, "r") as f:
        splits = f["split"][:]
        test_mask = splits == 2
        sequences = f["sequences"][:][test_mask].astype(np.float32)
        k562_acts = f["k562_activities"][:][test_mask]
        hepg2_acts = f["hepg2_activities"][:][test_mask]
        consensus = f["consensus_ranks"][:][test_mask]

    # Subsample for CKA (N^2 memory)
    if len(sequences) > args.max_samples:
        idx = np.random.RandomState(42).choice(len(sequences), args.max_samples, replace=False)
        sequences = sequences[idx]
        k562_acts = k562_acts[idx]
        hepg2_acts = hepg2_acts[idx]
        consensus = consensus[idx]

    print(f"Using {len(sequences)} paired test sequences")

    # Find models with 2+ experiment norms
    results = []
    for name in sorted(os.listdir(args.results_dir)):
        model_dir = os.path.join(args.results_dir, name)
        if not os.path.isdir(model_dir) or not os.path.exists(
            os.path.join(model_dir, "best_model.pt")
        ):
            continue

        model, config = load_model(model_dir, device)
        if model is None:
            continue

        n_exp = model.n_experiments
        print(f"\n{'='*60}")
        print(f"Model: {name} (n_experiments={n_exp})")

        if n_exp < 2:
            # Single-experiment model: extract denoised reps only
            reps_denoised = extract_reps_with_exp_id(model, sequences, None, device)
            result = {
                "model": name,
                "architecture": config.get("architecture"),
                "condition": config.get("condition"),
                "n_experiments": n_exp,
                "cka_exp0_exp1": None,
                "cka_exp0_denoised": None,
                "probe_accuracy": None,
            }
        else:
            # Multi-experiment model: compare experiment-specific reps
            reps_exp0 = extract_reps_with_exp_id(model, sequences, 0, device)
            reps_exp1 = extract_reps_with_exp_id(model, sequences, 1, device)
            reps_denoised = extract_reps_with_exp_id(model, sequences, None, device)

            # CKA between experiment norms
            cka_01 = compute_cka(reps_exp0, reps_exp1, kernel="linear")
            cka_0d = compute_cka(reps_exp0, reps_denoised, kernel="linear")
            cka_1d = compute_cka(reps_exp1, reps_denoised, kernel="linear")

            print(f"  CKA(exp0, exp1):      {cka_01:.4f}")
            print(f"  CKA(exp0, denoised):  {cka_0d:.4f}")
            print(f"  CKA(exp1, denoised):  {cka_1d:.4f}")

            # Experiment probe: classify which norm was used
            from sklearn.linear_model import LogisticRegression
            X_probe = np.concatenate([reps_exp0, reps_exp1])
            y_probe = np.concatenate([np.zeros(len(reps_exp0)), np.ones(len(reps_exp1))])
            perm = np.random.RandomState(42).permutation(len(X_probe))
            split_idx = int(0.8 * len(X_probe))
            train_idx, test_idx = perm[:split_idx], perm[split_idx:]

            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_probe[train_idx], y_probe[train_idx])
            probe_acc = clf.score(X_probe[test_idx], y_probe[test_idx])
            print(f"  Experiment probe accuracy: {probe_acc:.4f}")
            print(f"  (lower = more experiment-invariant)")

            # Representation-activity correlation
            from scipy.stats import pearsonr
            # Use first PC of representations as a simple summary
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(reps_denoised).flatten()
            act_corr = float(spearmanr(pc1, consensus)[0])
            print(f"  PC1-consensus Spearman: {act_corr:.4f}")

            result = {
                "model": name,
                "architecture": config.get("architecture"),
                "condition": config.get("condition"),
                "n_experiments": n_exp,
                "cka_exp0_exp1": float(cka_01),
                "cka_exp0_denoised": float(cka_0d),
                "cka_exp1_denoised": float(cka_1d),
                "probe_accuracy": float(probe_acc),
                "pc1_consensus_corr": float(act_corr),
            }

        results.append(result)
        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = os.path.join(args.output_dir, "noise_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'Model':<45} {'CKA(0,1)':>9} {'Probe':>7} {'PC1-Cons':>9}")
    print("-" * 75)
    for r in results:
        cka = f"{r['cka_exp0_exp1']:.4f}" if r['cka_exp0_exp1'] is not None else "N/A"
        probe = f"{r['probe_accuracy']:.4f}" if r['probe_accuracy'] is not None else "N/A"
        pc1 = f"{r.get('pc1_consensus_corr', 'N/A')}"
        if isinstance(pc1, float):
            pc1 = f"{pc1:.4f}"
        print(f"{r['model']:<45} {cka:>9} {probe:>7} {pc1:>9}")


if __name__ == "__main__":
    main()
