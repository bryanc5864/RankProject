#!/usr/bin/env python3
"""
Generate synthetic noise datasets for E3 experiments.

Takes K562 data and creates paired pseudo-experiments with controlled noise:
  Exp A: clean activity (ground truth)
  Exp B: noisy activity (GC-dependent / random offset / multiplicative)

This allows testing whether DISENTANGLE can recover clean signal when
the noise structure is known.

Usage:
    python scripts/generate_synthetic_noise.py \
        --input data/processed/dream_K562.h5 \
        --output_dir data/processed/synthetic_noise/
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def compute_gc_content(sequences):
    """Compute GC content per sequence from one-hot [N, L, 4] (order: A, C, G, T)."""
    c_count = sequences[:, :, 1].sum(axis=1)  # C
    g_count = sequences[:, :, 2].sum(axis=1)  # G
    seq_len = sequences.shape[1]
    return (c_count + g_count) / seq_len


def add_gc_dependent_noise(activities, gc_content, noise_scale=0.5, seed=42):
    """
    GC-dependent noise: noise magnitude scales with GC content.
    Sequences with extreme GC get more noise.
    """
    rng = np.random.RandomState(seed)
    # Scale noise by how far GC is from 0.5
    gc_deviation = np.abs(gc_content - 0.5) * 2  # [0, 1]
    noise_std = noise_scale * (0.5 + gc_deviation)
    noise = rng.normal(0, 1, size=len(activities)) * noise_std
    return activities + noise


def add_random_offset_noise(activities, noise_scale=0.5, seed=42):
    """
    Random offset noise: additive Gaussian noise independent of sequence.
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_scale, size=len(activities))
    return activities + noise


def add_multiplicative_noise(activities, noise_scale=0.3, seed=42):
    """
    Multiplicative noise: activity * (1 + epsilon).
    Preserves sign structure but distorts magnitudes.
    """
    rng = np.random.RandomState(seed)
    multiplier = 1.0 + rng.normal(0, noise_scale, size=len(activities))
    multiplier = np.clip(multiplier, 0.1, 5.0)  # prevent sign flips
    return activities * multiplier


def generate_synthetic_dataset(input_file, output_dir, noise_type, noise_scale=0.5):
    """
    Generate a synthetic noise dataset.

    Creates:
      - Single-experiment HDF5 for Exp A (clean) and Exp B (noisy)
      - Paired HDF5 with both measurements
    """
    noise_funcs = {
        "gc_dependent": add_gc_dependent_noise,
        "random_offset": add_random_offset_noise,
        "multiplicative": add_multiplicative_noise,
    }

    if noise_type not in noise_funcs:
        raise ValueError(f"Unknown noise type: {noise_type}")

    print(f"\nGenerating {noise_type} noise dataset (scale={noise_scale})...")

    with h5py.File(input_file, "r") as f:
        sequences = f["sequences"][:]
        activities = f["activities"][:]
        splits = f["split"][:]
        exp_ids = f["experiment_id"][:]
        stds = f["replicate_std"][:] if "replicate_std" in f else np.zeros_like(activities)

    n = len(sequences)
    gc_content = compute_gc_content(sequences.astype(np.float32))

    # Generate noisy activities
    if noise_type == "gc_dependent":
        noisy_activities = add_gc_dependent_noise(activities, gc_content, noise_scale)
    elif noise_type == "random_offset":
        noisy_activities = add_random_offset_noise(activities, noise_scale)
    elif noise_type == "multiplicative":
        noisy_activities = add_multiplicative_noise(activities, noise_scale)

    subdir = os.path.join(output_dir, noise_type)
    os.makedirs(subdir, exist_ok=True)

    # Write clean experiment (Exp A, experiment_id=0)
    clean_path = os.path.join(subdir, "synthetic_clean.h5")
    with h5py.File(clean_path, "w") as f:
        f.create_dataset("sequences", data=sequences, compression="gzip")
        f.create_dataset("activities", data=activities.astype(np.float32))
        f.create_dataset("split", data=splits)
        f.create_dataset("experiment_id", data=np.zeros(n, dtype=np.int32))
        f.create_dataset("replicate_std", data=stds.astype(np.float32))
    print(f"  Clean: {clean_path} ({n} sequences)")

    # Write noisy experiment (Exp B, experiment_id=1)
    noisy_path = os.path.join(subdir, "synthetic_noisy.h5")
    with h5py.File(noisy_path, "w") as f:
        f.create_dataset("sequences", data=sequences, compression="gzip")
        f.create_dataset("activities", data=noisy_activities.astype(np.float32))
        f.create_dataset("split", data=splits)
        f.create_dataset("experiment_id", data=np.ones(n, dtype=np.int32))
        f.create_dataset("replicate_std", data=(stds + noise_scale).astype(np.float32))
    print(f"  Noisy: {noisy_path} ({n} sequences)")

    # Write paired dataset
    # Compute consensus ranks from clean and noisy
    from scipy.stats import rankdata
    clean_ranks = rankdata(activities) / n
    noisy_ranks = rankdata(noisy_activities) / n
    consensus_ranks = (clean_ranks + noisy_ranks) / 2

    paired_path = os.path.join(subdir, "synthetic_paired.h5")
    with h5py.File(paired_path, "w") as f:
        f.create_dataset("sequences", data=sequences, compression="gzip")
        f.create_dataset("k562_activities", data=activities.astype(np.float32))
        f.create_dataset("hepg2_activities", data=noisy_activities.astype(np.float32))
        f.create_dataset("consensus_ranks", data=consensus_ranks.astype(np.float32))
        f.create_dataset("split", data=splits)
        f.create_dataset("k562_stds", data=stds.astype(np.float32))
        f.create_dataset("hepg2_stds", data=(stds + noise_scale).astype(np.float32))
    print(f"  Paired: {paired_path} ({n} sequences)")

    # Report noise statistics
    actual_noise = noisy_activities - activities
    train_mask = splits == 0
    corr_clean_noisy = float(np.corrcoef(activities[train_mask],
                                          noisy_activities[train_mask])[0, 1])
    print(f"  Noise stats: mean={actual_noise.mean():.4f}, std={actual_noise.std():.4f}")
    print(f"  Clean-noisy correlation (train): {corr_clean_noisy:.4f}")

    return clean_path, noisy_path, paired_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic noise datasets")
    parser.add_argument("--input", default="data/processed/dream_K562.h5",
                        help="Input K562 HDF5 file")
    parser.add_argument("--output_dir", default="data/processed/synthetic_noise/",
                        help="Output directory")
    parser.add_argument("--noise_scale", type=float, default=0.5,
                        help="Base noise scale")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for noise_type in ["gc_dependent", "random_offset", "multiplicative"]:
        generate_synthetic_dataset(
            args.input, args.output_dir, noise_type, args.noise_scale
        )

    print("\nAll synthetic datasets generated.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
