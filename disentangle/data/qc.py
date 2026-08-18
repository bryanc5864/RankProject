"""
Quality control checks for preprocessed data.

Usage:
    python -m data.qc --input_file data/processed/encode_lentimpra_K562.h5
"""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def run_qc(input_file: str):
    with h5py.File(input_file, "r") as f:
        sequences = f["sequences"][:]
        activities = f["activities"][:]

    print(f"=== QC Report: {input_file} ===")

    # check 1: sequence validity
    row_sums = sequences.sum(axis=2)
    assert (row_sums <= 1).all(), "Invalid one-hot encoding: multiple bases at same position"
    n_ambiguous = (row_sums == 0).sum()
    pct_ambiguous = n_ambiguous / row_sums.size * 100
    print(f"  Sequences: {len(sequences)}")
    print(f"  Sequence length: {sequences.shape[1]}")
    print(f"  Ambiguous bases (N): {pct_ambiguous:.2f}%")
    assert pct_ambiguous < 5, f"Too many ambiguous bases ({pct_ambiguous:.1f}% > 5%)"

    # check 2: activity distribution
    print(f"  Activity range: [{activities.min():.3f}, {activities.max():.3f}]")
    print(f"  Activity mean: {activities.mean():.3f}, std: {activities.std():.3f}")
    assert not np.any(np.isnan(activities)), "NaN values in activities"
    assert not np.any(np.isinf(activities)), "Inf values in activities"

    # check 3: GC content
    gc_content = (sequences[:, :, 1] + sequences[:, :, 2]).mean(axis=1)
    print(f"  GC content: mean={gc_content.mean():.3f}, std={gc_content.std():.3f}")

    # check 4: activity-GC correlation
    r, p = pearsonr(gc_content, activities)
    print(f"  Activity-GC correlation: r={r:.3f}, p={p:.2e}")
    if abs(r) > 0.3:
        print(f"  WARNING: Strong GC-activity correlation (r={r:.3f}). Potential confound.")

    # QC plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(activities, bins=100, alpha=0.7)
    axes[0].set_xlabel("Activity")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Activity Distribution")

    axes[1].hist(gc_content, bins=50, alpha=0.7)
    axes[1].set_xlabel("GC Content")
    axes[1].set_title("GC Content Distribution")

    axes[2].scatter(gc_content[::100], activities[::100], alpha=0.1, s=1)
    axes[2].set_xlabel("GC Content")
    axes[2].set_ylabel("Activity")
    axes[2].set_title(f"GC vs Activity (r={r:.3f})")

    plt.tight_layout()
    plot_path = input_file.replace(".h5", "_qc.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  QC plots saved to {plot_path}")
    print("  === QC PASSED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QC on preprocessed data")
    parser.add_argument("--input_file", required=True)
    args = parser.parse_args()
    run_qc(args.input_file)
