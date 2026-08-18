"""
Create experiment-aware train/validation/test splits.

Key principles:
- Held-out experiments (D5 ATAC-STARR-seq, D8 CAGI5, D2 HepG2) are never used in training.
- Paired sequences (for contrastive learning) are forced into the training set.
- Within each training experiment: 80% train / 10% val / 10% test.

Usage:
    python -m data.splits \
        --training_files data/processed/encode_lentimpra_K562.h5 data/processed/dream_K562.h5 \
        --paired_file data/processed/paired_sequences.h5 \
        --output_file data/processed/splits.json \
        --seed 42
"""

import argparse
import json

import h5py
import numpy as np


def create_splits(
    training_files: list[str],
    paired_file: str,
    output_file: str,
    seed: int = 42,
):
    np.random.seed(seed)

    # load paired sequence indices (must all go to train)
    paired_indices: dict[int, set[int]] = {}
    if paired_file:
        with h5py.File(paired_file, "r") as f:
            for pair_key in f["pairs"]:
                pair = f["pairs"][pair_key]
                exp_ids = pair["exp_ids"][:]
                indices = pair["indices"][:]
                for eid, idx in zip(exp_ids, indices):
                    eid = int(eid)
                    if eid not in paired_indices:
                        paired_indices[eid] = set()
                    paired_indices[eid].add(int(idx))

    splits = {}

    for exp_id, filepath in enumerate(training_files):
        with h5py.File(filepath, "r") as f:
            n_sequences = len(f["activities"])

        all_indices = set(range(n_sequences))
        forced_train = paired_indices.get(exp_id, set())
        remaining = list(all_indices - forced_train)
        np.random.shuffle(remaining)

        n_val = int(0.1 * n_sequences)
        n_test = int(0.1 * n_sequences)

        val_indices = remaining[:n_val]
        test_indices = remaining[n_val : n_val + n_test]
        train_indices = list(forced_train) + remaining[n_val + n_test :]

        splits[str(exp_id)] = {
            "train": sorted(train_indices),
            "val": sorted(val_indices),
            "test": sorted(test_indices),
            "n_total": n_sequences,
            "n_paired_in_train": len(forced_train),
            "filepath": filepath,
        }

        print(f"Experiment {exp_id} ({filepath}):")
        print(f"  Train: {len(train_indices)} ({len(forced_train)} paired)")
        print(f"  Val:   {len(val_indices)}")
        print(f"  Test:  {len(test_indices)}")

    with open(output_file, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSplits saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create experiment-aware data splits")
    parser.add_argument("--training_files", nargs="+", required=True)
    parser.add_argument("--paired_file", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_splits(args.training_files, args.paired_file, args.output_file, args.seed)
