"""
Standardize all datasets into a common HDF5 format.

Output format per file:
  /sequences:        one-hot encoded DNA [N, L, 4] (uint8)
  /activities:       measured activity values [N] (float32)
  /experiment_id:    integer experiment label [N] (int32)
  /sequence_strings: raw DNA strings [N] (variable-length)
  /replicate_std:    per-sequence replicate std [N] (float32, optional)
  /coordinates:      genomic positions [N, 3] (optional, for coordinate-based pairing)

Usage:
    python -m data.preprocess --dataset lentimpra --input_dir data/raw/encode_lentimpra/K562/ \
        --output_file data/processed/encode_lentimpra_K562.h5 --experiment_id 0
"""

import os
import argparse

import h5py
import numpy as np
import pandas as pd


def one_hot_encode(sequence: str) -> np.ndarray:
    """Convert DNA sequence to one-hot encoding in ACGT order."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((len(sequence), 4), dtype=np.uint8)
    for i, base in enumerate(sequence.upper()):
        if base in mapping:
            encoded[i, mapping[base]] = 1
    return encoded


def preprocess_lentimpra(input_dir: str, output_file: str, experiment_id: int):
    """Preprocess lentiMPRA data from ENCODE TSV files."""
    dfs = []
    for f in os.listdir(input_dir):
        if f.endswith(".tsv"):
            df = pd.read_csv(os.path.join(input_dir, f), sep="\t")
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No TSV files found in {input_dir}")

    data = pd.concat(dfs, ignore_index=True)

    # filter by minimum barcode count if column exists
    if "barcode_count" in data.columns:
        data = data[data["barcode_count"] >= 5].reset_index(drop=True)

    sequences = np.array([one_hot_encode(seq) for seq in data["sequence"]])
    activities = data["activity"].values.astype(np.float32)

    # Log-transform if all positive
    if activities.min() > 0:
        activities = np.log2(activities)

    _save_h5(output_file, sequences, activities, data["sequence"].values, experiment_id, data)


def preprocess_dream(input_dir: str, output_file: str, experiment_id: int,
                     cell_type: str = "K562"):
    """
    Preprocess DREAM Challenge lentiMPRA data.

    Args:
        input_dir: Directory containing HDF5 files
        output_file: Output path for DISENTANGLE-format HDF5
        experiment_id: Integer experiment label
        cell_type: 'K562' or 'HepG2'
    """
    filename = f"lentiMPRA_{cell_type}_activity_and_aleatoric_data.h5"
    h5_candidates = [
        os.path.join(input_dir, "data", filename),
        os.path.join(input_dir, filename),
    ]
    for h5_path in h5_candidates:
        if os.path.exists(h5_path):
            print(f"Using existing HDF5: {h5_path}")
            _preprocess_dream_h5(h5_path, output_file, experiment_id)
            return

    raise FileNotFoundError(
        f"No recognized DREAM data found in {input_dir}. "
        f"Expected {filename}"
    )


def _preprocess_dream_h5(h5_path: str, output_file: str, experiment_id: int):
    """Convert existing RankProject DREAM HDF5 to DISENTANGLE format.

    The source format uses capitalized split names (Train/Val/Test) with:
      X: [N, 230, 4] one-hot encoded sequences (float64)
      y: [N, 2] where col 0 = activity, col 1 = aleatoric uncertainty
    """
    split_names = {
        "Train": "train", "Val": "val", "Test": "test",
        "train": "train", "val": "val", "test": "test",
    }

    all_seqs = []
    all_acts = []
    all_stds = []
    split_labels = []

    with h5py.File(h5_path, "r") as src:
        for src_name, split_label in [("Train", 0), ("Val", 1), ("Test", 2),
                                       ("train", 0), ("val", 1), ("test", 2)]:
            if src_name not in src:
                continue
            grp = src[src_name]
            seqs = grp["X"][:]  # [N, 230, 4]
            y = grp["y"][:]     # [N, 2]: activity, aleatoric_std
            all_seqs.append(seqs)
            all_acts.append(y[:, 0])
            all_stds.append(y[:, 1])
            split_labels.append(np.full(len(seqs), split_label, dtype=np.int32))

    sequences = np.concatenate(all_seqs, axis=0)    # [N, 230, 4]
    activities = np.concatenate(all_acts, axis=0).astype(np.float32)
    replicate_stds = np.concatenate(all_stds, axis=0).astype(np.float32)
    splits = np.concatenate(split_labels, axis=0)    # 0=train, 1=val, 2=test

    # save with split information and replicate_std
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("sequences", data=sequences.astype(np.float32), compression="gzip")
        f.create_dataset("activities", data=activities)
        f.create_dataset("replicate_std", data=replicate_stds)
        f.create_dataset("experiment_id",
                         data=np.full(len(activities), experiment_id, dtype=np.int32))
        f.create_dataset("split", data=splits)

    print(f"Saved {len(activities)} sequences to {output_file}")
    print(f"  Sequence shape: {sequences.shape}")
    print(f"  Activity range: [{activities.min():.3f}, {activities.max():.3f}]")
    print(f"  Split counts: train={np.sum(splits==0)}, val={np.sum(splits==1)}, test={np.sum(splits==2)}")


def preprocess_starrseq(input_dir: str, output_file: str, experiment_id: int,
                        bin_size: int = 200):
    """
    Preprocess STARR-seq data (genome-wide).

    Bins the genome into fixed-size windows, computes activity per window,
    and extracts sequences from the reference genome.
    """
    # TODO: Implement based on specific file format from GEO
    raise NotImplementedError(
        "STARR-seq preprocessing requires specific file format handling. "
        "Implement based on the downloaded data format."
    )


def preprocess_cagi5(input_dir: str, output_file: str, experiment_id: int = 6):
    """
    Preprocess CAGI5 saturation mutagenesis data.

    Creates reference and alternate one-hot encodings for each variant.
    Reuses data from the existing RankProject if available.
    """
    # check for existing RankProject CAGI5 data
    cagi5_dir = os.path.join(input_dir)
    if not os.path.isdir(cagi5_dir):
        raise FileNotFoundError(f"CAGI5 directory not found: {cagi5_dir}")

    # TODO: Convert existing CAGI5 data to DISENTANGLE format
    # the existing RankProject already has CAGI5 evaluation code that can be referenced
    raise NotImplementedError(
        "CAGI5 preprocessing should leverage existing RankProject data at ../data/cagi5/. "
        "See ../src/evaluation/cagi5_eval.py for the existing loading logic."
    )


def _save_h5(output_file: str, sequences: np.ndarray, activities: np.ndarray,
             seq_strings: np.ndarray, experiment_id: int, data: pd.DataFrame = None):
    """Save preprocessed data to HDF5."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, "w") as f:
        f.create_dataset("sequences", data=sequences.astype(np.uint8), compression="gzip")
        f.create_dataset("activities", data=activities)
        f.create_dataset(
            "experiment_id",
            data=np.full(len(activities), experiment_id, dtype=np.int32),
        )
        f.create_dataset(
            "sequence_strings",
            data=seq_strings.astype("S"),
            compression="gzip",
        )

        if data is not None and "replicate_std" in data.columns:
            f.create_dataset(
                "replicate_std",
                data=data["replicate_std"].values.astype(np.float32),
            )

    print(f"Saved {len(activities)} sequences to {output_file}")
    print(f"  Activity range: [{activities.min():.2f}, {activities.max():.2f}]")
    print(f"  Sequence shape: {sequences.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets for DISENTANGLE")
    parser.add_argument(
        "--dataset",
        choices=["lentimpra", "dream", "starrseq", "cagi5", "kircher"],
        required=True,
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--experiment_id", type=int, required=True)
    parser.add_argument("--cell_type", default="K562",
                        help="Cell type for DREAM data (K562 or HepG2)")
    args = parser.parse_args()

    if args.dataset == "dream":
        preprocess_dream(args.input_dir, args.output_file, args.experiment_id,
                         cell_type=args.cell_type)
    elif args.dataset == "lentimpra":
        preprocess_lentimpra(args.input_dir, args.output_file, args.experiment_id)
    elif args.dataset == "starrseq":
        preprocess_starrseq(args.input_dir, args.output_file, args.experiment_id)
    elif args.dataset == "cagi5":
        preprocess_cagi5(args.input_dir, args.output_file, args.experiment_id)
