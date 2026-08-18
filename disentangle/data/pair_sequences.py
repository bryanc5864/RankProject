"""
Identify overlapping sequences across experiments for contrastive learning.

Supports exact string matching and genomic coordinate overlap.

Output: HDF5 mapping sequence_id -> [(experiment_id, index_in_experiment), ...]
for all sequences present in 2+ experiments.

Usage:
    python -m data.pair_sequences \
        --input_files data/processed/encode_lentimpra_K562.h5 data/processed/dream_K562.h5 \
        --output_file data/processed/paired_sequences.h5 \
        --method exact
"""

import argparse
from collections import defaultdict

import h5py
import numpy as np


def find_exact_pairs(input_files: list[str]) -> dict:
    """
    Find sequences appearing in multiple experiment files via exact string match.

    Returns:
        dict mapping sequence_string -> list of (experiment_id, within-experiment-index)
    """
    sequence_index = defaultdict(list)

    for exp_id, filepath in enumerate(input_files):
        with h5py.File(filepath, "r") as f:
            sequences = f["sequence_strings"][:]
            for idx, seq in enumerate(sequences):
                seq_str = seq.decode("utf-8") if isinstance(seq, bytes) else seq
                sequence_index[seq_str].append((exp_id, idx))

    # keep only sequences in 2+ experiments
    paired = {
        seq: locs
        for seq, locs in sequence_index.items()
        if len(set(loc[0] for loc in locs)) >= 2
    }

    print(f"Found {len(paired)} sequences in 2+ experiments")
    print(f"Total pairings: {sum(len(v) for v in paired.values())}")

    return paired


def find_coordinate_pairs(input_files: list[str], overlap_fraction: float = 0.5) -> dict:
    """
    Find sequences with overlapping genomic coordinates across experiments.
    Used for genome-wide assays where exact sequence matching won't work.
    """
    try:
        import pybedtools
    except ImportError:
        raise ImportError("pybedtools required for coordinate-based pairing")

    beds = []
    for exp_id, filepath in enumerate(input_files):
        with h5py.File(filepath, "r") as f:
            if "coordinates" not in f:
                print(f"Skipping {filepath}: no coordinate data")
                continue

            coords = f["coordinates"][:]
            entries = []
            for idx in range(len(coords)):
                chrom, start, end = coords[idx]
                if isinstance(chrom, bytes):
                    chrom = chrom.decode()
                entries.append(f"{chrom}\t{start}\t{end}\t{exp_id}:{idx}")

            bed = pybedtools.BedTool("\n".join(entries), from_string=True)
            beds.append((exp_id, bed))

    paired = defaultdict(list)
    for i in range(len(beds)):
        for j in range(i + 1, len(beds)):
            exp_i, bed_i = beds[i]
            exp_j, bed_j = beds[j]
            intersect = bed_i.intersect(
                bed_j, f=overlap_fraction, r=True, wa=True, wb=True
            )
            for hit in intersect:
                id_a = hit.fields[3]
                id_b = hit.fields[7]
                region_key = f"{hit.chrom}:{hit.start}-{hit.end}"
                paired[region_key].append(tuple(id_a.split(":")))
                paired[region_key].append(tuple(id_b.split(":")))

    return paired


def save_pairs(paired: dict, input_files: list[str], output_file: str):
    """Save paired sequences to HDF5 for efficient loading during training."""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, "w") as f:
        f.attrs["n_pairs"] = len(paired)
        f.attrs["input_files"] = [str(p) for p in input_files]

        pairs_group = f.create_group("pairs")

        for i, (seq, locations) in enumerate(paired.items()):
            pair_group = pairs_group.create_group(f"pair_{i}")
            exp_ids = [int(loc[0]) for loc in locations]
            indices = [int(loc[1]) for loc in locations]
            pair_group.create_dataset("exp_ids", data=np.array(exp_ids, dtype=np.int32))
            pair_group.create_dataset("indices", data=np.array(indices, dtype=np.int32))

    print(f"Saved {len(paired)} paired sequences to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find paired sequences across experiments")
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--method", choices=["exact", "coordinate"], default="exact")
    parser.add_argument("--overlap_fraction", type=float, default=0.5)
    args = parser.parse_args()

    if args.method == "exact":
        paired = find_exact_pairs(args.input_files)
    else:
        paired = find_coordinate_pairs(args.input_files, args.overlap_fraction)

    save_pairs(paired, args.input_files, args.output_file)
