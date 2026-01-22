#!/usr/bin/env python3
"""
Evaluate all trained models on CAGI5 saturation mutagenesis benchmark.

Usage:
    python scripts/evaluate_cagi5.py --results_dir results --output cagi5_results.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, kendalltau

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    DREAM_RNN, DREAM_RNN_SingleOutput, DREAM_RNN_DualHead,
    DREAM_RNN_DomainAdversarial, DREAM_RNN_BiasFactorized, DREAM_RNN_FullAdvanced
)


def one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA sequence to [4, seq_len]."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
        else:
            one_hot[:, i] = 0.25  # N or unknown
    return one_hot


def get_variant_sequence(ref_seq: str, ref_start: int, var_pos: int,
                         ref_allele: str, alt_allele: str, window: int = 230) -> str:
    """
    Create variant sequence centered on the variant position.

    Args:
        ref_seq: Full reference sequence
        ref_start: Genomic start position of ref_seq (1-based)
        var_pos: Genomic position of variant (1-based)
        ref_allele: Reference allele
        alt_allele: Alternate allele
        window: Output sequence length

    Returns:
        Variant sequence of length `window`
    """
    # Convert to 0-based index within ref_seq
    idx = var_pos - ref_start

    if idx < 0 or idx >= len(ref_seq):
        return None

    # Verify reference allele matches
    ref_in_seq = ref_seq[idx:idx + len(ref_allele)]
    if ref_in_seq.upper() != ref_allele.upper():
        # Try to handle edge cases
        pass

    # Create variant sequence
    var_seq = ref_seq[:idx] + alt_allele + ref_seq[idx + len(ref_allele):]

    # Extract window centered on variant
    center = idx + len(alt_allele) // 2
    half_window = window // 2
    start = center - half_window
    end = start + window

    # Handle boundaries
    if start < 0:
        pad_left = -start
        seq = 'N' * pad_left + var_seq[:window - pad_left]
    elif end > len(var_seq):
        pad_right = end - len(var_seq)
        seq = var_seq[start:] + 'N' * pad_right
    else:
        seq = var_seq[start:end]

    return seq[:window]


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device):
    """Load a model from checkpoint."""
    # Read config to determine model type
    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get('model', 'dream_rnn')
    n_bins = config.get('n_bins', 10)
    n_domains = config.get('n_domains', 10)

    # Create model
    if model_type == 'dream_rnn':
        n_outputs = n_bins if config.get('loss') == 'soft_classification' else 2
        model = DREAM_RNN(n_outputs=n_outputs)
    elif model_type == 'dream_rnn_single':
        model = DREAM_RNN_SingleOutput()
    elif model_type == 'dream_rnn_dual':
        model = DREAM_RNN_DualHead()
    elif model_type == 'dream_rnn_domain_adversarial':
        model = DREAM_RNN_DomainAdversarial(n_domains=n_domains)
    elif model_type == 'dream_rnn_bias_factorized':
        model = DREAM_RNN_BiasFactorized()
    elif model_type == 'dream_rnn_full_advanced':
        model = DREAM_RNN_FullAdvanced(n_domains=n_domains)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, config


def predict_batch(model, sequences: list, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Run model predictions on sequences."""
    # One-hot encode
    X = np.array([one_hot_encode(seq) for seq in sequences])

    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            out = model(batch)

            # Handle different output types
            if isinstance(out, tuple):
                out = out[0]  # Activity prediction
            if out.dim() > 1:
                if out.shape[1] > 1:
                    out = out[:, 0]  # First output is activity
                else:
                    out = out.squeeze(-1)

            predictions.extend(out.cpu().numpy().flatten())

    return np.array(predictions)


def evaluate_element(model, ref_data: dict, cagi5_df: pd.DataFrame,
                    device: torch.device, window: int = 230) -> dict:
    """Evaluate model on a single CAGI5 element."""
    ref_seq = ref_data['sequence']
    ref_start = ref_data['start']

    # Generate variant sequences
    sequences = []
    valid_indices = []

    for i, row in cagi5_df.iterrows():
        seq = get_variant_sequence(
            ref_seq, ref_start, row['Pos'], row['Ref'], row['Alt'], window
        )
        if seq and len(seq) == window:
            sequences.append(seq)
            valid_indices.append(i)

    if len(sequences) == 0:
        return None

    # Get predictions
    predictions = predict_batch(model, sequences, device)

    # Get ground truth
    ground_truth = cagi5_df.loc[valid_indices, 'Value'].values

    # Compute metrics
    spearman, _ = spearmanr(predictions, ground_truth)
    pearson, _ = pearsonr(predictions, ground_truth)
    kendall, _ = kendalltau(predictions, ground_truth)

    return {
        'n_variants': len(predictions),
        'spearman': spearman,
        'pearson': pearson,
        'kendall': kendall,
    }


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load reference sequences
    ref_path = Path(args.references)
    if not ref_path.exists():
        print(f"Reference file not found: {ref_path}")
        print("Run: python scripts/fetch_cagi5_references.py")
        sys.exit(1)

    with open(ref_path) as f:
        references = json.load(f)
    print(f"Loaded {len(references)} reference sequences")

    # Load CAGI5 data
    cagi5_dir = Path(args.cagi5_dir)
    cagi5_data = {}
    for tsv_file in cagi5_dir.glob("challenge_*.tsv"):
        element = tsv_file.stem.replace("challenge_", "")
        # Custom parsing: skip ## comment lines, use #Chrom line as header
        with open(tsv_file) as f:
            lines = f.readlines()
        # Find header line (starts with #Chrom)
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('#Chrom'):
                header_idx = i
                break
        if header_idx is None:
            print(f"Warning: Could not find header in {tsv_file}")
            continue
        # Parse with header
        header = lines[header_idx].lstrip('#').strip().split('\t')
        data_lines = [l.strip().split('\t') for l in lines[header_idx + 1:] if l.strip()]
        df = pd.DataFrame(data_lines, columns=header)
        # Convert numeric columns
        df['Pos'] = df['Pos'].astype(int)
        df['Value'] = df['Value'].astype(float)
        df['Confidence'] = df['Confidence'].astype(float)
        cagi5_data[element] = df
    print(f"Loaded {len(cagi5_data)} CAGI5 elements")

    # Find all experiment results
    results_dir = Path(args.results_dir)
    experiments = []
    for exp_dir in sorted(results_dir.iterdir()):
        if exp_dir.is_dir():
            checkpoint = exp_dir / "checkpoints" / "best_model.pth"
            config = exp_dir / "config.json"
            if checkpoint.exists() and config.exists():
                experiments.append({
                    'name': exp_dir.name.rsplit('_202', 1)[0],
                    'dir': exp_dir,
                    'checkpoint': checkpoint,
                    'config': config,
                })

    print(f"Found {len(experiments)} experiments to evaluate")
    print("=" * 80)

    # Evaluate each experiment
    all_results = []

    for exp in experiments:
        print(f"\nEvaluating: {exp['name']}")
        print("-" * 40)

        try:
            model, config = load_model(exp['checkpoint'], exp['config'], device)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue

        exp_results = {'experiment': exp['name']}
        element_spearman = []

        for element, df in cagi5_data.items():
            if element not in references:
                print(f"  {element}: No reference sequence")
                continue

            metrics = evaluate_element(model, references[element], df, device)
            if metrics is None:
                print(f"  {element}: No valid variants")
                continue

            print(f"  {element}: Spearman={metrics['spearman']:.4f}, "
                  f"Pearson={metrics['pearson']:.4f} (n={metrics['n_variants']})")

            exp_results[f"{element}_spearman"] = metrics['spearman']
            exp_results[f"{element}_pearson"] = metrics['pearson']
            exp_results[f"{element}_kendall"] = metrics['kendall']
            element_spearman.append(metrics['spearman'])

        # Mean across elements
        if element_spearman:
            exp_results['mean_spearman'] = np.mean(element_spearman)
            print(f"  MEAN Spearman: {exp_results['mean_spearman']:.4f}")

        all_results.append(exp_results)

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    results_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("CAGI5 SUMMARY - RANKED BY MEAN SPEARMAN")
    print("=" * 80)
    if 'mean_spearman' in results_df.columns:
        summary = results_df[['experiment', 'mean_spearman']].sort_values(
            'mean_spearman', ascending=False
        )
        for i, row in summary.iterrows():
            print(f"  {row['experiment']:<30} {row['mean_spearman']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on CAGI5")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing experiment results")
    parser.add_argument("--references", type=str, default="data/cagi5_references.json",
                       help="Path to reference sequences JSON")
    parser.add_argument("--cagi5_dir", type=str,
                       default="data/raw/dream_rnn_lentimpra/data/CAGI5",
                       help="Directory containing CAGI5 TSV files")
    parser.add_argument("--output", type=str, default="results/cagi5_evaluation.csv",
                       help="Output CSV file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    main(parser.parse_args())
