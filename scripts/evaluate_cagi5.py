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

# Cell-type to CAGI5 element mapping
K562_ELEMENTS = ['GP1BB', 'HBB', 'HBG1', 'PKLR']
HEPG2_ELEMENTS = ['F9', 'LDLR', 'SORT1']


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
        n_outputs = n_bins if config.get('loss') == 'soft_classification' else 1
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


def get_ref_sequence(ref_seq: str, ref_start: int, var_pos: int, window: int = 230) -> str:
    """
    Get reference sequence centered on the variant position.
    """
    # Convert to 0-based index within ref_seq
    idx = var_pos - ref_start

    if idx < 0 or idx >= len(ref_seq):
        return None

    # Extract window centered on position
    half_window = window // 2
    start = idx - half_window
    end = start + window

    # Handle boundaries
    if start < 0:
        pad_left = -start
        seq = 'N' * pad_left + ref_seq[:window - pad_left]
    elif end > len(ref_seq):
        pad_right = end - len(ref_seq)
        seq = ref_seq[start:] + 'N' * pad_right
    else:
        seq = ref_seq[start:end]

    return seq[:window]


def evaluate_element(model, ref_data: dict, cagi5_df: pd.DataFrame,
                    device: torch.device, window: int = 230,
                    min_confidence: float = None) -> dict:
    """
    Evaluate model on a single CAGI5 element.

    Uses Alt - Ref approach: variant effect = model(Alt) - model(Ref)

    Args:
        min_confidence: If set, filter variants to those with Confidence >= this value
    """
    # Apply confidence filtering
    if min_confidence is not None:
        cagi5_df = cagi5_df[cagi5_df['Confidence'] >= min_confidence].copy()
        if len(cagi5_df) == 0:
            return None

    ref_seq = ref_data['sequence']
    ref_start = ref_data['start']

    # Generate both ref and alt sequences for each variant
    alt_sequences = []
    ref_sequences = []
    valid_indices = []

    for i, row in cagi5_df.iterrows():
        alt_seq = get_variant_sequence(
            ref_seq, ref_start, row['Pos'], row['Ref'], row['Alt'], window
        )
        ref_seq_window = get_ref_sequence(
            ref_seq, ref_start, row['Pos'], window
        )

        if alt_seq and ref_seq_window and len(alt_seq) == window and len(ref_seq_window) == window:
            alt_sequences.append(alt_seq)
            ref_sequences.append(ref_seq_window)
            valid_indices.append(i)

    if len(alt_sequences) == 0:
        return None

    # Get predictions for both alt and ref
    alt_predictions = predict_batch(model, alt_sequences, device)
    ref_predictions = predict_batch(model, ref_sequences, device)

    # Variant effect = Alt - Ref
    predictions = alt_predictions - ref_predictions

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


def infer_cell_type(config: dict) -> str:
    """Infer cell type from experiment config data path."""
    data_path = config.get('data', '')
    if 'HepG2' in data_path or 'hepg2' in data_path.lower():
        return 'HepG2'
    return 'K562'


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

        cell_type = infer_cell_type(config)
        exp_results = {'experiment': exp['name'], 'cell_type': cell_type}

        # Determine matched elements for this model's cell type
        if cell_type == 'K562':
            matched_elements = K562_ELEMENTS
        else:
            matched_elements = HEPG2_ELEMENTS

        all_spearman = []
        all_pearson = []
        highconf_spearman = []
        highconf_pearson = []
        matched_all_spearman = []
        matched_highconf_spearman = []

        for element, df in cagi5_data.items():
            if element not in references:
                print(f"  {element}: No reference sequence")
                continue

            # Evaluate with all SNPs
            metrics_all = evaluate_element(model, references[element], df, device)
            if metrics_all is None:
                print(f"  {element}: No valid variants")
                continue

            # Evaluate with high-confidence SNPs (>= 0.1)
            metrics_hc = evaluate_element(
                model, references[element], df, device, min_confidence=0.1
            )

            print(f"  {element}: All Spearman={metrics_all['spearman']:.4f} (n={metrics_all['n_variants']})", end="")
            if metrics_hc:
                print(f", HC Spearman={metrics_hc['spearman']:.4f} (n={metrics_hc['n_variants']})")
            else:
                print(f", HC: no variants")

            # Store all-SNPs metrics
            exp_results[f"all_{element}_spearman"] = metrics_all['spearman']
            exp_results[f"all_{element}_pearson"] = metrics_all['pearson']
            exp_results[f"all_{element}_kendall"] = metrics_all['kendall']
            exp_results[f"all_{element}_n"] = metrics_all['n_variants']
            all_spearman.append(metrics_all['spearman'])
            all_pearson.append(metrics_all['pearson'])

            # Store high-confidence metrics
            if metrics_hc:
                exp_results[f"highconf_{element}_spearman"] = metrics_hc['spearman']
                exp_results[f"highconf_{element}_pearson"] = metrics_hc['pearson']
                exp_results[f"highconf_{element}_kendall"] = metrics_hc['kendall']
                exp_results[f"highconf_{element}_n"] = metrics_hc['n_variants']
                highconf_spearman.append(metrics_hc['spearman'])
                highconf_pearson.append(metrics_hc['pearson'])

            # Track matched elements
            if element in matched_elements:
                matched_all_spearman.append(metrics_all['spearman'])
                if metrics_hc:
                    matched_highconf_spearman.append(metrics_hc['spearman'])

        # Mean across ALL elements
        if all_spearman:
            exp_results['all_mean_spearman'] = np.mean(all_spearman)
            exp_results['all_mean_pearson'] = np.mean(all_pearson)
        if highconf_spearman:
            exp_results['highconf_mean_spearman'] = np.mean(highconf_spearman)
            exp_results['highconf_mean_pearson'] = np.mean(highconf_pearson)

        # Mean across MATCHED elements only
        if matched_all_spearman:
            exp_results['all_matched_mean_spearman'] = np.mean(matched_all_spearman)
            print(f"  MATCHED ({cell_type}) All Mean Spearman: {exp_results['all_matched_mean_spearman']:.4f}")
        if matched_highconf_spearman:
            exp_results['highconf_matched_mean_spearman'] = np.mean(matched_highconf_spearman)
            print(f"  MATCHED ({cell_type}) HC Mean Spearman: {exp_results['highconf_matched_mean_spearman']:.4f}")

        all_results.append(exp_results)

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    results_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("CAGI5 SUMMARY - RANKED BY MATCHED MEAN SPEARMAN (All SNPs)")
    print("=" * 80)
    if 'all_matched_mean_spearman' in results_df.columns:
        summary = results_df[['experiment', 'cell_type', 'all_matched_mean_spearman']].dropna().sort_values(
            'all_matched_mean_spearman', ascending=False
        )
        for _, row in summary.iterrows():
            print(f"  {row['experiment']:<35} [{row['cell_type']}] {row['all_matched_mean_spearman']:.4f}")

    print("\nCAGI5 SUMMARY - RANKED BY MATCHED MEAN SPEARMAN (High Confidence)")
    print("=" * 80)
    if 'highconf_matched_mean_spearman' in results_df.columns:
        summary = results_df[['experiment', 'cell_type', 'highconf_matched_mean_spearman']].dropna().sort_values(
            'highconf_matched_mean_spearman', ascending=False
        )
        for _, row in summary.iterrows():
            print(f"  {row['experiment']:<35} [{row['cell_type']}] {row['highconf_matched_mean_spearman']:.4f}")


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
