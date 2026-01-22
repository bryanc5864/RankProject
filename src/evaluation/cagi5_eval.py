"""
CAGI5 Saturation Mutagenesis Evaluation

Zero-shot evaluation on CAGI5 benchmark data for enhancers and promoters.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from .metrics import compute_all_metrics, spearman_correlation, kendall_tau, ndcg_score


# CAGI5 elements
CAGI5_ENHANCERS = ['IRF4', 'IRF6', 'MYCrs6983267', 'SORT1', 'MSMB', 'ZFAND3']
CAGI5_PROMOTERS = ['F9', 'GP1BB', 'HBB', 'HBG1', 'HNF4A', 'LDLR', 'PKLR', 'TERT-GBM', 'TERT-HEK293T']
CAGI5_ELEMENTS = CAGI5_ENHANCERS + CAGI5_PROMOTERS


def load_cagi5_element(data_dir: Union[str, Path], element: str,
                       use_challenge: bool = True) -> pd.DataFrame:
    """
    Load CAGI5 data for a single element.

    Args:
        data_dir: Directory containing CAGI5 TSV files
        element: Element name (e.g., 'IRF4', 'TERT-GBM')
        use_challenge: Use challenge set (True) or release set (False)

    Returns:
        DataFrame with columns: Chrom, Pos, Ref, Alt, Value, sequence (if available)
    """
    data_dir = Path(data_dir)
    prefix = 'challenge' if use_challenge else 'release'
    filepath = data_dir / f'{prefix}_{element}.tsv'

    if not filepath.exists():
        raise FileNotFoundError(f"CAGI5 file not found: {filepath}")

    # Read TSV, skip comment lines
    df = pd.read_csv(filepath, sep='\t', comment='#')

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    return df


def load_all_cagi5_data(data_dir: Union[str, Path],
                        use_challenge: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load CAGI5 data for all elements.

    Args:
        data_dir: Directory containing CAGI5 TSV files
        use_challenge: Use challenge set (True) or release set (False)

    Returns:
        Dictionary mapping element names to DataFrames
    """
    data_dir = Path(data_dir)
    data = {}

    for element in CAGI5_ELEMENTS:
        try:
            data[element] = load_cagi5_element(data_dir, element, use_challenge)
        except FileNotFoundError:
            print(f"Warning: Could not load {element}")

    return data


def get_variant_sequence(ref_seq: str, pos: int, ref: str, alt: str,
                         seq_len: int = 230, center: bool = True) -> str:
    """
    Generate variant sequence by substituting alt allele at position.

    Args:
        ref_seq: Reference sequence
        pos: 1-based position of variant
        ref: Reference allele
        alt: Alternate allele
        seq_len: Output sequence length
        center: Center the variant in the output sequence

    Returns:
        Variant sequence of length seq_len
    """
    # Convert to 0-based
    pos_0 = pos - 1

    # Verify reference allele matches
    if ref_seq[pos_0:pos_0 + len(ref)] != ref:
        raise ValueError(f"Reference mismatch at position {pos}: "
                        f"expected {ref}, got {ref_seq[pos_0:pos_0 + len(ref)]}")

    # Create variant sequence
    var_seq = ref_seq[:pos_0] + alt + ref_seq[pos_0 + len(ref):]

    # Extract window
    if center:
        # Center variant in output window
        var_pos_in_output = seq_len // 2
        start = pos_0 - var_pos_in_output
    else:
        start = 0

    # Handle boundary cases
    if start < 0:
        # Pad with N at beginning
        pad_left = -start
        seq = 'N' * pad_left + var_seq[:seq_len - pad_left]
    elif start + seq_len > len(var_seq):
        # Pad with N at end
        pad_right = start + seq_len - len(var_seq)
        seq = var_seq[start:] + 'N' * pad_right
    else:
        seq = var_seq[start:start + seq_len]

    return seq


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    """
    One-hot encode a DNA sequence.

    Args:
        seq: DNA sequence (A, C, G, T, N)

    Returns:
        One-hot encoded array [seq_len, 4]
    """
    encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}
    seq = seq.upper()

    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = encoding.get(base, -1)
        if idx >= 0:
            one_hot[i, idx] = 1.0
        else:
            # N or unknown: uniform distribution
            one_hot[i, :] = 0.25

    return one_hot


def evaluate_cagi5(model: torch.nn.Module,
                   data_dir: Union[str, Path],
                   device: torch.device = None,
                   seq_len: int = 230,
                   batch_size: int = 256,
                   use_challenge: bool = True,
                   reference_sequences: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on CAGI5 saturation mutagenesis data.

    Note: This requires reference sequences for each element to generate variant sequences.
    If reference_sequences is not provided, assumes the data files include pre-computed sequences.

    Args:
        model: Trained PyTorch model
        data_dir: Directory containing CAGI5 TSV files
        device: Device to run inference on
        seq_len: Sequence length expected by model
        batch_size: Batch size for inference
        use_challenge: Use challenge set (True) or release set (False)
        reference_sequences: Dict mapping element names to reference sequences

    Returns:
        Dictionary mapping element names to metric dictionaries
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = {}

    data_dir = Path(data_dir)
    cagi5_data = load_all_cagi5_data(data_dir, use_challenge)

    for element, df in cagi5_data.items():
        print(f"Evaluating {element}...")

        # Check if we have sequence data
        if 'sequence' in df.columns:
            sequences = df['sequence'].values
        elif reference_sequences is not None and element in reference_sequences:
            # Generate variant sequences
            ref_seq = reference_sequences[element]
            sequences = []
            for _, row in df.iterrows():
                try:
                    var_seq = get_variant_sequence(
                        ref_seq, row['Pos'], row['Ref'], row['Alt'], seq_len
                    )
                    sequences.append(var_seq)
                except Exception as e:
                    print(f"  Warning: Could not generate sequence for {element} "
                          f"pos {row['Pos']}: {e}")
                    sequences.append('N' * seq_len)
            sequences = np.array(sequences)
        else:
            print(f"  Skipping {element}: no sequence data available")
            continue

        # One-hot encode
        X = np.array([one_hot_encode_sequence(seq) for seq in sequences])
        X = np.transpose(X, (0, 2, 1))  # [N, 4, seq_len]

        # Ground truth effects
        y = df['Value'].values if 'Value' in df.columns else None
        if y is None:
            print(f"  Skipping {element}: no ground truth values")
            continue

        # Run inference in batches
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i + batch_size]).to(device)
                batch_pred = model(batch_X)

                # Handle multi-output models (take first output = activity)
                if batch_pred.dim() > 1 and batch_pred.shape[1] > 1:
                    batch_pred = batch_pred[:, 0]

                predictions.extend(batch_pred.cpu().numpy().flatten())

        predictions = np.array(predictions)

        # Compute metrics
        element_metrics = compute_all_metrics(predictions, y, k_values=[10, 50, 100])
        results[element] = element_metrics

        print(f"  Spearman: {element_metrics['spearman']:.4f}, "
              f"Kendall: {element_metrics['kendall']:.4f}, "
              f"Pearson: {element_metrics['pearson']:.4f}")

    return results


def summarize_cagi5_results(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Summarize CAGI5 evaluation results.

    Args:
        results: Dictionary from evaluate_cagi5

    Returns:
        DataFrame with per-element and aggregate metrics
    """
    rows = []

    for element, metrics in results.items():
        row = {'element': element}
        row.update(metrics)

        # Add element type
        if element in CAGI5_ENHANCERS:
            row['type'] = 'enhancer'
        elif element in CAGI5_PROMOTERS:
            row['type'] = 'promoter'
        else:
            row['type'] = 'unknown'

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add aggregate rows
    if len(df) > 0:
        # Overall mean
        mean_row = {'element': 'MEAN', 'type': 'aggregate'}
        for col in df.columns:
            if col not in ['element', 'type'] and df[col].dtype in [np.float64, np.float32]:
                mean_row[col] = df[col].mean()
        rows.append(mean_row)

        # Enhancer mean
        enh_df = df[df['type'] == 'enhancer']
        if len(enh_df) > 0:
            enh_row = {'element': 'ENHANCER_MEAN', 'type': 'aggregate'}
            for col in enh_df.columns:
                if col not in ['element', 'type'] and enh_df[col].dtype in [np.float64, np.float32]:
                    enh_row[col] = enh_df[col].mean()
            rows.append(enh_row)

        # Promoter mean
        prom_df = df[df['type'] == 'promoter']
        if len(prom_df) > 0:
            prom_row = {'element': 'PROMOTER_MEAN', 'type': 'aggregate'}
            for col in prom_df.columns:
                if col not in ['element', 'type'] and prom_df[col].dtype in [np.float64, np.float32]:
                    prom_row[col] = prom_df[col].mean()
            rows.append(prom_row)

    return pd.DataFrame(rows)
