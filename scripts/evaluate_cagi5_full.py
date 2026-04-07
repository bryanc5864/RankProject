#!/usr/bin/env python3
"""
Full CAGI5 evaluation with Spearman AND Pearson, plus confidence stratification.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    DREAM_RNN, DREAM_RNN_SingleOutput, DREAM_RNN_DualHead,
    DREAM_RNN_DomainAdversarial, DREAM_RNN_BiasFactorized, DREAM_RNN_FullAdvanced,
    DREAM_RNN_Distributional, DREAM_RNN_DistributionalDualHead,
    FactorizedEncoder, FactorizedEncoderVIB, FactorizedEncoderGCAdv, FactorizedEncoderFull,
)

K562_ELEMENTS = ['GP1BB', 'HBB', 'HBG1', 'PKLR']


def one_hot_encode(seq: str) -> np.ndarray:
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
        else:
            one_hot[:, i] = 0.25
    return one_hot


def get_variant_sequence(ref_seq, ref_start, var_pos, ref_allele, alt_allele, window=230):
    idx = var_pos - ref_start
    if idx < 0 or idx >= len(ref_seq):
        return None
    var_seq = ref_seq[:idx] + alt_allele + ref_seq[idx + len(ref_allele):]
    center = idx + len(alt_allele) // 2
    half_window = window // 2
    start = center - half_window
    end = start + window
    if start < 0:
        seq = 'N' * (-start) + var_seq[:window + start]
    elif end > len(var_seq):
        seq = var_seq[start:] + 'N' * (end - len(var_seq))
    else:
        seq = var_seq[start:end]
    return seq[:window]


def get_ref_sequence(ref_seq, ref_start, var_pos, window=230):
    idx = var_pos - ref_start
    if idx < 0 or idx >= len(ref_seq):
        return None
    half_window = window // 2
    start = idx - half_window
    end = start + window
    if start < 0:
        seq = 'N' * (-start) + ref_seq[:window + start]
    elif end > len(ref_seq):
        seq = ref_seq[start:] + 'N' * (end - len(ref_seq))
    else:
        seq = ref_seq[start:end]
    return seq[:window]


def load_model(checkpoint_path, config_path, device):
    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get('model', 'dream_rnn')
    n_bins = config.get('n_bins', 10)
    n_domains = config.get('n_domains', 10)

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
    elif model_type == 'dream_rnn_distributional':
        model = DREAM_RNN_Distributional()
    elif model_type == 'dream_rnn_distributional_dual':
        model = DREAM_RNN_DistributionalDualHead()
    elif model_type == 'factorized':
        model = FactorizedEncoder()
    elif model_type == 'factorized_vib':
        model = FactorizedEncoderVIB(vib_beta=config.get('vib_beta', 0.01))
    elif model_type == 'factorized_gc_adv':
        model = FactorizedEncoderGCAdv(n_gc_bins=config.get('gc_bins', 10))
    elif model_type == 'factorized_full':
        model = FactorizedEncoderFull(
            vib_beta=config.get('vib_beta', 0.01),
            n_gc_bins=config.get('gc_bins', 10)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model, config


def predict_batch(model, sequences, device, batch_size=256):
    X = np.array([one_hot_encode(seq) for seq in sequences])
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() > 1:
                if out.shape[1] > 1:
                    out = out[:, 0]
                else:
                    out = out.squeeze(-1)
            predictions.extend(out.cpu().numpy().flatten())
    return np.array(predictions)


def evaluate_element_stratified(model, ref_data, cagi5_df, device, window=230):
    """Evaluate with All, High-Conf, and Low-Conf stratification."""
    ref_seq = ref_data['sequence']
    ref_start = ref_data['start']

    alt_sequences = []
    ref_sequences = []
    valid_indices = []

    for i, row in cagi5_df.iterrows():
        alt_seq = get_variant_sequence(ref_seq, ref_start, row['Pos'], row['Ref'], row['Alt'], window)
        ref_seq_window = get_ref_sequence(ref_seq, ref_start, row['Pos'], window)
        if alt_seq and ref_seq_window and len(alt_seq) == window and len(ref_seq_window) == window:
            alt_sequences.append(alt_seq)
            ref_sequences.append(ref_seq_window)
            valid_indices.append(i)

    if len(alt_sequences) == 0:
        return None

    alt_preds = predict_batch(model, alt_sequences, device)
    ref_preds = predict_batch(model, ref_sequences, device)
    predictions = alt_preds - ref_preds

    valid_df = cagi5_df.loc[valid_indices].copy()
    ground_truth = valid_df['Value'].values
    confidence = valid_df['Confidence'].values

    # All variants
    sp_all, _ = spearmanr(predictions, ground_truth)
    pe_all, _ = pearsonr(predictions, ground_truth)

    # High confidence (>= 0.1)
    hc_mask = confidence >= 0.1
    if hc_mask.sum() > 10:
        sp_hc, _ = spearmanr(predictions[hc_mask], ground_truth[hc_mask])
        pe_hc, _ = pearsonr(predictions[hc_mask], ground_truth[hc_mask])
        n_hc = hc_mask.sum()
    else:
        sp_hc, pe_hc, n_hc = np.nan, np.nan, 0

    # Low confidence (< 0.1)
    lc_mask = confidence < 0.1
    if lc_mask.sum() > 10:
        sp_lc, _ = spearmanr(predictions[lc_mask], ground_truth[lc_mask])
        pe_lc, _ = pearsonr(predictions[lc_mask], ground_truth[lc_mask])
        n_lc = lc_mask.sum()
    else:
        sp_lc, pe_lc, n_lc = np.nan, np.nan, 0

    return {
        'n_all': len(predictions), 'spearman_all': sp_all, 'pearson_all': pe_all,
        'n_hc': n_hc, 'spearman_hc': sp_hc, 'pearson_hc': pe_hc,
        'n_lc': n_lc, 'spearman_lc': sp_lc, 'pearson_lc': pe_lc,
    }


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load references
    with open(args.references) as f:
        references = json.load(f)
    print(f"Loaded {len(references)} reference sequences")

    # Load CAGI5 data
    cagi5_dir = Path(args.cagi5_dir)
    cagi5_data = {}
    for tsv_file in cagi5_dir.glob("challenge_*.tsv"):
        element = tsv_file.stem.replace("challenge_", "")
        with open(tsv_file) as f:
            lines = f.readlines()
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('#Chrom'):
                header_idx = i
                break
        if header_idx is None:
            continue
        header = lines[header_idx].lstrip('#').strip().split('\t')
        data_lines = [l.strip().split('\t') for l in lines[header_idx + 1:] if l.strip()]
        df = pd.DataFrame(data_lines, columns=header)
        df['Pos'] = df['Pos'].astype(int)
        df['Value'] = df['Value'].astype(float)
        df['Confidence'] = df['Confidence'].astype(float)
        cagi5_data[element] = df
    print(f"Loaded {len(cagi5_data)} CAGI5 elements")

    # Find experiments
    results_dir = Path(args.results_dir)
    experiments = []
    for exp_dir in sorted(results_dir.iterdir()):
        if exp_dir.is_dir():
            checkpoint = exp_dir / "checkpoints" / "best_model.pth"
            config = exp_dir / "config.json"
            if checkpoint.exists() and config.exists():
                experiments.append({
                    'name': exp_dir.name.rsplit('_202', 1)[0],
                    'checkpoint': checkpoint,
                    'config': config,
                })

    print(f"Found {len(experiments)} experiments")
    print("=" * 80)

    all_results = []
    per_element_results = []

    for exp in experiments:
        print(f"\nEvaluating: {exp['name']}")

        try:
            model, config = load_model(exp['checkpoint'], exp['config'], device)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        exp_results = {'model': exp['name']}
        element_spearmans_all = []
        element_pearsons_all = []
        element_spearmans_hc = []
        element_pearsons_hc = []
        element_spearmans_lc = []
        element_pearsons_lc = []

        for element in K562_ELEMENTS:
            if element not in cagi5_data or element not in references:
                continue

            metrics = evaluate_element_stratified(model, references[element], cagi5_data[element], device)
            if metrics is None:
                continue

            print(f"  {element}: All Sp={metrics['spearman_all']:.3f} Pe={metrics['pearson_all']:.3f} | "
                  f"HC Sp={metrics['spearman_hc']:.3f} Pe={metrics['pearson_hc']:.3f} | "
                  f"LC Sp={metrics['spearman_lc']:.3f} Pe={metrics['pearson_lc']:.3f}")

            # Store per-element
            exp_results[f'{element}_sp_all'] = metrics['spearman_all']
            exp_results[f'{element}_pe_all'] = metrics['pearson_all']
            exp_results[f'{element}_sp_hc'] = metrics['spearman_hc']
            exp_results[f'{element}_pe_hc'] = metrics['pearson_hc']
            exp_results[f'{element}_sp_lc'] = metrics['spearman_lc']
            exp_results[f'{element}_pe_lc'] = metrics['pearson_lc']
            exp_results[f'{element}_n_all'] = metrics['n_all']
            exp_results[f'{element}_n_hc'] = metrics['n_hc']
            exp_results[f'{element}_n_lc'] = metrics['n_lc']

            element_spearmans_all.append(metrics['spearman_all'])
            element_pearsons_all.append(metrics['pearson_all'])
            if not np.isnan(metrics['spearman_hc']):
                element_spearmans_hc.append(metrics['spearman_hc'])
                element_pearsons_hc.append(metrics['pearson_hc'])
            if not np.isnan(metrics['spearman_lc']):
                element_spearmans_lc.append(metrics['spearman_lc'])
                element_pearsons_lc.append(metrics['pearson_lc'])

            # Per-element detailed
            per_element_results.append({
                'model': exp['name'], 'element': element,
                'n_all': metrics['n_all'], 'spearman_all': metrics['spearman_all'], 'pearson_all': metrics['pearson_all'],
                'n_hc': metrics['n_hc'], 'spearman_hc': metrics['spearman_hc'], 'pearson_hc': metrics['pearson_hc'],
                'n_lc': metrics['n_lc'], 'spearman_lc': metrics['spearman_lc'], 'pearson_lc': metrics['pearson_lc'],
            })

        # Means
        if element_spearmans_all:
            exp_results['mean_sp_all'] = np.mean(element_spearmans_all)
            exp_results['mean_pe_all'] = np.mean(element_pearsons_all)
        if element_spearmans_hc:
            exp_results['mean_sp_hc'] = np.mean(element_spearmans_hc)
            exp_results['mean_pe_hc'] = np.mean(element_pearsons_hc)
        if element_spearmans_lc:
            exp_results['mean_sp_lc'] = np.mean(element_spearmans_lc)
            exp_results['mean_pe_lc'] = np.mean(element_pearsons_lc)

        print(f"  MEAN: All Sp={exp_results.get('mean_sp_all', 0):.4f} Pe={exp_results.get('mean_pe_all', 0):.4f}")

        all_results.append(exp_results)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    pd.DataFrame(all_results).to_csv(output_dir / 'cagi5_full_metrics.csv', index=False)
    pd.DataFrame(per_element_results).to_csv(output_dir / 'cagi5_per_element_full.csv', index=False)

    print(f"\nResults saved to {output_dir}")

    # Print top models
    print("\n" + "=" * 80)
    print("TOP 10 BY MEAN SPEARMAN (ALL)")
    df = pd.DataFrame(all_results)
    if 'mean_sp_all' in df.columns:
        top = df.nlargest(10, 'mean_sp_all')[['model', 'mean_sp_all', 'mean_pe_all', 'mean_sp_hc', 'mean_sp_lc']]
        print(top.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/noise_resistant")
    parser.add_argument("--references", default="data/cagi5_references.json")
    parser.add_argument("--cagi5_dir", default="data/raw/dream_rnn_lentimpra/data/CAGI5")
    parser.add_argument("--output_dir", default="results/noise_resistant")
    parser.add_argument("--gpu", type=int, default=0)
    main(parser.parse_args())
