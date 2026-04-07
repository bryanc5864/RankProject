#!/usr/bin/env python3
"""
Evaluate Prix Fixe DREAM-RNN models on CAGI5 saturation mutagenesis benchmark.

For each run (MSE, RankNet, etc.), loads all 90 models (10 folds × 9 models)
and ensembles predictions. Also reports per-fold-ensemble results.

Usage:
    python scripts/evaluate_cagi5_prixfixe.py --gpu 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, kendalltau

# Prix Fixe code
PRIXFIXE_PARENT = '/home/bcheng/RankProject/data/raw/deboer_dream/benchmarks/human'
sys.path.insert(0, PRIXFIXE_PARENT)

from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

# CAGI5 element to cell-type mapping
K562_ELEMENTS = ['GP1BB', 'HBB', 'HBG1', 'PKLR']
HEPG2_ELEMENTS = ['F9', 'LDLR', 'SORT1']
ALL_ELEMENTS = K562_ELEMENTS + HEPG2_ELEMENTS


def build_model(device):
    """Build DREAM-RNN model using official Prix Fixe components."""
    generator = torch.Generator()
    generator.manual_seed(42)

    first = BHIFirstLayersBlock(
        in_channels=5, out_channels=320, seqsize=230,
        kernel_sizes=[9, 15], pool_size=1, dropout=0.2
    )
    core = BHICoreBlock(
        in_channels=first.out_channels, out_channels=320,
        seqsize=first.infer_outseqsize(), lstm_hidden_channels=320,
        kernel_sizes=[9, 15], pool_size=1, dropout1=0.2, dropout2=0.5
    )
    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)
    model = PrixFixeNet(first=first, core=core, final=final, generator=generator)
    return model.to(device)


def encode_sequence(seq, is_reverse=0):
    """Encode DNA sequence to 5-channel format (ACGT + reverse flag)."""
    mapping = {
        'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0],
        'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]
    }
    encoded = np.array([mapping.get(b.upper(), [0, 0, 0, 0]) for b in seq], dtype=np.float32)
    rev_channel = np.full((len(seq), 1), is_reverse, dtype=np.float32)
    encoded = np.concatenate([encoded, rev_channel], axis=1)
    return encoded.T  # (5, seq_len)


def predict_batch(model, sequences, device, batch_size=256):
    """Run model predictions on sequences."""
    X = np.array([encode_sequence(seq) for seq in sequences])
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            out = model(batch)
            predictions.extend(out.cpu().numpy().flatten())
    return np.array(predictions)


def get_variant_sequence(ref_seq, ref_start, var_pos, ref_allele, alt_allele, window=230):
    """Create variant sequence centered on the variant position."""
    idx = var_pos - ref_start
    if idx < 0 or idx >= len(ref_seq):
        return None

    var_seq = ref_seq[:idx] + alt_allele + ref_seq[idx + len(ref_allele):]
    center = idx + len(alt_allele) // 2
    half_window = window // 2
    start = center - half_window
    end = start + window

    if start < 0:
        pad_left = -start
        seq = 'N' * pad_left + var_seq[:window - pad_left]
    elif end > len(var_seq):
        pad_right = end - len(var_seq)
        seq = var_seq[start:] + 'N' * pad_right
    else:
        seq = var_seq[start:end]
    return seq[:window]


def get_ref_sequence(ref_seq, ref_start, var_pos, window=230):
    """Get reference sequence centered on the variant position."""
    idx = var_pos - ref_start
    if idx < 0 or idx >= len(ref_seq):
        return None

    half_window = window // 2
    start = idx - half_window
    end = start + window

    if start < 0:
        pad_left = -start
        seq = 'N' * pad_left + ref_seq[:window - pad_left]
    elif end > len(ref_seq):
        pad_right = end - len(ref_seq)
        seq = ref_seq[start:] + 'N' * pad_right
    else:
        seq = ref_seq[start:end]
    return seq[:window]


def compute_variant_effects(model, ref_data, cagi5_df, device, window=230):
    """Compute variant effects (alt - ref) for all variants."""
    ref_seq = ref_data['sequence']
    ref_start = ref_data['start']

    alt_sequences = []
    ref_sequences = []
    valid_indices = []

    for i, row in cagi5_df.iterrows():
        alt_seq = get_variant_sequence(
            ref_seq, ref_start, row['Pos'], row['Ref'], row['Alt'], window
        )
        ref_seq_window = get_ref_sequence(ref_seq, ref_start, row['Pos'], window)

        if alt_seq and ref_seq_window and len(alt_seq) == window and len(ref_seq_window) == window:
            alt_sequences.append(alt_seq)
            ref_sequences.append(ref_seq_window)
            valid_indices.append(i)

    if len(alt_sequences) == 0:
        return None, None

    alt_preds = predict_batch(model, alt_sequences, device)
    ref_preds = predict_batch(model, ref_sequences, device)

    effects = alt_preds - ref_preds
    ground_truth = cagi5_df.loc[valid_indices, 'Value'].values

    return effects, ground_truth


def evaluate_effects(predictions, ground_truth, min_confidence_df=None):
    """Compute correlation metrics."""
    if predictions is None or len(predictions) == 0:
        return None
    sp, _ = spearmanr(predictions, ground_truth)
    pe, _ = pearsonr(predictions, ground_truth)
    kt, _ = kendalltau(predictions, ground_truth)
    return {'spearman': sp, 'pearson': pe, 'kendall': kt, 'n': len(predictions)}


def load_cagi5_data(cagi5_dir):
    """Load all CAGI5 element data."""
    cagi5_data = {}
    for tsv_file in Path(cagi5_dir).glob("challenge_*.tsv"):
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
    return cagi5_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--references', type=str, default='data/cagi5_references.json')
    parser.add_argument('--cagi5_dir', type=str, default='data/raw/dream_rnn_lentimpra/data/CAGI5')
    parser.add_argument('--output', type=str, default='results/deboer_rankloss/cagi5_evaluation.csv')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load reference sequences and CAGI5 data
    with open(args.references) as f:
        references = json.load(f)
    cagi5_data = load_cagi5_data(args.cagi5_dir)
    print(f"Loaded {len(references)} references, {len(cagi5_data)} CAGI5 elements")

    # Define all runs to evaluate
    runs = {}
    base = Path('results')

    # MSE baseline
    mse_dir = base / 'deboer_official'
    if (mse_dir / 'cv_results.json').exists():
        runs['MSE_baseline'] = mse_dir

    # Rank loss runs
    rankloss_dir = base / 'deboer_rankloss'
    for sub in sorted(rankloss_dir.iterdir()) if rankloss_dir.exists() else []:
        if sub.is_dir() and (sub / 'cv_results.json').exists():
            runs[f'rank_{sub.name}'] = sub

    print(f"\nFound {len(runs)} completed runs: {list(runs.keys())}")

    all_results = []

    for run_name, run_dir in runs.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {run_name}")
        print(f"{'='*70}")

        # Find all model checkpoints
        model_paths = sorted(run_dir.glob("fold*_model*/weights/model_best.pth"))
        print(f"  Found {len(model_paths)} model checkpoints")

        if len(model_paths) == 0:
            continue

        # For each CAGI5 element, collect predictions from all models
        element_results = {}

        for element in cagi5_data:
            if element not in references:
                continue

            all_effects = []
            ground_truth = None

            for mp in model_paths:
                model = build_model(device)
                model.load_state_dict(torch.load(mp, map_location=device))
                model.eval()

                effects, gt = compute_variant_effects(
                    model, references[element], cagi5_data[element], device
                )

                if effects is not None:
                    all_effects.append(effects)
                    ground_truth = gt

                del model
                torch.cuda.empty_cache()

            if len(all_effects) == 0 or ground_truth is None:
                continue

            # Full ensemble (all 90 models)
            ensemble_effects = np.mean(all_effects, axis=0)
            metrics_full = evaluate_effects(ensemble_effects, ground_truth)

            # High-confidence subset
            hc_mask = cagi5_data[element]['Confidence'] >= 0.1
            # Need to figure out valid indices
            ref_seq = references[element]['sequence']
            ref_start = references[element]['start']
            valid_mask = []
            for _, row in cagi5_data[element].iterrows():
                alt_seq = get_variant_sequence(ref_seq, ref_start, row['Pos'], row['Ref'], row['Alt'])
                ref_seq_w = get_ref_sequence(ref_seq, ref_start, row['Pos'])
                valid_mask.append(alt_seq is not None and ref_seq_w is not None
                                  and len(alt_seq) == 230 and len(ref_seq_w) == 230)
            valid_mask = np.array(valid_mask)

            # Get high confidence indices among valid variants
            valid_indices = np.where(valid_mask)[0]
            hc_values = cagi5_data[element]['Confidence'].values
            hc_among_valid = hc_values[valid_mask] >= 0.1

            if hc_among_valid.sum() > 0:
                metrics_hc = evaluate_effects(
                    ensemble_effects[hc_among_valid],
                    ground_truth[hc_among_valid]
                )
            else:
                metrics_hc = None

            # Per-fold ensemble (average of 9 models within each fold)
            fold_effects = {}
            for mp in model_paths:
                fold_id = mp.parts[-3].split('_model')[0]  # e.g., "fold0"
                if fold_id not in fold_effects:
                    fold_effects[fold_id] = []
                # Find which index this model was in all_effects
                idx = list(model_paths).index(mp)
                fold_effects[fold_id].append(all_effects[idx])

            fold_ensemble_effects = []
            for fold_id in sorted(fold_effects.keys()):
                fold_ensemble_effects.append(np.mean(fold_effects[fold_id], axis=0))

            # Grand ensemble of fold ensembles (should be same as full ensemble)
            fold_spearman = [spearmanr(fe, ground_truth)[0] for fe in fold_ensemble_effects]

            element_results[element] = {
                'full_ensemble': metrics_full,
                'high_conf': metrics_hc,
                'per_fold_spearman': fold_spearman,
            }

            print(f"  {element}: Sp={metrics_full['spearman']:.4f} "
                  f"(n={metrics_full['n']})", end="")
            if metrics_hc:
                print(f", HC Sp={metrics_hc['spearman']:.4f} (n={metrics_hc['n']})")
            else:
                print()

        # Aggregate results
        result_row = {'experiment': run_name, 'cell_type': 'K562'}

        matched_sp = []
        all_sp = []

        for element, er in element_results.items():
            m = er['full_ensemble']
            result_row[f'all_{element}_spearman'] = m['spearman']
            result_row[f'all_{element}_pearson'] = m['pearson']
            result_row[f'all_{element}_kendall'] = m['kendall']
            result_row[f'all_{element}_n'] = m['n']
            all_sp.append(m['spearman'])

            if element in K562_ELEMENTS:
                matched_sp.append(m['spearman'])

            if er['high_conf']:
                hc = er['high_conf']
                result_row[f'highconf_{element}_spearman'] = hc['spearman']
                result_row[f'highconf_{element}_pearson'] = hc['pearson']
                result_row[f'highconf_{element}_n'] = hc['n']

        if matched_sp:
            result_row['matched_mean_spearman'] = np.mean(matched_sp)
            print(f"\n  K562-matched mean Spearman: {np.mean(matched_sp):.4f}")
        if all_sp:
            result_row['all_mean_spearman'] = np.mean(all_sp)
            print(f"  All elements mean Spearman: {np.mean(all_sp):.4f}")

        all_results.append(result_row)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Final comparison table
    print(f"\n{'='*70}")
    print("CAGI5 COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'Matched K562 Sp':>15} {'All Elements Sp':>15}")
    print("-" * 62)
    for _, row in results_df.iterrows():
        matched = row.get('matched_mean_spearman', float('nan'))
        all_mean = row.get('all_mean_spearman', float('nan'))
        print(f"  {row['experiment']:<28} {matched:>15.4f} {all_mean:>15.4f}")


if __name__ == '__main__':
    main()
