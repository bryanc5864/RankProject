#!/usr/bin/env python3
"""
Evaluate 1-fold Prix Fixe models on CAGI5.
Takes a base directory and evaluates all subdirs that contain model checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, kendalltau

PRIXFIXE_PARENT = '/home/bcheng/RankProject/data/raw/deboer_dream/benchmarks/human'
sys.path.insert(0, PRIXFIXE_PARENT)

from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

K562_ELEMENTS = ['GP1BB', 'HBB', 'HBG1', 'PKLR']


def build_model(device):
    generator = torch.Generator()
    generator.manual_seed(42)
    first = BHIFirstLayersBlock(in_channels=5, out_channels=320, seqsize=230,
                                 kernel_sizes=[9, 15], pool_size=1, dropout=0.2)
    core = BHICoreBlock(in_channels=first.out_channels, out_channels=320,
                        seqsize=first.infer_outseqsize(), lstm_hidden_channels=320,
                        kernel_sizes=[9, 15], pool_size=1, dropout1=0.2, dropout2=0.5)
    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)
    return PrixFixeNet(first=first, core=core, final=final, generator=generator).to(device)


def encode_sequence(seq):
    mapping = {'A': [1,0,0,0], 'G': [0,1,0,0], 'C': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    encoded = np.array([mapping.get(b.upper(), [0,0,0,0]) for b in seq], dtype=np.float32)
    rev = np.zeros((len(seq), 1), dtype=np.float32)
    return np.concatenate([encoded, rev], axis=1).T


def predict_batch(model, sequences, device, batch_size=256):
    X = np.array([encode_sequence(seq) for seq in sequences])
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            preds.extend(model(batch).cpu().numpy().flatten())
    return np.array(preds)


def get_variant_sequence(ref_seq, ref_start, var_pos, ref_allele, alt_allele, window=230):
    idx = var_pos - ref_start
    if idx < 0 or idx >= len(ref_seq):
        return None
    var_seq = ref_seq[:idx] + alt_allele + ref_seq[idx + len(ref_allele):]
    center = idx + len(alt_allele) // 2
    start = center - window // 2
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
    start = idx - window // 2
    end = start + window
    if start < 0:
        seq = 'N' * (-start) + ref_seq[:window + start]
    elif end > len(ref_seq):
        seq = ref_seq[start:] + 'N' * (end - len(ref_seq))
    else:
        seq = ref_seq[start:end]
    return seq[:window]


def load_cagi5_data(cagi5_dir):
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
        data_lines = [l.strip().split('\t') for l in lines[header_idx+1:] if l.strip()]
        df = pd.DataFrame(data_lines, columns=header)
        df['Pos'] = df['Pos'].astype(int)
        df['Value'] = df['Value'].astype(float)
        df['Confidence'] = df['Confidence'].astype(float)
        cagi5_data[element] = df
    return cagi5_data


def evaluate_run(run_name, run_dir, references, cagi5_data, device):
    model_paths = sorted(Path(run_dir).glob("fold*_model*/weights/model_best.pth"))
    if not model_paths:
        return None

    print(f"\n  {run_name}: {len(model_paths)} models")
    result = {'experiment': run_name, 'n_models': len(model_paths)}

    for element, df in cagi5_data.items():
        if element not in references:
            continue

        ref_seq = references[element]['sequence']
        ref_start = references[element]['start']

        # Build alt/ref sequences once
        alt_seqs, ref_seqs, valid_idx = [], [], []
        for i, row in df.iterrows():
            alt = get_variant_sequence(ref_seq, ref_start, row['Pos'], row['Ref'], row['Alt'])
            ref = get_ref_sequence(ref_seq, ref_start, row['Pos'])
            if alt and ref and len(alt) == 230 and len(ref) == 230:
                alt_seqs.append(alt)
                ref_seqs.append(ref)
                valid_idx.append(i)

        if not alt_seqs:
            continue

        ground_truth = df.loc[valid_idx, 'Value'].values
        confidence = df.loc[valid_idx, 'Confidence'].values

        # Ensemble predictions from all models
        all_effects = []
        for mp in model_paths:
            model = build_model(device)
            model.load_state_dict(torch.load(mp, map_location=device))
            model.eval()
            alt_preds = predict_batch(model, alt_seqs, device)
            ref_preds = predict_batch(model, ref_seqs, device)
            all_effects.append(alt_preds - ref_preds)
            del model
            torch.cuda.empty_cache()

        ensemble = np.mean(all_effects, axis=0)

        # All variants
        sp = spearmanr(ensemble, ground_truth)[0]
        pe = pearsonr(ensemble, ground_truth)[0]
        result[f'all_{element}_spearman'] = sp
        result[f'all_{element}_pearson'] = pe
        result[f'all_{element}_n'] = len(ensemble)

        # High confidence
        hc = confidence >= 0.1
        if hc.sum() > 5:
            sp_hc = spearmanr(ensemble[hc], ground_truth[hc])[0]
            pe_hc = pearsonr(ensemble[hc], ground_truth[hc])[0]
            result[f'hc_{element}_spearman'] = sp_hc
            result[f'hc_{element}_pearson'] = pe_hc
            result[f'hc_{element}_n'] = int(hc.sum())
            print(f"    {element}: Sp={sp:.4f} (HC={sp_hc:.4f}, n={int(hc.sum())})")
        else:
            print(f"    {element}: Sp={sp:.4f}")

    # Compute K562-matched mean
    matched = [result.get(f'all_{e}_spearman') for e in K562_ELEMENTS
               if f'all_{e}_spearman' in result]
    if matched:
        result['K562_matched_sp'] = np.mean(matched)

    all_sp = [v for k, v in result.items() if k.startswith('all_') and k.endswith('_spearman')]
    if all_sp:
        result['all_mean_sp'] = np.mean(all_sp)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='results/deboer_rankloss_1fold')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--references', type=str, default='data/cagi5_references.json')
    parser.add_argument('--cagi5_dir', type=str, default='data/raw/dream_rnn_lentimpra/data/CAGI5')
    parser.add_argument('--output', type=str, default='results/deboer_rankloss_1fold/cagi5_results.csv')
    # Also include the 90-model runs for comparison
    parser.add_argument('--include_90model', action='store_true')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(args.references) as f:
        references = json.load(f)
    cagi5_data = load_cagi5_data(args.cagi5_dir)
    print(f"Loaded {len(references)} references, {len(cagi5_data)} elements")

    base = Path(args.base_dir)
    all_results = []

    # Evaluate all subdirs that have model checkpoints
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        model_paths = list(sub.glob("fold*_model*/weights/model_best.pth"))
        if not model_paths:
            continue
        result = evaluate_run(sub.name, sub, references, cagi5_data, device)
        if result:
            all_results.append(result)

    # Optionally include 90-model baseline
    if args.include_90model:
        for name, path in [('MSE_90model', 'results/deboer_official'),
                           ('RankNet_90model', 'results/deboer_rankloss/combined_ranknet')]:
            if Path(path).exists():
                result = evaluate_run(name, path, references, cagi5_data, device)
                if result:
                    all_results.append(result)

    # Save and print
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*75}")
    print(f"{'Method':<30} {'K562 Match Sp':>13} {'All Sp':>10} {'Models':>7}")
    print('-' * 75)
    for _, row in df.sort_values('K562_matched_sp', ascending=False).iterrows():
        k562 = row.get('K562_matched_sp', float('nan'))
        allsp = row.get('all_mean_sp', float('nan'))
        n = row.get('n_models', '?')
        print(f"  {row['experiment']:<28} {k562:>13.4f} {allsp:>10.4f} {n:>7}")

    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
