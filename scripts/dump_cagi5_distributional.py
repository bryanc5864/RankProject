#!/usr/bin/env python3
"""
CAGI5 inference for DREAM_RNN_Distributional model (Bet 1).

Adapted from dump_cagi5_predictions.py for the 4-channel distributional model.
Each "method" here corresponds to a single seed; final ensemble averages 3 seeds.
"""

import argparse
import json
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

# Distributional model from this repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import DREAM_RNN_Distributional

K562_ELEMENTS = ['GP1BB', 'HBB', 'HBG1', 'PKLR']


def build_model(device):
    model = DREAM_RNN_Distributional(in_channels=4, seq_len=230)
    return model.to(device)


def encode_sequence(seq):
    # 4-channel ACGT one-hot (no reverse strand indicator)
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    encoded = np.array([mapping.get(b.upper(), [0,0,0,0]) for b in seq], dtype=np.float32)
    return encoded.T  # (4, seq_len)


def predict_batch(model, sequences, device, batch_size=256):
    X = np.array([encode_sequence(seq) for seq in sequences])
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            mu, _ = model(batch)
            preds.extend(mu.cpu().numpy().flatten())
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
        header_idx = next((i for i, l in enumerate(lines) if l.startswith('#Chrom')), None)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Paths to final_model.pth files for each seed')
    parser.add_argument('--method_name', default='bet1_heteroscedastic_v2',
                        help='Name to assign to the ensembled method')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--references', default='data/cagi5_references.json')
    parser.add_argument('--cagi5_dir', default='data/raw/dream_rnn_lentimpra/data/CAGI5')
    parser.add_argument('--output', default='results/noise_resistant/bet1_cagi5_predictions.pkl')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoints: {len(args.checkpoints)}")

    with open(args.references) as f:
        references = json.load(f)
    cagi5_data = load_cagi5_data(args.cagi5_dir)

    # Precompute alt/ref sequences per element
    element_setup = {}
    for element, df in cagi5_data.items():
        if element not in references:
            continue
        ref_seq = references[element]['sequence']
        ref_start = references[element]['start']
        alt_seqs, ref_seqs, valid_idx = [], [], []
        for i, row in df.iterrows():
            alt = get_variant_sequence(ref_seq, ref_start, row['Pos'], row['Ref'], row['Alt'])
            ref = get_ref_sequence(ref_seq, ref_start, row['Pos'])
            if alt and ref and len(alt) == 230 and len(ref) == 230:
                alt_seqs.append(alt); ref_seqs.append(ref); valid_idx.append(i)
        if not alt_seqs:
            continue
        element_setup[element] = {
            'alt_seqs': alt_seqs, 'ref_seqs': ref_seqs,
            'ground_truth': df.loc[valid_idx, 'Value'].values,
            'confidence': df.loc[valid_idx, 'Confidence'].values,
            'n': len(valid_idx),
        }

    print(f"Elements with valid sequences: {len(element_setup)}")

    # Run inference: per element, ensemble across all 3 seeds (mean of alt-ref)
    per_element_ensemble = {e: [] for e in element_setup}
    for mp in args.checkpoints:
        print(f"\nLoading {mp}")
        model = build_model(device)
        # Strip 'model_state_dict' wrapping if present
        sd = torch.load(mp, map_location=device, weights_only=False)
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        model.load_state_dict(sd)
        model.eval()
        for element, setup in element_setup.items():
            alt_p = predict_batch(model, setup['alt_seqs'], device)
            ref_p = predict_batch(model, setup['ref_seqs'], device)
            per_element_ensemble[element].append(alt_p - ref_p)
        del model
        torch.cuda.empty_cache()

    # Output container with a single "method" entry holding the seed-ensemble mean
    out = {}
    print(f"\n=== {args.method_name} per-element Spearman ===")
    k562_sps = []
    all_sps = []
    for element, setup in element_setup.items():
        ensemble = np.mean(per_element_ensemble[element], axis=0)
        out[element] = {
            'methods': [args.method_name],
            'predictions': np.array([ensemble]),
            'ground_truth': setup['ground_truth'],
            'confidence': setup['confidence'],
        }
        sp = spearmanr(ensemble, setup['ground_truth'])[0]
        all_sps.append(sp)
        if element in K562_ELEMENTS:
            k562_sps.append(sp)
        print(f"  {element}: Sp={sp:.4f} (n={setup['n']})")

    print(f"\nK562-matched mean Sp: {np.mean(k562_sps):.4f}")
    print(f"All-element mean Sp:  {np.mean(all_sps):.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
