#!/usr/bin/env python3
"""dump per-method per-variant alt-ref predictions so ensembling can iterate without inference.

per_element[element] holds methods (N names), predictions (N, V), ground_truth (V,)
and confidence (V,).
"""

import argparse
import json
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr

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
    parser.add_argument('--base_dir', default='results/deboer_rankloss_1fold_v4')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--references', default='data/cagi5_references.json')
    parser.add_argument('--cagi5_dir', default='data/raw/dream_rnn_lentimpra/data/CAGI5')
    parser.add_argument('--output', default='results/deboer_rankloss_1fold_v4/cagi5_predictions.pkl')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.references) as f:
        references = json.load(f)
    cagi5_data = load_cagi5_data(args.cagi5_dir)

    # discover methods
    base = Path(args.base_dir)
    method_paths = {}
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        models = sorted(sub.glob("fold*_model*/weights/model_best.pth"))
        if models:
            method_paths[sub.name] = models
    print(f"Found {len(method_paths)} methods")

    # precompute alt/ref sequences per element (so we don't redo per method)
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

    # output container
    out = {}
    for element, setup in element_setup.items():
        out[element] = {
            'methods': [], 'predictions': [],
            'ground_truth': setup['ground_truth'],
            'confidence': setup['confidence'],
        }

    # for each method, load each of its 9 models once, predict on all elements,
    # then move on. each method ensemble is mean over its 9 models.
    for method, models in method_paths.items():
        print(f"\n{method}: {len(models)} checkpoints")
        per_element_method = {e: [] for e in element_setup}
        for mp in models:
            model = build_model(device)
            model.load_state_dict(torch.load(mp, map_location=device))
            model.eval()
            for element, setup in element_setup.items():
                alt_p = predict_batch(model, setup['alt_seqs'], device)
                ref_p = predict_batch(model, setup['ref_seqs'], device)
                per_element_method[element].append(alt_p - ref_p)
            del model
            torch.cuda.empty_cache()
        for element in element_setup:
            ensemble = np.mean(per_element_method[element], axis=0)
            out[element]['methods'].append(method)
            out[element]['predictions'].append(ensemble)
            sp = spearmanr(ensemble, element_setup[element]['ground_truth'])[0]
            print(f"  {element}: Sp={sp:.4f}")

    # convert prediction lists to arrays
    for element in out:
        out[element]['predictions'] = np.array(out[element]['predictions'])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
