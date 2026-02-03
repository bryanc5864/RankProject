#!/usr/bin/env python3
"""
Motif probing analysis: Do model embeddings encode known regulatory motifs?

Probes embeddings for presence/count of MPRA-validated TF binding motifs
relevant to K562 (erythroid) and HepG2 (liver) enhancer activity.

Usage:
    python scripts/motif_probing.py --gpu 0 --n_samples 5000
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import h5py
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    DREAM_RNN, DREAM_RNN_SingleOutput, DREAM_RNN_DualHead,
    DREAM_RNN_DomainAdversarial, DREAM_RNN_BiasFactorized, DREAM_RNN_FullAdvanced
)

OUTPUT_DIR = Path('results/interpretability')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Motif Definitions ────────────────────────────────────────────────────────
# MPRA-validated TF binding motifs from Agarwal et al. 2024 and literature

MOTIFS = {
    # K562-specific (erythroid)
    'GATA1': {
        'pattern': r'[AT]GATA[AG]',
        'cell_type': 'K562',
        'description': 'GATA1 binding site (erythroid master regulator)',
    },
    'TAL1_Ebox': {
        'pattern': r'CA[ACGT]{2}TG',
        'cell_type': 'K562',
        'description': 'E-box (TAL1/SCL heterodimer)',
    },
    'KLF1_CACCC': {
        'pattern': r'C[ACGT]?CACCC',
        'cell_type': 'K562',
        'description': 'CACCC-box (KLF1/EKLF, beta-globin regulation)',
    },
    'NFE2_MARE': {
        'pattern': r'TGA[GC]TCA',
        'cell_type': 'K562',
        'description': 'MARE (NF-E2 / AP-1 core)',
    },
    'STAT5_GAS': {
        'pattern': r'TTC.{3}GAA',
        'cell_type': 'K562',
        'description': 'GAS element (STAT5, EPO/JAK2 signaling)',
    },
    # HepG2-specific (liver)
    'HNF4A_DR1': {
        'pattern': r'AGGTCA.AGGTCA',
        'cell_type': 'HepG2',
        'description': 'DR1 (HNF4A homodimer, liver master regulator)',
    },
    'HNF1A': {
        'pattern': r'GTTAAT.ATT[GA]AC',
        'cell_type': 'HepG2',
        'description': 'HNF1A palindrome (hepatocyte-specific)',
    },
    'FOXA': {
        'pattern': r'[TG][AG]TT[GT]AC',
        'cell_type': 'HepG2',
        'description': 'Forkhead (FOXA1/2 pioneer factor)',
    },
    'CEBP': {
        'pattern': r'T[TG].{2}G.AA[TG]',
        'cell_type': 'HepG2',
        'description': 'C/EBP (CEBPA/B, liver differentiation)',
    },
    'USF_Ebox': {
        'pattern': r'CACGTG',
        'cell_type': 'HepG2',
        'description': 'E-box (USF1/2, SORT1 enhancer)',
    },
    # Housekeeping (both cell types)
    'ETS': {
        'pattern': r'[AG]?GGA[AT]',
        'cell_type': 'both',
        'description': 'ETS family core (top activator across cell types)',
    },
    'SP1_GCbox': {
        'pattern': r'GG[GC]CGG',
        'cell_type': 'both',
        'description': 'GC-box (SP1, ubiquitous activator)',
    },
    'NRF1': {
        'pattern': r'GCGC[AT][CT]GCGC',
        'cell_type': 'both',
        'description': 'NRF1 (housekeeping activator)',
    },
    'CCAAT': {
        'pattern': r'CCAAT',
        'cell_type': 'both',
        'description': 'CCAAT-box (NF-Y complex)',
    },
}

# Known regulatory grammar (composite motifs / spacing rules)
GRAMMAR_PATTERNS = {
    'GATA_TAL1_composite': {
        'pattern': r'C[AT]G.TG.{7,9}[AT]GATAA',
        'cell_type': 'K562',
        'description': 'GATA1::TAL1 composite (E-box + 7-9bp + WGATAA)',
    },
    'tandem_GATA': {
        'pattern': r'[AT]GATA[AG].{1,20}[AT]GATA[AG]',
        'cell_type': 'K562',
        'description': 'Tandem GATA sites (cooperative binding)',
    },
    'SP1_near_GATA': {
        'pattern': r'GG[GC]CGG.{1,30}[AT]GATA[AG]',
        'cell_type': 'K562',
        'description': 'SP1 flanking GATA1 (chromatin opening)',
    },
    'HNF4A_FOXA_pair': {
        'pattern': r'AGGTCA.AGGTCA.{1,50}[TG][AG]TT[GT]AC',
        'cell_type': 'HepG2',
        'description': 'HNF4A DR1 + FOXA (liver enhancer grammar)',
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def onehot_to_sequence(x):
    """Convert one-hot (4, L) to nucleotide string. Encoding: A=0, C=1, G=2, T=3."""
    idx_to_nt = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    indices = np.argmax(x, axis=0)
    return ''.join(idx_to_nt[i] for i in indices)


def scan_motifs(sequences, motif_dict):
    """Scan sequences for motifs. Returns (n_seqs, n_motifs) count matrix."""
    motif_names = list(motif_dict.keys())
    compiled = {name: re.compile(info['pattern']) for name, info in motif_dict.items()}
    counts = np.zeros((len(sequences), len(motif_names)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, name in enumerate(motif_names):
            matches = compiled[name].findall(seq)
            counts[i, j] = len(matches)
            # Also scan reverse complement
            rc_seq = reverse_complement(seq)
            matches_rc = compiled[name].findall(rc_seq)
            counts[i, j] += len(matches_rc)
    return counts, motif_names


def reverse_complement(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(nt, 'N') for nt in reversed(seq))


def load_model(checkpoint_path, config_path, device):
    with open(config_path) as f:
        config = json.load(f)
    model_type = config.get('model', 'dream_rnn')
    n_bins = config.get('n_bins', 10)
    n_domains = config.get('n_domains', 10)

    if model_type == 'dream_rnn':
        n_out = n_bins if config.get('loss') == 'soft_classification' else 1
        model = DREAM_RNN(n_outputs=n_out)
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

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, config


def extract_embeddings(model, X, device, batch_size=512):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            if hasattr(model, 'get_embeddings'):
                emb = model.get_embeddings(batch)
            elif hasattr(model, 'backbone'):
                emb = model.backbone.get_embeddings(batch)
            else:
                if hasattr(model, 'first_block'):
                    x = model.first_block(batch)
                    x = model.core_block(x)
                    x = model.pointwise_conv(x)
                    x = model.global_avg_pool(x)
                    emb = x.squeeze(-1)
                else:
                    raise RuntimeError(f"Cannot extract embeddings from {type(model)}")
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def load_test_data(data_path):
    with h5py.File(data_path, 'r') as f:
        X = f['Test/X'][:].astype(np.float32)
        y_raw = f['Test/y'][:].astype(np.float32)
        y = y_raw[:, 0] if y_raw.ndim > 1 else y_raw
    X = np.transpose(X, (0, 2, 1))  # (batch, 4, seq_len)
    return X, y


def find_experiments(results_dir):
    results_dir = Path(results_dir)
    experiments = []
    skip = ['B2_soft_classification']
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        checkpoint = exp_dir / 'checkpoints' / 'best_model.pth'
        config = exp_dir / 'config.json'
        if checkpoint.exists() and config.exists():
            name = exp_dir.name.rsplit('_202', 1)[0]
            if any(s in name for s in skip):
                continue
            experiments.append({
                'name': name,
                'dir': exp_dir,
                'checkpoint': checkpoint,
                'config': config,
            })
    seen = {}
    for exp in experiments:
        seen[exp['name']] = exp
    return list(seen.values())


# ── Analysis 1: Motif Frequency vs Activity Correlation ─────────────────────

def analyze_motif_activity_correlation(sequences, y, motif_counts, motif_names):
    """Correlate motif presence with activity."""
    print("\n" + "=" * 60)
    print("Analysis 1: Motif-Activity Correlations")
    print("=" * 60)

    results = []
    for j, name in enumerate(motif_names):
        counts = motif_counts[:, j]
        present = (counts > 0).astype(float)
        n_present = int(present.sum())
        freq = n_present / len(sequences)

        if n_present > 10 and n_present < len(sequences) - 10:
            sp, _ = spearmanr(counts, y)
            # Mean activity difference
            mean_with = y[counts > 0].mean()
            mean_without = y[counts == 0].mean()
            delta = mean_with - mean_without
        else:
            sp = np.nan
            delta = np.nan

        motif_info = {**MOTIFS, **GRAMMAR_PATTERNS}.get(name, {})
        results.append({
            'motif': name,
            'cell_type': motif_info.get('cell_type', '?'),
            'frequency': freq,
            'n_sequences': n_present,
            'spearman_with_activity': sp,
            'mean_activity_delta': delta,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('spearman_with_activity', key=abs, ascending=False)
    print(df.to_string(index=False))
    return df


# ── Analysis 2: Linear Probing for Motif Features ──────────────────────────

def probe_motifs(experiments, device, X_k562, y_k562, X_hepg2, y_hepg2,
                 motif_counts_k562, motif_names_k562,
                 motif_counts_hepg2, motif_names_hepg2, n_samples):
    """Probe: can embeddings predict motif presence/count?"""
    print("\n" + "=" * 60)
    print("Analysis 2: Linear Probing for Motif Features")
    print("=" * 60)

    all_results = []

    for exp in experiments:
        print(f"  Probing: {exp['name']}")
        model, config = load_model(exp['checkpoint'], exp['config'], device)

        is_hepg2 = 'HepG2' in exp['name']
        X = X_hepg2 if is_hepg2 else X_k562
        y = y_hepg2 if is_hepg2 else y_k562
        motif_counts = motif_counts_hepg2 if is_hepg2 else motif_counts_k562
        motif_names = motif_names_hepg2 if is_hepg2 else motif_names_k562

        emb = extract_embeddings(model, X, device)

        # Split train/test
        n = len(emb)
        idx = np.random.RandomState(42).permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = idx[:split], idx[split:]

        scaler = StandardScaler()
        emb_train = scaler.fit_transform(emb[train_idx])
        emb_test = scaler.transform(emb[test_idx])

        for j, mname in enumerate(motif_names):
            counts = motif_counts[:, j]

            # Skip motifs too rare or too common
            present = (counts > 0).sum()
            if present < 50 or present > len(counts) - 50:
                continue

            # Regression probe: predict count
            target = counts
            ridge = Ridge(alpha=1.0)
            ridge.fit(emb_train, target[train_idx])
            pred = ridge.predict(emb_test)
            r2 = r2_score(target[test_idx], pred)

            # Classification probe: predict presence (binary)
            binary = (counts > 0).astype(int)
            lr = LogisticRegression(max_iter=500, C=1.0)
            lr.fit(emb_train, binary[train_idx])
            acc = accuracy_score(binary[test_idx], lr.predict(emb_test))

            motif_info = {**MOTIFS, **GRAMMAR_PATTERNS}.get(mname, {})
            all_results.append({
                'experiment': exp['name'],
                'motif': mname,
                'motif_cell_type': motif_info.get('cell_type', '?'),
                'count_r2': r2,
                'presence_accuracy': acc,
                'motif_frequency': present / len(counts),
            })

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    return df


# ── Analysis 3: Motif-Conditioned Embedding Analysis ────────────────────────

def motif_embedding_analysis(experiments, device, X, y, sequences,
                             motif_counts, motif_names, cell_label):
    """Compare embeddings of motif-containing vs non-containing sequences."""
    print(f"\n{'=' * 60}")
    print(f"Analysis 3: Motif-Conditioned Embeddings ({cell_label})")
    print("=" * 60)

    # Select a few key motifs with reasonable frequency
    key_motifs = []
    for j, name in enumerate(motif_names):
        n_present = (motif_counts[:, j] > 0).sum()
        if 100 < n_present < len(motif_counts) - 100:
            key_motifs.append((j, name))
    key_motifs = key_motifs[:8]  # limit

    if not key_motifs:
        print("  No motifs with sufficient variation, skipping.")
        return

    # Use a representative model (first non-HepG2 for K562, first HepG2 for HepG2)
    target_exps = [e for e in experiments
                   if ('HepG2' in e['name']) == (cell_label == 'HepG2')]
    if not target_exps:
        return

    # Pick MSE baseline if available
    baseline = [e for e in target_exps if 'baseline' in e['name']]
    ranknet = [e for e in target_exps if 'ranknet' in e['name'] or 'softsort' in e['name']]
    selected = (baseline + ranknet + target_exps)[:2]

    for exp in selected:
        print(f"  Model: {exp['name']}")
        model, config = load_model(exp['checkpoint'], exp['config'], device)
        emb = extract_embeddings(model, X, device)

        # For each motif, compute mean embedding distance between
        # motif-present and motif-absent groups
        for j, mname in key_motifs:
            present = motif_counts[:, j] > 0
            emb_with = emb[present].mean(axis=0)
            emb_without = emb[~present].mean(axis=0)
            cosine_sim = np.dot(emb_with, emb_without) / (
                np.linalg.norm(emb_with) * np.linalg.norm(emb_without) + 1e-8
            )
            l2_dist = np.linalg.norm(emb_with - emb_without)
            print(f"    {mname}: cosine_sim={cosine_sim:.4f}, "
                  f"L2_dist={l2_dist:.4f}, "
                  f"n_with={present.sum()}, n_without=(~present).sum()")

        del model
        torch.cuda.empty_cache()


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_motif_probing_results(probe_df, motif_activity_df_k562, motif_activity_df_hepg2):
    """Create comprehensive motif probing plots."""

    # --- Plot 1: Motif count R² heatmap ---
    if len(probe_df) > 0:
        pivot = probe_df.pivot_table(
            index='experiment', columns='motif', values='count_r2', aggfunc='first'
        )
        # Sort experiments and motifs
        if len(pivot) > 1 and len(pivot.columns) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(pivot.fillna(0), annot=True, fmt='.3f', cmap='RdYlBu_r',
                        center=0, ax=ax, annot_kws={'fontsize': 7})
            ax.set_title('Motif Count Prediction from Embeddings (Ridge R²)')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'motif_probe_r2_heatmap.png', dpi=150)
            plt.close()
            print(f"  Saved: {OUTPUT_DIR / 'motif_probe_r2_heatmap.png'}")

        # --- Plot 2: Motif presence accuracy heatmap ---
        pivot_acc = probe_df.pivot_table(
            index='experiment', columns='motif', values='presence_accuracy', aggfunc='first'
        )
        if len(pivot_acc) > 1 and len(pivot_acc.columns) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(pivot_acc.fillna(0.5), annot=True, fmt='.3f', cmap='RdYlBu_r',
                        vmin=0.5, vmax=1.0, ax=ax, annot_kws={'fontsize': 7})
            ax.set_title('Motif Presence Classification Accuracy from Embeddings')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'motif_probe_accuracy_heatmap.png', dpi=150)
            plt.close()
            print(f"  Saved: {OUTPUT_DIR / 'motif_probe_accuracy_heatmap.png'}")

    # --- Plot 3: Motif-activity correlation bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, df, title in [
        (axes[0], motif_activity_df_k562, 'K562'),
        (axes[1], motif_activity_df_hepg2, 'HepG2'),
    ]:
        df_plot = df.dropna(subset=['spearman_with_activity']).copy()
        df_plot = df_plot.sort_values('spearman_with_activity')
        colors = []
        for _, row in df_plot.iterrows():
            ct = row['cell_type']
            if ct == 'K562':
                colors.append('#e74c3c')
            elif ct == 'HepG2':
                colors.append('#3498db')
            else:
                colors.append('#95a5a6')
        ax.barh(df_plot['motif'], df_plot['spearman_with_activity'], color=colors)
        ax.set_xlabel('Spearman with Activity')
        ax.set_title(f'Motif-Activity Correlation ({title} Test Set)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'motif_activity_correlation.png', dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'motif_activity_correlation.png'}")

    # --- Plot 4: K562 vs HepG2 model comparison for cell-type motifs ---
    if len(probe_df) > 0:
        # For each motif, compare R² between K562 and HepG2 models
        k562_exps = probe_df[~probe_df['experiment'].str.contains('HepG2')]
        hepg2_exps = probe_df[probe_df['experiment'].str.contains('HepG2')]

        if len(k562_exps) > 0 and len(hepg2_exps) > 0:
            k562_mean = k562_exps.groupby('motif')['count_r2'].mean()
            hepg2_mean = hepg2_exps.groupby('motif')['count_r2'].mean()

            shared_motifs = sorted(set(k562_mean.index) & set(hepg2_mean.index))
            if shared_motifs:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(shared_motifs))
                width = 0.35
                ax.bar(x - width/2, [k562_mean.get(m, 0) for m in shared_motifs],
                       width, label='K562 models', color='#e74c3c', alpha=0.8)
                ax.bar(x + width/2, [hepg2_mean.get(m, 0) for m in shared_motifs],
                       width, label='HepG2 models', color='#3498db', alpha=0.8)
                ax.set_xlabel('Motif')
                ax.set_ylabel('Mean R² (count prediction)')
                ax.set_title('Motif Encoding: K562 vs HepG2 Models')
                ax.set_xticks(x)
                ax.set_xticklabels(shared_motifs, rotation=45, ha='right', fontsize=8)
                ax.legend()
                ax.axhline(y=0, color='black', linewidth=0.5)
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / 'motif_k562_vs_hepg2.png', dpi=150)
                plt.close()
                print(f"  Saved: {OUTPUT_DIR / 'motif_k562_vs_hepg2.png'}")

    # --- Plot 5: Per-model motif R² comparison (MSE vs ranking losses) ---
    if len(probe_df) > 0:
        # Group by loss type
        def get_loss_type(name):
            if 'baseline' in name:
                return 'MSE'
            elif 'plackett' in name:
                return 'Plackett-Luce'
            elif 'softsort' in name:
                return 'SoftSort'
            elif 'ranknet' in name:
                return 'RankNet'
            elif 'combined' in name or 'dual' in name:
                return 'Combined'
            elif 'domain' in name or 'bias' in name or 'advanced' in name:
                return 'Advanced'
            elif 'curriculum' in name:
                return 'Curriculum'
            return 'Other'

        probe_df['loss_type'] = probe_df['experiment'].apply(get_loss_type)

        # Average R² per loss type per motif
        loss_motif = probe_df.groupby(['loss_type', 'motif'])['count_r2'].mean().reset_index()
        pivot_loss = loss_motif.pivot(index='loss_type', columns='motif', values='count_r2')

        if len(pivot_loss) > 1 and len(pivot_loss.columns) > 1:
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(pivot_loss.fillna(0), annot=True, fmt='.3f', cmap='RdYlBu_r',
                        center=0, ax=ax, annot_kws={'fontsize': 8})
            ax.set_title('Motif Encoding by Loss Type (Mean R²)')
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'motif_by_loss_type.png', dpi=150)
            plt.close()
            print(f"  Saved: {OUTPUT_DIR / 'motif_by_loss_type.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    experiments = find_experiments(args.results_dir)
    print(f"Found {len(experiments)} experiments")
    for e in experiments:
        print(f"  - {e['name']}")

    # Load data
    print("\nLoading test data...")
    k562_X, k562_y = load_test_data(
        'data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5'
    )
    hepg2_X, hepg2_y = load_test_data(
        'data/raw/dream_rnn_lentimpra/data/lentiMPRA_HepG2_activity_data.h5'
    )

    # Subsample
    rng = np.random.default_rng(42)
    n = args.n_samples
    k562_idx = rng.choice(len(k562_X), min(n, len(k562_X)), replace=False)
    hepg2_idx = rng.choice(len(hepg2_X), min(n, len(hepg2_X)), replace=False)

    k562_X_sub = k562_X[k562_idx]
    k562_y_sub = k562_y[k562_idx]
    hepg2_X_sub = hepg2_X[hepg2_idx]
    hepg2_y_sub = hepg2_y[hepg2_idx]

    # Convert to sequences
    print("Converting to nucleotide sequences and scanning motifs...")
    k562_seqs = [onehot_to_sequence(x) for x in k562_X_sub]
    hepg2_seqs = [onehot_to_sequence(x) for x in hepg2_X_sub]

    # Combine individual motifs and grammar patterns
    all_motifs = {**MOTIFS, **GRAMMAR_PATTERNS}

    k562_counts, k562_motif_names = scan_motifs(k562_seqs, all_motifs)
    hepg2_counts, hepg2_motif_names = scan_motifs(hepg2_seqs, all_motifs)

    # Print motif frequency summary
    print("\n  K562 motif frequencies:")
    for j, name in enumerate(k562_motif_names):
        freq = (k562_counts[:, j] > 0).mean()
        mean_count = k562_counts[:, j].mean()
        print(f"    {name:25s}: {freq:.3f} ({(k562_counts[:, j] > 0).sum():5d} seqs), "
              f"mean count={mean_count:.2f}")

    print("\n  HepG2 motif frequencies:")
    for j, name in enumerate(hepg2_motif_names):
        freq = (hepg2_counts[:, j] > 0).mean()
        mean_count = hepg2_counts[:, j].mean()
        print(f"    {name:25s}: {freq:.3f} ({(hepg2_counts[:, j] > 0).sum():5d} seqs), "
              f"mean count={mean_count:.2f}")

    # Analysis 1: Motif-activity correlations
    print("\n--- K562 ---")
    motif_act_k562 = analyze_motif_activity_correlation(
        k562_seqs, k562_y_sub, k562_counts, k562_motif_names
    )
    print("\n--- HepG2 ---")
    motif_act_hepg2 = analyze_motif_activity_correlation(
        hepg2_seqs, hepg2_y_sub, hepg2_counts, hepg2_motif_names
    )

    # Analysis 2: Linear probing
    probe_df = probe_motifs(
        experiments, device,
        k562_X_sub, k562_y_sub, hepg2_X_sub, hepg2_y_sub,
        k562_counts, k562_motif_names,
        hepg2_counts, hepg2_motif_names,
        n
    )

    # Save probe results
    probe_df.to_csv(OUTPUT_DIR / 'motif_probing.csv', index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'motif_probing.csv'}")

    motif_act_k562.to_csv(OUTPUT_DIR / 'motif_activity_k562.csv', index=False)
    motif_act_hepg2.to_csv(OUTPUT_DIR / 'motif_activity_hepg2.csv', index=False)

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary: Motif Probe R² (count prediction from embeddings)")
    print("=" * 60)
    if len(probe_df) > 0:
        summary = probe_df.pivot_table(
            index='experiment', columns='motif', values='count_r2', aggfunc='first'
        )
        print(summary.round(3).to_string())

    # Analysis 3: Motif-conditioned embeddings
    motif_embedding_analysis(
        experiments, device, k562_X_sub, k562_y_sub, k562_seqs,
        k562_counts, k562_motif_names, 'K562'
    )
    motif_embedding_analysis(
        experiments, device, hepg2_X_sub, hepg2_y_sub, hepg2_seqs,
        hepg2_counts, hepg2_motif_names, 'HepG2'
    )

    # Plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    plot_motif_probing_results(probe_df, motif_act_k562, motif_act_hepg2)

    print("\n" + "=" * 60)
    print("All motif probing analyses complete.")
    print(f"Results saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
