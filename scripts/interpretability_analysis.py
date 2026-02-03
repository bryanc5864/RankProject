#!/usr/bin/env python3
"""
Interpretability analysis for trained MPRA models.

Analyses:
1. UMAP of learned embeddings (colored by activity, cell type)
2. CKA similarity between all model pairs
3. Integrated Gradients attribution maps
4. Linear probing: what do embeddings encode?
5. DeepSHAP nucleotide importance

Usage:
    python scripts/interpretability_analysis.py --gpu 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import umap
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    DREAM_RNN, DREAM_RNN_SingleOutput, DREAM_RNN_DualHead,
    DREAM_RNN_DomainAdversarial, DREAM_RNN_BiasFactorized, DREAM_RNN_FullAdvanced
)

OUTPUT_DIR = Path('results/interpretability')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    """Extract 256-dim embeddings from the shared backbone."""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            # All models share the same backbone structure
            if hasattr(model, 'get_embeddings'):
                emb = model.get_embeddings(batch)
            elif hasattr(model, 'backbone'):
                emb = model.backbone.get_embeddings(batch)
            else:
                # Manual extraction for models without get_embeddings
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


def get_predictions(model, X, device, batch_size=512):
    """Get scalar predictions from any model."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze()
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def load_test_data(data_path):
    """Load test set X and y from HDF5."""
    with h5py.File(data_path, 'r') as f:
        X = f['Test/X'][:].astype(np.float32)
        y_raw = f['Test/y'][:].astype(np.float32)
        y = y_raw[:, 0] if y_raw.ndim > 1 else y_raw
    X = np.transpose(X, (0, 2, 1))  # (batch, 4, seq_len)
    return X, y


def find_experiments(results_dir):
    """Find all valid experiments, return list of dicts."""
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
    # Deduplicate: keep latest per experiment name
    seen = {}
    for exp in experiments:
        seen[exp['name']] = exp
    return list(seen.values())


# ── Analysis 1: UMAP of Embeddings ──────────────────────────────────────────

def run_umap_analysis(experiments, device, n_samples=3000):
    """UMAP visualization of embeddings for each model, colored by activity."""
    print("\n" + "=" * 60)
    print("Analysis 1: UMAP of Learned Embeddings")
    print("=" * 60)

    # Load K562 and HepG2 test data
    k562_X, k562_y = load_test_data(
        'data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5'
    )
    hepg2_X, hepg2_y = load_test_data(
        'data/raw/dream_rnn_lentimpra/data/lentiMPRA_HepG2_activity_data.h5'
    )

    # Subsample for speed
    rng = np.random.default_rng(42)
    k562_idx = rng.choice(len(k562_X), min(n_samples, len(k562_X)), replace=False)
    hepg2_idx = rng.choice(len(hepg2_X), min(n_samples, len(hepg2_X)), replace=False)

    k562_X_sub, k562_y_sub = k562_X[k562_idx], k562_y[k562_idx]
    hepg2_X_sub, hepg2_y_sub = hepg2_X[hepg2_idx], hepg2_y[hepg2_idx]

    # Separate K562 and HepG2 experiments
    k562_exps = [e for e in experiments if 'HepG2' not in e['name']]
    hepg2_exps = [e for e in experiments if 'HepG2' in e['name']]

    for exps, X_sub, y_sub, cell_label in [
        (k562_exps, k562_X_sub, k562_y_sub, 'K562'),
        (hepg2_exps, hepg2_X_sub, hepg2_y_sub, 'HepG2'),
    ]:
        if not exps:
            continue

        n_models = len(exps)
        cols = min(4, n_models)
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, exp in enumerate(exps):
            print(f"  UMAP: {exp['name']}")
            try:
                model, config = load_model(exp['checkpoint'], exp['config'], device)
                emb = extract_embeddings(model, X_sub, device)

                reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
                umap_coords = reducer.fit_transform(emb)

                ax = axes[idx]
                scatter = ax.scatter(
                    umap_coords[:, 0], umap_coords[:, 1],
                    c=y_sub, cmap='viridis', s=1, alpha=0.5, rasterized=True
                )
                ax.set_title(exp['name'], fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(scatter, ax=ax, label='Activity', shrink=0.8)

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    Error: {e}")
                axes[idx].set_title(f"{exp['name']}\n(failed)", fontsize=9)

        # Hide unused axes
        for idx in range(len(exps), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'UMAP of Embeddings — {cell_label} Models\n(colored by activity)',
                     fontsize=14, y=1.02)
        fig.tight_layout()
        path = OUTPUT_DIR / f'umap_embeddings_{cell_label}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    # Cross-cell-type UMAP: embed K562 and HepG2 test data through same model
    print("\n  Cross-cell-type UMAP (best K562 model on both datasets)...")
    best_k562 = [e for e in k562_exps if 'R3_ranknet' in e['name']]
    if not best_k562:
        best_k562 = k562_exps[:1]
    if best_k562:
        exp = best_k562[0]
        model, _ = load_model(exp['checkpoint'], exp['config'], device)

        k562_emb = extract_embeddings(model, k562_X_sub, device)
        hepg2_emb = extract_embeddings(model, hepg2_X_sub, device)

        combined_emb = np.vstack([k562_emb, hepg2_emb])
        combined_y = np.concatenate([k562_y_sub, hepg2_y_sub])
        cell_labels = np.array(['K562'] * len(k562_emb) + ['HepG2'] * len(hepg2_emb))

        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
        umap_coords = reducer.fit_transform(combined_emb)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Color by cell type
        for ct, color in [('K562', 'tab:blue'), ('HepG2', 'tab:orange')]:
            mask = cell_labels == ct
            ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                        c=color, s=1, alpha=0.4, label=ct, rasterized=True)
        ax1.legend(markerscale=5)
        ax1.set_title(f'{exp["name"]} — colored by cell type')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Color by activity
        scatter = ax2.scatter(umap_coords[:, 0], umap_coords[:, 1],
                              c=combined_y, cmap='viridis', s=1, alpha=0.4, rasterized=True)
        plt.colorbar(scatter, ax=ax2, label='Activity')
        ax2.set_title(f'{exp["name"]} — colored by activity')
        ax2.set_xticks([])
        ax2.set_yticks([])

        fig.tight_layout()
        path = OUTPUT_DIR / 'umap_cross_cell_type.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

        del model
        torch.cuda.empty_cache()


# ── Analysis 2: CKA Similarity ──────────────────────────────────────────────

def linear_cka(X, Y):
    """Compute linear CKA between two representation matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def run_cka_analysis(experiments, device, n_samples=2000):
    """Compute CKA similarity matrix between all model pairs."""
    print("\n" + "=" * 60)
    print("Analysis 2: CKA Representation Similarity")
    print("=" * 60)

    # Use K562 test data as common reference
    X_test, y_test = load_test_data(
        'data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5'
    )
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sub = X_test[idx]

    # Extract embeddings for all models
    all_embeddings = {}
    names = []
    for exp in experiments:
        print(f"  Extracting: {exp['name']}")
        try:
            model, _ = load_model(exp['checkpoint'], exp['config'], device)
            emb = extract_embeddings(model, X_sub, device)
            all_embeddings[exp['name']] = emb
            names.append(exp['name'])
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Error: {e}")

    # Compute pairwise CKA
    n = len(names)
    cka_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_matrix[i, j] = linear_cka(all_embeddings[names[i]],
                                            all_embeddings[names[j]])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cka_matrix, xticklabels=names, yticklabels=names,
                annot=True, fmt='.2f', cmap='RdYlBu_r', vmin=0, vmax=1,
                square=True, ax=ax, annot_kws={'fontsize': 7})
    ax.set_title('Linear CKA Representation Similarity', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / 'cka_similarity_matrix.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save numeric results
    cka_df = pd.DataFrame(cka_matrix, index=names, columns=names)
    cka_df.to_csv(OUTPUT_DIR / 'cka_similarity.csv')
    print(f"  Saved: {OUTPUT_DIR / 'cka_similarity.csv'}")

    return cka_matrix, names


# ── Analysis 3: Integrated Gradients Attribution ─────────────────────────────

def run_attribution_analysis(experiments, device, n_seqs=50):
    """Integrated Gradients attribution for representative sequences."""
    print("\n" + "=" * 60)
    print("Analysis 3: Integrated Gradients Attribution Maps")
    print("=" * 60)

    from captum.attr import IntegratedGradients

    X_test, y_test = load_test_data(
        'data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5'
    )

    # Pick sequences spanning the activity range
    sorted_idx = np.argsort(y_test)
    # Sample evenly across quantiles
    pick_idx = sorted_idx[np.linspace(0, len(sorted_idx) - 1, n_seqs, dtype=int)]
    X_sub = X_test[pick_idx]
    y_sub = y_test[pick_idx]

    # Select a subset of models to compare
    target_names = ['B1_baseline_mse', 'R1_plackett_luce', 'R3_ranknet',
                    'R2_softsort', 'R4_combined']
    selected = [e for e in experiments if e['name'] in target_names]
    if not selected:
        selected = experiments[:5]

    all_attrs = {}
    for exp in selected:
        print(f"  IG: {exp['name']}")
        try:
            model, config = load_model(exp['checkpoint'], exp['config'], device)

            # Wrap model to return scalar
            class ScalarWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    out = self.m(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    if out.dim() > 1:
                        out = out.squeeze(-1)
                    return out

            wrapped = ScalarWrapper(model)
            ig = IntegratedGradients(wrapped)

            input_tensor = torch.FloatTensor(X_sub).to(device)
            input_tensor.requires_grad_(True)
            baseline = torch.zeros_like(input_tensor).to(device)

            # Disable cuDNN for RNN backward compatibility in eval mode
            prev_cudnn = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False

            # Compute attribution for each sequence individually
            attrs = []
            for i in range(len(X_sub)):
                attr = ig.attribute(
                    input_tensor[i:i+1], baselines=baseline[i:i+1],
                    n_steps=50, internal_batch_size=1
                )
                attrs.append(attr.detach().cpu().numpy())
            torch.backends.cudnn.enabled = prev_cudnn
            attrs = np.concatenate(attrs, axis=0)  # (n_seqs, 4, 230)
            all_attrs[exp['name']] = attrs

            del model, wrapped
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Error: {e}")

    if not all_attrs:
        print("  No attributions computed, skipping plots.")
        return

    # Plot: attribution heatmaps for high/medium/low activity sequences
    activity_groups = {
        'Low activity': np.argsort(y_sub)[:5],
        'Medium activity': np.argsort(np.abs(y_sub - np.median(y_sub)))[:5],
        'High activity': np.argsort(y_sub)[-5:],
    }

    model_names = list(all_attrs.keys())
    for group_name, group_idx in activity_groups.items():
        fig, axes = plt.subplots(len(model_names), 1,
                                 figsize=(14, 2.5 * len(model_names)))
        if len(model_names) == 1:
            axes = [axes]

        for ax_idx, mname in enumerate(model_names):
            # Average attribution across sequences in group, sum across nucleotides
            attr_group = all_attrs[mname][group_idx]  # (5, 4, 230)
            attr_importance = np.mean(np.abs(attr_group), axis=0)  # (4, 230)

            axes[ax_idx].imshow(attr_importance, aspect='auto', cmap='Reds',
                                interpolation='none')
            axes[ax_idx].set_ylabel(mname, fontsize=8, rotation=0,
                                    ha='right', va='center')
            axes[ax_idx].set_yticks([0, 1, 2, 3])
            axes[ax_idx].set_yticklabels(['A', 'C', 'G', 'T'], fontsize=7)
            if ax_idx < len(model_names) - 1:
                axes[ax_idx].set_xticks([])

        axes[-1].set_xlabel('Position (bp)')
        fig.suptitle(f'Integrated Gradients — {group_name}', fontsize=12)
        fig.tight_layout()
        safe_name = group_name.replace(' ', '_').lower()
        path = OUTPUT_DIR / f'ig_attributions_{safe_name}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    # Plot: attribution magnitude profile (position-wise, summed over nucleotides)
    fig, axes = plt.subplots(len(model_names), 1,
                             figsize=(14, 2 * len(model_names)), sharex=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax_idx, mname in enumerate(model_names):
        # Average |attribution| across all sequences and nucleotides -> per-position
        attr_all = all_attrs[mname]  # (n_seqs, 4, 230)
        pos_importance = np.mean(np.sum(np.abs(attr_all), axis=1), axis=0)  # (230,)
        axes[ax_idx].fill_between(range(230), pos_importance, alpha=0.7)
        axes[ax_idx].set_ylabel(mname, fontsize=8, rotation=0,
                                ha='right', va='center')
        axes[ax_idx].set_xlim(0, 229)

    axes[-1].set_xlabel('Position (bp)')
    fig.suptitle('Per-Position Attribution Magnitude (averaged over test sequences)',
                 fontsize=12)
    fig.tight_layout()
    path = OUTPUT_DIR / 'ig_position_profiles.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Analysis: attribution similarity between models
    print("\n  Attribution correlation between models:")
    # Flatten attributions and compute pairwise Pearson
    attr_corr = np.zeros((len(model_names), len(model_names)))
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            a1 = all_attrs[m1].flatten()
            a2 = all_attrs[m2].flatten()
            attr_corr[i, j] = pearsonr(a1, a2)[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attr_corr, xticklabels=model_names, yticklabels=model_names,
                annot=True, fmt='.3f', cmap='RdYlBu_r', vmin=-1, vmax=1,
                square=True, ax=ax, annot_kws={'fontsize': 9})
    ax.set_title('Attribution Correlation Between Models (Pearson)')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    path = OUTPUT_DIR / 'ig_attribution_correlation.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Analysis 4: Linear Probing ──────────────────────────────────────────────

def run_probing_analysis(experiments, device, n_samples=5000):
    """Probe what information embeddings encode."""
    print("\n" + "=" * 60)
    print("Analysis 4: Linear Probing of Representations")
    print("=" * 60)

    X_test, y_test = load_test_data(
        'data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5'
    )
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sub, y_sub = X_test[idx], y_test[idx]

    # Compute sequence-level features for probing
    # GC content
    gc_content = (X_sub[:, 1, :] + X_sub[:, 2, :]).mean(axis=1)  # C + G channels
    # Dinucleotide frequencies (simplified: just CpG)
    cpg_freq = np.array([
        (X_sub[i, 1, :-1] * X_sub[i, 2, 1:]).sum() / (X_sub.shape[2] - 1)
        for i in range(len(X_sub))
    ])

    results = []

    for exp in experiments:
        if 'HepG2' in exp['name']:
            continue  # Probe with K562 data only for K562 models

        print(f"  Probing: {exp['name']}")
        try:
            model, _ = load_model(exp['checkpoint'], exp['config'], device)
            emb = extract_embeddings(model, X_sub, device)
            scaler = StandardScaler()
            emb_scaled = scaler.fit_transform(emb)

            # Probe 1: Activity prediction (R² via Ridge regression)
            ridge = Ridge(alpha=1.0)
            activity_r2 = cross_val_score(ridge, emb_scaled, y_sub,
                                          cv=5, scoring='r2').mean()

            # Probe 2: Activity rank prediction
            y_ranks = y_sub.argsort().argsort().astype(float)
            rank_r2 = cross_val_score(ridge, emb_scaled, y_ranks,
                                      cv=5, scoring='r2').mean()

            # Probe 3: GC content prediction
            gc_r2 = cross_val_score(ridge, emb_scaled, gc_content,
                                    cv=5, scoring='r2').mean()

            # Probe 4: CpG frequency prediction
            cpg_r2 = cross_val_score(ridge, emb_scaled, cpg_freq,
                                     cv=5, scoring='r2').mean()

            # Probe 5: High vs Low activity classification
            y_binary = (y_sub > np.median(y_sub)).astype(int)
            lr = LogisticRegression(max_iter=500, C=1.0)
            class_acc = cross_val_score(lr, emb_scaled, y_binary,
                                        cv=5, scoring='accuracy').mean()

            # Probe 6: Random labels (noise ceiling)
            y_random = rng.permutation(y_sub)
            random_r2 = cross_val_score(ridge, emb_scaled, y_random,
                                        cv=5, scoring='r2').mean()

            row = {
                'experiment': exp['name'],
                'activity_r2': activity_r2,
                'rank_r2': rank_r2,
                'gc_content_r2': gc_r2,
                'cpg_freq_r2': cpg_r2,
                'highlow_accuracy': class_acc,
                'random_r2': random_r2,
            }
            results.append(row)
            print(f"    Activity R²={activity_r2:.4f}, GC R²={gc_r2:.4f}, "
                  f"Random R²={random_r2:.4f}")

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Error: {e}")

    if not results:
        return

    probe_df = pd.DataFrame(results)
    probe_df.to_csv(OUTPUT_DIR / 'linear_probing.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'linear_probing.csv'}")

    # Plot probing results
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Plot 1: Activity R² comparison
    probe_df_sorted = probe_df.sort_values('activity_r2', ascending=True)
    axes[0].barh(range(len(probe_df_sorted)), probe_df_sorted['activity_r2'], color='steelblue')
    axes[0].set_yticks(range(len(probe_df_sorted)))
    axes[0].set_yticklabels(probe_df_sorted['experiment'], fontsize=8)
    axes[0].set_xlabel('R²')
    axes[0].set_title('Activity Prediction from Embeddings')
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: GC content R² (noise indicator)
    probe_df_sorted2 = probe_df.sort_values('gc_content_r2', ascending=True)
    colors = ['salmon' if v > 0.5 else 'steelblue' for v in probe_df_sorted2['gc_content_r2']]
    axes[1].barh(range(len(probe_df_sorted2)), probe_df_sorted2['gc_content_r2'], color=colors)
    axes[1].set_yticks(range(len(probe_df_sorted2)))
    axes[1].set_yticklabels(probe_df_sorted2['experiment'], fontsize=8)
    axes[1].set_xlabel('R²')
    axes[1].set_title('GC Content Prediction\n(high = potential noise)')

    # Plot 3: Multi-probe comparison
    probes = ['activity_r2', 'rank_r2', 'gc_content_r2', 'cpg_freq_r2']
    probe_labels = ['Activity', 'Rank', 'GC Content', 'CpG Freq']
    x = np.arange(len(probe_df))
    width = 0.2
    for i, (probe, label) in enumerate(zip(probes, probe_labels)):
        axes[2].bar(x + i * width, probe_df[probe], width, label=label, alpha=0.8)
    axes[2].set_xticks(x + width * 1.5)
    axes[2].set_xticklabels(probe_df['experiment'], rotation=45, ha='right', fontsize=7)
    axes[2].set_ylabel('R²')
    axes[2].set_title('What Do Embeddings Encode?')
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / 'linear_probing.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Analysis 5: Prediction Scatter Comparison ────────────────────────────────

def run_prediction_analysis(experiments, device, n_samples=5000):
    """Compare model predictions: scatter plots and residual analysis."""
    print("\n" + "=" * 60)
    print("Analysis 5: Prediction Comparison & Residuals")
    print("=" * 60)

    X_test, y_test = load_test_data(
        'data/raw/dream_rnn_lentimpra/data/lentiMPRA_K562_activity_and_aleatoric_data.h5'
    )
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sub, y_sub = X_test[idx], y_test[idx]

    k562_exps = [e for e in experiments if 'HepG2' not in e['name']]
    target_names = ['B1_baseline_mse', 'R1_plackett_luce', 'R3_ranknet',
                    'R2_softsort', 'R4_combined', 'R5_dual_combined']
    selected = [e for e in k562_exps if e['name'] in target_names]
    if not selected:
        selected = k562_exps[:6]

    all_preds = {}
    for exp in selected:
        print(f"  Predicting: {exp['name']}")
        try:
            model, _ = load_model(exp['checkpoint'], exp['config'], device)
            preds = get_predictions(model, X_sub, device)
            all_preds[exp['name']] = preds
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Error: {e}")

    if len(all_preds) < 2:
        return

    model_names = list(all_preds.keys())

    # Plot 1: Pred vs True scatter
    n_models = len(model_names)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten()

    for i, mname in enumerate(model_names):
        preds = all_preds[mname]
        sp = spearmanr(preds, y_sub)[0]
        pr = pearsonr(preds, y_sub)[0]
        axes[i].scatter(y_sub, preds, s=1, alpha=0.3, rasterized=True)
        axes[i].set_xlabel('True Activity')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{mname}\nSpearman={sp:.4f}, Pearson={pr:.4f}', fontsize=9)
        # Fit line
        z = np.polyfit(y_sub, preds, 1)
        x_line = np.linspace(y_sub.min(), y_sub.max(), 100)
        axes[i].plot(x_line, np.polyval(z, x_line), 'r-', alpha=0.7)

    for i in range(len(model_names), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Predictions vs True Activity (K562 Test Set)', fontsize=13)
    fig.tight_layout()
    path = OUTPUT_DIR / 'prediction_scatter.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Plot 2: Inter-model prediction correlation
    n = len(model_names)
    pred_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pred_corr[i, j] = spearmanr(all_preds[model_names[i]],
                                          all_preds[model_names[j]])[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pred_corr, xticklabels=model_names, yticklabels=model_names,
                annot=True, fmt='.3f', cmap='RdYlBu_r', vmin=0.8, vmax=1.0,
                square=True, ax=ax)
    ax.set_title('Inter-Model Prediction Correlation (Spearman)')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    path = OUTPUT_DIR / 'prediction_correlation.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Plot 3: Residual analysis — where do models disagree?
    if 'B1_baseline_mse' in all_preds:
        baseline_preds = all_preds['B1_baseline_mse']
        fig, axes = plt.subplots(1, len(model_names) - 1,
                                 figsize=(5 * (len(model_names) - 1), 4))
        if len(model_names) == 2:
            axes = [axes]
        ax_idx = 0
        for mname in model_names:
            if mname == 'B1_baseline_mse':
                continue
            diff = all_preds[mname] - baseline_preds
            axes[ax_idx].scatter(y_sub, diff, s=1, alpha=0.3, rasterized=True)
            axes[ax_idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[ax_idx].set_xlabel('True Activity')
            axes[ax_idx].set_ylabel(f'{mname} - B1_mse')
            axes[ax_idx].set_title(f'Prediction Difference: {mname}', fontsize=9)
            ax_idx += 1

        fig.suptitle('How Ranking Models Differ from MSE Baseline', fontsize=12)
        fig.tight_layout()
        path = OUTPUT_DIR / 'residual_vs_baseline.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    experiments = find_experiments('results')
    print(f"Found {len(experiments)} experiments")
    for exp in experiments:
        print(f"  - {exp['name']}")

    # Run all analyses
    run_umap_analysis(experiments, device, n_samples=args.n_samples)
    run_cka_analysis(experiments, device, n_samples=args.n_samples)
    run_attribution_analysis(experiments, device, n_seqs=args.n_attr_seqs)
    run_probing_analysis(experiments, device, n_samples=args.n_samples)
    run_prediction_analysis(experiments, device, n_samples=args.n_samples)

    print("\n" + "=" * 60)
    print(f"All analyses complete. Results saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=3000,
                        help='Number of test samples for UMAP/CKA/probing')
    parser.add_argument('--n_attr_seqs', type=int, default=50,
                        help='Number of sequences for attribution analysis')
    main(parser.parse_args())
