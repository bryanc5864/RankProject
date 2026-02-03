#!/usr/bin/env python3
"""
Comprehensive model interpretability suite for DISENTANGLE.

Experiments:
  1. Integrated Gradients (IG) - input-level attribution maps
  2. In-Silico Mutagenesis (ISM) - effect of every single nucleotide mutation
  3. Motif discovery from attributions - extract & cluster top-attribution windows
  4. Representation geometry - PCA of latent spaces by cell type & activity
  5. Experiment-norm comparison - K562-normed vs HepG2-normed vs denoised reps
  6. BatchNorm parameter analysis - learned gamma/beta per experiment norm
  7. First-layer filter visualization - extract PWMs from conv1 filters
  8. Prediction head weight analysis - which latent dims drive predictions
  9. CKA between models - representation similarity across conditions
 10. Positional sensitivity profile - gradient magnitude across sequence
 11. High vs low activity attribution comparison
 12. Cross-experiment attribution consistency

Usage:
    python scripts/run_interpretability_suite.py --gpu 0
"""

import argparse
import json
import os
import sys
import warnings

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluate import load_model

# Key models to analyze
KEY_MODELS = {
    "bilstm": [
        "bilstm_baseline_mse_seed42",
        "bilstm_ranking_seed42",
        "bilstm_contrastive_only_seed42",
        "bilstm_ranking_contrastive_seed42",
        "bilstm_full_disentangle_seed42",
    ],
    "dilated_cnn": [
        "dilated_cnn_baseline_mse_seed42",
        "dilated_cnn_ranking_seed42",
        "dilated_cnn_contrastive_only_seed42",
        "dilated_cnn_ranking_contrastive_seed42",
        "dilated_cnn_full_disentangle_seed42",
    ],
}

N_SAMPLES = 2000  # sequences for attribution/ISM analysis
N_REPR_SAMPLES = 5000  # sequences for representation analysis


def load_test_data(data_file, n_samples=None, seed=42):
    """Load test split data."""
    with h5py.File(data_file, "r") as f:
        splits = f["split"][:]
        mask = splits == 2  # test
        seqs = f["sequences"][:][mask].astype(np.float32)
        acts = f["activities"][:][mask]
    if n_samples and n_samples < len(seqs):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(seqs), n_samples, replace=False)
        return seqs[idx], acts[idx]
    return seqs, acts


def load_paired_test_data(paired_file, n_samples=None, seed=42):
    """Load paired test data with both cell-type activities."""
    with h5py.File(paired_file, "r") as f:
        splits = f["split"][:]
        mask = splits == 2
        seqs = f["sequences"][:][mask].astype(np.float32)
        k562_acts = f["k562_activities"][:][mask]
        hepg2_acts = f["hepg2_activities"][:][mask]
        consensus = f["consensus_ranks"][:][mask]
    if n_samples and n_samples < len(seqs):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(seqs), n_samples, replace=False)
        return seqs[idx], k562_acts[idx], hepg2_acts[idx], consensus[idx]
    return seqs, k562_acts, hepg2_acts, consensus


# ============================================================================
# 1. Integrated Gradients
# ============================================================================
def compute_integrated_gradients(model, sequences, device, n_steps=50,
                                  experiment_id=None, batch_size=64):
    """Compute integrated gradients for input attribution."""
    torch.backends.cudnn.enabled = False
    model.train()  # needed for LSTM backward

    all_attrs = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.from_numpy(sequences[start:start+batch_size]).to(device)
        batch.requires_grad_(True)
        baseline = torch.zeros_like(batch)

        # Accumulate gradients along interpolation path
        attr_sum = torch.zeros_like(batch)
        for step in range(n_steps + 1):
            alpha = step / n_steps
            interp = baseline + alpha * (batch - baseline)
            interp = interp.detach().requires_grad_(True)

            if experiment_id is not None:
                out = model(interp, experiment_id=experiment_id)
            elif hasattr(model, "predict_denoised"):
                out = model.predict_denoised(interp)
            else:
                out = model(interp)

            out.sum().backward()
            attr_sum += interp.grad.detach()

        attrs = (attr_sum / (n_steps + 1)) * (batch.detach() - baseline)
        all_attrs.append(attrs.cpu().numpy())

    model.eval()
    torch.backends.cudnn.enabled = True
    return np.concatenate(all_attrs)


def run_integrated_gradients(models_dict, seqs, acts, device, output_dir):
    """Run IG for all models, compare attributions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Integrated Gradients")
    print("=" * 70)

    results = {}
    all_attrs = {}

    # Use subset for IG (expensive)
    n_ig = min(500, len(seqs))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(seqs), n_ig, replace=False)
    ig_seqs = seqs[idx]
    ig_acts = acts[idx]

    for name, (model, config) in models_dict.items():
        print(f"  Computing IG for {name}...")
        attrs = compute_integrated_gradients(model, ig_seqs, device)
        # Sum across nucleotide channels -> per-position importance
        pos_importance = np.abs(attrs).sum(axis=-1)  # [N, L]
        all_attrs[name] = attrs
        results[name] = {
            "mean_importance_per_pos": pos_importance.mean(axis=0).tolist(),
            "total_attribution_magnitude": float(np.abs(attrs).sum(axis=(1, 2)).mean()),
        }

    # Pairwise attribution correlation between models
    attr_correlations = {}
    model_names = list(all_attrs.keys())
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            a1 = np.abs(all_attrs[n1]).sum(axis=-1).flatten()
            a2 = np.abs(all_attrs[n2]).sum(axis=-1).flatten()
            r, _ = spearmanr(a1, a2)
            attr_correlations[f"{n1}_vs_{n2}"] = float(r)

    results["pairwise_attribution_correlation"] = attr_correlations

    # Save
    np.savez_compressed(
        os.path.join(output_dir, "integrated_gradients.npz"),
        **{f"attrs_{k}": v for k, v in all_attrs.items()},
        sequences=ig_seqs,
        activities=ig_acts,
    )

    print(f"  Attribution correlations:")
    for pair, r in sorted(attr_correlations.items()):
        print(f"    {pair}: {r:.4f}")

    return results


# ============================================================================
# 2. In-Silico Mutagenesis
# ============================================================================
def compute_ism(model, sequences, device, experiment_id=None, batch_size=256):
    """Compute in-silico mutagenesis: effect of every single mutation."""
    model.eval()
    n_seqs, seq_len, n_channels = sequences.shape
    ism_scores = np.zeros((n_seqs, seq_len, 4), dtype=np.float32)

    with torch.no_grad():
        # Get reference predictions
        ref_preds = []
        for i in range(0, n_seqs, batch_size):
            batch = torch.from_numpy(sequences[i:i+batch_size]).to(device)
            if hasattr(model, "predict_denoised"):
                out = model.predict_denoised(batch)
            else:
                out = model(batch)
            ref_preds.append(out.cpu().numpy())
        ref_preds = np.concatenate(ref_preds)

        # Mutate each position to each nucleotide
        for pos in range(seq_len):
            for nuc in range(4):
                mutant = sequences.copy()
                mutant[:, pos, :] = 0
                mutant[:, pos, nuc] = 1

                mut_preds = []
                for i in range(0, n_seqs, batch_size):
                    batch = torch.from_numpy(mutant[i:i+batch_size]).to(device)
                    if hasattr(model, "predict_denoised"):
                        out = model.predict_denoised(batch)
                    else:
                        out = model(batch)
                    mut_preds.append(out.cpu().numpy())
                ism_scores[:, pos, nuc] = np.concatenate(mut_preds) - ref_preds

    return ism_scores  # [N, L, 4] - change in prediction for each mutation


def run_ism_analysis(models_dict, seqs, acts, device, output_dir):
    """Run ISM for key models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: In-Silico Mutagenesis")
    print("=" * 70)

    results = {}
    n_ism = min(200, len(seqs))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(seqs), n_ism, replace=False)
    ism_seqs = seqs[idx]

    all_ism = {}
    for name, (model, config) in models_dict.items():
        print(f"  Computing ISM for {name}...")
        ism = compute_ism(model, ism_seqs, device)
        all_ism[name] = ism

        # Compute summary statistics
        max_effect = np.abs(ism).max(axis=-1)  # [N, L]
        results[name] = {
            "mean_max_effect_per_pos": max_effect.mean(axis=0).tolist(),
            "mean_absolute_effect": float(np.abs(ism).mean()),
            "max_absolute_effect": float(np.abs(ism).max()),
            "effect_std_per_pos": max_effect.std(axis=0).tolist(),
        }

    # ISM correlation between models
    ism_correlations = {}
    model_names = list(all_ism.keys())
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            e1 = np.abs(all_ism[n1]).max(axis=-1).flatten()
            e2 = np.abs(all_ism[n2]).max(axis=-1).flatten()
            r, _ = spearmanr(e1, e2)
            ism_correlations[f"{n1}_vs_{n2}"] = float(r)

    results["pairwise_ism_correlation"] = ism_correlations

    np.savez_compressed(
        os.path.join(output_dir, "ism_scores.npz"),
        **{f"ism_{k}": v for k, v in all_ism.items()},
        sequences=ism_seqs,
    )

    print(f"  ISM correlations:")
    for pair, r in sorted(ism_correlations.items()):
        print(f"    {pair}: {r:.4f}")

    return results


# ============================================================================
# 3. Motif Discovery from Attributions
# ============================================================================
def extract_motifs_from_attributions(attrs, sequences, window=15, n_motifs=200):
    """Extract top-attributed windows as candidate motifs."""
    n_seqs, seq_len, n_channels = attrs.shape
    pos_importance = np.abs(attrs).sum(axis=-1)  # [N, L]

    # Find top windows by sliding window importance
    motifs = []
    for seq_idx in range(n_seqs):
        for start in range(0, seq_len - window):
            score = pos_importance[seq_idx, start:start+window].sum()
            motifs.append((score, seq_idx, start))

    # Sort and take top motifs
    motifs.sort(key=lambda x: -x[0])
    top_motifs = motifs[:n_motifs]

    # Extract PWM-like representations
    motif_matrices = []
    for score, seq_idx, start in top_motifs:
        window_seq = sequences[seq_idx, start:start+window, :]
        window_attr = attrs[seq_idx, start:start+window, :]
        # Weight by attribution
        weighted = window_seq * np.abs(window_attr)
        motif_matrices.append(weighted)

    return np.array(motif_matrices), top_motifs


def run_motif_discovery(attrs_dict, sequences, output_dir, n_clusters=10):
    """Cluster attributed windows to find recurring motifs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Motif Discovery from Attributions")
    print("=" * 70)

    results = {}
    for name, attrs in attrs_dict.items():
        print(f"  Extracting motifs for {name}...")
        motif_matrices, top_motifs = extract_motifs_from_attributions(
            attrs, sequences, window=15, n_motifs=500
        )

        # Flatten and cluster
        flat = motif_matrices.reshape(len(motif_matrices), -1)
        n_clust = min(n_clusters, len(flat))
        kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        labels = kmeans.fit_predict(flat)

        # Build PWMs per cluster
        cluster_pwms = {}
        for c in range(n_clust):
            mask = labels == c
            if mask.sum() < 5:
                continue
            cluster_seqs = motif_matrices[mask]
            # Average attribution-weighted sequence
            pwm = cluster_seqs.mean(axis=0)
            # Normalize to get information content
            pwm_norm = np.abs(pwm)
            pwm_sum = pwm_norm.sum(axis=-1, keepdims=True)
            pwm_sum[pwm_sum == 0] = 1
            pwm_prob = pwm_norm / pwm_sum
            cluster_pwms[f"cluster_{c}"] = {
                "pwm": pwm_prob.tolist(),
                "count": int(mask.sum()),
                "mean_attribution": float(np.mean([top_motifs[i][0] for i in range(len(mask)) if mask[i]])),
                "consensus": "".join("ACGT"[np.argmax(row)] for row in pwm_prob),
            }

        results[name] = {
            "n_clusters": n_clust,
            "clusters": cluster_pwms,
        }
        print(f"    Found {len(cluster_pwms)} clusters with 5+ members")
        for cname, cdata in sorted(cluster_pwms.items(), key=lambda x: -x[1]["mean_attribution"]):
            print(f"      {cname}: n={cdata['count']}, consensus={cdata['consensus']}")

    return results


# ============================================================================
# 4. Representation Geometry
# ============================================================================
def extract_representations(model, sequences, device, experiment_id=None,
                            batch_size=512):
    """Extract representations from model."""
    model.eval()
    all_reps = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[i:i+batch_size]).to(device)
            reps = model.encode(batch, experiment_id=experiment_id)
            all_reps.append(reps.cpu().numpy())
    return np.concatenate(all_reps)


def run_representation_geometry(models_dict, k562_seqs, k562_acts,
                                 hepg2_seqs, hepg2_acts, device, output_dir):
    """Analyze representation geometry: PCA, separation metrics."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Representation Geometry")
    print("=" * 70)

    results = {}
    n_per_type = min(2500, len(k562_seqs), len(hepg2_seqs))
    rng = np.random.RandomState(42)
    k_idx = rng.choice(len(k562_seqs), n_per_type, replace=False)
    h_idx = rng.choice(len(hepg2_seqs), n_per_type, replace=False)
    combined_seqs = np.concatenate([k562_seqs[k_idx], hepg2_seqs[h_idx]])
    combined_acts = np.concatenate([k562_acts[k_idx], hepg2_acts[h_idx]])
    cell_labels = np.array([0] * n_per_type + [1] * n_per_type)

    for name, (model, config) in models_dict.items():
        print(f"  Analyzing representations for {name}...")

        # Denoised representations
        reps = extract_representations(model, combined_seqs, device,
                                        experiment_id=None)

        # PCA
        pca = PCA(n_components=min(50, reps.shape[1]))
        reps_pca = pca.fit_transform(reps)

        # Explained variance
        explained_var = pca.explained_variance_ratio_

        # Cell-type separability: train linear probe
        split = int(0.8 * len(reps))
        perm = rng.permutation(len(reps))
        train_idx, test_idx = perm[:split], perm[split:]
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(reps[train_idx], cell_labels[train_idx])
        cell_acc = clf.score(reps[test_idx], cell_labels[test_idx])

        # Activity prediction: linear probe
        reg = Ridge(alpha=1.0)
        reg.fit(reps[train_idx], combined_acts[train_idx])
        pred_acts = reg.predict(reps[test_idx])
        act_r2 = reg.score(reps[test_idx], combined_acts[test_idx])
        act_sp = float(spearmanr(pred_acts, combined_acts[test_idx])[0])

        # Representation statistics
        rep_norms = np.linalg.norm(reps, axis=1)
        k562_reps = reps[:n_per_type]
        hepg2_reps = reps[n_per_type:]

        # Inter/intra cluster distance ratio (cell type)
        k_center = k562_reps.mean(axis=0)
        h_center = hepg2_reps.mean(axis=0)
        inter_dist = float(np.linalg.norm(k_center - h_center))
        intra_k = float(np.mean(np.linalg.norm(k562_reps - k_center, axis=1)))
        intra_h = float(np.mean(np.linalg.norm(hepg2_reps - h_center, axis=1)))
        separation_ratio = inter_dist / (0.5 * (intra_k + intra_h) + 1e-10)

        # Effective dimensionality (participation ratio)
        eigenvalues = pca.explained_variance_
        participation_ratio = float(
            (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        )

        # Isotropy score: how uniformly directions are used
        cos_sims = []
        sample_reps = reps[rng.choice(len(reps), min(500, len(reps)), replace=False)]
        centered = sample_reps - sample_reps.mean(axis=0)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = centered / norms
        cos_matrix = normalized @ normalized.T
        upper_tri = cos_matrix[np.triu_indices(len(normalized), k=1)]
        isotropy = float(np.std(upper_tri))  # lower = more isotropic

        results[name] = {
            "pca_explained_variance_top10": explained_var[:10].tolist(),
            "pca_cumulative_var_10": float(explained_var[:10].sum()),
            "pca_cumulative_var_20": float(explained_var[:20].sum()) if len(explained_var) >= 20 else float(explained_var.sum()),
            "cell_type_probe_accuracy": float(cell_acc),
            "activity_probe_r2": float(act_r2),
            "activity_probe_spearman": act_sp,
            "mean_rep_norm": float(rep_norms.mean()),
            "std_rep_norm": float(rep_norms.std()),
            "cell_type_separation_ratio": separation_ratio,
            "effective_dimensionality": participation_ratio,
            "isotropy_score": isotropy,
        }

        print(f"    Cell probe acc={cell_acc:.4f}, Act R²={act_r2:.4f}, "
              f"Separation={separation_ratio:.4f}, EffDim={participation_ratio:.1f}")

    return results


# ============================================================================
# 5. Experiment-Norm Comparison
# ============================================================================
def run_experiment_norm_comparison(models_dict, paired_seqs, k562_acts,
                                    hepg2_acts, device, output_dir):
    """Compare K562-normed vs HepG2-normed vs denoised representations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Experiment-Norm Comparison")
    print("=" * 70)

    results = {}
    n_samples = min(2000, len(paired_seqs))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(paired_seqs), n_samples, replace=False)
    seqs = paired_seqs[idx]
    k_acts = k562_acts[idx]
    h_acts = hepg2_acts[idx]

    for name, (model, config) in models_dict.items():
        if model.n_experiments < 2:
            continue
        print(f"  Comparing norms for {name}...")

        reps_k562 = extract_representations(model, seqs, device, experiment_id=0)
        reps_hepg2 = extract_representations(model, seqs, device, experiment_id=1)
        reps_denoised = extract_representations(model, seqs, device, experiment_id=None)

        # How different are the representations?
        diff_k_h = np.linalg.norm(reps_k562 - reps_hepg2, axis=1)
        diff_k_d = np.linalg.norm(reps_k562 - reps_denoised, axis=1)
        diff_h_d = np.linalg.norm(reps_hepg2 - reps_denoised, axis=1)

        # CKA between norm variants
        cka_k_h = compute_linear_cka(reps_k562, reps_hepg2)
        cka_k_d = compute_linear_cka(reps_k562, reps_denoised)
        cka_h_d = compute_linear_cka(reps_hepg2, reps_denoised)

        # Which norm predicts which cell type better?
        split = int(0.8 * n_samples)
        perm = rng.permutation(n_samples)
        train_idx, test_idx = perm[:split], perm[split:]

        predictions = {}
        for norm_name, reps, target_name, targets in [
            ("k562_norm", reps_k562, "k562_activity", k_acts),
            ("k562_norm", reps_k562, "hepg2_activity", h_acts),
            ("hepg2_norm", reps_hepg2, "k562_activity", k_acts),
            ("hepg2_norm", reps_hepg2, "hepg2_activity", h_acts),
            ("denoised", reps_denoised, "k562_activity", k_acts),
            ("denoised", reps_denoised, "hepg2_activity", h_acts),
        ]:
            reg = Ridge(alpha=1.0)
            reg.fit(reps[train_idx], targets[train_idx])
            pred = reg.predict(reps[test_idx])
            sp = float(spearmanr(pred, targets[test_idx])[0])
            predictions[f"{norm_name}_predicts_{target_name}"] = sp

        results[name] = {
            "mean_diff_k562_hepg2": float(diff_k_h.mean()),
            "mean_diff_k562_denoised": float(diff_k_d.mean()),
            "mean_diff_hepg2_denoised": float(diff_h_d.mean()),
            "cka_k562_hepg2": float(cka_k_h),
            "cka_k562_denoised": float(cka_k_d),
            "cka_hepg2_denoised": float(cka_h_d),
            **predictions,
        }

        print(f"    CKA(K562,HepG2)={cka_k_h:.4f}, CKA(K562,Den)={cka_k_d:.4f}")
        for k, v in predictions.items():
            print(f"      {k}: {v:.4f}")

    return results


# ============================================================================
# 6. BatchNorm Parameter Analysis
# ============================================================================
def run_batchnorm_analysis(models_dict, output_dir):
    """Analyze learned BatchNorm parameters across experiment norms."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: BatchNorm Parameter Analysis")
    print("=" * 70)

    results = {}
    for name, (model, config) in models_dict.items():
        if model.n_experiments < 2:
            continue
        print(f"  Analyzing BatchNorm params for {name}...")

        norm_params = {}
        for exp_id, norm in enumerate(model.exp_norms):
            gamma = norm.weight.detach().cpu().numpy()
            beta = norm.bias.detach().cpu().numpy()
            running_mean = norm.running_mean.detach().cpu().numpy()
            running_var = norm.running_var.detach().cpu().numpy()
            norm_params[f"exp_{exp_id}"] = {
                "gamma_mean": float(gamma.mean()),
                "gamma_std": float(gamma.std()),
                "gamma_min": float(gamma.min()),
                "gamma_max": float(gamma.max()),
                "beta_mean": float(beta.mean()),
                "beta_std": float(beta.std()),
                "running_mean_norm": float(np.linalg.norm(running_mean)),
                "running_var_mean": float(running_var.mean()),
            }

        # Difference between norms
        g0 = model.exp_norms[0].weight.detach().cpu().numpy()
        g1 = model.exp_norms[1].weight.detach().cpu().numpy()
        b0 = model.exp_norms[0].bias.detach().cpu().numpy()
        b1 = model.exp_norms[1].bias.detach().cpu().numpy()
        m0 = model.exp_norms[0].running_mean.detach().cpu().numpy()
        m1 = model.exp_norms[1].running_mean.detach().cpu().numpy()

        gamma_diff = np.abs(g0 - g1)
        beta_diff = np.abs(b0 - b1)
        mean_diff = np.abs(m0 - m1)

        # Which dimensions differ most?
        top_gamma_dims = np.argsort(-gamma_diff)[:10].tolist()
        top_beta_dims = np.argsort(-beta_diff)[:10].tolist()
        top_mean_dims = np.argsort(-mean_diff)[:10].tolist()

        # Cosine similarity between norm params
        cos_gamma = float(np.dot(g0, g1) / (np.linalg.norm(g0) * np.linalg.norm(g1) + 1e-10))
        cos_beta = float(np.dot(b0, b1) / (np.linalg.norm(b0) * np.linalg.norm(b1) + 1e-10))

        results[name] = {
            "norm_params": norm_params,
            "gamma_diff_mean": float(gamma_diff.mean()),
            "gamma_diff_max": float(gamma_diff.max()),
            "beta_diff_mean": float(beta_diff.mean()),
            "beta_diff_max": float(beta_diff.max()),
            "running_mean_diff_mean": float(mean_diff.mean()),
            "cos_similarity_gamma": cos_gamma,
            "cos_similarity_beta": cos_beta,
            "top_differing_gamma_dims": top_gamma_dims,
            "top_differing_beta_dims": top_beta_dims,
            "top_differing_mean_dims": top_mean_dims,
        }

        print(f"    Gamma cosine sim: {cos_gamma:.4f}, "
              f"Beta cosine sim: {cos_beta:.4f}")
        print(f"    Mean gamma diff: {gamma_diff.mean():.4f}, "
              f"Mean beta diff: {beta_diff.mean():.4f}")

    return results


# ============================================================================
# 7. First-Layer Filter Visualization
# ============================================================================
def run_filter_analysis(models_dict, output_dir):
    """Extract and analyze first-layer convolutional filters as PWMs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: First-Layer Filter Analysis")
    print("=" * 70)

    results = {}
    for name, (model, config) in models_dict.items():
        arch = config.get("architecture", "")
        # Get first conv layer weights
        base = model.base_model
        if arch in ("dilated_cnn", "cnn"):
            conv = base.initial_conv[0]  # first Conv1d
        elif arch == "bilstm":
            conv = base.initial_conv[0]
        else:
            continue

        weights = conv.weight.detach().cpu().numpy()  # [out_ch, in_ch=4, kernel]
        n_filters, n_channels, kernel_size = weights.shape

        print(f"  {name}: {n_filters} filters, kernel={kernel_size}")

        filter_info = {}
        # Information content per filter
        for filt_idx in range(n_filters):
            w = weights[filt_idx]  # [4, kernel]
            # Normalize to get probability-like matrix
            w_abs = np.abs(w)
            w_sum = w_abs.sum(axis=0, keepdims=True)
            w_sum[w_sum == 0] = 1
            w_norm = w_abs / w_sum

            # Information content (relative to uniform)
            ic = np.log2(4) + np.sum(w_norm * np.log2(w_norm + 1e-10), axis=0)
            total_ic = float(ic.sum())

            # Consensus sequence
            consensus = "".join("ACGT"[np.argmax(w_abs[:, pos])] for pos in range(kernel_size))

            # Filter activation magnitude (L2 norm)
            magnitude = float(np.linalg.norm(w))

            filter_info[f"filter_{filt_idx}"] = {
                "consensus": consensus,
                "total_ic": total_ic,
                "magnitude": magnitude,
                "pwm": w_norm.tolist(),
            }

        # Cluster filters by similarity
        flat_weights = weights.reshape(n_filters, -1)
        n_clust = min(10, n_filters // 2)
        kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        labels = kmeans.fit_predict(flat_weights)

        cluster_sizes = {}
        for c in range(n_clust):
            members = [i for i in range(n_filters) if labels[i] == c]
            cluster_sizes[f"cluster_{c}"] = len(members)

        results[name] = {
            "n_filters": n_filters,
            "kernel_size": kernel_size,
            "filter_clusters": cluster_sizes,
            "top_ic_filters": sorted(
                [(k, v["total_ic"], v["consensus"])
                 for k, v in filter_info.items()],
                key=lambda x: -x[1]
            )[:20],
        }

        top5 = sorted(filter_info.items(), key=lambda x: -x[1]["total_ic"])[:5]
        for fname, fdata in top5:
            print(f"    {fname}: IC={fdata['total_ic']:.2f}, {fdata['consensus']}")

    return results


# ============================================================================
# 8. Prediction Head Weight Analysis
# ============================================================================
def run_prediction_head_analysis(models_dict, output_dir):
    """Analyze which latent dimensions drive predictions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Prediction Head Weight Analysis")
    print("=" * 70)

    results = {}
    for name, (model, config) in models_dict.items():
        weights = model.prediction_head.weight.detach().cpu().numpy().flatten()
        bias = model.prediction_head.bias.detach().cpu().numpy().flatten()

        abs_weights = np.abs(weights)
        top_dims = np.argsort(-abs_weights)[:20].tolist()
        top_weights = abs_weights[top_dims].tolist()

        # Sparsity: what fraction of dimensions contribute significantly?
        threshold = abs_weights.max() * 0.1
        n_significant = int((abs_weights > threshold).sum())
        gini = compute_gini(abs_weights)

        results[name] = {
            "n_dims": len(weights),
            "weight_mean": float(weights.mean()),
            "weight_std": float(weights.std()),
            "weight_max": float(abs_weights.max()),
            "top_20_dims": top_dims,
            "top_20_weights": top_weights,
            "n_significant_dims": n_significant,
            "gini_coefficient": float(gini),
            "bias": float(bias[0]) if len(bias) > 0 else 0.0,
        }

        print(f"  {name}: {n_significant}/{len(weights)} significant dims, "
              f"Gini={gini:.4f}")

    # Overlap in top dims between models
    model_names = list(results.keys())
    dim_overlaps = {}
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            s1 = set(results[n1]["top_20_dims"])
            s2 = set(results[n2]["top_20_dims"])
            overlap = len(s1 & s2)
            dim_overlaps[f"{n1}_vs_{n2}"] = overlap
    results["top20_dim_overlaps"] = dim_overlaps

    return results


def compute_gini(values):
    """Compute Gini coefficient of an array."""
    values = np.abs(values)
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * values).sum() / (n * values.sum() + 1e-10))


# ============================================================================
# 9. CKA Between Models
# ============================================================================
def compute_linear_cka(X, Y):
    """Compute linear CKA between two representation matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def run_cka_analysis(models_dict, seqs, device, output_dir):
    """Compute pairwise CKA between all models' representations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: CKA Between Models")
    print("=" * 70)

    n_cka = min(2000, len(seqs))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(seqs), n_cka, replace=False)
    cka_seqs = seqs[idx]

    all_reps = {}
    for name, (model, config) in models_dict.items():
        reps = extract_representations(model, cka_seqs, device, experiment_id=None)
        all_reps[name] = reps

    model_names = list(all_reps.keys())
    cka_matrix = np.zeros((len(model_names), len(model_names)))

    for i, n1 in enumerate(model_names):
        for j, n2 in enumerate(model_names):
            if i <= j:
                cka = compute_linear_cka(all_reps[n1], all_reps[n2])
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka

    results = {
        "model_names": model_names,
        "cka_matrix": cka_matrix.tolist(),
    }

    print("  CKA matrix:")
    header = "              " + "  ".join(f"{n[:12]:>12}" for n in model_names)
    print(header)
    for i, n1 in enumerate(model_names):
        row = f"{n1[:14]:<14} " + "  ".join(f"{cka_matrix[i, j]:12.4f}" for j in range(len(model_names)))
        print(row)

    return results


# ============================================================================
# 10. Positional Sensitivity Profile
# ============================================================================
def compute_gradient_magnitude(model, sequences, device, batch_size=128):
    """Compute gradient magnitude at each position."""
    torch.backends.cudnn.enabled = False
    model.train()

    all_grads = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.from_numpy(sequences[start:start+batch_size]).to(device)
        batch.requires_grad_(True)

        if hasattr(model, "predict_denoised"):
            out = model.predict_denoised(batch)
        else:
            out = model(batch)
        out.sum().backward()

        grads = batch.grad.detach().cpu().numpy()
        all_grads.append(grads)

    model.eval()
    torch.backends.cudnn.enabled = True
    return np.concatenate(all_grads)


def run_positional_sensitivity(models_dict, seqs, acts, device, output_dir):
    """Analyze which sequence positions have highest gradient magnitude."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Positional Sensitivity Profile")
    print("=" * 70)

    n_pos = min(1000, len(seqs))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(seqs), n_pos, replace=False)
    pos_seqs = seqs[idx]
    pos_acts = acts[idx]

    results = {}
    for name, (model, config) in models_dict.items():
        print(f"  Computing gradients for {name}...")
        grads = compute_gradient_magnitude(model, pos_seqs, device)

        # Per-position gradient magnitude
        grad_mag = np.sqrt((grads ** 2).sum(axis=-1))  # [N, L]
        mean_profile = grad_mag.mean(axis=0)  # [L]

        # High vs low activity comparison
        high_idx = np.argsort(pos_acts)[-len(pos_acts)//5:]
        low_idx = np.argsort(pos_acts)[:len(pos_acts)//5]
        high_profile = grad_mag[high_idx].mean(axis=0)
        low_profile = grad_mag[low_idx].mean(axis=0)

        # Where are models most sensitive?
        top_positions = np.argsort(-mean_profile)[:20].tolist()

        # Entropy of positional sensitivity (is it concentrated or spread?)
        p = mean_profile / (mean_profile.sum() + 1e-10)
        entropy = float(-np.sum(p * np.log2(p + 1e-10)))

        results[name] = {
            "mean_sensitivity_profile": mean_profile.tolist(),
            "high_activity_profile": high_profile.tolist(),
            "low_activity_profile": low_profile.tolist(),
            "top_20_positions": top_positions,
            "sensitivity_entropy": entropy,
            "profile_correlation_high_low": float(spearmanr(high_profile, low_profile)[0]),
        }

        print(f"    Entropy={entropy:.2f}, "
              f"High-Low corr={spearmanr(high_profile, low_profile)[0]:.4f}, "
              f"Top positions: {top_positions[:5]}")

    return results


# ============================================================================
# 11. High vs Low Activity Attribution Comparison
# ============================================================================
def run_high_low_comparison(models_dict, seqs, acts, device, output_dir):
    """Compare what models attend to for high vs low activity sequences."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: High vs Low Activity Attribution")
    print("=" * 70)

    n_cmp = min(500, len(seqs))
    rng = np.random.RandomState(42)

    # Top and bottom 20%
    sorted_idx = np.argsort(acts)
    n_extreme = len(acts) // 5
    high_idx = sorted_idx[-n_extreme:]
    low_idx = sorted_idx[:n_extreme]

    if len(high_idx) > n_cmp:
        high_idx = rng.choice(high_idx, n_cmp, replace=False)
    if len(low_idx) > n_cmp:
        low_idx = rng.choice(low_idx, n_cmp, replace=False)

    n_ig = min(200, len(high_idx))
    high_seqs = seqs[high_idx[:n_ig]]
    low_seqs = seqs[low_idx[:n_ig]]

    results = {}
    for name, (model, config) in models_dict.items():
        print(f"  Computing high/low IG for {name}...")
        high_attrs = compute_integrated_gradients(model, high_seqs, device)
        low_attrs = compute_integrated_gradients(model, low_seqs, device)

        # Position importance comparison
        high_imp = np.abs(high_attrs).sum(axis=-1).mean(axis=0)  # [L]
        low_imp = np.abs(low_attrs).sum(axis=-1).mean(axis=0)   # [L]

        # Where does the model look differently for high vs low?
        diff_imp = high_imp - low_imp
        top_diff_pos = np.argsort(-np.abs(diff_imp))[:20].tolist()

        # Nucleotide preference differences
        high_nuc_pref = high_attrs.mean(axis=0)  # [L, 4]
        low_nuc_pref = low_attrs.mean(axis=0)

        results[name] = {
            "high_mean_magnitude": float(np.abs(high_attrs).mean()),
            "low_mean_magnitude": float(np.abs(low_attrs).mean()),
            "magnitude_ratio": float(np.abs(high_attrs).mean() / (np.abs(low_attrs).mean() + 1e-10)),
            "profile_correlation": float(spearmanr(high_imp, low_imp)[0]),
            "top_differential_positions": top_diff_pos,
            "high_importance_profile": high_imp.tolist(),
            "low_importance_profile": low_imp.tolist(),
        }

        print(f"    High/Low magnitude ratio: {results[name]['magnitude_ratio']:.4f}, "
              f"Profile corr: {results[name]['profile_correlation']:.4f}")

    return results


# ============================================================================
# 12. Cross-Experiment Attribution Consistency
# ============================================================================
def run_cross_experiment_attributions(models_dict, paired_seqs, device, output_dir):
    """Compare attributions using K562 norm vs HepG2 norm on same sequences."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 12: Cross-Experiment Attribution Consistency")
    print("=" * 70)

    n_cross = min(200, len(paired_seqs))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(paired_seqs), n_cross, replace=False)
    cross_seqs = paired_seqs[idx]

    results = {}
    for name, (model, config) in models_dict.items():
        if model.n_experiments < 2:
            continue
        print(f"  Computing cross-experiment IG for {name}...")

        # IG using K562 normalization
        attrs_k562 = compute_integrated_gradients(
            model, cross_seqs, device, experiment_id=0
        )
        # IG using HepG2 normalization
        attrs_hepg2 = compute_integrated_gradients(
            model, cross_seqs, device, experiment_id=1
        )
        # IG using denoised
        attrs_denoised = compute_integrated_gradients(
            model, cross_seqs, device, experiment_id=None
        )

        # Position-level correlation between norms
        imp_k = np.abs(attrs_k562).sum(axis=-1).flatten()
        imp_h = np.abs(attrs_hepg2).sum(axis=-1).flatten()
        imp_d = np.abs(attrs_denoised).sum(axis=-1).flatten()

        corr_k_h = float(spearmanr(imp_k, imp_h)[0])
        corr_k_d = float(spearmanr(imp_k, imp_d)[0])
        corr_h_d = float(spearmanr(imp_h, imp_d)[0])

        # Per-sequence correlation (how consistent are attributions per sequence?)
        per_seq_corr = []
        for i in range(n_cross):
            sk = np.abs(attrs_k562[i]).sum(axis=-1)
            sh = np.abs(attrs_hepg2[i]).sum(axis=-1)
            r, _ = spearmanr(sk, sh)
            if not np.isnan(r):
                per_seq_corr.append(r)

        results[name] = {
            "attribution_corr_k562_hepg2": corr_k_h,
            "attribution_corr_k562_denoised": corr_k_d,
            "attribution_corr_hepg2_denoised": corr_h_d,
            "per_sequence_corr_mean": float(np.mean(per_seq_corr)),
            "per_sequence_corr_std": float(np.std(per_seq_corr)),
        }

        print(f"    K562 vs HepG2 attr corr: {corr_k_h:.4f}, "
              f"Per-seq mean: {np.mean(per_seq_corr):.4f}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--k562_data", default="data/processed/dream_K562.h5")
    parser.add_argument("--hepg2_data", default="data/processed/dream_HepG2.h5")
    parser.add_argument("--paired_data", default="data/processed/paired_K562_HepG2.h5")
    parser.add_argument("--output_dir", default="results/interpretability/")
    parser.add_argument("--architecture", default="dilated_cnn",
                        choices=["bilstm", "dilated_cnn", "both"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    k562_seqs, k562_acts = load_test_data(args.k562_data, N_REPR_SAMPLES)
    hepg2_seqs, hepg2_acts = load_test_data(args.hepg2_data, N_REPR_SAMPLES)
    paired_seqs, paired_k562, paired_hepg2, paired_consensus = \
        load_paired_test_data(args.paired_data)
    print(f"  K562: {len(k562_seqs)}, HepG2: {len(hepg2_seqs)}, "
          f"Paired: {len(paired_seqs)}")

    # Load models
    print("Loading models...")
    archs = ["bilstm", "dilated_cnn"] if args.architecture == "both" else [args.architecture]
    models_dict = {}
    for arch in archs:
        for model_name in KEY_MODELS.get(arch, []):
            model_dir = os.path.join(args.results_dir, model_name)
            if not os.path.exists(os.path.join(model_dir, "best_model.pt")):
                print(f"  Skipping {model_name} (not found)")
                continue
            model, config = load_model(model_dir, device)
            if model is not None:
                models_dict[model_name] = (model, config)
                print(f"  Loaded {model_name}")

    print(f"\n{len(models_dict)} models loaded")

    # ---- Run all experiments ----
    all_results = {}

    # 1. Integrated Gradients
    all_results["integrated_gradients"] = run_integrated_gradients(
        models_dict, k562_seqs, k562_acts, device, args.output_dir
    )

    # 2. In-Silico Mutagenesis
    all_results["ism"] = run_ism_analysis(
        models_dict, k562_seqs, k562_acts, device, args.output_dir
    )

    # 3. Motif Discovery (uses IG results)
    ig_data = np.load(os.path.join(args.output_dir, "integrated_gradients.npz"))
    attrs_dict = {}
    for name in models_dict:
        key = f"attrs_{name}"
        if key in ig_data:
            attrs_dict[name] = ig_data[key]
    all_results["motif_discovery"] = run_motif_discovery(
        attrs_dict, ig_data["sequences"], args.output_dir
    )

    # 4. Representation Geometry
    all_results["representation_geometry"] = run_representation_geometry(
        models_dict, k562_seqs, k562_acts, hepg2_seqs, hepg2_acts,
        device, args.output_dir
    )

    # 5. Experiment-Norm Comparison
    all_results["experiment_norm_comparison"] = run_experiment_norm_comparison(
        models_dict, paired_seqs, paired_k562, paired_hepg2,
        device, args.output_dir
    )

    # 6. BatchNorm Parameter Analysis
    all_results["batchnorm_analysis"] = run_batchnorm_analysis(
        models_dict, args.output_dir
    )

    # 7. First-Layer Filter Analysis
    all_results["filter_analysis"] = run_filter_analysis(
        models_dict, args.output_dir
    )

    # 8. Prediction Head Weight Analysis
    all_results["prediction_head"] = run_prediction_head_analysis(
        models_dict, args.output_dir
    )

    # 9. CKA Between Models
    all_results["cka"] = run_cka_analysis(
        models_dict, k562_seqs, device, args.output_dir
    )

    # 10. Positional Sensitivity
    all_results["positional_sensitivity"] = run_positional_sensitivity(
        models_dict, k562_seqs, k562_acts, device, args.output_dir
    )

    # 11. High vs Low Activity
    all_results["high_low_comparison"] = run_high_low_comparison(
        models_dict, k562_seqs, k562_acts, device, args.output_dir
    )

    # 12. Cross-Experiment Attribution Consistency
    all_results["cross_experiment_attributions"] = run_cross_experiment_attributions(
        models_dict, paired_seqs, device, args.output_dir
    )

    # Save all results
    output_file = os.path.join(args.output_dir, "interpretability_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("INTERPRETABILITY SUITE COMPLETE")
    print("=" * 70)
    print(f"  12 experiments completed")
    print(f"  {len(models_dict)} models analyzed")
    print(f"  Results: {output_file}")


if __name__ == "__main__":
    main()
