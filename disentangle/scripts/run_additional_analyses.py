#!/usr/bin/env python3
"""
Additional analyses for DISENTANGLE paper.

Implements 7 analyses (E1, E2, F1, F2, F3, F4, F5) on existing 42 models.
Follows the pattern of scripts/run_interpretability_suite.py.

Analyses:
  E1: Verify/Fix Tier 2 Non-Circularity (multi-point extraction)
  E2: Stratified CAGI5 Analysis
  F1: Learning Dynamics
  F2: Representation Sensitivity Analysis
  F3: Noise Fraction Estimation
  F4: Experiment Probe Multi-Point Extraction
  F5: GC Content Decomposition

Usage:
    python scripts/run_additional_analyses.py --gpu 0
"""

import argparse
import csv
import json
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr, rankdata

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluate import (
    load_model, evaluate_cagi5_element,
    one_hot_encode_for_disentangle, get_variant_sequence, get_ref_sequence,
)

# ============================================================================
# Constants
# ============================================================================

# 10 key models for expensive analyses (F2, F4, E2)
KEY_MODELS_10 = [
    "bilstm_baseline_mse_seed42",
    "bilstm_ranking_seed42",
    "bilstm_contrastive_only_seed42",
    "bilstm_full_disentangle_seed42",
    "bilstm_ranking_contrastive_seed42",
    "dilated_cnn_baseline_mse_seed42",
    "dilated_cnn_ranking_seed42",
    "dilated_cnn_contrastive_only_seed42",
    "dilated_cnn_full_disentangle_seed42",
    "dilated_cnn_ranking_contrastive_seed42",
]

# New experiment models for expensive analyses
KEY_MODELS_NEW = [
    "bilstm_two_stage_seed42",
    "dilated_cnn_two_stage_seed42",
    "bilstm_variant_contrastive_seed42",
    "dilated_cnn_variant_contrastive_seed42",
    "bilstm_hierarchical_contrastive_seed42",
    "dilated_cnn_hierarchical_contrastive_seed42",
    "bilstm_quantile_mse_seed42",
    "bilstm_e3_gc_dependent_baseline_seed42",
    "bilstm_e3_gc_dependent_disentangle_seed42",
]

K562_ELEMENTS = ['GP1BB', 'HBB', 'HBG1', 'PKLR']
HEPG2_ELEMENTS = ['F9', 'LDLR', 'SORT1']


def find_all_models(results_dir):
    """Find all trained model directories."""
    models = []
    for name in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, name)
        if (os.path.isdir(d) and
            os.path.exists(os.path.join(d, "best_model.pt")) and
            os.path.exists(os.path.join(d, "config.json"))):
            models.append(name)
    return models


def load_test_data(data_file, n_samples=None, seed=42):
    """Load test split data."""
    with h5py.File(data_file, "r") as f:
        splits = f["split"][:]
        mask = splits == 2
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
        k562_stds = f["k562_stds"][:][mask] if "k562_stds" in f else None
        hepg2_stds = f["hepg2_stds"][:][mask] if "hepg2_stds" in f else None
    if n_samples and n_samples < len(seqs):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(seqs), n_samples, replace=False)
        return (seqs[idx], k562_acts[idx], hepg2_acts[idx], consensus[idx],
                k562_stds[idx] if k562_stds is not None else None,
                hepg2_stds[idx] if hepg2_stds is not None else None)
    return seqs, k562_acts, hepg2_acts, consensus, k562_stds, hepg2_stds


# ============================================================================
# E1: Verify/Fix Tier 2 Non-Circularity
# ============================================================================
def run_e1_noncircular(models_dict, paired_file, device, output_dir):
    """
    E1: Multi-point extraction for non-circular cross-experiment evaluation.

    For each model, compute predictions from:
      1. Pre-BN raw: encoder output -> prediction head (no BN)
      2. Denoised: averaged BN -> prediction head
      3. Experiment-specific: K562-BN and HepG2-BN -> prediction head

    Compare each against actual K562 and HepG2 activities.
    """
    print("\n" + "=" * 70)
    print("E1: Non-Circular Multi-Point Cross-Experiment Evaluation")
    print("=" * 70)

    with h5py.File(paired_file, "r") as f:
        splits = f["split"][:]
        mask = splits == 2
        sequences = f["sequences"][:][mask].astype(np.float32)
        k562_acts = f["k562_activities"][:][mask]
        hepg2_acts = f["hepg2_activities"][:][mask]
        consensus = f["consensus_ranks"][:][mask]

    results = {}
    batch_size = 512

    for name, (model, config) in models_dict.items():
        print(f"  {name}...")
        model_results = {}

        def predict_batch(seqs, fn):
            preds = []
            with torch.no_grad():
                for i in range(0, len(seqs), batch_size):
                    batch = torch.from_numpy(seqs[i:i+batch_size]).to(device)
                    p = fn(batch)
                    preds.append(p.cpu().numpy())
            return np.concatenate(preds)

        # 1. Pre-BN raw
        preds_raw = predict_batch(sequences,
            lambda x: model.prediction_head(model.base_model.encode(x)).squeeze(-1))
        model_results["raw_spearman_k562"] = float(spearmanr(preds_raw, k562_acts)[0])
        model_results["raw_spearman_hepg2"] = float(spearmanr(preds_raw, hepg2_acts)[0])

        # 2. Denoised
        preds_den = predict_batch(sequences, lambda x: model.predict_denoised(x))
        model_results["denoised_spearman_k562"] = float(spearmanr(preds_den, k562_acts)[0])
        model_results["denoised_spearman_hepg2"] = float(spearmanr(preds_den, hepg2_acts)[0])
        model_results["denoised_spearman_consensus"] = float(spearmanr(preds_den, consensus)[0])

        # 3. Experiment-specific
        if model.n_experiments >= 2:
            preds_k = predict_batch(sequences,
                lambda x: model(x, experiment_id=0))
            preds_h = predict_batch(sequences,
                lambda x: model(x, experiment_id=1))
            model_results["k562norm_spearman_k562"] = float(spearmanr(preds_k, k562_acts)[0])
            model_results["k562norm_spearman_hepg2"] = float(spearmanr(preds_k, hepg2_acts)[0])
            model_results["hepg2norm_spearman_k562"] = float(spearmanr(preds_h, k562_acts)[0])
            model_results["hepg2norm_spearman_hepg2"] = float(spearmanr(preds_h, hepg2_acts)[0])

        results[name] = model_results
        print(f"    raw_k={model_results['raw_spearman_k562']:.4f} "
              f"den_k={model_results['denoised_spearman_k562']:.4f} "
              f"den_h={model_results['denoised_spearman_hepg2']:.4f}")

    return results


# ============================================================================
# E2: Stratified CAGI5 Analysis
# ============================================================================
def run_e2_stratified_cagi5(models_dict, cagi5_dir, references_file, device,
                             output_dir, window=230):
    """
    E2: Stratified CAGI5 analysis.

    For each CAGI5 element, stratify variants by:
      a) Relative position in element (5 bins)
      b) Transition vs transversion
      c) |effect size| quartiles
      d) GC content of +-10bp context

    Requires min 10 variants per stratum.
    """
    print("\n" + "=" * 70)
    print("E2: Stratified CAGI5 Analysis")
    print("=" * 70)

    if not os.path.exists(references_file) or not os.path.isdir(cagi5_dir):
        print("  CAGI5 data not found, skipping E2")
        return []

    with open(references_file) as f:
        references = json.load(f)

    # Load CAGI5 data
    cagi5_data = {}
    for tsv_file in sorted(os.listdir(cagi5_dir)):
        if not tsv_file.startswith("challenge_") or not tsv_file.endswith(".tsv"):
            continue
        element = tsv_file.replace("challenge_", "").replace(".tsv", "")
        filepath = os.path.join(cagi5_dir, tsv_file)
        with open(filepath) as f_tsv:
            lines = f_tsv.readlines()
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

    # Transition / transversion classification
    transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

    all_rows = []
    for model_name, (model, config) in models_dict.items():
        print(f"  {model_name}...")

        for element, df in cagi5_data.items():
            if element not in references:
                continue
            ref_data = references[element]
            ref_seq = ref_data['sequence']
            ref_start = ref_data['start']
            element_len = len(ref_seq)

            # Get variant predictions
            alt_seqs, ref_seqs, valid_idx = [], [], []
            for i, row in df.iterrows():
                alt_s = get_variant_sequence(ref_seq, ref_start, row['Pos'],
                                             row['Ref'], row['Alt'], window)
                ref_s = get_ref_sequence(ref_seq, ref_start, row['Pos'], window)
                if alt_s and ref_s and len(alt_s) == window and len(ref_s) == window:
                    alt_seqs.append(alt_s)
                    ref_seqs.append(ref_s)
                    valid_idx.append(i)

            if len(alt_seqs) < 20:
                continue

            # Get predictions
            from evaluate import predict_sequences_disentangle
            alt_preds = predict_sequences_disentangle(model, alt_seqs, device)
            ref_preds = predict_sequences_disentangle(model, ref_seqs, device)
            var_effects = alt_preds - ref_preds
            ground_truth = df.loc[valid_idx, 'Value'].values
            valid_df = df.loc[valid_idx].copy()
            valid_df['var_effect'] = var_effects
            valid_df['gt'] = ground_truth

            # Compute relative position
            valid_df['rel_pos'] = (valid_df['Pos'] - ref_start) / max(element_len, 1)
            valid_df['pos_bin'] = pd.cut(valid_df['rel_pos'], bins=5, labels=False)

            # Transition vs transversion
            valid_df['is_transition'] = valid_df.apply(
                lambda r: (r['Ref'], r['Alt']) in transitions
                if len(r['Ref']) == 1 and len(r['Alt']) == 1 else False, axis=1)

            # Effect size quartiles
            valid_df['abs_effect'] = valid_df['gt'].abs()
            valid_df['effect_quartile'] = pd.qcut(
                valid_df['abs_effect'], q=4, labels=False, duplicates='drop')

            # GC context
            gc_vals = []
            for _, row in valid_df.iterrows():
                idx = row['Pos'] - ref_start
                start = max(0, idx - 10)
                end = min(len(ref_seq), idx + 11)
                context = ref_seq[start:end].upper()
                gc = (context.count('G') + context.count('C')) / max(len(context), 1)
                gc_vals.append(gc)
            valid_df['gc_context'] = gc_vals
            valid_df['gc_quartile'] = pd.qcut(
                valid_df['gc_context'], q=4, labels=False, duplicates='drop')

            # Compute stratified correlations
            strat_configs = [
                ("position", "pos_bin", 5),
                ("mutation_type", "is_transition", 2),
                ("effect_size", "effect_quartile", 4),
                ("gc_context", "gc_quartile", 4),
            ]

            for strat_name, strat_col, n_strata in strat_configs:
                for stratum in sorted(valid_df[strat_col].unique()):
                    mask = valid_df[strat_col] == stratum
                    subset = valid_df[mask]
                    if len(subset) < 10:
                        continue
                    sp = float(spearmanr(subset['var_effect'], subset['gt'])[0])
                    if np.isnan(sp):
                        continue
                    all_rows.append({
                        "model": model_name,
                        "element": element,
                        "stratification": strat_name,
                        "stratum": str(stratum),
                        "n_variants": len(subset),
                        "spearman": sp,
                    })

    # Write CSV
    output_path = os.path.join(output_dir, "stratified_cagi5.csv")
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(output_path, index=False)
        print(f"  Wrote {len(all_rows)} rows to {output_path}")
    else:
        print("  No stratified results generated")

    return all_rows


# ============================================================================
# F1: Learning Dynamics
# ============================================================================
def run_f1_learning_dynamics(results_dir, eval_csv_path, output_dir):
    """
    F1: Learning dynamics analysis.

    Load history.json for all models, extract per-epoch train_loss components
    and val_spearman. Cross-reference final metrics with CAGI5.
    """
    print("\n" + "=" * 70)
    print("F1: Learning Dynamics")
    print("=" * 70)

    all_models = find_all_models(results_dir)
    results = {}

    # Load evaluation CSV for CAGI5 cross-reference
    eval_data = {}
    if os.path.exists(eval_csv_path):
        df_eval = pd.read_csv(eval_csv_path)
        for _, row in df_eval.iterrows():
            eval_data[row['model']] = row.to_dict()

    for model_name in all_models:
        history_path = os.path.join(results_dir, model_name, "history.json")
        config_path = os.path.join(results_dir, model_name, "config.json")

        if not os.path.exists(history_path):
            continue

        with open(history_path) as f:
            history = json.load(f)
        with open(config_path) as f:
            config = json.load(f)

        # Extract learning dynamics
        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_spearmans = [h.get("val_spearman", 0) for h in history]

        # Extract loss components
        components = defaultdict(list)
        for h in history:
            for k, v in h.get("train_components", {}).items():
                components[k].append(v)

        # Best epoch
        best_epoch = int(np.argmax(val_spearmans))
        best_val_spearman = float(max(val_spearmans))

        # Convergence speed: epoch to reach 90% of best val spearman
        threshold = 0.9 * best_val_spearman if best_val_spearman > 0 else 0
        convergence_epoch = len(epochs)
        for i, vs in enumerate(val_spearmans):
            if vs >= threshold:
                convergence_epoch = i
                break

        model_result = {
            "architecture": config.get("architecture", "unknown"),
            "condition": config.get("condition", "unknown"),
            "seed": config.get("seed", 0),
            "n_epochs": len(epochs),
            "best_epoch": best_epoch,
            "best_val_spearman": best_val_spearman,
            "convergence_epoch_90pct": convergence_epoch,
            "final_train_loss": float(train_losses[-1]),
            "train_loss_trajectory": train_losses,
            "val_spearman_trajectory": val_spearmans,
            "loss_components": {k: v for k, v in components.items()},
        }

        # Cross-reference with CAGI5
        if model_name in eval_data:
            ev = eval_data[model_name]
            cagi5_key = "cagi5_all_mean_spearman"
            if cagi5_key in ev and ev[cagi5_key] != "":
                model_result["cagi5_all_mean_spearman"] = float(ev[cagi5_key])
            matched_key = "cagi5_all_matched_mean_spearman"
            if matched_key in ev and ev[matched_key] != "":
                model_result["cagi5_matched_mean_spearman"] = float(ev[matched_key])

        results[model_name] = model_result
        print(f"  {model_name}: {len(epochs)} epochs, best_sp={best_val_spearman:.4f} "
              f"@ epoch {best_epoch}, conv_90%=epoch {convergence_epoch}")

    # Save
    output_path = os.path.join(output_dir, "learning_dynamics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    return results


# ============================================================================
# F2: Representation Sensitivity Analysis
# ============================================================================
def run_f2_sensitivity(models_dict, k562_data_file, device, output_dir,
                       n_sequences=1000, batch_size=256):
    """
    F2: Representation sensitivity analysis.

    Compute: sensitivity = mean|f(seq) - f(seq_mut)| / mean|f(seq)|
    for 1000 test sequences with all 690 single-nt mutations each.
    """
    print("\n" + "=" * 70)
    print("F2: Representation Sensitivity Analysis")
    print("=" * 70)

    seqs, acts = load_test_data(k562_data_file, n_sequences)
    n_seqs, seq_len, n_channels = seqs.shape
    # Total mutations per sequence: L * (C-1) = 230 * 3 = 690
    n_muts_per_seq = seq_len * (n_channels - 1)

    results = {}

    for name, (model, config) in models_dict.items():
        print(f"  {name} ({n_seqs} seqs x {n_muts_per_seq} mutations)...")
        model.eval()

        # Get reference predictions
        ref_preds = []
        with torch.no_grad():
            for i in range(0, n_seqs, batch_size):
                batch = torch.from_numpy(seqs[i:i+batch_size]).to(device)
                p = model.predict_denoised(batch)
                ref_preds.append(p.cpu().numpy())
        ref_preds = np.concatenate(ref_preds)

        # Compute sensitivity per sequence
        all_sensitivities = []
        all_mean_abs_diffs = []

        for seq_idx in range(n_seqs):
            if seq_idx % 100 == 0 and seq_idx > 0:
                print(f"    seq {seq_idx}/{n_seqs}...")

            seq = seqs[seq_idx]  # [L, 4]

            # Generate all single-nt mutations for this sequence
            mutations = []
            for pos in range(seq_len):
                orig_nuc = np.argmax(seq[pos])
                for nuc in range(n_channels):
                    if nuc == orig_nuc:
                        continue
                    mut = seq.copy()
                    mut[pos, :] = 0
                    mut[pos, nuc] = 1
                    mutations.append(mut)

            mutations = np.array(mutations)  # [690, L, 4]

            # Predict in batches
            mut_preds = []
            with torch.no_grad():
                for i in range(0, len(mutations), batch_size):
                    batch = torch.from_numpy(mutations[i:i+batch_size]).to(device)
                    p = model.predict_denoised(batch)
                    mut_preds.append(p.cpu().numpy())
            mut_preds = np.concatenate(mut_preds)

            diffs = np.abs(mut_preds - ref_preds[seq_idx])
            mean_abs_diff = float(diffs.mean())
            ref_abs = abs(float(ref_preds[seq_idx]))
            sensitivity = mean_abs_diff / max(ref_abs, 1e-6)

            all_sensitivities.append(sensitivity)
            all_mean_abs_diffs.append(mean_abs_diff)

        results[name] = {
            "mean_sensitivity": float(np.mean(all_sensitivities)),
            "median_sensitivity": float(np.median(all_sensitivities)),
            "std_sensitivity": float(np.std(all_sensitivities)),
            "mean_abs_diff": float(np.mean(all_mean_abs_diffs)),
            "sensitivities": all_sensitivities[:100],  # save subset
        }

        print(f"    sensitivity: mean={np.mean(all_sensitivities):.4f}, "
              f"median={np.median(all_sensitivities):.4f}")

    # Save
    output_path = os.path.join(output_dir, "sensitivity_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    return results


# ============================================================================
# F3: Noise Fraction Estimation
# ============================================================================
def run_f3_noise_estimation(paired_file, output_dir):
    """
    F3: Estimate noise fraction using replicate STDs.

    noise_var = mean(k562_stds^2) + mean(hepg2_stds^2)
    total_var = Var(k562_activities) + Var(hepg2_activities)
    noise_fraction = noise_var / total_var

    Also compute cross-cell-type Spearman on paired test set.
    """
    print("\n" + "=" * 70)
    print("F3: Noise Fraction Estimation")
    print("=" * 70)

    with h5py.File(paired_file, "r") as f:
        k562_acts = f["k562_activities"][:]
        hepg2_acts = f["hepg2_activities"][:]
        splits = f["split"][:]
        k562_stds = f["k562_stds"][:] if "k562_stds" in f else None
        hepg2_stds = f["hepg2_stds"][:] if "hepg2_stds" in f else None

    results = {}

    for split_name, split_code in [("train", 0), ("test", 2), ("all", None)]:
        mask = np.ones(len(k562_acts), dtype=bool) if split_code is None else (splits == split_code)
        k_acts = k562_acts[mask]
        h_acts = hepg2_acts[mask]

        split_results = {
            "n_samples": int(mask.sum()),
            "k562_var": float(np.var(k_acts)),
            "hepg2_var": float(np.var(h_acts)),
            "total_var": float(np.var(k_acts) + np.var(h_acts)),
            "cross_spearman": float(spearmanr(k_acts, h_acts)[0]),
            "cross_pearson": float(pearsonr(k_acts, h_acts)[0]),
        }

        if k562_stds is not None and hepg2_stds is not None:
            k_stds = k562_stds[mask]
            h_stds = hepg2_stds[mask]
            noise_var_k = float(np.mean(k_stds ** 2))
            noise_var_h = float(np.mean(h_stds ** 2))
            noise_var = noise_var_k + noise_var_h
            total_var = np.var(k_acts) + np.var(h_acts)

            split_results["noise_var_k562"] = noise_var_k
            split_results["noise_var_hepg2"] = noise_var_h
            split_results["noise_var_total"] = noise_var
            split_results["noise_fraction"] = float(noise_var / max(total_var, 1e-10))
            split_results["signal_to_noise"] = float(max(total_var, 1e-10) / max(noise_var, 1e-10))

            print(f"  {split_name}: noise_frac={split_results['noise_fraction']:.4f}, "
                  f"SNR={split_results['signal_to_noise']:.2f}, "
                  f"cross_sp={split_results['cross_spearman']:.4f}")
        else:
            print(f"  {split_name}: cross_sp={split_results['cross_spearman']:.4f} "
                  f"(no replicate STDs available)")

        results[split_name] = split_results

    # Save
    output_path = os.path.join(output_dir, "noise_estimation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    return results


# ============================================================================
# F4: Experiment Probe Multi-Point Extraction
# ============================================================================
def run_f4_probe_multipoint(models_dict, k562_file, hepg2_file, device,
                             output_dir, batch_size=512):
    """
    F4: Run experiment probe + activity probe at 3 extraction points.

    1. Pre-BN raw: model.base_model.encode(x)
    2. Denoised: model.encode(x, experiment_id=None)
    3. Experiment-specific: model.encode(x, experiment_id=0) and =1

    Also includes random-label control for experiment probe.
    """
    print("\n" + "=" * 70)
    print("F4: Experiment Probe Multi-Point Extraction")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression, Ridge

    # Load data
    n_samples = 3000
    k562_seqs, k562_acts = load_test_data(k562_file, n_samples)
    hepg2_seqs, hepg2_acts = load_test_data(hepg2_file, n_samples)

    n_per_type = min(len(k562_seqs), len(hepg2_seqs))
    combined_seqs = np.concatenate([k562_seqs[:n_per_type], hepg2_seqs[:n_per_type]])
    exp_labels = np.array([0] * n_per_type + [1] * n_per_type)
    acts_all = np.concatenate([k562_acts[:n_per_type], hepg2_acts[:n_per_type]])

    rng = np.random.RandomState(42)
    n = len(combined_seqs)
    perm = rng.permutation(n)
    split_idx = int(0.8 * n)
    train_idx, test_idx = perm[:split_idx], perm[split_idx:]

    # Random labels for control
    random_labels = rng.randint(0, 2, size=n)

    all_rows = []

    for model_name, (model, config) in models_dict.items():
        print(f"  {model_name}...")
        model.eval()

        def extract_reps(seqs, fn):
            reps = []
            with torch.no_grad():
                for i in range(0, len(seqs), batch_size):
                    batch = torch.from_numpy(seqs[i:i+batch_size]).to(device)
                    r = fn(batch)
                    reps.append(r.cpu().numpy())
            return np.concatenate(reps)

        extraction_points = {
            "pre_bn_raw": lambda x: model.base_model.encode(x),
            "denoised": lambda x: model.encode(x, experiment_id=None),
        }
        if model.n_experiments >= 2:
            extraction_points["k562_specific"] = lambda x: model.encode(x, experiment_id=0)
            extraction_points["hepg2_specific"] = lambda x: model.encode(x, experiment_id=1)

        for point_name, fn in extraction_points.items():
            reps = extract_reps(combined_seqs, fn)

            # Experiment probe
            exp_clf = LogisticRegression(max_iter=1000, C=1.0)
            exp_clf.fit(reps[train_idx], exp_labels[train_idx])
            exp_acc = float(exp_clf.score(reps[test_idx], exp_labels[test_idx]))

            # Random-label control
            rand_clf = LogisticRegression(max_iter=1000, C=1.0)
            rand_clf.fit(reps[train_idx], random_labels[train_idx])
            rand_acc = float(rand_clf.score(reps[test_idx], random_labels[test_idx]))

            # Activity probe
            act_reg = Ridge(alpha=1.0)
            act_reg.fit(reps[train_idx], acts_all[train_idx])
            act_r2 = float(act_reg.score(reps[test_idx], acts_all[test_idx]))
            act_preds = act_reg.predict(reps[test_idx])
            act_sp = float(spearmanr(act_preds, acts_all[test_idx])[0])

            row = {
                "model": model_name,
                "extraction_point": point_name,
                "experiment_probe_accuracy": exp_acc,
                "random_label_accuracy": rand_acc,
                "activity_probe_r2": act_r2,
                "activity_probe_spearman": act_sp,
            }
            all_rows.append(row)

            print(f"    {point_name}: exp_acc={exp_acc:.4f}, "
                  f"rand_acc={rand_acc:.4f}, act_R2={act_r2:.4f}")

    # Save CSV
    output_path = os.path.join(output_dir, "probe_multipoint.csv")
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(all_rows)} rows to {output_path}")

    return all_rows


# ============================================================================
# F5: GC Content Decomposition
# ============================================================================
def run_f5_gc_decomposition(models_dict, k562_file, cagi5_dir, references_file,
                             device, output_dir, batch_size=512):
    """
    F5: GC content decomposition.

    1. Partial correlation: Spearman(pred, activity | GC) via residualization
    2. GC-stratified CAGI5: bin variants by GC context quartile
    """
    print("\n" + "=" * 70)
    print("F5: GC Content Decomposition")
    print("=" * 70)

    seqs, acts = load_test_data(k562_file)

    # Compute GC content: C=idx1, G=idx2
    gc = (seqs[:, :, 1].sum(axis=1) + seqs[:, :, 2].sum(axis=1)) / seqs.shape[1]

    results = {}

    for name, (model, config) in models_dict.items():
        print(f"  {name}...")
        model.eval()

        # Get predictions
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch = torch.from_numpy(seqs[i:i+batch_size]).to(device)
                p = model.predict_denoised(batch)
                all_preds.append(p.cpu().numpy())
        preds = np.concatenate(all_preds)

        # Raw Spearman
        raw_sp = float(spearmanr(preds, acts)[0])

        # Partial correlation via residualization
        # Residualize both pred and activity with respect to GC
        from numpy.polynomial.polynomial import polyfit, polyval
        gc_pred_coeffs = polyfit(gc, preds, deg=3)
        gc_act_coeffs = polyfit(gc, acts, deg=3)
        pred_resid = preds - polyval(gc, gc_pred_coeffs)
        act_resid = acts - polyval(gc, gc_act_coeffs)
        partial_sp = float(spearmanr(pred_resid, act_resid)[0])

        # GC correlation with prediction and activity
        gc_pred_sp = float(spearmanr(gc, preds)[0])
        gc_act_sp = float(spearmanr(gc, acts)[0])

        results[name] = {
            "raw_spearman": raw_sp,
            "partial_spearman_given_gc": partial_sp,
            "gc_pred_spearman": gc_pred_sp,
            "gc_activity_spearman": gc_act_sp,
            "gc_explained_fraction": float(1 - partial_sp / max(abs(raw_sp), 1e-6)),
        }

        print(f"    raw_sp={raw_sp:.4f}, partial_sp|GC={partial_sp:.4f}, "
              f"gc_explained={results[name]['gc_explained_fraction']:.4f}")

    # GC-stratified CAGI5
    gc_cagi5_results = {}
    if os.path.exists(references_file) and os.path.isdir(cagi5_dir):
        with open(references_file) as f:
            references = json.load(f)

        # Only run for key models
        for model_name in list(models_dict.keys())[:10]:
            model, config = models_dict[model_name]

            for tsv_file in sorted(os.listdir(cagi5_dir)):
                if not tsv_file.startswith("challenge_") or not tsv_file.endswith(".tsv"):
                    continue
                element = tsv_file.replace("challenge_", "").replace(".tsv", "")
                if element not in references:
                    continue

                ref_data = references[element]
                ref_seq = ref_data['sequence']
                ref_start = ref_data['start']

                filepath = os.path.join(cagi5_dir, tsv_file)
                with open(filepath) as f_tsv:
                    lines = f_tsv.readlines()
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

                # Compute GC context for each variant
                gc_vals = []
                for _, row in df.iterrows():
                    idx = row['Pos'] - ref_start
                    start = max(0, idx - 10)
                    end = min(len(ref_seq), idx + 11)
                    context = ref_seq[start:end].upper()
                    gc_val = (context.count('G') + context.count('C')) / max(len(context), 1)
                    gc_vals.append(gc_val)
                df['gc_context'] = gc_vals

                try:
                    df['gc_quartile'] = pd.qcut(df['gc_context'], q=4, labels=False, duplicates='drop')
                except ValueError:
                    continue

                for q in sorted(df['gc_quartile'].unique()):
                    sub = df[df['gc_quartile'] == q]
                    if len(sub) < 10:
                        continue
                    metrics = evaluate_cagi5_element(
                        model, ref_data, sub.reset_index(drop=True), device)
                    if metrics:
                        key = f"{model_name}_{element}_q{q}"
                        gc_cagi5_results[key] = {
                            "model": model_name,
                            "element": element,
                            "gc_quartile": int(q),
                            "n_variants": metrics['n_variants'],
                            "spearman": metrics['spearman'],
                        }

    results["gc_stratified_cagi5"] = gc_cagi5_results

    # Save
    output_path = os.path.join(output_dir, "gc_decomposition.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Additional analyses for DISENTANGLE paper")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--k562_data", default="data/processed/dream_K562.h5")
    parser.add_argument("--hepg2_data", default="data/processed/dream_HepG2.h5")
    parser.add_argument("--paired_data", default="data/processed/paired_K562_HepG2.h5")
    parser.add_argument("--cagi5_dir",
                        default="../data/raw/dream_rnn_lentimpra/data/CAGI5")
    parser.add_argument("--cagi5_references", default="../data/cagi5_references.json")
    parser.add_argument("--eval_csv", default="results/evaluation_summary_full.csv")
    parser.add_argument("--output_dir", default="results/additional_analyses/")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Analyses to skip (e1 e2 f1 f2 f3 f4 f5)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all models
    all_model_names = find_all_models(args.results_dir)
    print(f"Found {len(all_model_names)} trained models")

    # Load key models (for expensive analyses) - original 10 + new experiments
    key_model_names = KEY_MODELS_10 + KEY_MODELS_NEW
    key_models = {}
    for name in key_model_names:
        model_dir = os.path.join(args.results_dir, name)
        if not os.path.exists(os.path.join(model_dir, "best_model.pt")):
            print(f"  Key model not found: {name}")
            continue
        model, config = load_model(model_dir, device)
        if model is not None:
            key_models[name] = (model, config)
            print(f"  Loaded key model: {name}")

    # Load all 42 models (for cheap analyses)
    all_models = {}
    for name in all_model_names:
        if name in key_models:
            all_models[name] = key_models[name]
            continue
        model_dir = os.path.join(args.results_dir, name)
        model, config = load_model(model_dir, device)
        if model is not None:
            all_models[name] = (model, config)

    print(f"\nLoaded: {len(key_models)} key models, {len(all_models)} total models")

    # ---- Run analyses ----
    all_results = {}

    # F3: Noise Fraction Estimation (no model loading needed, cheapest)
    if "f3" not in args.skip:
        all_results["f3_noise_estimation"] = run_f3_noise_estimation(
            args.paired_data, args.output_dir
        )

    # F1: Learning Dynamics (reads history.json, no model loading needed)
    if "f1" not in args.skip:
        all_results["f1_learning_dynamics"] = run_f1_learning_dynamics(
            args.results_dir, args.eval_csv, args.output_dir
        )

    # E1: Non-Circular Multi-Point (all models, moderate cost)
    if "e1" not in args.skip:
        all_results["e1_noncircular"] = run_e1_noncircular(
            all_models, args.paired_data, device, args.output_dir
        )

    # F5: GC Content Decomposition (all models, moderate cost)
    if "f5" not in args.skip:
        all_results["f5_gc_decomposition"] = run_f5_gc_decomposition(
            all_models, args.k562_data, args.cagi5_dir, args.cagi5_references,
            device, args.output_dir
        )

    # F4: Experiment Probe Multi-Point (key models, moderate cost)
    if "f4" not in args.skip:
        all_results["f4_probe_multipoint"] = run_f4_probe_multipoint(
            key_models, args.k562_data, args.hepg2_data, device, args.output_dir
        )

    # E2: Stratified CAGI5 (key models, expensive)
    if "e2" not in args.skip:
        all_results["e2_stratified_cagi5"] = run_e2_stratified_cagi5(
            key_models, args.cagi5_dir, args.cagi5_references, device,
            args.output_dir
        )

    # F2: Representation Sensitivity (key models, most expensive)
    if "f2" not in args.skip:
        all_results["f2_sensitivity"] = run_f2_sensitivity(
            key_models, args.k562_data, device, args.output_dir
        )

    # Save master results
    master_path = os.path.join(args.output_dir, "additional_analyses_master.json")
    with open(master_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ALL ADDITIONAL ANALYSES COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {args.output_dir}")
    for name in all_results:
        print(f"    {name}: OK")


if __name__ == "__main__":
    main()
