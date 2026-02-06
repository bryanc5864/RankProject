#!/usr/bin/env python3
"""
Unified evaluation script for DISENTANGLE models.

Runs Tier 1 (within-experiment), Tier 2 (cross-experiment), and
Tier 3 (CAGI5 variant effect) evaluation.

Usage:
    python evaluate.py --results_dir results/ --data data/processed/dream_K562.h5
"""

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.linear_model import LogisticRegression, Ridge
from models.encoders import build_encoder
from models.wrapper import DisentangleWrapper

# CAGI5 constants
K562_ELEMENTS = ['GP1BB', 'HBB', 'HBG1', 'PKLR']
HEPG2_ELEMENTS = ['F9', 'LDLR', 'SORT1']


def load_model(model_dir, device):
    """Load a trained model from a results directory."""
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return None, None

    with open(config_path) as f:
        config = json.load(f)

    architecture = config["architecture"]
    n_data_files = len(config.get("data_files", [1]))
    has_paired = config.get("paired_data") is not None
    n_experiments = max(n_data_files, 2 if has_paired else 1)

    # Infer n_experiments from checkpoint if it has more norms than expected
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    n_exp_in_ckpt = sum(1 for k in state_dict if k.startswith("exp_norms.") and k.endswith(".weight"))
    if n_exp_in_ckpt > n_experiments:
        n_experiments = n_exp_in_ckpt

    encoder = build_encoder(architecture, config)
    model = DisentangleWrapper(encoder, n_experiments=n_experiments, config=config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_within_experiment(model, data_file, split="test", device="cpu",
                               batch_size=512):
    """Tier 1: Within-experiment evaluation metrics."""
    split_code = {"train": 0, "val": 1, "test": 2}[split]

    with h5py.File(data_file, "r") as f:
        splits = f["split"][:]
        mask = splits == split_code
        sequences = f["sequences"][:][mask].astype(np.float32)
        activities = f["activities"][:][mask]

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[i:i+batch_size]).to(device)
            if hasattr(model, "predict_denoised"):
                preds = model.predict_denoised(batch)
            else:
                preds = model(batch)
            all_preds.append(preds.cpu().numpy())

    preds = np.concatenate(all_preds)

    metrics = {
        "pearson": float(pearsonr(preds, activities)[0]),
        "spearman": float(spearmanr(preds, activities)[0]),
        "kendall": float(kendalltau(preds, activities)[0]),
        "mse": float(np.mean((preds - activities) ** 2)),
        "n_samples": len(activities),
    }

    # NDCG@k for top predictions
    for k in [100, 500, 1000]:
        if k <= len(activities):
            metrics[f"ndcg@{k}"] = compute_ndcg(activities, preds, k)

    # Extreme value correlation (top/bottom 10%)
    n = len(activities)
    n_extreme = max(int(0.1 * n), 50)
    top_idx = np.argsort(activities)[-n_extreme:]
    bottom_idx = np.argsort(activities)[:n_extreme]
    extreme_idx = np.concatenate([top_idx, bottom_idx])
    metrics["spearman_extreme"] = float(
        spearmanr(preds[extreme_idx], activities[extreme_idx])[0]
    )

    return metrics


def compute_ndcg(true_relevance, predicted_scores, k):
    """Compute NDCG@k."""
    pred_order = np.argsort(-predicted_scores)[:k]
    true_order = np.argsort(-true_relevance)[:k]

    # Use 2^relevance - 1 as gains (normalize relevance to [0,1] first)
    rel_min, rel_max = true_relevance.min(), true_relevance.max()
    if rel_max == rel_min:
        return 1.0
    norm_rel = (true_relevance - rel_min) / (rel_max - rel_min)

    dcg = sum(norm_rel[pred_order[i]] / np.log2(i + 2) for i in range(k))
    idcg = sum(norm_rel[true_order[i]] / np.log2(i + 2) for i in range(k))

    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_cross_experiment(model, k562_file, hepg2_file, paired_file,
                               device="cpu", batch_size=512):
    """Tier 2: Cross-experiment transfer evaluation.

    Train on K562, evaluate on HepG2 (and vice versa) for matched sequences.
    """
    with h5py.File(paired_file, "r") as f:
        sequences = f["sequences"][:].astype(np.float32)
        k562_acts = f["k562_activities"][:]
        hepg2_acts = f["hepg2_activities"][:]
        consensus = f["consensus_ranks"][:]
        splits = f["split"][:]

    # Only use test split
    test_mask = splits == 2
    if test_mask.sum() == 0:
        return {}

    sequences = sequences[test_mask]
    k562_acts = k562_acts[test_mask]
    hepg2_acts = hepg2_acts[test_mask]
    consensus = consensus[test_mask]

    # Get predictions
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[i:i+batch_size]).to(device)
            if hasattr(model, "predict_denoised"):
                preds = model.predict_denoised(batch)
            else:
                preds = model(batch)
            all_preds.append(preds.cpu().numpy())
    preds = np.concatenate(all_preds)

    metrics = {
        "cross_pearson_k562": float(pearsonr(preds, k562_acts)[0]),
        "cross_spearman_k562": float(spearmanr(preds, k562_acts)[0]),
        "cross_pearson_hepg2": float(pearsonr(preds, hepg2_acts)[0]),
        "cross_spearman_hepg2": float(spearmanr(preds, hepg2_acts)[0]),
        "cross_pearson_consensus": float(pearsonr(preds, consensus)[0]),
        "cross_spearman_consensus": float(spearmanr(preds, consensus)[0]),
        "cross_n_samples": len(preds),
    }
    return metrics


def one_hot_encode_for_disentangle(seq, seq_len=230):
    """One-hot encode a DNA sequence to [seq_len, 4] for DISENTANGLE models."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((seq_len, 4), dtype=np.float32)
    for i, base in enumerate(seq[:seq_len].upper()):
        if base in mapping:
            one_hot[i, mapping[base]] = 1.0
        else:
            one_hot[i, :] = 0.25
    return one_hot


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
        seq = 'N' * (-start) + var_seq[:window + start]
    elif end > len(var_seq):
        seq = var_seq[start:] + 'N' * (end - len(var_seq))
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
        seq = 'N' * (-start) + ref_seq[:window + start]
    elif end > len(ref_seq):
        seq = ref_seq[start:] + 'N' * (end - len(ref_seq))
    else:
        seq = ref_seq[start:end]
    return seq[:window]


def predict_sequences_disentangle(model, sequences, device, batch_size=256,
                                   experiment_id=None):
    """Get predictions for a list of DNA string sequences using DISENTANGLE model.

    Args:
        model: DISENTANGLE model
        sequences: List of DNA string sequences
        device: torch device
        batch_size: Batch size for inference
        experiment_id: If None, use denoised (averaged BN). If int, use that
                      experiment's BN layer (0=K562, 1=HepG2 for matched inference).
    """
    X = np.array([one_hot_encode_for_disentangle(seq) for seq in sequences])
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).to(device)
            if experiment_id is not None and hasattr(model, 'predict_matched'):
                out = model.predict_matched(batch, experiment_id)
            elif hasattr(model, 'predict_denoised'):
                out = model.predict_denoised(batch)
            else:
                out = model(batch)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)


def evaluate_cagi5_element(model, ref_data, cagi5_df, device, window=230,
                           min_confidence=None, experiment_id=None):
    """Evaluate model on a single CAGI5 element, with optional confidence filtering.

    Uses Alt - Ref approach: variant effect = model(Alt) - model(Ref)

    Args:
        experiment_id: If provided, use matched BN inference (0=K562, 1=HepG2).
                      If None, use denoised (averaged) inference.
    """
    if min_confidence is not None:
        cagi5_df = cagi5_df[cagi5_df['Confidence'] >= min_confidence].copy()
        if len(cagi5_df) == 0:
            return None

    ref_seq = ref_data['sequence']
    ref_start = ref_data['start']

    alt_seqs, ref_seqs, valid_idx = [], [], []
    for i, row in cagi5_df.iterrows():
        alt_s = get_variant_sequence(ref_seq, ref_start, row['Pos'],
                                     row['Ref'], row['Alt'], window)
        ref_s = get_ref_sequence(ref_seq, ref_start, row['Pos'], window)
        if alt_s and ref_s and len(alt_s) == window and len(ref_s) == window:
            alt_seqs.append(alt_s)
            ref_seqs.append(ref_s)
            valid_idx.append(i)

    if len(alt_seqs) == 0:
        return None

    alt_preds = predict_sequences_disentangle(model, alt_seqs, device,
                                               experiment_id=experiment_id)
    ref_preds = predict_sequences_disentangle(model, ref_seqs, device,
                                               experiment_id=experiment_id)
    variant_effects = alt_preds - ref_preds
    ground_truth = cagi5_df.loc[valid_idx, 'Value'].values

    return {
        'n_variants': len(variant_effects),
        'spearman': float(spearmanr(variant_effects, ground_truth)[0]),
        'pearson': float(pearsonr(variant_effects, ground_truth)[0]),
    }


def infer_cell_types(config):
    """Infer which cell types a DISENTANGLE model was trained on."""
    data_files = config.get("data_files", [])
    cell_types = set()
    for f in data_files:
        f_lower = f.lower()
        if 'k562' in f_lower:
            cell_types.add('K562')
        if 'hepg2' in f_lower:
            cell_types.add('HepG2')
    if not cell_types:
        cell_types.add('K562')  # default
    return cell_types


def evaluate_cagi5(model, cagi5_dir, references_file, device, config=None,
                   window=230):
    """Tier 3: CAGI5 saturation mutagenesis variant effect prediction.

    Matches the reference evaluate_cagi5.py:
    - Evaluates ALL 15 CAGI5 elements
    - Evaluates twice: all SNPs and high-confidence (>=0.1) SNPs
    - Computes matched means based on cell-type
    - NEW: Also evaluates with matched-BN inference (cell-type-specific BN)
    """
    if not os.path.exists(references_file):
        print(f"  CAGI5 references not found: {references_file}")
        return {}
    if not os.path.isdir(cagi5_dir):
        print(f"  CAGI5 data not found: {cagi5_dir}")
        return {}

    with open(references_file) as f:
        references = json.load(f)

    # Load CAGI5 TSV files
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

    # Determine matched elements based on training cell types
    cell_types = infer_cell_types(config) if config else {'K562'}
    matched_elements = set()
    if 'K562' in cell_types:
        matched_elements.update(K562_ELEMENTS)
    if 'HepG2' in cell_types:
        matched_elements.update(HEPG2_ELEMENTS)

    # Element to experiment_id mapping for matched-BN inference
    element_to_exp_id = {}
    for elem in K562_ELEMENTS:
        element_to_exp_id[elem] = 0  # K562 BN
    for elem in HEPG2_ELEMENTS:
        element_to_exp_id[elem] = 1  # HepG2 BN

    # Check if model supports matched-BN inference
    has_matched_bn = (hasattr(model, 'predict_matched') and
                      hasattr(model, 'n_experiments') and
                      model.n_experiments >= 2)

    metrics = {}
    all_spearman = []
    all_pearson = []
    highconf_spearman = []
    highconf_pearson = []
    matched_all_spearman = []
    matched_highconf_spearman = []

    # Matched-BN metrics (new)
    matched_bn_all_spearman = []
    matched_bn_highconf_spearman = []

    for element, df in cagi5_data.items():
        if element not in references:
            continue

        # Evaluate with all SNPs (denoised/averaged BN)
        metrics_all = evaluate_cagi5_element(
            model, references[element], df, device, window
        )
        if metrics_all is None:
            continue

        # Evaluate with high-confidence SNPs (>= 0.1) (denoised)
        metrics_hc = evaluate_cagi5_element(
            model, references[element], df, device, window, min_confidence=0.1
        )

        # Store all-SNPs metrics
        metrics[f"cagi5_all_{element}_spearman"] = metrics_all['spearman']
        metrics[f"cagi5_all_{element}_pearson"] = metrics_all['pearson']
        metrics[f"cagi5_all_{element}_n"] = metrics_all['n_variants']
        all_spearman.append(metrics_all['spearman'])
        all_pearson.append(metrics_all['pearson'])

        # Store high-confidence metrics
        if metrics_hc:
            metrics[f"cagi5_highconf_{element}_spearman"] = metrics_hc['spearman']
            metrics[f"cagi5_highconf_{element}_pearson"] = metrics_hc['pearson']
            metrics[f"cagi5_highconf_{element}_n"] = metrics_hc['n_variants']
            highconf_spearman.append(metrics_hc['spearman'])
            highconf_pearson.append(metrics_hc['pearson'])

        # Track matched elements (denoised)
        if element in matched_elements:
            matched_all_spearman.append(metrics_all['spearman'])
            if metrics_hc:
                matched_highconf_spearman.append(metrics_hc['spearman'])

        # Matched-BN inference: use cell-type-specific BN layer
        if has_matched_bn and element in element_to_exp_id:
            exp_id = element_to_exp_id[element]

            # All SNPs with matched BN
            metrics_matched_all = evaluate_cagi5_element(
                model, references[element], df, device, window,
                experiment_id=exp_id
            )
            if metrics_matched_all:
                metrics[f"cagi5_matched_bn_all_{element}_spearman"] = metrics_matched_all['spearman']
                matched_bn_all_spearman.append(metrics_matched_all['spearman'])

            # High-confidence with matched BN
            metrics_matched_hc = evaluate_cagi5_element(
                model, references[element], df, device, window,
                min_confidence=0.1, experiment_id=exp_id
            )
            if metrics_matched_hc:
                metrics[f"cagi5_matched_bn_hc_{element}_spearman"] = metrics_matched_hc['spearman']
                matched_bn_highconf_spearman.append(metrics_matched_hc['spearman'])

    # Mean across ALL elements (denoised)
    if all_spearman:
        metrics["cagi5_all_mean_spearman"] = float(np.mean(all_spearman))
        metrics["cagi5_all_mean_pearson"] = float(np.mean(all_pearson))
    if highconf_spearman:
        metrics["cagi5_highconf_mean_spearman"] = float(np.mean(highconf_spearman))
        metrics["cagi5_highconf_mean_pearson"] = float(np.mean(highconf_pearson))

    # Mean across MATCHED elements only (denoised)
    if matched_all_spearman:
        metrics["cagi5_all_matched_mean_spearman"] = float(np.mean(matched_all_spearman))
    if matched_highconf_spearman:
        metrics["cagi5_highconf_matched_mean_spearman"] = float(np.mean(matched_highconf_spearman))

    # Mean across matched-BN elements (NEW - cell-type-specific BN)
    if matched_bn_all_spearman:
        metrics["cagi5_matched_bn_all_mean_spearman"] = float(np.mean(matched_bn_all_spearman))
    if matched_bn_highconf_spearman:
        metrics["cagi5_matched_bn_hc_mean_spearman"] = float(np.mean(matched_bn_highconf_spearman))

    return metrics


def extract_representations(model, data_file, split="test", device="cpu",
                            batch_size=512, experiment_id=None):
    """Extract learned representations from a model for a dataset."""
    split_code = {"train": 0, "val": 1, "test": 2}[split]

    with h5py.File(data_file, "r") as f:
        splits = f["split"][:]
        mask = splits == split_code
        sequences = f["sequences"][:][mask].astype(np.float32)
        activities = f["activities"][:][mask]

    all_reps = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[i:i+batch_size]).to(device)
            reps = model.encode(batch, experiment_id=experiment_id)
            all_reps.append(reps.cpu().numpy())

    return np.concatenate(all_reps), activities


def evaluate_multipoint_cross_experiment(model, paired_file, device="cpu",
                                          batch_size=512):
    """
    E1: Non-circular cross-experiment evaluation with multi-point extraction.

    Evaluates predictions from 3 extraction points against actual activities:
      1. Pre-BN raw: encoder output -> prediction head (no BN)
      2. Denoised: averaged BN output -> prediction head
      3. Experiment-specific: experiment BN -> prediction head

    Returns correlations vs actual K562 and HepG2 activities.
    """
    with h5py.File(paired_file, "r") as f:
        sequences = f["sequences"][:].astype(np.float32)
        k562_acts = f["k562_activities"][:]
        hepg2_acts = f["hepg2_activities"][:]
        splits = f["split"][:]

    test_mask = splits == 2
    if test_mask.sum() == 0:
        return {}

    sequences = sequences[test_mask]
    k562_acts = k562_acts[test_mask]
    hepg2_acts = hepg2_acts[test_mask]

    metrics = {}

    def get_preds_at_point(model, seqs, point_name, experiment_id=None):
        """Get predictions for a specific extraction point."""
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch = torch.from_numpy(seqs[i:i+batch_size]).to(device)
                if point_name == "pre_bn_raw":
                    # Raw encoder output, skip BN, go directly to prediction head
                    h = model.base_model.encode(batch)
                    preds = model.prediction_head(h).squeeze(-1)
                elif point_name == "denoised":
                    preds = model.predict_denoised(batch)
                elif point_name == "experiment_specific":
                    h = model.encode(batch, experiment_id=experiment_id)
                    preds = model.prediction_head(h).squeeze(-1)
                else:
                    preds = model.predict_denoised(batch)
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds)

    # 1. Pre-BN raw predictions
    preds_raw = get_preds_at_point(model, sequences, "pre_bn_raw")
    metrics["e1_raw_spearman_k562"] = float(spearmanr(preds_raw, k562_acts)[0])
    metrics["e1_raw_spearman_hepg2"] = float(spearmanr(preds_raw, hepg2_acts)[0])

    # 2. Denoised predictions
    preds_den = get_preds_at_point(model, sequences, "denoised")
    metrics["e1_denoised_spearman_k562"] = float(spearmanr(preds_den, k562_acts)[0])
    metrics["e1_denoised_spearman_hepg2"] = float(spearmanr(preds_den, hepg2_acts)[0])

    # 3. Experiment-specific predictions
    if model.n_experiments >= 2:
        preds_k = get_preds_at_point(model, sequences, "experiment_specific", experiment_id=0)
        preds_h = get_preds_at_point(model, sequences, "experiment_specific", experiment_id=1)
        metrics["e1_k562norm_spearman_k562"] = float(spearmanr(preds_k, k562_acts)[0])
        metrics["e1_k562norm_spearman_hepg2"] = float(spearmanr(preds_k, hepg2_acts)[0])
        metrics["e1_hepg2norm_spearman_k562"] = float(spearmanr(preds_h, k562_acts)[0])
        metrics["e1_hepg2norm_spearman_hepg2"] = float(spearmanr(preds_h, hepg2_acts)[0])

    metrics["e1_n_samples"] = len(sequences)
    return metrics


def evaluate_representation_probing(model, k562_file, hepg2_file, device="cpu",
                                     batch_size=512):
    """
    Tier 4: Representation probing analysis.

    1. Experiment probe: can a linear classifier predict which experiment
       a representation came from? Lower accuracy = more invariant.
    2. Activity probe R²: can a linear regressor predict activity from
       representations? Higher R² = more useful features.
    3. Feature overlap: cosine similarity between experiment-probe and
       activity-probe weight vectors.
    """
    # Extract denoised representations (experiment_id=None)
    reps_k562, acts_k562 = extract_representations(
        model, k562_file, "test", device, batch_size, experiment_id=None
    )
    reps_hepg2, acts_hepg2 = extract_representations(
        model, hepg2_file, "test", device, batch_size, experiment_id=None
    )

    # Combine for experiment probing
    reps_all = np.concatenate([reps_k562, reps_hepg2])
    exp_labels = np.concatenate([
        np.zeros(len(reps_k562)),
        np.ones(len(reps_hepg2))
    ])
    acts_all = np.concatenate([acts_k562, acts_hepg2])

    # Shuffle and split into probe train/test (80/20)
    n = len(reps_all)
    perm = np.random.RandomState(42).permutation(n)
    split_idx = int(0.8 * n)
    train_idx, test_idx = perm[:split_idx], perm[split_idx:]

    metrics = {}

    # 1. Experiment probe accuracy
    exp_clf = LogisticRegression(max_iter=1000, C=1.0)
    exp_clf.fit(reps_all[train_idx], exp_labels[train_idx])
    exp_acc = exp_clf.score(reps_all[test_idx], exp_labels[test_idx])
    metrics["probe_experiment_accuracy"] = float(exp_acc)

    # 2. Activity probe R²
    act_reg = Ridge(alpha=1.0)
    act_reg.fit(reps_all[train_idx], acts_all[train_idx])
    act_r2 = act_reg.score(reps_all[test_idx], acts_all[test_idx])
    act_preds = act_reg.predict(reps_all[test_idx])
    act_sp = float(spearmanr(act_preds, acts_all[test_idx])[0])
    metrics["probe_activity_r2"] = float(act_r2)
    metrics["probe_activity_spearman"] = act_sp

    # 3. Feature overlap: cosine similarity between weight vectors
    exp_weights = exp_clf.coef_.flatten()
    act_weights = act_reg.coef_.flatten()
    cos_sim = np.dot(exp_weights, act_weights) / (
        np.linalg.norm(exp_weights) * np.linalg.norm(act_weights) + 1e-10
    )
    metrics["probe_feature_overlap"] = float(np.abs(cos_sim))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DISENTANGLE models")
    parser.add_argument("--results_dir", default="results/",
                        help="Directory containing model subdirectories")
    parser.add_argument("--data", default="data/processed/dream_K562.h5")
    parser.add_argument("--hepg2_data", default="data/processed/dream_HepG2.h5")
    parser.add_argument("--paired_data", default="data/processed/paired_K562_HepG2.h5")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cagi5_dir",
                        default="../data/raw/dream_rnn_lentimpra/data/CAGI5")
    parser.add_argument("--cagi5_references", default="../data/cagi5_references.json")
    parser.add_argument("--output", default="results/evaluation_summary.csv")
    parser.add_argument("--incremental", action="store_true",
                        help="Skip models already in output CSV, append new results")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load existing results if incremental mode
    existing_models = set()
    existing_results = []
    existing_fieldnames = []
    if args.incremental and os.path.exists(args.output):
        import csv as csv_mod
        with open(args.output, "r") as f:
            reader = csv_mod.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            for row in reader:
                existing_results.append(row)
                existing_models.add(row.get("model", ""))
        print(f"Incremental mode: {len(existing_models)} models already evaluated")

    # Find all model directories
    model_dirs = []
    for name in sorted(os.listdir(args.results_dir)):
        d = os.path.join(args.results_dir, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "best_model.pt")):
            if args.incremental and name in existing_models:
                continue
            model_dirs.append((name, d))

    if not model_dirs:
        print("No new models to evaluate.")
        if existing_results:
            print(f"  ({len(existing_models)} already in {args.output})")
        return

    print(f"Found {len(model_dirs)} models to evaluate")

    all_results = []
    for name, model_dir in model_dirs:
        print(f"\nEvaluating: {name}")
        model, config = load_model(model_dir, device)
        if model is None:
            print(f"  Skipping (failed to load)")
            continue

        # Tier 1: Within-experiment
        tier1 = evaluate_within_experiment(model, args.data, "test", device)
        print(f"  Tier1 test: Pearson={tier1['pearson']:.4f}, "
              f"Spearman={tier1['spearman']:.4f}, MSE={tier1['mse']:.4f}")

        # Tier 2: Cross-experiment (if paired data exists)
        tier2 = {}
        if os.path.exists(args.paired_data):
            tier2 = evaluate_cross_experiment(
                model, args.data, args.hepg2_data, args.paired_data, device
            )
            if tier2:
                print(f"  Tier2 cross: K562_spearman={tier2['cross_spearman_k562']:.4f}, "
                      f"HepG2_spearman={tier2['cross_spearman_hepg2']:.4f}, "
                      f"Consensus_spearman={tier2['cross_spearman_consensus']:.4f}")

        # Tier 3: CAGI5 variant effect prediction
        tier3 = {}
        if os.path.exists(args.cagi5_dir) and os.path.exists(args.cagi5_references):
            tier3 = evaluate_cagi5(model, args.cagi5_dir, args.cagi5_references,
                                   device, config=config)
            if "cagi5_all_mean_spearman" in tier3:
                print(f"  Tier3 CAGI5: all_mean_sp={tier3['cagi5_all_mean_spearman']:.4f}", end="")
                if "cagi5_highconf_mean_spearman" in tier3:
                    print(f", hc_mean_sp={tier3['cagi5_highconf_mean_spearman']:.4f}", end="")
                if "cagi5_all_matched_mean_spearman" in tier3:
                    print(f", matched_all={tier3['cagi5_all_matched_mean_spearman']:.4f}", end="")
                if "cagi5_highconf_matched_mean_spearman" in tier3:
                    print(f", matched_hc={tier3['cagi5_highconf_matched_mean_spearman']:.4f}", end="")
                print()

        # E1: Multi-point cross-experiment evaluation (non-circular)
        e1_metrics = {}
        if os.path.exists(args.paired_data):
            e1_metrics = evaluate_multipoint_cross_experiment(
                model, args.paired_data, device
            )
            if e1_metrics:
                print(f"  E1 multipoint: raw_k={e1_metrics.get('e1_raw_spearman_k562', 0):.4f}, "
                      f"den_k={e1_metrics.get('e1_denoised_spearman_k562', 0):.4f}, "
                      f"den_h={e1_metrics.get('e1_denoised_spearman_hepg2', 0):.4f}")

        # Tier 4: Representation probing
        tier4 = {}
        if os.path.exists(args.data) and os.path.exists(args.hepg2_data):
            tier4 = evaluate_representation_probing(
                model, args.data, args.hepg2_data, device
            )
            print(f"  Tier4 probe: exp_acc={tier4['probe_experiment_accuracy']:.4f}, "
                  f"act_R2={tier4['probe_activity_r2']:.4f}, "
                  f"overlap={tier4['probe_feature_overlap']:.4f}")

        result = {
            "model": name,
            "architecture": config.get("architecture", "unknown"),
            "condition": config.get("condition", "unknown"),
            "seed": config.get("seed", 0),
            **tier1,
            **tier2,
            **e1_metrics,
            **tier3,
            **tier4,
        }
        all_results.append(result)

    # Write CSV summary
    if all_results:
        import csv
        # Merge with existing results in incremental mode
        if args.incremental and existing_results:
            combined = existing_results + all_results
        else:
            combined = all_results

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fieldnames = list(combined[0].keys())
        # Ensure all results have all fields
        for r in combined:
            for k in fieldnames:
                r.setdefault(k, "")
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined)

        print(f"\nResults saved to {args.output}")

        # Print summary table
        print("\n" + "=" * 100)
        print(f"{'Model':<45} {'Pearson':>8} {'Spearman':>9} {'MSE':>8} "
              f"{'Cross-Sp':>9}")
        print("-" * 100)
        for r in all_results:
            cross_sp = r.get("cross_spearman_consensus", "")
            if isinstance(cross_sp, float):
                cross_sp = f"{cross_sp:.4f}"
            print(f"{r['model']:<45} {r['pearson']:8.4f} {r['spearman']:9.4f} "
                  f"{r['mse']:8.4f} {cross_sp:>9}")


if __name__ == "__main__":
    main()
