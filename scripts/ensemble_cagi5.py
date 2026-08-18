#!/usr/bin/env python3
"""cross-method ensembling on the dumped CAGI5 predictions, plus effect-size stratification.

ensembles: uniform over all, uniform over top-K by K562 Spearman, Borda rank
aggregation, and ridge with learned weights. ridge overfits the 4-element K562
supervision set - uniform top-5 wins.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

K562_ELEMENTS = ['GP1BB', 'HBB', 'HBG1', 'PKLR']


def k562_mean(per_element_sp):
    """Mean Spearman over K562 elements present in the dict."""
    vals = [per_element_sp[e] for e in K562_ELEMENTS if e in per_element_sp]
    return np.mean(vals) if vals else float('nan')


def all_mean(per_element_sp):
    vals = list(per_element_sp.values())
    return np.mean(vals) if vals else float('nan')


def eval_predictions(predictions_dict, ground_truth_dict, confidence_dict=None):
    """Compute per-element Spearman and aggregates given a {element: predictions} dict."""
    per_el = {}
    per_el_hc = {}
    for element, preds in predictions_dict.items():
        gt = ground_truth_dict[element]
        per_el[element] = spearmanr(preds, gt)[0]
        if confidence_dict is not None:
            hc = confidence_dict[element] >= 0.1
            if hc.sum() > 5:
                per_el_hc[element] = spearmanr(preds[hc], gt[hc])[0]
    return {
        'per_element': per_el,
        'K562_matched_sp': k562_mean(per_el),
        'all_mean_sp': all_mean(per_el),
        'per_element_hc': per_el_hc,
        'K562_matched_sp_hc': k562_mean(per_el_hc) if per_el_hc else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', default='results/deboer_rankloss_1fold_v4/cagi5_predictions.pkl')
    parser.add_argument('--output_csv', default='results/deboer_rankloss_1fold_v4/ensemble_results.csv')
    args = parser.parse_args()

    with open(args.predictions, 'rb') as f:
        data = pickle.load(f)

    # all methods should be in the same order across elements
    methods = data[next(iter(data))]['methods']
    print(f"Methods ({len(methods)}):")
    for m in methods: print(f"  {m}")
    print(f"\nElements ({len(data)}): {sorted(data.keys())}")

    # build per-element ground truth and confidence
    gt = {e: data[e]['ground_truth'] for e in data}
    conf = {e: data[e]['confidence'] for e in data}

    print("\n=== Single methods (K562-matched Sp / All-element Sp) ===")
    rows = []
    for i, m in enumerate(methods):
        preds_dict = {e: data[e]['predictions'][i] for e in data}
        ev = eval_predictions(preds_dict, gt, conf)
        rows.append({
            'method': m, 'type': 'single',
            'K562_sp': ev['K562_matched_sp'],
            'All_sp': ev['all_mean_sp'],
            'K562_hc_sp': ev['K562_matched_sp_hc'],
        })
    single_df = pd.DataFrame(rows).sort_values('K562_sp', ascending=False)
    print(single_df.to_string(index=False))

    print("\n=== A1. Uniform mean over ALL methods ===")
    uniform_all = {e: data[e]['predictions'].mean(axis=0) for e in data}
    ev = eval_predictions(uniform_all, gt, conf)
    print(f"  K562_sp={ev['K562_matched_sp']:.4f}  All_sp={ev['all_mean_sp']:.4f}  HC_K562={ev['K562_matched_sp_hc']:.4f}")
    rows.append({'method': 'uniform_all17', 'type': 'ensemble',
                 'K562_sp': ev['K562_matched_sp'], 'All_sp': ev['all_mean_sp'],
                 'K562_hc_sp': ev['K562_matched_sp_hc']})

    print("\n=== A2. Uniform mean over top-K methods (ranked by K562_sp) ===")
    rank_order = single_df['method'].tolist()
    method_idx = {m: i for i, m in enumerate(methods)}
    for K in [3, 5, 7, 10]:
        top = rank_order[:K]
        idx = [method_idx[m] for m in top]
        uniform_topk = {e: data[e]['predictions'][idx].mean(axis=0) for e in data}
        ev = eval_predictions(uniform_topk, gt, conf)
        print(f"  K={K} ({','.join(top[:3])}...): K562_sp={ev['K562_matched_sp']:.4f}  All_sp={ev['all_mean_sp']:.4f}")
        rows.append({'method': f'uniform_top{K}', 'type': 'ensemble',
                     'K562_sp': ev['K562_matched_sp'], 'All_sp': ev['all_mean_sp'],
                     'K562_hc_sp': ev['K562_matched_sp_hc']})

    print("\n=== A3. Borda rank-aggregation ===")
    # for each element and each method, rank predictions; sum ranks across methods  final score
    for K in [5, 10, len(methods)]:
        top = rank_order[:K]
        idx = [method_idx[m] for m in top]
        borda_dict = {}
        for e in data:
            preds = data[e]['predictions'][idx]  # K × V
            ranks = np.array([rankdata(p) for p in preds])
            borda_dict[e] = ranks.sum(axis=0)
        ev = eval_predictions(borda_dict, gt, conf)
        print(f"  K={K}: K562_sp={ev['K562_matched_sp']:.4f}  All_sp={ev['all_mean_sp']:.4f}")
        rows.append({'method': f'borda_top{K}', 'type': 'ensemble',
                     'K562_sp': ev['K562_matched_sp'], 'All_sp': ev['all_mean_sp'],
                     'K562_hc_sp': ev['K562_matched_sp_hc']})

    print("\n=== A4. RidgeCV with 5-fold on K562 elements ===")
    # train: concat alt-ref preds across 4 K562 elements; target: ground truth
    X_blocks = []
    y_blocks = []
    el_marker = []  # remember element of each row
    for e in K562_ELEMENTS:
        if e in data:
            X_blocks.append(data[e]['predictions'].T)  # V × N
            y_blocks.append(gt[e])
            el_marker.extend([e] * data[e]['predictions'].shape[1])
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    el_marker = np.array(el_marker)
    # K-fold CV: predict each fold using ridge fit on other folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_pred = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        rcv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        rcv.fit(X[tr], y[tr])
        cv_pred[te] = rcv.predict(X[te])
    # Per-element Sp from CV preds
    per_el_ridge = {}
    per_el_hc_ridge = {}
    for e in K562_ELEMENTS:
        mask = el_marker == e
        per_el_ridge[e] = spearmanr(cv_pred[mask], y[mask])[0]
        hc_mask = mask & (np.concatenate([conf[ee] for ee in K562_ELEMENTS if ee in data]) >= 0.1)
        if hc_mask.sum() > 5:
            per_el_hc_ridge[e] = spearmanr(cv_pred[hc_mask], y[hc_mask])[0]
    K562_ridge = np.mean(list(per_el_ridge.values()))
    K562_hc_ridge = np.mean(list(per_el_hc_ridge.values())) if per_el_hc_ridge else float('nan')
    print(f"  K562_sp={K562_ridge:.4f}  HC_K562={K562_hc_ridge:.4f}")
    # use weights from one final fit on all data, apply to all elements (incl non-K562)
    rcv_final = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    rcv_final.fit(X, y)
    ridge_all = {}
    for e in data:
        ridge_all[e] = rcv_final.predict(data[e]['predictions'].T)
    ev = eval_predictions(ridge_all, gt, conf)
    print(f"  (full-fit applied to all elements) All_sp={ev['all_mean_sp']:.4f}")
    rows.append({'method': 'ridgeCV_K562', 'type': 'ensemble',
                 'K562_sp': K562_ridge, 'All_sp': ev['all_mean_sp'],
                 'K562_hc_sp': K562_hc_ridge})

    # save summary
    df = pd.DataFrame(rows).sort_values('K562_sp', ascending=False)
    df.to_csv(args.output_csv, index=False)
    print(f"\n=== FINAL RANKING (all methods + ensembles) ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {args.output_csv}")

    print("\n=== B. Effect-size stratified per-element analysis ===")
    print("Bin variants by |ground truth| (quartiles); report Spearman per bin.")
    # use top single method + best ensemble for this
    top_method = single_df.iloc[0]['method']
    top_idx = method_idx[top_method]
    print(f"\nStratification for top single method: {top_method}")
    for e in K562_ELEMENTS:
        if e not in data:
            continue
        y_e = gt[e]
        p_e = data[e]['predictions'][top_idx]
        abs_y = np.abs(y_e)
        # quartiles
        q = np.quantile(abs_y, [0.25, 0.5, 0.75])
        bins = np.digitize(abs_y, q)
        print(f"  {e}:")
        for b in range(4):
            mask = bins == b
            n = mask.sum()
            if n > 5:
                sp = spearmanr(p_e[mask], y_e[mask])[0]
                print(f"    |y| bin {b} (n={n}, |y| range {abs_y[mask].min():.3f}-{abs_y[mask].max():.3f}): Sp={sp:.4f}")


if __name__ == '__main__':
    main()
