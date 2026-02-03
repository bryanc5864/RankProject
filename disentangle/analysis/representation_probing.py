"""
Linear probing analysis to determine what information is encoded
in model representations.

Probes:
1. Experiment/batch probe: predict experiment ID from representations
2. Activity probe: predict activity from representations
3. GC content probe: predict GC content from representations
4. Batch-activity feature overlap: correlation between batch-predictive
   and activity-predictive dimensions

Produces Figure 2 of the paper.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def run_all_probes(
    representations: np.ndarray,
    experiment_ids: np.ndarray,
    activities: np.ndarray,
    gc_contents: np.ndarray = None,
) -> dict:
    """
    Run all probing analyses.

    Args:
        representations: [N, D] model representations
        experiment_ids: [N] integer experiment labels
        activities: [N] measured activities
        gc_contents: [N] GC content per sequence (optional)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(representations)

    results = {}

    # Probe 1: Experiment ID (classification)
    n_classes = len(np.unique(experiment_ids))
    probe_exp = LogisticRegression(max_iter=1000, C=1.0)
    scores_exp = cross_val_score(probe_exp, X, experiment_ids, cv=5, scoring="accuracy")
    results["experiment_probe_accuracy"] = float(scores_exp.mean())
    results["experiment_probe_std"] = float(scores_exp.std())
    results["experiment_probe_chance"] = 1.0 / n_classes
    print(
        f"Experiment probe: {scores_exp.mean():.3f} +/- {scores_exp.std():.3f} "
        f"(chance: {1.0/n_classes:.3f})"
    )

    # Probe 2: Activity (regression)
    probe_act = Ridge(alpha=1.0)
    scores_act = cross_val_score(probe_act, X, activities, cv=5, scoring="r2")
    results["activity_probe_r2"] = float(scores_act.mean())
    results["activity_probe_std"] = float(scores_act.std())
    print(f"Activity probe R2: {scores_act.mean():.3f} +/- {scores_act.std():.3f}")

    # Probe 3: GC content (regression)
    if gc_contents is not None:
        probe_gc = Ridge(alpha=1.0)
        scores_gc = cross_val_score(probe_gc, X, gc_contents, cv=5, scoring="r2")
        results["gc_probe_r2"] = float(scores_gc.mean())
        results["gc_probe_std"] = float(scores_gc.std())
        print(f"GC probe R2: {scores_gc.mean():.3f} +/- {scores_gc.std():.3f}")

    # Probe 4: Batch-activity feature overlap
    probe_exp_full = LogisticRegression(max_iter=1000, C=1.0)
    probe_exp_full.fit(X, experiment_ids)
    batch_importance = np.abs(probe_exp_full.coef_).mean(axis=0)

    probe_act_full = Ridge(alpha=1.0)
    probe_act_full.fit(X, activities)
    activity_importance = np.abs(probe_act_full.coef_)

    overlap = float(np.corrcoef(batch_importance, activity_importance)[0, 1])
    results["batch_activity_feature_overlap"] = overlap
    results["batch_feature_importance"] = batch_importance.tolist()
    results["activity_feature_importance"] = activity_importance.tolist()

    print(f"Batch-activity feature overlap: {overlap:.3f}")
    if overlap > 0.3:
        print(
            "WARNING: High overlap between batch-predictive and "
            "activity-predictive features!"
        )

    return results


def compare_baseline_vs_disentangle(
    baseline_reps: np.ndarray,
    disentangle_reps: np.ndarray,
    experiment_ids: np.ndarray,
    activities: np.ndarray,
    gc_contents: np.ndarray = None,
) -> tuple[dict, dict]:
    """
    Compare probing results between baseline and DISENTANGLE.

    Expected:
    - Baseline: high experiment probe, high batch-activity overlap
    - DISENTANGLE: near-chance experiment probe, low overlap
    - Both: similar activity probe R^2
    """
    print("=== BASELINE MODEL ===")
    baseline_results = run_all_probes(
        baseline_reps, experiment_ids, activities, gc_contents
    )

    print("\n=== DISENTANGLE MODEL ===")
    disentangle_results = run_all_probes(
        disentangle_reps, experiment_ids, activities, gc_contents
    )

    return baseline_results, disentangle_results
