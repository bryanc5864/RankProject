"""
Shared evaluation metrics used across all tiers.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score


def compute_all_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute the full suite of evaluation metrics."""
    return {
        "pearson": float(pearsonr(predictions, targets)[0]),
        "pearson_pvalue": float(pearsonr(predictions, targets)[1]),
        "spearman": float(spearmanr(predictions, targets)[0]),
        "spearman_pvalue": float(spearmanr(predictions, targets)[1]),
        "kendall": float(kendalltau(predictions, targets)[0]),
        "kendall_pvalue": float(kendalltau(predictions, targets)[1]),
        "mse": float(mean_squared_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
        "n_sequences": len(targets),
    }


def compute_ndcg(true_activities: np.ndarray, predicted_scores: np.ndarray,
                 k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    Measures how well the model identifies top regulatory elements.
    """
    pred_order = np.argsort(-predicted_scores)

    dcg = 0.0
    for i in range(min(k, len(true_activities))):
        dcg += true_activities[pred_order[i]] / np.log2(i + 2)

    ideal_order = np.argsort(-true_activities)
    idcg = 0.0
    for i in range(min(k, len(true_activities))):
        idcg += true_activities[ideal_order[i]] / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_direction_accuracy(predictions: np.ndarray, activities: np.ndarray,
                               threshold: float = 1.0,
                               n_samples: int = 100000) -> float:
    """
    For pairs with activity difference > threshold,
    what fraction does the model rank correctly?
    """
    n = len(predictions)
    correct = 0
    total = 0

    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue

        act_diff = activities[i] - activities[j]
        if abs(act_diff) < threshold:
            continue

        pred_diff = predictions[i] - predictions[j]
        if np.sign(act_diff) == np.sign(pred_diff):
            correct += 1
        total += 1

    return correct / total if total > 0 else float("nan")
