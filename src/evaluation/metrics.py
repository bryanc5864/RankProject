"""
Evaluation Metrics for Ranking Models

Includes rank-based metrics (Spearman, Kendall, NDCG) and standard regression metrics.
"""

import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from typing import Dict, Optional, Union, List
import warnings


def spearman_correlation(pred: Union[torch.Tensor, np.ndarray],
                         target: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Spearman rank correlation coefficient.

    Args:
        pred: Predictions [n_samples]
        target: Ground truth [n_samples]

    Returns:
        Spearman correlation coefficient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = spearmanr(pred, target)

    return float(corr) if not np.isnan(corr) else 0.0


def kendall_tau(pred: Union[torch.Tensor, np.ndarray],
                target: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Kendall's tau rank correlation coefficient.

    Measures the ordinal association between predictions and targets.
    More robust to outliers than Spearman.

    Args:
        pred: Predictions [n_samples]
        target: Ground truth [n_samples]

    Returns:
        Kendall's tau coefficient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau, _ = kendalltau(pred, target)

    return float(tau) if not np.isnan(tau) else 0.0


def pearson_correlation(pred: Union[torch.Tensor, np.ndarray],
                        target: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        pred: Predictions [n_samples]
        target: Ground truth [n_samples]

    Returns:
        Pearson correlation coefficient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = pearsonr(pred, target)

    return float(corr) if not np.isnan(corr) else 0.0


def ndcg_score(pred: Union[torch.Tensor, np.ndarray],
               target: Union[torch.Tensor, np.ndarray],
               k: Optional[int] = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).

    Measures ranking quality with emphasis on top positions.

    Args:
        pred: Predictions [n_samples]
        target: Ground truth relevance scores [n_samples]
        k: Cutoff for NDCG@k (None = use all)

    Returns:
        NDCG score in [0, 1]
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    n = len(pred)
    if k is None:
        k = n
    k = min(k, n)

    # Get ranking by predictions (indices that would sort pred descending)
    pred_ranking = np.argsort(-pred)

    # DCG: sum of (2^rel - 1) / log2(rank + 1) for top-k
    dcg = 0.0
    for i in range(k):
        idx = pred_ranking[i]
        rel = target[idx]
        # Use gain = rel directly for regression targets (not 2^rel - 1)
        dcg += rel / np.log2(i + 2)  # +2 because rank is 1-indexed

    # Ideal DCG: sort by true relevance
    ideal_ranking = np.argsort(-target)
    idcg = 0.0
    for i in range(k):
        idx = ideal_ranking[i]
        rel = target[idx]
        idcg += rel / np.log2(i + 2)

    if idcg < 1e-10:
        return 1.0 if dcg < 1e-10 else 0.0

    return float(dcg / idcg)


def mean_reciprocal_rank(pred: Union[torch.Tensor, np.ndarray],
                         target: Union[torch.Tensor, np.ndarray],
                         threshold: Optional[float] = None) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    For regression, items above threshold (or top quartile) are considered relevant.

    Args:
        pred: Predictions [n_samples]
        target: Ground truth [n_samples]
        threshold: Relevance threshold (default: 75th percentile)

    Returns:
        MRR score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    if threshold is None:
        threshold = np.percentile(target, 75)

    # Get ranking by predictions
    pred_ranking = np.argsort(-pred)

    # Find rank of first relevant item
    for rank, idx in enumerate(pred_ranking, 1):
        if target[idx] >= threshold:
            return 1.0 / rank

    return 0.0


def precision_at_k(pred: Union[torch.Tensor, np.ndarray],
                   target: Union[torch.Tensor, np.ndarray],
                   k: int, threshold: Optional[float] = None) -> float:
    """
    Compute Precision@k.

    Fraction of top-k predictions that are truly relevant.

    Args:
        pred: Predictions [n_samples]
        target: Ground truth [n_samples]
        k: Number of top predictions to consider
        threshold: Relevance threshold (default: 75th percentile)

    Returns:
        Precision@k score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    if threshold is None:
        threshold = np.percentile(target, 75)

    n = len(pred)
    k = min(k, n)

    # Get top-k by predictions
    top_k_indices = np.argsort(-pred)[:k]

    # Count how many are relevant
    n_relevant = np.sum(target[top_k_indices] >= threshold)

    return float(n_relevant / k)


def recall_at_k(pred: Union[torch.Tensor, np.ndarray],
                target: Union[torch.Tensor, np.ndarray],
                k: int, threshold: Optional[float] = None) -> float:
    """
    Compute Recall@k.

    Fraction of all relevant items that appear in top-k predictions.

    Args:
        pred: Predictions [n_samples]
        target: Ground truth [n_samples]
        k: Number of top predictions to consider
        threshold: Relevance threshold (default: 75th percentile)

    Returns:
        Recall@k score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    if threshold is None:
        threshold = np.percentile(target, 75)

    n = len(pred)
    k = min(k, n)

    # Total relevant items
    total_relevant = np.sum(target >= threshold)
    if total_relevant == 0:
        return 1.0

    # Get top-k by predictions
    top_k_indices = np.argsort(-pred)[:k]

    # Count how many relevant items are in top-k
    n_relevant_in_top_k = np.sum(target[top_k_indices] >= threshold)

    return float(n_relevant_in_top_k / total_relevant)


def pairwise_accuracy(pred: Union[torch.Tensor, np.ndarray],
                      target: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute pairwise ranking accuracy.

    Fraction of pairs where the model correctly predicts relative ordering.

    Args:
        pred: Predictions [n_samples]
        target: Ground truth [n_samples]

    Returns:
        Pairwise accuracy in [0, 1]
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    n = len(pred)
    if n < 2:
        return 1.0

    # Compute all pairs
    pred_diff = pred[:, np.newaxis] - pred[np.newaxis, :]
    target_diff = target[:, np.newaxis] - target[np.newaxis, :]

    # Count concordant pairs (same sign)
    # Exclude diagonal and ties in target
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    target_diff_masked = target_diff[mask]
    pred_diff_masked = pred_diff[mask]

    # Exclude ties in target
    non_tie = np.abs(target_diff_masked) > 1e-10
    if non_tie.sum() == 0:
        return 1.0

    concordant = np.sign(pred_diff_masked[non_tie]) == np.sign(target_diff_masked[non_tie])

    return float(concordant.mean())


def compute_all_metrics(pred: Union[torch.Tensor, np.ndarray],
                        target: Union[torch.Tensor, np.ndarray],
                        k_values: List[int] = [10, 50, 100]) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        pred: Predictions [n_samples]
        target: Ground truth [n_samples]
        k_values: Values of k for NDCG@k and Precision@k

    Returns:
        Dictionary of all metrics
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    metrics = {
        'pearson': pearson_correlation(pred, target),
        'spearman': spearman_correlation(pred, target),
        'kendall': kendall_tau(pred, target),
        'pairwise_accuracy': pairwise_accuracy(pred, target),
        'mse': float(np.mean((pred - target) ** 2)),
        'mae': float(np.mean(np.abs(pred - target))),
    }

    # Add NDCG and Precision at various k
    for k in k_values:
        if k <= len(pred):
            metrics[f'ndcg@{k}'] = ndcg_score(pred, target, k=k)
            metrics[f'precision@{k}'] = precision_at_k(pred, target, k=k)
            metrics[f'recall@{k}'] = recall_at_k(pred, target, k=k)

    return metrics
