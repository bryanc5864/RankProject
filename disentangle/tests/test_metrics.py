"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from evaluation.metrics import compute_all_metrics, compute_ndcg, compute_direction_accuracy


def test_compute_all_metrics_perfect():
    """Perfect predictions should give correlation 1.0."""
    preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_all_metrics(preds, targets)
    assert metrics["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["spearman"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)


def test_ndcg_perfect():
    """Perfect ranking should give NDCG = 1.0."""
    true = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ndcg = compute_ndcg(true, pred, k=5)
    assert ndcg == pytest.approx(1.0, abs=1e-6)


def test_ndcg_reversed():
    """Reversed ranking should give low NDCG."""
    true = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ndcg = compute_ndcg(true, pred, k=5)
    assert ndcg < 0.8


def test_direction_accuracy_perfect():
    """Perfect ranking should give direction accuracy near 1.0."""
    preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    acc = compute_direction_accuracy(preds, targets, threshold=0.5)
    assert acc > 0.99
