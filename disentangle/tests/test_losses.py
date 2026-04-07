"""Unit tests for all loss functions."""

import pytest
import torch


@pytest.fixture
def ranking_config():
    return {
        "ranking_margin": 1.0,
        "noise_threshold": 0.1,
        "ranking_temperature": 0.5,
        "n_pairs_per_sample": 8,
    }


@pytest.fixture
def contrastive_config():
    return {"contrastive_temperature": 0.07}


@pytest.fixture
def consensus_config():
    return {"consensus_margin": 1.0, "margin": 1.0, "temperature": 0.1}


def test_ranking_loss_correct_ordering(ranking_config):
    """Loss should be near 0 for perfect ranking with large margin."""
    from training.losses.ranking import AdaptiveMarginRankingLoss

    loss_fn = AdaptiveMarginRankingLoss(ranking_config)
    activities = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    loss = loss_fn(predictions, activities)
    assert loss.item() < 0.1, f"Loss should be near 0 for perfect ranking, got {loss.item()}"


def test_ranking_loss_reversed_ordering(ranking_config):
    """Loss should be high for reversed ranking."""
    from training.losses.ranking import AdaptiveMarginRankingLoss

    loss_fn = AdaptiveMarginRankingLoss(ranking_config)
    activities = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0])
    loss = loss_fn(predictions, activities)
    assert loss.item() > 0.5, f"Loss should be high for reversed ranking, got {loss.item()}"


def test_contrastive_loss_positive_closer(contrastive_config):
    """Loss should be low when positive pairs are much closer than negatives."""
    from training.losses.contrastive import NoiseContrastiveLoss

    loss_fn = NoiseContrastiveLoss(contrastive_config)
    anchor = torch.randn(8, 64)
    positive = anchor + 0.01 * torch.randn(8, 64)
    negative = torch.randn(8, 4, 64)
    loss = loss_fn(anchor, positive, negative)
    assert loss.item() < 2.0, f"Loss should be low for similar positives, got {loss.item()}"


def test_contrastive_loss_gradient(contrastive_config):
    """Gradients flow through contrastive loss."""
    from training.losses.contrastive import NoiseContrastiveLoss

    loss_fn = NoiseContrastiveLoss(contrastive_config)
    anchor = torch.randn(4, 32, requires_grad=True)
    positive = torch.randn(4, 32, requires_grad=True)
    negative = torch.randn(4, 4, 32)
    loss = loss_fn(anchor, positive, negative)
    loss.backward()
    assert anchor.grad is not None


def test_consensus_loss_basic(consensus_config):
    """Consensus loss runs without errors."""
    from training.losses.consensus import ConsensusLoss

    loss_fn = ConsensusLoss(consensus_config)
    predictions = torch.randn(16)
    targets = torch.rand(16)
    loss = loss_fn(predictions, targets)
    assert loss.item() >= 0


def test_mse_loss():
    """MSE loss baseline works correctly."""
    from training.losses.mse import MSELoss

    loss_fn = MSELoss()
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    loss = loss_fn(predictions, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
