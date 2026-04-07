"""Unit tests for all encoder architectures."""

import pytest
import torch

from models.encoders import build_encoder


@pytest.fixture(params=["cnn", "dilated_cnn", "bilstm", "transformer"])
def architecture(request):
    return request.param


@pytest.fixture
def config(architecture):
    base = {"hidden_dim": 64, "dropout": 0.0}
    if architecture == "cnn":
        base.update({
            "sequence_length": 230,
            "n_filters": [32, 32, 32],
            "filter_sizes": [7, 5, 3],
            "pool_sizes": [3, 3, 3],
        })
    elif architecture == "dilated_cnn":
        base.update({
            "sequence_length": 230,
            "n_filters_dilated": 64,
            "n_dilated_layers": 3,
        })
    elif architecture == "bilstm":
        base.update({
            "sequence_length": 230,
            "lstm_hidden": 32,
            "n_lstm_layers": 1,
        })
    elif architecture == "transformer":
        base.update({
            "sequence_length": 230,
            "n_heads": 4,
            "n_transformer_layers": 2,
        })
    return base


def test_forward_pass(architecture, config):
    """Model produces correct output shape."""
    model = build_encoder(architecture, config)
    x = torch.randn(4, 230, 4)
    out = model(x)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_encode(architecture, config):
    """Encode produces representations of correct dimension."""
    model = build_encoder(architecture, config)
    x = torch.randn(4, 230, 4)
    reps = model.encode(x)
    assert reps.shape == (4, config["hidden_dim"]), f"Expected (4, {config['hidden_dim']}), got {reps.shape}"


def test_gradient_flow(architecture, config):
    """Gradients flow through the model."""
    model = build_encoder(architecture, config)
    x = torch.randn(4, 230, 4)
    out = model(x)
    loss = out.sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients flowing through model"
