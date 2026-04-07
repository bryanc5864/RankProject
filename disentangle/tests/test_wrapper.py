"""Unit tests for the DisentangleWrapper."""

import torch

from models.encoders import build_encoder
from models.wrapper import DisentangleWrapper


def test_wrapper_forward():
    """Wrapper produces correct output shape."""
    config = {
        "hidden_dim": 64,
        "sequence_length": 230,
        "n_filters": [32, 32],
        "filter_sizes": [7, 5],
        "pool_sizes": [3, 3],
        "dropout": 0.0,
        "projection_dim": 32,
    }
    encoder = build_encoder("cnn", config)
    model = DisentangleWrapper(encoder, n_experiments=3, config=config)

    x = torch.randn(4, 230, 4)
    out = model(x, experiment_id=0)
    assert out.shape == (4,)


def test_wrapper_denoised():
    """Denoised prediction averages across experiment normalizations."""
    config = {
        "hidden_dim": 64,
        "sequence_length": 230,
        "n_filters": [32, 32],
        "filter_sizes": [7, 5],
        "pool_sizes": [3, 3],
        "dropout": 0.0,
        "projection_dim": 32,
    }
    encoder = build_encoder("cnn", config)
    model = DisentangleWrapper(encoder, n_experiments=3, config=config)

    x = torch.randn(4, 230, 4)
    out = model.predict_denoised(x)
    assert out.shape == (4,)


def test_wrapper_project():
    """Projection head produces correct dimension."""
    config = {
        "hidden_dim": 64,
        "sequence_length": 230,
        "n_filters": [32, 32],
        "filter_sizes": [7, 5],
        "pool_sizes": [3, 3],
        "dropout": 0.0,
        "projection_dim": 32,
    }
    encoder = build_encoder("cnn", config)
    model = DisentangleWrapper(encoder, n_experiments=3, config=config)

    x = torch.randn(4, 230, 4)
    proj = model.project(x, experiment_id=1)
    assert proj.shape == (4, 32)


def test_wrapper_gradient_flow():
    """Gradients flow through the full wrapper."""
    config = {
        "hidden_dim": 64,
        "sequence_length": 230,
        "n_filters": [32, 32],
        "filter_sizes": [7, 5],
        "pool_sizes": [3, 3],
        "dropout": 0.0,
        "projection_dim": 32,
    }
    encoder = build_encoder("cnn", config)
    model = DisentangleWrapper(encoder, n_experiments=3, config=config)

    x = torch.randn(4, 230, 4)
    out = model(x, experiment_id=0)
    loss = out.sum()
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad
