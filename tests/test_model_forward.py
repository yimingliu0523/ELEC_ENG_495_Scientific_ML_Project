from __future__ import annotations

import torch

from src.models import build_model


def test_mlp_forward_shape() -> None:
    model = build_model("mlp", {"state_dim": 3, "history_steps": 1, "hidden_dims": [32, 32], "activation": "relu"})
    inputs = torch.randn(5, 3)
    outputs = model(inputs)
    assert outputs.shape == (5, 3)


def test_resnet_forward_shape() -> None:
    model = build_model("resnet", {"state_dim": 3, "history_steps": 2, "hidden_dims": [32, 32], "activation": "relu"})
    inputs = torch.randn(5, 6)
    outputs = model(inputs)
    assert outputs.shape == (5, 3)


def test_neural_ode_forward_shape() -> None:
    model = build_model("neural_ode", {"state_dim": 3, "hidden_dims": [32, 32], "activation": "tanh", "solver": "rk4"})
    initial_state = torch.randn(4, 3)
    t_grid = torch.linspace(0.0, 0.5, 6)
    outputs = model(initial_state, t_grid)
    assert outputs.shape == (4, 6, 3)
