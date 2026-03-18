from __future__ import annotations

import numpy as np
import torch

from src.evaluation.rollout import compute_rollout_metrics, recursive_rollout_discrete_model, rollout_neural_ode_model
from src.models import build_model


def test_discrete_rollout_runs() -> None:
    model = build_model("mlp", {"state_dim": 3, "history_steps": 1, "hidden_dims": [16], "activation": "relu"})
    initial_history = np.zeros((1, 3), dtype=np.float32)
    rollout = recursive_rollout_discrete_model(model, initial_history, horizon=5, device=torch.device("cpu"))
    assert rollout.shape == (5, 3)


def test_neural_ode_rollout_runs() -> None:
    model = build_model("neural_ode", {"state_dim": 3, "hidden_dims": [16], "activation": "tanh", "solver": "rk4"})
    initial_state = np.zeros(3, dtype=np.float32)
    t_grid = np.linspace(0.0, 0.5, 6, dtype=np.float32)
    rollout = rollout_neural_ode_model(model, initial_state, t_grid, device=torch.device("cpu"))
    assert rollout.shape == (6, 3)


def test_rollout_metrics_are_finite() -> None:
    true_rollout = np.zeros((10, 3), dtype=float)
    predicted_rollout = np.ones((10, 3), dtype=float)
    metrics = compute_rollout_metrics(true_rollout, predicted_rollout)
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mae"])
