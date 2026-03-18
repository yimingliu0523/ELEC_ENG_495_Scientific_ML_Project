"""Rollout helpers for discrete predictors and Neural ODEs."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.metrics import mae, rmse


@torch.no_grad()
def recursive_rollout_discrete_model(
    model: torch.nn.Module,
    initial_history: np.ndarray | torch.Tensor,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    history = torch.as_tensor(initial_history, dtype=torch.float32, device=device)
    squeeze_output = history.ndim == 2
    if squeeze_output:
        history = history.unsqueeze(0)

    window = history.clone()
    predictions = []
    model.eval()
    for _ in range(horizon):
        inputs = window.reshape(window.shape[0], -1)
        next_state = model(inputs)
        predictions.append(next_state)
        if window.shape[1] == 1:
            window = next_state.unsqueeze(1)
        else:
            window = torch.cat([window[:, 1:, :], next_state.unsqueeze(1)], dim=1)

    rollout = torch.stack(predictions, dim=1)
    return rollout[0] if squeeze_output else rollout


@torch.no_grad()
def rollout_neural_ode_model(
    model: torch.nn.Module,
    initial_state: np.ndarray | torch.Tensor,
    t_grid: np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    state = torch.as_tensor(initial_state, dtype=torch.float32, device=device)
    times = torch.as_tensor(t_grid, dtype=torch.float32, device=device)
    model.eval()
    return model.predict_rollout(state, times)


def compute_rollout_metrics(true_rollout: np.ndarray, predicted_rollout: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(true_rollout, predicted_rollout),
        "mae": mae(true_rollout, predicted_rollout),
    }


def compute_error_curve(true_rollout: np.ndarray, predicted_rollout: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((true_rollout - predicted_rollout) ** 2, axis=-1))
