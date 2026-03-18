"""Loss functions for Lorenz forecasting experiments."""

from __future__ import annotations

import torch
from torch import nn


_MSE = nn.MSELoss()


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _MSE(prediction, target)


def weighted_rollout_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    step_weight: float = 0.98,
) -> torch.Tensor:
    horizon = prediction.shape[1]
    weights = torch.tensor(
        [step_weight**idx for idx in range(horizon)],
        device=prediction.device,
        dtype=prediction.dtype,
    )
    errors = torch.mean((prediction - target) ** 2, dim=-1)
    weighted = errors * weights.unsqueeze(0)
    return weighted.mean()


def derivative_matching_loss(
    predicted_derivative: torch.Tensor,
    target_derivative: torch.Tensor,
) -> torch.Tensor:
    return _MSE(predicted_derivative, target_derivative)


def l2_regularization(model: torch.nn.Module, weight_decay: float = 1e-6) -> torch.Tensor:
    penalty = torch.zeros(1, device=next(model.parameters()).device)
    for parameter in model.parameters():
        penalty = penalty + torch.sum(parameter ** 2)
    return weight_decay * penalty
