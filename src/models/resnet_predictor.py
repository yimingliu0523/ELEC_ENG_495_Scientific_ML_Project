"""Residual discrete-time predictor for Lorenz dynamics."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from src.models.mlp_predictor import ACTIVATIONS


class ResNetPredictor(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        history_steps: int = 1,
        hidden_dims: Iterable[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        activation_cls = ACTIVATIONS[activation.lower()]
        input_dim = state_dim * history_steps
        dims = [input_dim, *hidden_dims, state_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_cls())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.delta_network = nn.Sequential(*layers)
        self.state_dim = state_dim
        self.history_steps = history_steps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        delta = self.delta_network(inputs)
        current_state = inputs[..., -self.state_dim :]
        return current_state + delta

    def predict_next(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)
