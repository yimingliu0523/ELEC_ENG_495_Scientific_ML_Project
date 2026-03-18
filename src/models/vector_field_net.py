"""Neural network parameterization of a continuous-time vector field."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from src.models.mlp_predictor import ACTIVATIONS


class VectorFieldNet(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        hidden_dims: Iterable[int] = (128, 128),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        activation_cls = ACTIVATIONS[activation.lower()]
        dims = [state_dim, *hidden_dims, state_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_cls())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)
        self.state_dim = state_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
