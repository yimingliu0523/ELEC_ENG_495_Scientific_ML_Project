"""Neural ODE model with optional torchdiffeq support and RK4 fallback."""

from __future__ import annotations

import torch
from torch import nn

from src.models.vector_field_net import VectorFieldNet

try:
    from torchdiffeq import odeint
except ImportError:  # pragma: no cover
    odeint = None


class NeuralODEModel(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        hidden_dims: tuple[int, ...] = (128, 128),
        activation: str = "tanh",
        solver: str = "rk4",
    ) -> None:
        super().__init__()
        self.vector_field = VectorFieldNet(state_dim=state_dim, hidden_dims=hidden_dims, activation=activation)
        self.state_dim = state_dim
        self.solver = solver

    def rhs(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.vector_field(state)

    def _integrate_rk4(self, initial_state: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        squeeze_output = initial_state.ndim == 1
        if squeeze_output:
            initial_state = initial_state.unsqueeze(0)
        states = [initial_state]
        current = initial_state
        for idx in range(len(t_grid) - 1):
            t = t_grid[idx]
            dt = t_grid[idx + 1] - t_grid[idx]
            k1 = self.rhs(t, current)
            k2 = self.rhs(t + 0.5 * dt, current + 0.5 * dt * k1)
            k3 = self.rhs(t + 0.5 * dt, current + 0.5 * dt * k2)
            k4 = self.rhs(t + dt, current + dt * k3)
            current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            states.append(current)
        rollout = torch.stack(states, dim=1)
        return rollout[0] if squeeze_output else rollout

    def forward(self, initial_state: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        if odeint is not None:
            squeeze_output = initial_state.ndim == 1
            state = initial_state.unsqueeze(0) if squeeze_output else initial_state
            solution = odeint(self.rhs, state, t_grid, method=self.solver)
            solution = solution.permute(1, 0, 2)
            return solution[0] if squeeze_output else solution
        return self._integrate_rk4(initial_state, t_grid)

    def predict_rollout(self, initial_state: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        return self.forward(initial_state, t_grid)
