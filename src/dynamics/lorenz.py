"""Lorenz system right-hand side implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def get_default_lorenz_params() -> dict[str, float]:
    return {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}


def lorenz_rhs_numpy(
    state: np.ndarray,
    t: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """Compute the Lorenz derivative for a single state or a batch."""

    state_arr = np.asarray(state, dtype=float)
    x = state_arr[..., 0]
    y = state_arr[..., 1]
    z = state_arr[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.stack([dx, dy, dz], axis=-1)


def lorenz_rhs_batch_numpy(states: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    return lorenz_rhs_numpy(states, 0.0, sigma=sigma, rho=rho, beta=beta)


def lorenz_rhs_torch(
    t: Any,
    state: "torch.Tensor",
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for lorenz_rhs_torch.")

    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz], dim=-1)
