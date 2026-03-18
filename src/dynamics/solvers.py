"""Numerical integration routines for Lorenz trajectories."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.dynamics.lorenz import lorenz_rhs_numpy

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover
    solve_ivp = None


def integrate_rk4(
    rhs: Callable[..., np.ndarray],
    initial_state: np.ndarray,
    time_grid: np.ndarray,
    rhs_kwargs: dict | None = None,
) -> np.ndarray:
    """Integrate an ODE with fixed-step RK4 over an arbitrary time grid."""

    rhs_kwargs = rhs_kwargs or {}
    states = np.zeros((len(time_grid), initial_state.shape[-1]), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)

    for idx in range(len(time_grid) - 1):
        t = float(time_grid[idx])
        dt = float(time_grid[idx + 1] - time_grid[idx])
        current = states[idx]
        k1 = rhs(current, t, **rhs_kwargs)
        k2 = rhs(current + 0.5 * dt * k1, t + 0.5 * dt, **rhs_kwargs)
        k3 = rhs(current + 0.5 * dt * k2, t + 0.5 * dt, **rhs_kwargs)
        k4 = rhs(current + dt * k3, t + dt, **rhs_kwargs)
        states[idx + 1] = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return states


def integrate_scipy(
    initial_state: np.ndarray,
    time_grid: np.ndarray,
    sigma: float,
    rho: float,
    beta: float,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    if solve_ivp is None:  # pragma: no cover
        raise ImportError("SciPy is required for integrate_scipy.")

    solution = solve_ivp(
        lambda t, y: lorenz_rhs_numpy(y, t, sigma=sigma, rho=rho, beta=beta),
        t_span=(float(time_grid[0]), float(time_grid[-1])),
        y0=np.asarray(initial_state, dtype=float),
        t_eval=np.asarray(time_grid, dtype=float),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"SciPy integration failed: {solution.message}")
    return solution.y.T


def simulate_trajectory(
    initial_state: np.ndarray,
    time_grid: np.ndarray,
    params: dict[str, float] | None = None,
    solver: str = "rk4",
    **solver_kwargs,
) -> np.ndarray:
    params = params or {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}

    if solver == "rk4":
        return integrate_rk4(
            lorenz_rhs_numpy,
            np.asarray(initial_state, dtype=float),
            np.asarray(time_grid),
            rhs_kwargs=params,
        )
    if solver == "scipy":
        return integrate_scipy(np.asarray(initial_state, dtype=float), np.asarray(time_grid), **params, **solver_kwargs)
    raise ValueError(f"Unknown solver '{solver}'.")
