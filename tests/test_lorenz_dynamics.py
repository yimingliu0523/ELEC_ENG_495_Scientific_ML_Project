from __future__ import annotations

import numpy as np

from src.dynamics.lorenz import get_default_lorenz_params, lorenz_rhs_numpy
from src.dynamics.solvers import integrate_rk4, simulate_trajectory


def test_lorenz_rhs_output_shape() -> None:
    params = get_default_lorenz_params()
    state = np.array([1.0, 2.0, 3.0])
    derivative = lorenz_rhs_numpy(state, 0.0, **params)
    assert derivative.shape == (3,)
    assert np.all(np.isfinite(derivative))


def test_rk4_integration_shape() -> None:
    params = get_default_lorenz_params()
    time_grid = np.linspace(0.0, 0.1, 11)
    initial_state = np.array([1.0, 1.0, 1.0])
    trajectory = integrate_rk4(lorenz_rhs_numpy, initial_state, time_grid, rhs_kwargs=params)
    assert trajectory.shape == (11, 3)
    assert np.all(np.isfinite(trajectory))


def test_simulate_trajectory_wrapper() -> None:
    time_grid = np.linspace(0.0, 0.1, 11)
    trajectory = simulate_trajectory(np.array([1.0, 1.0, 1.0]), time_grid, solver="rk4")
    assert trajectory.shape == (11, 3)
