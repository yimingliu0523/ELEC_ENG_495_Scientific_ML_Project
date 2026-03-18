"""High-level Lorenz trajectory simulation utilities."""

from __future__ import annotations

import numpy as np

from src.dynamics.lorenz import get_default_lorenz_params
from src.dynamics.solvers import simulate_trajectory


DEFAULT_INITIAL_CONDITION_RANGES = {
    "x": (-15.0, 15.0),
    "y": (-20.0, 20.0),
    "z": (5.0, 35.0),
}


def sample_initial_conditions(
    num_trajectories: int,
    ranges: dict[str, tuple[float, float]] | None = None,
    seed: int = 0,
) -> np.ndarray:
    ranges = ranges or DEFAULT_INITIAL_CONDITION_RANGES
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(*ranges["x"], size=num_trajectories)
    y0 = rng.uniform(*ranges["y"], size=num_trajectories)
    z0 = rng.uniform(*ranges["z"], size=num_trajectories)
    return np.stack([x0, y0, z0], axis=-1)


def add_gaussian_noise(trajectories: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    if noise_std <= 0.0:
        return trajectories.copy()
    rng = np.random.default_rng(seed)
    return trajectories + rng.normal(loc=0.0, scale=noise_std, size=trajectories.shape)


def generate_single_trajectory(
    initial_state: np.ndarray,
    time_grid: np.ndarray,
    params: dict[str, float] | None = None,
    solver: str = "rk4",
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    params = params or get_default_lorenz_params()
    trajectory = simulate_trajectory(initial_state, time_grid, params=params, solver=solver)
    return add_gaussian_noise(trajectory[None, ...], noise_std, seed=seed)[0]


def generate_multiple_trajectories(
    num_trajectories: int,
    time_grid: np.ndarray,
    params: dict[str, float] | None = None,
    initial_condition_ranges: dict[str, tuple[float, float]] | None = None,
    seed: int = 0,
    solver: str = "rk4",
    noise_std: float = 0.0,
) -> dict[str, np.ndarray]:
    params = params or get_default_lorenz_params()
    initial_conditions = sample_initial_conditions(num_trajectories, initial_condition_ranges, seed=seed)
    clean_trajectories = np.stack(
        [simulate_trajectory(x0, time_grid, params=params, solver=solver) for x0 in initial_conditions],
        axis=0,
    )
    noisy_trajectories = add_gaussian_noise(clean_trajectories, noise_std=noise_std, seed=seed + 1)
    return {
        "initial_conditions": initial_conditions,
        "clean_trajectories": clean_trajectories,
        "trajectories": noisy_trajectories,
    }
