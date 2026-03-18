"""Time-series and sensitivity plots for Lorenz trajectories."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.dynamics.dataset_generation import build_time_grid
from src.dynamics.lorenz import get_default_lorenz_params
from src.dynamics.simulation import generate_single_trajectory
from src.utils.io import ensure_dir
from src.utils.plotting import save_figure, set_plot_style


STATE_LABELS = ["x(t)", "y(t)", "z(t)"]


def plot_time_series(time_grid: np.ndarray, trajectory: np.ndarray, output_path: str, title: str = "Lorenz trajectory time series") -> None:
    set_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(time_grid, trajectory[:, idx], linewidth=1.2)
        ax.set_ylabel(STATE_LABELS[idx])
    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    save_figure(fig, output_path, tight=False)
    plt.close(fig)


def plot_time_series_comparison(
    time_grid: np.ndarray,
    true_trajectory: np.ndarray,
    predicted_trajectory: np.ndarray,
    output_path: str,
    title: str = "True vs predicted time series",
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(time_grid, true_trajectory[:, idx], label="True", linewidth=1.5)
        ax.plot(time_grid, predicted_trajectory[:, idx], label="Predicted", linewidth=1.2, linestyle="--")
        ax.set_ylabel(STATE_LABELS[idx])
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    save_figure(fig, output_path, tight=False)
    plt.close(fig)


def plot_sensitivity_to_initial_conditions(
    time_grid: np.ndarray,
    trajectory_a: np.ndarray,
    trajectory_b: np.ndarray,
    output_path: str,
    title: str = "Sensitivity to nearby initial conditions",
) -> None:
    set_plot_style()
    divergence = np.linalg.norm(trajectory_a - trajectory_b, axis=-1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(time_grid, trajectory_a[:, 0], label="Trajectory A")
    axes[0].plot(time_grid, trajectory_b[:, 0], label="Trajectory B", linestyle="--")
    axes[0].set_ylabel("x(t)")
    axes[0].legend(loc="upper right")
    axes[1].plot(time_grid, divergence, color="#9a031e")
    axes[1].set_ylabel("State distance")
    axes[1].set_xlabel("Time")
    fig.suptitle(title)
    save_figure(fig, output_path, tight=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reference Lorenz time-series figure.")
    parser.add_argument("--output", default="results/figures/figure_time_series.png")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--total-time", type=float, default=25.0)
    args = parser.parse_args()

    time_grid = build_time_grid(total_time=args.total_time, dt=args.dt)
    initial_state = np.array([1.0, 1.0, 1.0], dtype=float)
    trajectory = generate_single_trajectory(initial_state, time_grid, params=get_default_lorenz_params())
    ensure_dir("results/figures")
    plot_time_series(time_grid, trajectory, args.output)


if __name__ == "__main__":
    main()
