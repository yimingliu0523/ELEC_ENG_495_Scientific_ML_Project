"""Phase portrait visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import annotate_subplots, save_figure, set_plot_style


PROJECTIONS = [((0, 1), "x", "y"), ((0, 2), "x", "z"), ((1, 2), "y", "z")]


def plot_phase_portraits(
    true_states: np.ndarray,
    predicted_states: np.ndarray,
    output_path: str,
    title: str = "Phase portrait comparison",
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.30)
    for ax, (axes_idx, xlabel, ylabel) in zip(axes, PROJECTIONS):
        ax.plot(true_states[:, axes_idx[0]], true_states[:, axes_idx[1]], label="True", linewidth=1.2)
        ax.plot(
            predicted_states[:, axes_idx[0]],
            predicted_states[:, axes_idx[1]],
            label="Predicted",
            linewidth=1.0,
            linestyle="--",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    annotate_subplots(axes, y=-0.27)
    axes[0].legend(loc="upper right")
    fig.suptitle(title)
    save_figure(fig, output_path, tight=False)
    plt.close(fig)


def plot_solver_comparison(
    time_grid: np.ndarray,
    trajectories: dict[str, np.ndarray],
    output_path: str,
    state_index: int = 0,
    title: str = "Numerical solver comparison",
) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, trajectory in trajectories.items():
        ax.plot(time_grid[: trajectory.shape[0]], trajectory[:, state_index], label=label, linewidth=1.3)
    ax.set_xlabel("Time")
    ax.set_ylabel(["x", "y", "z"][state_index])
    ax.legend(loc="upper right")
    ax.set_title(title)
    save_figure(fig, output_path)
    plt.close(fig)
