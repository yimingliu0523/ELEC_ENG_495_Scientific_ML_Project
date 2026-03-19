"""Attractor visualizations for Lorenz trajectories and learned rollouts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import annotate_subplots, save_figure, set_plot_style


def plot_reference_attractor(
    trajectory: np.ndarray,
    output_path: str,
    title: str = "Lorenz attractor",
) -> None:
    set_plot_style()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_attractor_projections(
    true_states: np.ndarray,
    predicted_states: np.ndarray,
    output_path: str,
    title: str = "Attractor projection comparison",
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)
    projections = [((0, 1), "x", "y"), ((0, 2), "x", "z")]
    for ax, (axes_idx, xlabel, ylabel) in zip(axes, projections):
        ax.plot(true_states[:, axes_idx[0]], true_states[:, axes_idx[1]], label="True", linewidth=1.0)
        ax.plot(
            predicted_states[:, axes_idx[0]],
            predicted_states[:, axes_idx[1]],
            label="Predicted",
            linewidth=1.0,
            linestyle="--",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    annotate_subplots(axes, y=-0.25)
    axes[0].legend(loc="upper right")
    fig.suptitle(title)
    save_figure(fig, output_path, tight=False)
    plt.close(fig)
