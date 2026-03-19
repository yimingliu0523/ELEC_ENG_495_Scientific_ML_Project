"""Error-curve and summary plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plotting import annotate_subplots, save_figure, set_plot_style


STATE_NAMES = ["x", "y", "z"]


def plot_error_growth_curve(
    time_axis: np.ndarray,
    error_curves: dict[str, np.ndarray],
    output_path: str,
    title: str = "Rollout error growth",
) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, curve in error_curves.items():
        ax.plot(time_axis[: len(curve)], curve, label=label, linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("RMSE")
    ax.legend(loc="upper left")
    ax.set_title(title)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_one_step_scatter(
    true_states: np.ndarray,
    predicted_states: np.ndarray,
    output_path: str,
    title: str = "One-step prediction quality",
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.32)
    for idx, ax in enumerate(axes):
        ax.scatter(true_states[:, idx], predicted_states[:, idx], s=6, alpha=0.35)
        min_val = min(true_states[:, idx].min(), predicted_states[:, idx].min())
        max_val = max(true_states[:, idx].max(), predicted_states[:, idx].max())
        ax.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1.0)
        ax.set_xlabel(f"True {STATE_NAMES[idx]}")
        ax.set_ylabel(f"Predicted {STATE_NAMES[idx]}")
    annotate_subplots(axes, y=-0.28)
    fig.suptitle(title)
    save_figure(fig, output_path, tight=False)
    plt.close(fig)


def plot_noise_robustness(
    dataframe: pd.DataFrame,
    output_path: str,
    metric_column: str = "rollout_rmse",
    title: str = "Robustness to observation noise",
) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name, group in dataframe.groupby("model"):
        group = group.sort_values("noise_level")
        ax.plot(group["noise_level"], group[metric_column], marker="o", label=model_name)
    ax.set_xlabel("Noise standard deviation")
    ax.set_ylabel(metric_column.replace("_", " "))
    ax.legend(loc="upper left")
    ax.set_title(title)
    save_figure(fig, output_path)
    plt.close(fig)
