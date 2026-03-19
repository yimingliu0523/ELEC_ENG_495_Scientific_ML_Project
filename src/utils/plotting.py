"""Matplotlib styling utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import resolve_path


COLOR_CYCLE = ["#0f4c5c", "#e36414", "#6a994e", "#5f0f40", "#9a031e", "#437f97"]


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),
            "figure.figsize": (8, 5),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "savefig.dpi": 250,
            "figure.dpi": 120,
            "font.size": 11,
        }
    )


def annotate_subplots(
    axes: plt.Axes | np.ndarray | list[plt.Axes],
    labels: list[str] | tuple[str, ...] | None = None,
    y: float = -0.18,
) -> None:
    axis_array = np.atleast_1d(axes).ravel()
    if labels is None:
        labels = [f"({chr(ord('a') + idx)})" for idx in range(len(axis_array))]

    for ax, label in zip(axis_array, labels):
        ax.text(
            0.5,
            y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )


def save_figure(fig: plt.Figure, path_like: str | Path, tight: bool = True) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path
