"""Numerical metrics shared across training and evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def state_statistics(states: np.ndarray) -> dict[str, np.ndarray]:
    flat_states = states.reshape(-1, states.shape[-1])
    return {
        "mean": flat_states.mean(axis=0),
        "std": flat_states.std(axis=0),
        "covariance": np.cov(flat_states.T),
    }


def covariance_distance(true_states: np.ndarray, pred_states: np.ndarray) -> float:
    true_cov = state_statistics(true_states)["covariance"]
    pred_cov = state_statistics(pred_states)["covariance"]
    return float(np.linalg.norm(true_cov - pred_cov))


def projected_histogram_distance(
    true_states: np.ndarray,
    pred_states: np.ndarray,
    bins: int = 50,
    axes: tuple[int, int] = (0, 2),
) -> float:
    true_hist, x_edges, y_edges = np.histogram2d(
        true_states[..., axes[0]].ravel(),
        true_states[..., axes[1]].ravel(),
        bins=bins,
        density=True,
    )
    pred_hist, _, _ = np.histogram2d(
        pred_states[..., axes[0]].ravel(),
        pred_states[..., axes[1]].ravel(),
        bins=[x_edges, y_edges],
        density=True,
    )
    return float(np.mean(np.abs(true_hist - pred_hist)))


def occupancy_distance(true_states: np.ndarray, pred_states: np.ndarray, bins: int = 16) -> float:
    flat_true = true_states.reshape(-1, true_states.shape[-1])
    flat_pred = pred_states.reshape(-1, pred_states.shape[-1])
    mins = np.minimum(flat_true.min(axis=0), flat_pred.min(axis=0))
    maxs = np.maximum(flat_true.max(axis=0), flat_pred.max(axis=0))
    hist_true, edges = np.histogramdd(flat_true, bins=bins, range=list(zip(mins, maxs)), density=True)
    hist_pred, _ = np.histogramdd(flat_pred, bins=edges, density=True)
    return float(np.mean(np.abs(hist_true - hist_pred)))


def error_growth_curve(true_rollout: np.ndarray, pred_rollout: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((true_rollout - pred_rollout) ** 2, axis=-1))


def summarise_rollout_metrics(horizons: Iterable[int], metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for horizon in horizons:
        values = metrics[horizon]
        summary[f"rollout_rmse_{horizon}"] = values["rmse"]
        summary[f"rollout_mae_{horizon}"] = values["mae"]
    return summary
