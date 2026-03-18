"""Simple attractor-level statistics for chaotic dynamics evaluation."""

from __future__ import annotations

import numpy as np

from src.utils.metrics import covariance_distance, occupancy_distance, projected_histogram_distance, state_statistics


def compute_attractor_metrics(true_states: np.ndarray, predicted_states: np.ndarray) -> dict[str, float | list[float]]:
    true_stats = state_statistics(true_states)
    predicted_stats = state_statistics(predicted_states)
    return {
        "mean_distance": float(np.linalg.norm(true_stats["mean"] - predicted_stats["mean"])),
        "std_distance": float(np.linalg.norm(true_stats["std"] - predicted_stats["std"])),
        "covariance_distance": covariance_distance(true_states, predicted_states),
        "occupancy_distance": occupancy_distance(true_states, predicted_states),
        "projected_density_xy": projected_histogram_distance(true_states, predicted_states, axes=(0, 1)),
        "projected_density_xz": projected_histogram_distance(true_states, predicted_states, axes=(0, 2)),
        "true_mean": true_stats["mean"].tolist(),
        "predicted_mean": predicted_stats["mean"].tolist(),
        "true_std": true_stats["std"].tolist(),
        "predicted_std": predicted_stats["std"].tolist(),
    }
