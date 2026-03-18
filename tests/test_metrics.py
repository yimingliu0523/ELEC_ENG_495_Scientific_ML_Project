from __future__ import annotations

import numpy as np

from src.evaluation.attractor_metrics import compute_attractor_metrics
from src.utils.metrics import covariance_distance, mae, mse, occupancy_distance, rmse


def test_basic_metrics_finite() -> None:
    true_states = np.zeros((20, 3), dtype=float)
    predicted_states = np.ones((20, 3), dtype=float)
    assert np.isfinite(mse(true_states, predicted_states))
    assert np.isfinite(mae(true_states, predicted_states))
    assert np.isfinite(rmse(true_states, predicted_states))
    assert np.isfinite(covariance_distance(true_states, predicted_states))
    assert np.isfinite(occupancy_distance(true_states, predicted_states))


def test_attractor_metrics_finite() -> None:
    true_states = np.random.randn(100, 3)
    predicted_states = true_states + 0.1 * np.random.randn(100, 3)
    metrics = compute_attractor_metrics(true_states, predicted_states)
    assert np.isfinite(metrics["mean_distance"])
    assert np.isfinite(metrics["occupancy_distance"])
