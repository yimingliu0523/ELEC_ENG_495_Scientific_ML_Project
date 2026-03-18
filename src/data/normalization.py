"""Normalization utilities for trajectory datasets."""

from __future__ import annotations

import numpy as np


def compute_normalization_stats(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    flat = trajectories.reshape(-1, trajectories.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std}


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (states - mean) / std


def denormalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return states * std + mean
