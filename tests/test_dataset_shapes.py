from __future__ import annotations

import numpy as np

from src.data.dataset import ContinuousTrajectoryDataset, OneStepDataset, ShortWindowDataset


def _write_tiny_dataset(path) -> None:
    trajectories = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    np.savez_compressed(
        path,
        time_grid=np.linspace(0.0, 0.7, 8, dtype=np.float32),
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
        train_trajectories=trajectories,
        val_trajectories=trajectories,
        test_trajectories=trajectories,
        train_initial_conditions=trajectories[:, 0],
        val_initial_conditions=trajectories[:, 0],
        test_initial_conditions=trajectories[:, 0],
        params=np.array([10.0, 28.0, 8.0 / 3.0], dtype=np.float32),
    )


def test_one_step_dataset_shapes(tmp_path) -> None:
    dataset_path = tmp_path / "tiny_dataset.npz"
    _write_tiny_dataset(dataset_path)
    dataset = OneStepDataset(dataset_path, "train", history_steps=2, prediction_horizon=1, normalize=False)
    inputs, targets = dataset[0]
    assert inputs.shape == (6,)
    assert targets.shape == (3,)
    assert len(dataset) == 12


def test_short_window_dataset_shapes(tmp_path) -> None:
    dataset_path = tmp_path / "tiny_dataset.npz"
    _write_tiny_dataset(dataset_path)
    dataset = ShortWindowDataset(dataset_path, "val", history_steps=2, rollout_horizon=3, normalize=False)
    inputs, targets = dataset[0]
    assert inputs.shape == (6,)
    assert targets.shape == (3, 3)


def test_continuous_dataset_shapes(tmp_path) -> None:
    dataset_path = tmp_path / "tiny_dataset.npz"
    _write_tiny_dataset(dataset_path)
    dataset = ContinuousTrajectoryDataset(dataset_path, "test", segment_length=4, stride=2, normalize=False)
    segment, t_grid = dataset[0]
    assert segment.shape == (4, 3)
    assert t_grid.shape == (4,)
