"""DataLoader construction helpers for Lorenz experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import ContinuousTrajectoryDataset, OneStepDataset, ShortWindowDataset, load_dataset_arrays


class LorenzDataModule:
    def __init__(self, dataset_path: str | Path, batch_size: int = 128, num_workers: int = 0) -> None:
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.arrays = load_dataset_arrays(dataset_path)

    @property
    def normalization(self) -> dict[str, np.ndarray]:
        return {"mean": self.arrays["mean"], "std": self.arrays["std"]}

    @property
    def time_grid(self) -> np.ndarray:
        return self.arrays["time_grid"]

    def one_step_loaders(
        self,
        history_steps: int = 1,
        prediction_horizon: int = 1,
        normalize: bool = True,
        flatten_history: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = OneStepDataset(self.dataset_path, "train", history_steps, prediction_horizon, normalize, flatten_history)
        val_ds = OneStepDataset(self.dataset_path, "val", history_steps, prediction_horizon, normalize, flatten_history)
        test_ds = OneStepDataset(self.dataset_path, "test", history_steps, prediction_horizon, normalize, flatten_history)
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
            DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
        )

    def short_window_loaders(
        self,
        history_steps: int,
        rollout_horizon: int,
        normalize: bool = True,
        flatten_history: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = ShortWindowDataset(self.dataset_path, "train", history_steps, rollout_horizon, normalize, flatten_history)
        val_ds = ShortWindowDataset(self.dataset_path, "val", history_steps, rollout_horizon, normalize, flatten_history)
        test_ds = ShortWindowDataset(self.dataset_path, "test", history_steps, rollout_horizon, normalize, flatten_history)
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
            DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
        )

    def continuous_loaders(
        self,
        segment_length: int,
        stride: int = 1,
        normalize: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = ContinuousTrajectoryDataset(self.dataset_path, "train", segment_length, stride=stride, normalize=normalize)
        val_ds = ContinuousTrajectoryDataset(self.dataset_path, "val", segment_length, stride=stride, normalize=normalize)
        test_ds = ContinuousTrajectoryDataset(self.dataset_path, "test", segment_length, stride=stride, normalize=normalize)
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
            DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
        )
