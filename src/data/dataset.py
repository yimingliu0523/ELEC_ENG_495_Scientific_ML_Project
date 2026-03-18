"""PyTorch datasets for Lorenz forecasting experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.normalization import normalize_states
from src.utils.io import resolve_path


STATE_DIM = 3


def load_dataset_arrays(dataset_path: str | Path) -> dict[str, np.ndarray]:
    path = resolve_path(dataset_path)
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


class OneStepDataset(Dataset):
    """Map a current state or short history window to a future state."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        history_steps: int = 1,
        prediction_horizon: int = 1,
        normalize: bool = True,
        flatten_history: bool = True,
    ) -> None:
        arrays = load_dataset_arrays(dataset_path)
        trajectories = arrays[f"{split}_trajectories"].astype(np.float32)
        mean = arrays["mean"].astype(np.float32)
        std = arrays["std"].astype(np.float32)

        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        max_start = trajectories.shape[1] - history_steps - prediction_horizon + 1
        for trajectory in trajectories:
            for start_idx in range(max_start):
                history = trajectory[start_idx : start_idx + history_steps]
                target = trajectory[start_idx + history_steps + prediction_horizon - 1]
                inputs.append(history)
                targets.append(target)

        input_array = np.stack(inputs, axis=0)
        target_array = np.stack(targets, axis=0)

        if normalize:
            input_array = normalize_states(input_array, mean=mean, std=std)
            target_array = normalize_states(target_array, mean=mean, std=std)

        if flatten_history:
            input_array = input_array.reshape(input_array.shape[0], -1)

        self.inputs = torch.from_numpy(input_array.astype(np.float32))
        self.targets = torch.from_numpy(target_array.astype(np.float32))
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        self.history_steps = history_steps
        self.prediction_horizon = prediction_horizon

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


class ShortWindowDataset(Dataset):
    """Return a history window and the following rollout segment."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        history_steps: int,
        rollout_horizon: int,
        normalize: bool = True,
        flatten_history: bool = True,
    ) -> None:
        arrays = load_dataset_arrays(dataset_path)
        trajectories = arrays[f"{split}_trajectories"].astype(np.float32)
        mean = arrays["mean"].astype(np.float32)
        std = arrays["std"].astype(np.float32)

        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        max_start = trajectories.shape[1] - history_steps - rollout_horizon + 1
        for trajectory in trajectories:
            for start_idx in range(max_start):
                history = trajectory[start_idx : start_idx + history_steps]
                rollout = trajectory[start_idx + history_steps : start_idx + history_steps + rollout_horizon]
                inputs.append(history)
                targets.append(rollout)

        input_array = np.stack(inputs, axis=0)
        target_array = np.stack(targets, axis=0)

        if normalize:
            input_array = normalize_states(input_array, mean=mean, std=std)
            target_array = normalize_states(target_array, mean=mean, std=std)

        if flatten_history:
            input_array = input_array.reshape(input_array.shape[0], -1)

        self.inputs = torch.from_numpy(input_array.astype(np.float32))
        self.targets = torch.from_numpy(target_array.astype(np.float32))
        self.time_grid = torch.from_numpy(arrays["time_grid"].astype(np.float32))
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


class ContinuousTrajectoryDataset(Dataset):
    """Return short continuous-time segments for Neural ODE training."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        segment_length: int,
        stride: int = 1,
        normalize: bool = True,
    ) -> None:
        arrays = load_dataset_arrays(dataset_path)
        trajectories = arrays[f"{split}_trajectories"].astype(np.float32)
        mean = arrays["mean"].astype(np.float32)
        std = arrays["std"].astype(np.float32)
        time_grid = arrays["time_grid"].astype(np.float32)

        samples: list[tuple[int, int]] = []
        max_start = trajectories.shape[1] - segment_length + 1
        for traj_idx in range(trajectories.shape[0]):
            for start_idx in range(0, max_start, stride):
                samples.append((traj_idx, start_idx))

        self.trajectories = trajectories
        self.time_grid = time_grid
        self.segment_length = segment_length
        self.samples = samples
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        traj_idx, start_idx = self.samples[index]
        segment = self.trajectories[traj_idx, start_idx : start_idx + self.segment_length]
        t_segment = self.time_grid[start_idx : start_idx + self.segment_length]
        t_segment = t_segment - t_segment[0]
        if self.normalize:
            segment = normalize_states(segment, mean=self.mean, std=self.std)
        return torch.from_numpy(segment.astype(np.float32)), torch.from_numpy(t_segment.astype(np.float32))
