"""Dataset generation pipeline for Lorenz forecasting experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.normalization import compute_normalization_stats
from src.dynamics.lorenz import get_default_lorenz_params
from src.dynamics.simulation import generate_multiple_trajectories
from src.utils.io import ensure_dir, save_json, write_markdown_table


@dataclass
class DatasetBundle:
    time_grid: np.ndarray
    params: dict[str, float]
    split_trajectories: dict[str, np.ndarray]
    split_initial_conditions: dict[str, np.ndarray]
    normalization: dict[str, np.ndarray]
    metadata: dict


def build_time_grid(total_time: float, dt: float) -> np.ndarray:
    num_steps = int(round(total_time / dt)) + 1
    return np.linspace(0.0, total_time, num_steps, dtype=float)


def _split_counts(config: dict) -> dict[str, int]:
    return {
        "train": int(config["train_trajectories"]),
        "val": int(config["val_trajectories"]),
        "test": int(config["test_trajectories"]),
    }


def generate_dataset_bundle(config: dict) -> DatasetBundle:
    seed = int(config["seed"])
    params = config.get("system", get_default_lorenz_params())
    data_cfg = config["data"]
    counts = _split_counts(data_cfg)
    time_grid = build_time_grid(total_time=float(data_cfg["total_time"]), dt=float(data_cfg["dt"]))
    ranges = data_cfg.get("initial_condition_ranges")
    solver = data_cfg.get("solver", "rk4")
    noise_std = float(data_cfg.get("noise_std", 0.0))

    split_trajectories: dict[str, np.ndarray] = {}
    split_initial_conditions: dict[str, np.ndarray] = {}
    offset = 0
    for split_name, count in counts.items():
        payload = generate_multiple_trajectories(
            num_trajectories=count,
            time_grid=time_grid,
            params=params,
            initial_condition_ranges=ranges,
            seed=seed + offset,
            solver=solver,
            noise_std=noise_std,
        )
        split_trajectories[split_name] = payload["trajectories"]
        split_initial_conditions[split_name] = payload["initial_conditions"]
        offset += 17

    normalization = compute_normalization_stats(split_trajectories["train"])
    metadata = {
        "seed": seed,
        "noise_std": noise_std,
        "dt": float(data_cfg["dt"]),
        "total_time": float(data_cfg["total_time"]),
        "solver": solver,
        "counts": counts,
        "params": params,
        "initial_condition_ranges": ranges,
    }
    return DatasetBundle(
        time_grid=time_grid,
        params=params,
        split_trajectories=split_trajectories,
        split_initial_conditions=split_initial_conditions,
        normalization=normalization,
        metadata=metadata,
    )


def save_dataset_bundle(bundle: DatasetBundle, output_root: str | Path) -> dict[str, Path]:
    output_root = ensure_dir(output_root)
    raw_dir = ensure_dir(output_root / "raw")
    processed_dir = ensure_dir(output_root / "processed")
    table_dir = ensure_dir("results/tables")

    raw_path = raw_dir / "lorenz_raw_dataset.npz"
    processed_path = processed_dir / "lorenz_dataset.npz"
    metadata_path = processed_dir / "lorenz_dataset_metadata.json"

    np.savez_compressed(
        raw_path,
        time_grid=bundle.time_grid,
        params=np.array([bundle.params["sigma"], bundle.params["rho"], bundle.params["beta"]], dtype=float),
        train_trajectories=bundle.split_trajectories["train"],
        val_trajectories=bundle.split_trajectories["val"],
        test_trajectories=bundle.split_trajectories["test"],
        train_initial_conditions=bundle.split_initial_conditions["train"],
        val_initial_conditions=bundle.split_initial_conditions["val"],
        test_initial_conditions=bundle.split_initial_conditions["test"],
    )

    np.savez_compressed(
        processed_path,
        time_grid=bundle.time_grid,
        mean=bundle.normalization["mean"],
        std=bundle.normalization["std"],
        params=np.array([bundle.params["sigma"], bundle.params["rho"], bundle.params["beta"]], dtype=float),
        train_trajectories=bundle.split_trajectories["train"],
        val_trajectories=bundle.split_trajectories["val"],
        test_trajectories=bundle.split_trajectories["test"],
        train_initial_conditions=bundle.split_initial_conditions["train"],
        val_initial_conditions=bundle.split_initial_conditions["val"],
        test_initial_conditions=bundle.split_initial_conditions["test"],
    )

    save_json(bundle.metadata, metadata_path)

    summary_rows = []
    for split_name, trajectories in bundle.split_trajectories.items():
        summary_rows.append(
            {
                "split": split_name,
                "num_trajectories": trajectories.shape[0],
                "dt": bundle.metadata["dt"],
                "trajectory_length": trajectories.shape[1],
                "noise_level": bundle.metadata["noise_std"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(table_dir / "dataset_summary.csv", index=False)
    write_markdown_table(summary_df, table_dir / "dataset_summary.md")

    return {
        "raw_path": raw_path,
        "processed_path": processed_path,
        "metadata_path": metadata_path,
        "dataset_table_path": table_dir / "dataset_summary.csv",
    }
