"""Robustness experiments for noise and step-size shifts."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_dataset_arrays
from src.data.normalization import denormalize_states, normalize_states
from src.dynamics.dataset_generation import build_time_grid
from src.dynamics.simulation import generate_multiple_trajectories
from src.evaluation.evaluate_model import load_trained_model
from src.evaluation.rollout import compute_rollout_metrics, recursive_rollout_discrete_model, rollout_neural_ode_model
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_markdown_table
from src.visualization.plot_error_curves import plot_noise_robustness


def run_robustness_experiments(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    device = torch.device(config.get("device", "cpu"))
    arrays = load_dataset_arrays(config["dataset_path"])
    mean = arrays["mean"]
    std = arrays["std"]

    noise_levels = config.get("noise_levels", [0.0, 0.01, 0.05])
    dt_values = config.get("dt_values", [0.01, 0.02])
    total_time = float(config.get("total_time", 25.0))
    num_trajectories = int(config.get("num_trajectories", 12))
    rollout_horizon = int(config.get("rollout_horizon", 100))
    seed = int(config.get("seed", 7))
    params = config.get("system", {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0})
    ranges = config.get("initial_condition_ranges")

    rows = []
    for model_name in config["models"]:
        model, _, train_config, _ = load_trained_model(model_name, config, device)
        history_steps = int(train_config.get("model", {}).get("history_steps", 1))
        for dt in dt_values:
            time_grid = build_time_grid(total_time=total_time, dt=float(dt))
            clean_payload = generate_multiple_trajectories(
                num_trajectories=num_trajectories,
                time_grid=time_grid,
                params=params,
                initial_condition_ranges=ranges,
                seed=seed,
                noise_std=0.0,
            )
            clean_trajectories = clean_payload["clean_trajectories"]

            for noise_level in noise_levels:
                observed_payload = generate_multiple_trajectories(
                    num_trajectories=num_trajectories,
                    time_grid=time_grid,
                    params=params,
                    initial_condition_ranges=ranges,
                    seed=seed,
                    noise_std=float(noise_level),
                )
                observed = observed_payload["trajectories"]
                max_available = observed.shape[1] - 1 if model_name == "neural_ode" else observed.shape[1] - history_steps
                horizon = min(rollout_horizon, max_available)

                true_rollouts = []
                pred_rollouts = []
                if model_name == "neural_ode":
                    eval_time = time_grid[: horizon + 1] - time_grid[0]
                    for clean, noisy in zip(clean_trajectories, observed):
                        initial_state = normalize_states(noisy[0], mean, std)
                        pred_norm = rollout_neural_ode_model(model, initial_state, eval_time, device=device).cpu().numpy()
                        pred = denormalize_states(pred_norm, mean, std)[1:]
                        true_rollouts.append(clean[1 : horizon + 1])
                        pred_rollouts.append(pred)
                else:
                    for clean, noisy in zip(clean_trajectories, observed):
                        history = normalize_states(noisy[:history_steps], mean, std)
                        pred_norm = recursive_rollout_discrete_model(model, history, horizon, device=device).cpu().numpy()
                        pred = denormalize_states(pred_norm, mean, std)
                        true_rollouts.append(clean[history_steps : history_steps + horizon])
                        pred_rollouts.append(pred)

                true_array = np.concatenate(true_rollouts, axis=0)
                pred_array = np.concatenate(pred_rollouts, axis=0)
                metrics = compute_rollout_metrics(true_array, pred_array)
                rows.append(
                    {
                        "model": model_name,
                        "noise_level": float(noise_level),
                        "dt": float(dt),
                        "rollout_rmse": metrics["rmse"],
                        "rollout_mae": metrics["mae"],
                    }
                )

    dataframe = pd.DataFrame(rows)
    table_dir = ensure_dir(config.get("table_dir", "results/tables"))
    figure_dir = ensure_dir(config.get("figure_dir", "results/figures"))
    report_asset_dir = ensure_dir(config.get("report_asset_dir", "results/report_assets"))

    csv_path = table_dir / "robustness_results.csv"
    md_path = table_dir / "robustness_results.md"
    dataframe.to_csv(csv_path, index=False)
    write_markdown_table(dataframe, md_path)

    default_dt = min(dt_values)
    plot_noise_robustness(
        dataframe[dataframe["dt"] == default_dt],
        str(figure_dir / "figure_noise_robustness.png"),
        metric_column="rollout_rmse",
        title=f"Noise robustness at dt={default_dt}",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = dataframe[dataframe["noise_level"] == min(noise_levels)]
    for current_model, group in grouped.groupby("model"):
        group = group.sort_values("dt")
        ax.plot(group["dt"], group["rollout_rmse"], marker="o", label=current_model)
    ax.set_xlabel("Evaluation dt")
    ax.set_ylabel("Rollout RMSE")
    ax.set_title("Cross-step-size robustness")
    ax.legend(loc="upper left")
    step_size_path = figure_dir / "figure_step_size_robustness.png"
    fig.savefig(step_size_path, bbox_inches="tight", dpi=250)
    plt.close(fig)

    for path in [csv_path, md_path, figure_dir / "figure_noise_robustness.png", step_size_path]:
        target = report_asset_dir / path.name
        target.write_bytes(path.read_bytes())

    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness experiments for trained Lorenz models.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_robustness_experiments(args.config)


if __name__ == "__main__":
    main()
