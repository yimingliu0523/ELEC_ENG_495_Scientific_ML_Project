"""Evaluate a trained Lorenz model on predictive and attractor-level metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ContinuousTrajectoryDataset, OneStepDataset, load_dataset_arrays
from src.data.normalization import denormalize_states, normalize_states
from src.evaluation.attractor_metrics import compute_attractor_metrics
from src.evaluation.rollout import compute_error_curve, compute_rollout_metrics, recursive_rollout_discrete_model, rollout_neural_ode_model
from src.models import build_model
from src.training.trainer import count_parameters
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.utils.metrics import mae, mse, summarise_rollout_metrics
from src.visualization.plot_attractors import plot_attractor_projections
from src.visualization.plot_error_curves import plot_error_growth_curve, plot_one_step_scatter
from src.visualization.plot_phase_portraits import plot_phase_portraits
from src.visualization.plot_time_series import plot_time_series_comparison


def _model_record(eval_config: dict, model_name: str) -> dict:
    record = eval_config["models"].get(model_name)
    if record is None:
        raise ValueError(f"Model '{model_name}' is missing from the evaluation config.")
    return record


def load_trained_model(model_name: str, eval_config: dict, device: torch.device):
    record = _model_record(eval_config, model_name)
    checkpoint_path = Path(record["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = load_config(record["train_config"])
    model_section = train_config.get("model", {})
    checkpoint_model_name = checkpoint.get("model_name", model_name)
    model_kwargs = checkpoint.get("model_kwargs") or {key: value for key, value in model_section.items() if key != "name"}
    model = build_model(checkpoint_model_name, model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, train_config, record


@torch.no_grad()
def one_step_predictions_discrete(
    model: torch.nn.Module,
    dataset_path: str,
    history_steps: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = OneStepDataset(dataset_path, "test", history_steps=history_steps, normalize=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    targets = []
    for inputs, target in loader:
        pred = model(inputs.to(device)).cpu().numpy()
        predictions.append(pred)
        targets.append(target.numpy())
    pred_norm = np.concatenate(predictions, axis=0)
    target_norm = np.concatenate(targets, axis=0)
    return denormalize_states(pred_norm, mean, std), denormalize_states(target_norm, mean, std)


@torch.no_grad()
def one_step_predictions_neural_ode(
    model: torch.nn.Module,
    dataset_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ContinuousTrajectoryDataset(dataset_path, "test", segment_length=2, stride=1, normalize=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    targets = []
    for segments, t_grid in loader:
        segments = segments.to(device)
        t_grid = t_grid.to(device)
        pred = model(segments[:, 0], t_grid[0])[:, 1].cpu().numpy()
        predictions.append(pred)
        targets.append(segments[:, 1].cpu().numpy())
    pred_norm = np.concatenate(predictions, axis=0)
    target_norm = np.concatenate(targets, axis=0)
    return denormalize_states(pred_norm, mean, std), denormalize_states(target_norm, mean, std)


@torch.no_grad()
def generate_rollout_arrays(
    model_name: str,
    model: torch.nn.Module,
    trajectories: np.ndarray,
    time_grid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    max_horizon: int,
    history_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_rollouts = []
    pred_rollouts = []

    if model_name == "neural_ode":
        eval_time = time_grid[: max_horizon + 1] - time_grid[0]
        for trajectory in trajectories:
            initial_state = normalize_states(trajectory[0], mean, std)
            pred_norm = rollout_neural_ode_model(model, initial_state, eval_time, device=device).cpu().numpy()
            pred = denormalize_states(pred_norm, mean, std)[1:]
            true = trajectory[1 : max_horizon + 1]
            true_rollouts.append(true)
            pred_rollouts.append(pred)
        return np.stack(true_rollouts), np.stack(pred_rollouts), time_grid[1 : max_horizon + 1]

    for trajectory in trajectories:
        history = normalize_states(trajectory[:history_steps], mean, std)
        pred_norm = recursive_rollout_discrete_model(model, history, max_horizon, device=device).cpu().numpy()
        pred = denormalize_states(pred_norm, mean, std)
        true = trajectory[history_steps : history_steps + max_horizon]
        true_rollouts.append(true)
        pred_rollouts.append(pred)
    return np.stack(true_rollouts), np.stack(pred_rollouts), time_grid[history_steps : history_steps + max_horizon]


def evaluate_model(model_name: str, config_path: str) -> dict:
    eval_config = load_config(config_path)
    device = torch.device(eval_config.get("device", "cpu"))
    model, checkpoint, train_config, record = load_trained_model(model_name, eval_config, device)

    dataset_path = eval_config.get("dataset_path") or train_config["data"]["dataset_path"]
    arrays = load_dataset_arrays(dataset_path)
    mean = arrays["mean"]
    std = arrays["std"]
    test_trajectories = arrays["test_trajectories"]
    time_grid = arrays["time_grid"]
    history_steps = int(train_config.get("model", {}).get("history_steps", 1))
    batch_size = int(eval_config.get("batch_size", 512))

    figure_dir = ensure_dir(eval_config.get("figure_dir", "results/figures"))
    log_dir = ensure_dir(eval_config.get("log_dir", "results/logs"))
    report_asset_dir = ensure_dir(eval_config.get("report_asset_dir", "results/report_assets"))

    if model_name == "neural_ode":
        one_step_pred, one_step_true = one_step_predictions_neural_ode(model, dataset_path, mean, std, device, batch_size)
    else:
        one_step_pred, one_step_true = one_step_predictions_discrete(model, dataset_path, history_steps, mean, std, device, batch_size)

    one_step_metrics = {
        "one_step_mse": mse(one_step_true, one_step_pred),
        "one_step_mae": mae(one_step_true, one_step_pred),
    }

    requested_horizons = [int(h) for h in eval_config.get("horizons", [10, 50, 100, 500, 1000])]
    available_horizon = test_trajectories.shape[1] - 1 if model_name == "neural_ode" else test_trajectories.shape[1] - history_steps
    valid_horizons = [h for h in requested_horizons if h <= available_horizon]
    max_horizon = max(valid_horizons)
    num_eval_trajectories = int(eval_config.get("num_eval_trajectories", min(10, len(test_trajectories))))
    selected = test_trajectories[:num_eval_trajectories]
    true_rollouts, pred_rollouts, rollout_time = generate_rollout_arrays(
        model_name,
        model,
        selected,
        time_grid,
        mean,
        std,
        max_horizon,
        history_steps,
        device,
    )

    rollout_metrics = {}
    for horizon in valid_horizons:
        rollout_metrics[horizon] = compute_rollout_metrics(true_rollouts[:, :horizon], pred_rollouts[:, :horizon])

    mean_error_curve = compute_error_curve(true_rollouts, pred_rollouts).mean(axis=0)
    attractor_metrics = compute_attractor_metrics(true_rollouts.reshape(-1, 3), pred_rollouts.reshape(-1, 3))

    scatter_path = figure_dir / f"figure_{model_name}_one_step.png"
    rollout_path = figure_dir / f"figure_{model_name}_rollout.png"
    phase_path = figure_dir / f"figure_{model_name}_phase_portraits.png"
    attractor_path = figure_dir / f"figure_{model_name}_attractor_projections.png"
    error_path = figure_dir / f"figure_{model_name}_error_growth.png"

    plot_one_step_scatter(one_step_true, one_step_pred, str(scatter_path), title=f"{model_name.upper()} one-step prediction quality")
    plot_time_series_comparison(
        rollout_time,
        true_rollouts[0],
        pred_rollouts[0],
        str(rollout_path),
        title=f"{model_name.upper()} rollout comparison",
    )
    plot_phase_portraits(
        true_rollouts[0],
        pred_rollouts[0],
        str(phase_path),
        title=f"{model_name.upper()} phase portraits",
    )
    plot_attractor_projections(
        true_rollouts.reshape(-1, 3),
        pred_rollouts.reshape(-1, 3),
        str(attractor_path),
        title=f"{model_name.upper()} attractor reconstruction",
    )
    plot_error_growth_curve(
        rollout_time,
        {model_name.upper(): mean_error_curve},
        str(error_path),
        title=f"{model_name.upper()} rollout error growth",
    )

    for path in [scatter_path, rollout_path, phase_path, attractor_path, error_path]:
        target = report_asset_dir / path.name
        target.write_bytes(path.read_bytes())

    summary = {
        "model": model_name,
        "checkpoint_path": str(record["checkpoint"]),
        "params_count": count_parameters(model),
        **one_step_metrics,
        **summarise_rollout_metrics(valid_horizons, rollout_metrics),
        "attractor_density_distance": attractor_metrics["occupancy_distance"],
        "attractor_metrics": attractor_metrics,
        "valid_horizons": valid_horizons,
        "error_curve": mean_error_curve.tolist(),
        "rollout_time": rollout_time.tolist(),
        "figures": {
            "one_step": str(scatter_path),
            "rollout": str(rollout_path),
            "phase": str(phase_path),
            "attractor": str(attractor_path),
            "error_growth": str(error_path),
        },
        "best_val_loss": checkpoint.get("best_val_loss"),
    }
    summary_path = log_dir / f"{model_name}_evaluation_summary.json"
    save_json(summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Lorenz model.")
    parser.add_argument("--model", required=True, choices=["mlp", "resnet", "neural_ode"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate_model(args.model, args.config)


if __name__ == "__main__":
    main()
