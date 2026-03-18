"""Train the baseline MLP one-step predictor."""

from __future__ import annotations

import torch

from src.data.datamodule import LorenzDataModule
from src.data.dataset import load_dataset_arrays
from src.data.normalization import denormalize_states, normalize_states
from src.evaluation.rollout import recursive_rollout_discrete_model
from src.models import build_model
from src.training.trainer import train_supervised_model
from src.utils.config import build_config_argparser, load_config, save_config_snapshot
from src.utils.io import ensure_dir, save_json
from src.utils.seeds import set_seed
from src.visualization.plot_time_series import plot_time_series_comparison


def main() -> None:
    parser = build_config_argparser("Train the Lorenz MLP baseline.")
    args = parser.parse_args()
    config = load_config(args.config)

    set_seed(int(config["seed"]))
    device = torch.device(config.get("device", "cpu"))

    dataset_path = config["data"]["dataset_path"]
    training_cfg = config["training"]
    model_kwargs = {key: value for key, value in config["model"].items() if key != "name"}

    datamodule = LorenzDataModule(
        dataset_path,
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    train_loader, val_loader, _ = datamodule.one_step_loaders(
        history_steps=int(model_kwargs.get("history_steps", 1)),
        prediction_horizon=int(training_cfg.get("prediction_horizon", 1)),
        normalize=bool(training_cfg.get("normalize", True)),
        flatten_history=True,
    )

    model = build_model("mlp", model_kwargs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    checkpoint_dir = ensure_dir(config["output"]["checkpoint_dir"])
    figure_dir = ensure_dir(config["output"]["figure_dir"])

    summary = train_supervised_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=int(training_cfg["epochs"]),
        output_dir=checkpoint_dir,
        run_name="mlp",
        extra_checkpoint_data={"model_name": "mlp", "model_kwargs": model_kwargs},
    )
    save_config_snapshot(config, checkpoint_dir, filename="mlp_config_used.yaml")

    checkpoint = torch.load(summary["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    arrays = load_dataset_arrays(dataset_path)
    mean = arrays["mean"]
    std = arrays["std"]
    val_trajectory = arrays["val_trajectories"][0]
    time_grid = arrays["time_grid"]
    history_steps = int(model_kwargs.get("history_steps", 1))
    rollout_horizon = min(int(training_cfg.get("sample_rollout_horizon", 200)), val_trajectory.shape[0] - history_steps)
    history = normalize_states(val_trajectory[:history_steps], mean, std)
    with torch.no_grad():
        prediction_norm = recursive_rollout_discrete_model(model, history, rollout_horizon, device=device).cpu().numpy()
    prediction = denormalize_states(prediction_norm, mean, std)
    target = val_trajectory[history_steps : history_steps + rollout_horizon]
    figure_path = figure_dir / "mlp_validation_rollout.png"
    plot_time_series_comparison(
        time_grid[history_steps : history_steps + rollout_horizon],
        target,
        prediction,
        str(figure_path),
        title="MLP validation rollout",
    )

    summary["sample_rollout_figure"] = str(figure_path)
    save_json(summary, checkpoint_dir / "mlp_train_summary.json")


if __name__ == "__main__":
    main()
