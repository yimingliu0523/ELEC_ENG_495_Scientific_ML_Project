"""Train the Neural ODE Lorenz model."""

from __future__ import annotations

import torch

from src.data.datamodule import LorenzDataModule
from src.data.dataset import load_dataset_arrays
from src.data.normalization import denormalize_states, normalize_states
from src.evaluation.rollout import rollout_neural_ode_model
from src.models import build_model
from src.training.trainer import train_sequence_model
from src.utils.config import build_config_argparser, load_config, save_config_snapshot
from src.utils.io import ensure_dir, save_json
from src.utils.seeds import set_seed
from src.visualization.plot_time_series import plot_time_series_comparison


def main() -> None:
    parser = build_config_argparser("Train the Lorenz Neural ODE model.")
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
    train_loader, val_loader, _ = datamodule.continuous_loaders(
        segment_length=int(training_cfg["segment_length"]),
        stride=int(training_cfg.get("segment_stride", 1)),
        normalize=bool(training_cfg.get("normalize", True)),
    )

    model = build_model("neural_ode", model_kwargs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    checkpoint_dir = ensure_dir(config["output"]["checkpoint_dir"])
    figure_dir = ensure_dir(config["output"]["figure_dir"])

    summary = train_sequence_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=int(training_cfg["epochs"]),
        output_dir=checkpoint_dir,
        run_name="neural_ode",
        extra_checkpoint_data={"model_name": "neural_ode", "model_kwargs": model_kwargs},
    )
    save_config_snapshot(config, checkpoint_dir, filename="neural_ode_config_used.yaml")

    checkpoint = torch.load(summary["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    arrays = load_dataset_arrays(dataset_path)
    mean = arrays["mean"]
    std = arrays["std"]
    val_trajectory = arrays["val_trajectories"][0]
    time_grid = arrays["time_grid"]
    rollout_horizon = min(int(training_cfg.get("sample_rollout_horizon", 200)), val_trajectory.shape[0] - 1)
    eval_time = time_grid[: rollout_horizon + 1] - time_grid[0]
    initial_state = normalize_states(val_trajectory[0], mean, std)
    with torch.no_grad():
        prediction_norm = rollout_neural_ode_model(model, initial_state, eval_time, device=device).cpu().numpy()
    prediction = denormalize_states(prediction_norm, mean, std)[1:]
    target = val_trajectory[1 : rollout_horizon + 1]
    figure_path = figure_dir / "neural_ode_validation_rollout.png"
    plot_time_series_comparison(
        time_grid[1 : rollout_horizon + 1],
        target,
        prediction,
        str(figure_path),
        title="Neural ODE validation rollout",
    )

    summary["sample_rollout_figure"] = str(figure_path)
    save_json(summary, checkpoint_dir / "neural_ode_train_summary.json")


if __name__ == "__main__":
    main()
