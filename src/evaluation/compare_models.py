"""Compare trained Lorenz models using saved evaluation summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json, write_markdown_table
from src.visualization.plot_error_curves import plot_error_growth_curve


def compare_models(config_path: str) -> dict[str, Path]:
    config = load_config(config_path)
    log_dir = ensure_dir(config.get("log_dir", "results/logs"))
    table_dir = ensure_dir(config.get("table_dir", "results/tables"))
    figure_dir = ensure_dir(config.get("figure_dir", "results/figures"))
    report_asset_dir = ensure_dir(config.get("report_asset_dir", "results/report_assets"))

    evaluation_rows = []
    model_rows = []
    error_curves = {}
    time_axis = None

    for model_name in config["models"]:
        summary = load_json(log_dir / f"{model_name}_evaluation_summary.json")
        training_summary_path = Path(config["models"][model_name].get("training_summary", f"results/checkpoints/{model_name}_train_summary.json"))
        training_summary = load_json(training_summary_path) if training_summary_path.exists() else {}

        evaluation_rows.append(
            {
                "model": model_name,
                "one_step_mse": summary.get("one_step_mse"),
                "rollout_rmse_50": summary.get("rollout_rmse_50"),
                "rollout_rmse_100": summary.get("rollout_rmse_100"),
                "rollout_rmse_500": summary.get("rollout_rmse_500"),
                "mean_abs_error": summary.get("one_step_mae"),
                "attractor_density_distance": summary.get("attractor_density_distance"),
                "notes": config["models"][model_name].get("notes", ""),
            }
        )
        model_rows.append(
            {
                "model": model_name,
                "parameter_count": training_summary.get("params_count", summary.get("params_count")),
                "training_epochs": training_summary.get("epochs"),
                "training_time_sec": training_summary.get("train_time_sec"),
                "best_validation_loss": training_summary.get("best_val_loss", summary.get("best_val_loss")),
            }
        )
        error_curves[model_name.upper()] = summary["error_curve"]
        time_axis = summary["rollout_time"]

    evaluation_df = pd.DataFrame(evaluation_rows)
    model_df = pd.DataFrame(model_rows)

    evaluation_csv = table_dir / "final_evaluation_table.csv"
    evaluation_md = table_dir / "final_evaluation_table.md"
    model_csv = table_dir / "model_summary.csv"
    model_md = table_dir / "model_summary.md"

    evaluation_df.to_csv(evaluation_csv, index=False)
    write_markdown_table(evaluation_df, evaluation_md)
    model_df.to_csv(model_csv, index=False)
    write_markdown_table(model_df, model_md)

    plot_error_growth_curve(
        pd.Series(time_axis).to_numpy(),
        {key: pd.Series(value).to_numpy() for key, value in error_curves.items()},
        str(figure_dir / "figure_error_growth_all_models.png"),
        title="Error growth across learned models",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["one_step_mse", "rollout_rmse_100", "attractor_density_distance"]
    titles = ["One-step MSE", "Rollout RMSE (100)", "Attractor density distance"]
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(evaluation_df["model"], evaluation_df[metric], color=["#0f4c5c", "#e36414", "#6a994e"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    summary_figure = figure_dir / "figure_final_summary.png"
    fig.savefig(summary_figure, bbox_inches="tight", dpi=250)
    plt.close(fig)

    for path in [evaluation_csv, evaluation_md, model_csv, model_md, figure_dir / "figure_error_growth_all_models.png", summary_figure]:
        target = report_asset_dir / path.name
        target.write_bytes(path.read_bytes())

    return {
        "evaluation_csv": evaluation_csv,
        "evaluation_md": evaluation_md,
        "model_csv": model_csv,
        "model_md": model_md,
        "error_figure": figure_dir / "figure_error_growth_all_models.png",
        "summary_figure": summary_figure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all evaluated Lorenz models.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    compare_models(args.config)


if __name__ == "__main__":
    main()
