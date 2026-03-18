"""Generate publication-ready figures for the report and presentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.dynamics.dataset_generation import build_time_grid
from src.dynamics.lorenz import get_default_lorenz_params
from src.dynamics.simulation import generate_single_trajectory
from src.dynamics.solvers import simulate_trajectory
from src.evaluation.compare_models import compare_models
from src.evaluation.robustness import run_robustness_experiments
from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.visualization.plot_attractors import plot_attractor_projections, plot_reference_attractor
from src.visualization.plot_phase_portraits import plot_solver_comparison
from src.visualization.plot_time_series import plot_sensitivity_to_initial_conditions, plot_time_series


def generate_reference_figures(data_config_path: str, figure_dir: Path, report_asset_dir: Path) -> None:
    data_config = load_config(data_config_path)
    params = data_config.get("system", get_default_lorenz_params())
    dt = float(data_config["data"]["dt"])
    total_time = float(data_config["data"]["total_time"])
    time_grid = build_time_grid(total_time=total_time, dt=dt)

    initial_state = np.array([1.0, 1.0, 1.0], dtype=float)
    trajectory = generate_single_trajectory(initial_state, time_grid, params=params)
    plot_reference_attractor(trajectory, str(figure_dir / "figure_true_attractor.png"), title="True Lorenz attractor")
    plot_time_series(time_grid, trajectory, str(figure_dir / "figure_time_series.png"))

    perturbed_state = initial_state + np.array([1e-5, 0.0, 0.0], dtype=float)
    perturbed_trajectory = generate_single_trajectory(perturbed_state, time_grid, params=params)
    plot_sensitivity_to_initial_conditions(
        time_grid,
        trajectory,
        perturbed_trajectory,
        str(figure_dir / "figure_sensitivity.png"),
    )

    fine_grid = build_time_grid(total_time=total_time, dt=0.005)
    coarse_grid = build_time_grid(total_time=total_time, dt=0.02)
    fine_trajectory = simulate_trajectory(initial_state, fine_grid, params=params, solver="rk4")
    medium_trajectory = simulate_trajectory(initial_state, time_grid, params=params, solver="rk4")
    coarse_trajectory = simulate_trajectory(initial_state, coarse_grid, params=params, solver="rk4")

    coarse_interp = np.column_stack([np.interp(fine_grid, coarse_grid, coarse_trajectory[:, idx]) for idx in range(3)])
    medium_interp = np.column_stack([np.interp(fine_grid, time_grid, medium_trajectory[:, idx]) for idx in range(3)])
    plot_solver_comparison(
        fine_grid,
        {
            "RK4 dt=0.005": fine_trajectory,
            "RK4 dt=0.01": medium_interp,
            "RK4 dt=0.02": coarse_interp,
        },
        str(figure_dir / "figure_solver_comparison.png"),
        state_index=0,
        title="Solver / step-size comparison on x(t)",
    )

    for name in [
        "figure_true_attractor.png",
        "figure_time_series.png",
        "figure_sensitivity.png",
        "figure_solver_comparison.png",
    ]:
        source = figure_dir / name
        (report_asset_dir / name).write_bytes(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready Lorenz figures.")
    parser.add_argument("--config", default="configs/eval_default.yaml")
    args = parser.parse_args()

    eval_config = load_config(args.config)
    figure_dir = ensure_dir(eval_config.get("figure_dir", "results/figures"))
    report_asset_dir = ensure_dir(eval_config.get("report_asset_dir", "results/report_assets"))
    data_config_path = eval_config.get("data_config", "configs/data_default.yaml")

    generate_reference_figures(data_config_path, figure_dir, report_asset_dir)

    try:
        compare_models(args.config)
    except FileNotFoundError:
        pass

    robustness_config = "configs/ablation_noise.yaml"
    if Path(robustness_config).exists():
        try:
            run_robustness_experiments(robustness_config)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
