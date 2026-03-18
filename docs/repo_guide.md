# Repository Guide

## Typical workflow

1. Generate data with `scripts/generate_data.py`.
2. Train the three models under `src/training/`.
3. Evaluate each model with `src/evaluation/evaluate_model.py`.
4. Aggregate tables with `src/evaluation/compare_models.py`.
5. Run `src/evaluation/robustness.py` for noise and step-size ablations.
6. Build report-ready figures with `src/visualization/make_report_figures.py`.

## Where to look

- `src/dynamics/`: Lorenz equations, RK4 integration, simulation utilities, and dataset generation.
- `src/data/`: PyTorch dataset views for one-step, windowed, and continuous-time training.
- `src/models/`: MLP, residual predictor, vector-field network, and Neural ODE.
- `src/training/`: loss definitions, shared training loops, and CLI training entry points.
- `src/evaluation/`: rollout metrics, attractor statistics, robustness experiments, and model comparison.
- `src/visualization/`: reusable plotting utilities plus the report-figure entry point.

## Output conventions

- `results/data/`: raw and processed datasets.
- `results/checkpoints/`: model checkpoints and train summaries.
- `results/logs/`: evaluation summary JSON files.
- `results/figures/`: all generated figures.
- `results/tables/`: dataset, model, evaluation, and robustness tables.
- `results/report_assets/`: clean figure and table copies for direct inclusion in the write-up.

## Testing

Run lightweight checks with:

```bash
python -m pytest -q
```

These tests cover dynamics shapes, rollout utilities, dataset indexing, model forward passes, and metric sanity.
