#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

$PYTHON_BIN scripts/generate_data.py --config configs/data_default.yaml
$PYTHON_BIN -m src.training.train_mlp --config configs/train_mlp.yaml
$PYTHON_BIN -m src.training.train_resnet --config configs/train_resnet.yaml
$PYTHON_BIN -m src.training.train_neural_ode --config configs/train_neural_ode.yaml
$PYTHON_BIN -m src.evaluation.evaluate_model --model mlp --config configs/eval_default.yaml
$PYTHON_BIN -m src.evaluation.evaluate_model --model resnet --config configs/eval_default.yaml
$PYTHON_BIN -m src.evaluation.evaluate_model --model neural_ode --config configs/eval_default.yaml
$PYTHON_BIN -m src.evaluation.compare_models --config configs/eval_default.yaml
$PYTHON_BIN -m src.evaluation.robustness --config configs/ablation_noise.yaml
$PYTHON_BIN -m src.visualization.make_report_figures --config configs/eval_default.yaml
