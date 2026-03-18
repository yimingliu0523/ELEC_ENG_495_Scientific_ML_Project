#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

$PYTHON_BIN -m src.evaluation.evaluate_model --model mlp --config configs/eval_default.yaml
$PYTHON_BIN -m src.evaluation.evaluate_model --model resnet --config configs/eval_default.yaml
$PYTHON_BIN -m src.evaluation.evaluate_model --model neural_ode --config configs/eval_default.yaml
$PYTHON_BIN -m src.evaluation.compare_models --config configs/eval_default.yaml
$PYTHON_BIN -m src.evaluation.robustness --config configs/ablation_noise.yaml
