#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

$PYTHON_BIN -m src.training.train_mlp --config configs/train_mlp.yaml
$PYTHON_BIN -m src.training.train_resnet --config configs/train_resnet.yaml
$PYTHON_BIN -m src.training.train_neural_ode --config configs/train_neural_ode.yaml
