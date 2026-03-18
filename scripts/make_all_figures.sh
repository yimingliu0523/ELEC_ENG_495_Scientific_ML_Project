#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

$PYTHON_BIN -m src.visualization.make_report_figures --config configs/eval_default.yaml
