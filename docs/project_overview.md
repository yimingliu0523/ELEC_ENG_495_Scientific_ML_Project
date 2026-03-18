# Project Overview

This repository studies a standard chaotic benchmark in Scientific Machine Learning: the Lorenz system. The goal is not simply to minimize one-step prediction error, but to understand how different learning paradigms behave when a continuous-time system is chaotic and long-horizon pointwise tracking is fundamentally fragile.

The project compares three modeling routes:

1. Classical numerical simulation of the true Lorenz equations.
2. A discrete-time MLP predictor trained on state-transition supervision.
3. A residual discrete-time model that predicts state increments.
4. A Neural ODE that learns a continuous-time vector field.

The scientific question is: when do these models look good under local supervision, and when do they preserve or destroy the geometry of the learned dynamics?

The repository is organized around reproducibility:

- YAML configs define data generation, training, evaluation, and ablation settings.
- Scripts save checkpoints, logs, summary tables, and report-ready figures.
- Evaluation includes both local predictive metrics and qualitative dynamical diagnostics.
- The results directory is structured to support a short report and a conference-style talk.

Expected narrative:

- Chaotic divergence makes long-horizon pointwise mismatch inevitable.
- Good short-horizon prediction does not imply good global dynamics.
- Attractor reconstruction is a more meaningful long-term test than exact trajectory tracking.
