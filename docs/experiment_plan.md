# Experiment Plan

## Stage A. Understand the system

Goals:

- simulate a standard Lorenz trajectory
- visualize the 3D attractor
- plot `x(t), y(t), z(t)`
- demonstrate divergence from nearby initial conditions
- compare solver behavior across step sizes

Outputs:

- `figure_true_attractor.png`
- `figure_time_series.png`
- `figure_sensitivity.png`
- `figure_solver_comparison.png`

## Stage B. Build supervised datasets

Default dataset design:

- train trajectories: 100
- validation trajectories: 20
- test trajectories: 20
- total simulation time: 25
- default step size: 0.01

Derived views:

- one-step transition pairs
- short-window rollout segments
- continuous trajectory segments for Neural ODE training

## Stage C. Train the MLP baseline

Primary question:

- how good can one-step prediction get?
- how quickly does recursive rollout degrade?

## Stage D. Train the residual predictor

Primary question:

- does predicting state increments better align with discretized dynamics?

## Stage E. Train the Neural ODE

Primary question:

- does learning a continuous-time vector field improve short-horizon fidelity and step-size robustness?

## Stage F. Compare models

Produce:

- one-step metrics
- rollout RMSE at multiple horizons
- phase portrait plots
- attractor projection comparisons
- summary tables for report and README

## Stage G. Robustness studies

Required ablations:

1. Observation noise at levels `0.0`, `0.01`, `0.05`.
2. Step-size shift between `dt = 0.01` and `dt = 0.02`.

Optional extension:

- smaller-data training subsets such as `20`, `50`, and `100` trajectories.

## Expected interpretation

- one-step performance can be misleading in chaotic systems
- residual parameterization may improve local rollout behavior
- Neural ODEs are attractive for vector-field learning and step-size transfer
- attractor-level diagnostics are necessary to evaluate long-term qualitative fidelity
