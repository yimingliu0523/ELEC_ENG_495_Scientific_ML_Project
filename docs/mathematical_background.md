# Mathematical Background

## The Lorenz system

We consider the autonomous ODE

\[
\dot{x} = \sigma (y - x), \qquad
\dot{y} = x(\rho - z) - y, \qquad
\dot{z} = xy - \beta z.
\]

The state is \(s(t) = [x(t), y(t), z(t)]^\top\). Throughout the project we use the standard chaotic parameter choice

- \(\sigma = 10\)
- \(\rho = 28\)
- \(\beta = 8/3\)

These parameters place the system in the classic butterfly-attractor regime.

## Role of the parameters

- `sigma` controls the coupling between `x` and `y` and is often interpreted as the Prandtl number in the original convection model.
- `rho` is the forcing parameter and strongly affects stability and the onset of chaos.
- `beta` controls damping in the `z` direction.

## Equilibria

The Lorenz system has three equilibrium points:

- the origin `(0, 0, 0)`
- two nonzero symmetric equilibria
  `(+sqrt(beta (rho - 1)), +sqrt(beta (rho - 1)), rho - 1)` and its sign-flipped counterpart

Under the standard parameter choice, these fixed points are unstable, which contributes to the observed chaotic motion.

## Sensitivity to initial conditions

Chaos means that two trajectories started extremely close to one another can separate rapidly over time. This makes exact long-horizon state tracking difficult even when the learned model captures some meaningful aspects of the underlying dynamics.

That point matters for evaluation: large long-horizon pointwise error does not automatically mean the model learned nothing useful. A model may still recover the rough attractor shape, marginal state statistics, and phase portrait geometry.

## Three learning views

### One-step supervised prediction

A discrete-time predictor is trained on pairs `(s_t, s_{t+1})`. This is a local approximation to the flow map over one step of size `dt`.

### Rollout forecasting

At test time, the model is recursively fed its own predictions. Small one-step errors can accumulate and compound, so rollout error can grow much faster than one-step validation loss suggests.

### Learning the vector field

A Neural ODE learns a function `f_theta(s)` intended to approximate `ds/dt`. Its prediction is obtained by integrating the learned vector field through time. This is closer in spirit to the true continuous-time system and naturally supports varying evaluation step sizes.

## Why chaotic divergence is not the whole story

For chaotic systems, the right question is not “does the model track the exact trajectory forever?” but rather:

- does it produce accurate local predictions?
- does it remain stable over a meaningful short horizon?
- does it preserve the qualitative attractor geometry?
- does it reproduce coarse temporal statistics and projected densities?

That distinction motivates the mixed evaluation protocol used throughout this repo.
