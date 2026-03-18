# Presentation Storyline

## Goal

Deliver a clean 12-minute conference-style talk that emphasizes the central SciML lesson: local predictive accuracy is not enough for chaotic dynamical systems.

## Suggested slide flow

1. Motivation
   Chaotic systems stress-test learned dynamics.
2. Problem setup
   Introduce the Lorenz equations and the butterfly attractor.
3. Why forecasting chaos is hard
   Sensitivity to initial conditions and rollout compounding.
4. Methods
   Numerical solver, MLP baseline, residual predictor, Neural ODE.
5. Dataset and experiment design
   Trajectory splits, step size, and evaluation horizons.
6. Baseline one-step results
   Show that one-step prediction can look strong.
7. Rollout results
   Highlight short-horizon vs long-horizon mismatch.
8. Attractor reconstruction
   Compare true and learned phase portraits / projected densities.
9. Robustness
   Noise and step-size sensitivity.
10. Takeaways
    Good one-step fits are not enough; attractor-level evaluation matters.

## Verbal emphasis

- Be explicit that long-horizon pointwise divergence is expected in chaotic systems.
- Avoid overselling exact trajectory matching as the only success criterion.
- Frame the Neural ODE as a better structural model, not a magic solution.

## Recommended closing line

A good SciML model for chaotic dynamics should be judged not only by what it predicts next, but by the dynamical world it reconstructs over time.
