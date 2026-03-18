# 12-Minute Presentation Outline

## Slide 1. Title and motivation

- Why Lorenz is a clean benchmark for Scientific ML
- Local accuracy vs global dynamical fidelity

## Slide 2. Lorenz system basics

- Equations
- Standard parameters
- Reference attractor figure

## Slide 3. Why chaos is challenging

- Nearby initial conditions diverge
- Long-horizon pointwise prediction is fragile

## Slide 4. Methods compared

- Numerical solver
- MLP predictor
- Residual predictor
- Neural ODE

## Slide 5. Dataset and protocol

- train / val / test splits by trajectory
- rollout horizons
- robustness settings

## Slide 6. One-step results

- prediction scatter
- short narrative on supervised fit quality

## Slide 7. Rollout results

- time-series comparison
- error-growth figure

## Slide 8. Attractor reconstruction

- true vs predicted projections
- explain why this matters

## Slide 9. Robustness

- noise curve
- step-size transfer

## Slide 10. Final takeaways

- one-step error can mislead
- continuous-time structure helps
- attractor fidelity is the right long-term lens
