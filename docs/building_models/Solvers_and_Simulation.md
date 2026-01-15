---
layout: default
title: Solvers & Simulation (Building Models)
---
## Solvers & Simulation (Building Models)

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Connections.md" style="margin-right: 2em;">&#8592; Previous: Connections</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="../Simulation.md" style="margin-left: 2em;">Next: Simulation (Torch + JAX) &#8594;</a>
</div>

This page explains the **global solver mode** (`network_evaluation_method`) and when to use each option.

If you want the backend-agnostic overview (Torch + JAX), start here instead:

- `Simulation.md`

---

## `network_evaluation_method` (global evaluator)

The `network_evaluation_method` field in `SimulationConfig` controls how the **whole network** is evaluated over time.

> **Parallel Solver:** Looking for time-parallel training? Check out the [ParaRNN Parallel Solver](../DeeperDives/ParaRNN_Solver.md) for O(log T) parallelization.

![Decision Tree for Global Solver](../Figures/Building_Models/model_building_fig3.jpg)

### Layerwise (default, fastest)

Process each layer completely before moving to next:

```text
Layer 0 [all timesteps] → Layer 1 [all timesteps] → Layer 2 [all timesteps]
```

- **Use for**: feedforward networks (DAG graphs)
- **Don’t use for**: networks with backward/recurrent connections

### Stepwise Gauss-Seidel

Process all layers at each timestep using freshest available states:

```text
t=0: Layer 0 → Layer 1 → Layer 2 (using just-computed states)
t=1: Layer 0 → Layer 1 → Layer 2 (using just-computed states)
...
```

- **Use for**: networks with recurrent/feedback connections
- **Cost**: sequential updates per timestep
- **Feedback delay**: backward edges (higher ID → lower ID) incur a one-timestep ($\Delta t$) delay

### Stepwise Jacobi

Process all layers at each timestep using snapshot from the previous timestep:

```text
t=0: Compute all φ from s[t-1] → Update all layers in parallel
t=1: Compute all φ from s[t] → Update all layers in parallel
...
```

- **Use for**: networks with recurrent connections
- **Note**: conceptually parallelizable

---

## Early stopping (stepwise only)

Stepwise solvers can terminate early when the network reaches steady state.

```python
from soen_toolkit.core import SimulationConfig

SimulationConfig(
    network_evaluation_method="stepwise_gauss_seidel",
    early_stopping_forward_pass=True,

    # Simple mode (patience-based)
    early_stopping_tolerance=1e-6,
    early_stopping_patience=3,
    early_stopping_min_steps=10,

    # Windowed mode (preferred for noisy systems)
    steady_window_min=50,
    steady_tol_abs=1e-5,
    steady_tol_rel=1e-3,
)
```

When `steady_window_min > 0`, the “windowed” detector:

- tracks $\max(\lvert\Delta s\rvert)$ and $\max(\lvert s\rvert)$ over a trailing window
- stops when $\max(\lvert\Delta s\rvert) \leq$ `steady_tol_abs` + `steady_tol_rel` $\times \max(\lvert s\rvert)$

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Connections.md" style="margin-right: 2em;">&#8592; Previous: Connections</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="../Simulation.md" style="margin-left: 2em;">Next: Simulation (Torch + JAX) &#8594;</a>
</div>
