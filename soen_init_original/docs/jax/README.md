---
layout: default
title: JAX Backend - Overview
---
# JAX Backend (Overview)

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="../Simulation.md" style="margin-right: 2em;">&#8592; Previous: Simulation</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="../Training_Models.md" style="margin-left: 2em;">Next: Training Models &#8594;</a>
</div>

This page explains how the **JAX backend** is structured today (12th December 2025), how it relates to the **Torch** codepath, and what the current limitations are. The goal is to make it easy to answer:

- **Where does the JAX forward pass live?**
- **What is the "model" in JAX?**
- **How do training and the GUI reuse the same forward?**
- **Where do parameters live and how are they updated?**

---

## Big picture

The JAX backend uses a **single forward implementation** (`unified_forward.forward`) driven by a **topology object** (`JAXModel`), and training wraps that topology in an **Equinox module** (`SoenEqxModel`) so Optax can update parameters cleanly.

---

## How this ties to Torch

In SOEN-Toolkit, the Torch model (`SOENModelCore`) remains the canonical object you build/save/load across the toolkit.

When you run JAX:

1. You start from a Torch model (either created by config/YAML or via the GUI).
2. You call the model’s conversion method (via `port_to_jax`) which produces a `JAXModel`.
3. The JAX forward pass runs from that `JAXModel` using the unified forward implementation.

Why this design?

- **Torch** is the easiest place to author model building, file formats (`.soen`), and rich tooling.
- **JAX** is used for high-performance simulation/training while reusing the same conceptual network structure.

---

## The key building blocks (what to learn first)

### 1) `JAXModel` (topology + runtime knobs)

File: `src/soen_toolkit/utils/port_to_jax/jax_model.py`

`JAXModel` is the JAX-side “model blueprint”. It contains:

- A list of `LayerSpec` and `ConnectionSpec` objects
- Global settings like `dt`, `network_evaluation_method`, `input_type`
- Optional **override hooks** used by training (e.g. connection overrides)
- Caches that are prepared before JIT for performance

It also implements `__call__`, but that is intentionally thin: it delegates to the unified forward pass.

### 2) `unified_forward.forward(...)` (the single source of truth)

File: `src/soen_toolkit/utils/port_to_jax/unified_forward.py`

This is the **single forward implementation** used by:

- JAX training
- The GUI state trajectory runner when using the JAX backend
- Any other JAX inference/simulation call

It handles:

- Layerwise evaluation
- Stepwise solvers (Jacobi / Gauss-Seidel) via `unified_stepwise`
- Optional state carry-in (`initial_states`, `s1_inits`, `s2_inits`)
- Optional parameter overrides (`conn_override`, `internal_conn_override`, `layer_param_override`)
- Optional traces (`ForwardTrace`) for telemetry (e.g. per-layer φ histories)

### 3) `SoenEqxModel` (Equinox wrapper for trainable parameters)

File: `src/soen_toolkit/utils/port_to_jax/eqx_model.py`

`SoenEqxModel` exists because Optax wants “a pytree of parameters”.

It holds:

- `topology: JAXModel` as a **static** field (not trained; not updated)
- trainable leaves:
  - `layer_params`
  - `connection_params`
  - `internal_connections`

Calling `SoenEqxModel(x)` simply calls `unified_forward.forward(...)` with the parameter overrides populated from the module’s leaves.

### 4) Layer implementations (`layers_jax.py`)

File: `src/soen_toolkit/utils/port_to_jax/layers_jax.py`

These are the per-layer simulation kernels used by the unified forward pass.

Important note: some layer types are still **placeholders** (e.g. `Soma`/`Synapse`) and will fail fast if you try to build/run them in supported paths.

### 5) `ForwardTrace` (optional telemetry)

File: `src/soen_toolkit/utils/port_to_jax/forward_trace.py`

`ForwardTrace` is how JAX forward can return “extra stuff” without mutation:

- per-layer state histories (always)
- optional `phi_by_layer`
- NOCC auxiliary states (`s1_final_by_layer`, `s2_final_by_layer`) so callers can carry state between runs

### 6) Surrogate spiking ops (`soen_toolkit.ops`)

Folder: `src/soen_toolkit/ops/`

This provides backend-agnostic primitives for “hard” operations with surrogate gradients:

- `spike_torch` (custom autograd)
- `spike_jax` (custom VJP)
- a registry of surrogates (e.g. triangle derivative), with a consistent interface

---

## The call chain (training vs GUI)

### GUI / State trajectory plotting (JAX backend)

- Torch model → `port_to_jax(prepare=True)` → `JAXModel`
- GUI caches and JITs `JAXModel.__call__`
- `JAXModel.__call__` → `unified_forward.forward(...)`

### JAX training

- Torch model → convert → `JAXModel`
- Build initial parameter arrays
- Wrap in `SoenEqxModel(topology=jax_model, ...)`
- Training calls `SoenEqxModel(x)` which → `unified_forward.forward(...)`

So: **both** flows hit the same forward implementation.

---

## Current limitations / caveats

- **`g`-history tracing is not exposed in JAX yet**
  - Some GUI metrics that require `g` will fail fast on the JAX backend.
- **Placeholder layers** (`Soma`, `Synapse`) exist as scaffolding
  - They are registered so the intended architecture is visible, but they are not implemented and should fail fast.
- **Equinox integration is “params-first”**
  - The topology is still owned by `JAXModel`.
  - Parameters live in `SoenEqxModel` to match the standard JAX/Optax approach.
- **Checkpointing**
  - `.soen` is still the cross-backend portable format.
  - JAX `.pkl` checkpoints are for resuming JAX training (topology + params + opt_state).

---

## Where to look next

- **Simulation & solvers (backend-agnostic)**: `docs/Simulation.md`
- **Model building**: `docs/Building_Models.md`
- **Training configs**: `docs/Training_Models.md`

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="../Simulation.md" style="margin-right: 2em;">&#8592; Previous: Simulation</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="../Training_Models.md" style="margin-left: 2em;">Next: Training Models &#8594;</a>
</div>
