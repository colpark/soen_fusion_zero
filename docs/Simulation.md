---
layout: default
title: Simulation (Torch + JAX)
---
# Simulation (Torch + JAX)

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Building_Models.md" style="margin-right: 2em;">&#8592; Previous: Building Models</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Training_Models.md" style="margin-left: 2em;">Next: Training Models &#8594;</a>
</div>

This page explains what “simulation” means in SOEN-Toolkit and how it works in **both backends**:

- **Torch backend**: `SOENModelCore(...)` forward pass
- **JAX backend**: `JAXModel(...)` / `SoenEqxModel(...)` forward pass

The goal is to give you a single mental model for:

- Shapes and conventions (`[B, T, D]`, and why outputs often include `t=0`)
- Global solver/evaluation methods (layerwise vs stepwise)
- State histories and what the GUI plots
- What is shared between Torch and JAX, and what differs today

---

## What “simulation” means here

A SOEN model is a network of dynamical elements (dendrites/multipliers/etc.). A “forward pass” is not a single matrix multiply — it is a time evolution that produces:

- **A full state history per layer** (what the GUI plots)
- A final layer output sequence (often the last layer’s state over time)

---

## Input and output shapes (the most important convention)

### Inputs

Most simulation entrypoints expect:

- `x`: shape `[batch, time, features]` (written as `[B, T, D_in]`)

### Outputs

Many SOEN layers expose state histories including the initial condition:

- `history`: shape `[B, T+1, D]`

The extra `+1` is because the simulation commonly includes an explicit initial state at `t=0`.

This is why you often see code that drops the first timestep (`history[:, 1:, :]`) for supervised learning targets.

---

## Global network evaluation methods

The global solver/evaluator controls how the network is evaluated over time given a graph of layers and connections.

### `layerwise`

Compute an entire layer’s time evolution before moving to the next layer.

This is fast and simple, and matches feed-forward graphs well.

### `stepwise_*` (feedback / recurrent graphs)

For networks with feedback (or when you want coupled updates), you evaluate “one timestep at a time”:

- `stepwise_jacobi`: update all layers for a timestep using the previous timestep’s states
- `stepwise_gauss_seidel`: update layers sequentially within the timestep (uses newest states)

In the JAX backend, these stepwise methods are implemented via `jax.lax.scan`.

---

## What the GUI uses

The GUI’s “state trajectory” tool runs the model forward and visualizes per-layer histories.

- Torch backend: uses `SOENModelCore` forward and extracts histories directly.
- JAX backend: converts the Torch model to `JAXModel`, then calls `JAXModel.__call__` (JIT compiled).

In the JAX backend, the GUI typically requests an optional `ForwardTrace` so it can:

- collect per-layer `phi` histories when requested (`track_phi=True`)
- carry NOCC auxiliary states between runs (no mutation)

---

## What is shared between Torch and JAX (today)

- Same conceptual layer graph: layers + connections + global simulation config
- Same high-level evaluation methods: layerwise vs stepwise solvers
- Same idea of “history over time” per layer

---

## What differs / current limitations (JAX)

- **Some telemetry is not exposed yet**
  - In particular, `g`-history tracing is not currently available in the JAX backend (some GUI metrics will fail fast).

---

## Where to learn the JAX internals

- `docs/jax/README.md` (overview and building blocks)

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Building_Models.md" style="margin-right: 2em;">&#8592; Previous: Building Models</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Training_Models.md" style="margin-left: 2em;">Next: Training Models &#8594;</a>
</div>
