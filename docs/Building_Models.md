---
layout: default
title: Building Models - Complete Guide
---
# Building Models: Complete Guide

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Getting_Started.md" style="margin-right: 2em;">&#8592; Previous: Getting Started</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Training_Models.md" style="margin-left: 2em;">Next: Training Models &#8594;</a>
</div>

This guide shows you how to construct SOEN models using the spec-based API. You'll learn to define layers, connect them into networks, and configure simulation parameters. We recommend launching the model creation GUI (python -m soen_toolkit.model_creation_gui) alongside this guide—it provides visual feedback as you experiment with the concepts below. Full GUI documentation at: [GUI_Tools](GUI_Tools.md).

> **New:** Prefer a PyTorch-native API? Check out [PyTorch-Style API (Graph & Sequential)](PyTorch_API.md) for an imperative model building approach. This guide covers the spec-based (config/YAML) approach, which is equally powerful and often preferred for reproducibility and hyperparameter tuning.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [The Three Building Blocks](#the-three-building-blocks)
- [Simulation (Torch + JAX)](Simulation.md)
- [Model I/O: Build, Save, Load, and File Formats](#model-io-build-save-load-and-file-formats)
- [Layer Types Catalog](building_models/Layer_Types.md)
- [Connection Patterns](building_models/Connections.md)
- [Dynamic V2 Connections: Programmable Weights](#dynamic-v2-connections-programmable-weights)
- [Connection Weight Initialization](#connection-weight-initialization)
- [Solver Modes Explained](#solver-modes-explained)
- [Complete Examples](#complete-examples)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

**5-Minute Example:** Create a simple 2-layer feedforward network.

```python
from soen_toolkit.core import SOENModelCore, SimulationConfig, LayerConfig, ConnectionConfig

# 1. Global simulation settings
sim_config = SimulationConfig(
    dt=37,                  # Time step (dimensionless units) ~0.1ns per step (this depends on `omega_c` specified in `src/soen_toolkit/physics/constants.py`)
    network_evaluation_method="layerwise" # Global evaluation method. This means we fully evaluate all time steps of the layer before moving to the next
)

# 2. Define layers
layers = [
    LayerConfig(
        layer_id=0,
        layer_type="Linear",      # Input layer (no dynamics)
        params={"dim": 10}
    ),
    LayerConfig(
        layer_id=1,
        layer_type="SingleDendrite",  # SOEN dendrite circuit
        params={
            "dim": 5,
            "solver": "FE",               # Forward Euler solver
            "source_func_type": "RateArray", # Lookup table from circuit sims
            "bias_current": 1.7,          # Activation threshold
            "phi_offset": 0.23,
            "gamma_plus": 0.001,          # Drive term gain
            "gamma_minus": 0.001,         # Leak rate
        }
    )
]

# 3. Define connections
connections = [
    ConnectionConfig(
        from_layer=0,
        to_layer=1,
        connection_type="dense"  # All-to-all connectivity
    )
]

# 4. Build and use
model = SOENModelCore(sim_config, layers, connections)

# Run a forward pass
import torch
input_seq = torch.randn(2, 100, 10)  # (batch, time, input_dim)
output, history = model(input_seq)

print(f"Output shape: {output.shape}")  # (2, 101, 5) - includes t=0 initial condition
```

> **What just happened?**
>
> - **Layer 0**: 10 static input nodes (no dynamics)
> - **Layer 1**: 5 superconducting dendrites that integrate inputs over time, governed by differential equations from circuit physics
> - **Output**: The state of all 5 dendrites at each timestep (101 timesteps total, including initial state)"

---

## Architecture Overview

SOEN models are built from three core components that define the network structure and behavior:

![Model Building Architecture](Figures/Building_Models/model_building_fig1.jpg)

**Key Concepts:**

1. **Layers** define computation (e.g., SOEN dendrites, LSTM cells, linear transforms)
2. **Connections** define information flow between layers (weight matrices)
3. **SimulationConfig** controls how the network evolves over time

---

## The Three Building Blocks


### 1. SimulationConfig: Global Settings

Controls how the entire network behaves during evaluation.

```python
from soen_toolkit.core import SimulationConfig

sim_config = SimulationConfig(
    dt=37,                        # Integration time step
    dt_learnable=False,             # Make dt trainable?
    input_type="state",             # "state" or "flux" - how layer 0 is fed
    track_phi=False,                # Record input flux history?
    track_g=False,                  # Record source function values?
    track_s=False,                  # Record state history?
    track_power=False,              # Track power (SingleDendrite only)
    network_evaluation_method="layerwise",      # Network evaluation method
    
    # Early stopping (for stepwise solvers)
    early_stopping_forward_pass=False,
    early_stopping_tolerance=1e-6,
    early_stopping_patience=1,
    
    # Windowed steady-state detection (preferred for early stopping)
    steady_window_min=50,
    steady_tol_abs=1e-5,
    steady_tol_rel=1e-3,
)
```

#### Key Parameters

| Parameter | What It Does | When to Change |
|-------|----------|----------|
| `dt` | Time step size (dimensionless) | Smaller for stiff dynamics, larger for speed |
| `dt_learnable` | Allow dt to be optimized during training | Experimental feature; rarely needed in practice |
| `input_type` | How to interpret layer 0 | "state": clamp input layer to provided values (most common). "flux": drive input layer with flux signal (for dynamic inputs) |
| `network_evaluation_method` | Network evaluation method | Use "stepwise_*" for recurrent/feedback, use layerwise if the graph is a DAG as it is faster|
| `track_*` | Enable trajectory recording | `True` for analysis, `False` for speed |
| `early_stopping_*` | Stop when network reaches steady state | Used when running evolution until equilibrium |

> **Physical Meaning of `dt`:**
> 
> - In simulation: `dt` is dimensionless
> - In hardware: physical seconds per step = $\frac{dt}{\omega_c}$
> - Use `soen_toolkit.physics.constants.get_omega_c()` for conversions
> - The characteristic frequency $\omega_c$​ varies with fabrication process. Always use `soen_toolkit.physics.constants.get_omega_c()` for unit conversions—never hard-code values.

---

### 2. LayerConfig: Defining Computation

Each layer is a group of neurons/nodes that share properties.

```python
from soen_toolkit.core import LayerConfig, NoiseConfig, PerturbationConfig

layer = LayerConfig(
    layer_id=1,                     # Unique integer ID
    model_id=0,                     # Sub-module ID (for visualization)
    layer_type="SingleDendrite",    # Type of computation
    description="Hidden layer",     # Optional documentation
    params={                        # Type-specific parameters
        "dim": 20,                  # Number of nodes
        "solver": "FE",             # Integration method
        "source_func_type": "RateArray", # Nonlinearity
        # ... more type-specific params
    },
    noise=NoiseConfig(              # Optional: stochastic noise per timestep
        phi=0.01,                   # Noise on input flux
        s=0.005,                    # Noise on state
        relative=False              # Absolute vs relative scaling
    ),
    perturb=PerturbationConfig(     # Optional: deterministic offset per forward
        bias_current_mean=0.0,
        bias_current_std=0.05,
    )
)
```

#### Layer ID Rules

- Must be unique integers (0, 1, 5, 10 is valid—no need for consecutive IDs)
- Layer 0 is always the input layer
- Evaluated in topological order regardless of IDs

#### Parameter Initialization

SOEN layers need initial parameter values. You can specify distributions:

```python
params = {
    "dim": 10,
    "bias_current": {
        "distribution": "uniform",
        "params": {"min": 1.5, "max": 2.0}
    },
    "gamma_plus": {
        "distribution": "lognormal",
        "params": {"mean": -6.9, "std": 0.2}  # log-space mean; actual values ~1e-3
    },
    "phi_offset": 0.23,  # Scalar = constant for all nodes
}
```

**Available Distributions:**

| Distribution | Parameters | Use Case |
|----------|-------|------|
| `constant` | `value` | All nodes identical |
| `normal` | `mean`, `std` | Gaussian distribution around a center point |
| `uniform` | `min`, `max` | When you want to uniformly sample from a fixed range |
| `lognormal` | `mean`, `std` (in log space) | Positive values, long tail |
| `loguniform` | `min`, `max` (in log space) | Positive values, wide range |
| `linear` | `min`, `max` | Evenly spaced values |
| `loglinear` | `min`, `max` (in log space) | Exponentially spaced |
| `fan_out` | `scale`, `node_fan_outs` | This feature has been built in because our dendrites' inductance values, and therefore $\gamma^+$ values depend on the number of downstream connections, and so this option automates this process |

#### Noise vs Perturbation

![Noise vs Perturbation Diagram](Figures/Building_Models/model_building_fig2.jpg)

> **Tip:** Use `NoiseConfig` for runtime variability, `PerturbationConfig` for fixed per-batch offsets.

---

### 3. ConnectionConfig: Wiring Layers Together

Connections define how information flows between layers.

```python
from soen_toolkit.core import ConnectionConfig

connection = ConnectionConfig(
    from_layer=0,              # Source layer ID
    to_layer=1,                # Target layer ID
    connection_type="dense",   # Connectivity pattern (for backwards compatibility)
    params={},                 # Pattern-specific parameters
    learnable=True,            # Train these weights?
    noise=NoiseConfig(),       # Optional noise on weights
    perturb=PerturbationConfig()  # Optional weight mismatch
)
```

**Parameter Structure:** Connections support two parameter formats for backwards compatibility:

1. **Nested format (recommended):** Separates structure and initialization
```python
params={
    "structure": {
        "type": "dense",
        "params": {"allow_self_connections": True}
    },
    "init": {
        "name": "xavier_uniform",
        "params": {"gain": 1.0}
    }
}
```

2. **Flat format (legacy):** Parameters at the top level
```python
params={
    "init": "xavier_uniform",
    "gain": 1.0
}
```

Both formats work identically. The nested format is clearer for complex configurations.

#### Connection Matrix Shape

```python
# For connection from layer i (N nodes) to layer j (M nodes):
# Connection matrix J has shape: (M, N)
# 
# During forward pass:
# phi_j = s_i @ J.T
# where s_i is (batch, time, N) and phi_j becomes (batch, time, M)
```

> **Important:** Connection matrices are stored in `model.connections` as a `ParameterDict` with keys following the format `"J_{from_layer}_to_{to_layer}"`. For example, `"J_0_to_1"` represents a connection from layer 0 to layer 1, and `"J_1_to_1"` represents an internal (recurrent) connection within layer 1.

---

## Model I/O: Build, Save, Load, and File Formats

This section explains how to build models from specs (specification/architecture files), and how to save/load models in different formats.

### Build from YAML/JSON Spec

Use `SOENModelCore.build` to construct a model from a spec file or a Python dict:

```python
from soen_toolkit.core import SOENModelCore

# YAML or YML (recommended for human editing)
model = SOENModelCore.build("config.yaml")

# Minimal JSON spec in the same schema as YAML
model = SOENModelCore.build("config.json")

# Directly from a dict
spec = {
    "simulation": {"dt": 37, "network_evaluation_method": "layerwise"},
    "layers": [
        {"layer_id": 0, "layer_type": "Linear", "params": {"dim": 10}},
        {"layer_id": 1, "layer_type": "SingleDendrite", "params": {
            "dim": 5, "solver": "FE", "source_func_type": "RateArray",
            "bias_current": 1.7, "gamma_plus": 0.001, "gamma_minus": 0.001
        }}
    ],
    "connections": [
        {"from_layer": 0, "to_layer": 1, "connection_type": "dense"}
    ]
}
model = SOENModelCore.build(spec)
```

- Seed support: If the YAML/JSON spec includes a top-level `seed:` or `simulation.seed`, it will be applied during build and stored on the model for round-tripping.
- JSON detection: If the JSON file is an exported model (see below), it is auto-detected and fully loaded (architecture + weights). If it is a minimal spec (like the dict above), it will be parsed as a spec.

### Save Models

Two common targets depending on your needs:

- Binary full model (.soen or .pth) — includes weights and configuration. Best for training checkpoints and exact resume.
  ```python
  model.save("checkpoint.soen")   # or .pth; identical content
  # Optional metadata can be toggled
  model.save("checkpoint.pth", include_metadata=True)
  ```
  Contains: `state_dict`, `simulation`, `layers_config`, `connections_config`, `connection_masks` (if present), and `dt`/`dt_learnable`.

- Exported JSON (.json) — readable JSON containing configuration and tensors. Good for inspection, interoperability, and diffs.
  ```python
  model.save("model.json")
  ```
  Layout: `simulation`, `layers.{layer_id}.{config|parameters|buffers}`, `connections.{config|matrices|masks}`, plus a `global_matrix` convenience view.

- Config-only YAML (.yaml/.yml) — architecture/spec without weights.
  ```python
  from soen_toolkit.core.model_yaml import dump_model_to_yaml
  dump_model_to_yaml(model, "architecture.yaml")
  ```
  Includes `simulation`, `layers`, `connections`, and (if available) the original `seed` used to build the model.

### Load Models

Use `SOENModelCore.load` for files saved via `model.save` and for exported JSON. It auto-detects by extension.

```python
from soen_toolkit.core import SOENModelCore

# Binary full model
m = SOENModelCore.load("checkpoint.soen")

# Exported JSON (architecture + weights)
m = SOENModelCore.load("model.json")
```

### Choosing a Format

- Use .soen/.pth when you need an exact training checkpoint or the fastest save/load.
- Use exported .json when you want a readable snapshot for inspection, versioning, or tool interop.
- Use YAML spec when you want a compact, editable architecture definition without weights.


---


## Layer Types Catalog

This section has been moved into a dedicated page:

- `building_models/Layer_Types.md`

## Connection Patterns

This section has been moved into a dedicated page:

- `building_models/Connections.md`

---

## Dynamic V2 Connections: Programmable Weights

This section explains how to use **dynamic_v2 connections** to implement multiplier v2 circuit physics. See also the [MultiplierNOCC layer](building_models/Layer_Types.md#multipliernocc-no-collection-coil) section if you want a multiplier circuit as a computational layer instead.

### What are Dynamic Connections?

Dynamic connections implement time-varying, data-dependent weights using superconducting circuit physics. Instead of fixed matrix multiplications, each connection computes its output using coupled differential equations that respond to both the input signal and a programmable weight signal.

**Two versions exist:**

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>WICC</th>
      <th>NOCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Name</strong></td>
      <td>With Collection Coil</td>
      <td>No Collection Coil</td>
    </tr>
    <tr>
      <td><strong>Circuit</strong></td>
      <td>Single-loop multiplier</td>
      <td>Dual-loop multiplier, no collection coil</td>
    </tr>
    <tr>
      <td><strong>Edge States</strong></td>
      <td>One state per edge: $s_{ij}$</td>
      <td>Two states per edge: $s^1_{ij}, s^2_{ij}$</td>
    </tr>
    <tr>
      <td><strong>Output State</strong></td>
      <td>Direct edge state</td>
      <td>Aggregated per-node: $m_i$</td>
    </tr>
    <tr>
      <td><strong>Parameters</strong></td>
      <td><code>gamma_plus</code>, <code>bias_current</code>, <code>j_in</code>, <code>j_out</code></td>
      <td><code>alpha</code>, <code>beta</code>, <code>beta_out</code>, <code>ib</code>, <code>j_in</code>, <code>j_out</code></td>
    </tr>
  </tbody>
</table>

**Coupling Parameters:**
- `j_in`: Input coupling gain that scales the upstream state before entering the multiplier circuit (default: 0.38)
- `j_out`: Output coupling gain that scales the circuit output state (default: 0.38)

### V2 Connection Physics

When you specify `mode="NOCC"`, each edge $(i \to j)$ runs coupled ODEs per timestep:

**Per-edge dynamics:**
$$\beta \dot{s}^1_{ij} = g(\phi_x + \phi_y, i_b - s^1_{ij}) - \beta_{\text{out}} \dot{m}_j - \alpha s^1_{ij}$$
$$\beta \dot{s}^2_{ij} = g(\phi_x - \phi_y, -i_b + s^2_{ij}) - \beta_{\text{out}} \dot{m}_j - \alpha s^2_{ij}$$

**Per-node aggregation:**
$$(\beta + 2N\beta_{\text{out}}) \dot{m}_j = \sum_i (g^1_{ij} + g^2_{ij}) - \alpha m_j$$

where:
- $\phi_x$ = input flux from upstream layer (scaled by `j_in`)
- $\phi_y$ = programmable weight (learned parameter)
- $N$ = fan-in (number of incoming edges) to node $j$
- $m_j$ = aggregated output state for node $j$ (scaled by `j_out` before output)

### Usage Example

#### Spec-Based API (YAML)

```yaml
connections:
  - from_layer: 0
    to_layer: 1
    connection_type: dense
    learnable: true
    mode: NOCC           # Uses defaults
```

#### Spec-Based API (YAML) — Custom Parameters

```yaml
connections:
  - from_layer: 0
    to_layer: 1
    connection_type: dense
    learnable: true
    mode: NOCC
    connection_params:    # New simplified API
      alpha: 1.64053      # Dimensionless resistance
      beta: 303.85        # Inductance of incoming branches
      beta_out: 91.156    # Inductance of output branch
      ib: 2.1             # Bias current
      j_in: 0.38          # Input coupling gain (default: 0.38)
      j_out: 0.38         # Output coupling gain (default: 0.38)
```

#### PyTorch-Style API (Graph) — Simple

```python
from soen_toolkit.nn import Graph, layers, structure, init

g = Graph(dt=37, network_evaluation_method="layerwise")
g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(dim=5, solver="FE", source_func_type="RateArray",
                                      bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3))

# Simple: uses all defaults
g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform(), mode="NOCC")
```

#### PyTorch-Style API (Graph) — Custom Parameters

```python
# Customized: override default physics parameters
g.connect(0, 1, 
          structure=structure.dense(), 
          init=init.uniform(-0.15, 0.15),
          mode="NOCC",
          connection_params={"alpha": 1.5, "beta": 350.0, "beta_out": 100.0, "ib": 2.0})
```

### V2 Parameters Explained

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `alpha` | 1.64053 | > 0 | Dimensionless resistance. Physical: ~2Ω |
| `beta` | 303.85 | > 0 | Inductance of incoming branches. Physical: ~1nH |
| `beta_out` | 91.156 | > 0 | Inductance of output branch. Physical: ~300pH |
| `ib` | 2.1 | > 0 | Bias current (threshold). Physical: ~210μA |
| `j_in` | 0.38 | > 0 | Input coupling gain (scales upstream state) |
| `j_out` | 0.38 | > 0 | Output coupling gain (scales circuit output) |
| `source_func` | "RateArray" | string | Nonlinearity lookup. Options: "RateArray", "Tanh", etc. |

These default values correspond to physical parameters from hardware specifications. Adjust them if your hardware target uses different component values. The coupling gains `j_in` and `j_out` control signal scaling at circuit boundaries.


---

## Connection Weight Initialization

How initial connection weights are set when building the model. Connection matrices are stored in `model.connections` with keys following the format `J_{from_layer}_to_{to_layer}`. For example:
- `J_0_to_1`: Connection from layer 0 to layer 1
- `J_1_to_1`: Internal (recurrent) connection within layer 1

### Supported methods

- **normal**: mean, std
- **uniform**: min, max (aliases `a`/`b` accepted)
- **xavier_normal**: gain
- **xavier_uniform**: gain
- **kaiming_normal**: nonlinearity, a
- **kaiming_uniform**: nonlinearity, a
- **orthogonal**: gain
- **constant**: value
- **linear**: min, max
- **custom**: weights_file (for pre-trained or custom weights)

### Where to specify

You can provide initialization in two equivalent ways:

1) Under a `structure` block (recommended; keeps structure and init together)

```python
ConnectionConfig(
    from_layer=0,
    to_layer=1,
    connection_type="dense",
    params={
        "structure": {"type": "dense", "params": {"allow_self_connections": True}},
        "init": {"name": "uniform", "params": {"min": -0.2, "max": 0.2}},
    },
)
```

2) Flat on `params` (legacy-compatible)

```python
ConnectionConfig(
    from_layer=0,
    to_layer=1,
    connection_type="dense",
    params={
        "init": "xavier_uniform",
        "gain": 1.0,
    },
)
```

### Notes

- The initializer always respects the connection mask built by the chosen structure.
- For `uniform`, you may use `a`/`b` as aliases for `min`/`max`.
- Constraints can be applied via `params["constraints"] = {"min": ..., "max": ...}` and are enforced after initialization and during training steps.
- Learnability (`learnable=True/False`) is respected irrespective of the initializer choice.

### Custom Weights from Files

You can load pre-trained weights or custom weight matrices directly from `.npy` or `.npz` files. This is useful for transfer learning, initializing from external sources, or using hand-crafted weights.

#### File Format

Weights must be stored as a 2D array with shape `[to_nodes, from_nodes]` (matching the connection direction):
- `.npy` files: Direct NumPy binary format
- `.npz` files: Must contain the weights under the key `"weights"`

#### Creating and Saving Weights

```python
import numpy as np
from soen_toolkit.utils.weights_utils import save_weights_to_npy, save_weights_to_npz

# Create custom weights (shape matches your connection)
custom_weights = np.random.randn(5, 10).astype(np.float32)  # 5 targets, 10 sources

# Save as .npy (simplest)
save_weights_to_npy(custom_weights, "my_weights.npy")

# Or save as .npz (with metadata)
save_weights_to_npz(custom_weights, "my_weights.npz", key="weights")
```

#### Using Custom Weights in Models

**With Config API:**

```python
ConnectionConfig(
    from_layer=0,
    to_layer=1,
    connection_type="dense",
    params={
        "init": "custom",
        "weights_file": "path/to/my_weights.npy",
    },
)
```

**With PyTorch-style API:**

```python
from soen_toolkit.nn import Graph, init, structure

g = Graph(dt=37)
g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(dim=5))
g.connect(0, 1,
          structure=structure.dense(),
          init=init.custom_weights("my_weights.npy"))
```

**With YAML:**

```yaml
connections:
  - from: 0
    to: 1
    type: dense
    params:
      init: custom
      weights_file: path/to/my_weights.npy
```

#### Validation and Utilities

```python
from soen_toolkit.utils.weights_utils import (
    load_weights_from_file,
    validate_weight_shape,
)

# Load and validate weights
weights = load_weights_from_file("my_weights.npy", from_nodes=10, to_nodes=5)

# Check shape matches your connection before building
is_valid = validate_weight_shape(weights, from_nodes=10, to_nodes=5)
```

---

## Neuron Polarity Constraints (Dale's Principle)

Enforce excitatory/inhibitory neuron types where each neuron's outgoing connections maintain a consistent sign. This implements Dale's principle, a fundamental constraint in biological neural networks stating that a neuron releases the same neurotransmitters at all of its synapses.

**Key Point:** Polarity is a **layer property** (specified per neuron in the layer), not a connection property. The constraints are automatically applied to all outgoing connections from that layer.

### Polarity Values

- `1`: **Excitatory** - all outgoing weights from this neuron must be ≥ 0
- `-1`: **Inhibitory** - all outgoing weights from this neuron must be ≤ 0
- `0`: **Normal** - unrestricted, any sign allowed

### Polarity Init Options

**Built-in Methods (No File Needed):**
- `"alternating"` or `"50_50"`: Alternating pattern [1, -1, 1, -1, ...]
- `{"excitatory_ratio": 0.8, "seed": 42}`: Random with custom ratio

**Custom File:**
Use the `polarity_utils` module to generate and save custom patterns:

```python
from soen_toolkit.utils.polarity_utils import (
    generate_alternating_polarity,
    generate_random_polarity,
    save_polarity,
)

# Generate custom pattern
polarity = generate_alternating_polarity(num_neurons=100)

# Or random with specific ratio
polarity = generate_random_polarity(
    num_neurons=100,
    excitatory_ratio=0.8,
    seed=42
)

# Save to file
save_polarity(polarity, "layer0_polarity.npy")

# Then use in layer config:
# params={"dim": 100, "polarity_file": "layer0_polarity.npy"}
```

### Using Polarity in Models

**Option 1: Simple Init Method (No File Required)**

For common cases like 50:50 alternating, just use `polarity_init`:

```yaml
layers:
  - layer_id: 0
    layer_type: SingleDendrite
    params:
      dim: 100
      polarity_init: "alternating"  # or "50_50" - generates [1, -1, 1, -1, ...]

  - layer_id: 1
    layer_type: SingleDendrite
    params:
      dim: 50

connections:
  - from_layer: 0
    to_layer: 1
    connection_type: dense
    params:
      init: uniform
      min: -0.5
      max: 0.5
```

**Option 2: Custom File (For Complex Patterns)**

For custom polarity patterns, generate and load from file:

```yaml
layers:
  - layer_id: 0
    layer_type: SingleDendrite
    params:
      dim: 100
      polarity_file: "layer0_polarity.npy"  # Load custom pattern from file

  - layer_id: 1
    layer_type: SingleDendrite
    params:
      dim: 50

connections:
  - from_layer: 0
    to_layer: 1
    connection_type: dense
    params:
      init: uniform
```

**With Config API:**

```python
from soen_toolkit.core.configs import ConnectionConfig, LayerConfig, SimulationConfig
from soen_toolkit.core.soen_model_core import SOENModelCore

# Simple method: Use polarity_init (no file needed!)
sim = SimulationConfig(dt=37)
layers = [
    LayerConfig(
        layer_id=0, 
        layer_type="SingleDendrite", 
        params={"dim": 50, "polarity_init": "alternating"}  # Simple 50:50
    ),
    LayerConfig(
        layer_id=1, 
        layer_type="SingleDendrite", 
        params={"dim": 30}
    ),
]
connections = [
    ConnectionConfig(
        from_layer=0,
        to_layer=1,
        connection_type="dense",
        params={"init": "normal", "std": 0.1},
    ),
]

model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=connections)
```

**Advanced: Random Polarity with Custom Ratio**

```python
# 80% excitatory, 20% inhibitory (biologically realistic)
LayerConfig(
    layer_id=0,
    layer_type="SingleDendrite",
    params={
        "dim": 100,
        "polarity_init": {
            "excitatory_ratio": 0.8,
            "seed": 42  # For reproducibility
        }
    }
)
```

### How It Works

1. **Layer Definition**: Polarity is specified in the layer configuration (one value per neuron)
2. **Build Time**: When connections are built, the system checks if the SOURCE layer has polarity
3. **Constraint Generation**: Per-edge constraint matrices are automatically generated for all outgoing connections
4. **Initialization**: Weights are initialized normally, then constraints are applied
5. **Training**: After each optimizer step, `model.enforce_param_constraints()` clamps weights to satisfy polarity constraints
6. **Zero Overhead**: Constraint matrices are pre-computed once at build time, so runtime cost is minimal

### Important Considerations

**Recurrent Connections:**
When using polarity with recurrent (self-) connections, mixed excitatory/inhibitory neurons will result in many weights being clamped to zero. For example, with 50:50 alternating polarity:
- Excitatory→Excitatory: positive weights ✓
- Inhibitory→Inhibitory: negative weights ✓
- Excitatory→Inhibitory: clamped to 0 (excitatory can't be negative)
- Inhibitory→Excitatory: clamped to 0 (inhibitory can't be positive)

This is expected and reflects biological constraints, but reduces effective connectivity.

**Combining with Scalar Constraints:**
Polarity constraints (from layer) combine with standard min/max constraints (from connection):

```yaml
layers:
  - layer_id: 0
    params:
      dim: 100
      polarity_file: "polarity.npy"  # Polarity in layer

connections:
  - from_layer: 0
    to_layer: 1
    params:
      constraints:
        min: -0.5
        max: 0.5  # Scalar constraint in connection
```

For excitatory neurons in layer 0: effective constraint is `[0.0, 0.5]` (max of mins, min of maxs)

**Incompatibility with Dynamic Weights:**
Neuron polarity is **not compatible** with dynamic weight modes (WICC/NOCC). Attempting to use both will raise an error:

```python
# This will fail:
layers = [
    LayerConfig(layer_id=0, layer_type="SingleDendrite", 
                params={"dim": 10, "polarity_file": "polarity.npy"}),  # Has polarity
    ...
]
connections = [
    ConnectionConfig(
        from_layer=0,
        to_layer=1,
        params={"connection_params": {"mode": "WICC"}},  # ERROR! Dynamic weights incompatible
    ),
]
```

### Biological Context

Dale's principle is named after Henry Dale, who discovered that neurons typically release the same set of neurotransmitters at all synapses. In cortical networks:
- ~80% of neurons are excitatory (glutamatergic)
- ~20% are inhibitory (GABAergic)

This constraint is fundamental to biological neural computation and can improve model interpretability and biological realism.

### Visualization

When visualizing models with polarity using `model.visualize_grid_of_grids()`, neurons are automatically color-coded:
- **Green nodes**: Excitatory neurons (polarity = 1)
- **Red nodes**: Inhibitory neurons (polarity = -1)
- **Default color**: Normal neurons (polarity = 0) or layers without polarity

This makes it easy to visually verify polarity patterns and understand network structure.

### Performance Notes

- Constraint application overhead: <1% of training time
- Constraint matrices are pre-computed at model build time
- Memory overhead: 2 × (to_nodes × from_nodes) × 4 bytes per connection with polarity

---

## Solver Modes Explained

This section has been moved into a smaller page to keep `Building_Models.md` from growing without bound:

- `building_models/Solvers_and_Simulation.md`

For the backend-agnostic overview (Torch + JAX), see:

- `Simulation.md`

---

## Complete Examples

### Example 1: Simple Classifier

Build a 3-layer network for sequence classification.

```python
from soen_toolkit.core import SOENModelCore, SimulationConfig, LayerConfig, ConnectionConfig

# Network: Input (10) → Hidden (50 dendrites) → Output (5 classes)

sim_config = SimulationConfig(dt=37, track_power=True)

layers = [
    LayerConfig(
        layer_id=0,
        layer_type="Linear",
        description="Input features",
        params={"dim": 10}
    ),
    LayerConfig(
        layer_id=1,
        layer_type="SingleDendrite",
        description="Recurrent SOEN layer",
        params={
            "dim": 50,
            "solver": "FE",
            "source_func_type": "RateArray",
            "bias_current": {
                "distribution": "uniform",
                "params": {"min": 1.5, "max": 1.9}
            },
            "gamma_plus": 0.0003,
            "gamma_minus": 0.001,
            "phi_offset": 0.23,
        }
    ),
    LayerConfig(
        layer_id=2,
        layer_type="DendriteReadout",
        description="Classification logits",
        params={
            "dim": 5,
            "source_func_type": "RateArray",
            "bias_current": 1.7
        }
    )
]

connections = [
    ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense"),
    ConnectionConfig(from_layer=1, to_layer=2, connection_type="dense")
]

model = SOENModelCore(sim_config, layers, connections)
model.save("classifier.pth")

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

![Model diagram 1](Figures/Building_Models/model_building_fig4.jpg)

---

### Example 2: Recurrent Network with Feedback Loops

Create a network with backward connections using stepwise solver.

```python
from soen_toolkit.core import NoiseConfig

sim_config = SimulationConfig(
    dt=37,
    network_evaluation_method="stepwise_gauss_seidel",  # Required for feedback
    early_stopping_forward_pass=True,       # Stop at steady state
    steady_window_min=50,
    steady_tol_abs=1e-5,
)

layers = [
    LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 10}),
    LayerConfig(
        layer_id=1,
        layer_type="SingleDendrite",
        params={
            "dim": 20,
            "solver": "FE",
            "source_func_type": "RateArray",
            "bias_current": 1.7,
            "gamma_plus": 0.001,
            "gamma_minus": 0.001,
        }
    ),
    LayerConfig(
        layer_id=2,
        layer_type="SingleDendrite",
        params={
            "dim": 15,
            "solver": "FE",
            "source_func_type": "RateArray",
            "bias_current": 1.7,
            "gamma_plus": 0.001,
            "gamma_minus": 0.001,
        }
    ),
]

connections = [
    # Feedforward path
    ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense"),
    ConnectionConfig(from_layer=1, to_layer=2, connection_type="dense"),
    
    # Feedback connection (requires stepwise solver)
    ConnectionConfig(
        from_layer=2,
        to_layer=1,
        connection_type="sparse",
        params={"sparsity": 0.3}
    ),
]

model = SOENModelCore(sim_config, layers, connections)
```

![Model diagram 3](Figures/Building_Models/model_building_fig6.jpg)

> **Important:** Feedback connections (higher layer_id → lower layer_id) require `network_evaluation_method="stepwise_gauss_seidel"` or `"stepwise_jacobi"`.

---

## Troubleshooting

### Issue 1: Model won't build

**Error:** `ValueError: Unsupported layer type 'X'`

**Solution:** Check available layer types:

```python
from soen_toolkit.core.layer_registry import LAYER_TYPE_MAP
print(list(LAYER_TYPE_MAP.keys()))
```

Available types:
- Physical: `"SingleDendrite"`, `"Multiplier"`, `"DendriteReadout"`, `"Readout"`
- Virtual: `"Linear"`, `"Input"`, `"ScalingLayer"`, `"NonLinear"`, `"RNN"`, `"LSTM"`, `"GRU"`, `"MinGRU"`


---

### Issue 3: NaN or exploding values

**Possible causes:**
1. `dt` too large for dynamics
2. `gamma_plus` or `gamma_minus` too large
3. If using soen-type layers and you chose to learn the layer-specific parameters such as $\gamma^{+}$ or $\gamma^{-}$ make sure to set constraints to force an upper bound on them.

---


**Debugging Tip:** Enable tracking to inspect internal states:

- Note that if you use the model creation gui, you do not have to worry about any manual tracking flags, just navigate to the 'Analyse' tab, and then 'Plot State Trajectory'. We will cover this in more depth in the GUI documentation. 

```python
sim_config = SimulationConfig(
    track_phi=True,    # Track inputs to each layer
    track_g=True,      # Track source function values
    track_s=True,      # Track state trajectories
    track_power=True,  # Track power consumption (SingleDendrite only)
)
# to access the histories that were tracked
output, histories = model(input_seq)
phi_hist = model.get_phi_history()    # List of tensors, one per layer
g_hist = model.get_g_history()        # List of tensors, one per layer
power_hist = model.get_power_history()  # List of total power tensors, one per layer
```

---

## Next Steps

Now that you can build models:

1. **Training Models** → [Training_Models](Training_Models.md) - Learn how to train your network
2. **GUI Tools** → [GUI_Tools](GUI_Tools.md) - Explore visual interfaces for model creation
3. **Advanced Features** → [Advanced_Features](Advanced_Features.md) - Custom layers, quantization, robustness

---


<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Getting_Started.md" style="margin-right: 2em;">&#8592; Previous: Getting Started</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Training_Models.md" style="margin-left: 2em;">Next: Training Models &#8594;</a>
</div>