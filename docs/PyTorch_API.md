---
layout: default
title: PyTorch-Style API (Graph & Sequential)
---


# PyTorch-Style API (Graph & Sequential)

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="index.md" class="nav-home">Home</a>
</div>

<span style="color: red; font-weight: bold;">**New feature and still in development.**</span>

This guide covers the imperative, PyTorch-native API for building SOEN models. If you're coming from a PyTorch background, this API will feel familiar and intuitive.

> **Note:** This API is fully compatible with the spec-based (YAML/config) approach. Under the hood, it compiles to the same `SimulationConfig`, `LayerConfig`, and `ConnectionConfig` objects. Choose whichever style suits your workflow.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Why Use This API?](#why-use-this-api)
- [Graph: Imperative Model Builder](#graph-imperative-model-builder)
- [Sequential: Feedforward Networks](#sequential-feedforward-networks)
- [Complete Examples](#complete-examples)
- [API Reference](#api-reference)
- [Migration from Spec-Based API](#migration-from-spec-based-api)

---

## Quick Start

**Simple feedforward network:**

```python
from soen_toolkit.nn import Sequential, layers

net = Sequential([
    layers.Linear(dim=10),
    layers.SingleDendrite(dim=50), # uses defaults under the hood
    layers.NonLinear(dim=5)
])

# Standard PyTorch usage
import torch
x = torch.randn(2, 100, 10)
output = net(x)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
```

**Custom graph with specific connections:**

```python
from soen_toolkit.nn import Graph, layers, init, structure

g = Graph(dt=37, network_evaluation_method="layerwise")
g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(
    dim=5, solver="FE", source_func_type="RateArray",
    bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
))
g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

output = g(x)
```

---

## Why Use This API?

### PyTorch-Native Experience
- Subclass `nn.Module` → works with standard PyTorch tools
- `model.parameters()` → works with any optimizer
- `forward(x)` → clean interface, returns output only
- `.to(device)`, `.train()`, `.eval()` → standard PyTorch patterns

### Code-First Workflow
- Build models in Python, not YAML (though YAML still supported)
- IDE autocomplete and type hints
- Easier to debug and iterate
- Natural for researchers coming from PyTorch

### Near Equal Feature Support to SOENModelCore objects
- All layer types supported
- All connection patterns (dense, sparse, power-law, etc.)
- Fixed and dynamic connections
- Constraints, learnability, masks
- Histories and tracking
- Save/load compatible
- However, currently, not all of the SOENModelCore methods have been migrated across to this API


---

## Graph: Imperative Model Builder

`Graph` is the main container for building SOEN networks imperatively. You add layers, define connections, and call `forward(x)`.

### Basic Usage

```python
from soen_toolkit.nn import Graph, layers, init, structure

# Create graph with simulation config
g = Graph(
    dt=37,
    network_evaluation_method="layerwise",
    track_phi=False,
    track_s=False
)

# Add layers
g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(
    dim=5, solver="FE", source_func_type="RateArray",
    bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
))

# Define connections
g.connect(
    from_layer=0,
    to_layer=1,
    structure=structure.dense(),
    init=init.xavier_uniform(gain=1.0)
)

# Forward pass
x = torch.randn(2, 100, 10)
output = g(x)
```

### What g.compile() does

`g.compile()` converts the Graph's specifications into a runnable model and registers parameters with PyTorch. Concretely, it:

- Converts the Graph specs to core configs: builds `SimulationConfig`, one `LayerConfig` per layer, and one `ConnectionConfig` per connection
- Initializes all connection weights using the requested initializer and applies masks/structure
- Validates dimensions and mask shapes; raises informative errors if mismatched
- Builds the underlying `SOENModelCore` and registers all parameters/buffers so `g.parameters()` is complete
- Applies learnability flags and prepares constraint enforcement

You usually do not need to call `compile()` manually because several methods auto-compile on first use:

- `g(x)` (forward) — auto-compiles if needed
- `g.summary()` — auto-compiles if needed
- `g.visualize(...)` and `g.visualize_grid_of_grids(...)` — auto-compile if needed
- `g.compute_summary()` — auto-compiles if needed

If a graph hasn't been compiled yet, string representations may show `Graph(layers=..., connections=..., not compiled)`. This is informational; the first forward/summary/visualize call will compile it automatically.

### Using MultiplierNOCC Layers

Add multiplier v2 circuit layers to your model when you need multiplier computation nodes (not just programmable weights on connections). A MultiplierNOCC layer can integrate two input fluxes and produce an output that approximates their product.

**When to use MultiplierNOCCLayer vs NOCC connections:**

| Use Case | Choice |
|----------|--------|
| Want multiplier **nodes** in the network | `layers.MultiplierNOCC(...)` |
| Want programmable **weights** between layers | `mode="NOCC"` connection (or legacy `"dynamic_v2"`) |
| Both: multiplier nodes AND v2-physics connections | Use both together |

**Example: MultiplierNOCC Layer as a computational node:**

```python
from soen_toolkit.nn import Graph, layers, structure, init, dynamic_v2

g = Graph(dt=37, network_evaluation_method="layerwise")

# Input layer
g.add_layer(0, layers.Linear(dim=10))

# Multiplier v2 computational layer (with default parameters)
g.add_layer(1, layers.MultiplierNOCC(dim=20, solver="FE", source_func_type="RateArray"))

# Output layer
g.add_layer(2, layers.DendriteReadout(dim=5, source_func_type="RateArray", bias_current=1.7))

# Connect input to multiplier with fixed weights
g.connect(0, 1, structure=structure.dense(), init=init.uniform(-0.15, 0.15))

# Connect multiplier to output with NOCC (v2 physics, uses defaults)
g.connect(1, 2, structure=structure.dense(), init=init.uniform(-0.15, 0.15),
          mode="NOCC", dynamic=dynamic_v2())  # Can also use legacy "dynamic_v2"

output = g(x)
```

**Customizing MultiplierNOCC Parameters:**

```python
# Override default parameters if needed
g.add_layer(1, layers.MultiplierNOCC(
    dim=20,
    solver="FE",
    source_func_type="RateArray",
    phi_y=0.15,                  # Secondary input/weight term (default: 0.1)
    ib=2.0,                      # Bias current (default: 2.1)
    alpha=1.5,                   # Dimensionless resistance (default: 1.64053)
    beta=350.0,                  # Inductance of incoming branches (default: 303.85)
    beta_out=100.0               # Inductance of output branch (default: 91.156)
))

# Optionally customize NOCC connection parameters too
g.connect(1, 2, structure=structure.dense(), init=init.uniform(-0.15, 0.15),
          mode="NOCC",  # Can also use legacy "dynamic_v2"
          dynamic=dynamic_v2(alpha=1.5, beta=350.0, beta_out=100.0, ib=2.0))
```

**Key Parameters for MultiplierNOCCLayer:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `dim` | — | Number of multiplier nodes |
| `solver` | "FE" | Integration method (only FE supported) |
| `source_func_type` | "RateArray" | Nonlinearity; typically "RateArray" |
| `phi_y` | 0.1 | Secondary input flux (weight term) |
| `ib` | 2.1 | Bias current |
| `alpha` | 1.64053 | Dimensionless resistance |
| `beta` | 303.85 | Inductance of incoming branches |
| `beta_out` | 91.156 | Inductance of output branch |

For detailed physics and parameter guidance, see [Building_Models.md: MultiplierNOCC Layer](Building_Models.md#multiplierv2).

---

### Connection Options

All connection patterns from the spec-based API are available:

```python
# Dense connectivity
g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

# Sparse connectivity
g.connect(0, 1, 
          structure=structure.sparse(sparsity=0.3),
          init=init.uniform(-0.2, 0.2))

# Block structure
g.connect(0, 1,
          structure=structure.block_structure(
              block_count=4,
              connection_mode="diagonal",
              within_block_density=0.8,
              cross_block_density=0.1
          ),
          init=init.xavier_uniform())

# Power law (distance-dependent)
g.connect(0, 1,
          structure=structure.power_law(alpha=2.0, expected_fan_out=4),
          init=init.normal(mean=0.0, std=0.1))

# Custom connectivity from file
g.connect(0, 1,
          structure=structure.custom(mask_file="my_mask.npz"),
          init=init.xavier_uniform())
```

### Weight Initialization

```python
from soen_toolkit.nn import init

# Available initializers
init.normal(mean=0.0, std=0.1)
init.uniform(min=-0.24, max=0.24)
init.linear(min=0.0, max=1.0)
init.xavier_normal(gain=1.0)
init.xavier_uniform(gain=1.0)
init.kaiming_normal(nonlinearity="relu", a=0.0)
init.kaiming_uniform(nonlinearity="relu", a=0.0)
init.orthogonal(gain=1.0)
init.constant(value=1.0)
```

### Dynamic Connections

Enable multiplier-based programmable weights using circuit physics. Three options:

**Fixed Weights (Default):**

```python
# No dynamic= needed, mode defaults to "fixed"
g.connect(
    0, 1,
    structure=structure.dense(),
    init=init.uniform(-0.15, 0.15)
)
```

**WICC - With Collection Coil (V1, Faster, Traditional Multiplier):**

```python
g.connect(
    0, 1,
    structure=structure.dense(),
    init=init.uniform(-0.15, 0.15),
    mode="WICC"  # Uses defaults: gamma_plus=0.001, bias_current=2.0
)
```

**NOCC - No Collection Coil (V2, Hardware-Compatible):**

```python
g.connect(
    0, 1,
    structure=structure.dense(),
    init=init.uniform(-0.15, 0.15),
    mode="NOCC"  # Uses defaults
)
```

**Customizing Parameters:**

```python
# For any dynamic mode, customize parameters as needed
g.connect(
    0, 1,
    structure=structure.dense(),
    init=init.uniform(-0.15, 0.15),
    mode="NOCC",
    connection_params={
        "alpha": 1.5,          # Adjust resistance
        "beta": 350.0,         # Adjust incoming inductance
        "beta_out": 100.0,     # Adjust output inductance
        "ib": 2.0              # Adjust bias current
    }
)
```

**Quick Comparison:**

| Connection Type | Settling Time | Use Case | Parameters | Mode Name |
|-----------------|---------------|----------|------------|-----------|
| **Fixed** | Instant | Standard weights | None | `"fixed"` |
| **WICC (V1)** | ~10ns | Fast programmable weights | `gamma_plus`, `bias_current` | `"WICC"` |
| **NOCC (V2)** | ~30ns | Hardware v2 design, no collection coil | `alpha`, `beta`, `beta_out`, `ib` | `"NOCC"` |

See [Building_Models.md: Dynamic V2 Connections](Building_Models.md#dynamic-v2-connections-programmable-weights) for detailed physics and parameter tuning.

> **Note:** Use `mode="WICC"` or `mode="NOCC"` with optional `connection_params` dict to customize parameters. Legacy `dynamic()` and `dynamic_v2()` helper functions are deprecated but still supported for backward compatibility.

### Recurrent/Feedback Networks

Use stepwise solvers for feedback connections:

```python
g = Graph(dt=37, network_evaluation_method="stepwise_gauss_seidel")

g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(...))
g.add_layer(2, layers.SingleDendrite(...))

# Forward connections
g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
g.connect(1, 2, structure=structure.dense(), init=init.xavier_uniform())

# Feedback connection
g.connect(2, 1, structure=structure.sparse(0.3), init=init.uniform(-0.1, 0.1))
```

### History Tracking

Enable tracking and access histories after forward pass:

```python
g = Graph(dt=37, track_phi=True, track_s=True, track_power=True)
# ... add layers and connections ...

output = g(x)

# Access histories
phi_hist = g.phi_history  # List of tensors, one per layer
s_hist = g.s_history
power_hist = g.power_history
```

### Standard PyTorch Integration

```python
# Works with optimizers
optimizer = torch.optim.Adam(g.parameters(), lr=1e-3)

# Move to device
g = g.to("cuda")

# Save and load
g.save("model.soen")
loaded_g = Graph.load("model.soen")

# State dict
state = g.state_dict()
g.load_state_dict(state)

# From YAML (interop with spec-based API)
g = Graph.from_yaml("config.yaml")
```

---

## Sequential: Feedforward Networks

`Sequential` is a convenience wrapper for simple feedforward networks. It automatically creates dense connections between consecutive layers.

### Basic Usage

```python
from soen_toolkit.nn import Sequential, layers

net = Sequential([
    layers.Linear(dim=10),
    layers.SingleDendrite(dim=50, solver="FE", source_func_type="RateArray",
                         bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3),
    layers.DendriteReadout(dim=5, source_func_type="RateArray", bias_current=1.7)
])

output = net(x)
```

### Appending Layers

```python
net = Sequential([layers.Linear(dim=10)])

net.append(layers.SingleDendrite(
    dim=5, solver="FE", source_func_type="RateArray",
    bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
))

output = net(x)
```

### Custom Initialization

Override default connection init:

```python
from soen_toolkit.nn import init

net = Sequential(
    [
        layers.Linear(dim=10),
        layers.SingleDendrite(...)
    ],
    connection_init=init.xavier_uniform(gain=1.0)
)
```

### Manual Connection Control

Disable auto-connect and add connections manually:

```python
from soen_toolkit.nn import structure, init

net = Sequential(
    [layers.Linear(dim=10), layers.SingleDendrite(...)],
    auto_connect=False
)

net.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
```

---

## Complete Examples

### Example 1: Simple Classifier

```python
from soen_toolkit.nn import Sequential, layers
import torch
import torch.nn as nn

# Build model
net = Sequential([
    layers.Linear(dim=784),
    layers.SingleDendrite(
        dim=128, solver="FE", source_func_type="RateArray",
        bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
    ),
    layers.DendriteReadout(dim=10, source_func_type="RateArray", bias_current=1.7)
])

# Training loop
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        
        output = net(x_batch)  # (batch, time, classes)
        logits = output[:, -1, :]  # Final timestep
        
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
```

### Example 2: Hybrid SOEN + LSTM

```python
from soen_toolkit.nn import Graph, layers, init, structure

g = Graph(dt=37, network_evaluation_method="layerwise")

g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.LSTM(dim=32))
g.add_layer(2, layers.SingleDendrite(
    dim=5, solver="FE", source_func_type="RateArray",
    bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
))

g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
g.connect(1, 2, structure=structure.dense(), init=init.xavier_uniform())

output = g(x)
```

### Example 3: Model with Dynamic Connections

```python
from soen_toolkit.nn import Graph, layers, init, structure, dynamic

g = Graph(dt=37, network_evaluation_method="layerwise")

g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(
    dim=5, solver="FE", source_func_type="RateArray",
    bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
))

g.connect(
    0, 1,
    structure=structure.sparse(0.5),
    init=init.uniform(-0.15, 0.15),
    mode="WICC",  # Canonical name (legacy "dynamic" also works)
    dynamic=dynamic(source_func="RateArray", gamma_plus=1e-3, bias_current=2.0)
)

output = g(x)
```

---

## API Reference

### Layers

All layer types from `soen_toolkit.nn.layers`:

- `Linear(dim)` - Passthrough layer
- `SingleDendrite(dim, solver, source_func_type, bias_current, gamma_plus, gamma_minus, ...)` - SOEN dendrite
- `MultiplierWICC(dim, solver, source_func_type, phi_y, bias_current, gamma_plus)` - WICC multiplier circuit (FE only, legacy: `Multiplier`)
- `MultiplierNOCC(dim, solver, source_func_type, phi_y, ib, alpha, beta, beta_out)` - Multiplier V2 circuit (FE only)
- `DendriteReadout(dim, source_func_type, bias_current, phi_offset)` - Readout layer
- `NonLinear(dim, source_func_type, phi_offset, bias_current)` - Static nonlinearity
- `ScalingLayer(dim, scale_factor)` - Learnable scaling
- `RNN(dim)`, `LSTM(dim)`, `GRU(dim)`, `MinGRU(dim)` - Standard RNN variants

### Structure (Connectivity)

From `soen_toolkit.nn.structure`:

- `dense()` - All-to-all
- `one_to_one()` - Diagonal (requires same dimensions)
- `sparse(sparsity)` - Random connections with probability
- `block_structure(block_count, connection_mode, within_block_density, cross_block_density)` - Block-based
- `power_law(alpha, expected_fan_out)` - Distance-dependent with power-law
- `exponential(d_0, expected_fan_out)` - Distance-dependent with exponential
- `constant_fan_out(expected_fan_out)` - Fixed number of targets per source
- `custom(mask_file, npz_key)` - Load connectivity from .npz file

### Initialization

From `soen_toolkit.nn.init`:

- `normal(mean, std)`
- `uniform(min, max)`
- `xavier_normal(gain)`, `xavier_uniform(gain)`
- `kaiming_normal(nonlinearity, a)`, `kaiming_uniform(nonlinearity, a)`
- `orthogonal(gain)`
- `constant(value)`

### Dynamic Connections

From `soen_toolkit.nn.dynamic`:

- `dynamic(source_func, gamma_plus, bias_current)` - WICC (With Collection Coil) - V1 multiplier physics
- `dynamic_v2(source_func, alpha, beta, beta_out, ib)` - NOCC (No Collection Coil) - V2 multiplier physics

Mode names: Use canonical `"WICC"` or `"NOCC"`. Legacy names (`"dynamic"`, `"dynamic_v1"`, `"dynamic_v2"`, etc.) still work.

---

## Migration from Spec-Based API

### Before (YAML/Config)

```python
from soen_toolkit.core import SOENModelCore, SimulationConfig, LayerConfig, ConnectionConfig

sim = SimulationConfig(dt=37, network_evaluation_method="layerwise")

layers = [
    LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 10}),
    LayerConfig(layer_id=1, layer_type="SingleDendrite", params={
        "dim": 5, "solver": "FE", "source_func_type": "RateArray",
        "bias_current": 1.7, "gamma_plus": 1e-3, "gamma_minus": 1e-3
    })
]

connections = [
    ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={
        "init": {"name": "xavier_uniform", "params": {"gain": 1.0}}
    })
]

model = SOENModelCore(sim, layers, connections)
```

### After (PyTorch-Style)

```python
from soen_toolkit.nn import Graph, layers, init, structure

g = Graph(dt=37, network_evaluation_method="layerwise")
g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(
    dim=5, solver="FE", source_func_type="RateArray",
    bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
))
g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform(gain=1.0))
```

### Interoperability

You can mix both approaches:

```python
# Load spec-based model
g = Graph.from_yaml("config.yaml")

# Use with PyTorch tools
optimizer = torch.optim.Adam(g.parameters(), lr=1e-3)

# Save back
g.save("model.soen")
```

---

---

## Fine-Grained Parameter Control

The PyTorch API supports three levels of parameter control:

### 1. Distributions

Instead of scalar values, specify distributions for layer parameters:

```python
from soen_toolkit.nn import param_specs

g.add_layer(1, layers.SingleDendrite(
    dim=50,
    solver="FE",
    source_func_type="RateArray",
    # Uniform distribution
    bias_current=param_specs.uniform(1.5, 2.0),
    # Log-normal distribution (good for strictly positive parameters)
    gamma_plus=param_specs.lognormal(mean=-6.9, std=0.2),
    # Normal distribution
    gamma_minus=param_specs.normal(0.001, 0.0002),
    # Constant value
    phi_offset=0.23
))
```

**Available distributions:**
- `param_specs.uniform(min, max)` - Uniform distribution
- `param_specs.normal(mean, std)` - Normal/Gaussian distribution
- `param_specs.lognormal(mean, std)` - Log-normal distribution (in log space)
- `param_specs.constant(value)` - Constant value for all nodes
- `param_specs.linear(min, max)` - Linearly spaced values across nodes
- `param_specs.loglinear(min, max)` - Exponentially spaced values (in log space)
- `param_specs.loguniform(min, max)` - Log-uniform distribution (in log space)

### 2. Learnability Control

Control which parameters are trainable:

```python
# Method 1: Via ParamSpec
g.add_layer(1, layers.SingleDendrite(
    dim=50,
    solver="FE",
    source_func_type="RateArray",
    bias_current=param_specs.uniform(1.5, 2.0, learnable=True),
    gamma_plus=param_specs.constant(0.001, learnable=False),
    gamma_minus=0.001,
    phi_offset=0.23
))

# Method 2: Via add_layer (recommended for explicit control)
g.add_layer(
    1,
    layers.SingleDendrite(
        dim=50,
        solver="FE",
        source_func_type="RateArray",
        bias_current=1.7,
        gamma_plus=0.001,
        gamma_minus=0.001,
        phi_offset=0.23
    ),
    learnable_params={
        "bias_current": True,   # Train bias_current
        "gamma_plus": True,     # Train gamma_plus
        "gamma_minus": False,   # Freeze gamma_minus
        "phi_offset": False     # Freeze phi_offset
    }
)
```

**Note:** Method 2 takes precedence over Method 1 if both are specified.

### 3. Constraints

Enforce min/max bounds during training:

```python
g.add_layer(1, layers.SingleDendrite(
    dim=50,
    solver="FE",
    source_func_type="RateArray",
    bias_current=param_specs.uniform(
        1.5, 2.0,
        learnable=True,
        constraints={"min": 0.0, "max": 5.0}  # Enforced during training
    ),
    gamma_plus=param_specs.lognormal(
        -6.9, 0.2,
        learnable=True,
        constraints={"min": 0.0, "max": 0.01}  # Keep positive and bounded
    ),
    gamma_minus=0.001,
    phi_offset=0.23
))
```

**Constraints are automatically enforced** by the core model after each parameter update.

### Complete Example

Combining all three features:

```python
from soen_toolkit import nn

g = nn.Graph(dt=37, network_evaluation_method="layerwise")

g.add_layer(0, nn.layers.Linear(dim=10))

g.add_layer(
    1,
    nn.layers.SingleDendrite(
        dim=50,
        solver="FE",
        source_func_type="RateArray",
        # Distribution + learnability + constraints
        bias_current=nn.param_specs.uniform(
            1.5, 2.0,
            learnable=True,
            constraints={"min": 0.0, "max": 5.0}
        ),
        # Log-normal for strictly positive parameters
        gamma_plus=nn.param_specs.lognormal(
            -6.9, 0.2,
            learnable=True,
            constraints={"min": 0.0, "max": 0.01}
        ),
        # Frozen parameter
        gamma_minus=nn.param_specs.constant(0.001, learnable=False),
        phi_offset=0.23
    )
)

g.connect(0, 1, structure=nn.structure.dense(), init=nn.init.xavier_uniform())

# Train normally
optimizer = torch.optim.Adam(g.parameters(), lr=1e-3)
# ... training loop ...
# Constraints are automatically enforced after each optimizer step
```

### Old Dict Syntax (Still Supported)

The original dict syntax still works:

```python
g.add_layer(1, layers.SingleDendrite(
    dim=50,
    solver="FE",
    source_func_type="RateArray",
    bias_current={
        "distribution": "uniform",
        "params": {"min": 1.5, "max": 2.0},
        "learnable": True,
        "constraints": {"min": 0.0, "max": 5.0}
    },
    gamma_plus=0.001,
    gamma_minus=0.001,
    phi_offset=0.23
))
```

But the `param_specs` helpers are recommended for better readability and IDE autocomplete.

---

## Full Access to SOENModelCore Features

`Graph` and `Sequential` are **full wrappers** around `SOENModelCore`, giving you access to all features:

### Visualization

```python
# Generate network diagram
g.visualize(
    save_path="my_network",
    file_format="svg",  # or "png", "pdf", "jpg"
    orientation="LR",   # or "TB" for top-bottom
    simple_view=True
)
```

### Model Summary

```python
# Print summary to console
g.summary()

# Get summary as pandas DataFrame
df = g.summary(return_df=True, print_summary=False)

# Get summary statistics as dict
stats = g.compute_summary()
print(f"Total parameters: {stats['total_parameters']}")
```

### Save and Load

```python
# Save in different formats
g.save("model.soen")   # Binary format (recommended)
g.save("model.pth")    # PyTorch format
g.save("model.json")   # JSON format (human-readable)

# Load
g_loaded = Graph.load("model.soen")
```

### State Dict Management

```python
# Save/load state dict
state = g.state_dict()
torch.save(state, "weights.pth")

# Load into new model
state = torch.load("weights.pth")
g.load_state_dict(state)
```

### Device Management

```python
# Move to GPU
g_cuda = g.to('cuda')

# Move to MPS (Apple Silicon)
g_mps = g.to('mps')

# Move to CPU
g_cpu = g.to('cpu')

# Mixed precision
g_half = g.to(dtype=torch.float16)
```

### Direct Core Access

For advanced use, access the underlying `SOENModelCore`:

```python
# Access compiled core
core = g._compiled_core

# Access layers directly
for layer in core.layers:
    print(type(layer).__name__)

# Access connection matrices
for name, weight in core.connections.items():
    print(f"{name}: {weight.shape}")

# Use any SOENModelCore method
core.enforce_param_constraints()
core.reset_states()
# ... etc
```

### Example: Complete Workflow

```python
from soen_toolkit import nn
import torch

# Build model
g = nn.Graph(dt=37, network_evaluation_method="layerwise")
g.add_layer(0, nn.layers.Linear(dim=10))
g.add_layer(1, nn.layers.SingleDendrite(
    dim=50,
    solver="FE",
    source_func_type="RateArray",
    bias_current=nn.param_specs.uniform(1.5, 2.0, learnable=True)
))
g.connect(0, 1, structure=nn.structure.dense(), init=nn.init.xavier_uniform())

# Visualize architecture
g.visualize(save_path="my_model", file_format="png")

# Print summary
g.summary()

# Train
g.compile()  # Ensure parameters are registered
optimizer = torch.optim.Adam(g.parameters(), lr=1e-3)

for epoch in range(10):
    optimizer.zero_grad()
    output = g(x_train)
    loss = criterion(output[:, -1, :], y_train)
    loss.backward()
    optimizer.step()
    
    # Save checkpoint
    if (epoch + 1) % 5 == 0:
        g.save(f"checkpoint_epoch_{epoch+1}.soen")

# Final save
g.save("final_model.soen")

# Later: Load and evaluate
g_loaded = nn.Graph.load("final_model.soen")
g_loaded.eval()
with torch.no_grad():
    output = g_loaded(x_test)
```

**See `examples_features.py` for more comprehensive examples.**

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="index.md" class="nav-home">Home</a>
</div>
 
