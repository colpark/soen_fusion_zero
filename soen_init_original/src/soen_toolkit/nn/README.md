# SOEN Toolkit PyTorch API (`soen_toolkit.nn`)

This package provides a PyTorch-native API for building SOEN models, offering an imperative, `nn.Module`-based interface as an alternative to the config/YAML-based approach.

## Quick Start

```python
from soen_toolkit import nn
import torch

# Build a model using Graph
g = nn.Graph(dt=37, network_evaluation_method="layerwise")
g.add_layer(0, nn.layers.Linear(dim=10))
g.add_layer(1, nn.layers.SingleDendrite(
    dim=50, solver="FE", source_func_type="RateArray",
    bias_current=1.7, gamma_plus=0.001, gamma_minus=0.001
))
g.connect(0, 1, structure=nn.structure.dense(), init=nn.init.xavier_uniform())

# Or use Sequential for simple feedforward networks
net = nn.Sequential([
    nn.layers.Linear(dim=10),
    nn.layers.SingleDendrite(dim=50, ...),
    nn.layers.NonLinear(dim=5, source_func_type="Tanh")
])

# Use like any PyTorch model
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
output = net(input_tensor)
```

## Features

### ✅ Full Feature Parity with SOENModelCore

The PyTorch API provides **complete access** to all SOENModelCore features:

| Feature | Available | Method/API |
|---------|-----------|------------|
| **Model Building** | ✅ | `Graph`, `Sequential` |
| **Parameters** | ✅ | `.parameters()`, `.named_parameters()` |
| **Training** | ✅ | Standard PyTorch (any optimizer/loss) |
| **Device Management** | ✅ | `.to('cuda')`, `.to('mps')`, etc. |
| **Visualization** | ✅ | `.visualize()` |
| **Summary** | ✅ | `.summary()`, `.compute_summary()` |
| **Save/Load** | ✅ | `.save()`, `Graph.load()` |
| **State Dict** | ✅ | `.state_dict()`, `.load_state_dict()` |
| **Distributions** | ✅ | `param_specs.uniform()`, etc. |
| **Learnability Control** | ✅ | `learnable_params={...}` |
| **Constraints** | ✅ | `param_specs.*(constraints={...})` |
| **Dynamic Connections** | ✅ | `mode="WICC"/"NOCC"`, `dynamic=...` |
| **Connection Patterns** | ✅ | `structure.dense()`, `structure.sparse()`, etc. |
| **Weight Init** | ✅ | `init.xavier_uniform()`, etc. |

### 🎯 Key Advantages

1. **PyTorch-Native**: Subclasses `nn.Module`, works with standard PyTorch tools
2. **Imperative**: Build models programmatically, not just from configs
3. **IDE-Friendly**: Full autocomplete and type hints
4. **Modular**: No changes to core code, purely additive wrapper
5. **Compatible**: Can convert to/from config-based models

## API Components

### Core Classes

- **`Graph`**: Flexible graph container for custom architectures
- **`Sequential`**: Convenience wrapper for feedforward networks

### Helper Namespaces

- **`layers`**: Layer factories (`SingleDendrite`, `Linear`, `NonLinear`, etc.)
- **`init`**: Weight initialization (`xavier_uniform`, `kaiming_normal`, etc.)
- **`structure`**: Connection patterns (`dense`, `sparse`, etc.)
- **`param_specs`**: Parameter distributions and control
- **`dynamic`**: Dynamic connection helpers

## Examples

### 1. Basic Usage

```python
from soen_toolkit.nn import Graph, layers, structure, init

g = Graph(dt=37, network_evaluation_method="layerwise")
g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(dim=50, ...))
g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

output = g(input_tensor)
```

### 2. Fine-Grained Parameter Control

```python
from soen_toolkit.nn import param_specs

g.add_layer(1, layers.SingleDendrite(
    dim=50,
    bias_current=param_specs.uniform(1.5, 2.0, learnable=True, 
                                     constraints={"min": 0.0, "max": 5.0}),
    gamma_plus=param_specs.lognormal(-6.9, 0.2, learnable=True),
    gamma_minus=param_specs.constant(0.001, learnable=False)
))
```

### 3. Visualization and Summary

```python
# Visualize network
g.visualize(save_path="my_network", file_format="svg")

# Print summary
g.summary()

# Get statistics
stats = g.compute_summary()
print(f"Total parameters: {stats['total_parameters']:,}")
```

### 4. Save and Load

```python
# Save in different formats
g.save("model.soen")   # Binary (recommended)
g.save("model.json")   # JSON (human-readable)

# Load
g_loaded = Graph.load("model.soen")
```

### 5. Training

```python
# Standard PyTorch training loop
g.compile()  # Ensure parameters are registered
optimizer = torch.optim.Adam(g.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = g(x_train)
    loss = criterion(output[:, -1, :], y_train)
    loss.backward()
    optimizer.step()
```

## Example Files

- **`examples.py`**: Basic usage of Graph and Sequential
- **`examples_advanced.py`**: Fine-grained parameter control
- **`examples_features.py`**: All SOENModelCore features
- **`FEATURES.md`**: Complete feature comparison table

## Tests

All functionality is tested:
- `test_graph.py`: Graph class tests
- `test_sequential.py`: Sequential class tests
- `test_param_specs.py`: Parameter control tests
- `test_features.py`: Feature parity verification

Run with: `pytest src/soen_toolkit/nn/tests/`

## Documentation

See [`docs/04a_PyTorch_API.md`](../../docs/04a_PyTorch_API.md) for complete documentation with examples.

## Implementation Details

- **Zero Core Changes**: The PyTorch API is a pure wrapper, no modifications to `soen_toolkit.core`
- **Lazy Compilation**: Models compile on first forward pass or explicit `.compile()` call
- **Full Delegation**: All SOENModelCore features accessible via delegation to `_compiled_core`
- **Config Generation**: Internally converts imperative specs to `SimulationConfig`, `LayerConfig`, `ConnectionConfig`

## Comparison: Config vs. PyTorch API

| Aspect | Config/YAML | PyTorch API |
|--------|-------------|-------------|
| **Style** | Declarative | Imperative |
| **Serialization** | Native | Via `.save()` |
| **IDE Support** | Limited | Full autocomplete |
| **Reproducibility** | Excellent | Excellent (via `.save()`) |
| **Hyperparameter Tuning** | Excellent | Good (via `param_specs`) |
| **PyTorch Integration** | Good | Excellent |
| **Learning Curve** | Steeper | Gentler (if from PyTorch) |

**Both are equally powerful** - choose based on your workflow and preferences.

## Summary

The `soen_toolkit.nn` API provides a **complete, PyTorch-native interface** to SOEN models with:
- Full feature parity with `SOENModelCore`
- Standard PyTorch `nn.Module` patterns
- Fine-grained parameter control
- All visualization, summary, and I/O features
- Zero disruption to existing code

**It's the same powerful SOEN toolkit, with a more familiar interface for PyTorch users.**

