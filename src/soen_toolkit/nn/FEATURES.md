# PyTorch API Feature Parity

The `soen_toolkit.nn` API (`Graph` and `Sequential`) provides **full access** to all `SOENModelCore` features.

## Feature Comparison

| Feature | SOENModelCore | Graph/Sequential | Notes |
|---------|---------------|------------------|-------|
| **Model Building** | ✓ Config-based | ✓ Imperative | Both approaches available |
| **Forward Pass** | ✓ `.forward(x)` | ✓ `.forward(x)` or `(x)` | Full history tracking support |
| **Parameters** | ✓ `.parameters()` | ✓ `.parameters()` | Standard PyTorch interface |
| **State Dict** | ✓ `.state_dict()` | ✓ `.state_dict()` | Compatible with PyTorch checkpoints |
| **Device Management** | ✓ `.to(device)` | ✓ `.to(device)` | CPU, CUDA, MPS all supported |
| **Visualization** | ✓ `.visualize()` | ✓ `.visualize()` | Graphviz-based network diagrams |
| **Summary** | ✓ `.summary()` | ✓ `.summary()` | Tables, DataFrames, histograms |
| **Save/Load** | ✓ `.save()/.load()` | ✓ `.save()/.load()` | .soen, .pth, .json formats |
| **Parameter Control** | ✓ Distributions | ✓ `param_specs.*` | Uniform, normal, lognormal, etc. |
| **Learnability** | ✓ Per-parameter | ✓ Per-parameter | Fine-grained freeze/train control |
| **Constraints** | ✓ Min/max | ✓ Min/max | Automatically enforced |
| **Dynamic Connections** | ✓ Hidden multipliers | ✓ `mode="WICC"/"NOCC"` | Full multiplier dynamics (WICC: v1, NOCC: v2) |
| **Connection Patterns** | ✓ Dense, sparse, etc. | ✓ `structure.*` | All patterns available |
| **Weight Init** | ✓ Xavier, Kaiming, etc. | ✓ `init.*` | All initializers available |
| **Training** | ✓ Standard PyTorch | ✓ Standard PyTorch | Works with any optimizer/loss |
| **Quantization** | ✓ `.quantize()` | ✓ `.quantize()` | Via `_compiled_core` |
| **Noise** | ✓ Noise configs | ✓ Noise configs | Full support |
| **History Tracking** | ✓ φ, s, g, power | ✓ φ, s, g, power | Via attributes after forward |

## API Methods

### Graph and Sequential Methods

```python
# Core PyTorch methods
g.forward(x)                    # Forward pass
g.parameters()                  # Get parameters
g.named_parameters()            # Get named parameters
g.state_dict()                  # Get state dictionary
g.load_state_dict(state)        # Load state dictionary
g.to(device)                    # Move to device
g.train() / g.eval()            # Set training mode

# SOENModelCore features
g.compile()                     # Explicit compilation
g.visualize(**kwargs)           # Network visualization
g.summary(**kwargs)             # Model summary
g.compute_summary()             # Summary as dict
g.save(path)                    # Save model
Graph.load(path)                # Load model (class method)
Graph.from_core(core)           # Create from SOENModelCore

# Advanced (via _compiled_core)
g._compiled_core.quantize()     # Quantize model
g._compiled_core.reset_states() # Reset internal states
g._compiled_core.enforce_param_constraints()  # Apply constraints
```

## Examples

### 1. Visualization
```python
g.visualize(save_path="network", file_format="svg", orientation="LR")
```

### 2. Summary
```python
g.summary()  # Print to console
df = g.summary(return_df=True)  # As DataFrame
stats = g.compute_summary()  # As dict
```

### 3. Save/Load
```python
g.save("model.soen")  # Save
g_new = Graph.load("model.soen")  # Load
```

### 4. Device Management
```python
g_cuda = g.to('cuda')
g_mps = g.to('mps')
```

### 5. Fine-Grained Parameter Control
```python
from soen_toolkit.nn import param_specs

g.add_layer(1, layers.SingleDendrite(
    dim=50,
    bias_current=param_specs.uniform(1.5, 2.0, learnable=True, 
                                     constraints={"min": 0.0, "max": 5.0}),
    gamma_plus=param_specs.lognormal(-6.9, 0.2, learnable=True),
))
```

