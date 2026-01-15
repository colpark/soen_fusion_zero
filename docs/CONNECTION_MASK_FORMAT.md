---
layout: default
title: Custom Connection Mask Format
---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Building_Models.md">&#8592; Back to Building Models Guide</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
</div>

# Custom Connection Mask Format

## Overview

You can provide custom connectivity patterns by uploading .npz files containing mask arrays. This allows you to use your own code to generate connectivity structures and load them into SOEN models through any API (YAML/JSON, PyTorch, or GUI).

---

## File Format

### Basic Structure

- **Format:** NumPy `.npz` archive (created with `numpy.savez()`)
- **Required key:** `"mask"` (must always be named "mask")
- **Array shape:** `[to_dim, from_dim]` where:
  - `to_dim` is the destination layer size (number of nodes)
  - `from_dim` is the source layer size (number of nodes)
- **Data type:** `float32` or `float64`

### Shape Convention

The mask shape follows the standard connection matrix convention used throughout the toolkit:

```
mask.shape = (to_nodes, from_nodes)
mask[i, j] = 1 if source node j connects to destination node i, 0 otherwise
```

**How masks work:**
- The mask is applied to the weight matrix to **enforce connectivity**
- After weight initialization, non-connected positions are zeroed: `W = W * mask`
- During training, the mask is reapplied after gradient updates to maintain connectivity
- Forward pass uses the masked weights: `phi_dst = s_src @ W.T` where W has been masked

---

## Mask Format

- **Values:** 0 (no connection) or 1 (connection exists)
- **Purpose:** Define connectivity structure only
- **Weight initialization:** Connection weights are initialized separately using your chosen init method (e.g., `xavier_uniform`, `normal`)
- **Example:** `mask[i, j] = 1` means source node j connects to destination node i

The mask is purely structural - it defines **which** connections exist, not their weight values. This keeps things simple and follows the same pattern as other connectivity types like `dense` or `sparse`.

---

## Creating Compatible Files

### Python Example

```python
import numpy as np

# Example: Custom 5x10 connectivity (10 source nodes -> 5 destination nodes)
mask = np.zeros((5, 10), dtype=np.float32)

# Connect source node 0 to destination nodes 0, 1, 2
mask[0:3, 0] = 1.0

# Connect source node 5 to all destination nodes
mask[:, 5] = 1.0

# Connect source nodes 7-9 to destination node 4
mask[4, 7:10] = 1.0

# Save to file
np.savez("my_custom_mask.npz", mask=mask)

print(f"Mask shape: {mask.shape}")
print(f"Number of connections: {np.sum(mask > 0)}")
```

### Note: One Mask Per File

Each .npz file should contain one mask with the key "mask". If you need multiple connectivity patterns, create separate .npz files for each connection.

```python
import numpy as np

# Create different masks
mask1 = np.ones((5, 10), dtype=np.float32)
mask2 = np.random.rand(10, 5).astype(np.float32)

# Save to separate files
np.savez("layer0_to_layer1.npz", mask=mask1)
np.savez("layer1_to_layer2.npz", mask=mask2)
```

### Advanced: Structured Connectivity

```python
import numpy as np

def create_block_diagonal_mask(n_blocks, block_size):
    """Create block diagonal connectivity pattern."""
    total_size = n_blocks * block_size
    mask = np.zeros((total_size, total_size), dtype=np.float32)
    
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        mask[start:end, start:end] = 1.0
    
    return mask

def create_distance_based_mask(from_dim, to_dim, max_distance):
    """Create connectivity based on node distance."""
    mask = np.zeros((to_dim, from_dim), dtype=np.float32)
    
    for i in range(to_dim):
        for j in range(from_dim):
            distance = abs(i - j)
            if distance <= max_distance:
                mask[i, j] = 1.0
    
    return mask

# Use the functions
block_mask = create_block_diagonal_mask(n_blocks=4, block_size=5)
distance_mask = create_distance_based_mask(from_dim=20, to_dim=20, max_distance=3)

# Save to separate files
np.savez("block_diagonal_mask.npz", mask=block_mask)
np.savez("distance_based_mask.npz", mask=distance_mask)
```

---

## Usage Examples

### YAML/JSON Config

```yaml
simulation:
  dt: 37
  network_evaluation_method: layerwise

layers:
  - layer_id: 0
    layer_type: Linear
    params:
      dim: 10
  - layer_id: 1
    layer_type: SingleDendrite
    params:
      dim: 5
      solver: FE
      source_func_type: RateArray
      bias_current: 1.7
      gamma_plus: 0.001
      gamma_minus: 0.001

connections:
  - from_layer: 0
    to_layer: 1
    connection_type: custom
    params:
      structure:
        type: custom
        params:
          mask_file: "./masks/my_mask.npz"
      init:
        name: xavier_uniform
        params:
          gain: 1.0
```

### PyTorch API

```python
from soen_toolkit.nn import Graph, layers, structure, init

g = Graph(dt=37, network_evaluation_method="layerwise")

g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(
    dim=5, solver="FE", source_func_type="RateArray",
    bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
))

# Use custom mask for connectivity structure
g.connect(0, 1,
          structure=structure.custom("my_mask.npz"),
          init=init.xavier_uniform())
```

### Direct Config API

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
    ConnectionConfig(
        from_layer=0,
        to_layer=1,
        connection_type="custom",
        params={
            "structure": {
                "type": "custom",
                "params": {"mask_file": "./masks/my_mask.npz"}
            },
            "init": {
                "name": "xavier_uniform",
                "params": {"gain": 1.0}
            }
        }
    )
]

model = SOENModelCore(sim, layers, connections)
```

### GUI Usage

1. Open the model creation GUI: `python -m soen_toolkit.model_creation_gui`
2. Create or edit a connection between layers
3. In the connection dialog, select **"custom"** from the connectivity type dropdown
4. Click **"Browse..."** next to the "Mask File" field to select your `.npz` file
   - Or type/paste the full file path directly into the field
5. Configure weight initialization method
6. Click OK to create the connection

---

## Validation

The toolkit automatically validates custom masks when building the model:

- **File exists and is readable**
- **File is valid .npz format**
- **Specified key exists in the archive**
- **Array shape matches** `[to_dim, from_dim]`
- **Values are binary:** only 0 or 1 values allowed

**Example error messages:**

```
ValueError: Mask file not found: ./masks/my_mask.npz
ValueError: Key 'my_mask' not found in .npz file. Available keys: ['mask', 'other']
ValueError: Mask shape mismatch. Expected (5, 10) (to_nodes, from_nodes), got (10, 5)
ValueError: Mask must contain only 0 or 1 values. Found unique values: [0.0, 0.5, 1.0]
```

---

## Important Notes

### Mask Embedding

- **Masks are embedded in saved models** - no external file dependency after building
- The .npz file is only needed during model construction
- When you save a model (`.save()`), the mask tensor is saved with the model
- You can move or delete the .npz file after the model is built

### Path Handling

- Use **absolute paths** or **paths relative to your working directory**
- Relative paths are resolved at model build time

### Shape Requirements

- Shape must **exactly match**: `mask.shape == (to_layer.dim, from_layer.dim)`
- For internal connections (same layer): `mask.shape == (dim, dim)`
- The toolkit will raise an error if shapes don't match

### Weight Initialization

- The mask defines **connectivity only** (which connections exist)
- Weights are initialized by your chosen method (e.g., `xavier_uniform`, `normal`)
- This is the same pattern used by all other connectivity types


---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Building_Models.md">&#8592; Back to Building Models Guide</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
</div>

