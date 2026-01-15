---
layout: default
title: Connections (Patterns)
---
## Connection Patterns

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Layer_Types.md" style="margin-right: 2em;">&#8592; Previous: Layer Types</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Solvers_and_Simulation.md" style="margin-left: 2em;">Next: Solvers &amp; Simulation &#8594;</a>
</div>

This page describes **connection patterns** (how layers are wired together).

For the connection mask file format reference, see:

- `../CONNECTION_MASK_FORMAT.md`

---

### Basic patterns

#### 1. Dense (all-to-all)

Every node in source layer connects to every node in target layer.

```python
from soen_toolkit.core import ConnectionConfig

ConnectionConfig(
    from_layer=0,
    to_layer=1,
    connection_type="dense",
)
```

**Matrix:** dense ($M \times N$)

![Illustration of dense connectivity](../Figures/Building_Models/model_building_fig8.jpg)

---

#### 2. Random sparse

Random connections with specified probability.

```python
from soen_toolkit.core import ConnectionConfig

ConnectionConfig(
    from_layer=0,
    to_layer=1,
    connection_type="sparse",
    params={"sparsity": 0.3},  # 30% connection probability
)
```

**Matrix:** sparse ($M \times N$) with ~sparsity×$M$×$N$ nonzero entries

![Illustration of sparse connectivity](../Figures/Building_Models/model_building_fig9.jpg)

---

#### 3. One-to-one

Direct 1:1 mapping (layers must be same size).

```python
from soen_toolkit.core import ConnectionConfig

ConnectionConfig(
    from_layer=0,
    to_layer=1,
    connection_type="one_to_one",
)
```

**Matrix:** diagonal ($M \times M$)

---

### Advanced patterns

#### 4. Power law (distance-dependent)

Probability of connection decreases with physical distance, mimicking cortical connectivity. Nodes are conceptually arranged on a grid.

```python
from soen_toolkit.core import ConnectionConfig

ConnectionConfig(
    from_layer=0,
    to_layer=1,
    connection_type="power_law",
    params={
        "alpha": 2.0,            # Power law exponent (higher = more local)
        "expected_fan_out": 4,   # Number of targets sampled per source node
    },
)
```

---

#### 5. Exponential decay

Similar to power law but with exponential distance penalty.

```python
from soen_toolkit.core import ConnectionConfig

ConnectionConfig(
    connection_type="exponential",
    params={
        "d_0": 2.0,              # Characteristic length scale for decay
        "expected_fan_out": 4,   # Number of targets sampled per source node
    },
)
```

---

#### 6. Block structure

Creates blocks of dense connectivity (useful for modular networks).

```python
from soen_toolkit.core import ConnectionConfig

ConnectionConfig(
    connection_type="block_structure",
    params={
        "block_size": 5,    # Size of each block
        "p_in": 1.0,        # Connection probability within blocks
        "p_out": 0.0,       # Connection probability between blocks
    },
)
```

---

#### 7. Custom (from file)

Load connectivity from your own `.npz` file.

```python
from soen_toolkit.core import ConnectionConfig

ConnectionConfig(
    from_layer=0,
    to_layer=1,
    connection_type="custom",
    params={"mask_file": "./masks/my_custom_connectivity.npz"},
)
```

**How it works:**

- Load a NumPy `.npz` file containing a binary connectivity mask (0/1 values)
- Mask defines connectivity structure only
- Weights are initialized separately by your chosen init method
- Mask shape must be `[to_dim, from_dim]`

#### Example: creating a custom mask

```python
import numpy as np

# Create custom 5x10 connectivity pattern
mask = np.zeros((5, 10), dtype=np.float32)

# Connect source node 0 to destination nodes 0, 1, 2
mask[0:3, 0] = 1.0

# Connect source node 5 to all destination nodes
mask[:, 5] = 1.0

# Save to file
np.savez("my_custom_mask.npz", mask=mask)
```

See `../CONNECTION_MASK_FORMAT.md` for detailed format specification and more examples.

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Layer_Types.md" style="margin-right: 2em;">&#8592; Previous: Layer Types</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Solvers_and_Simulation.md" style="margin-left: 2em;">Next: Solvers &amp; Simulation &#8594;</a>
</div>
