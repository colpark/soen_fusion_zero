---
layout: default
title: Layer Types (Catalog)
---
## Layer Types Catalog

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="../Building_Models.md" style="margin-right: 2em;">&#8592; Previous: Building Models</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Connections.md" style="margin-left: 2em;">Next: Connections &#8594;</a>
</div>

This page lists the available **layer types** and when to use them.

If you’re looking for JAX-specific implementation details (topology vs params, unified forward, etc.), see:

- `../jax/README.md`

---

### Physical Layers (model real hardware)

> **Layer naming:** Physical multiplier layers use names that match their connection mode counterparts:
>
> - **MultiplierWICC** (With Collection Coil)
> - **MultiplierNOCC** (No Collection Coil)
>
> Legacy name `"Multiplier"` is still supported as an alias for `MultiplierWICC`.

#### SingleDendrite

**What:** The fundamental SOEN computational unit — a superconducting integrator with nonlinear dynamics.

**Circuit physics:** Two coupled loops containing Josephson junctions. Input flux drives current into an integration loop that slowly decays.

**Phenomenological model:**

$$\boxed{\dot{s} = \gamma^{+} g(\phi, i_b - s) - \gamma^{-} s}$$

where:

- $s$: state (integrated current)
- $\phi$: input flux (weighted sum of upstream states)
- $g(\cdot)$: nonlinear source function
- $i_b$: bias current (defines the threshold of source functions)
- $\gamma^{+}$: drive term gain
- $\gamma^{-}$: leak rate

![Phenomenological Model of Dendrite: Diagram](../Figures/Building_Models/model_building_fig10_dendrite_diagram.jpg)
*Simplified diagram to illustrate the phenomenological model of GreatSky's superconducting dendrite circuit. This can be thought of as a node/unit/cell in the physical neural network.*

**When to use:**

- Building SOEN networks (i.e not virtual)
- Need temporal integration and nonlinear dynamics
- Modeling actual dendrite circuits

**Parameters:**

```python
from soen_toolkit.core import LayerConfig

LayerConfig(
    layer_type="SingleDendrite",
    params={
        "dim": 20,                          # Number of dendrites

        # === Solver ===
        "solver": "FE",                     # "FE" (Forward Euler) or "PS" (Parallel Scan)*

        # === Nonlinearity ===
        "source_func_type": "RateArray",    # "RateArray" (lookup table) or analytic key

        # === Circuit Parameters ===
        "bias_current": 1.7,                # Dimensionless bias (threshold)
        "phi_offset": 0.23,                 # Flux offset
        "gamma_plus": 0.001,                # Drive gain on g(·)
        "gamma_minus": 0.001,               # Leak rate (larger = faster decay)
    }
)
```

**Parameter constraints** (enforced during training):

- `bias_current >= 0`
- `gamma_plus >= 0`
- `gamma_minus >= 0`

> **PS solver limitation:** Parallel Scan only works when:
>
> 1. No internal connections, AND
> 2. The source function does not depend on the layer's state $s$ (i.e., `uses_squid_current` is `False`).
>
> For state-dependent source functions or internal connections, use `solver="FE"`.

---

#### MultiplierWICC (With Collection Coil)

> **Detailed physics:** For complete equations, physical parameter mappings, and implementation details, see [Multiplier Equations Reference](../DeeperDives/Multiplier_Equations_Reference.md).

**What:** Circuit that computes approximate analog multiplication using WICC physics with collection coil design — state is proportional to the product of two input fluxes.

**Layer vs connection:**

- **MultiplierWICC layer**: A computational layer with multiplier nodes (use when you want multiplier circuits as neurons)
- **WICC connection mode**: Programmable weights between layers (use when you want dynamic/programmable weights)

Both use the same circuit physics but serve different architectural purposes.

**Equation:**

$$\boxed{\dot{s} = \gamma^{+} \cdot g_1 - \gamma^{+} \cdot g_2 - \gamma^{-} \cdot s}$$

where $g_1 = g(\phi_{\text{in}} + \phi_w, i_b - s)$ and $g_2 = g(\phi_{\text{in}} - \phi_w, i_b + s)$

![Phenomenological Model of Multiplier: Diagram](../Figures/Building_Models/model_building_fig10_multiplier_diagram.jpg)
*Simplified diagram to illustrate the phenomenological model of GreatSky's superconducting multiplier circuit. Note the two input fluxes in contrast to the dendrites' one.*

**When to use as a layer:**

- Approximate analog multiplication as a computational operation
- Simpler dynamics than NOCC (1 state per circuit)
- Collection coil design

**Parameters:**

```python
from soen_toolkit.core import LayerConfig

LayerConfig(
    layer_type="MultiplierWICC",  # Legacy: "Multiplier" also works
    params={
        "dim": 10,
        "solver": "FE",                      # Only FE supported
        "source_func_type": "RateArray",     # Single source function key
        "phi_y": 0.1,                        # Secondary input term
        "bias_current": 2.0,
        "gamma_plus": 0.001,                 # Drive gain
        "gamma_minus": 0.001,                # Leak/damping term
    }
)
```

**Multiplication accuracy:**
The multiplication is approximate and works best when:

- Both inputs are well matched in magnitude
- Circuit is in steady-state
- Input fluxes are in range $\pm 0.15$ around $\phi=0$ or offset by $0.5\Phi_0$
- Both $\phi_{\text{in}}$ and $\phi_w$ are in similar ranges

---

#### MultiplierNOCC (No Collection Coil)

> **Detailed physics:** For complete equations, physical parameter mappings, and implementation details, see [Multiplier Equations Reference](../DeeperDives/Multiplier_Equations_Reference.md).
> **Important distinction:** This section describes **MultiplierNOCC layer** — a computational layer type. Do not confuse with **NOCC connection mode**:
>
> - **MultiplierNOCC layer**: a computational layer with multiplier nodes (like SingleDendrite)
> - **NOCC connection mode** (`mode="NOCC"`): programmable weights between any two layers
>
> Both use the same circuit physics but serve different architectural purposes. Use the layer for multiplier computation nodes. Use NOCC connections for programmable weights.

**What:** New multiplier circuit with dual SQUID states and aggregated output. Uses a more compact flux collection mechanism without collection coils.

**Circuit summary:** NOCC uses 3 coupled ODEs — two SQUID states per edge ($s_1$, $s_2$) plus one aggregated output state per destination node ($m$). The effective dimensionless inductance is $\beta_{\text{eff}} = \beta + 2N\beta_{\text{out}}$ where $N$ is fan-in.

See the [detailed reference](../DeeperDives/Multiplier_Equations_Reference.md) for complete equations and coupling details.

**Parameters:**

```python
from soen_toolkit.core import LayerConfig

LayerConfig(
    layer_type="MultiplierNOCC",
    params={
        "dim": 10,
        "solver": "FE",                      # Only FE supported
        "source_func_type": "RateArray",
        "phi_y": 0.1,                        # Secondary input/weight term
        "ib": 2.1,                           # Bias current
        "alpha": 1.64053,                    # Dimensionless resistance
        "beta": 303.85,                      # Inductance of incoming branches
        "beta_out": 91.156,                  # Inductance of output branch
    }
)
```

**Default physical parameter mappings:**

| Parameter | Physical meaning | Default | Key |
|---|---|---:|---|
| $\beta$ | inductance of each branch | 303.85 | `beta` |
| $\beta_{out}$ | output inductance | 91.156 | `beta_out` |
| $i_b$ | bias current | 2.1 | `ib` |
| $R$ | resistance | 1.64053 | `alpha` |

**Key differences from WICC:**

- Time to steady state is longer: ~30ns (NOCC) vs ~10ns (WICC)
- Circuit topology: dual SQUID states per edge vs single state
- Output: aggregated $m$ state vs direct $s$ state
- Hardware: no collection coil (NOCC) vs with collection coil (WICC)

**Using V2 with dynamic connections:**

```python
# example using the Graph API (as opposed to the configs)
from soen_toolkit.nn import Graph, layers, structure, init

g = Graph(dt=37, network_evaluation_method="layerwise")
g.add_layer(0, layers.Linear(dim=10))
g.add_layer(1, layers.SingleDendrite(dim=5, ...))

g.connect(
    0, 1,
    structure=structure.dense(),
    init=init.uniform(-0.15, 0.15),
    mode="NOCC"
)
```

---

#### DendriteReadout

**What:** Specialized readout layer that outputs the source function value directly without any temporal integration.

**Phenomenological model:**

$$\boxed{s(t) = g(\phi(t), i_b)}$$

**When to use:**

- Reading the state stored in a single dendrite circuit
- Hardware-compatible readout mechanism

**Parameters:**

```python
from soen_toolkit.core import LayerConfig

LayerConfig(
    layer_type="DendriteReadout",
    params={
        "dim": 10,                      # Number of readout circuits
        "source_func_type": "RateArray",
        "bias_current": 1.7,
        "phi_offset": 0.0,
    }
)
```

> **Training stability:** `DendriteReadout` can lead to unstable gradients. To mitigate:
>
> - Connect one-to-one from output layer to readout circuits
> - Freeze those connections during training

---

### Virtual layers (abstract operations)

These don't correspond to hardware but are useful for hybrid networks or preprocessing.

#### Linear

**What:** Simple passthrough layer — no dynamics, just holds values.

**When to use:**

1. **As the input layer (layer 0)**:
   - When using `input_type="state"`, the first layer's states are directly overridden with input signals
   - While any layer type works at ID=0 with `input_type="state"` (since states are overwritten), using Linear is clearest

2. **As an intermediate passthrough layer:**
   - Creates a linear combination of upstream states via matrix-vector multiplication (the upstream-φ computation)
   - Simply passes $\phi$ (weighted inputs) directly through as its state: $s = \phi$
   - Useful when you need a linear projection without adding temporal dynamics

```python
from soen_toolkit.core import LayerConfig

LayerConfig(
    layer_id=0,
    layer_type="Linear",
    params={"dim": 10}  # Only parameter needed
)
```

> **Note:** Layer 0 should almost always be Linear unless you have a specific reason otherwise.

---

#### ScalingLayer

**What:** Applies a learnable per-feature scaling factor to inputs.

**When to use:**

- Need learnable gain control for each feature
- Normalizing or amplifying specific input channels
- Adding trainable scaling between layers

**Parameters:**

```python
from soen_toolkit.core import LayerConfig

LayerConfig(
    layer_type="ScalingLayer",
    params={
        "dim": 10,
        "scale_factor": 1.0,  # Initial scaling (can be learned)
    }
)
```

**Behavior:** Each input feature is multiplied by its scale factor: $s = \phi \cdot \text{scale factor}$

---

#### NonLinear

**What:** Applies a configurable nonlinearity (source function) to input without temporal dynamics. Identical to Linear, but each timestep is passed through a non-linearity.

**When to use:**

- Need static nonlinear transformations
- Testing different activation functions
- Creating hybrid networks with custom nonlinearities

**Parameters:**

```python
from soen_toolkit.core import LayerConfig

LayerConfig(
    layer_type="NonLinear",
    params={
        "dim": 10,
        "source_func_type": "Tanh",   # Or "RateArray", "SimpleGELU", etc.
        "phi_offset": 0.0,            # Shift input before nonlinearity
        "bias_current": 1.7,          # Only used if the source function accepts it
    }
)
```

**Available activation functions:**

- `"Tanh"`, `"SimpleGELU"`, `"Telu"`
- `"RateArray"` (lookup table from circuit simulations)
- And others. See `soen_toolkit.core.source_functions.SOURCE_FUNCTIONS.keys()` for a full list.

**Behavior:** Computes $s = g(\phi + \phi_\text{offset}, i_b)$ where $g$ is the chosen source function.

> **Key difference:** Unlike SingleDendrite, this has no temporal integration — nonlinearity is applied instantaneously at each timestep.

---

#### RNN / LSTM / GRU / MinGRU

**What:** Standard recurrent neural network variants from PyTorch, wrapped for compatibility. `MinGRU` is a custom lightweight (and parallelisable over time) variant of GRU.

**When to use:**

- Hybrid SOEN + traditional RNN networks
- Comparing against baselines
- Preprocessing sequences before SOEN layers

```python
from soen_toolkit.core import LayerConfig

LayerConfig(
    layer_type="LSTM",
    description="Feature extraction",
    params={"dim": 64},  # Sets both input_size and hidden_size
)
```

**Available types:** `"RNN"`, `"LSTM"`, `"GRU"`, `"MinGRU"`

> **Note:** These are wrappers around single-layer `nn.Module` instances where `input_size` and `hidden_size` are both set to the layer's `dim`. Parameters like `num_layers` or `dropout` are not configurable through this interface.

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="../Building_Models.md" style="margin-right: 2em;">&#8592; Previous: Building Models</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Connections.md" style="margin-left: 2em;">Next: Connections &#8594;</a>
</div>
