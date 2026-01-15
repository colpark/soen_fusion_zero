---
layout: default
title: Multiplier Dynamic Weights - Equation Reference
---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="../Building_Models.md" style="margin-right: 2em;">&#8592; Back to: Building Models</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="JJs.md" style="margin-left: 2em;">More Physics: Josephson Junctions &#8594;</a>
</div>

# Multiplier Dynamic Weights: Complete Equation Reference

## Purpose

Dynamic weights enable **on-chip re-programmability** for superconducting electronic networks using physical multiplier circuits. Unlike fixed weights (matrix multiplication via static mutual inductances), these are **dynamical systems** with internal states that approximate analog multiplication in steady-state.

**Key characteristics:**
- Physical SQUID-based circuits that evolve according to circuit ODEs
- Approximate multiplication: $\text{output} \approx \propto \phi_{\text{in}} \times \phi_w$ in steady-state
- Works best in narrow range around $\phi=0$ or offset by $\frac{1}{2}\Phi_0$ (approximately $\pm 0.15$ either side)
- Adds computational cost (more ODEs) but enables weight reprogramming without hardware changes


---

## 1. WICC (With Collection Coil) - V1 Multiplier

### Core Dynamics

**Single ODE per edge:**

$$\boxed{\frac{ds}{dt} = \gamma^{+} \cdot g_1 - \gamma^{+} \cdot g_2 - \gamma^{-} \cdot s}$$

where:
$$g_1 = g(\phi_{\text{in}} + \phi_w, i_b - s)$$
$$g_2 = g(\phi_{\text{in}} - \phi_w, i_b + s)$$

**Parameters:**
- $s$ = multiplier edge state (dimensionless current)
- $\gamma^{+}$ = dimensionless drive gain (default: 0.001)
- $\gamma^{-}$ = dimensionless leak/damping term (default: 0.001)
- $\phi_{\text{in}}$ = input flux from upstream state
- $\phi_w$ = weight flux (programmable control parameter)
- $i_b$ = bias current (default: 2.0)
- $g(\phi, i_b)$ = SQUID source function (nonlinear voltage response)

**Implementation location:** `src/soen_toolkit/core/layers/physical/dynamics/multiplier.py`

### Input/Output Transformations

**Per-edge input flux:**
$$\phi_{\text{in}}^{ij} = s_j \cdot J_{\text{in}}$$

where $J_{\text{in}}$ is the input coupling coefficient (default: 0.38).

**Per-edge weight flux:**
$$\phi_w^{ij} = J_{\text{eff}}[i, j]$$

The weight matrix entries directly represent the control flux values.

**Output to destination node:**
$$\phi_{\text{out}}^i = \sum_j \left(s_{ij} \cdot J_{\text{out}}^i\right)$$

### Fan-In Dependent Output Coupling

For WICC, the output coupling $J_{\text{out}}$ is **automatically computed** based on fan-in:

$$\boxed{J_{\text{out}}^i = \frac{2.5 \times 10^{-10}}{(40 \times 10^{-12}) \cdot N_{\text{in}}^i + 3.35 \times 10^{-10}}}$$

where:
- $N_{\text{in}}^i$ = number of incoming connections to destination node $i$

**Example:** For fan-in = 10: $J_{\text{out}} \approx 0.34$

**Implementation location:** `src/soen_toolkit/core/mixins/builders.py`, lines 254-278

---

## 2. NOCC (No Collection Coil) - V2 Multiplier

### Overview

NOCC uses a more compact circuit design without collection coils. Each edge has **two SQUID states** ($s_1$, $s_2$) representing left and right branches, plus one **aggregated output state** ($m$) per destination node. These are coupled through three ODEs that must be solved together.

### Core Dynamics (Three Coupled ODEs)

#### 2.1 Aggregated Output State

$$\boxed{\beta_{\text{eff}} \cdot \frac{dm_i}{dt} = \sum_j\left(g_1^{ij} - g_2^{ij}\right) - \alpha \cdot m_i}$$

where the **effective dimensionless inductance** is:

$$\boxed{\beta_{\text{eff}} = \beta + 2 \cdot N \cdot \beta_{\text{out}}}$$

**Rearranged for Forward Euler integration:**
$$\frac{dm}{dt} = \frac{\sum_j\left(g_1^{ij} - g_2^{ij}\right) - \alpha \cdot m}{\beta_{\text{eff}}}$$

**Parameters:**
- $m$ = aggregated output state per destination node (dimensionless current)
- $\beta$ = dimensionless inductance of incoming branches (default: 303.85)
- $\beta_{\text{out}}$ = dimensionless inductance of output branch (default: 91.156)
- $N$ = fan-in to destination node
- $\alpha$ = dimensionless resistance (default: 1.64053)

#### 2.2 Left Branch SQUID State

$$\boxed{\beta \cdot \frac{ds_1^{ij}}{dt} = g_1^{ij} - \beta_{\text{out}} \cdot \frac{dm_i}{dt} - \alpha \cdot s_1^{ij}}$$

**Rearranged:**
$$\frac{ds_1}{dt} = \frac{g_1 - \beta_{\text{out}} \cdot \frac{dm}{dt} - \alpha \cdot s_1}{\beta}$$

#### 2.3 Right Branch SQUID State

$$\boxed{\beta \cdot \frac{ds_2^{ij}}{dt} = g_2^{ij} - \beta_{\text{out}} \cdot \frac{dm_i}{dt} - \alpha \cdot s_2^{ij}}$$

**Rearranged:**
$$\frac{ds_2}{dt} = \frac{g_2 - \beta_{\text{out}} \cdot \frac{dm}{dt} - \alpha \cdot s_2}{\beta}$$

**Implementation location:** `src/soen_toolkit/core/layers/physical/dynamics/multiplier_v2.py`

### Source Function Terms

The two source functions for left and right branches:

$$g_1 = g(\phi_{\text{in}} + \phi_w, i_b - s_1)$$

$$g_2 = g(\phi_{\text{in}} - \phi_w, -i_b + s_2)$$

Note on negative bias: The right branch uses $-i_b$. For RateArray (which supports currents 0.95 to 2.5), the implementation uses $\lvert i_{\text{squid}} \rvert$, relying on physical symmetry around $I=0$.

### NOCC Physical Parameter Mappings

| Parameter | Symbol | Default | Physical Mapping | Description |
|-----------|--------|---------|------------------|-------------|
| alpha | $\alpha$ | 1.64053 | $R \approx 2\,\Omega$ | Dimensionless resistance |
| beta | $\beta$ | 303.85 | $L_{\text{branch}} \approx 1\,\text{nH}$ | Dimensionless inductance (incoming) |
| beta_out | $\beta_{\text{out}}$ | 91.156 | $L_{\text{out}} \approx 300\,\text{pH}$ | Dimensionless inductance (output) |
| ib | $i_b$ | 2.1 | $I_b \approx 210\,\mu\text{A}$ | Bias current |
| phi_y | $\phi_w$ | 0.1 | - | Weight flux |

### Input/Output Transformations

**Per-edge input flux:**
$$\phi_{\text{in}}^{ij} = s_j \cdot J_{\text{in}}$$

**Per-edge flux terms:**
$$\phi_a^{ij} = \phi_{\text{in}}^{ij} + \phi_w^{ij}$$
$$\phi_b^{ij} = \phi_{\text{in}}^{ij} - \phi_w^{ij}$$

**Output (using aggregated $m$ state):**
$$\phi_{\text{out}}^i = m_i \cdot J_{\text{out}}$$

**Key difference from WICC:** $J_{\text{out}}$ is a **fixed parameter** (default: 0.38), **NOT** fan-in dependent.

---

## 3. Integration Method

Both implementations use **Forward Euler** integration:

$$x(t + \Delta t) = x(t) + \Delta t \cdot \frac{dx}{dt}$$

### NOCC Integration Order

For NOCC, the solving order matters due to coupling:

1. Compute source functions $g_1$ and $g_2$ using current edge states
2. Aggregate and solve for $\frac{dm}{dt}$ using $\beta_{\text{eff}}$
3. Use $\frac{dm}{dt}$ to solve for $\frac{ds_1}{dt}$ and $\frac{ds_2}{dt}$
4. Update all three states: $s_1(t+\Delta t)$, $s_2(t+\Delta t)$, $m(t+\Delta t)$

---

## 4. Key Implementation Differences

| Aspect | WICC (V1) | NOCC (V2) |
|--------|-----------|-----------|
| **States** | 1 per edge: $s$ | 2 per edge: $s_1, s_2$ + 1 per node: $m$ |
| **ODEs per timestep** | $E$ edges | $2E + D$ destinations |
| **Coupling parameters** | $\gamma^{+}, \gamma^{-}$ | $\alpha, \beta, \beta_{\text{out}}$ |
| **Effective inductance** | Implicit in $\gamma^{+}$ | Explicit: $\beta_{\text{eff}} = \beta + 2N\beta_{\text{out}}$ |
| **Damping/leak** | $\gamma^{-} \cdot s$ term | $\alpha \cdot s$ in all three ODEs |
| **Output coupling** | $J_{\text{out}}$ (fan-in dependent) | $J_{\text{out}}$ (fixed) |
| **Source functions** | $g_1(\phi + \phi_w, i_b - s)$, $g_2(\phi - \phi_w, i_b + s)$ | $g_1, g_2$ with opposite branches |
| **Circuit design** | Collection coil present | No collection coil (more compact) |

---

## 5. Steady-State Behavior

In steady-state ($\frac{ds}{dt} \approx 0$), both circuits approximate multiplication:

### WICC Steady-State

$$\Rightarrow s_{ss} \approx  \propto \phi_{\text{in}} \times \phi_w \quad$$

### NOCC Steady-State 

$$\Rightarrow m_{ss} \approx \propto \phi_{\text{in}} \times \phi_w \quad$$

**Accuracy:** Multiplication approximation works best when:
- Both inputs are in optimal range: $\pm 0.15$ around $\phi=0$ or offset by $0.5\Phi_0$
- Each input is held long enough to allow the circuit to reach steady-state
- Both $\phi_{\text{in}}$ and $\phi_w$ are well-matched in magnitude

---

## 6. Optional: Half-Flux Offset

Both implementations support shifting the operating point by half a flux quantum:

$$\phi_w^{ij} \rightarrow \phi_w^{ij} + 0.5$$

Enabled by setting `half_flux_offset: True` in connection parameters. This:
- Moves operating point from $\phi=0$ to $\phi=0.5\Phi_0$


---

## 7. Usage Examples

### WICC Connection

```python
from soen_toolkit.nn import connection

conn_wicc = connection(
    from_layer="layer1",
    to_layer="layer2",
    mode="WICC",
    connection_params={
        "gamma_plus": 0.001,
        "gamma_minus": 0.001,
        "bias_current": 2.0,
        "j_in": 0.38,
        "j_out": 0.38,  # Auto-computed if omitted
    }
)
```

### NOCC Connection

```python
conn_nocc = connection(
    from_layer="layer1",
    to_layer="layer2",
    mode="NOCC",
    connection_params={
        "alpha": 1.64053,
        "beta": 303.85,
        "beta_out": 91.156,
        "ib": 2.1,
        "j_in": 0.38,
        "j_out": 0.38,  # Fixed, not fan-in dependent
    }
)
```

### Fixed Weights (Baseline)

```python
conn_fixed = connection(
    from_layer="layer1",
    to_layer="layer2",
    mode="fixed"  # Just matrix multiply: φ_out = s @ J^T
)
```

---

## 8. Code Locations

### PyTorch Implementations
- **WICC layer dynamics:** `src/soen_toolkit/core/layers/physical/dynamics/multiplier.py`
- **NOCC layer dynamics:** `src/soen_toolkit/core/layers/physical/dynamics/multiplier_v2.py`
- **Connection ops (both):** `src/soen_toolkit/core/utils/connection_ops.py`
  - `MultiplierOp` class for WICC
  - `MultiplierNOCCOp` class for NOCC
- **J_out auto-computation:** `src/soen_toolkit/core/mixins/builders.py` (lines 254-278)

### JAX Implementations
- **Layers:** `src/soen_toolkit/utils/port_to_jax/layers_jax.py`
- **Optimized kernels:** `src/soen_toolkit/utils/port_to_jax/fast_kernels_v2.py`
- **Model integration:** `src/soen_toolkit/utils/port_to_jax/jax_model.py`

---

## Summary

Dynamic weights provide **hardware-realistic programmable connections** through physical multiplier circuits:

- **WICC** (V1): Simpler design with collection coil, 1 state per edge, fan-in dependent $J_{\text{out}}$
- **NOCC** (V2): More compact without collection coil, 3 coupled states (2 per edge + 1 per node), fixed $J_{\text{out}}$
- Both approximate multiplication in steady-state within narrow flux ranges
- Both add computational cost (additional ODEs) compared to fixed weights
- Enable on-chip weight reprogramming without hardware changes

Understanding the coupled dynamics, proper parameter scaling, and operating ranges is essential for accurate hardware-realistic neural network simulations.

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="../Building_Models.md" style="margin-right: 2em;">&#8592; Back to: Building Models</a>
  &nbsp;|&nbsp;
  <a href="../index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="JJs.md" style="margin-left: 2em;">More Physics: Josephson Junctions &#8594;</a>
</div>
