# Dendritic Dynamics 🔒 HARDWARE-FIXED

## Classification: DO NOT MODIFY

This module contains the ODE kernels that implement SOEN neuron dynamics.
These equations are derived from circuit physics and represent the actual
behavior of superconducting optoelectronic dendrites.

---

## Contents

| File | ODE | Modification Risk |
|------|-----|-------------------|
| `single_dendrite.py` | ds/dt = γ⁺g(φ) - γ⁻s | 🔴 CRITICAL |
| `multiplier.py` | WICC multiplicative dynamics | 🔴 CRITICAL |
| `multiplier_v2.py` | NOCC multiplicative dynamics | 🔴 CRITICAL |
| `*_coeffs.py` | Coefficient computations | 🔴 CRITICAL |

---

## The Core Equation

```python
# single_dendrite.py:45 - THE HEART OF SOEN
return gamma_plus * g_val - gamma_minus * state
```

This single line implements:

```
ds/dt = γ⁺ · g(φ) - γ⁻ · s
```

| Term | Physical Origin |
|------|-----------------|
| γ⁺ · g(φ) | Current injection from photon-induced conductance |
| γ⁻ · s | Inductance-based decay (L/R time constant) |

---

## Physical Interpretation

```
                    ┌─────────────────────────┐
                    │   Superconducting Loop  │
    Photons ──SPD──►│                         │
                    │   Current s circulates  │
                    │   indefinitely          │
                    └───────────┬─────────────┘
                                │
                    ds/dt = γ⁺g(φ) - γ⁻s
                    ─────   ─────   ────
                      │       │       │
                      │       │       └── Decay (inductance)
                      │       └────────── Input (photon-induced)
                      └────────────────── Rate of change
```

---

## The SQUID Current

```python
# single_dendrite.py:35
squid_current = bias_current - state
```

When current `s` is stored in the loop, the current available for the SQUID is:
```
I_SQUID = I_bias - s
```

This determines the operating point on the g(φ) curve.

---

## Why This Is Fixed

The ODE structure comes from Kirchhoff's laws applied to the superconducting circuit:

1. **Current conservation** at the SPD-loop junction
2. **Flux quantization** in the superconducting loop
3. **Josephson relations** for the SQUID junction

Changing the ODE structure means you're modeling a different circuit.

---

## What CAN Be Changed (With Caution)

| Parameter | Adjustable? | Notes |
|-----------|-------------|-------|
| γ⁺ value | ⚠️ Within physical limits | Device-dependent |
| γ⁻ value | ⚠️ Within physical limits | Device-dependent |
| bias_current | ⚠️ Within operating range | Affects sensitivity |
| ODE STRUCTURE | ❌ NO | Fixed by physics |

---

## Discretization

The continuous ODE is discretized using Forward Euler:

```python
# single_dendrite.py:82-85
alpha = 1.0 - dt * gamma_minus
beta = dt * gamma_plus
return alpha * s_prev + beta * g_val
```

This is a numerical approximation. The SOLVER can be changed (Forward Euler
vs. ParaRNN), but the underlying ODE being solved must remain the same.

---

## Imported By

- `soen_toolkit.hardware_fixed` (re-exports)
- `core/layers/physical/*.py` (layer implementations)
- `core/soen_model_core.py` (model forward pass)
