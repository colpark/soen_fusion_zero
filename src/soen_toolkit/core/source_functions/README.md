# Source Functions 🔒 HARDWARE-FIXED

## Classification: DO NOT MODIFY

This module contains the source function g(φ) implementations that encode
the SQUID response curve - the heart of SOEN neuron behavior.

---

## Contents

| File | Purpose | Modification Risk |
|------|---------|-------------------|
| `rate_array.py` | Lookup table interpolation | 🔴 CRITICAL |
| `base_rate_array.soen` | Pre-computed g(φ) data | 🔴 CRITICAL |
| `heaviside.py` | Simplified step approximation | 🟡 MODERATE |
| `analytic.py` | Analytic approximations | 🟡 MODERATE |
| `registry.py` | Source function registry | 🟢 LOW |

---

## What is g(φ)?

The source function g(φ) maps magnetic flux to effective conductance:

```
g: φ (flux in units of Φ₀) → conductance (dimensionless)
```

### Key Properties (Fixed by Physics)

1. **Periodicity**: g(φ) = g(φ + 1), period = Φ₀
2. **Symmetry**: g(φ) = g(1 - φ), mirror around φ = 0.5
3. **Shape**: Determined by Josephson junction physics

```
        g(φ)
          │     ___
          │    /   \
          │   /     \
          │  /       \
          │ /         \
          └──────┬──────► φ
                0.5
```

---

## Why This Is Fixed

The g(φ) curve is **measured or simulated from the physical device**:

1. Fabricate a SQUID device
2. Measure conductance vs. applied flux
3. Fit or tabulate the response
4. Store in `base_rate_array.soen`

The lookup table in `rate_array.py` interpolates this measured data.
Changing it means you're simulating a different device.

---

## The Lookup Table

```python
# rate_array.py
phi_mod = torch.remainder(phi, 1.0)      # Apply periodicity
phi_eff = torch.minimum(phi_mod, 1.0 - phi_mod)  # Apply symmetry
return self._interpolate(phi_eff, squid_current)  # 2D interpolation
```

The data file `base_rate_array.soen` contains pre-computed values from
device physics simulation.

---

## Valid Reasons to Modify

1. **New device measurements**: If you have measured g(φ) for a new device,
   generate a new `.soen` file with the new data.

2. **Alternative device topology**: Different SQUID geometries may have
   different response curves.

3. **Analytical studies**: Using `TanhSourceFunction` for simplified
   analytical work (with explicit acknowledgment of approximation).

---

## Adding New Source Functions

If you need a new source function (for a different device):

1. Implement the `SourceFunctionProtocol` interface
2. Register in `registry.py`
3. Document the physical basis
4. Ensure periodicity and symmetry properties are preserved

---

## Imported By

- `soen_toolkit.hardware_fixed` (re-exports)
- `core/layers/physical/dynamics/` (ODE kernels)
- `core/layers/physical/*.py` (layer implementations)
