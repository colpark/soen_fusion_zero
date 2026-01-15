# Physics Module 🔒 HARDWARE-FIXED

## Classification: DO NOT MODIFY

This module contains fundamental physical constants that define superconducting behavior.

---

## Contents

| File | Contents | Modification Risk |
|------|----------|-------------------|
| `constants.py` | Φ₀, h, e, I_c, R_JJ, ω_c | 🔴 CRITICAL |

---

## Why These Are Fixed

### Universal Physical Constants
```python
PLANCK_H = 6.626×10⁻³⁴ J·s      # Law of physics
ELEMENTARY_CHARGE_E = 1.602×10⁻¹⁹ C  # Law of physics
```
These are fundamental constants of nature. Changing them is physically meaningless.

### Derived Constants
```python
PHI0 = h / (2e) ≈ 2.07×10⁻¹⁵ Wb  # Flux quantum
```
Derived from fundamental constants - changing the derivation would be wrong.

### Device Parameters
```python
DEFAULT_IC = 100 μA   # Critical current from fabrication
DEFAULT_RJJ = 1.22 Ω  # Junction resistance from fabrication
```
These represent specific device properties. Changing them means simulating
a different physical device.

---

## Valid Reasons to Modify

1. **Simulating different hardware**: If you have a different device with
   different I_c, R_JJ values, update these consistently.

2. **New device measurements**: If improved measurements provide better values.

In both cases, ensure all derived quantities (ω_c, β_c) are updated consistently.

---

## Imported By

- `soen_toolkit.hardware_fixed` (re-exports)
- `core/layers/physical/` (dynamics calculations)
- `utils/physical_mappings/` (time conversions)
