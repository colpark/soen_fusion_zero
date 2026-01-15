# Hardware-Fixed Components 🔒

## ⚠️ DO NOT MODIFY THESE MODULES

This package contains components that implement the **physical behavior of SOEN hardware**.
Modifying these will cause your simulation to no longer represent real superconducting
optoelectronic devices.

---

## What's in Here

| Module | Purpose | Why Fixed |
|--------|---------|-----------|
| **Physical Constants** | Φ₀, I_c, R_JJ, ω_c | Universal constants & device properties |
| **Source Functions** | g(φ) lookup tables | Measured/simulated device response |
| **Dendritic Dynamics** | ds/dt = γ⁺g(φ) - γ⁻s | Circuit physics equations |
| **Spike Mechanism** | Hard threshold behavior | Josephson junction switching |

---

## Consequences of Modification

If you modify these components:

1. ❌ Your simulation **no longer represents SOEN hardware**
2. ❌ Trained weights **won't transfer** to physical devices
3. ❌ Results have **no physical meaning**
4. ❌ You're studying a different system, not SOEN

---

## The Core Physics

### Flux Quantum
```python
Φ₀ = h / (2e) ≈ 2.07 × 10⁻¹⁵ Wb
```
This is the fundamental unit of magnetic flux in superconductors.

### Dendritic ODE
```python
ds/dt = γ⁺ · g(φ) - γ⁻ · s
```
This equation governs how SOEN dendrites integrate signals.

### Source Function
```
g(φ) is periodic with period Φ₀
g(φ) = g(1 - φ)  (mirror symmetry)
```
The shape of g(φ) is determined by device fabrication.

---

## When You Might Need to Change These

The **only** valid reason to modify these is if you're:
1. Simulating a **different physical device** (not standard SOEN)
2. Have **new device measurements** to incorporate
3. Doing **theoretical research** on modified physics

In all cases, document your changes extensively and understand that
you're no longer simulating standard SOEN hardware.

---

## Usage

```python
# Import hardware-fixed components (for reference)
from soen_toolkit.hardware_fixed import (
    DEFAULT_PHI0,           # Flux quantum
    DEFAULT_IC,             # Critical current
    SingleDendriteDynamics, # ODE kernel
    RateArray,              # g(φ) source function
)

# These values should be treated as READ-ONLY
print(f"Flux quantum: {DEFAULT_PHI0} Wb")
```

---

## See Also

- `reports/hardware_software_split_architecture.md` - Full classification
- `reports/hardware_vs_software_parameters.md` - Parameter breakdown
- `reports/code_concept_mapping.md` - Physics ↔ code mapping
