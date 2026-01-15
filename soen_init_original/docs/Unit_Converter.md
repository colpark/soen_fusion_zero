---
layout: default
title: Unit Converter
---

# SOEN Phenomenological Model Converter

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Advanced_Features.md" style="margin-right: 2em;">&#8592; Previous: Advanced Features</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Cloud_Training.md" style="margin-left: 2em;">Next: Cloud Training &#8594;</a>
</div>

A short guide and tool for translating between SI units and the dimensionless quantities used in the SOEN phenomenological model.
It summarizes the formulas, shows common conversions, and provides Python and HTTP APIs for programmatic use.
Start by setting the base device parameters ($I_c$, $\gamma_c$, $\beta_c$); the converter derives $c_j$, $r_{jj}$, $\omega_c$, and $\tau_0$ so all conversions are self-consistent. In the phenomenological model for SOENs, all parameters and variables are dimensionless. The equations and conversions are laid out in this document. This tool can be access via the model_creation_gui in the `Unit Conversion` tab. Simply click launch and the unit conversion webpage should embed within the gui. 

---

## Fundamental Constants

All fundamental constants are defined in `soen_toolkit.physics.constants`.

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Magnetic Flux Quantum | $\Phi_0$ | $2.067833848 \times 10^{-15}$ | Wb |
| Elementary Charge | $e$ | $1.602176634 \times 10^{-19}$ | C |
| Planck Constant | $h$ | $6.62607015 \times 10^{-34}$ | J·s |
| Reduced Planck Constant | $\hbar$ | $1.054571817 \times 10^{-34}$ | J·s |

---

## Base Parameters

### Input Parameters

These are the three fundamental parameters that define your superconducting junction:

| Parameter | Symbol | Default Value | Unit | Description |
|-----------|--------|---------------|------|-------------|
| Critical Current | $I_c$ | $100 \times 10^{-6}$ | A | Maximum supercurrent the junction can sustain |
| Capacitance Proportionality | $\gamma_c$ | $1.5 \times 10^{-9}$ | F/A | Junction capacitance per unit critical current |
| Stewart-McCumber Parameter | $\beta_c$ | $0.3$ | - | Dimensionless damping parameter |

---

## Derived Physical Parameters

All derived parameters are calculated from the base parameters above:

### Junction Capacitance
$$c_j = \gamma_c I_c$$

The total capacitance of the Josephson junction.

### Junction Resistance
$$r_{jj} = \sqrt{\frac{\beta_c \Phi_0}{2\pi c_j I_c}}$$

The characteristic resistance of the junction, related to the normal-state resistance.

### Junction Voltage
$$V_j = I_c r_{jj}$$

The characteristic voltage scale of the junction.

### Characteristic Time
$$\tau_0 = \frac{\Phi_0}{2\pi I_c r_{jj}}$$

The fundamental time scale of the junction dynamics.

### Josephson Frequency
$$\omega_c = \frac{2\pi I_c r_{jj}}{\Phi_0}$$

The characteristic angular frequency for Josephson oscillations.

### Plasma Frequency
$$\omega_p = \sqrt{\frac{2\pi}{\Phi_0 \gamma_c}}$$

The natural oscillation frequency of the junction.

### Dimensionless Time Constant
$$\tau = \frac{\beta_L}{\alpha}$$

The ratio of inductance to resistance parameters.

---

## Physical ↔ Dimensionless Conversions

### Physical to Dimensionless

| Physical Quantity | Symbol | Dimensionless Form | Formula |
|-------------------|--------|-------------------|---------|
| Current | $I$ [A] | $i$ | $i = \frac{I}{I_c}$ |
| Flux | $\Phi$ [Wb] | $\phi$ | $\phi = \frac{\Phi}{\Phi_0}$ |
| Inductance | $L$ [H] | $\beta_L$ | $\beta_L = \frac{2\pi I_c L}{\Phi_0}$ |
| Time | $t$ [s] | $t'$ | $t' = t\omega_c$ |
| Leak Resistance | $r_{\text{leak}}$ [Ω] | $\alpha$ | $\alpha = \frac{r_{\text{leak}}}{r_{jj}}$ |
| Flux Quantum Rate | $G_{fq}$ [Hz] | $g_{fq}$ | $g_{fq} = \frac{2\pi G_{fq}}{\omega_c}$ |
| Physical Tau | $\tau_{\text{phys}}$ [s] | $\gamma_-$ | $\gamma_- = \frac{1}{\tau_{\text{phys}} \cdot \omega_c}$ |

### Dimensionless to Physical

| Dimensionless Quantity | Symbol | Physical Form | Formula |
|------------------------|--------|---------------|---------|
| Current | $i$ | $I$ [A] | $I = i I_c$ |
| Flux | $\phi$ | $\Phi$ [Wb] | $\Phi = \phi \Phi_0$ |
| Inductance | $\beta_L$ | $L$ [H] | $L = \frac{\beta_L \Phi_0}{2\pi I_c}$ |
| Inverse Inductance | $\gamma_+ = \frac{1}{\beta_L}$ | $L$ [H] | $L = \frac{\Phi_0}{2\pi \gamma_+ I_c}$ |
| Time | $t'$ | $t$ [s] | $t = \frac{t'}{\omega_c}$ |
| Resistance | $\alpha$ | $r_{\text{leak}}$ [Ω] | $r_{\text{leak}} = \alpha r_{jj}$ |
| Time Constant | $\gamma_- = \frac{\alpha}{\beta_L} = \frac{1}{\tau}$ | $\tau$ | $\tau = \frac{1}{\gamma_-}$ |
| Flux Quantum Rate | $g_{fq}$ | $G_{fq}$ [Hz] | $G_{fq} = \frac{g_{fq}\omega_c}{2\pi}$ |

---

## API Access

### Python API

- Use `soen_toolkit.utils.physical_mappings.soen_conversion_utils.PhysicalConverter`:

```python
from soen_toolkit.utils.physical_mappings.soen_conversion_utils import PhysicalConverter
from soen_toolkit.physics import constants as phys

# Initialize with base parameters (defaults shown)
converter = PhysicalConverter(I_c=phys.DEFAULT_IC, gamma_c=phys.DEFAULT_GAMMA_C, beta_c=phys.DEFAULT_BETA_C)

# Scalar conversions
i = converter.physical_to_dimensionless_current(25e-6)   # I -> i
I = converter.dimensionless_to_physical_current(0.25)    # i -> I

beta_L = converter.physical_to_dimensionless_inductance(1e-9)  # L -> beta_L
gamma_plus = converter.beta_L_to_gamma(beta_L)                  # beta_L -> gamma_+
L = converter.dimensionless_to_physical_inductance(beta_L)      # beta_L -> L

# Time conversions
t_prime = converter.physical_to_dimensionless_time(5e-9)  # t [s] -> t'
t_seconds = converter.dimensionless_to_physical_time(37)  # t' -> t [s]

# Tau / gamma_- / alpha relationships
gamma_minus = 0.02
tau_dimless = converter.gamma_minus_to_tau(gamma_minus)           # 1/gamma_-
tau_seconds = converter.dimensionless_gamma_minus_to_physical_tau(gamma_minus)
alpha = converter.gamma_minus_to_alpha_beta_L(gamma_minus, beta_L)
r_leak = converter.dimensionless_to_physical_resistance(alpha)

# Update base parameters and read back derived values
converter.I_c = 200e-6
params = converter.get_base_parameters()  # dict with I_c, gamma_c, beta_c, c_j, r_jj, omega_c, omega_p, tau_0, V_j
```

- Ergonomic one-shot and batch conversions:

```python
# One value
L = converter.to('L', gamma_plus=0.8)

# Multiple targets at once
out = converter.to(['beta_L', 'L'], gamma_plus=0.8)
# => {'beta_L': 1/0.8, 'L': (beta_L * Phi0)/(2*pi*I_c)}

# Context style with lazy caching
ctx = converter.inputs(gamma_plus=0.8)
beta_L = ctx.beta_L
L = ctx.L
```

- Works with NumPy arrays and PyTorch tensors (shapes preserved):

```python
import numpy as np
gp = np.array([0.5, 1.0, 2.0])
res = converter.convert_many({'gamma_plus': gp}, targets=['beta_L', 'L'])
# res['beta_L'] and res['L'] are numpy arrays matching gp's shape
```

- Time helpers are also available in `soen_toolkit.physics.constants`:

```python
from soen_toolkit.physics import constants as phys

t_prime = phys.seconds_to_dimensionless_time(5e-9)
t_seconds = phys.dimensionless_time_to_seconds(37)
```

### HTTP API (optional)

A small service exposes the converter over HTTP: `python -m soen_toolkit.utils.physical_mappings.main`.

Endpoints (JSON in/out):

- GET `/get_constants`
- POST `/update_base_parameters`
- POST `/convert` (generic batch)
- POST `/convert_to_dimensionless`
- POST `/convert_to_physical`
- POST `/calculate_derived`

Examples (assuming `http://localhost:5001`):

```bash
# Update base parameters
curl -s -X POST http://localhost:5001/update_base_parameters \
  -H 'Content-Type: application/json' \
  -d '{"I_c": 1e-4, "gamma_c": 1.5e-9, "beta_c": 0.3}'

# Generic conversion: gamma_plus -> beta_L, L
curl -s -X POST http://localhost:5001/convert \
  -H 'Content-Type: application/json' \
  -d '{"inputs": {"gamma_plus": [0.5, 1.0, 2.0]}, "targets": ["beta_L", "L"]}'

# Physical -> dimensionless
curl -s -X POST http://localhost:5001/convert_to_dimensionless \
  -H 'Content-Type: application/json' \
  -d '{"I": 2.5e-5, "L": 1e-9, "t": 5e-9, "r_leak": 5.0, "G_fq": 1.0e9, "tau_physical": 1e-9}'

# Dimensionless -> physical
curl -s -X POST http://localhost:5001/convert_to_physical \
  -H 'Content-Type: application/json' \
  -d '{"i": 0.25, "beta_L": 1.2, "t_prime": 37, "alpha": 0.1, "g_fq": 0.2, "gamma_minus": 0.02}'
```

---

## Tips & Best Practices

### Usage Guidelines
- Set base parameters ($I_c$, $\gamma_c$, $\beta_c$) first, then perform conversions
- All conversions use the currently set base parameters
- Default values are for "typical" superconducting dendrites

### Physical Interpretation
- **Current normalization**: $i = 1$ corresponds to the critical current
- **Flux normalization**: $\phi = 1$ corresponds to one flux quantum
- **Time normalization**: Uses the Josephson frequency as the characteristic scale
- **Inductance normalization**: $\beta_L$ represents the ratio of inductive to Josephson energy

### Parameter Relationships
The dimensionless parameters have important relationships:
- $\gamma_+ = \frac{1}{\beta_L}$ (inverse inductance)
- $\gamma_- = \frac{\alpha}{\beta_L}$ (damping rate)
- $\tau = \frac{\beta_L}{\alpha} = \frac{1}{\gamma_-}$ (time constant)

---

## Quick Reference

### Fundamental Relations
- **Flux quantum**: $\Phi_0 = \frac{h}{2e}$ (magnetic flux quantum)
- **Normalized current**: $i = \frac{I}{I_c}$ (current in units of critical current)
- **Normalized flux**: $\phi = \frac{\Phi}{\Phi_0}$ (flux in units of flux quanta)
- **Normalized inductance**: $\beta_L = \frac{2\pi I_c L}{\Phi_0}$ (dimensionless inductance)
- **Inverse inductance**: $\gamma_+ = \frac{1}{\beta_L}$ (alternative parameterization)

### Key Frequencies
- **Josephson frequency**: $\omega_c = \frac{2\pi I_c r_{jj}}{\Phi_0} = \frac{2eI_cr_{jj}}{\hbar}$
- **Plasma frequency**: $\omega_p = \sqrt{\frac{2\pi I_c}{\Phi_0 c_j}} = \sqrt{\frac{1}{LC}}$ analogy

### RCSJ Model Context
The Resistively and Capacitively Shunted Junction (RCSJ) model describes Josephson junction dynamics:
- $\beta_c = \frac{2\pi I_c r_{jj}^2 c_j}{\Phi_0}$ is the Stewart-McCumber parameter
- Overdamped regime: $\beta_c \ll 1$
- Underdamped regime: $\beta_c \gg 1$

---

## About

This tool converts between physical and dimensionless quantities used in the SOEN Phenomenological Model. Use it to calculate parameters for your simulations and to understand the relationships between physical and model quantities.

For typical superconducting dendrites:
- $I_c \sim 100$ µA
- $\gamma_c \sim 1.5$ nF/A
- $\beta_c \sim 0.3$ (overdamped)

<div align="center" style="margin-top: 2em;">
  <a href="Advanced_Features.md" style="margin-right: 2em;">&#8592; Previous: Advanced Features</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Cloud_Training.md" style="margin-left: 2em;">Next: Cloud Training &#8594;</a>
</div>

