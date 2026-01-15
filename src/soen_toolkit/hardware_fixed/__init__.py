"""Hardware-Fixed SOEN Components - DO NOT MODIFY.

This package re-exports all physics-rooted components that represent
the actual behavior of SOEN hardware. Modifying these will cause your
simulation to no longer represent real superconducting optoelectronic devices.

Components included:
    - Physical constants (Φ₀, I_c, R_JJ, ω_c)
    - Source function g(φ) implementations
    - Dendritic dynamics ODE kernels
    - Spike forward pass behavior

WARNING: Changes to these modules mean your trained weights
         will NOT transfer correctly to physical hardware.

See: reports/hardware_software_split_architecture.md for full documentation.
"""

from __future__ import annotations

# =============================================================================
# PHYSICAL CONSTANTS
# These are fundamental to superconducting physics - NEVER CHANGE
# =============================================================================
from soen_toolkit.physics.constants import (
    # Universal constants
    PLANCK_H,
    ELEMENTARY_CHARGE_E,
    # Derived constants
    DEFAULT_PHI0,  # Flux quantum Φ₀ = h/(2e)
    # Device fabrication parameters
    DEFAULT_IC,    # Critical current
    DEFAULT_RJJ,   # Junction resistance
    DEFAULT_GAMMA_C,  # Capacitance proportionality
    DEFAULT_BETA_C,   # Stewart-McCumber parameter
    # Time scale
    get_omega_c,   # Characteristic angular frequency
    set_omega_c,
    # Time conversion utilities
    dimensionless_time_to_seconds,
    seconds_to_dimensionless_time,
    dt_nanoseconds_per_step,
    dt_seconds_per_step,
)

# =============================================================================
# SOURCE FUNCTIONS g(φ)
# These encode the SQUID response curve - determined by device physics
# =============================================================================
from soen_toolkit.core.source_functions import (
    RateArray,           # Primary: lookup table from device simulation
    HeavisideFit,        # Simplified step approximation
    HeavisideFitStateDep,  # State-dependent Heaviside
    TanhSourceFunction,  # Analytic approximation
    get_source_function, # Registry lookup
)

# =============================================================================
# DENDRITIC DYNAMICS
# These implement the core ODE: ds/dt = γ⁺g(φ) - γ⁻s
# The STRUCTURE of this equation is fixed by circuit physics
# =============================================================================
from soen_toolkit.core.layers.physical.dynamics import (
    SingleDendriteDynamics,  # Core dendritic ODE kernel
)

# Also expose the full layer implementations that use these dynamics
from soen_toolkit.core.layers.physical import (
    SingleDendrite,      # Full single-dendrite layer
    Multiplier,          # WICC multiplicative synapse layer
    MultiplierV2,        # NOCC multiplicative synapse layer
    Soma,                # Soma/threshold layer
)

# =============================================================================
# SPIKE MECHANISM
# Hard threshold behavior - represents Josephson junction switching
# =============================================================================
from soen_toolkit.ops.spike import (
    spike_torch,  # Forward: (x > threshold) - this behavior is fixed
    spike_jax,    # JAX version
)

# =============================================================================
# MODULE METADATA
# =============================================================================

__all__ = [
    # Physical constants
    "PLANCK_H",
    "ELEMENTARY_CHARGE_E",
    "DEFAULT_PHI0",
    "DEFAULT_IC",
    "DEFAULT_RJJ",
    "DEFAULT_GAMMA_C",
    "DEFAULT_BETA_C",
    "get_omega_c",
    "set_omega_c",
    "dimensionless_time_to_seconds",
    "seconds_to_dimensionless_time",
    "dt_nanoseconds_per_step",
    "dt_seconds_per_step",
    # Source functions
    "RateArray",
    "HeavisideFit",
    "HeavisideFitStateDep",
    "TanhSourceFunction",
    "get_source_function",
    # Dynamics
    "SingleDendriteDynamics",
    # Physical layers
    "SingleDendrite",
    "Multiplier",
    "MultiplierV2",
    "Soma",
    # Spike mechanism
    "spike_torch",
    "spike_jax",
]

# Classification metadata
CLASSIFICATION = "HARDWARE_FIXED"
MODIFICATION_RISK = "CRITICAL"
RATIONALE = """
These components implement the physics of superconducting optoelectronic devices.
They are derived from:
1. Universal physical constants (h, e, Φ₀)
2. Device fabrication parameters (I_c, R_JJ)
3. Measured/simulated device response curves (g(φ))
4. Circuit physics equations (ds/dt = γ⁺g - γ⁻s)

Modifying these means your simulation no longer represents SOEN hardware.
"""
