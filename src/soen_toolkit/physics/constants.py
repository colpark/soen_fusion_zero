"""Centralized physical constants and helpers for time conversions.

Units:
- Phi0: Weber (Wb)
- Ic: Ampere (A)
- omega_c: angular frequency in rad/s

This module provides a single source of truth for default values used across
the toolkit and convenience helpers for converting between dimensionless and
physical time based on omega_c. Use `set_omega_c` to override the default at
runtime, or set the environment variable `SOEN_OMEGA_C` before import.
"""

from __future__ import annotations

import math
import os
from typing import Union

NumberLike = Union[int, float]


# ---- Fundamental physical constants (SI) ----
# CODATA 2019 exact values (https://physics.nist.gov/cuu/Constants/)
_PLANCK_H_SI = 6.62607015e-34  # Planck constant [J·s]
_ELEMENTARY_CHARGE_E_SI = 1.602176634e-19  # Elementary charge [C]

# Expose named constants
PLANCK_H: float = _PLANCK_H_SI  # Planck constant [J·s]
ELEMENTARY_CHARGE_E: float = _ELEMENTARY_CHARGE_E_SI  # Elementary charge [C]

# Magnetic flux quantum Φ0 = h/(2e)
_PHI0_SI: float = PLANCK_H / (2.0 * ELEMENTARY_CHARGE_E)

# Defaults (can be overridden)
DEFAULT_PHI0: float = _PHI0_SI  # Weber (Wb)
DEFAULT_IC: float = 100e-6  # Ampere (A)
DEFAULT_RJJ: float = 1.2191160341258176  # Ohm (Ω) junction characteristic resistance

# Default base-parameter guesses commonly used by tools (centralized here)
DEFAULT_GAMMA_C: float = 1.5e-9  # Capacitance proportionality (F/A)

# Ensure DEFAULT_BETA_C matches DEFAULT_RJJ, DEFAULT_GAMMA_C and DEFAULT_IC:
# From r_jj^2 = (β_c Φ0) / (2π c_j I_c), with c_j = γ_c I_c →
# β_c = r_jj^2 * (2π γ_c I_c^2) / Φ0
DEFAULT_BETA_C: float = (DEFAULT_RJJ * DEFAULT_RJJ) * (2.0 * math.pi * DEFAULT_GAMMA_C * DEFAULT_IC * DEFAULT_IC) / DEFAULT_PHI0

# Default omega_c computed from defaults: ω_c = (2π I_c R) / Φ0
_DEFAULT_OMEGA_C: float = (2.0 * math.pi * DEFAULT_IC * DEFAULT_RJJ) / DEFAULT_PHI0

# Backing store for current omega_c (rad/s)
_OMEGA_C: float | None = None


def _env_omega_c() -> float | None:
    try:
        value = os.environ.get("SOEN_OMEGA_C")
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def get_omega_c() -> float:
    """Return the current default omega_c (rad/s).

    Resolution order:
    1) Explicitly set via `set_omega_c`
    2) Environment variable `SOEN_OMEGA_C`
    3) Module default `_DEFAULT_OMEGA_C`
    """
    global _OMEGA_C
    if _OMEGA_C is not None:
        return _OMEGA_C
    env = _env_omega_c()
    if env is not None:
        return env
    return _DEFAULT_OMEGA_C


def set_omega_c(value: NumberLike) -> None:
    """Override the default omega_c (rad/s) used across the toolkit.

    Note: Existing objects that captured old values (e.g., layer instances with
    `self.wc`) will not be retroactively updated.
    """
    global _OMEGA_C
    _OMEGA_C = float(value)


def dimensionless_time_to_seconds(t_dimless: NumberLike, omega_c: NumberLike | None = None) -> float:
    """Convert dimensionless time t' to seconds t = t'/omega_c."""
    oc = float(omega_c) if omega_c is not None else get_omega_c()
    return float(t_dimless) / oc


def seconds_to_dimensionless_time(t_seconds: NumberLike, omega_c: NumberLike | None = None) -> float:
    """Convert physical seconds t to dimensionless time t' = t*omega_c."""
    oc = float(omega_c) if omega_c is not None else get_omega_c()
    return float(t_seconds) * oc


def dt_seconds_per_step(dt_dimless: NumberLike, omega_c: NumberLike | None = None) -> float:
    """Seconds per simulation step given dimensionless dt: Δt = dt/omega_c."""
    return dimensionless_time_to_seconds(dt_dimless, omega_c=omega_c)


def dt_nanoseconds_per_step(dt_dimless: NumberLike, omega_c: NumberLike | None = None) -> float:
    """Nanoseconds per simulation step from dimensionless dt."""
    return dt_seconds_per_step(dt_dimless, omega_c=omega_c) * 1e9


__all__ = [
    "DEFAULT_BETA_C",
    "DEFAULT_GAMMA_C",
    "DEFAULT_IC",
    "DEFAULT_PHI0",
    "DEFAULT_RJJ",
    "ELEMENTARY_CHARGE_E",
    "PLANCK_H",
    "dimensionless_time_to_seconds",
    "dt_nanoseconds_per_step",
    "dt_seconds_per_step",
    "get_omega_c",
    "seconds_to_dimensionless_time",
    "set_omega_c",
]
