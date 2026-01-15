"""Timebase: single source of truth for dt, sampling rate, and time conversions."""

from __future__ import annotations

from dataclasses import dataclass

from soen_toolkit.physics.constants import get_omega_c


@dataclass(frozen=True)
class Timebase:
    """Immutable timebase specification for simulation.

    Provides centralized time conversions between dimensionless and physical units.
    All time calculations should flow through this class to ensure consistency.
    """

    dt: float  # dimensionless time step
    omega_c: float  # characteristic angular frequency (rad/s)

    @property
    def step_seconds(self) -> float:
        """Physical time per step in seconds: Δt = dt / ω_c"""
        return self.dt / self.omega_c

    @property
    def step_ns(self) -> float:
        """Physical time per step in nanoseconds."""
        return self.step_seconds * 1e9

    @property
    def sampling_rate_hz(self) -> float:
        """Sampling rate in Hz: f_s = 1 / Δt"""
        return 1.0 / self.step_seconds if self.step_seconds > 0 else 0.0

    @property
    def nyquist_hz(self) -> float:
        """Nyquist frequency in Hz: f_nyq = f_s / 2"""
        return self.sampling_rate_hz / 2.0

    def total_ns(self, steps: int) -> float:
        """Total simulation time in nanoseconds for given number of steps."""
        return self.step_ns * steps

    def total_seconds(self, steps: int) -> float:
        """Total simulation time in seconds for given number of steps."""
        return self.step_seconds * steps

    @classmethod
    def from_dt(cls, dt: float, omega_c: float | None = None) -> Timebase:
        """Construct timebase from dimensionless dt.

        Args:
            dt: Dimensionless time step
            omega_c: Characteristic frequency (rad/s). If None, uses global default.

        Returns:
            Timebase instance
        """
        oc = omega_c if omega_c is not None else get_omega_c()
        return cls(dt=dt, omega_c=oc)

    @classmethod
    def from_total_ns(cls, total_ns: float, steps: int, omega_c: float | None = None) -> Timebase:
        """Construct timebase from total physical time and step count.

        Computes dt such that: total_ns = steps * (dt / omega_c) * 1e9
        Solving for dt: dt = (total_ns * 1e-9) * omega_c / steps

        Args:
            total_ns: Total simulation time in nanoseconds
            steps: Number of simulation steps
            omega_c: Characteristic frequency (rad/s). If None, uses global default.

        Returns:
            Timebase instance

        Raises:
            ValueError: If steps <= 0 or total_ns < 0
        """
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        if total_ns < 0:
            msg = f"total_ns must be non-negative, got {total_ns}"
            raise ValueError(msg)

        oc = omega_c if omega_c is not None else get_omega_c()
        dt = (total_ns * 1e-9) * oc / steps
        return cls(dt=dt, omega_c=oc)
