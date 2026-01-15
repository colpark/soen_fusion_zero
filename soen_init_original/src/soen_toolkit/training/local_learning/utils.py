"""Utility functions for local learning."""

from soen_toolkit.utils.physical_mappings.soen_conversion_utils import PhysicalConverter


def get_dt_for_time_constant(time_constant_ns: float) -> float:
    """Convert physical time constant to dimensionless dt.

    Args:
        time_constant_ns: Time constant in nanoseconds (e.g., 0.1 for 0.1ns)

    Returns:
        Dimensionless dt value for use in simulation config

    Example:
        >>> dt = get_dt_for_time_constant(0.1)  # 0.1 ns
        >>> print(f"dt = {dt:.2f}")  # dt ≈ 37.0
    """
    converter = PhysicalConverter()
    time_constant_seconds = time_constant_ns * 1e-9
    return converter.physical_to_dimensionless_time(time_constant_seconds)
