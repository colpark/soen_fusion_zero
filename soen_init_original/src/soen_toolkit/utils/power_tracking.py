import torch


def convert_power_to_physical(power_dimensionless: torch.Tensor, Ic: float, Phi0: float, wc: float) -> torch.Tensor:
    """Convert dimensionless power to Watts."""
    return Phi0 * Ic * wc * power_dimensionless / (2 * torch.pi)


def convert_energy_to_physical(energy_dimensionless: torch.Tensor, Ic: float, Phi0: float) -> torch.Tensor:
    """Convert dimensionless energy to Joules."""
    return Phi0 * Ic * energy_dimensionless / (2 * torch.pi)
