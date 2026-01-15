from __future__ import annotations

import warnings

import torch


def adjust_first_layer_flux(external_phi: torch.Tensor, first_dim: int, *, warn: bool = True) -> torch.Tensor:
    """Pad/slice external flux to match first-layer dim."""
    if external_phi.shape[-1] == first_dim:
        return external_phi
    if external_phi.shape[-1] < first_dim:
        pad_amount = first_dim - external_phi.shape[-1]
        padding = torch.zeros(*external_phi.shape[:-1], pad_amount, device=external_phi.device, dtype=external_phi.dtype)
        return torch.cat([external_phi, padding], dim=-1)
    if warn:
        warnings.warn(
            f"External input for 'flux' mode has {external_phi.shape[-1]} channels, but first layer has dim {first_dim}. Slicing input.",
            UserWarning,
            stacklevel=2,
        )
    return external_phi[..., :first_dim]


def adjust_first_layer_state(external_phi: torch.Tensor, first_dim: int, *, warn: bool = True) -> torch.Tensor:
    """Pad/slice external state to match first-layer dim."""
    if external_phi.shape[-1] == first_dim:
        return external_phi
    if external_phi.shape[-1] < first_dim:
        pad_amount = first_dim - external_phi.shape[-1]
        padding = torch.zeros(*external_phi.shape[:-1], pad_amount, device=external_phi.device, dtype=external_phi.dtype)
        return torch.cat([external_phi, padding], dim=-1)
    if warn:
        warnings.warn(
            f"External input for 'state' mode has {external_phi.shape[-1]} channels, but first layer has dim {first_dim}. Slicing input.",
            UserWarning,
            stacklevel=2,
        )
    return external_phi[..., :first_dim]
