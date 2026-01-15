from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch


def compute_phi_with_offset(
    phi: torch.Tensor,
    params: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return the total flux presented to the source function.

    Adds the per-dimension ``phi_offset`` parameter when it is present while
    preserving the input tensor's dtype/device and supporting broadcastable
    shapes.
    """
    phi_offset = params.get("phi_offset")
    if phi_offset is None:
        return phi

    offset = phi_offset.to(device=phi.device, dtype=phi.dtype)
    return phi + offset
