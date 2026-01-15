"""Validation helpers shared across layers."""

from __future__ import annotations

import torch


def validate_sequence_input(tensor: torch.Tensor, *, dim: int) -> tuple[int, int, int]:
    """Ensure tensor follows ``[batch, steps, dim]`` convention."""
    if tensor.dim() != 3:
        msg = f"Expected tensor with shape [batch, steps, dim]; received {tuple(tensor.shape)}"
        raise ValueError(
            msg,
        )
    if tensor.shape[-1] != dim:
        msg = f"Input last dimension {tensor.shape[-1]} does not match layer dim {dim}"
        raise ValueError(
            msg,
        )
    return tuple(tensor.shape)  # type: ignore[return-value]


def broadcast_initial_state(
    initial_state: torch.Tensor | None,
    *,
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalise initial state to shape ``[batch, dim]``."""
    if initial_state is None:
        return torch.zeros(batch, dim, device=device, dtype=dtype)

    init = initial_state.to(device=device, dtype=dtype)
    if init.dim() == 1:
        if init.shape[0] != dim:
            msg = f"Initial state length {init.shape[0]} does not match layer dim {dim}"
            raise ValueError(
                msg,
            )
        return init.view(1, -1).expand(batch, -1).clone()
    if init.shape == (batch, dim):
        return init.clone()
    msg = f"Initial state must have shape [dim] or [batch, dim]; received {tuple(init.shape)}"
    raise ValueError(
        msg,
    )


__all__ = ["broadcast_initial_state", "validate_sequence_input"]
