"""Synapse layer (spike -> analog smoothing) - skeleton implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from soen_toolkit.core.layers.common import Constraint, InitializerSpec, ParameterDef, SoenLayerBase, validate_sequence_input

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parameter_defs(dim: int) -> Sequence[ParameterDef]:
    return (
        ParameterDef(
            name="alpha",
            default=0.9,
            constraint=Constraint(min=0.0, max=1.0),
            initializer=InitializerSpec(method="constant", params={"value": 0.9}),
        ),
    )


class SynapseLayer(SoenLayerBase):
    """First-order low-pass filter applied to spike trains.

    y_{t+1} = alpha * y_t + (1 - alpha) * z_t
    """

    layer_type = "Synapse"
    _is_placeholder: bool = True
    _placeholder_reason: str = "Synapse dynamics/state handling not finalized yet."

    def __init__(self, *, dim: int, dt: float | torch.Tensor) -> None:
        super().__init__(dt=dt, dim=dim, parameters=_parameter_defs(dim))

    def forward(self, spikes: torch.Tensor, *, initial_state: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError(
            "SynapseLayer is a placeholder skeleton and is not implemented yet.\n\n"
            "The current class exists only to reserve the API surface (params, registry integration, JAX port wiring).\n"
            "Do not use it for training/simulation yet: synapse dynamics/units and state handling are not finalized."
        )

        validate_sequence_input(spikes, dim=self.dim)
        B, T, D = spikes.shape
        if D != self.dim:
            raise ValueError(f"SynapseLayer expected dim={self.dim}, got {D}")

        params = self.parameter_values()
        alpha = params["alpha"]

        if initial_state is None:
            y0 = torch.zeros((B, D), device=spikes.device, dtype=spikes.dtype)
        else:
            y0 = initial_state

        ys = []
        y = y0
        one_minus_alpha = 1.0 - alpha
        for t in range(T):
            z = spikes[:, t, :]
            y = alpha * y + one_minus_alpha * z
            ys.append(y)
        y_hist = torch.stack([y0, *ys], dim=1)
        return y_hist


__all__ = ["SynapseLayer"]

