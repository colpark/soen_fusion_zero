"""Readout layer for layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import torch

from soen_toolkit.core.layers.common import (
    Constraint,
    InitializerSpec,
    ParameterDef,
    PhysicalLayerBase,
)
from soen_toolkit.core.noise import build_noise_strategies
from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parameter_defs(dim: int) -> Sequence[ParameterDef]:
    return (
        ParameterDef(
            name="phi_offset",
            default=0.0,
            initializer=InitializerSpec(method="constant", params={"value": 0.0}),
        ),
        ParameterDef(
            name="bias_current",
            default=1.7,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="constant", params={"value": 1.7}),
        ),
    )


class ReadoutLayer(PhysicalLayerBase):
    """Static readout mapping φ → g(φ) each timestep."""

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        source_func_type: str = "RateArray",
    ) -> None:
        super().__init__(dt=dt, dim=dim, parameters=_parameter_defs(dim))
        self._source_function = self._build_source_function(source_func_type)

    def solver(self) -> NoReturn:
        msg = "ReadoutLayer does not use a solver"
        raise RuntimeError(msg)

    def forward(self, phi: torch.Tensor, *, noise_config=None) -> torch.Tensor:  # type: ignore[override]
        self.validate_input(phi)
        self.apply_parameter_constraints()

        self.feature_stack.set_noise_config(noise_config)
        self._on_noise_config_updated(noise_config)

        params = self.parameter_values()
        phi_offset = params["phi_offset"]
        bias_current = params["bias_current"]
        noise = build_noise_strategies(noise_config)

        phi = phi + phi_offset
        phi = noise.apply(phi, "phi")

        g_val = self._source_function.g(phi, squid_current=bias_current)
        g_val = noise.apply(g_val, "g")
        g_val = noise.apply(g_val, "s")

        batch, _steps, dim = g_val.shape
        zeros = torch.zeros(batch, 1, dim, device=g_val.device, dtype=g_val.dtype)
        return torch.cat([zeros, g_val], dim=1)

    def _build_source_function(self, func_type: str):
        if func_type not in SOURCE_FUNCTIONS:
            msg = f"Unknown source function type '{func_type}'"
            raise ValueError(msg)
        return SOURCE_FUNCTIONS[func_type]()


__all__ = ["ReadoutLayer"]
