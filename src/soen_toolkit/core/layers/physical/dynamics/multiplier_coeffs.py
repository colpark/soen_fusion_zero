"""Coefficient provider for multiplier parallel scan solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from soen_toolkit.core.layers.common.solvers.parallel_scan import CoefficientProvider

if TYPE_CHECKING:
    from collections.abc import Mapping


class MultiplierCoefficientProvider(CoefficientProvider):
    """Map multiplier ODE into scan coefficients when g has no state dependence."""

    def __init__(self, source_function) -> None:
        self._source_function = source_function

    def coefficients(
        self,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_plus = params["gamma_plus"]
        observable = self._observable(phi, params)
        a = torch.ones_like(phi)
        b = dt * gamma_plus * observable
        return a, b

    def observable(
        self,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._observable(phi, params)

    def _observable(
        self,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        phi_y = params["phi_y"]
        bias = params.get("bias_current", torch.zeros_like(phi))
        phi_a = phi + phi_y
        phi_b = phi - phi_y
        sf = self._source_function
        g_a = sf.g(phi_a, squid_current=bias)
        g_b = sf.g(phi_b, squid_current=bias)
        return g_a - g_b


__all__ = ["MultiplierCoefficientProvider"]
