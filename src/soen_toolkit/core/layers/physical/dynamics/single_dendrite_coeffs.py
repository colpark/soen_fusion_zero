"""Coefficient provider for the single-dendrite parallel scan solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from soen_toolkit.core.layers.common.solvers.parallel_scan import CoefficientProvider

if TYPE_CHECKING:
    from collections.abc import Mapping


class SingleDendriteCoefficientProvider(CoefficientProvider):
    """Translate the single-dendrite ODE into scan coefficients.

    ``source_function`` is the same object the legacy FE solver uses.  We reuse
    its ``g`` evaluation while mapping the recurrence parameters into the
    solver-neutral ``a`` / ``b`` representation.
    """

    def __init__(self, source_function) -> None:
        self._source_function = source_function

    def coefficients(
        self,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi_eff, bias = self._prepare_phi_and_bias(phi, params)
        gamma_plus = params["gamma_plus"]
        gamma_minus = params["gamma_minus"]

        # Parallel scan for phi-only or simple functions
        if getattr(self._source_function, "uses_squid_current", False):
            # Squid-current functions are not PS compatible
            msg = "Parallel scan does not support state dependent source functions"
            raise RuntimeError(msg)

        # Phi-only functions (bias_current is total bias for these)
        observable = self._source_function.g(phi_eff, squid_current=bias)
        a = 1.0 - dt * gamma_minus
        b = dt * gamma_plus * observable
        return a, b

    def observable(
        self,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        phi_eff, bias = self._prepare_phi_and_bias(phi, params)
        # For parallel scan, only phi-only functions are supported
        return self._source_function.g(phi_eff, squid_current=bias)

    @staticmethod
    def _prepare_phi_and_bias(
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias = params.get("bias_current", torch.zeros_like(phi))
        # phi has already been adjusted by the solver via compute_phi_with_offset
        return phi, bias


__all__ = ["SingleDendriteCoefficientProvider"]
