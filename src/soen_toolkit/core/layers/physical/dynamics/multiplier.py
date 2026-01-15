"""Multiplier layer dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping


class MultiplierDynamics(nn.Module):
    """Compute ``ds/dt`` for the multiplier layer."""

    def __init__(self, *, source_function) -> None:
        super().__init__()
        self._source_function = source_function

    def forward(
        self,
        state: torch.Tensor,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        phi_y = params["phi_y"]
        gamma_plus = params["gamma_plus"]
        gamma_minus = params["gamma_minus"]
        bias_current = params["bias_current"]

        phi_a = phi + phi_y
        phi_b = phi - phi_y

        sf = self._source_function
        if getattr(sf, "uses_squid_current", False):
            squid_current_a = bias_current - state
            squid_current_b = bias_current + state
            g_a = sf.g(phi_a, squid_current=squid_current_a)
            g_b = sf.g(phi_b, squid_current=squid_current_b)
        else:
            g_a = sf.g(phi_a, squid_current=bias_current)
            g_b = sf.g(phi_b, squid_current=bias_current)

        return gamma_plus * (g_a - g_b) - gamma_minus * state


__all__ = ["MultiplierDynamics"]
