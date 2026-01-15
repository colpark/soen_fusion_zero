"""Single-dendrite ODE kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping


class SingleDendriteDynamics(nn.Module):
    """Compute ``ds/dt`` for the single-dendrite model."""

    def __init__(self, *, source_function) -> None:
        super().__init__()
        self._source_function = source_function

    def forward(
        self,
        state: torch.Tensor,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        gamma_plus = params["gamma_plus"]
        gamma_minus = params["gamma_minus"]
        bias_current = params["bias_current"]

        # phi has already been adjusted by the solver via compute_phi_with_offset
        phi_eff = phi

        # Always compute squid current for consistency
        squid_current = bias_current - state

        # Handle different source function interfaces
        if getattr(self._source_function, "uses_squid_current", False):
            # Functions that expect squid current (RateArray, HeavisideFitStateDep)
            g_val = self._source_function.g(phi_eff, squid_current=squid_current)
        else:
            # Phi-only functions use total bias (Tanh, etc.)
            g_val = self._source_function.g(phi_eff, squid_current=bias_current)

        return gamma_plus * g_val - gamma_minus * state

    def step(
        self,
        s_prev: torch.Tensor,
        phi_t: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Discrete step function for ParaRNN solver.

        Computes s_{t+1} = s_t + dt * ds/dt using Forward Euler discretization.
        This is equivalent to: s_{t+1} = alpha * s_t + beta * g(phi, squid_current)
        where alpha = (1 - dt * gamma_minus) and beta = dt * gamma_plus.

        Args:
            s_prev: Previous state [B, D]
            phi_t: Input flux at current timestep [B, D]
            params: Parameter dict with gamma_plus, gamma_minus, bias_current
            dt: Timestep size (scalar tensor)

        Returns:
            Next state [B, D]
        """
        gamma_plus = params["gamma_plus"]
        gamma_minus = params["gamma_minus"]
        bias_current = params["bias_current"]

        # Compute squid current
        squid_current = bias_current - s_prev

        # Compute source function value
        if getattr(self._source_function, "uses_squid_current", False):
            g_val = self._source_function.g(phi_t, squid_current=squid_current)
        else:
            g_val = self._source_function.g(phi_t, squid_current=bias_current)

        # Discrete step: s_next = s_prev * (1 - dt*gamma_minus) + dt*gamma_plus*g
        alpha = 1.0 - dt * gamma_minus
        beta = dt * gamma_plus
        return alpha * s_prev + beta * g_val


__all__ = ["SingleDendriteDynamics"]
