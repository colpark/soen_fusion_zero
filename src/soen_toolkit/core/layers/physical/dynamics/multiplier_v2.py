"""Multiplier V2 layer dynamics with dual SQUID states and aggregated output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping


# Smoothing parameter for soft_abs: controls the "sharpness" of the abs approximation
# Smaller values = closer to true abs but sharper transition
# Larger values = smoother but less accurate near zero
_SOFT_ABS_EPS = 0.01


def soft_abs(x: torch.Tensor, eps: float = _SOFT_ABS_EPS) -> torch.Tensor:
    """Smooth approximation of abs(x) using sqrt(x² + ε).

    Unlike torch.abs(), this function has a continuous derivative everywhere,
    which is important for numerical stability in Forward Euler integration.
    The derivative at x=0 is 0 (vs undefined for abs).

    Args:
        x: Input tensor
        eps: Smoothing parameter (default: 0.01)

    Returns:
        Smooth approximation of abs(x)
    """
    return torch.sqrt(x * x + eps)


class MultiplierNOCCDynamics(nn.Module):
    """Compute coupled ODEs for multiplier v2 circuit.

    This version uses a new flux collection mechanism with:
    - Two SQUID states per edge: s1_ij (left branch) and s2_ij (right branch)
    - One aggregated output state per node: m_i (post fan-in)
    - Coupled dynamics: solve for dot_m first, then use it to solve SQUID states

    ODEs:
        (beta + 2*N*beta_out) * dot_m_i = sum_j(g1_ij + g2_ij) - alpha * m_i
        beta * dot_s1_ij = g1_ij - beta_out * dot_m_i - alpha * s1_ij
        beta * dot_s2_ij = g2_ij - beta_out * dot_m_i - alpha * s2_ij

    where:
        g1_ij = g(phi_x + phi_y, squid_current=bias_current - s1_ij)
        g2_ij = g(phi_x - phi_y, squid_current=-bias_current + s2_ij)
    """

    def __init__(self, *, source_function) -> None:
        super().__init__()
        self._source_function = source_function

    def forward(
        self,
        state: MultiplierNOCCState,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> MultiplierNOCCState:
        """Compute time derivatives for all three state components.

        Args:
            state: MultiplierNOCCState with s1, s2, m tensors
            phi: Input flux [B, D] where D is number of nodes
            params: Dictionary containing:
                - phi_y: Weight flux [B, D] or [D]
                - alpha: Dimensionless resistance (scalar or per-node)
                - beta: Inductance of incoming branches (scalar or per-node)
                - beta_out: Inductance of output branch (scalar or per-node)
                - bias_current: Bias current (scalar or per-node)
                - fan_in: Number of incoming edges per node [D] (optional, defaults to 1)
                - internal_J: Internal connectivity matrix [D, D] (optional)

        Returns:
            MultiplierNOCCState with dot_s1, dot_s2, dot_m
        """
        phi_y = params["phi_y"]
        alpha = params["alpha"]
        beta = params["beta"]
        beta_out = params["beta_out"]
        bias_current = params["bias_current"]

        # Extract state components
        s1 = state.s1  # [B, D] - left branch SQUID states
        s2 = state.s2  # [B, D] - right branch SQUID states
        m = state.m  # [B, D] - aggregated output states

        # Compute phi inputs (phi_a and phi_b for the two branches)
        phi_a = phi + phi_y  # [B, D]
        phi_b = phi - phi_y  # [B, D]

        # Compute source functions for both branches
        sf = self._source_function
        if getattr(sf, "uses_squid_current", False):
            # State-dependent source functions
            # NOTE: The v2 circuit specification uses squid_current_2 = -bias_current + s2
            # which can be negative. However, RateArray only supports positive currents (0.95 to 2.5).
            # We use soft_abs() (smooth approximation of abs) to enable RateArray lookups while
            # maintaining a continuous derivative for numerical stability in Forward Euler.
            # This is physically justified if RateArray is symmetric around current=0.
            squid_current_1 = bias_current - s1
            squid_current_2 = -bias_current + s2
            g1 = sf.g(phi_a, squid_current=soft_abs(squid_current_1))
            g2 = sf.g(phi_b, squid_current=soft_abs(squid_current_2))
        else:
            # Phi-only source functions, i.e if g is not dependent on s
            g1 = sf.g(phi_a, squid_current=bias_current)
            g2 = sf.g(phi_b, squid_current=-bias_current)

        # Get fan-in (default to 1 if not provided)
        fan_in = params.get("fan_in")
        if fan_in is None:
            fan_in = torch.ones_like(m[0])  # [D]

        # Step 1: Compute dot_m using aggregated source terms
        # (beta + 2*N*beta_out) * dot_m_i = sum_j(g1_ij - g2_ij) - alpha * m_i
        # Use subtraction to reflect opposite branch orientation
        g_sum = g1 - g2  # [B, D]

        # Apply internal connectivity if present to compute actual fan-in contribution
        if "internal_J" in params:
            # For internal connections, we need to gather contributions
            # This is handled at the layer level, here we just use the provided g_sum
            pass

        # Compute effective dimensionless inductance: beta + 2*N*beta_out
        # fan_in should be [D], broadcast to [B, D]
        fan_in_bc = fan_in.view(1, -1) if fan_in.dim() == 1 else fan_in
        beta_eff = beta + 2 * fan_in_bc * beta_out

        # Solve for dot_m
        dot_m = (g_sum - alpha * m) / beta_eff  # [B, D]

        # Step 2: Compute dot_s1 and dot_s2 using dot_m
        # beta * dot_s1_ij = g1_ij - beta_out * dot_m_i - alpha * s1_ij
        # beta * dot_s2_ij = g2_ij - beta_out * dot_m_i - alpha * s2_ij
        dot_s1 = (g1 - beta_out * dot_m - alpha * s1) / beta  # [B, D]
        dot_s2 = (g2 - beta_out * dot_m - alpha * s2) / beta  # [B, D]

        return MultiplierNOCCState(s1=dot_s1, s2=dot_s2, m=dot_m)


class MultiplierNOCCState:
    """Container for the three state components of multiplier v2.

    Attributes:
        s1: Left branch SQUID states [B, D] or [B, E] for edges
        s2: Right branch SQUID states [B, D] or [B, E] for edges
        m: Aggregated output states [B, D]
    """

    def __init__(self, s1: torch.Tensor, s2: torch.Tensor, m: torch.Tensor) -> None:
        self.s1 = s1
        self.s2 = s2
        self.m = m

    def __add__(self, other: MultiplierNOCCState) -> MultiplierNOCCState:
        """Add two states element-wise."""
        return MultiplierNOCCState(
            s1=self.s1 + other.s1,
            s2=self.s2 + other.s2,
            m=self.m + other.m,
        )

    def __mul__(self, scalar: float | torch.Tensor) -> MultiplierNOCCState:
        """Multiply state by scalar."""
        return MultiplierNOCCState(
            s1=self.s1 * scalar,
            s2=self.s2 * scalar,
            m=self.m * scalar,
        )

    __rmul__ = __mul__


__all__ = ["MultiplierNOCCDynamics", "MultiplierNOCCState"]
