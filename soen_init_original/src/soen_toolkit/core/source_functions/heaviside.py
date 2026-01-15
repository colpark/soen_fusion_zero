from __future__ import annotations

from typing import NoReturn

import torch

from .base import SourceFunctionBase, SourceFunctionInfo


class HeavisideFitStateDep(SourceFunctionBase):
    info = SourceFunctionInfo(
        key="Heaviside_state_dep",
        title="Heaviside (state dependent)",
        description="State-dependent smooth Heaviside fit.",
        category="SOEN",
        uses_squid_current=True,
        supports_coefficients=False,
    )

    def __init__(self, A=0.37091212, B=0.31903101, C=1.06435066, K=1.92138556, M=2.50322787, N=2.62706077, epsilon=1e-6) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.K = K
        self.M = M
        self.N = N
        self.epsilon = epsilon

    # squid_current is the total current in the integration loop (SQUID CURRENT)
    # this could have popped up elsewhere as well. It's worth checking.
    def g(self, phi: torch.Tensor, *, squid_current: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        # Note: squid_current is the total current through the SQUID part of the dendrite circuit
        if squid_current is None: # we might be able to remove this. When would squid_current ever be None?
            squid_current = torch.full_like(phi, 1.7)
        bias_diff = torch.clamp(squid_current - self.C, min=self.epsilon)
        # Add epsilon to cos_term before fractional power to avoid NaN gradients at cos_term=0
        cos_term = torch.abs(torch.cos(torch.pi * phi)) + self.epsilon
        disc = self.A * (bias_diff**self.K) - self.B * (cos_term**self.M)
        activation = torch.sigmoid(100.0 * disc)
        return activation * (torch.clamp(disc, min=self.epsilon) ** (1 / self.N))

    def get_coefficients(self, *args, **kwargs) -> NoReturn:  # pragma: no cover - not supported
        msg = "Parallel scan integration is not supported for Heaviside fit."
        raise NotImplementedError(msg)
