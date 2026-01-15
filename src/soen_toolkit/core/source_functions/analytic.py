# FILEPATH: src/soen_toolkit/core/source_functions/analytic.py

from __future__ import annotations

import torch

from .base import SourceFunctionBase, SourceFunctionInfo, default_coefficients


class _AnalyticSource(SourceFunctionBase):
    _abstract = True

    def get_coefficients(self, phi, gamma_plus, gamma_minus, dt, **kwargs):
        return default_coefficients(phi, gamma_plus, gamma_minus, dt, self.g(phi, **kwargs))


class TanhSourceFunction(_AnalyticSource):
    info = SourceFunctionInfo(
        key="Tanh",
        title="Tanh",
        description="Hyperbolic tangent nonlinearity.",
        category="Analytic",
    )

    def g(self, phi: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.tanh(phi)


class TeLUSourceFunction(_AnalyticSource):
    info = SourceFunctionInfo(
        key="Telu",
        title="TeLU",
        description="Tanh-exp linear unit approximation.",
        category="Analytic",
    )

    def g(self, phi: torch.Tensor, **kwargs) -> torch.Tensor:
        return phi * torch.tanh(torch.exp(phi))


class SimpleGELUSourceFunction(_AnalyticSource):
    info = SourceFunctionInfo(
        key="SimpleGELU",
        title="Simple GELU",
        description="Simplified GELU approximation.",
        category="Analytic",
    )

    def g(self, phi: torch.Tensor, **kwargs) -> torch.Tensor:
        return 0.5 * phi * torch.tanh(0.8 * phi) + 0.5 * phi


class ReLUSourceFunction(_AnalyticSource):
    info = SourceFunctionInfo(
        key="ReLU",
        title="ReLU",
        description="Rectified Linear Unit nonlinearity.",
        category="Analytic",
    )

    def g(self, phi: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.relu(phi)


class TanhGauss1p7IBFitSourceFunction(_AnalyticSource):
    info = SourceFunctionInfo(
        key="TanhGauss1p7IBFit",
        title="Tanh Gaussian 1.7 IB Fit",
        description="Single-bias (≈1.7) tanh(Gaussian) approximation of rate array; clamps squid_current to [0.7, 1.7].",
        category="Analytic",
        uses_squid_current=True,
        supports_coefficients=False,
    )

    def __init__(
        self,
        *,
        sigma: float = 0.12632733583450317,
        s_power: float = 2.5484097003936768,
        amp: float = 0.7845368385314941,
        bias_center: float = 1.7,
        bias_span: float = 1.0,
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.s_power = float(s_power)
        self.amp = float(amp)
        self.bias_center = float(bias_center)
        self.bias_span = float(bias_span)

    def g(self, phi: torch.Tensor, *, squid_current: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Assumes squid_current in [bias_center - bias_span, bias_center].
        Maps squid_current -> s via s = bias_center - squid_current, then applies tanh(Gaussian) in phi.
        """
        if squid_current is None:
            squid_current = torch.full_like(phi, self.bias_center)

        bias_low = self.bias_center - self.bias_span
        bias_high = self.bias_center

        # Softly clamp squid_current to valid range
        bias_clamped = torch.clamp(squid_current, bias_low, bias_high)

        s_val = torch.clamp(self.bias_center - bias_clamped, 0.0, 1.0)
        # Add epsilon to base to avoid NaN gradients for fractional power at 0
        s_term = torch.clamp(1.0 - s_val, min=1e-6) ** self.s_power

        phi_mod = torch.remainder(phi, 1.0)
        gauss = torch.exp(-((phi_mod - 0.5) ** 2) / (2.0 * self.sigma * self.sigma))
        phi_term = torch.tanh(gauss)
        return self.amp * s_term * phi_term
