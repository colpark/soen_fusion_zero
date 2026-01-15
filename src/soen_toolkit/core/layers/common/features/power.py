"""Power and energy tracking feature stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import weakref

import torch

from soen_toolkit.utils.power_tracking import (
    convert_energy_to_physical,
    convert_power_to_physical,
)

from .base import FeatureContext, FeatureHook, StepPayload

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class PowerStorage:
    power_bias_dimensionless: torch.Tensor | None = None
    power_diss_dimensionless: torch.Tensor | None = None
    energy_bias_dimensionless: torch.Tensor | None = None
    energy_diss_dimensionless: torch.Tensor | None = None
    power_bias: torch.Tensor | None = None
    power_diss: torch.Tensor | None = None
    energy_bias: torch.Tensor | None = None
    energy_diss: torch.Tensor | None = None


class PowerTrackingFeature(FeatureHook):
    """Accumulate dimensionless power/energy values per timestep."""

    def __init__(self, *, Ic: float, Phi0: float, wc: float) -> None:
        self.Ic = Ic
        self.Phi0 = Phi0
        self.wc = wc
        self._storage = PowerStorage()
        self._bias_power_steps: list[torch.Tensor] = []
        self._diss_power_steps: list[torch.Tensor] = []
        self._layer_ref = None
        self._bulk_ready = False

    def attach_layer(self, layer) -> None:
        self._layer_ref = weakref.ref(layer) # type: ignore[assignment]

    def on_integration_start(self, **kwargs) -> None:
        self._storage = PowerStorage()
        self._bias_power_steps = []
        self._diss_power_steps = []
        self._bulk_ready = False
        layer = self._layer_ref() if self._layer_ref is not None else None
        if layer is not None:
            layer.power_bias_dimensionless = None
            layer.power_diss_dimensionless = None
            layer.energy_bias_dimensionless = None
            layer.energy_diss_dimensionless = None
            layer.power_bias = None
            layer.power_diss = None
            layer.energy_bias = None
            layer.energy_diss = None

    def on_after_step(
        self,
        *,
        context: FeatureContext,
        step_index: int,
        payload: StepPayload,
    ) -> None:
        if payload.ds_dt is None:
            return

        prev_state = payload.extras.get("prev_state")
        if prev_state is None:
            return

        gamma_plus = payload.params.get("gamma_plus")
        gamma_minus = payload.params.get("gamma_minus")
        bias_current = payload.params.get("bias_current")

        if gamma_plus is None or gamma_minus is None or bias_current is None:
            return

        eps = 1e-12
        safe_gp = gamma_plus.clamp_min(eps)
        g_val = (payload.ds_dt + gamma_minus * prev_state) / safe_gp

        payload.extras["g"] = g_val

        # Power bias: g * Ib (total bias current, not effective)
        # Note: bias_current in params is always the total bias current
        power_bias = g_val * bias_current
        # Power dissipation: g * Ib - s * ds/gamma_plus (using post-update state)
        power_diss = power_bias - payload.state * (payload.ds_dt / safe_gp)

        self._bias_power_steps.append(power_bias.detach().clone())
        self._diss_power_steps.append(power_diss.detach().clone())

    def on_scan_batch(
        self,
        *,
        context: FeatureContext,
        history: torch.Tensor,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        observables: torch.Tensor,
        ds_dt: torch.Tensor,
    ) -> bool:
        gamma_plus = params.get("gamma_plus")
        if gamma_plus is None:
            return False
        gamma_minus = params.get("gamma_minus")
        if gamma_minus is None:
            gamma_minus = torch.zeros_like(gamma_plus)
        bias_current = params.get("bias_current")
        if bias_current is None:
            return False

        eps = 1e-12
        prev_state = history[:, :-1, :]
        post_state = history[:, 1:, :]  # Post-update states
        safe_gp = gamma_plus.clamp_min(eps)
        g_val = (ds_dt + gamma_minus * prev_state) / safe_gp

        # Power bias: g * Ib (total bias current, not effective)
        # Note: bias_current in params is always the total bias current
        power_bias_dimless = g_val * bias_current
        # Power dissipation: g * Ib - s * ds/gamma_plus (using post-update state)
        power_diss_dimless = power_bias_dimless - post_state * (ds_dt / safe_gp)

        dt = context.dt.to(device=power_bias_dimless.device, dtype=power_bias_dimless.dtype)
        while dt.dim() < power_bias_dimless.dim():
            dt = dt.unsqueeze(-1)

        energy_bias_dimless = torch.cumsum(power_bias_dimless * dt, dim=1)
        energy_diss_dimless = torch.cumsum(power_diss_dimless * dt, dim=1)

        power_bias = convert_power_to_physical(power_bias_dimless, self.Ic, self.Phi0, self.wc)
        power_diss = convert_power_to_physical(power_diss_dimless, self.Ic, self.Phi0, self.wc)
        energy_bias = convert_energy_to_physical(energy_bias_dimless, self.Ic, self.Phi0)
        energy_diss = convert_energy_to_physical(energy_diss_dimless, self.Ic, self.Phi0)

        self._storage = PowerStorage(
            power_bias_dimensionless=power_bias_dimless,
            power_diss_dimensionless=power_diss_dimless,
            energy_bias_dimensionless=energy_bias_dimless,
            energy_diss_dimensionless=energy_diss_dimless,
            power_bias=power_bias,
            power_diss=power_diss,
            energy_bias=energy_bias,
            energy_diss=energy_diss,
        )

        layer = self._layer_ref() if self._layer_ref is not None else None
        if layer is not None:
            layer.power_bias_dimensionless = power_bias_dimless
            layer.power_diss_dimensionless = power_diss_dimless
            layer.energy_bias_dimensionless = energy_bias_dimless
            layer.energy_diss_dimensionless = energy_diss_dimless
            layer.power_bias = power_bias
            layer.power_diss = power_diss
            layer.energy_bias = energy_bias
            layer.energy_diss = energy_diss

        self._bias_power_steps = []
        self._diss_power_steps = []
        self._bulk_ready = True
        return True

    @property
    def storage(self) -> PowerStorage:
        return self._storage

    def on_integration_end(
        self,
        *,
        context: FeatureContext,
        history: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> None:
        if self._bulk_ready:
            return

        if not self._bias_power_steps:
            return

        power_bias_dimless = torch.stack(self._bias_power_steps, dim=1)
        power_diss_dimless = torch.stack(self._diss_power_steps, dim=1)

        dt = context.dt.to(device=power_bias_dimless.device, dtype=power_bias_dimless.dtype)
        while dt.dim() < power_bias_dimless.dim():
            dt = dt.unsqueeze(-1)

        energy_bias_dimless = torch.cumsum(power_bias_dimless * dt, dim=1)
        energy_diss_dimless = torch.cumsum(power_diss_dimless * dt, dim=1)

        power_bias = convert_power_to_physical(power_bias_dimless, self.Ic, self.Phi0, self.wc)
        power_diss = convert_power_to_physical(power_diss_dimless, self.Ic, self.Phi0, self.wc)
        energy_bias = convert_energy_to_physical(energy_bias_dimless, self.Ic, self.Phi0)
        energy_diss = convert_energy_to_physical(energy_diss_dimless, self.Ic, self.Phi0)

        self._storage = PowerStorage(
            power_bias_dimensionless=power_bias_dimless,
            power_diss_dimensionless=power_diss_dimless,
            energy_bias_dimensionless=energy_bias_dimless,
            energy_diss_dimensionless=energy_diss_dimless,
            power_bias=power_bias,
            power_diss=power_diss,
            energy_bias=energy_bias,
            energy_diss=energy_diss,
        )

        layer = self._layer_ref() if self._layer_ref is not None else None
        if layer is not None:
            layer.power_bias_dimensionless = power_bias_dimless
            layer.power_diss_dimensionless = power_diss_dimless
            layer.energy_bias_dimensionless = energy_bias_dimless
            layer.energy_diss_dimensionless = energy_diss_dimless
            layer.power_bias = power_bias
            layer.power_diss = power_diss
            layer.energy_bias = energy_bias
            layer.energy_diss = energy_diss


__all__ = ["PowerStorage", "PowerTrackingFeature"]
