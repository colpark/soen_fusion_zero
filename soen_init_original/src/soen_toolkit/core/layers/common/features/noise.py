"""Noise and perturbation feature implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from soen_toolkit.core.noise import NoiseSettings, build_noise_strategies

from .base import FeatureContext, FeatureHook, StepPayload

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch


class NoiseFeature(FeatureHook):
    """Apply noise or perturbation strategies to tensors during integration."""

    def __init__(self, noise_config: Mapping[str, object] | None = None, *, layer_type: str | None = None) -> None:
        self._config = noise_config
        self._noise: NoiseSettings | None = None
        self._param_baseline: dict[str, torch.Tensor] = {}
        self._layer_type = layer_type
        self._layer_ref = None

    def set_noise_config(self, config: Mapping[str, object] | NoiseSettings | None) -> None:
        self._config = config # type: ignore[assignment]
        self._noise = None

    def attach_layer(self, layer) -> None:
        self._layer_ref = layer
        if getattr(layer, "layer_type", None) is not None:
            self._layer_type = layer.layer_type

    def on_before_step(
        self,
        *,
        context: FeatureContext,
        step_index: int,
        payload: StepPayload,
    ) -> StepPayload:
        if self._noise is None:
            return payload

        payload.phi = self._noise.apply(payload.phi, "phi")

        for name, base in self._param_baseline.items():
            payload.params[name] = self._noise.apply(base, name) # type: ignore[index]

        g_val = payload.extras.get("g")
        if g_val is not None:
            payload.extras["g"] = self._noise.apply(g_val, "g")

        return payload

    def on_after_step(
        self,
        *,
        context: FeatureContext,
        step_index: int,
        payload: StepPayload,
    ) -> None:
        if self._noise is None:
            return
        payload.state = self._noise.apply(payload.state, "s")

    def on_integration_start(
        self,
        *,
        context: FeatureContext,
        state: torch.Tensor,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> None:
        self._prepare_noise(context=context, state=state, phi=phi, params=params)

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
        # When no noise is configured we do not need per-step callbacks and can stay vectorised.
        return self._noise is None

    def _prepare_noise(
        self,
        *,
        context: FeatureContext,
        state: torch.Tensor,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> None:
        if self._config is None:
            self._noise = None
            self._param_baseline = {}
            return
        if isinstance(self._config, NoiseSettings):
            settings = self._config
        else:
            settings = build_noise_strategies(self._config) # type: ignore[assignment,arg-type]
        if settings is not None and settings.is_trivial():
            self._noise = None
            self._param_baseline = {}
            return
        self._noise = settings
        self._param_baseline = {name: tensor.clone() for name, tensor in params.items()}
        if self._noise is not None:
            self._noise.reset()


__all__ = ["NoiseFeature"]
