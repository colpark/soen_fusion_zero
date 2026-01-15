"""Feature hook interfaces for layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableMapping

    import torch


class FeatureContext(Protocol):
    """Minimal context a solver provides to feature hooks."""

    @property
    def dt(self) -> torch.Tensor:  # pragma: no cover - protocol definition
        ...


@dataclass(slots=True)
class StepPayload:
    """Data exchanged between solver and feature hooks at each timestep."""

    state: torch.Tensor
    phi: torch.Tensor
    params: Mapping[str, torch.Tensor]
    ds_dt: torch.Tensor | None = None
    extras: MutableMapping[str, torch.Tensor] = field(default_factory=dict)


class FeatureHook:
    """Base class for optional behaviours (noise, tracking, etc.)."""

    def on_integration_start(
        self,
        *,
        context: FeatureContext,
        state: torch.Tensor,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> None:
        """Called once before timesteps are processed."""

    def on_before_step(
        self,
        *,
        context: FeatureContext,
        step_index: int,
        payload: StepPayload,
    ) -> StepPayload:
        """Called before dynamics evaluation; may mutate payload or return replacement."""
        return payload

    def on_after_step(
        self,
        *,
        context: FeatureContext,
        step_index: int,
        payload: StepPayload,
    ) -> None:
        """Called after state update; may record tracking tensors."""

    def on_integration_end(
        self,
        *,
        context: FeatureContext,
        history: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> None:
        """Called once integration is complete."""

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
        """Optional bulk-processing hook.

        Return ``True`` when the feature handled the entire batch and no per-step
        callbacks are required. Default implementation requests a fallback.
        """
        return False


class CompositeFeature(FeatureHook):
    """Utility to fan out hook calls to child features."""

    def __init__(self, features: Iterable[FeatureHook] | None = None) -> None:
        self._features = list(features or [])
        self._layer = None

    def add(self, feature: FeatureHook) -> None:
        self._features.append(feature)
        if self._layer is not None:
            attach = getattr(feature, "attach_layer", None)
            if callable(attach):
                attach(self._layer)

    def prepend(self, feature: FeatureHook) -> None:
        self._features.insert(0, feature)
        if self._layer is not None:
            attach = getattr(feature, "attach_layer", None)
            if callable(attach):
                attach(self._layer)

    def clone(self) -> CompositeFeature:
        clone = CompositeFeature(list(self._features))
        clone._layer = self._layer
        if self._layer is not None:
            clone.attach_layer(self._layer)
        return clone

    def has_feature(self, feature_type: type) -> bool:
        return any(isinstance(feature, feature_type) for feature in self._features)

    def set_noise_config(self, noise_config) -> None:
        for feature in self._features:
            setter = getattr(feature, "set_noise_config", None)
            if callable(setter):
                setter(noise_config)
        if self._layer is not None and hasattr(self._layer, "_on_noise_config_updated"):
            self._layer._on_noise_config_updated(noise_config)

    def attach_layer(self, layer) -> None:
        self._layer = layer
        for feature in self._features:
            attach = getattr(feature, "attach_layer", None)
            if callable(attach):
                attach(layer)

    def on_integration_start(self, **kwargs) -> None:
        for feature in self._features:
            feature.on_integration_start(**kwargs)

    def on_before_step(self, **kwargs) -> StepPayload:
        payload = kwargs.pop("payload")
        context = kwargs.pop("context")
        step_index = kwargs.pop("step_index")
        for feature in self._features:
            payload = feature.on_before_step(
                context=context,
                step_index=step_index,
                payload=payload,
            )
        return payload

    def on_after_step(self, **kwargs) -> None:
        for feature in self._features:
            feature.on_after_step(**kwargs)

    def on_integration_end(self, **kwargs) -> None:
        for feature in self._features:
            feature.on_integration_end(**kwargs)

    def on_scan_batch(self, **kwargs) -> bool:
        handled = True
        for feature in self._features:
            if not feature.on_scan_batch(**kwargs):
                handled = False
        return handled


__all__ = ["CompositeFeature", "FeatureContext", "FeatureHook", "StepPayload"]
