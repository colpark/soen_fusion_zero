"""State/flux/source tracking feature stubs."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from .base import FeatureContext, FeatureHook, StepPayload

if TYPE_CHECKING:
    from collections.abc import Mapping


class HistoryTrackingFeature(FeatureHook):
    """Collect sequences of tensors for debugging/visualisation."""

    def __init__(self, *, track_phi: bool = False, track_g: bool = False, track_s: bool = False) -> None:
        self.track_phi = track_phi
        self.track_g = track_g
        self.track_s = track_s
        self._history: dict[str, list[torch.Tensor]] = defaultdict(list)

    def on_integration_start(self, **kwargs) -> None:
        self._history.clear()

    def on_before_step(
        self,
        *,
        context: FeatureContext,
        step_index: int,
        payload: StepPayload,
    ) -> StepPayload:
        if self.track_phi:
            self._history["phi"].append(payload.phi.detach().clone())
        return payload

    def on_after_step(
        self,
        *,
        context: FeatureContext,
        step_index: int,
        payload: StepPayload,
    ) -> None:
        if self.track_s:
            self._history["state"].append(payload.state.detach().clone())
        if self.track_g:
            g_val = payload.extras.get("g")
            if g_val is not None:
                self._history["g"].append(g_val.detach().clone())

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
        handled = False
        if self.track_phi:
            self._history["phi"] = phi.detach().clone() # type: ignore[assignment]
            handled = True
        if self.track_s:
            self._history["state"] = history[:, 1:, :].detach().clone() # type: ignore[assignment]
            handled = True
        if self.track_g:
            self._history["g"] = observables.detach().clone() # type: ignore[assignment]
            handled = True
        return handled

    def histories(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in self._history.items():
            if isinstance(value, torch.Tensor):
                out[key] = value
            elif isinstance(value, list) and value:
                out[key] = torch.stack(value, dim=1)
        return out


__all__ = ["HistoryTrackingFeature"]
