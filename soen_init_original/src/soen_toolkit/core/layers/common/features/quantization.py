"""Quantisation-aware training feature."""

from __future__ import annotations

from typing import TYPE_CHECKING

from soen_toolkit.utils.quantization import ste_snap

from .base import FeatureContext, FeatureHook, StepPayload

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class QuantizationFeature(FeatureHook):
    """Apply straight-through estimator quantisation to selected parameters."""

    def __init__(
        self,
        *,
        param_names: Sequence[str],
        codebook: torch.Tensor,
        stochastic: bool = False,
        active: bool = True,
    ) -> None:
        self._param_names = list(param_names)
        self._codebook = codebook
        self._stochastic = stochastic
        self._active = active

    def on_before_step(
        self,
        *,
        context: FeatureContext,
        step_index: int,
        payload: StepPayload,
    ) -> StepPayload:
        if not self._active:
            return payload
        for name in self._param_names:
            tensor = payload.params.get(name)
            if tensor is None:
                continue
            payload.params[name] = ste_snap( # type: ignore[index]
                tensor,
                self._codebook.to(device=tensor.device, dtype=tensor.dtype),
                stochastic=self._stochastic,
            )
        return payload


__all__ = ["QuantizationFeature"]
