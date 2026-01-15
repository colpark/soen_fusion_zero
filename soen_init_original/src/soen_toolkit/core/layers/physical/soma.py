"""Soma layer (spiking) - skeleton implementation.

Conceptually:
- Continuous dendrite-like dynamics (same shape contract as SingleDendrite)
- Spike emission head: hard threshold with surrogate gradients
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from soen_toolkit.core.layers.common import (
    Constraint,
    FeatureHook,
    ForwardEulerSolver,
    InitializerSpec,
    ParameterDef,
    PhysicalLayerBase,
)
from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS
from soen_toolkit.ops.spike import spike_torch
from soen_toolkit.ops.surrogates import SurrogateSpec

from .dynamics import SingleDendriteDynamics

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _parameter_defs(dim: int) -> Sequence[ParameterDef]:
    return (
        ParameterDef(
            name="phi_offset",
            default=0.23,
            initializer=InitializerSpec(method="constant", params={"value": 0.23}),
        ),
        ParameterDef(
            name="bias_current",
            default=1.7,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="constant", params={"value": 1.7}),
        ),
        ParameterDef(
            name="gamma_plus",
            default=0.001,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="fan_out"),
            transform="log",
        ),
        ParameterDef(
            name="gamma_minus",
            default=0.001,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="constant", params={"value": 0.001}),
            transform="log",
        ),
        ParameterDef(
            name="threshold",
            default=0.0,
            initializer=InitializerSpec(method="constant", params={"value": 0.0}),
        ),
    )


class SomaLayer(PhysicalLayerBase):
    """Spiking soma layer (skeleton).

    This intentionally reuses the SingleDendrite dynamics for the continuous state
    evolution; the defining behavior is the spike head and its surrogate gradient.
    """

    layer_type = "Soma"
    _is_placeholder: bool = True
    _placeholder_reason: str = "Soma coupling / threshold circuit physics not implemented yet."

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        features: Iterable[FeatureHook] | None = None,
        source_func_type: str = "RateArray",
        surrogate: SurrogateSpec | None = None,
    ) -> None:
        super().__init__(
            dt=dt,
            dim=dim,
            parameters=_parameter_defs(dim),
            init_context={},
            features=features,
        )
        # For now, reuse the same continuous dynamics as SingleDendrite.
        self._source_function = self._build_source_function(source_func_type)
        self._dynamics = SingleDendriteDynamics(source_function=self._source_function)
        self._solver = ForwardEulerSolver(dynamics=self._dynamics, feature=self.feature_stack, layer=self)
        self.surrogate = surrogate or SurrogateSpec()

    def forward(  # type: ignore[override]
        self,
        phi: torch.Tensor,
        *,
        noise_config=None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "SomaLayer is a placeholder skeleton and is not implemented yet.\n\n"
            "The current class exists only to reserve the API surface (params, registry integration, JAX port wiring).\n"
            "Do not use it for training/simulation yet: the soma coupling / threshold circuit physics are not defined."
        )

        self.validate_input(phi)
        initial_state_tensor = self.prepare_initial_state(phi, initial_state)

        self.apply_parameter_constraints()
        params = self.parameter_values()

        self.feature_stack.set_noise_config(noise_config)
        self._on_noise_config_updated(noise_config)

        # Continuous state history [B, T+1, D]
        state_hist = self._solver.integrate(
            state=_StateWrapper(initial_state_tensor),
            phi=phi,
            params=params,
            dt=self.dt,
        )

        # Spike head: hard threshold, surrogate gradients
        thr = params["threshold"]
        return spike_torch(state_hist, threshold=thr, surrogate=self.surrogate)

    @staticmethod
    def _build_source_function(func_type: str):
        if func_type not in SOURCE_FUNCTIONS:
            raise ValueError(f"Unknown source function type '{func_type}'")
        return SOURCE_FUNCTIONS[func_type]()


class _StateWrapper:
    """Minimal state wrapper matching solver expectations (same as SingleDendrite pattern)."""

    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

