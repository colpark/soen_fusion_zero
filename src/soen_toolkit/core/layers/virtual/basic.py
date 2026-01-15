"""Basic virtual layers (linear, scaling, non-linear) for layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn
import warnings

import torch

from soen_toolkit.core.layers.common import (
    Constraint,
    FeatureHook,
    InitializerSpec,
    ParameterDef,
    PhysicalLayerBase,
)
from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class LinearLayer(PhysicalLayerBase):
    """Pass-through linear layer supporting optional noise/perturb hooks."""

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        features: Sequence[FeatureHook] | None = None,
    ) -> None:
        super().__init__(dt=dt, dim=dim, parameters=(), features=features)
        self._supports_g_tracking = False

    def solver(self) -> NoReturn:  # pragma: no cover - linear layer has no solver
        msg = "LinearLayer does not use a solver"
        raise RuntimeError(msg)

    def forward(  # type: ignore[override]
        self,
        phi: torch.Tensor,
        *,
        noise_config=None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, steps, dim = self.validate_input(phi)
        self._clear_histories()

        # Adjust dimension mismatches to mirror legacy behaviour
        if dim != self.dim:
            if dim < self.dim:
                pad = torch.zeros(batch, steps, self.dim - dim, device=phi.device, dtype=phi.dtype)
                phi_adjusted = torch.cat([phi, pad], dim=-1)
            else:
                warnings.warn(
                    f"LinearLayer: slicing input from {dim} to layer dim {self.dim}",
                    UserWarning,
                    stacklevel=2,
                )
                phi_adjusted = phi[..., : self.dim]
        else:
            phi_adjusted = phi

        self.feature_stack.set_noise_config(noise_config)
        self._on_noise_config_updated(noise_config)

        phi_noisy = self._apply_noise(phi_adjusted, noise_config, "phi")
        state = self.prepare_initial_state(phi_noisy, initial_state)
        state = self._apply_noise(state, noise_config, "s")

        history = torch.empty(batch, steps + 1, self.dim, device=phi.device, dtype=phi.dtype)
        history[:, 0, :] = state
        history[:, 1:, :] = phi_noisy

        for t in range(steps):
            self._add_phi_to_history(phi_noisy[:, t, :])
            self._add_state_to_history(history[:, t + 1, :])

        return history


def _scale_parameter_defs() -> Sequence[ParameterDef]:
    return (
        ParameterDef(
            name="scale_factor",
            default=1.0,
            initializer=InitializerSpec(method="constant", params={"value": 1.0}),
            constraint=Constraint(min=None, max=None),
        ),
    )


class ScalingLayer(PhysicalLayerBase):
    """Layer applying a learnable per-feature scaling."""

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        features: Sequence[FeatureHook] | None = None,
    ) -> None:
        super().__init__(dt=dt, dim=dim, parameters=_scale_parameter_defs(), features=features)
        self._supports_g_tracking = False

    def solver(self) -> NoReturn:  # pragma: no cover - no solver required
        msg = "ScalingLayer does not use a solver"
        raise RuntimeError(msg)

    def forward(  # type: ignore[override]
        self,
        phi: torch.Tensor,
        *,
        noise_config=None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, steps, dim = self.validate_input(phi)
        self._clear_histories()

        if dim != self.dim:
            if dim < self.dim:
                pad = torch.zeros(batch, steps, self.dim - dim, device=phi.device, dtype=phi.dtype)
                phi_adjusted = torch.cat([phi, pad], dim=-1)
            else:
                warnings.warn(
                    f"ScalingLayer: slicing input from {dim} to layer dim {self.dim}",
                    UserWarning,
                    stacklevel=2,
                )
                phi_adjusted = phi[..., : self.dim]
        else:
            phi_adjusted = phi

        self.feature_stack.set_noise_config(noise_config)
        self._on_noise_config_updated(noise_config)

        phi_noisy = self._apply_noise(phi_adjusted, noise_config, "phi")

        self.apply_parameter_constraints()
        scale = self.parameter_values()["scale_factor"].view(1, 1, -1)
        scaled = phi_noisy * scale
        scaled = self._apply_noise(scaled, noise_config, "s")

        state = self.prepare_initial_state(phi_noisy, initial_state)
        history = torch.empty(batch, steps + 1, self.dim, device=phi.device, dtype=phi.dtype)
        history[:, 0, :] = state
        history[:, 1:, :] = scaled

        for t in range(steps):
            self._add_phi_to_history(phi_noisy[:, t, :])
            self._add_state_to_history(scaled[:, t, :])

        return history


def _nonlinear_parameter_defs() -> Sequence[ParameterDef]:
    return (
        ParameterDef(
            name="phi_offset",
            default=0.0,
            initializer=InitializerSpec(method="constant", params={"value": 0.0}),
        ),
        ParameterDef(
            name="bias_current",
            default=1.7,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="constant", params={"value": 1.7}),
        ),
    )


class NonLinearLayer(PhysicalLayerBase):
    """Applies a configurable source-function nonlinearity to the input."""

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        source_func_type: str = "Tanh",
        source_func_kwargs: Mapping[str, object] | None = None,
        features: Sequence[FeatureHook] | None = None,
    ) -> None:
        super().__init__(dt=dt, dim=dim, parameters=_nonlinear_parameter_defs(), features=features)
        self._supports_g_tracking = True
        self._source_init_kwargs = dict(source_func_kwargs or {})
        self._source_function = self._build_source_function(source_func_type)

    @property
    def source_function(self):
        return self._source_function

    def solver(self) -> NoReturn:  # pragma: no cover - nonlinear layer has no solver
        msg = "NonLinearLayer does not use a solver"
        raise RuntimeError(msg)

    def forward(  # type: ignore[override]
        self,
        phi: torch.Tensor,
        *,
        noise_config=None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, steps, dim = self.validate_input(phi)
        self._clear_histories()

        if dim != self.dim:
            if dim < self.dim:
                pad = torch.zeros(batch, steps, self.dim - dim, device=phi.device, dtype=phi.dtype)
                phi_adjusted = torch.cat([phi, pad], dim=-1)
            else:
                warnings.warn(
                    f"NonLinearLayer: slicing input from {dim} to layer dim {self.dim}",
                    UserWarning,
                    stacklevel=2,
                )
                phi_adjusted = phi[..., : self.dim]
        else:
            phi_adjusted = phi

        self.feature_stack.set_noise_config(noise_config)
        self._on_noise_config_updated(noise_config)

        phi_noisy = self._apply_noise(phi_adjusted, noise_config, "phi")

        self.apply_parameter_constraints()
        params = self.parameter_values()

        phi_offset = params.get("phi_offset")
        if phi_offset is not None:
            phi_shifted = phi_noisy + phi_offset.view(1, 1, -1)
        else:
            phi_shifted = phi_noisy

        bias_current = params.get("bias_current")
        g_kwargs: dict[str, object] = {}
        if bias_current is not None:
            # bias_current parameter in virtual layers acts as the squid_current for the source function
            g_kwargs["squid_current"] = bias_current.view(1, 1, -1)

        g_val = self._source_function.g(phi_shifted, **g_kwargs)
        g_val = self._apply_noise(g_val, noise_config, "g")

        state = self.prepare_initial_state(g_val, initial_state)
        state = self._apply_noise(state, noise_config, "s")

        g_val = self._apply_noise(g_val, noise_config, "s")

        history = torch.empty(batch, steps + 1, self.dim, device=phi.device, dtype=phi.dtype)
        history[:, 0, :] = state
        history[:, 1:, :] = g_val

        for t in range(steps):
            self._add_phi_to_history(phi_noisy[:, t, :])
            self._add_g_to_history(g_val[:, t, :])
            self._add_state_to_history(history[:, t + 1, :])

        return history

    def _build_source_function(self, func_type: str):
        if func_type not in SOURCE_FUNCTIONS:
            msg = f"Unknown source function type '{func_type}'"
            raise ValueError(msg)
        source_ctor = SOURCE_FUNCTIONS[func_type]
        return source_ctor(**self._source_init_kwargs)


InputLayer = LinearLayer  # Backwards compatibility alias


def _softmax_parameter_defs() -> Sequence[ParameterDef]:
    return (
        ParameterDef(
            name="beta",
            default=1.0,
            initializer=InitializerSpec(method="constant", params={"value": 1.0}),
            constraint=Constraint(min=0.01, max=None),  # Avoid division issues with very small beta
        ),
    )


class SoftmaxLayer(PhysicalLayerBase):
    """Applies temperature-scaled softmax normalization across features.

    The softmax is computed as: softmax(beta * phi) where beta is a learnable
    temperature parameter. Higher beta values produce sharper distributions.
    """

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        features: Sequence[FeatureHook] | None = None,
    ) -> None:
        super().__init__(dt=dt, dim=dim, parameters=_softmax_parameter_defs(), features=features)
        self._supports_g_tracking = False

    def solver(self) -> NoReturn:  # pragma: no cover - no solver required
        msg = "SoftmaxLayer does not use a solver"
        raise RuntimeError(msg)

    def forward(  # type: ignore[override]
        self,
        phi: torch.Tensor,
        *,
        noise_config=None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, steps, dim = self.validate_input(phi)
        self._clear_histories()

        if dim != self.dim:
            if dim < self.dim:
                pad = torch.zeros(batch, steps, self.dim - dim, device=phi.device, dtype=phi.dtype)
                phi_adjusted = torch.cat([phi, pad], dim=-1)
            else:
                warnings.warn(
                    f"SoftmaxLayer: slicing input from {dim} to layer dim {self.dim}",
                    UserWarning,
                    stacklevel=2,
                )
                phi_adjusted = phi[..., : self.dim]
        else:
            phi_adjusted = phi

        self.feature_stack.set_noise_config(noise_config)
        self._on_noise_config_updated(noise_config)

        phi_noisy = self._apply_noise(phi_adjusted, noise_config, "phi")

        self.apply_parameter_constraints()
        beta = self.parameter_values()["beta"].view(1, 1, -1)

        # Apply temperature-scaled softmax across features (dim=-1)
        # Note: beta is per-feature, so we scale each feature's contribution
        scaled_phi = phi_noisy * beta
        softmax_out = torch.softmax(scaled_phi, dim=-1)
        softmax_out = self._apply_noise(softmax_out, noise_config, "s")

        state = self.prepare_initial_state(phi_noisy, initial_state)
        history = torch.empty(batch, steps + 1, self.dim, device=phi.device, dtype=phi.dtype)
        history[:, 0, :] = state
        history[:, 1:, :] = softmax_out

        for t in range(steps):
            self._add_phi_to_history(phi_noisy[:, t, :])
            self._add_state_to_history(softmax_out[:, t, :])

        return history


__all__ = ["InputLayer", "LinearLayer", "NonLinearLayer", "ScalingLayer", "SoftmaxLayer"]
