"""Common base classes for layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import torch
from torch import nn

from soen_toolkit.core.noise import NoiseSettings, build_noise_strategies
from soen_toolkit.utils.quantization import ste_snap

from .features import CompositeFeature, FeatureHook, NoiseFeature, PowerTrackingFeature
from .parameters import ParameterDef, ParameterRegistry
from .validators import broadcast_initial_state, validate_sequence_input

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class SoenLayerBase(nn.Module):
    """Lightweight baseline for future layers."""

    # Type hints for registered buffers (mypy doesn't track register_buffer)
    _dt: torch.Tensor

    def __init__(
        self,
        *,
        dt: float | torch.Tensor,
        dim: int,
        parameters: Sequence[ParameterDef] = (),
        dtype: torch.dtype = torch.float32,
        init_context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        dt_tensor = self._coerce_dt(dt, dtype=dtype)
        self.register_buffer("_dt", dt_tensor, persistent=True)
        self._param_registry = ParameterRegistry(
            self,
            width=self.dim,
            dtype=dtype,
            device=dt_tensor.device,
            init_context=init_context,
        )
        for definition in parameters:
            self._param_registry.add(definition)

    # ------------------------------------------------------------------
    # dt handling
    # ------------------------------------------------------------------
    @property
    def dt(self) -> torch.Tensor:
        return self._dt

    @dt.setter
    def dt(self, value: float | torch.Tensor) -> None:
        new_dt = self._coerce_dt(value, dtype=self._dt.dtype)
        self._dt.data.copy_(new_dt)

    # ------------------------------------------------------------------
    # parameter helpers
    # ------------------------------------------------------------------
    def parameter_values(self) -> dict[str, torch.Tensor]:
        return self._param_registry.named_tensors()

    def apply_parameter_constraints(self) -> None:
        self._param_registry.apply_constraints()

    def set_parameter(
        self,
        name: str,
        value: float | torch.Tensor,
        *,
        learnable: bool | None = None,
    ) -> None:
        """Set a parameter value by name, handling log-transforms automatically.

        This is a convenience method that wraps override_parameter for simple
        value assignments. For advanced initialization (e.g., using specific
        initializers), use the registry's override_parameter directly.

        Args:
            name: Parameter name (e.g., "gamma_minus", not "log_gamma_minus")
            value: New value in real-space (automatically converted if log-transformed)
            learnable: Optionally change whether the parameter is learnable
        """
        self._param_registry.override_parameter(name, value=value, learnable=learnable)

    def apply_qat_ste_to_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply straight-through quantisation to ``weight`` when QAT is active."""
        if not getattr(self, "_qat_ste_active", False):
            return weight
        if not getattr(self, "_qat_internal_active", False):
            return weight
        codebook = getattr(self, "_qat_codebook", None)
        if codebook is None:
            return weight
        stochastic = bool(getattr(self, "_qat_stochastic_rounding", False))
        return ste_snap(weight, codebook, stochastic=stochastic)

    def _apply_noise(self, tensor: torch.Tensor, noise_config, noise_key: str) -> torch.Tensor:
        """Compatibility shim for legacy noise helpers."""
        if noise_config is None:
            return tensor
        if not isinstance(noise_config, NoiseSettings):
            noise_config = build_noise_strategies(noise_config)
        return noise_config.apply(tensor, noise_key)

    def _reset_noise_cache(self) -> None:
        self._noise_offsets: dict[str, int] = {}

    # ------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs) -> NoReturn:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _coerce_dt(self, value: float | torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        return torch.tensor(float(value), dtype=dtype)


class PhysicalLayerBase(SoenLayerBase):
    """Base class for physical SOEN layers with feature support."""

    def __init__(
        self,
        *,
        dt: float | torch.Tensor,
        dim: int,
        parameters: Sequence[ParameterDef] = (),
        dtype: torch.dtype = torch.float32,
        init_context: Mapping[str, object] | None = None,
        features: Iterable[FeatureHook] | None = None,
    ) -> None:
        super().__init__(
            dt=dt,
            dim=dim,
            parameters=parameters,
            dtype=dtype,
            init_context=init_context,
        )
        default_layer_type = self.__class__.__name__
        default_layer_type = default_layer_type.removesuffix("Layer")
        self.layer_type = getattr(self, "layer_type", default_layer_type)
        incoming_features = list(features or [])
        feature_stack = CompositeFeature(incoming_features)
        feature_stack.attach_layer(self)
        if not feature_stack.has_feature(NoiseFeature):
            feature_stack.prepend(NoiseFeature(layer_type=self.layer_type))
        self._feature_stack = feature_stack
        self.track_power = False
        self.track_phi = False
        self.track_g = False
        self.track_s = False
        self._phi_history: list[torch.Tensor] = []
        self._g_history: list[torch.Tensor] = []
        self._state_history: list[torch.Tensor] = []
        self._current_noise_config: Mapping[str, object] | NoiseSettings | None = None
        # default physical constants for features that expect them
        from soen_toolkit.physics.constants import DEFAULT_IC, DEFAULT_PHI0, get_omega_c

        self.Phi0 = DEFAULT_PHI0
        self.Ic = DEFAULT_IC
        self.wc = float(get_omega_c())

    # Tracking helpers -------------------------------------------------
    def set_tracking_flags(
        self,
        *,
        phi: bool | None = None,
        g: bool | None = None,
        s: bool | None = None,
        power: bool | None = None,
    ) -> None:
        if phi is not None:
            self.track_phi = bool(phi)
        if g is not None:
            self.track_g = bool(g)
        if s is not None:
            self.track_s = bool(s)
        if power is not None:
            # Only allow power tracking on SingleDendrite layers
            layer_type = getattr(self, "layer_type", self.__class__.__name__)
            if power and layer_type != "SingleDendrite":
                import warnings

                warnings.warn(
                    f"Power tracking is only supported for SingleDendrite layers, not {layer_type}. Power tracking will be disabled for this layer.",
                    UserWarning,
                    stacklevel=2,
                )
                self.track_power = False
            else:
                self.track_power = bool(power)
                if power and not self.feature_stack.has_feature(PowerTrackingFeature):
                    self.feature_stack.add(
                        PowerTrackingFeature(
                            Ic=float(getattr(self, "Ic", self.Ic)),
                            Phi0=float(getattr(self, "Phi0", self.Phi0)),
                            wc=float(getattr(self, "wc", self.wc)),
                        ),
                    )

    def _clear_histories(self) -> None:
        self._phi_history = []
        self._g_history = []
        self._state_history = []

    # Backwards-compatibility: legacy code called _clear_phi_history/_add_phi_to_history
    def _clear_phi_history(self) -> None:  # pragma: no cover - legacy API
        self._phi_history = []

    def _clear_g_history(self) -> None:  # pragma: no cover - legacy API
        self._g_history = []

    def _clear_state_history(self) -> None:  # pragma: no cover - legacy API
        self._state_history = []

    def _add_phi_to_history(self, phi: torch.Tensor) -> None:
        if self.track_phi:
            self._phi_history.append(phi.detach().clone())

    def _add_g_to_history(self, g_val: torch.Tensor) -> None:
        if self.track_g:
            self._g_history.append(g_val.detach().clone())

    def _add_state_to_history(self, state: torch.Tensor) -> None:
        if self.track_s:
            self._state_history.append(state.detach().clone())

    def _set_phi_history_sequence(self, phi: torch.Tensor) -> None:
        if self.track_phi:
            self._phi_history = [phi.detach().clone()]

    def _set_g_history_sequence(self, g_val: torch.Tensor) -> None:
        if self.track_g:
            self._g_history = [g_val.detach().clone()]

    def _set_state_history_sequence(self, state: torch.Tensor) -> None:
        if self.track_s:
            self._state_history = [state.detach().clone()]

    def get_phi_history(self) -> torch.Tensor | None:
        if self.track_phi and self._phi_history:
            stored = self._phi_history[0]
            if len(self._phi_history) == 1 and stored.dim() == 3:
                return stored
            return torch.stack(self._phi_history, dim=1)
        return None

    def get_g_history(self) -> torch.Tensor | None:
        if self.track_g and self._g_history:
            stored = self._g_history[0]
            if len(self._g_history) == 1 and stored.dim() == 3:
                return stored
            return torch.stack(self._g_history, dim=1)
        return None

    def get_state_history(self) -> torch.Tensor | None:
        if self.track_s and self._state_history:
            stored = self._state_history[0]
            if len(self._state_history) == 1 and stored.dim() == 3:
                return stored
            return torch.stack(self._state_history, dim=1)
        return None

    def validate_input(self, tensor: torch.Tensor) -> tuple[int, int, int]:
        return validate_sequence_input(tensor, dim=self.dim)

    def prepare_initial_state(
        self,
        tensor: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _, dim = tensor.shape
        return broadcast_initial_state(
            initial_state,
            batch=batch,
            dim=dim,
            device=tensor.device,
            dtype=tensor.dtype,
        )

    def add_feature(self, feature: FeatureHook) -> None:
        self._feature_stack.add(feature)

    @property
    def feature_stack(self) -> CompositeFeature:
        return self._feature_stack

    def _on_noise_config_updated(self, noise_config) -> None:
        self._current_noise_config = noise_config

    def _convert_power_and_energy(self) -> None:
        """Convert dimensionless power/energy to physical units."""
        if not getattr(self, "track_power", False):
            return

        from soen_toolkit.utils.power_tracking import (
            convert_energy_to_physical,
            convert_power_to_physical,
        )

        # Only convert if dimensionless values exist
        if hasattr(self, "power_bias_dimensionless") and self.power_bias_dimensionless is not None:
            self.power_bias = convert_power_to_physical(
                self.power_bias_dimensionless,
                self.Ic,
                self.Phi0,
                self.wc,
            )
        if hasattr(self, "power_diss_dimensionless") and self.power_diss_dimensionless is not None:
            self.power_diss = convert_power_to_physical(
                self.power_diss_dimensionless,
                self.Ic,
                self.Phi0,
                self.wc,
            )
        if hasattr(self, "energy_bias_dimensionless") and self.energy_bias_dimensionless is not None:
            self.energy_bias = convert_energy_to_physical(
                self.energy_bias_dimensionless,
                self.Ic,
                self.Phi0,
            )
        if hasattr(self, "energy_diss_dimensionless") and self.energy_diss_dimensionless is not None:
            self.energy_diss = convert_energy_to_physical(
                self.energy_diss_dimensionless,
                self.Ic,
                self.Phi0,
            )

    def _set_zero_power_tracking(self, batch: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize zero power tracking tensors."""
        if not getattr(self, "track_power", False):
            return

        shape = (batch, seq_len, self.dim)
        self.power_bias_dimensionless = torch.zeros(shape, device=device, dtype=dtype)
        self.power_diss_dimensionless = torch.zeros_like(self.power_bias_dimensionless)
        self.energy_bias_dimensionless = torch.zeros_like(self.power_bias_dimensionless)
        self.energy_diss_dimensionless = torch.zeros_like(self.power_bias_dimensionless)

        # Convert to physical units
        self._convert_power_and_energy()


class VirtualLayerBase(SoenLayerBase):
    """Placeholder for conventional NN layers."""


__all__ = ["PhysicalLayerBase", "SoenLayerBase", "VirtualLayerBase"]
