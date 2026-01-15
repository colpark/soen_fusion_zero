"""Torch recurrent wrappers for layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from soen_toolkit.core.layers.common import (
    ConnectivityModule,
    ParameterDef,
    PhysicalLayerBase,
    apply_connectivity,
    resolve_connectivity_matrix,
)
from soen_toolkit.core.noise import NoiseSettings, build_noise_strategies

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class _NoiseHelper:
    """Lightweight helper so recurrent layers treat noise uniformly."""

    def __init__(self, config: Mapping[str, object] | NoiseSettings | None) -> None:
        if config is None:
            self._strategy: NoiseSettings | None = None
        elif isinstance(config, NoiseSettings):
            self._strategy = config
        else:
            from soen_toolkit.core.configs import NoiseConfig, PerturbationConfig
            from typing import cast
            # Safe best-effort cast assuming proper config structure if dict
            # or try to detect types. build_noise_strategies accepts dict | NoiseConfig
            self._strategy = build_noise_strategies(cast("NoiseConfig | dict[str, object] | None", config))

    def reset(self) -> None:
        if self._strategy is not None:
            self._strategy.reset()

    def apply(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        if self._strategy is None:
            return tensor
        return self._strategy.apply(tensor, key)


class SequentialLayerBase(PhysicalLayerBase):
    """Shared sequential loop scaffold for virtual recurrent layers."""

    def __init__(
        self,
        *,
        dt: float | torch.Tensor,
        dim: int,
        parameters: Sequence[ParameterDef] = (),
        connectivity: torch.Tensor | None = None,
        connectivity_spec: Mapping[str, object] | None = None,
        connectivity_constraints: Mapping[str, float] | None = None,
        learnable_connectivity: bool = True,
    ) -> None:
        super().__init__(dt=dt, dim=dim, parameters=parameters)
        self.connectivity: ConnectivityModule | None = None
        matrix = resolve_connectivity_matrix(
            dim=dim,
            connectivity=connectivity,
            spec=connectivity_spec,
        )
        if matrix is not None:
            self.connectivity = ConnectivityModule(
                dim=dim,
                init=matrix,
                learnable=learnable_connectivity,
                constraints=connectivity_constraints,
            )
            self.add_module("internal_connectivity", self.connectivity)

    # ------------------------------------------------------------------
    # Hooks subclasses must implement
    # ------------------------------------------------------------------
    def _init_hidden(self, initial_state: torch.Tensor) -> object:
        raise NotImplementedError

    def _step(
        self,
        phi_t: torch.Tensor,
        hidden: object,
    ) -> tuple[torch.Tensor, object]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Forward loop
    # ------------------------------------------------------------------
    def forward(  # type: ignore[override]
        self,
        phi: torch.Tensor,
        *,
        noise_config: Mapping[str, object] | NoiseSettings | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, steps, dim = self.validate_input(phi)
        init_state = self.prepare_initial_state(phi, initial_state)

        self._clear_histories()
        if self.connectivity is not None:
            self.connectivity.apply_constraints()
            internal_weight = self.connectivity.materialised()
            connectivity_params = {
                "internal_J": self.apply_qat_ste_to_weight(internal_weight),
            }
        else:
            connectivity_params = None

        noise = _NoiseHelper(noise_config)
        noise.reset()

        history = torch.empty(batch, steps + 1, dim, device=phi.device, dtype=phi.dtype)
        history[:, 0, :] = init_state

        hidden = self._init_hidden(init_state)
        current_state = init_state

        self.feature_stack.set_noise_config(noise_config)
        self._on_noise_config_updated(noise_config)

        for t in range(steps):
            phi_t = phi[:, t, :]
            if connectivity_params is not None:
                phi_t = apply_connectivity(state=current_state, phi=phi_t, params=connectivity_params)
            phi_t = noise.apply(phi_t, "phi")
            self._add_phi_to_history(phi_t)

            next_state, hidden = self._step(phi_t, hidden)
            next_state = noise.apply(next_state, "s")

            self._add_state_to_history(next_state)
            history[:, t + 1, :] = next_state
            current_state = next_state

        return history

    @property
    def internal_J(self) -> torch.Tensor:
        if self.connectivity is None:
            msg = "Layer has no internal connectivity"
            raise AttributeError(msg)
        return self.connectivity.weight

    @internal_J.setter
    def internal_J(self, value: torch.Tensor) -> None:
        if self.connectivity is None:
            msg = "Layer has no internal connectivity"
            raise AttributeError(msg)
        with torch.no_grad():
            self.connectivity.weight.copy_(value)

    @property
    def internal_J_constraints(self) -> Mapping[str, float] | None:
        if self.connectivity is None:
            return None
        return self.connectivity.constraints

    @internal_J_constraints.setter
    def internal_J_constraints(self, value: Mapping[str, float] | None) -> None:
        if self.connectivity is None:
            msg = "Layer has no internal connectivity"
            raise AttributeError(msg)
        self.connectivity.constraints = dict(value or {})


class TorchRecurrentLayerBase(SequentialLayerBase):
    """Sequential wrapper around ``nn.RNNBase`` modules."""

    def __init__(
        self,
        *,
        dt: float | torch.Tensor,
        dim: int,
        module: nn.RNNBase,
        connectivity: torch.Tensor | None = None,
        connectivity_spec: Mapping[str, object] | None = None,
        connectivity_constraints: Mapping[str, float] | None = None,
        learnable_connectivity: bool = True,
    ) -> None:
        super().__init__(
            dt=dt,
            dim=dim,
            connectivity=connectivity,
            connectivity_spec=connectivity_spec,
            connectivity_constraints=connectivity_constraints,
            learnable_connectivity=learnable_connectivity,
        )
        self.module = module
        self.add_module("core", self.module)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _init_hidden(self, initial_state: torch.Tensor) -> torch.Tensor:
        return initial_state.unsqueeze(0)

    def _step(  # type: ignore[override]
        self,
        phi_t: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, next_hidden = self.module(phi_t.unsqueeze(1), hidden)
        next_state = output.squeeze(1)
        return next_state, next_hidden


class RNNLayer(TorchRecurrentLayerBase):
    """Single-layer ``nn.RNN`` wrapper."""

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        solver: str | None = None,
        nonlinearity: str = "tanh",
        connectivity: torch.Tensor | None = None,
        connectivity_spec: Mapping[str, object] | None = None,
        connectivity_constraints: Mapping[str, float] | None = None,
        learnable_connectivity: bool = True,
    ) -> None:
        module = nn.RNN(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            nonlinearity=nonlinearity,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        super().__init__(
            dt=dt,
            dim=dim,
            module=module,
            connectivity=connectivity,
            connectivity_spec=connectivity_spec,
            connectivity_constraints=connectivity_constraints,
            learnable_connectivity=learnable_connectivity,
        )


class GRULayer(TorchRecurrentLayerBase):
    """Single-layer ``nn.GRU`` wrapper."""

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        solver: str | None = None,
        connectivity: torch.Tensor | None = None,
        connectivity_spec: Mapping[str, object] | None = None,
        connectivity_constraints: Mapping[str, float] | None = None,
        learnable_connectivity: bool = True,
    ) -> None:
        module = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        super().__init__(
            dt=dt,
            dim=dim,
            module=module,
            connectivity=connectivity,
            connectivity_spec=connectivity_spec,
            connectivity_constraints=connectivity_constraints,
            learnable_connectivity=learnable_connectivity,
        )


class LSTMLayer(SequentialLayerBase):
    """Single-layer ``nn.LSTM`` wrapper."""

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        solver: str | None = None,
        connectivity: torch.Tensor | None = None,
        connectivity_spec: Mapping[str, object] | None = None,
        connectivity_constraints: Mapping[str, float] | None = None,
        learnable_connectivity: bool = True,
    ) -> None:
        super().__init__(
            dt=dt,
            dim=dim,
            connectivity=connectivity,
            connectivity_spec=connectivity_spec,
            connectivity_constraints=connectivity_constraints,
            learnable_connectivity=learnable_connectivity,
        )
        self.module = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.add_module("core", self.module)

    def _init_hidden(self, initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = initial_state.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def _step(  # type: ignore[override]
        self,
        phi_t: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        output, (h_next, c_next) = self.module(phi_t.unsqueeze(1), hidden)
        next_state = output.squeeze(1)
        return next_state, (h_next, c_next)


class MinGRULayer(SequentialLayerBase):
    """Minimal GRU-style recurrent update with optional internal connectivity."""

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        solver: str | None = None,
        connectivity: torch.Tensor | None = None,
        connectivity_spec: Mapping[str, object] | None = None,
        connectivity_constraints: Mapping[str, float] | None = None,
        learnable_connectivity: bool = True,
    ) -> None:
        super().__init__(
            dt=dt,
            dim=dim,
            connectivity=connectivity,
            connectivity_spec=connectivity_spec,
            connectivity_constraints=connectivity_constraints,
            learnable_connectivity=learnable_connectivity,
        )
        self.hidden_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.add_module("hidden_proj", self.hidden_proj)
        self.add_module("gate_proj", self.gate_proj)

    def _init_hidden(self, initial_state: torch.Tensor) -> torch.Tensor:
        return initial_state

    def _step(self, phi_t: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        gate = self.gate_proj(phi_t)
        z = torch.sigmoid(gate)
        hidden_proj = self.hidden_proj(phi_t)
        g_val = torch.tanh(hidden_proj)
        next_state = (1 - z) * hidden + z * g_val
        return next_state, next_state


__all__ = [
    "GRULayer",
    "LSTMLayer",
    "MinGRULayer",
    "RNNLayer",
]
