"""Multiplier layer for layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from soen_toolkit.core.layers.common import (
    ConnectivityModule,
    Constraint,
    FeatureHook,
    ForwardEulerSolver,
    InitializerSpec,
    ParameterDef,
    PhysicalLayerBase,
    apply_connectivity,
    resolve_connectivity_matrix,
)
from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS

from .dynamics import MultiplierDynamics

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import torch


def _parameter_defs(dim: int) -> Sequence[ParameterDef]:
    return (
        ParameterDef(
            name="phi_y",
            default=0.1,
            initializer=InitializerSpec(method="constant", params={"value": 0.1}),
        ),
        ParameterDef(
            name="bias_current",
            default=2.0,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="constant", params={"value": 2.0}),
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
            initializer=InitializerSpec(method="fan_out"),
            transform="log",
        ),
    )


class MultiplierLayer(PhysicalLayerBase):
    """Leakless multiplier layer using the layers infrastructure."""

    def __init__(
        self,
        *,
        dim: int,
        dt: float | torch.Tensor,
        solver: str = "FE",
        source_func_type: str = "RateArray",
        node_fan_outs: Sequence[int] | None = None,
        features: Iterable[FeatureHook] | None = None,
        connectivity: torch.Tensor | None = None,
        connectivity_constraints: Mapping[str, float] | None = None,
        learnable_connectivity: bool = True,
        connectivity_spec: Mapping[str, object] | None = None,
        connectivity_mode: str | None = None,
        connectivity_dynamic: Mapping[str, object] | None = None,
        connectivity_params: Mapping[str, object] | None = None,
    ) -> None:
        init_context: Mapping[str, object] = {
            "node_fan_outs": list(node_fan_outs) if node_fan_outs is not None else None,
        }
        super().__init__(
            dt=dt,
            dim=dim,
            parameters=_parameter_defs(dim),
            init_context=init_context,
            features=features,
        )
        self._source_function = self._build_source_function(source_func_type)
        self._dynamics = MultiplierDynamics(source_function=self._source_function)
        self._solver_name = solver.upper()
        self.connectivity: ConnectivityModule | None = None
        self._internal_conn_mode: str = "fixed"
        self._internal_conn_params: dict = {}
        self._internal_edge_indices: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        self._internal_conn_state = None

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

            # Parse internal connection config using unified parser
            from soen_toolkit.core.utils.connection_ops import parse_connection_config

            raw_internal_config = {
                "mode": connectivity_mode,
                "connection_params": connectivity_params or connectivity_dynamic,
            }
            self._internal_conn_mode, self._internal_conn_params = parse_connection_config(raw_internal_config)

            # Precompute edge indices for dynamic internal modes
            if self._internal_conn_mode in ("WICC", "NOCC"):
                try:
                    from soen_toolkit.core.utils.connection_ops import build_edge_index

                    mask = matrix.abs() > 0
                    src_idx, dst_idx = build_edge_index(mask, matrix)
                except Exception:
                    dummy = torch.zeros(dim, dim, dtype=torch.float32)
                    src_idx, dst_idx = build_edge_index(None, dummy)
                self._internal_edge_indices = (src_idx, dst_idx)

        self._solver = self._build_solver(self._solver_name)

    def solver(self) -> ForwardEulerSolver:
        return self._solver

    def forward(  # type: ignore[override]
        self,
        phi: torch.Tensor,
        *,
        noise_config=None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Reset internal connection state at start of forward pass for determinism
        if self._internal_conn_state is not None:
            self._internal_conn_state.edge_state = None

        self.validate_input(phi)
        initial_state_tensor = self.prepare_initial_state(phi, initial_state)

        self.apply_parameter_constraints()
        if self.connectivity is not None:
            self.connectivity.apply_constraints()
        params = self.parameter_values()
        if self.connectivity is not None:
            internal_weight = self.connectivity.materialised()
            params["internal_J"] = self.apply_qat_ste_to_weight(internal_weight)

        self.feature_stack.set_noise_config(noise_config)
        self._on_noise_config_updated(noise_config)

        return self._solver.integrate(
            state=_StateWrapper(initial_state_tensor),
            phi=phi,
            params=params,
            dt=self.dt,
        )

    def _build_solver(self, solver_name: str) -> ForwardEulerSolver:
        if solver_name != "FE":
            msg = f"Multiplier only supports Forward Euler solver, got '{solver_name}'"
            raise ValueError(msg)

        phi_transform = None
        if self.connectivity is not None:
            phi_transform = self._apply_connectivity
        return ForwardEulerSolver(
            dynamics=self._dynamics,
            feature=self.feature_stack,
            phi_transform=phi_transform,
            layer=self,
        )

    def _build_source_function(self, func_type: str):
        if func_type not in SOURCE_FUNCTIONS:
            msg = f"Unknown source function type '{func_type}'"
            raise ValueError(msg)
        return SOURCE_FUNCTIONS[func_type]()

    def _apply_connectivity(
        self,
        *,
        state: torch.Tensor,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply internal connectivity (static or dynamic)."""
        internal_weight = params.get("internal_J")
        if internal_weight is None:
            return phi

        # If mode is dynamic, use dynamic application
        if self._internal_conn_mode in ("WICC", "NOCC"):
            from soen_toolkit.core.utils.connection_ops import ConnectionState, apply_connection_step

            if self._internal_conn_state is None:
                self._internal_conn_state = ConnectionState()
            phi_add = apply_connection_step(state, internal_weight, self._internal_conn_mode, self._internal_conn_params, self._internal_edge_indices, self._internal_conn_state, self.dt)
            return phi + phi_add
        else:
            # Fixed mode: static matrix multiply
            return apply_connectivity(state=state, phi=phi, params=params, key="internal_J")


class _StateWrapper:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values


__all__ = ["MultiplierLayer"]
