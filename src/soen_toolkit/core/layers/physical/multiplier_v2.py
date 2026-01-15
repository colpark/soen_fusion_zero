"""Multiplier V2 layer for new flux collection mechanism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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

from .dynamics import MultiplierNOCCDynamics, MultiplierNOCCState

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import torch


def _parameter_defs(dim: int) -> Sequence[ParameterDef]:
    """Define parameters for multiplier v2 with recommended defaults."""
    return (
        ParameterDef(
            name="phi_y",
            default=0.1,
            initializer=InitializerSpec(method="constant", params={"value": 0.1}),
        ),
        ParameterDef(
            name="bias_current",
            default=2.1,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="constant", params={"value": 2.1}),
        ),
        ParameterDef(
            name="alpha",
            default=1.64053,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="constant", params={"value": 1.64053}),
        ),
        ParameterDef(
            name="beta",
            default=303.85,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="constant", params={"value": 303.85}),
        ),
        ParameterDef(
            name="beta_out",
            default=91.156,
            constraint=Constraint(min=0.0),
            initializer=InitializerSpec(method="constant", params={"value": 91.156}),
        ),
    )


class MultiplierNOCCLayer(PhysicalLayerBase):
    """Multiplier v2 layer with dual SQUID states and aggregated output.

    This layer uses a multiplier circuit design without collection coils.
    For external flux input, each node has an implicit incoming edge with
    states s1 and s2 [B, D]. The aggregated output state m is per node [B, D].

    Parameters:
        - alpha: Dimensionless resistance (default: 1.64053)
        - beta: Inductance of incoming branches (default: 303.85)
        - beta_out: Inductance of output branch (default: 91.156)
        - ib: Bias current (default: 2.1)
        - phi_y: Secondary input/weight term (default: 0.1)

    Physical parameter mappings:
        - beta_1 ≈ 1nH → beta = 303.85
        - beta_out ≈ 300pH → beta_out = 91.156
        - i_b ≈ 210μA → ib = 2.1
        - R ≈ 2Ω → alpha = 1.64053
    """

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
        """Initialize multiplier v2 layer.

        Args:
            dim: Number of nodes/circuits in the layer
            dt: Time step for integration
            solver: Solver type (only "FE" supported for v2)
            source_func_type: Source function key (e.g., "RateArray")
            node_fan_outs: Fan-out information for initialization
            features: Optional feature hooks
            connectivity: Optional internal connectivity matrix
            connectivity_constraints: Constraints on internal connectivity
            learnable_connectivity: Whether internal connectivity is learnable
            connectivity_spec: Specification for connectivity structure
        """
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
        self._dynamics = MultiplierNOCCDynamics(source_function=self._source_function)
        self._solver_name = solver.upper()

        if self._solver_name != "FE":
            msg = f"MultiplierNOCC only supports Forward Euler solver, got '{solver}'"
            raise ValueError(msg)

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
        s1_init: torch.Tensor | None = None,
        s2_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the multiplier v2 layer.

        Args:
            phi: Input flux [B, T, D] or [B, D]
            noise_config: Optional noise configuration
            initial_state: Optional initial m state [B, D] (optional, defaults to zero)
            s1_init: Optional initial s1 state [B, D] (for stepwise solvers)
            s2_init: Optional initial s2 state [B, D] (for stepwise solvers)

        Returns:
            Output flux [B, T, D] or [B, D] (m state)
        """
        # Reset internal connection state at start of forward pass for determinism
        if self._internal_conn_state is not None:
            self._internal_conn_state.edge_state = None

        self.validate_input(phi)

        if phi.dim() != 3:
            msg = f"Expected phi with shape [batch, steps, dim], received {tuple(phi.shape)}"
            raise ValueError(msg)

        batch, steps, dim = phi.shape
        device = phi.device
        dtype = phi.dtype

        # Initialize v2 states
        if s1_init is not None:
            s1 = s1_init.to(device=device, dtype=dtype)
        else:
            s1 = torch.zeros(batch, dim, device=device, dtype=dtype)
        if s2_init is not None:
            s2 = s2_init.to(device=device, dtype=dtype)
        else:
            s2 = torch.zeros(batch, dim, device=device, dtype=dtype)

        if initial_state is not None:
            m = initial_state.to(device=device, dtype=dtype)
        else:
            m = torch.zeros(batch, dim, device=device, dtype=dtype)

        self.apply_parameter_constraints()
        if self.connectivity is not None:
            self.connectivity.apply_constraints()
        params = self.parameter_values()

        # Compute fan-in for each node
        if self.connectivity is not None:
            internal_weight = self.connectivity.materialised()
            params["internal_J"] = self.apply_qat_ste_to_weight(internal_weight)
            # Fan-in is number of incoming connections per node
            fan_in = (internal_weight.abs() > 0).sum(dim=1).float()  # [D]
        else:
            fan_in = torch.ones(self.dim, device=device, dtype=dtype)
        params["fan_in"] = fan_in

        # Expand parameters
        expanded_params = {}
        for name, tensor in params.items():
            tensor = tensor.to(device=device, dtype=dtype)
            if tensor.dim() == 0:
                expanded_params[name] = tensor.view(1, 1).expand(batch, dim)
            elif tensor.dim() == 1:
                expanded_params[name] = tensor.view(1, -1).expand(batch, -1)
            elif tensor.dim() == 2:
                expanded_params[name] = tensor
            else:
                expanded_params[name] = tensor

        self.feature_stack.set_noise_config(noise_config)
        self._on_noise_config_updated(noise_config)

        # Integrate over time
        dt_val = self.dt.to(device=device, dtype=dtype)
        history = torch.empty(batch, steps + 1, dim, device=device, dtype=dtype)
        history[:, 0, :] = m  # Initial m state

        for t in range(steps):
            phi_t = phi[:, t, :]  # [B, D]

            # Apply internal connectivity if present (static or dynamic)
            if self.connectivity is not None:
                internal_weight = expanded_params["internal_J"]
                if self._internal_conn_mode in ("WICC", "NOCC"):
                    # Dynamic mode: use connection_ops
                    from soen_toolkit.core.utils.connection_ops import ConnectionState, apply_connection_step

                    if self._internal_conn_state is None:
                        self._internal_conn_state = ConnectionState()
                    phi_add = apply_connection_step(m, internal_weight, self._internal_conn_mode, self._internal_conn_params, self._internal_edge_indices, self._internal_conn_state, dt_val)
                    phi_t = phi_t + phi_add
                else:
                    # Fixed mode: static matrix multiply
                    phi_t = phi_t + (m @ internal_weight.t())

            # Create state wrapper for dynamics
            state = MultiplierNOCCState(s1=s1, s2=s2, m=m)

            # Compute derivatives
            d_state = self._dynamics(state, phi_t, expanded_params)

            # Update states with Forward Euler
            s1 = s1 + dt_val * d_state.s1
            s2 = s2 + dt_val * d_state.s2
            m = m + dt_val * d_state.m

            history[:, t + 1, :] = m

        # Store final s1, s2 for retrieval in stepwise solvers
        self._s1_final = s1
        self._s2_final = s2

        # Output is the m state
        return history

    def _build_solver(self, solver_name: str) -> ForwardEulerSolver:
        """Build Forward Euler solver for v2 dynamics."""
        if solver_name != "FE":
            msg = f"Solver '{solver_name}' not supported for MultiplierNOCC"
            raise NotImplementedError(msg)

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
        """Build source function from registry."""
        if func_type not in SOURCE_FUNCTIONS:
            msg = f"Unknown source function type '{func_type}'"
            raise ValueError(msg)
        return SOURCE_FUNCTIONS[func_type]()

    def _apply_connectivity(
        self,
        *,
        state: MultiplierNOCCState,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply internal connectivity to phi input (static or dynamic)."""
        internal_weight = params.get("internal_J")
        if internal_weight is None:
            return phi

        # If mode is dynamic, use dynamic application
        if self._internal_conn_mode in ("WICC", "NOCC"):
            from soen_toolkit.core.utils.connection_ops import ConnectionState, apply_connection_step

            if self._internal_conn_state is None:
                self._internal_conn_state = ConnectionState()
            phi_add = apply_connection_step(state.m, internal_weight, self._internal_conn_mode, self._internal_conn_params, self._internal_edge_indices, self._internal_conn_state, self.dt)
            return phi + phi_add
        else:
            # Fixed mode: static matrix multiply
            # For v2, we apply connectivity using the m state
            return apply_connectivity(state=state.m, phi=phi, params=params, key="internal_J")


__all__ = ["MultiplierNOCCLayer"]
