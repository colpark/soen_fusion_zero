"""Single-dendrite layer for layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from soen_toolkit.core.layers.common import (
    ConnectivityModule,
    Constraint,
    FeatureHook,
    ForwardEulerSolver,
    InitializerSpec,
    ParallelScanSolver,
    ParameterDef,
    ParaRNNSolver,
    PhysicalLayerBase,
    apply_connectivity,
    resolve_connectivity_matrix,
)
from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS
from soen_toolkit.core.utils.connection_ops import build_edge_index

from .dynamics import SingleDendriteCoefficientProvider, SingleDendriteDynamics

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


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
    )


class SingleDendriteLayer(PhysicalLayerBase):
    """Single-dendrite layer composed from parameters, dynamics, and solver."""

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
        self._dynamics = SingleDendriteDynamics(source_function=self._source_function)
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
                    mask = matrix.abs() > 0
                    src_idx, dst_idx = build_edge_index(mask, matrix)
                except Exception:
                    dummy = torch.zeros(dim, dim, dtype=torch.float32)
                    src_idx, dst_idx = build_edge_index(None, dummy)
                self._internal_edge_indices = (src_idx, dst_idx)
        self._solver = self._build_solver(self._solver_name)

    # ------------------------------------------------------------------
    # PyTorch API
    # ------------------------------------------------------------------
    def solver(self) -> ForwardEulerSolver | ParaRNNSolver | ParallelScanSolver:
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
        # Legacy helpers expect raw configs during forward passes
        self._on_noise_config_updated(noise_config)

        return self._solver.integrate(
            state=_StateWrapper(initial_state_tensor),
            phi=phi,
            params=params,
            dt=self.dt,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_solver(self, solver_name: str) -> ForwardEulerSolver | ParaRNNSolver | ParallelScanSolver:
        if solver_name == "FE":
            phi_transform = None
            if self.connectivity is not None:
                if self._internal_conn_mode in ("WICC", "NOCC"):
                    phi_transform = self._apply_connectivity_dynamic
                else:
                    phi_transform = self._apply_connectivity
            return ForwardEulerSolver(
                dynamics=self._dynamics,
                feature=self.feature_stack,
                phi_transform=phi_transform,
                layer=self,
            )
        if solver_name == "PS":
            if self.connectivity is not None:
                msg = "Parallel scan solver does not support internal connectivity."
                raise RuntimeError(msg)

            # Check if source function can provide coefficients
            if not getattr(self._source_function.info, "supports_coefficients", True):
                msg = f"Parallel scan solver requires source function with coefficient support. '{self._source_function.info.key}' does not support coefficients."
                raise RuntimeError(msg)

            coeff_provider = SingleDendriteCoefficientProvider(self._source_function)
            return ParallelScanSolver(
                coeff_provider=coeff_provider,
                feature=self.feature_stack,
                layer=self,
            )
        if solver_name == "PARARNN":
            # ParaRNN solver uses Newton's method with parallel prefix scan.
            #
            # The key requirement is that the Jacobian J = ∂f/∂s_{t-1} must be
            # DIAGONAL for efficient parallel computation. Mathematically, ParaRNN
            # works with any Jacobian structure, but:
            #   - Dense Jacobian: O(d³) matrix multiply per step → too slow
            #   - Diagonal Jacobian: O(d) element-wise ops → very fast
            #
            # For SingleDendrite with recurrent weights J:
            #   phi_conn = phi + s @ J.T
            #   s_next = α·s + β·g(phi_conn)
            #   ∂s_next/∂s = α·I + β·diag(g') @ J.T
            #
            # This is diagonal IFF J is diagonal (element-wise recurrent weights).
            # Dense J creates off-diagonal terms that accumulate numerical errors
            # over long sequences. Feature mixing across neurons should be done
            # via inter-layer connections, similar to Mamba/Transformer designs.
            connectivity_fn = None
            if self.connectivity is not None:
                conn_weight = self.connectivity.materialised()
                # Check if connectivity is diagonal (element-wise recurrent weights)
                if conn_weight.dim() == 2:
                    off_diag = conn_weight - torch.diag(torch.diag(conn_weight))
                    if off_diag.abs().max() > 1e-8:
                        msg = (
                            "ParaRNN solver requires diagonal (element-wise) recurrent weights. "
                            "Dense weight matrices create O(d³) Jacobian operations that are "
                            "computationally prohibitive. Use Forward Euler for dense connectivity, "
                            "or restructure to use diagonal recurrent weights with inter-layer "
                            "connections for feature mixing."
                        )
                        raise RuntimeError(msg)
                connectivity_fn = self._apply_connectivity

            step_provider = _ParaRNNStepProvider(
                dynamics=self._dynamics,
                connectivity_fn=connectivity_fn,
            )
            return ParaRNNSolver(
                step_provider=step_provider,
                feature=self.feature_stack,
                layer=self,
            )
        msg = f"Solver '{solver_name}' not yet available in v2 prototype"
        raise NotImplementedError(msg)

    def _build_source_function(self, func_type: str):
        if func_type not in SOURCE_FUNCTIONS:
            msg = f"Unknown source function type '{func_type}'"
            raise ValueError(msg)
        return SOURCE_FUNCTIONS[func_type]()

    @staticmethod
    def _apply_connectivity(
        *,
        state: torch.Tensor,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return apply_connectivity(state=state, phi=phi, params=params, key="internal_J")

    def _apply_connectivity_dynamic(
        self,
        *,
        state: torch.Tensor,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply dynamic internal connectivity using unified helper."""
        internal_weight = params.get("internal_J")
        if internal_weight is None:
            return phi
        # Use unified connection helper
        from soen_toolkit.core.utils.connection_ops import ConnectionState, apply_connection_step

        if self._internal_conn_state is None:
            self._internal_conn_state = ConnectionState()
        phi_add = apply_connection_step(state, internal_weight, self._internal_conn_mode, self._internal_conn_params, self._internal_edge_indices, self._internal_conn_state, self.dt)
        return phi + phi_add


class _StateWrapper:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values


class _ParaRNNStepProvider:
    """Step provider adapter for ParaRNN solver.

    Wraps the dynamics step function and optionally applies internal connectivity.
    """

    def __init__(self, *, dynamics: SingleDendriteDynamics, connectivity_fn=None) -> None:
        self._dynamics = dynamics
        self._connectivity_fn = connectivity_fn

    def step(
        self,
        s_prev: torch.Tensor,
        phi_t: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute next state applying connectivity and dynamics step."""
        # Apply internal connectivity if present
        if self._connectivity_fn is not None:
            phi_t = self._connectivity_fn(state=s_prev, phi=phi_t, params=params)

        # Apply phi offset
        phi_offset = params.get("phi_offset")
        if phi_offset is not None:
            phi_t = phi_t + phi_offset

        # Use dynamics step function
        return self._dynamics.step(s_prev, phi_t, params, dt)


__all__ = ["SingleDendriteLayer"]
