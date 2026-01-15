# FILEPATH: src/soen_toolkit/core/soen_model_core.py
"""Core SOEN model implementation.

This module provides the main ``SOENModelCore`` class which orchestrates
layers and connections for superconducting neural network simulations.

State Management Architecture
==============================

The model manages two types of stateful components for dynamic connections:

1. Layer-Level Internal State (intra-layer connections):
   - Location: layer._internal_conn_state
   - Lifecycle: Reset automatically at start of layer.forward()
   - Scope: Single layer's self-connections (internal connectivity)
   - Used by: SingleDendrite, Multiplier, MultiplierNOCC with internal_J

2. Model-Level Connection State (inter-layer connections, stepwise only):
   - Location: model._conn_states (edge states for WICC/NOCC connections)
   - Location: model._multiplier_nocc_layer_states (s1/s2 states for MultiplierNOCC layers)
   - Lifecycle: Reset automatically at start of stepwise forward passes
   - Scope: All inter-layer dynamic connections
   - Used by: _forward_single_timestep() (Gauss-Seidel), _forward_single_timestep_jacobi()

CRITICAL: All state must be reset between forward passes to ensure deterministic
behavior. This is handled automatically by calling reset_stateful_components()
at the start of each stepwise forward pass. Users can also call this method
manually if needed (e.g., model.reset_stateful_components()).

For more details, see the reset_stateful_components() method documentation.
"""

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Union, cast

import torch
from torch import nn

# Visualization imports moved to visualize() and visualize_grid_of_grids() methods (lazy-load)
# to avoid importing graphviz at startup
from soen_toolkit.core.configs import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
)
from soen_toolkit.core.layer_registry import is_multiplier_nocc
from soen_toolkit.core.mixins import (
    BuildersMixin,
    ConfigUpdateMixin,
    ConstraintsMixin,
    GradientAnalysisMixin,
    IOMixin,
    QuantizationMixin,
    SummaryMixin,
)
from soen_toolkit.core.noise import NoiseSettings, build_noise_strategies
from soen_toolkit.core.utils.graph import build_inbound_by_target, precompute_external_J
from soen_toolkit.core.utils.inputs import (
    adjust_first_layer_flux,
    adjust_first_layer_state,
)
from soen_toolkit.core.utils.tracking import (
    begin_step_accumulators,
    collect_step_for_layer,
    rebuild_histories_stepwise,
)

if TYPE_CHECKING:
    from pathlib import Path

# Library modules should not configure global logging.
logger = logging.getLogger(__name__)


####################################
# Utilities and Helpers
####################################


####################################
# Main SOENModelCore Class
####################################


class SOENModelCore(
    nn.Module,
    GradientAnalysisMixin,
    QuantizationMixin,
    SummaryMixin,
    ConstraintsMixin,
    ConfigUpdateMixin,
    IOMixin,
    BuildersMixin,
):
    """Core Model containing the network architecture and forward pass.

    Attributes:
        dt (float): Time step for integration
        sim_config (SimulationConfig): Configuration for simulation parameters
        layers_config (List[LayerConfig]): Configuration for each layer
        connections_config (List[ConnectionConfig]): Configuration for connections
        noise_config (Dict): Configuration for noise during training and evaluation
        layers (nn.ModuleList): List of layer modules
        connections (nn.ParameterDict): Dictionary of connection weight matrices
        connection_masks (Dict): Dictionary of connection masks
        connection_constraints (Dict): Dictionary of constraints for connections

    """

    layers: nn.ModuleList
    connections: nn.ParameterDict
    phi_history: list[torch.Tensor]
    layer_nodes: dict[int, int]
    connection_constraint_min_matrices: dict[str, torch.Tensor]
    connection_constraint_max_matrices: dict[str, torch.Tensor]
    connection_masks: dict[str, torch.Tensor]
    connection_constraints: dict[str, dict[str, float]]
    _multiplier_nocc_layer_states: dict[int, dict[str, torch.Tensor]]
    _conn_states: dict[str, Any]

    def __init__(
        self,
        sim_config: SimulationConfig | None = None,
        layers_config: list[LayerConfig] | None = None,
        connections_config: list[ConnectionConfig] | None = None,
    ) -> None:
        """Initialize a SOEN model.

        Args:
            layers_config: List of layer configurations.
            connections_config: List of connection configurations.
            sim_config: Simulation configuration.

        """
        super().__init__()

        # Set default values if not provided
        self.layers_config = sorted(layers_config or [], key=lambda x: x.layer_id)
        self.connections_config = connections_config or []
        self.sim_config = sim_config or SimulationConfig()

        # Extract simulation parameters
        if getattr(self.sim_config, "dt_learnable", False):
            self.dt = nn.Parameter(torch.tensor(float(self.sim_config.dt), dtype=torch.float32), requires_grad=True)

        else:
            self.dt = float(self.sim_config.dt)
        self.num_layers = len(self.layers_config)

        # Initialize model components
        self.layers = nn.ModuleList()
        self.phi_history = []  # Initialize phi_history for the model
        self.layer_nodes = self._collect_layer_dimensions()

        # Per-edge constraint matrices for neuron polarity (excitatory/inhibitory)
        # Must be initialized BEFORE _build_connections() is called
        self.connection_constraint_min_matrices: dict[str, torch.Tensor] = {}
        self.connection_constraint_max_matrices: dict[str, torch.Tensor] = {}

        # Create connection structures
        self.connections, self.connection_masks, self.connection_constraints = self._build_connections()
        self.connection_noise_settings = self._build_connection_noise_settings()

        # Precompute per-layer noise settings once; reset each forward
        self._layer_noise_settings: dict[int, NoiseSettings] = {}
        for cfg in self.layers_config:
            try:
                self._layer_noise_settings[cfg.layer_id] = build_noise_strategies(cfg.noise, cfg.perturb)
            except Exception as e:
                msg = f"Failed to build noise strategies for layer {cfg.layer_id} ({cfg.layer_type}). Check noise and perturbation configuration. Error: {e}"
                raise ValueError(msg) from e

        # Calculate fan-out for each node
        node_fan_outs = self._calculate_fan_outs()

        # Create layer modules
        self._create_layers(node_fan_outs)

        # Centralise dt: bind layers to the model's dt tensor when learnable
        if bool(getattr(self.sim_config, "dt_learnable", False)):
            for layer in self.layers:
                if hasattr(layer, "set_dt_reference"):
                    l_any = cast(Any, layer)
                    l_any.set_dt_reference(self.dt if isinstance(self.dt, torch.Tensor) else torch.tensor(float(self.dt), dtype=torch.float32))
                else:
                    with contextlib.suppress(Exception):
                        cast(Any, layer).dt = self.dt

        # Enforce parameter constraints
        self.enforce_param_constraints()

        # QAT-STE configuration (inactive by default)
        self._qat_ste_active: bool = False
        self._qat_codebook: torch.Tensor | None = None
        self._qat_target_connection_names: set[str] | None = None
        self._qat_stochastic_rounding: bool = False

    @classmethod
    def build(
        cls,
        config: Union[str, "Path", dict[str, Any]],
    ) -> "SOENModelCore":
        """Construct a model from a YAML/JSON spec path or a config dict.

        Accepts:
            - str/Path to .yaml/.yml: uses YAML model spec parser
            - str/Path to .json: auto-detects JSON schema
                - if exported JSON (with connections.config/matrices): uses JSON loader
                - if minimal spec (simulation/layers/connections lists): uses YAML-style parser
            - dict: treated as a YAML-style config (simulation/layers/connections)
        """
        # Local imports to avoid circular dependencies
        import json as _json
        from pathlib import Path as _Path

        # Helper to build from parsed YAML-style dict
        def _from_yaml_dict(d: dict[str, Any]) -> "SOENModelCore":
            from soen_toolkit.core.model_yaml import parse_model_yaml

            sim_cfg, layers_cfg, conns_cfg = parse_model_yaml(d)
            return cls(
                sim_config=sim_cfg,
                layers_config=layers_cfg,
                connections_config=conns_cfg,
            )

        # Path-like inputs
        if isinstance(config, (str, _Path)):
            path = _Path(str(config))
            suffix = path.suffix.lower()
            if suffix in {".yaml", ".yml"}:
                # Delegate to existing YAML builder
                from soen_toolkit.core.model_yaml import build_model_from_yaml

                return build_model_from_yaml(path)

            if suffix == ".json":
                # Detect JSON schema and dispatch accordingly
                with path.open("r") as f:
                    data = _json.load(f) or {}

                # Exported JSON has connections as a dict with config/matrices
                connections_obj = data.get("connections", {})
                if isinstance(connections_obj, dict) and (("config" in connections_obj) or ("matrices" in connections_obj)):
                    from soen_toolkit.utils.model_tools import model_from_json

                    return model_from_json(str(path))

                # Otherwise expect a minimal spec (YAML-style structure)
                if isinstance(data.get("layers"), list) and isinstance(data.get("simulation", {}), dict):
                    return _from_yaml_dict(data)

                msg = f"Unrecognized JSON schema in '{path}'. Expected exported model JSON or a spec with 'simulation'/'layers'/'connections'."
                raise ValueError(
                    msg,
                )

            msg = f"Unsupported file extension '{suffix}' for '{path}'. Supported: .yaml/.yml or .json"
            raise ValueError(
                msg,
            )

        # Dict-like input treated as YAML-style spec
        if isinstance(config, dict):
            return _from_yaml_dict(config)

        msg = "config must be a path to a YAML/JSON spec or a dict with keys 'simulation', 'layers', 'connections'"
        raise TypeError(
            msg,
        )

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    def reset_stateful_components(self) -> None:
        """Reset all stateful components for deterministic forward passes.

        This method clears dynamic state that can persist between forward passes:
        - Connection edge states for dynamic WICC/NOCC connections (_conn_states)
        - MultiplierNOCC layer internal states s1/s2 (_multiplier_nocc_layer_states)
        - Individual layer internal connection states

        Called automatically at the start of each forward pass in stepwise solvers
        to ensure deterministic behavior. Can also be called manually if needed.

        Example:
            >>> model.reset_stateful_components()
            >>> out1, _ = model(x)  # Clean state
            >>> out2, _ = model(x)  # Identical to out1

        Note:
            Layerwise solver resets layer-level internal states automatically
            in each layer's forward() method. Stepwise solvers require model-level
            state reset, which this method provides.
        """
        # Reset model-level connection edge states (stepwise solvers only)
        # These track dynamic WICC/NOCC connection states between layers
        if hasattr(self, "_conn_states"):
            for state in self._conn_states.values():
                cast(Any, state).edge_state = None

        # Reset MultiplierNOCC layer auxiliary states (stepwise solvers only)
        # These track s1 and s2 states that persist across timesteps
        if hasattr(self, "_multiplier_nocc_layer_states"):
            self._multiplier_nocc_layer_states = {}

        # Reset individual layer internal connection states
        # This handles intra-layer (self) connections
        for layer in self.layers:
            l_any = cast(Any, layer)
            if hasattr(l_any, "_internal_conn_state") and l_any._internal_conn_state is not None:
                l_any._internal_conn_state.edge_state = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        external_phi: torch.Tensor,
        *,
        initial_states: dict[int, torch.Tensor] | None = None,
        s1_inits: dict[int, torch.Tensor] | None = None,
        s2_inits: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through the model.

        Noise is injected into each layer according to its specific noise configuration.

        Args:
            external_phi: Input tensor of shape [batch, seq_len, input_dim]
            initial_states: Optional dict mapping layer_id to initial state tensors
            s1_inits: Optional dict mapping layer_id to s1 initial states (for MultiplierNOCC)
            s2_inits: Optional dict mapping layer_id to s2 initial states (for MultiplierNOCC)

        Returns:
            Tuple containing:
                - Final layer output tensor
                - List of all layer output tensors

        """
        if external_phi.dim() != 3:
            msg = f"Expected external_phi shape [batch, seq_len, input_dim], got {external_phi.shape}"
            raise ValueError(
                msg,
            )

        # Enforce all constraints once per forward BEFORE computation so this pass
        # uses clamped parameters (both layer params and connections, plus masks).
        self.enforce_param_constraints()

        # Centralised dt: layers are bound to the model's dt parameter (if learnable)
        # during construction; no per-forward mutation is performed here.

        batch, seq_len, _ = external_phi.shape
        device = external_phi.device

        # Reset connection noise so perturbations differ each forward pass
        for setting in self.connection_noise_settings.values():
            if hasattr(setting, "reset"):
                setting.reset()

        # Reset per-layer noise offsets (GaussianPerturbation caches per batch)
        for setting in getattr(self, "_layer_noise_settings", {}).values():
            if hasattr(setting, "reset"):
                setting.reset()

        # Global solver mode selection
        solver_mode = str(getattr(self.sim_config, "network_evaluation_method", "layerwise")).lower()
        if solver_mode not in {"layerwise", "stepwise_gauss_seidel", "stepwise_jacobi"}:
            solver_mode = "layerwise"

        # Early stopping is only supported in stepwise solvers.
        if getattr(self.sim_config, "early_stopping_forward_pass", False) and solver_mode == "layerwise":
            msg = "early_stopping_forward_pass is not supported with network_evaluation_method='layerwise'. Use 'stepwise_gauss_seidel' or 'stepwise_jacobi', or disable early stopping."
            raise ValueError(
                msg,
            )

        # Stepwise network solvers (synchronous updates per time step)
        if solver_mode == "stepwise_gauss_seidel":
            return self._forward_single_timestep(external_phi, initial_states=initial_states, s1_inits=s1_inits, s2_inits=s2_inits)
        if solver_mode == "stepwise_jacobi":
            return self._forward_single_timestep_jacobi(external_phi, initial_states=initial_states, s1_inits=s1_inits, s2_inits=s2_inits)

        # Dictionary to store output history for each layer
        s_histories = {}

        # Build per-layer NoiseSettings once per forward to honor runtime edits
        per_layer_noise: dict[int, NoiseSettings] = {}
        for cfg in self.layers_config:
            per_layer_noise[cfg.layer_id] = build_noise_strategies(cfg.noise, cfg.perturb)

        # Process first layer based on input type
        first_layer_cfg = self.layers_config[0]
        first_noise_settings = per_layer_noise[first_layer_cfg.layer_id]
        s_histories = self._process_first_layer(
            external_phi,
            batch,
            seq_len,
            device,
            first_noise_settings,
            initial_states.get(first_layer_cfg.layer_id) if initial_states is not None else None,
        )

        # Process remaining layers
        for idx in range(1, self.num_layers):
            curr_cfg = self.layers_config[idx]
            curr_id = curr_cfg.layer_id
            curr_dim = self.layer_nodes[curr_id]
            curr_noise_settings = per_layer_noise[curr_cfg.layer_id]

            # Collect inputs from all previous layers
            upstream_phi = self._collect_upstream_contributions(
                curr_id,
                batch,
                seq_len,
                curr_dim,
                device,
                s_histories,
                curr_noise_settings,
            )

            # Process the current layer with its specific noise config and optional initial state
            init_state = None
            if initial_states is not None:
                init_state = initial_states.get(curr_id)

            # Check for MultiplierNOCC s1/s2 initial states
            s1_init = None
            s2_init = None
            if s1_inits is not None:
                s1_init = s1_inits.get(curr_id)
            if s2_inits is not None:
                s2_init = s2_inits.get(curr_id)

            # Try to pass all available initial states to the layer
            try:
                if s1_init is not None or s2_init is not None:
                    # MultiplierNOCC layer with s1/s2 states
                    s_layer = self.layers[idx](
                        upstream_phi,
                        noise_config=curr_noise_settings,
                        initial_state=init_state,
                        s1_init=s1_init,
                        s2_init=s2_init
                    )
                else:
                    # Standard layer
                    s_layer = self.layers[idx](upstream_phi, noise_config=curr_noise_settings, initial_state=init_state)
            except TypeError:
                # Fallback if layer doesn't accept some of these arguments
                s_layer = self.layers[idx](upstream_phi, noise_config=curr_noise_settings)

            s_histories[curr_id] = s_layer

        # Get final layer output
        final_layer_id = self.layers_config[-1].layer_id
        final_history = s_histories[final_layer_id]

        # Populate model's phi_history if tracking is enabled
        if self.sim_config.track_phi:
            self.phi_history = self.get_phi_history()

        # All constraints were applied before computation; no need to repeat here.

        return final_history, list(s_histories.values())

    def _forward_single_timestep(
        self,
        external_phi: torch.Tensor,
        *,
        initial_states: dict[int, torch.Tensor] | None = None,
        s1_inits: dict[int, torch.Tensor] | None = None,
        s2_inits: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Advance the entire network one time step at a time.

        Uses a Gauss–Seidel style sweep within each time step: layers are updated
        in fixed layer_id order using the freshest available neighbor states
        inside the same step. Feedforward edges propagate immediately within the
        step. Backward/feedback edges (higher id -> lower id) incur exactly a
        one-step (Δt) delay.

        Each layer is advanced one step by reusing its own solver with a
        length‑1 sequence and the current state as the initial condition.
        """
        batch, seq_len, _ = external_phi.shape
        device = external_phi.device

        # Reset all stateful components for determinism
        self.reset_stateful_components()

        # Map layer_id -> index in self.layers / self.layers_config
        id_to_idx: dict[int, int] = {cfg.layer_id: i for i, cfg in enumerate(self.layers_config)}

        # Build inbound adjacency (only inter‑layer connections; internal handled by the layer)
        inbound_by_target = build_inbound_by_target(self.connections_config, self.connections)

        # Precompute effective external connection matrices for this forward
        # using the authoritative connections_config (avoids brittle key parsing).
        # QAT‑STE and connection noise on J are applied once per forward.
        J_eff_by_key: dict[str, torch.Tensor] = precompute_external_J(self)

        # Prepare per‑layer noise settings (fresh objects per forward to honor runtime edits)
        per_layer_noise: dict[int, NoiseSettings] = {}
        for cfg in self.layers_config:
            per_layer_noise[cfg.layer_id] = build_noise_strategies(cfg.noise, cfg.perturb)

        # Stepwise: full tracking aggregation implemented for φ/g/s and power/energy.

        # Prepare initial states and output history buffers
        s_prev: dict[int, torch.Tensor] = {}
        s_histories: dict[int, torch.Tensor] = {}
        phi_accum, g_accum, pb_accum, pd_accum = begin_step_accumulators(self.layers_config)

        for cfg in self.layers_config:
            dim = self.layer_nodes[cfg.layer_id]
            hist = torch.empty(batch, seq_len + 1, dim, device=device, dtype=external_phi.dtype)
            # Initial state per layer
            init = None
            if initial_states is not None:
                init = initial_states.get(cfg.layer_id)
            if init is not None:
                if init.dim() == 1:
                    base = init.to(device=device, dtype=external_phi.dtype).view(1, -1)
                    s0 = base.expand(batch, -1).clone()
                else:
                    s0 = init.to(device=device, dtype=external_phi.dtype).clone()
                    if s0.shape != (batch, dim):
                        msg = f"initial_state for layer {cfg.layer_id} must be [batch, {dim}] or [{dim}]"
                        raise ValueError(msg)
            else:
                s0 = torch.zeros(batch, dim, device=device, dtype=external_phi.dtype)
            hist[:, 0, :] = s0
            s_prev[cfg.layer_id] = s0
            s_histories[cfg.layer_id] = hist

        # Helper: external input handling for first layer
        first_layer_id = self.layers_config[0].layer_id
        input_type = getattr(self.sim_config, "input_type", "flux").lower()

        # If input_type is 'state', pre‑adjust external input dims for first layer
        external_phi_adjusted = None
        if input_type == "state":
            first_dim = self.layer_nodes[first_layer_id]
            external_phi_adjusted = adjust_first_layer_state(external_phi, first_dim, warn=True)
            # Propagate initial s0 through subsequent Input layers in order
            for cfg in self.layers_config[1:]:
                lid = cfg.layer_id
                # Only adjust Input layers (placeholders)
                try:
                    if str(cfg.layer_type).lower() != "input":
                        continue
                except Exception:
                    continue
                dim = self.layer_nodes[lid]
                phi0 = torch.zeros(batch, dim, device=device, dtype=external_phi.dtype)
                for src_id, key in inbound_by_target.get(lid, []):
                    s_src0 = s_prev.get(src_id)
                    if s_src0 is None:
                        continue
                    J_eff = J_eff_by_key.get(key)
                    if J_eff is None:
                        continue
                    phi0 = phi0 + torch.matmul(s_src0, J_eff.t())
                s_histories[lid][:, 0, :] = phi0
                s_prev[lid] = phi0
        # If input_type is 'flux', also pre-adjust once per forward for consistency/perf
        external_flux_adjusted = None
        if input_type == "flux":
            first_dim = self.layer_nodes[first_layer_id]
            external_flux_adjusted = adjust_first_layer_flux(external_phi, first_dim, warn=True)
        # Early stopping configuration (stepwise only)
        es_active: bool = bool(getattr(self.sim_config, "early_stopping_forward_pass", False))
        es_tol: float = float(getattr(self.sim_config, "early_stopping_tolerance", 1e-6))
        es_patience: int = int(getattr(self.sim_config, "early_stopping_patience", 1))
        es_min_steps: int = int(getattr(self.sim_config, "early_stopping_min_steps", 1))
        es_consec_ok = 0
        effective_seq_len = seq_len
        # Windowed steady-state configuration
        wsize: int = int(getattr(self.sim_config, "steady_window_min", 0))
        w_abs: float = float(getattr(self.sim_config, "steady_tol_abs", 1e-5))
        w_rel: float = float(getattr(self.sim_config, "steady_tol_rel", 1e-3))
        win_delta_maxes: list[float] = []
        win_state_maxes: list[float] = []

        # Time loop -----------------------------------------------------------
        for t in range(seq_len):
            step_max_delta = 0.0
            step_max_state = 0.0
            # Always Gauss–Seidel: update in order using freshest sources
            s_next_map: dict[int, torch.Tensor] = {}
            step_max_state = 0.0
            for cfg in self.layers_config:
                lid = cfg.layer_id
                layer_idx = id_to_idx[lid]
                layer_mod = cast(Any, self.layers[layer_idx])
                dim = self.layer_nodes[lid]
                # Avoid within-step history growth: clear histories before step advance
                try:
                    layer_mod._clear_phi_history()
                    layer_mod._clear_g_history()
                    layer_mod._clear_state_history()
                except Exception:
                    pass
                noise_cfg = per_layer_noise[lid]

                if (lid == first_layer_id) and (input_type == "state"):
                    prev_state = s_prev[lid]
                    s_next = cast(torch.Tensor, external_phi_adjusted)[:, t, :]
                    if noise_cfg and getattr(noise_cfg, "s", None) is not None:
                        s_next = layer_mod._apply_noise(s_next, noise_cfg, "s")
                    # Track max |Δs| for early stopping
                    if es_active:
                        d = torch.amax(torch.abs(s_next - prev_state)).item()
                        step_max_delta = max(step_max_delta, d)
                    # Track max |s| for windowed relative tolerance
                    step_state = torch.amax(torch.abs(s_next)).item()
                    step_max_state = max(step_max_state, step_state)
                else:
                    if (lid == first_layer_id) and (input_type == "flux"):
                        phi_sum = cast(torch.Tensor, external_flux_adjusted)[:, t, :]
                    else:
                        phi_sum = torch.zeros(batch, dim, device=device, dtype=external_phi.dtype)

                    for src_id, key in inbound_by_target.get(lid, []):
                        # Use freshest available state this step; fallback to previous step
                        s_src = s_next_map.get(src_id, s_prev[src_id])
                        J_eff = J_eff_by_key.get(key)
                        if J_eff is None:
                            continue
                        # Use unified connection helper
                        from soen_toolkit.core.utils.connection_ops import ConnectionState, apply_connection_step
                        from soen_toolkit.core.utils.graph import apply_connection_noise_step

                        # Apply per-timestep noise for GaussianNoise (perturbation already in J_eff)
                        conn_noise = self.connection_noise_settings.get(key)
                        J_noisy = apply_connection_noise_step(J_eff, conn_noise)

                        mode = getattr(self, "_connection_modes", {}).get(key, "fixed")
                        params = getattr(self, "_connection_params", {}).get(key, {})
                        edge_indices = getattr(self, "_connection_edge_maps", {}).get(key, (None, None))
                        if not hasattr(self, "_conn_states"):
                            self._conn_states = {}
                        if key not in self._conn_states:
                            self._conn_states[key] = ConnectionState()
                        phi_add = apply_connection_step(s_src, J_noisy, mode, params, edge_indices, self._conn_states[key], self.dt)
                        phi_sum = phi_sum + phi_add

                    up_phi = phi_sum.unsqueeze(1)
                    init_state = s_prev[lid]

                    # Handle MultiplierNOCC layer internal states
                    layer_type = str(cfg.layer_type).lower()
                    s1_init = None
                    s2_init = None
                    if is_multiplier_nocc(layer_type):
                        if not hasattr(self, "_multiplier_nocc_layer_states"):
                            self._multiplier_nocc_layer_states = {}
                        nocc_layer_state: dict[str, torch.Tensor] = self._multiplier_nocc_layer_states.get(lid, {})
                        s1_init = nocc_layer_state.get("s1")
                        s2_init = nocc_layer_state.get("s2")

                    if hasattr(layer_mod, "forward_euler_integration"):
                        try:
                            if is_multiplier_nocc(layer_type):
                                s_hist_step = layer_mod.forward_euler_integration(
                                    up_phi,
                                    noise_config=noise_cfg,
                                    initial_state=init_state,
                                    s1_init=s1_init,
                                    s2_init=s2_init,
                                )
                            else:
                                s_hist_step = layer_mod.forward_euler_integration(up_phi, noise_config=noise_cfg, initial_state=init_state)
                        except TypeError:
                            s_hist_step = layer_mod.forward_euler_integration(up_phi, noise_config=noise_cfg)
                    else:
                        try:
                            if is_multiplier_nocc(layer_type):
                                s_hist_step = layer_mod(up_phi, noise_config=noise_cfg, initial_state=init_state, s1_init=s1_init, s2_init=s2_init)
                            else:
                                s_hist_step = layer_mod(up_phi, noise_config=noise_cfg, initial_state=init_state)
                        except TypeError:
                            s_hist_step = layer_mod(up_phi, noise_config=noise_cfg)
                    s_next = s_hist_step[:, 1, :]

                    # Store updated s1, s2 for next iteration
                    if layer_type in ("multipliernocc", "multiplier_nocc", "multiplierv2", "multiplier_v2") and hasattr(layer_mod, "_s1_final"):
                        self._multiplier_nocc_layer_states[lid] = {
                            "s1": layer_mod._s1_final,
                            "s2": layer_mod._s2_final,
                        }

                    # Track max |Δs| for early stopping
                    if es_active:
                        d = torch.amax(torch.abs(s_next - init_state)).item()
                        step_max_delta = max(step_max_delta, d)
                    # Track max |s| for windowed relative tolerance
                    step_state = torch.amax(torch.abs(s_next)).item()
                    step_max_state = max(step_max_state, step_state)

                    collect_step_for_layer(
                        lid,
                        layer_mod,
                        self.sim_config,
                        phi_accum,
                        g_accum,
                        pb_accum,
                        pd_accum,
                    )

                s_histories[lid][:, t + 1, :] = s_next
                s_prev[lid] = s_next
                s_next_map[lid] = s_next

            # Early stopping decision after completing this time step
            if es_active:
                # Update windowed buffers
                if wsize > 0:
                    win_delta_maxes.append(step_max_delta)
                    win_state_maxes.append(step_max_state)
                    if len(win_delta_maxes) > wsize:
                        win_delta_maxes.pop(0)
                    if len(win_state_maxes) > wsize:
                        win_state_maxes.pop(0)
                # Decide by mode
                if (t + 1) >= es_min_steps:
                    if wsize > 0 and len(win_delta_maxes) >= wsize:
                        thr = w_abs + w_rel * max(1e-12, *win_state_maxes)
                        if max(win_delta_maxes) <= thr:
                            effective_seq_len = t + 1
                            break
                    else:
                        if step_max_delta <= es_tol:
                            es_consec_ok += 1
                        else:
                            es_consec_ok = 0
                        if es_consec_ok >= es_patience:
                            effective_seq_len = t + 1
                            break

        # Final outputs -------------------------------------------------------
        final_layer_id = self.layers_config[-1].layer_id
        final_history = s_histories[final_layer_id]

        rebuild_histories_stepwise(
            self,
            seq_len=effective_seq_len,
            batch=batch,
            device=device,
            dtype=external_phi.dtype,
            s_histories=s_histories,
            phi_accum=phi_accum,
            g_accum=g_accum,
            pb_accum=pb_accum,
            pd_accum=pd_accum,
            first_layer_id=first_layer_id,
            input_type=input_type,
        )

        # Slice histories to the effective length (early stop may shorten seq)
        if effective_seq_len != seq_len:
            for lid in list(s_histories.keys()):
                s_histories[lid] = s_histories[lid][:, : effective_seq_len + 1, :]
            final_history = final_history[:, : effective_seq_len + 1, :]

        # Return histories in layer order
        ordered_histories = [s_histories[cfg.layer_id] for cfg in self.layers_config]
        return final_history, ordered_histories

    def _forward_single_timestep_jacobi(
        self,
        external_phi: torch.Tensor,
        *,
        initial_states: dict[int, torch.Tensor] | None = None,
        s1_inits: dict[int, torch.Tensor] | None = None,
        s2_inits: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Advance the entire network one time step at a time (Jacobi).

        Snapshot-synchronous updates: for each timestep t, compute φ_t for every
        layer from s_prev only, then advance each layer once using that φ_t. This
        makes per-layer micro-steps independent and amenable to parallelization.
        """
        batch, seq_len, _ = external_phi.shape
        device = external_phi.device

        # Reset all stateful components for determinism
        self.reset_stateful_components()

        id_to_idx: dict[int, int] = {cfg.layer_id: i for i, cfg in enumerate(self.layers_config)}

        inbound_by_target = build_inbound_by_target(self.connections_config, self.connections)
        J_eff_by_key: dict[str, torch.Tensor] = precompute_external_J(self)

        per_layer_noise: dict[int, NoiseSettings] = {}
        for cfg in self.layers_config:
            per_layer_noise[cfg.layer_id] = build_noise_strategies(cfg.noise, cfg.perturb)

        # Prepare initial states and output buffers
        s_prev: dict[int, torch.Tensor] = {}
        s_histories: dict[int, torch.Tensor] = {}
        for cfg in self.layers_config:
            dim = self.layer_nodes[cfg.layer_id]
            hist = torch.empty(batch, seq_len + 1, dim, device=device, dtype=external_phi.dtype)
            init = None
            if initial_states is not None:
                init = initial_states.get(cfg.layer_id)
            if init is not None:
                if init.dim() == 1:
                    base = init.to(device=device, dtype=external_phi.dtype).view(1, -1)
                    s0 = base.expand(batch, -1).clone()
                else:
                    s0 = init.to(device=device, dtype=external_phi.dtype).clone()
                    if s0.shape != (batch, dim):
                        msg = f"initial_state for layer {cfg.layer_id} must be [batch, {dim}] or [{dim}]"
                        raise ValueError(msg)
            else:
                s0 = torch.zeros(batch, dim, device=device, dtype=external_phi.dtype)
            hist[:, 0, :] = s0
            s_prev[cfg.layer_id] = s0
            s_histories[cfg.layer_id] = hist

        first_layer_id = self.layers_config[0].layer_id
        input_type = getattr(self.sim_config, "input_type", "flux").lower()

        external_phi_adjusted = None
        if input_type == "state":
            first_dim = self.layer_nodes[first_layer_id]
            external_phi_adjusted = adjust_first_layer_state(external_phi, first_dim, warn=True)

        external_flux_adjusted = None
        if input_type == "flux":
            first_dim = self.layer_nodes[first_layer_id]
            external_flux_adjusted = adjust_first_layer_flux(external_phi, first_dim, warn=True)

        # Accumulators for tracking
        phi_accum, g_accum, pb_accum, pd_accum = begin_step_accumulators(self.layers_config)

        # Initialize layer internal states for MultiplierNOCC
        if not hasattr(self, "_multiplier_nocc_layer_states"):
            self._multiplier_nocc_layer_states = {}

        for cfg in self.layers_config:
            layer_type = str(cfg.layer_type).lower()
            if is_multiplier_nocc(layer_type):
                dim = self.layer_nodes[cfg.layer_id]
                if cfg.layer_id not in self._multiplier_nocc_layer_states:
                    self._multiplier_nocc_layer_states[cfg.layer_id] = {
                        "s1": torch.zeros(batch, dim, device=device, dtype=external_phi.dtype),
                        "s2": torch.zeros(batch, dim, device=device, dtype=external_phi.dtype),
                    }

        # Early stopping configuration (stepwise only)
        es_active: bool = bool(getattr(self.sim_config, "early_stopping_forward_pass", False))
        es_tol: float = float(getattr(self.sim_config, "early_stopping_tolerance", 1e-6))
        es_patience: int = int(getattr(self.sim_config, "early_stopping_patience", 1))
        es_min_steps: int = int(getattr(self.sim_config, "early_stopping_min_steps", 1))
        es_consec_ok = 0
        effective_seq_len = seq_len
        # Windowed steady-state configuration
        wsize: int = int(getattr(self.sim_config, "steady_window_min", 0))
        w_abs: float = float(getattr(self.sim_config, "steady_tol_abs", 1e-5))
        w_rel: float = float(getattr(self.sim_config, "steady_tol_rel", 1e-3))
        win_delta_maxes: list[float] = []
        win_state_maxes: list[float] = []

        for t in range(seq_len):
            step_max_delta = 0.0
            step_max_state = 0.0
            # Build φ from s_prev for all layers
            phi_t_map: dict[int, torch.Tensor] = {}
            for cfg in self.layers_config:
                lid = cfg.layer_id
                dim = self.layer_nodes[lid]
                if (lid == first_layer_id) and (input_type == "flux"):
                    phi_sum = cast(torch.Tensor, external_flux_adjusted)[:, t, :]
                else:
                    phi_sum = torch.zeros(batch, dim, device=device, dtype=external_phi.dtype)
                for src_id, key in inbound_by_target.get(lid, []):
                    s_src = s_prev[src_id]
                    J_eff = J_eff_by_key.get(key)
                    if J_eff is None:
                        continue
                    # Jacobi requires special handling for dynamic connections:
                    # use CURRENT edge state for phi, then update for next iteration
                    from soen_toolkit.core.utils.connection_ops import (
                        ConnectionState,
                        MultiplierNOCCOp,
                        MultiplierOp,
                        StaticMatrixOp,
                    )
                    from soen_toolkit.core.utils.graph import apply_connection_noise_step

                    mode = getattr(self, "_connection_modes", {}).get(key, "fixed")
                    params = getattr(self, "_connection_params", {}).get(key, {})
                    src_idx, dst_idx = getattr(self, "_connection_edge_maps", {}).get(key, (None, None))

                    if mode == "fixed" or src_idx is None or dst_idx is None:
                        # Apply per-timestep noise for GaussianNoise (perturbation already in J_eff)
                        conn_noise = self.connection_noise_settings.get(key)
                        J_noisy = apply_connection_noise_step(J_eff, conn_noise)
                        phi_add = StaticMatrixOp.step(s_src, J_noisy)
                    elif mode == "WICC":
                        if not hasattr(self, "_conn_states"):
                            self._conn_states = {}
                        if key not in self._conn_states:
                            self._conn_states[key] = ConnectionState()
                        state_obj = self._conn_states[key]
                        # Allocate WICC edge state if needed
                        E = int(src_idx.numel())
                        B = s_src.shape[0]
                        if state_obj.edge_state is None or state_obj.edge_state.shape != (B, E):
                            state_obj.edge_state = torch.zeros(B, E, device=s_src.device, dtype=s_src.dtype)
                        # Jacobi: use current edge state for phi, then update
                        j_out = params.get("j_out", 0.38)
                        phi_prev = torch.zeros(B, dim, device=s_src.device, dtype=s_src.dtype)
                        # Scatter edge states first, then apply j_out (supports scalar or per-destination vector)
                        phi_prev.index_add_(dim=1, index=dst_idx, source=state_obj.edge_state)
                        if isinstance(j_out, torch.Tensor):
                            phi_prev = phi_prev * j_out.to(device=s_src.device, dtype=s_src.dtype)
                        else:
                            phi_prev = phi_prev * float(j_out)
                        # Update edge state for next iteration
                        phi_y_add = 0.5 if params.get("half_flux_offset", False) else 0.0
                        _, state_obj.edge_state = MultiplierOp.step(
                            s_src,
                            J_eff,
                            src_idx,
                            dst_idx,
                            state_obj.edge_state,
                            dt=self.dt,
                            gamma_plus=params.get("gamma_plus", 0.001),
                            gamma_minus=params.get("gamma_minus", 0.001),
                            bias_current=params.get("bias_current", 2.0),
                            j_in=params.get("j_in", 0.38),
                            j_out=j_out,
                            source_func_key=params.get("source_func", "RateArray"),
                            phi_y_add=phi_y_add,
                        )
                        phi_add = phi_prev
                    elif mode == "NOCC":
                        if not hasattr(self, "_conn_states"):
                            self._conn_states = {}
                        if key not in self._conn_states:
                            self._conn_states[key] = ConnectionState()
                        state_obj = self._conn_states[key]
                        # Allocate NOCC edge state if needed (tuple of s1, s2, m)
                        E = int(src_idx.numel())
                        B = s_src.shape[0]
                        D = J_eff.shape[0]
                        if state_obj.edge_state is None or not isinstance(state_obj.edge_state, tuple) or state_obj.edge_state[0].shape != (B, E) or state_obj.edge_state[2].shape != (B, D):
                            state_obj.edge_state = (
                                torch.zeros(B, E, device=s_src.device, dtype=s_src.dtype),
                                torch.zeros(B, E, device=s_src.device, dtype=s_src.dtype),
                                torch.zeros(B, D, device=s_src.device, dtype=s_src.dtype),
                            )
                        # Jacobi: use current m state for phi, then update
                        j_out = params.get("j_out", 0.38)
                        phi_prev = state_obj.edge_state[2] * j_out  # Current m state
                        # Update edge states for next iteration
                        phi_y_add = 0.5 if params.get("half_flux_offset", False) else 0.0
                        _, state_obj.edge_state = MultiplierNOCCOp.step(
                            s_src,
                            J_eff,
                            src_idx,
                            dst_idx,
                            state_obj.edge_state,
                            dt=self.dt,
                            alpha=params.get("alpha", 1.64053),
                            beta=params.get("beta", 303.85),
                            beta_out=params.get("beta_out", 91.156),
                            bias_current=params.get("bias_current", 2.1),
                            j_in=params.get("j_in", 0.38),
                            j_out=j_out,
                            source_func_key=params.get("source_func", "RateArray"),
                            phi_y_add=phi_y_add,
                        )
                        phi_add = phi_prev
                    else:
                        phi_add = StaticMatrixOp.step(s_src, J_eff)

                    phi_sum = phi_sum + phi_add
                phi_t_map[lid] = phi_sum

            # Advance each layer independently using φ_t and s_prev as init
            for cfg in self.layers_config:
                lid = cfg.layer_id
                layer_idx = id_to_idx[lid]
                layer_mod = cast(Any, self.layers[layer_idx])
                try:
                    layer_mod._clear_phi_history()
                    layer_mod._clear_g_history()
                    layer_mod._clear_state_history()
                except Exception:
                    pass
                noise_cfg = per_layer_noise[lid]
                if (lid == first_layer_id) and (input_type == "state"):
                    prev_state = s_prev[lid]
                    s_next = cast(torch.Tensor, external_phi_adjusted)[:, t, :]
                    if noise_cfg and getattr(noise_cfg, "s", None) is not None:
                        s_next = layer_mod._apply_noise(s_next, noise_cfg, "s")
                else:
                    up_phi = phi_t_map[lid].unsqueeze(1)
                    init_state = s_prev[lid]

                    # Handle MultiplierNOCC layer internal states
                    layer_type = str(cfg.layer_type).lower()
                    s1_init = None
                    s2_init = None
                    if is_multiplier_nocc(layer_type):
                        if not hasattr(self, "_multiplier_nocc_layer_states"):
                            self._multiplier_nocc_layer_states = {}
                        nocc_layer_state: dict[str, torch.Tensor] = self._multiplier_nocc_layer_states.get(lid, {})
                        s1_init = nocc_layer_state.get("s1")
                        s2_init = nocc_layer_state.get("s2")

                    if hasattr(layer_mod, "forward_euler_integration"):
                        try:
                            if is_multiplier_nocc(layer_type):
                                s_hist_step = layer_mod.forward_euler_integration(
                                    up_phi,
                                    noise_config=noise_cfg,
                                    initial_state=init_state,
                                    s1_init=s1_init,
                                    s2_init=s2_init,
                                )
                            else:
                                s_hist_step = layer_mod.forward_euler_integration(up_phi, noise_config=noise_cfg, initial_state=init_state)
                        except TypeError:
                            s_hist_step = layer_mod.forward_euler_integration(up_phi, noise_config=noise_cfg)
                    else:
                        try:
                            if is_multiplier_nocc(layer_type):
                                s_hist_step = layer_mod(up_phi, noise_config=noise_cfg, initial_state=init_state, s1_init=s1_init, s2_init=s2_init)
                            else:
                                s_hist_step = layer_mod(up_phi, noise_config=noise_cfg, initial_state=init_state)
                        except TypeError:
                            s_hist_step = layer_mod(up_phi, noise_config=noise_cfg)
                    s_next = s_hist_step[:, 1, :]

                    # Store updated s1, s2 for next iteration
                    if layer_type in ("multipliernocc", "multiplier_nocc", "multiplierv2", "multiplier_v2") and hasattr(layer_mod, "_s1_final"):
                        self._multiplier_nocc_layer_states[lid] = {
                            "s1": layer_mod._s1_final,
                            "s2": layer_mod._s2_final,
                        }

                    collect_step_for_layer(
                        lid,
                        layer_mod,
                        self.sim_config,
                        phi_accum,
                        g_accum,
                        pb_accum,
                        pd_accum,
                    )

                # Track max |Δs| for early stopping and |s| for windowed rel tol
                if es_active:
                    baseline = prev_state if ((lid == first_layer_id) and (input_type == "state")) else init_state
                    d = torch.amax(torch.abs(s_next - baseline)).item()
                    step_max_delta = max(step_max_delta, d)
                step_state = torch.amax(torch.abs(s_next)).item()
                step_max_state = max(step_max_state, step_state)

                s_histories[lid][:, t + 1, :] = s_next
                # Note: s_prev updated after layer loop or immediately? For Jacobi, we can update per layer since φ only used s_prev snapshot.
                s_prev[lid] = s_next

            # Early stopping decision after completing this time step
            if es_active:
                # Update windowed buffers
                if wsize > 0:
                    win_delta_maxes.append(step_max_delta)
                    win_state_maxes.append(step_max_state)
                    if len(win_delta_maxes) > wsize:
                        win_delta_maxes.pop(0)
                    if len(win_state_maxes) > wsize:
                        win_state_maxes.pop(0)
                if (t + 1) >= es_min_steps:
                    if wsize > 0 and len(win_delta_maxes) >= wsize:
                        thr = w_abs + w_rel * max(1e-12, *win_state_maxes)
                        if max(win_delta_maxes) <= thr:
                            effective_seq_len = t + 1
                            break
                    else:
                        if step_max_delta <= es_tol:
                            es_consec_ok += 1
                        else:
                            es_consec_ok = 0
                        if es_consec_ok >= es_patience:
                            effective_seq_len = t + 1
                            break

        final_layer_id = self.layers_config[-1].layer_id
        final_history = s_histories[final_layer_id]

        rebuild_histories_stepwise(
            self,
            seq_len=effective_seq_len,
            batch=batch,
            device=device,
            dtype=external_phi.dtype,
            s_histories=s_histories,
            phi_accum=phi_accum,
            g_accum=g_accum,
            pb_accum=pb_accum,
            pd_accum=pd_accum,
            first_layer_id=first_layer_id,
            input_type=input_type,
        )

        # Slice histories to the effective length (early stop may shorten seq)
        if effective_seq_len != seq_len:
            for lid in list(s_histories.keys()):
                s_histories[lid] = s_histories[lid][:, : effective_seq_len + 1, :]
            final_history = final_history[:, : effective_seq_len + 1, :]

        ordered_histories = [s_histories[cfg.layer_id] for cfg in self.layers_config]
        return final_history, ordered_histories

    def _process_first_layer(
        self,
        external_phi: torch.Tensor,
        batch: int,
        seq_len: int,
        device: torch.device,
        noise_settings: NoiseSettings,
        initial_state: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        # Delegate to utility for maintainability
        from soen_toolkit.core.utils.solver_utils import process_first_layer as _proc

        return _proc(self, external_phi, batch, seq_len, device, noise_settings, initial_state)

    def _collect_upstream_contributions(
        self,
        curr_id: int,
        batch: int,
        seq_len: int,
        curr_dim: int,
        device: torch.device,
        s_histories: dict,
        noise_settings: NoiseSettings,
    ) -> torch.Tensor:
        # Delegate to utility for maintainability
        from soen_toolkit.core.utils.solver_utils import (
            collect_upstream_contributions as _collect,
        )

        return _collect(self, curr_id, batch, seq_len, curr_dim, device, s_histories, noise_settings)

    def visualize(
        self,
        save_path: str | None = None,
        file_format: str = "png",
        dpi: int = 300,
        open_file: bool = False,
        edge_color: str = "black",
        bg_color: str = "white",
        simple_view: bool = True,
        show_descriptions: bool = False,
        show_internal: bool = True,
        # GUI-aligned options (new)
        orientation: str = "LR",
        edge_routing: str = "true",
        edge_thickness: float = 0.5,
        arrow_size: float = 0.5,
        layer_spacing: float = 1.0,
        inter_color: str = "#000000",
        intra_color: str = "#ff0000",
        layer_color: str = "#eae5be",
        show_layer_outline: bool = False,
        show_intra: bool = True,
        show_desc: bool = False,
        show_conn_type: bool = False,
        show_node_ids: bool = False,
        show_layer_ids: bool = True,
        show_neuron_polarity: bool = False,
        show_connection_polarity: bool = False,
        # Modern style passthrough
        theme: str = "default",
        font_name: str | None = None,
        title_font_size: int | None = None,
        desc_font_size: int | None = None,
        nodesep: float | None = None,
    ) -> str:
        """Visualize the model architecture using Graphviz.

        Saves the figure to disk and, in notebooks, also displays it inline for SVG/PNG/JPG.

        Args:
            save_path: Base path (no extension). Defaults to `Figures/Network_Diagrams/saved_network_diagram`.
            file_format: "png" (default), "svg", "jpg", or "pdf".
            dpi: DPI used by Graphviz and for raster outputs.
            open_file: If True, opens the saved file in the OS viewer.

            orientation: Graph layout direction ("LR" or "TB").
            edge_routing: Graphviz splines mode: "true", "false", or "ortho".
            edge_thickness: Edge line width.
            arrow_size: Arrowhead scaling.
            layer_spacing: Separation between layer columns (inches).
            inter_color: Color for inter-layer edges.
            intra_color: Color for intra-layer edges/loops.
            layer_color: Layer box background color.
            show_layer_outline: Draw an outline around layer boxes.
            show_intra: Show intra-layer connections.
            show_desc: Show layer descriptions beneath nodes (overlaid, no layout impact).
            show_conn_type: In simple view, label edges with `connection_type`.
            show_node_ids: In detailed view, show neuron indices.
            show_layer_ids: Show a title row with layer ID.
            show_neuron_polarity: In detailed view, color neurons by polarity (red=excitatory, blue=inhibitory).
            show_connection_polarity: In detailed view, color edges by weight sign (red=positive, blue=negative).

            edge_color: Back-compat
            if customized and `inter_color` is default, replaces `inter_color`.
            bg_color: Canvas background color.
            simple_view: If True (default), compact per-layer boxes. If False, detailed port-per-neuron view.
            show_descriptions: Back-compat alias for `show_desc`.
            show_internal: Back-compat alias for `show_intra`.

        Returns:
            str: Full file path of the saved diagram.

        """
        # Local imports for visualization
        from soen_toolkit.utils.visualization import visualize

        return visualize(
            model=self,
            save_path=save_path,
            file_format=file_format,
            dpi=dpi,
            open_file=open_file,
            edge_color=edge_color,
            bg_color=bg_color,
            simple_view=simple_view,
            show_descriptions=show_descriptions,
            show_internal=show_internal,
            # GUI-aligned passthrough
            orientation=orientation,
            edge_routing=edge_routing,
            edge_thickness=edge_thickness,
            arrow_size=arrow_size,
            layer_spacing=layer_spacing,
            inter_color=inter_color,
            intra_color=intra_color,
            layer_color=layer_color,
            show_layer_outline=show_layer_outline,
            show_intra=show_intra,
            show_desc=show_desc,
            show_conn_type=show_conn_type,
            show_node_ids=show_node_ids,
            show_layer_ids=show_layer_ids,
            show_neuron_polarity=show_neuron_polarity,
            show_connection_polarity=show_connection_polarity,
            # Modern style passthrough
            theme=theme,
            font_name=font_name,
            title_font_size=title_font_size,
            desc_font_size=desc_font_size,
            nodesep=nodesep,
        )

    def visualize_grid_of_grids(
        self,
        save_path: str | None = None,
        file_format: str = "png",
        dpi: int = 200,
        show: bool = True,
        **kwargs,
    ) -> str | None:
        """Visualize layers as a grid-of-grids (each layer drawn as a module).

        This uses a Matplotlib renderer inspired by the attached prototype.
        It ignores module IDs and treats each layer independently with:
          - top row = inputs, last row = outputs;
          - intra-layer edges from J_i_to_i;
          - inter-layer edges connect output row of source to input row of target.
        """
        # Local imports for visualization
        from soen_toolkit.utils.visualization import visualize_grid_of_grids

        return visualize_grid_of_grids(
            model=self,
            save_path=save_path,
            file_format=file_format,
            dpi=dpi,
            show=show,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Unified summary for GUI/API
    # ------------------------------------------------------------------

    def export_to_json(self, filename: str | None = None, *, return_json: bool = False) -> str | None:
        """Deprecated: Use model.save("path.json") instead."""
        msg = "export_to_json has been removed. Use model.save('<path>.json') to export JSON."
        raise RuntimeError(
            msg,
        )

    # ------------------------------------------------------------------
    # Convenience: Port current model instance to a ready JAXModel
    # ------------------------------------------------------------------
    def port_to_jax(
        self,
        *,
        prepare: bool = True,
        input_type: str | None = None,
        network_evaluation_method: str | None = None,
        force: bool = True,
    ):
        """Convert the current (possibly edited) model to a JAXModel.

        - Applies current dt, connection weights, and modes.
        - Optionally override input_type/network_evaluation_method for the JAXModel only.
        - If prepare=True, preloads tables outside JIT.
        """
        from soen_toolkit.utils.port_to_jax.convert import convert_core_model_to_jax

        jax_model = convert_core_model_to_jax(self)
        if input_type is not None:
            try:
                jax_model.input_type = str(input_type)
            except Exception:
                pass
        if network_evaluation_method is not None:
            try:
                jax_model.network_evaluation_method = str(network_evaluation_method)
            except Exception:
                pass
        return jax_model.prepare() if prepare else jax_model

    # ----------------------------------------------------------------------------
    # Graph Operations (Model as Source of Truth)
    # ----------------------------------------------------------------------------

    def subgraph(self, keep_layers: list[int]) -> "SOENModelCore":
        """Create a new independent model instance containing only the specified layers.

        Preserves all weights, masks, constraints, and attributes (including polarity).
        Connections are kept only if both source and target are in keep_layers.

        Args:
            keep_layers: List of layer IDs to include in the new model.

        Returns:
            A new, independent SOENModelCore instance.
        """
        from soen_toolkit.core.configs import ConnectionConfig, LayerConfig
        from soen_toolkit.utils.model_tools import rebuild_model_preserving_state

        keep_set = set(keep_layers)

        # Filter layers
        new_layers_cfg = [
            LayerConfig(
                layer_id=cfg.layer_id,
                model_id=getattr(cfg, "model_id", 0),
                layer_type=cfg.layer_type,
                params=dict(cfg.params),  # Deep copy of params (includes polarity list)
                description=getattr(cfg, "description", ""),
                noise=getattr(cfg, "noise", None),
                perturb=getattr(cfg, "perturb", None),
            )
            for cfg in self.layers_config
            if cfg.layer_id in keep_set
        ]

        if not new_layers_cfg:
            raise ValueError(f"Subgraph selection resulted in 0 layers. Requested: {keep_layers}")

        # Filter connections
        new_connections_cfg = []
        for cc in self.connections_config:
            if (cc.from_layer in keep_set) and (cc.to_layer in keep_set):
                new_connections_cfg.append(
                    ConnectionConfig(
                        from_layer=cc.from_layer,
                        to_layer=cc.to_layer,
                        connection_type=cc.connection_type,
                        params=dict(cc.params) if cc.params is not None else None,
                        learnable=cc.learnable,
                        noise=getattr(cc, "noise", None),
                        perturb=getattr(cc, "perturb", None),
                    )
                )

        # Rebuild using state preservation utility
        # preserve_mode="all" ensures weights/masks are copied from self to new_model
        return rebuild_model_preserving_state(
            base_model=self,
            sim_config=self.sim_config,
            layers_config=new_layers_cfg,
            connections_config=new_connections_cfg,
            preserve_mode="all",
        )

    def prune_layers(self, remove_layers: list[int]) -> "SOENModelCore":
        """Create a new model with specific layers removed.

        Convenience wrapper around subgraph().

        Args:
            remove_layers: List of layer IDs to exclude.

        Returns:
            A new, independent SOENModelCore instance.
        """
        all_ids = {cfg.layer_id for cfg in self.layers_config}
        remove_set = set(remove_layers)
        keep_ids = sorted(list(all_ids - remove_set))
        return self.subgraph(keep_layers=keep_ids)

    def set_layer_description(self, layer_id: int, description: str) -> None:
        """Update a layer's description in-place (live metadata update).

        Updates both the configuration and the instantiated layer module.
        Does not require a model rebuild.

        Args:
            layer_id: ID of the layer to update.
            description: New description string.

        Raises:
            ValueError: If layer_id is not found.
        """
        # Update config
        found = False
        for cfg in self.layers_config:
            if cfg.layer_id == layer_id:
                cfg.description = description
                found = True
                break

        if not found:
            raise ValueError(f"Layer ID {layer_id} not found in model.")

        # Update live module if possible
        # Need to find the index of this layer in self.layers
        # We can rely on the sorted order property of layers_config matching layers
        # (enforced in __init__ via sorted(layers_config...))
        # Double check via index mapping

        # Efficiently find index
        # self.layers is nn.ModuleList, parallel to self.layers_config (sorted by ID)
        # So we can find the index in layers_config and apply to layers
        idx = -1
        for i, cfg in enumerate(self.layers_config):
            if cfg.layer_id == layer_id:
                idx = i
                break

        if idx != -1 and idx < len(self.layers):
            layer_module = cast(Any, self.layers[idx])
            layer_module.description = description
            # Also update internal dict if it exists?
            # Some layers might store it elsewhere, but attribute set is standard.

    # ----------------------------------------------------------------------------
    # Post‑training weight quantisation (matches robustness tool algorithms)
    # ----------------------------------------------------------------------------
