# FILEPATH: src/soen_toolkit/core/mixins/builders.py

from __future__ import annotations

from functools import lru_cache
import inspect
from typing import TYPE_CHECKING, Any, cast
import warnings

import torch
from torch import nn

from soen_toolkit.core.layer_registry import LAYER_TYPE_MAP
from soen_toolkit.core.layers.common.connectivity_metadata import build_weight
from soen_toolkit.core.layers.common.parameters import Constraint
from soen_toolkit.core.noise import NoiseSettings, build_noise_strategies
from soen_toolkit.core.utils.connection_ops import build_edge_index
from soen_toolkit.utils.polarity_utils import (
    POLARITY_ENFORCEMENT_DEFAULT,
    apply_polarity_enforcement,
)

if TYPE_CHECKING:
    from soen_toolkit.core.configs import ConnectionConfig, LayerConfig, SimulationConfig


class BuildersMixin:
    """Creation helpers: dimensions, connections, fan-outs, layers, learnability."""

    if TYPE_CHECKING:
        layers_config: list[LayerConfig]
        connections_config: list[ConnectionConfig]
        sim_config: SimulationConfig
        layer_nodes: dict[int, int]
        connection_constraint_min_matrices: dict[str, torch.Tensor]
        connection_constraint_max_matrices: dict[str, torch.Tensor]
        dt: float | nn.Parameter
        layers: nn.ModuleList
        connections: nn.ParameterDict
        connection_masks: dict[str, torch.Tensor]
        connection_constraints: dict[str, Any]
        _connection_modes: dict[str, str]
        _connection_params: dict[str, dict]
        _connection_edge_maps: dict[str, tuple[torch.Tensor, torch.Tensor]]
        _phi_history: list[torch.Tensor]
        latest_all_states: list[torch.Tensor] | None
        num_layers: int

    def _using_v2_layers(self) -> bool:
        # Deprecated method: the toolkit now always uses the unified layer implementation.
        # Kept for backward compatibility in case external code references it.
        return True

    def _apply_layer_params(self, layer: nn.Module, config: dict) -> None:
        if not hasattr(layer, "_param_registry"):
            return

        registry = getattr(layer, "_param_registry", None)
        if registry is None:
            return

        defs = getattr(registry, "_defs", {})
        getattr(registry, "_attr_names", {})

        for key, spec in config.items():
            skip_keys = {
                "dim",
                "solver",
                "source_func",
                "source_func_type",
                "description",
                "node_fan_outs",
                "learnable_params",
                "connectivity",
                "connectivity_spec",
                "connectivity_constraints",
                "learnable_connectivity",
                "internal_J",
                "connectivity_mode",
                "connectivity_multiplier",
            }
            if key in skip_keys:
                continue

            if key not in defs:
                continue

            override_method: str | None = None
            override_params: dict[str, object] = {}
            override_value: float | None = None
            learnable_flag = None
            constraint_obj = None

            if isinstance(spec, dict):
                if "distribution" in spec:
                    override_method = spec.get("distribution")
                    override_params = dict(spec.get("params", {}))
                elif "value" in spec:
                    override_method = "constant"
                    override_params = {"value": spec.get("value")}

                if "value" in spec and override_method is None:
                    override_value = float(spec["value"])

                if "learnable" in spec:
                    learnable_flag = spec.get("learnable")

                cons = spec.get("constraints")
                if isinstance(cons, dict):
                    constraint_obj = Constraint(
                        min=cons.get("min"),
                        max=cons.get("max"),
                    )
            elif spec is not None:
                override_method = "constant"
                override_value = float(spec)
                override_params = {"value": override_value}

            registry.override_parameter(
                key,
                method=override_method,
                params=override_params if override_params else None,
                value=override_value,
                learnable=learnable_flag,
                constraint=constraint_obj,
            )

    def _collect_layer_dimensions(self) -> dict[int, int]:
        return {cfg.layer_id: cfg.params["dim"] for cfg in self.layers_config}

    def _build_connections(self) -> tuple[nn.ParameterDict, dict, dict]:
        connections = nn.ParameterDict()
        connection_masks = {}
        connection_constraints = {}
        internal_specs: dict[int, dict] = {}
        # Per-connection mode, parameters, and edge maps
        self._connection_modes: dict[str, str] = {}
        self._connection_params: dict[str, dict] = {}
        self._connection_edge_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Precompute for O(1) layer lookup
        layer_cfg_map = {cfg.layer_id: cfg for cfg in self.layers_config}

        # Pre-compute number of input sources per layer for flux_balanced initialization
        # This counts how many connections feed INTO each layer (external + internal + feedback)
        input_sources_per_layer: dict[int, int] = {cfg.layer_id: 0 for cfg in self.layers_config}
        for conn in self.connections_config:
            input_sources_per_layer[conn.to_layer] = input_sources_per_layer.get(conn.to_layer, 0) + 1

        for conn in self.connections_config:
            if (conn.from_layer not in self.layer_nodes) or (conn.to_layer not in self.layer_nodes):
                msg = f"Connection refers to undefined layer(s): {conn.from_layer} -> {conn.to_layer}"
                raise ValueError(
                    msg,
                )

            from_nodes = self.layer_nodes[conn.from_layer]
            to_nodes = self.layer_nodes[conn.to_layer]
            # Make a shallow copy so we can annotate with connection scope
            original_params = dict(conn.params or {})
            raw_params = original_params.copy()
            constraints_override = raw_params.pop("constraints", None)
            learnable_override = raw_params.pop("learnable", None)
            raw_params.pop("J", None)
            structure_info = raw_params.pop("structure", None)
            init_info = raw_params.pop("init", None)
            allow_self = raw_params.pop("allow_self_connections", True)

            if isinstance(structure_info, dict):
                effective_conn_type = structure_info.get("type", conn.connection_type)
                struct_params = dict(structure_info.get("params", {}) or {})
            else:
                effective_conn_type = conn.connection_type
                struct_params = raw_params.copy()

            if isinstance(init_info, dict):
                init_name = init_info.get("name", "normal")
                init_params = dict(init_info.get("params", {}) or {})
            else:
                init_name = init_info if isinstance(init_info, str) else original_params.get("init", "normal")
                init_params = {}

            if init_name == "uniform":
                if "a" in init_params and "min" not in init_params:
                    init_params["min"] = init_params.pop("a")
                if "b" in init_params and "max" not in init_params:
                    init_params["max"] = init_params.pop("b")

            if structure_info is not None:
                params = struct_params.copy()
                params["allow_self_connections"] = allow_self
                params["init"] = init_name
                params.update(init_params)
            else:
                params = original_params.copy()
                params.pop("constraints", None)
                params.pop("learnable", None)
                params.setdefault("allow_self_connections", allow_self)

            # Auto-inject num_input_sources for flux_balanced initialization
            if init_name == "flux_balanced":
                num_sources = input_sources_per_layer.get(conn.to_layer, 1)
                params.setdefault("num_input_sources", max(1, num_sources))

            # Extract connection mode and params using unified parser
            from soen_toolkit.core.utils.connection_ops import parse_connection_config

            conn_mode, conn_params = parse_connection_config(original_params)

            # Annotate scope for connectivity builders that need to distinguish
            # intra-layer vs inter-layer geometry (e.g., power_law)
            if conn.from_layer == conn.to_layer:
                params.setdefault("connection_scope", "internal")
                struct_params.setdefault("connection_scope", "internal")
            else:
                params.setdefault("connection_scope", "external")
                struct_params.setdefault("connection_scope", "external")

            # Internal connections are handled inside the layer for certain layer types
            layer_type = layer_cfg_map[conn.from_layer].layer_type

            # Note: Intra-layer connections are treated like any other connection
            # and added to model.connections with key J_{layer}_to_{layer}

            # Auto-populate visualization_metadata for hierarchical_blocks
            if effective_conn_type == "hierarchical_blocks" and conn.from_layer == conn.to_layer:
                levels = struct_params.get("levels")
                base_size = struct_params.get("base_size")
                if levels is not None and base_size is not None:
                    # Initialize visualization_metadata if it doesn't exist
                    if conn.visualization_metadata is None:
                        conn.visualization_metadata = {}
                    # Add hierarchical structure info if not already present
                    if "hierarchical" not in conn.visualization_metadata:
                        conn.visualization_metadata["hierarchical"] = {
                            "levels": int(levels),
                            "base_size": int(base_size),
                        }

            weight, mask = build_weight(effective_conn_type, from_nodes, to_nodes, dict(params))

            if conn.from_layer == conn.to_layer:
                layer_cfg_map[conn.from_layer]
                key = f"J_{conn.from_layer}_to_{conn.to_layer}"
            else:
                solver_mode = str(getattr(getattr(self, "sim_config", object()), "network_evaluation_method", "layerwise")).lower()
                if solver_mode not in {"layerwise", "stepwise_gauss_seidel", "stepwise_jacobi"}:
                    solver_mode = "layerwise"
                allow_backward = solver_mode in {"stepwise_gauss_seidel", "stepwise_jacobi"}
                if (conn.to_layer <= conn.from_layer) and (not allow_backward):
                    old_mode = getattr(self.sim_config, "network_evaluation_method", "layerwise")
                    try:
                        self.sim_config.network_evaluation_method = "stepwise_gauss_seidel"
                    except Exception as e:
                        msg = f"Failed to set network_evaluation_method to 'stepwise_gauss_seidel' for backward connection {conn.from_layer}->{conn.to_layer}. Error: {e}"
                        raise RuntimeError(msg) from e
                    warnings.warn(
                        (
                            f"Detected backward/feedback connection {conn.from_layer}->{conn.to_layer} with network_evaluation_method='{old_mode}'. "
                            "Switching to network_evaluation_method='stepwise_gauss_seidel' automatically (feedback requires a stepwise solver)."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                key = f"J_{conn.from_layer}_to_{conn.to_layer}"

            # Store internal connectivity specs for layers that need them
            if conn.from_layer == conn.to_layer and layer_type in LAYER_TYPE_MAP:
                internal_specs[conn.from_layer] = {
                    "matrix": weight,
                    "mask": mask,
                    "spec": {"type": effective_conn_type, "params": struct_params},
                    "learnable": bool(conn.learnable if learnable_override is None else learnable_override),
                    "constraints": constraints_override,
                    "mode": conn_mode,
                    "dynamic_params": conn_params,
                }

                # Build polarity constraint matrices for internal (recurrent) connections
                source_layer_cfg_internal = layer_cfg_map.get(conn.from_layer)
                if source_layer_cfg_internal:
                    polarity_file_internal = source_layer_cfg_internal.params.get("polarity_file")
                    polarity_init_internal = source_layer_cfg_internal.params.get("polarity_init")
                    polarity_internal = None

                    # Check incompatibility with dynamic weights
                    if (polarity_file_internal or polarity_init_internal) and conn_mode in {"WICC", "NOCC"}:
                        msg = f"Neuron polarity constraints incompatible with dynamic weights (internal connection {key} uses {conn_mode} mode)"
                        raise ValueError(msg)

                    if polarity_file_internal:
                        from soen_toolkit.core.layers.common.connectivity_metadata import load_neuron_polarity
                        polarity_internal = load_neuron_polarity(polarity_file_internal)
                    elif polarity_init_internal:
                        from soen_toolkit.utils.polarity_utils import (
                            generate_alternating_polarity,
                            generate_excitatory_polarity,
                            generate_inhibitory_polarity,
                            generate_random_polarity,
                        )

                        if isinstance(polarity_init_internal, dict):
                            excitatory_ratio = polarity_init_internal.get("excitatory_ratio", 0.8)
                            seed = polarity_init_internal.get("seed")
                            polarity_internal = torch.as_tensor(generate_random_polarity(from_nodes, excitatory_ratio=excitatory_ratio, seed=seed), dtype=torch.int8)
                        elif polarity_init_internal in {"alternating", "50_50"}:
                            polarity_internal = torch.as_tensor(generate_alternating_polarity(from_nodes), dtype=torch.int8)
                        elif polarity_init_internal == "excitatory":
                            polarity_internal = torch.as_tensor(generate_excitatory_polarity(from_nodes), dtype=torch.int8)
                        elif polarity_init_internal == "inhibitory":
                            polarity_internal = torch.as_tensor(generate_inhibitory_polarity(from_nodes), dtype=torch.int8)
                        else:
                            msg = f"Unknown polarity_init value: {polarity_init_internal}"
                            raise ValueError(msg)

                    if polarity_internal is not None and len(polarity_internal) == from_nodes:
                        # Apply polarity enforcement to initial weights
                        enforcement_method = source_layer_cfg_internal.params.get(
                            "polarity_enforcement_method", POLARITY_ENFORCEMENT_DEFAULT
                        )
                        weight = apply_polarity_enforcement(weight, polarity_internal, enforcement_method)
                        # Update the internal_specs matrix with enforced weights
                        internal_specs[conn.from_layer]["matrix"] = weight

                        # Build per-element constraint matrices
                        min_mat_internal = torch.full((to_nodes, from_nodes), -float('inf'), dtype=weight.dtype, device=weight.device)
                        max_mat_internal = torch.full((to_nodes, from_nodes), float('inf'), dtype=weight.dtype, device=weight.device)

                        for src in range(from_nodes):
                            if polarity_internal[src] == 1:  # Excitatory
                                min_mat_internal[:, src] = 0.0
                            elif polarity_internal[src] == -1:  # Inhibitory
                                max_mat_internal[:, src] = 0.0

                        # Merge with scalar constraints
                        if constraints_override:
                            scalar_min_internal = constraints_override.get("min", -float("inf"))
                            scalar_max_internal = constraints_override.get("max", float("inf"))
                            min_mat_internal = torch.maximum(min_mat_internal, torch.tensor(scalar_min_internal, dtype=min_mat_internal.dtype, device=min_mat_internal.device))
                            max_mat_internal = torch.minimum(max_mat_internal, torch.tensor(scalar_max_internal, dtype=max_mat_internal.dtype, device=max_mat_internal.device))

                        # Store internal connection constraint matrices
                        self.connection_constraint_min_matrices[key] = min_mat_internal
                        self.connection_constraint_max_matrices[key] = max_mat_internal

            if key in connections:
                msg = f"Multiple connection definitions for the same mapping '{key}'. The current implementation supports only one connection between a given pair of layers."
                raise ValueError(
                    msg,
                )

            # Weight and mask already built above, no need to rebuild
            is_learnable = conn.learnable if learnable_override is None else bool(learnable_override)

            connections[key] = nn.Parameter(weight, requires_grad=is_learnable)
            connection_masks[key] = mask

            if constraints_override is not None:
                connection_constraints[key] = constraints_override or {}
            else:
                connection_constraints[key] = raw_params.get("J", {"min": -float("inf"), "max": float("inf")})

            # Build per-edge constraint matrices from SOURCE LAYER neuron polarity
            source_layer_cfg = layer_cfg_map.get(conn.from_layer)
            polarity = None

            if source_layer_cfg:
                # Check params['polarity'] first (explicit list/array)
                polarity_explicit = source_layer_cfg.params.get("polarity")
                polarity_file = source_layer_cfg.params.get("polarity_file")
                polarity_init = source_layer_cfg.params.get("polarity_init")

                # Check incompatibility with dynamic weights first
                if (polarity_explicit is not None or polarity_file or polarity_init) and conn_mode in {"WICC", "NOCC"}:
                    msg = f"Neuron polarity constraints incompatible with dynamic weights (connection {key} uses {conn_mode} mode)"
                    raise ValueError(msg)

                if polarity_explicit is not None:
                    # Handle list or numpy array
                    if isinstance(polarity_explicit, list):
                        polarity = torch.tensor(polarity_explicit, dtype=torch.int8)
                    elif isinstance(polarity_explicit, torch.Tensor):
                        polarity = polarity_explicit.to(dtype=torch.int8)
                    else:
                        # Assume numpy or similar iterable
                        polarity = torch.as_tensor(polarity_explicit, dtype=torch.int8)
                elif polarity_file:
                    from soen_toolkit.core.layers.common.connectivity_metadata import load_neuron_polarity
                    polarity = load_neuron_polarity(polarity_file)
                elif polarity_init:
                    # Generate polarity on the fly
                    from soen_toolkit.utils.polarity_utils import (
                        generate_alternating_polarity,
                        generate_excitatory_polarity,
                        generate_inhibitory_polarity,
                        generate_random_polarity,
                    )

                    if isinstance(polarity_init, dict):
                        # Allow dict with excitatory_ratio and optional seed
                        excitatory_ratio = polarity_init.get("excitatory_ratio", 0.8)
                        seed = polarity_init.get("seed")
                        polarity = torch.as_tensor(generate_random_polarity(from_nodes, excitatory_ratio=excitatory_ratio, seed=seed), dtype=torch.int8)
                    elif polarity_init in {"alternating", "50_50"}:
                        polarity = torch.as_tensor(generate_alternating_polarity(from_nodes), dtype=torch.int8)
                    elif polarity_init == "excitatory":
                        polarity = torch.as_tensor(generate_excitatory_polarity(from_nodes), dtype=torch.int8)
                    elif polarity_init == "inhibitory":
                        polarity = torch.as_tensor(generate_inhibitory_polarity(from_nodes), dtype=torch.int8)
                    else:
                        msg = f"Unknown polarity_init value: {polarity_init}. Use 'alternating', '50_50', 'excitatory', 'inhibitory', or a dict with 'excitatory_ratio'"
                        raise ValueError(msg)

                if polarity is not None:
                    if len(polarity) != from_nodes:
                        msg = f"Polarity length {len(polarity)} for layer {conn.from_layer} != from_nodes {from_nodes}"
                        raise ValueError(msg)

                    # Apply polarity enforcement to initial weights
                    enforcement_method = source_layer_cfg.params.get(
                        "polarity_enforcement_method", POLARITY_ENFORCEMENT_DEFAULT
                    )
                    weight = apply_polarity_enforcement(weight, polarity, enforcement_method)
                    # Update the connection parameter with enforced weights
                    connections[key] = nn.Parameter(weight, requires_grad=is_learnable)

                    # Initialize with inf
                    min_mat = torch.full((to_nodes, from_nodes), -float('inf'), dtype=weight.dtype, device=weight.device)
                    max_mat = torch.full((to_nodes, from_nodes), float('inf'), dtype=weight.dtype, device=weight.device)

                    # Apply polarity constraints column-wise (per source neuron)
                    for src in range(from_nodes):
                        if polarity[src] == 1:  # Excitatory
                            min_mat[:, src] = 0.0
                        elif polarity[src] == -1:  # Inhibitory
                            max_mat[:, src] = 0.0
                        # polarity[src] == 0: no change (unrestricted)

                    # Merge with scalar constraints
                    scalar_min = constraints_override.get("min", -float("inf")) if constraints_override else -float("inf")
                    scalar_max = constraints_override.get("max", float("inf")) if constraints_override else float("inf")
                    min_mat = torch.maximum(min_mat, torch.tensor(scalar_min, dtype=min_mat.dtype, device=min_mat.device))
                    max_mat = torch.minimum(max_mat, torch.tensor(scalar_max, dtype=max_mat.dtype, device=max_mat.device))

                    # Store in model-level dicts (initialized in __init__)
                    self.connection_constraint_min_matrices[key] = min_mat
                    self.connection_constraint_max_matrices[key] = max_mat

            # Record connection mode, parameters, and edge indices
            self._connection_modes[key] = conn_mode
            self._connection_params[key] = conn_params

            try:
                src_idx, dst_idx = build_edge_index(mask, weight)
            except Exception:
                # Fallback: assume dense
                src_idx, dst_idx = build_edge_index(None, weight)
            self._connection_edge_maps[key] = (src_idx, dst_idx)

            # Auto-compute WICC j_out per-destination using fan-in formula (hidden from user)
            # J_out = (2.5e-10) / ((40e-12) * N + 3.35e-10), where N is fan-in of destination node
            try:
                if self._connection_modes.get(key) == "WICC":
                    # Respect explicit user-provided j_out; only auto-compute when not provided
                    dyn_block = original_params.get("connection_params") or original_params.get("dynamic") or original_params.get("multiplier") or {}
                    if isinstance(dyn_block, dict) and ("j_out" in dyn_block):
                        # User specified j_out; do not override
                        pass
                    else:
                        # Compute fan-in per destination from mask when available, otherwise from weight
                        if mask is not None:
                            fan_in_vec = (mask > 0).sum(dim=1).to(torch.float32)
                        else:
                            fan_in_vec = (weight != 0).sum(dim=1).to(torch.float32)

                        # Constants in SI units
                        num = torch.as_tensor(2.5e-10, dtype=torch.float32, device=weight.device)
                        coeff = torch.as_tensor(40e-12, dtype=torch.float32, device=weight.device)
                        offset = torch.as_tensor(3.35e-10, dtype=torch.float32, device=weight.device)

                        j_out_vec = num / (coeff * fan_in_vec + offset)  # shape [D]
                        # Store computed vector; runtime ops will broadcast correctly
                        self._connection_params[key]["j_out"] = j_out_vec
            except Exception:
                # Fail-safe: do not block model build if computation fails; leave params as-is
                # come back and check this, it might be better to return an error. Why would this calc ever fail? If it does this is a sign of a deeper issue.
                # In general, we want to avoid silent failures like this.
                pass

        # Do not create legacy aliases like internal_<id>. Only keep unified keys J_<id>_to_<id>.

        self._internal_connectivity_settings = internal_specs
        return connections, connection_masks, connection_constraints

    def _build_connection_noise_settings(self) -> dict[str, NoiseSettings]:
        settings = {}
        for conn in self.connections_config:
            key = f"J_{conn.from_layer}_to_{conn.to_layer}"
            settings[key] = build_noise_strategies(conn.noise, conn.perturb)
        return settings

    def _calculate_fan_outs(self) -> dict[int, list[int]]:
        node_fan_outs = {layer_id: [0] * dim for layer_id, dim in self.layer_nodes.items()}

        for key, mask in self.connection_masks.items():
            # Vectorized fan-out: mask shape [to_dim, from_dim]
            # Count outgoing connections per source node (column-wise)
            counts = (mask > 0).sum(dim=0).to(torch.int64).tolist()

            if key.startswith("J_"):
                # Key format: J_<from>_to_<to>
                try:
                    from_layer = int(key.split("_")[1])
                except (ValueError, IndexError) as e:
                    msg = f"Malformed connection key '{key}': expected format 'J_<from>_to_<to>'. Error: {e}"
                    raise ValueError(msg) from e
                # Accumulate
                target_list = node_fan_outs.get(from_layer)
                if target_list is not None:
                    # In case shapes mismatch, accumulate up to min length
                    upto = min(len(target_list), len(counts))
                    for i in range(upto):
                        target_list[i] += counts[i]
            elif key.startswith("internal_"):
                # Legacy format: internal_<layer> (should not appear in new code but handle for backward compat)
                try:
                    layer_id = int(key.split("_")[1])
                except (ValueError, IndexError) as e:
                    msg = f"Malformed connection key '{key}': expected format 'internal_<layer>'. Error: {e}"
                    raise ValueError(msg) from e
                target_list = node_fan_outs.get(layer_id)
                if target_list is not None:
                    upto = min(len(target_list), len(counts))
                    for i in range(upto):
                        target_list[i] += counts[i]

        # Replace zeros with ones to avoid zero fan-out
        for layer_id, fan_outs in node_fan_outs.items():
            node_fan_outs[layer_id] = [v if v != 0 else 1 for v in fan_outs]

        return node_fan_outs

    def _create_layers(self, node_fan_outs: dict[int, list[int]]) -> None:
        # Get layer registry
        registry = LAYER_TYPE_MAP

        internal_specs = getattr(self, "_internal_connectivity_settings", {})

        for cfg in self.layers_config:
            layer_cls = registry.get(cfg.layer_type)
            if layer_cls is None:
                msg = f"Unknown layer type: {cfg.layer_type}"
                raise ValueError(msg)

            if bool(getattr(layer_cls, "_is_placeholder", False)):
                reason = str(getattr(layer_cls, "_placeholder_reason", "")).strip()
                reason_msg = f" Reason: {reason}" if reason else ""
                raise NotImplementedError(
                    f"Layer type '{cfg.layer_type}' is registered but not supported yet (placeholder).{reason_msg} "
                    "Remove this layer from the model spec or implement it before building."
                )

            layer_params = cfg.params.copy()
            dim_value = layer_params.get("dim", 1)

            for param_key, param_value in layer_params.items():
                if param_key.startswith("gamma_plus") and isinstance(param_value, dict) and param_value.get("distribution") == "fan_out":
                    param_value.setdefault("params", {})
                    if cfg.layer_id in node_fan_outs:
                        param_value["params"]["node_fan_outs"] = node_fan_outs[cfg.layer_id]

            @lru_cache(maxsize=128)
            def _param_names(cls):
                return frozenset(inspect.signature(cls.__init__).parameters.keys())

            param_names = _param_names(layer_cls)
            dt_value = self.dt if isinstance(self.dt, torch.Tensor) else float(self.dt)

            # Only enable power tracking for SingleDendrite layers
            enable_power_tracking = getattr(self.sim_config, "track_power", False) and cfg.layer_type == "SingleDendrite"
            track_flags = {
                "track_power": enable_power_tracking,
                "track_phi": getattr(self.sim_config, "track_phi", False),
                "track_g": getattr(self.sim_config, "track_g", False),
                "track_s": getattr(self.sim_config, "track_s", False),
            }

            kwargs = {
                "dim": dim_value,
                "dt": dt_value,
            }

            if "solver" in param_names:
                kwargs["solver"] = cfg.params.get("solver", "FE")
            if "source_func_type" in param_names:
                if cfg.layer_type in {"Linear", "Input"}:
                    kwargs["source_func_type"] = None
                elif "source_func" in cfg.params:
                    kwargs["source_func_type"] = cfg.params["source_func"]

            # Targeted constructor-kwargs passthrough for virtual layers that are configured
            # via cfg.params (and do not use the ParameterRegistry).
            if cfg.layer_type in {"LeakyGRU", "leakyGRU"}:
                for key in (
                    "reset_bias",
                    "tau_init",
                    "tau_spacing",
                    "tau_eps",
                    "candidate_diag",
                    "train_alpha",
                ):
                    if key in cfg.params:
                        kwargs[key] = cfg.params[key]

            if "internal_J" in layer_params and "connectivity" in param_names:
                internal_tensor = layer_params.get("internal_J")
                kwargs.setdefault("connectivity", internal_tensor)

            internal_info = internal_specs.get(cfg.layer_id)
            if internal_info:
                if "connectivity" in param_names and internal_info.get("matrix") is not None:
                    kwargs.setdefault("connectivity", internal_info["matrix"])
                if "connectivity_spec" in param_names and internal_info.get("spec") is not None:
                    kwargs.setdefault("connectivity_spec", internal_info["spec"])
                if "learnable_connectivity" in param_names and internal_info.get("learnable") is not None:
                    kwargs.setdefault("learnable_connectivity", internal_info["learnable"])
                if "connectivity_constraints" in param_names and internal_info.get("constraints") is not None:
                    kwargs.setdefault("connectivity_constraints", internal_info["constraints"])
            # Pass mode and dynamic params from internal_specs (not cfg.params)
            if internal_info:
                if "connectivity_mode" in param_names and internal_info.get("mode") is not None:
                    kwargs.setdefault("connectivity_mode", internal_info["mode"])
                # Use new API name (connectivity_params) and fallback to old name for backward compat
                if "connectivity_params" in param_names and internal_info.get("dynamic_params") is not None:
                    kwargs.setdefault("connectivity_params", internal_info["dynamic_params"])
                elif "connectivity_dynamic" in param_names and internal_info.get("dynamic_params") is not None:
                    kwargs.setdefault("connectivity_dynamic", internal_info["dynamic_params"])

            layer = layer_cls(**kwargs)

            # Post-construction handling
            layer.layer_type = cfg.layer_type
            if hasattr(layer, "feature_stack"):
                layer.feature_stack.attach_layer(layer)

            for flag, value in track_flags.items():
                setattr(layer, flag, value)
            if hasattr(layer, "set_tracking_flags"):
                layer.set_tracking_flags(
                    phi=track_flags["track_phi"],
                    g=track_flags["track_g"],
                    s=track_flags["track_s"],
                    power=track_flags["track_power"],
                )
            self._apply_layer_params(layer, layer_params)

            if getattr(layer, "connectivity", None) is not None:
                key = f"J_{cfg.layer_id}_to_{cfg.layer_id}"
                if key not in self.connections:
                    self.connections[key] = layer.connectivity.weight
                info = internal_specs.get(cfg.layer_id, {})
                if key not in self.connection_masks and info.get("mask") is not None:
                    self.connection_masks[key] = info["mask"]
                if key not in self.connection_constraints and info.get("constraints") is not None:
                    self.connection_constraints[key] = info["constraints"] or {}

                # Build polarity constraint matrices for layer-internal connectivity
                if key not in self.connection_constraint_min_matrices:
                    polarity_explicit_layer = cfg.params.get("polarity")
                    polarity_file_layer = cfg.params.get("polarity_file")
                    polarity_init_layer = cfg.params.get("polarity_init")
                    polarity_layer = None

                    if polarity_explicit_layer is not None:
                        if isinstance(polarity_explicit_layer, list):
                            polarity_layer = torch.tensor(polarity_explicit_layer, dtype=torch.int8)
                        elif isinstance(polarity_explicit_layer, torch.Tensor):
                            polarity_layer = polarity_explicit_layer.to(dtype=torch.int8)
                        else:
                            polarity_layer = torch.as_tensor(polarity_explicit_layer, dtype=torch.int8)
                    elif polarity_file_layer:
                        from soen_toolkit.core.layers.common.connectivity_metadata import load_neuron_polarity
                        polarity_layer = load_neuron_polarity(polarity_file_layer)
                    elif polarity_init_layer:
                        from soen_toolkit.utils.polarity_utils import (
                            generate_alternating_polarity,
                            generate_excitatory_polarity,
                            generate_inhibitory_polarity,
                            generate_random_polarity,
                        )

                        dim = cfg.params.get("dim", dim_value)
                        if isinstance(polarity_init_layer, dict):
                            excitatory_ratio = polarity_init_layer.get("excitatory_ratio", 0.8)
                            seed = polarity_init_layer.get("seed")
                            polarity_layer = torch.as_tensor(generate_random_polarity(dim, excitatory_ratio=excitatory_ratio, seed=seed), dtype=torch.int8)
                        elif polarity_init_layer in {"alternating", "50_50"}:
                            polarity_layer = torch.as_tensor(generate_alternating_polarity(dim), dtype=torch.int8)
                        elif polarity_init_layer == "excitatory":
                            polarity_layer = torch.as_tensor(generate_excitatory_polarity(dim), dtype=torch.int8)
                        elif polarity_init_layer == "inhibitory":
                            polarity_layer = torch.as_tensor(generate_inhibitory_polarity(dim), dtype=torch.int8)
                        else:
                            msg = f"Unknown polarity_init value: {polarity_init_layer}"
                            raise ValueError(msg)

                    if polarity_layer is not None:
                        weight = layer.connectivity.weight
                        dim = weight.shape[0]

                        if len(polarity_layer) == dim:
                            # Apply polarity enforcement to initial weights
                            enforcement_method = cfg.params.get(
                                "polarity_enforcement_method", POLARITY_ENFORCEMENT_DEFAULT
                            )
                            enforced_weight = apply_polarity_enforcement(weight, polarity_layer, enforcement_method)
                            # Update the layer's connectivity weight with enforced weights
                            with torch.no_grad():
                                layer.connectivity.weight.copy_(enforced_weight)

                            # Build per-element constraint matrices
                            min_mat_layer = torch.full((dim, dim), -float('inf'), dtype=weight.dtype, device=weight.device)
                            max_mat_layer = torch.full((dim, dim), float('inf'), dtype=weight.dtype, device=weight.device)

                            for src in range(dim):
                                if polarity_layer[src] == 1:  # Excitatory
                                    min_mat_layer[:, src] = 0.0
                                elif polarity_layer[src] == -1:  # Inhibitory
                                    max_mat_layer[:, src] = 0.0

                            # Merge with scalar constraints if present
                            constraints_layer = info.get("constraints")
                            if constraints_layer:
                                scalar_min_layer = constraints_layer.get("min", -float("inf"))
                                scalar_max_layer = constraints_layer.get("max", float("inf"))
                                min_mat_layer = torch.maximum(min_mat_layer, torch.tensor(scalar_min_layer, dtype=min_mat_layer.dtype, device=min_mat_layer.device))
                                max_mat_layer = torch.minimum(max_mat_layer, torch.tensor(scalar_max_layer, dtype=max_mat_layer.dtype, device=max_mat_layer.device))

                            # Store layer-internal connection constraint matrices
                            self.connection_constraint_min_matrices[key] = min_mat_layer
                            self.connection_constraint_max_matrices[key] = max_mat_layer

            if (not hasattr(layer, "solver")) and ("solver" in param_names):
                layer.solver = str(cfg.params.get("solver", "FE")).upper()

            self.layers.append(layer)

            self._set_parameter_learnability(
                layer=layer,
                layer_type=cfg.layer_type,
                learnable_dict=cfg.params.get("learnable_params", {}),
            )

        for idx, cfg in enumerate(self.layers_config):
            new_key = f"J_{cfg.layer_id}_to_{cfg.layer_id}"
            old_key = f"internal_{cfg.layer_id}"
            if old_key in self.connections and new_key not in self.connections:
                self.connections[new_key] = self.connections[old_key]
                if old_key in self.connection_masks and new_key not in self.connection_masks:
                    self.connection_masks[new_key] = self.connection_masks[old_key]
                if old_key in self.connection_constraints and new_key not in self.connection_constraints:
                    self.connection_constraints[new_key] = self.connection_constraints[old_key]

            if new_key in self.connections:
                layer_obj = self.layers[idx]
                if getattr(layer_obj, "connectivity", None) is not None:
                    l_conn = cast(Any, getattr(layer_obj, "connectivity"))
                    self.connections[new_key] = l_conn.weight
                    if new_key in self.connection_constraints:
                        l_conn.constraints = self.connection_constraints[new_key]
                    continue
                if hasattr(layer_obj, "internal_J"):
                    l_any = cast(Any, layer_obj)
                    l_any.internal_J = self.connections[new_key]
                    if new_key in self.connection_constraints and hasattr(
                        l_any,
                        "internal_J_constraints",
                    ):
                        l_any.internal_J_constraints = self.connection_constraints[new_key]
                    self._set_parameter_learnability(
                        layer=layer_obj,
                        layer_type=cfg.layer_type,
                        learnable_dict=cfg.params.get("learnable_params", {}),
                    )

        # -------------------------------
        # Validation: PS compatibility
        # -------------------------------
        for idx, cfg in enumerate(self.layers_config):
            layer = self.layers[idx]
            solver = str(cfg.params.get("solver", getattr(layer, "solver", "FE"))).upper()
            if solver != "PS":
                continue

            # Disallow PS if internal connectivity is present (φ depends on s)
            if getattr(layer, "internal_J", None) is not None:
                msg = f"Layer {cfg.layer_id} ({cfg.layer_type}) uses solver='PS' but has an internal connection. Parallel scan requires φ independent of s."
                raise ValueError(
                    msg,
                )

            # Disallow PS if the source function doesn't support coefficients
            sf = getattr(layer, "source_function", None)
            if sf is not None and not getattr(sf.info, "supports_coefficients", True):
                msg = f"Layer {cfg.layer_id} ({cfg.layer_type}) uses solver='PS' with source function '{sf.info.key}' that does not support parallel scan coefficients."
                raise ValueError(
                    msg,
                )

    def _set_parameter_learnability(
        self,
        layer: nn.Module,
        layer_type: str,
        learnable_dict: dict,
        show_logs: bool = False,
    ) -> None:
        # Modern layers use ParameterRegistry for parameter management.
        # Check dynamically if the layer supports it.
        registry = getattr(layer, "_param_registry", None)

        if registry is not None:
            # These layers use ParameterRegistry for declaring parameters, which may expose
            # log-domain tensors (e.g. ``log_gamma_plus``). Respect the catalog key provided
            # via ``learnable_params`` by normalizing parameter names before lookup.
            attr_names = getattr(registry, "_attr_names", {})
            attr_to_base = {attr: base for base, attr in attr_names.items()}

            # Build set of valid param names for validation
            valid_param_names: set[str] = set()
            matched_learnable_keys: set[str] = set()

            for param_name, param_obj in layer.named_parameters():
                base_name = attr_to_base.get(param_name)
                if base_name is None and param_name.startswith("log_"):
                    # Fallback for transformed tensors that are not registered via the
                    # ParameterRegistry (kept for clarity).
                    base_name = param_name[len("log_") :]

                # Track valid param names (both base and actual)
                valid_param_names.add(param_name)
                if base_name is not None:
                    valid_param_names.add(base_name)

                lookup_order = []
                if base_name is not None:
                    lookup_order.append(base_name)
                lookup_order.append(param_name)

                is_learnable = True
                for key in lookup_order:
                    if key in learnable_dict:
                        is_learnable = bool(learnable_dict[key])
                        matched_learnable_keys.add(key)
                        break

                param_obj.requires_grad = is_learnable

            # Validate: all keys in learnable_dict should match actual params
            # Fail-fast on invalid keys to catch config errors early
            if learnable_dict:
                unmatched_keys = set(learnable_dict.keys()) - matched_learnable_keys
                if unmatched_keys:
                    msg = (
                        f"Invalid parameter names in learnable_params for layer type '{layer_type}': "
                        f"{sorted(unmatched_keys)}. "
                        f"Valid parameter names are: {sorted(valid_param_names)}. "
                        f"Check your YAML config for typos or wrong parameter names."
                    )
                    raise ValueError(msg)
            return

        # If we reach here, the layer doesn't have ParameterRegistry - this is a bug
        msg = (
            f"Layer type '{layer_type}' does not have ParameterRegistry (_param_registry attribute). "
            f"All layers should inherit from SoenLayerBase which provides ParameterRegistry. "
            f"This indicates an implementation error in the layer class."
        )
        raise RuntimeError(msg)
