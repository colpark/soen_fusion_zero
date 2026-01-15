from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import jax.numpy as jnp


@dataclass(eq=False)
class Topology:
    """Static network topology for parameter conversion."""

    dt: float
    num_layers: int
    layer_dims: tuple[int, ...]  # dimensions per layer
    layer_kinds: tuple[int, ...]  # int codes: 0=Multiplier, 1=SingleDendrite, 2=MinGRU, 3=Linear, 4=NonLinear, 5=Scaling, 6=GRU
    layer_source_keys: tuple[str, ...]  # source function names
    layer_ids: tuple[int, ...]  # layer ids in sorted order
    # Edges in CSR-like format
    edge_starts: tuple[int, ...]  # start index in edges for each layer's inbound
    edge_from_layers: tuple[int, ...]  # source layer indices
    edge_matrices: jnp.ndarray  # [E, max_dst_dim, max_src_dim] padded connection matrices
    edge_dst_dims: tuple[int, ...]  # actual dst dims for each edge
    edge_src_dims: tuple[int, ...]  # actual src dims for each edge
    edge_masks: jnp.ndarray | None = None  # [E, max_dst_dim, max_src_dim] padded masks or None
    network_evaluation_method: str = "layerwise"
    # Internal connectivity per layer (if any)
    internal_js: list[jnp.ndarray | None] = field(default_factory=list)  # [L] each can be None or [D,D] or [B,D,D]
    internal_masks: list[jnp.ndarray | None] = field(default_factory=list)  # [L] each can be None or [D,D] structural mask for internal connections
    internal_constraint_mins: list[jnp.ndarray | None] = field(default_factory=list)  # [L] per-element min constraints for internal connections
    internal_constraint_maxs: list[jnp.ndarray | None] = field(default_factory=list)  # [L] per-element max constraints for internal connections
    # Optional mapping for trainable connections (edge order alignment)
    edge_from_ids: tuple[int, ...] = ()
    edge_to_ids: tuple[int, ...] = ()
    # Input semantics ("flux" | "state")
    input_type: str = "flux"
    # Learnability tracking
    edge_learnable: tuple[bool, ...] = ()  # Learnability flag for each edge in edge order
    internal_learnable: dict[int, bool] | None = None  # Learnability flag for internal connections per layer_id
    connection_constraints: dict[str, dict[str, float]] | None = None  # Connection constraints propagated from Torch
    # Per-element constraint matrices for polarity enforcement (padded to match edge_matrices)
    edge_constraint_mins: jnp.ndarray | None = None  # [E, max_dst_dim, max_src_dim] per-element min constraints
    edge_constraint_maxs: jnp.ndarray | None = None  # [E, max_dst_dim, max_src_dim] per-element max constraints


# Layer kind codes
KIND_MULTIPLIER = 0
KIND_SINGLEDENDRITE = 1
KIND_MINGRU = 2
KIND_LINEAR = 3
KIND_NONLINEAR = 4
KIND_SCALING = 5
KIND_GRU = 6
KIND_MULTIPLIERV2 = 7
KIND_LSTM = 8


def build_topology(jax_model) -> Topology:
    """Convert JAXModel to static topology."""
    layers_sorted = sorted(jax_model.layers, key=lambda spec: spec.layer_id)
    L = len(layers_sorted)

    layer_dims = tuple(s.dim for s in layers_sorted)

    # Map layer kinds to codes
    kind_codes = []
    source_keys = []
    internal_js = []
    internal_masks = []
    for s in layers_sorted:
        k = s.kind.lower()
        if k == "multiplier":
            kind_codes.append(KIND_MULTIPLIER)
        elif k in ("multiplierv2", "multiplier_v2", "multipliernocc", "nocc"):
            kind_codes.append(KIND_MULTIPLIERV2)
        elif k in ("singledendrite", "single_dendrite", "dendrite"):
            kind_codes.append(KIND_SINGLEDENDRITE)
        elif k == "mingru":
            kind_codes.append(KIND_MINGRU)
        elif k == "gru":
            kind_codes.append(KIND_GRU)
        elif k == "lstm":
            kind_codes.append(KIND_LSTM)
        elif k == "linear":
            kind_codes.append(KIND_LINEAR)
        elif k == "nonlinear":
            kind_codes.append(KIND_NONLINEAR)
        elif k in ("scalinglayer", "scaling"):
            kind_codes.append(KIND_SCALING)
        else:
            msg = f"Unknown layer kind: {k}"
            raise ValueError(msg)

        source_keys.append(s.source_key or "Heaviside_state_dep")
        internal_js.append(s.internal_J)
        internal_masks.append(getattr(s, "internal_mask", None))

    # Build internal connection learnability dict
    internal_learnable: dict[int, bool] = {}
    for s in layers_sorted:
        if s.internal_J is not None:
            internal_learnable[int(s.layer_id)] = bool(getattr(s, "internal_J_learnable", True))

    layer_kinds = tuple(kind_codes)
    layer_ids_sorted = tuple(int(s.layer_id) for s in layers_sorted)

    # Build edge structure (CSR-like)
    lid_to_idx = {s.layer_id: i for i, s in enumerate(layers_sorted)}
    inbound: dict[int, list[Any]] = {spec.layer_id: [] for spec in layers_sorted}
    for c in jax_model.connections:
        inbound[c.to_layer].append(c)

    # Collect all edges
    all_edges: list[Any] = []
    edge_starts_list = [0]
    for s in layers_sorted:
        edges_for_layer = inbound[s.layer_id]
        all_edges.extend(edges_for_layer)
        edge_starts_list.append(len(all_edges))

    edge_starts = tuple(edge_starts_list)

    if len(all_edges) == 0:
        # No connections
        edge_from_layers: tuple[int, ...] = ()
        edge_matrices = jnp.zeros((0, 1, 1), dtype=jnp.float32)
        edge_dst_dims: tuple[int, ...] = ()
        edge_src_dims: tuple[int, ...] = ()
        edge_learnable: tuple[bool, ...] = ()
    else:
        edge_from_layers = tuple(lid_to_idx[c.from_layer] for c in all_edges)
        edge_from_ids = tuple(int(c.from_layer) for c in all_edges)
        edge_to_ids = tuple(int(c.to_layer) for c in all_edges)

        # Pad all J matrices to same size
        max_dst = max(c.J.shape[0] for c in all_edges)
        max_src = max(c.J.shape[1] for c in all_edges)

        edge_matrices_list = []
        edge_masks_list = []
        edge_dst_dims_list = []
        edge_src_dims_list = []
        edge_learnable_list = []

        for c in all_edges:
            J = c.J
            dst_dim, src_dim = J.shape
            # Pad to max size
            J_padded = jnp.zeros((max_dst, max_src), dtype=J.dtype)
            J_padded = J_padded.at[:dst_dim, :src_dim].set(J)
            edge_matrices_list.append(J_padded)
            # Mask (if present)
            if c.mask is not None:
                M = c.mask
                M_padded = jnp.zeros((max_dst, max_src), dtype=M.dtype)
                M_padded = M_padded.at[:dst_dim, :src_dim].set(M)
                edge_masks_list.append(M_padded)
            else:
                edge_masks_list.append(jnp.ones((max_dst, max_src), dtype=J.dtype))
            edge_dst_dims_list.append(dst_dim)
            edge_src_dims_list.append(src_dim)
            # Extract learnability flag
            edge_learnable_list.append(bool(getattr(c, "learnable", True)))

        edge_matrices = jnp.stack(edge_matrices_list, axis=0)
        edge_masks = jnp.stack(edge_masks_list, axis=0)
        edge_dst_dims = tuple(edge_dst_dims_list)
        edge_src_dims = tuple(edge_src_dims_list)
        edge_learnable = tuple(edge_learnable_list)

    connection_constraints = getattr(jax_model, "connection_constraints", None)

    # Build padded constraint matrices for constraint enforcement
    # This includes both per-element constraints (polarity) AND scalar constraints
    edge_constraint_mins = None
    edge_constraint_maxs = None
    constraint_min_mats = getattr(jax_model, "connection_constraint_min_matrices", None)
    constraint_max_mats = getattr(jax_model, "connection_constraint_max_matrices", None)

    # Check if we have ANY constraints to apply (per-element OR scalar)
    has_any_constraints = (
        (constraint_min_mats and len(constraint_min_mats) > 0) or
        (constraint_max_mats and len(constraint_max_mats) > 0) or
        (connection_constraints and len(connection_constraints) > 0)
    )

    if has_any_constraints and len(all_edges) > 0:
        max_dst = max(c.J.shape[0] for c in all_edges)
        max_src = max(c.J.shape[1] for c in all_edges)

        min_mats_list = []
        max_mats_list = []

        for c in all_edges:
            dst_dim, src_dim = c.J.shape
            key = f"J_{c.from_layer}_to_{c.to_layer}"

            # Start with per-element constraint matrices if they exist (e.g., polarity)
            if constraint_min_mats and key in constraint_min_mats:
                min_mat = jnp.asarray(constraint_min_mats[key][:dst_dim, :src_dim], dtype=jnp.float32)
            else:
                min_mat = jnp.full((dst_dim, src_dim), -float('inf'), dtype=jnp.float32)

            if constraint_max_mats and key in constraint_max_mats:
                max_mat = jnp.asarray(constraint_max_mats[key][:dst_dim, :src_dim], dtype=jnp.float32)
            else:
                max_mat = jnp.full((dst_dim, src_dim), float('inf'), dtype=jnp.float32)

            # Merge scalar constraints from connection_constraints dict
            # Scalar constraints are combined with per-element constraints (more restrictive wins)
            if connection_constraints and key in connection_constraints:
                scalar_cons = connection_constraints[key]
                scalar_min = scalar_cons.get("min", -float("inf"))
                scalar_max = scalar_cons.get("max", float("inf"))
                if scalar_min > -float("inf"):
                    # Take the more restrictive (higher) minimum
                    min_mat = jnp.maximum(min_mat, jnp.float32(scalar_min))
                if scalar_max < float("inf"):
                    # Take the more restrictive (lower) maximum
                    max_mat = jnp.minimum(max_mat, jnp.float32(scalar_max))

            # Pad to max size
            min_padded = jnp.full((max_dst, max_src), -float('inf'), dtype=jnp.float32)
            min_padded = min_padded.at[:dst_dim, :src_dim].set(min_mat)
            max_padded = jnp.full((max_dst, max_src), float('inf'), dtype=jnp.float32)
            max_padded = max_padded.at[:dst_dim, :src_dim].set(max_mat)

            min_mats_list.append(min_padded)
            max_mats_list.append(max_padded)

        edge_constraint_mins = jnp.stack(min_mats_list, axis=0)
        edge_constraint_maxs = jnp.stack(max_mats_list, axis=0)

    # Build internal connection constraint matrices (polarity + scalar constraints)
    internal_constraint_mins_list: list[jnp.ndarray | None] = []
    internal_constraint_maxs_list: list[jnp.ndarray | None] = []
    constraint_min_mats = getattr(jax_model, "connection_constraint_min_matrices", None)
    constraint_max_mats = getattr(jax_model, "connection_constraint_max_matrices", None)

    for s in layers_sorted:
        if s.internal_J is None:
            internal_constraint_mins_list.append(None)
            internal_constraint_maxs_list.append(None)
        else:
            dim = s.internal_J.shape[0]
            key = f"J_{s.layer_id}_to_{s.layer_id}"

            # Start with per-element constraint matrices if they exist (e.g., polarity)
            if constraint_min_mats and key in constraint_min_mats:
                min_mat = jnp.asarray(constraint_min_mats[key][:dim, :dim], dtype=jnp.float32)
            else:
                min_mat = jnp.full((dim, dim), -float('inf'), dtype=jnp.float32)

            if constraint_max_mats and key in constraint_max_mats:
                max_mat = jnp.asarray(constraint_max_mats[key][:dim, :dim], dtype=jnp.float32)
            else:
                max_mat = jnp.full((dim, dim), float('inf'), dtype=jnp.float32)

            # Merge scalar constraints from connection_constraints dict
            if connection_constraints and key in connection_constraints:
                scalar_cons = connection_constraints[key]
                scalar_min = scalar_cons.get("min", -float("inf"))
                scalar_max = scalar_cons.get("max", float("inf"))
                if scalar_min > -float("inf"):
                    min_mat = jnp.maximum(min_mat, jnp.float32(scalar_min))
                if scalar_max < float("inf"):
                    max_mat = jnp.minimum(max_mat, jnp.float32(scalar_max))

            internal_constraint_mins_list.append(min_mat)
            internal_constraint_maxs_list.append(max_mat)

    return Topology(
        dt=jax_model.dt,
        num_layers=L,
        layer_dims=layer_dims,
        layer_kinds=layer_kinds,
        layer_source_keys=tuple(source_keys),
        layer_ids=layer_ids_sorted,
        edge_starts=edge_starts,
        edge_from_layers=edge_from_layers,
        edge_matrices=edge_matrices,
        edge_masks=(edge_masks if len(all_edges) > 0 else None),
        edge_dst_dims=edge_dst_dims,
        edge_src_dims=edge_src_dims,
        network_evaluation_method=jax_model.network_evaluation_method,
        internal_js=internal_js,
        internal_masks=internal_masks,
        internal_constraint_mins=internal_constraint_mins_list,
        internal_constraint_maxs=internal_constraint_maxs_list,
        edge_from_ids=edge_from_ids if len(all_edges) > 0 else (),
        edge_to_ids=tuple(s.layer_id for s in layers_sorted for _ in []) if False else tuple(int(c.to_layer) for c in all_edges) if len(all_edges) > 0 else (),
        input_type=str(getattr(jax_model, "input_type", "flux")),
        edge_learnable=edge_learnable,
        internal_learnable=internal_learnable if internal_learnable else None,
        connection_constraints=connection_constraints if connection_constraints else None,
        edge_constraint_mins=edge_constraint_mins,
        edge_constraint_maxs=edge_constraint_maxs,
    )


def convert_params_to_arrays(param_tree: dict, topology: Topology, connection_constraints: dict[str, dict[str, float]] | None = None) -> tuple[list, jnp.ndarray]:
    """Convert param tree to layer-ordered arrays and connection matrix."""
    L = topology.num_layers
    layer_params = []

    for i in range(L):
        kind = int(topology.layer_kinds[i])
        dim = int(topology.layer_dims[i])

        # Robust mapping: use the layer id ordering captured in the topology
        layer_id = int(topology.layer_ids[i])
        pdict = param_tree.get("layers", {}).get(layer_id, {})

        if kind == KIND_MULTIPLIER:
            # [phi_y, bias_current, gamma_plus, gamma_minus] each [D] -> stack to [4, D]
            # FAIL-FAST: These parameters are required, no silent defaults
            try:
                phi_y = jnp.asarray(pdict["phi_y"])
                bias_current = jnp.asarray(pdict["bias_current"])
                gamma_plus = jnp.asarray(pdict["gamma_plus"])
                gamma_minus = jnp.asarray(pdict["gamma_minus"])
            except KeyError as e:
                msg = f"Multiplier layer {layer_id} missing required parameter: {e}"
                raise ValueError(msg) from e
            layer_params.append(jnp.stack([phi_y, bias_current, gamma_plus, gamma_minus], axis=0))
        elif kind == KIND_MULTIPLIERV2:
            # [phi_y, bias_current, alpha, beta, beta_out] each [D] -> [5, D]
            # FAIL-FAST: These parameters are required, no silent defaults
            try:
                phi_y = jnp.asarray(pdict["phi_y"])
                bias_current = jnp.asarray(pdict["bias_current"])
                alpha = jnp.asarray(pdict["alpha"])
                beta = jnp.asarray(pdict["beta"])
                beta_out = jnp.asarray(pdict["beta_out"])
            except KeyError as e:
                msg = f"MultiplierV2 layer {layer_id} missing required parameter: {e}"
                raise ValueError(msg) from e
            layer_params.append(jnp.stack([phi_y, bias_current, alpha, beta, beta_out], axis=0))
        elif kind == KIND_SINGLEDENDRITE:
            # [phi_offset, bias_current, gamma_plus, gamma_minus] each [D] -> [4, D]
            # FAIL-FAST: These parameters are required, no silent defaults
            try:
                phi_offset = jnp.asarray(pdict["phi_offset"])
                bias_current = jnp.asarray(pdict["bias_current"])
                gamma_plus = jnp.asarray(pdict["gamma_plus"])
                gamma_minus = jnp.asarray(pdict["gamma_minus"])
            except KeyError as e:
                msg = f"SingleDendrite layer {layer_id} missing required parameter: {e}"
                raise ValueError(msg) from e
            layer_params.append(jnp.stack([phi_offset, bias_current, gamma_plus, gamma_minus], axis=0))
        elif kind == KIND_MINGRU:
            # [W_hidden, W_gate] each [D, D] -> [2, D, D]
            W_hidden = jnp.asarray(pdict.get("W_hidden", jnp.eye(dim)))
            W_gate = jnp.asarray(pdict.get("W_gate", jnp.eye(dim)))
            layer_params.append(jnp.stack([W_hidden, W_gate], axis=0))
        elif kind == KIND_LINEAR:
            # No params
            layer_params.append(jnp.zeros((0, dim)))  # Empty placeholder
        elif kind == KIND_NONLINEAR:
            # [phi_offset, bias_current] each [D] -> [2, D]
            # NonLinear params are OPTIONAL (per convert.py lines 323-326)
            # Use zeros if not provided (allows source function to work with raw phi)
            phi_offset = jnp.asarray(pdict.get("phi_offset", jnp.zeros(dim)))
            bias_current = jnp.asarray(pdict.get("bias_current", jnp.zeros(dim)))
            layer_params.append(jnp.stack([phi_offset, bias_current], axis=0))
        elif kind == KIND_SCALING:
            # [scale_factor] [D] -> [1, D]
            # FAIL-FAST: This parameter is required, no silent default
            try:
                scale_factor = jnp.asarray(pdict["scale_factor"])
            except KeyError as e:
                msg = f"Scaling layer {layer_id} missing required parameter: {e}"
                raise ValueError(msg) from e
            layer_params.append(scale_factor[None, :])  # [1, D]
        elif kind == KIND_GRU:
            # GRU: [W_ih(3,D,D), W_hh(3,D,D), b_ih(3,D), b_hh(3,D)] as tuple for JAX
            W_ih = jnp.asarray(pdict.get("weight_ih", jnp.stack([jnp.eye(dim), jnp.eye(dim), jnp.eye(dim)], axis=0)))
            W_hh = jnp.asarray(pdict.get("weight_hh", jnp.stack([jnp.eye(dim), jnp.eye(dim), jnp.eye(dim)], axis=0)))
            b_ih = jnp.asarray(pdict.get("bias_ih", jnp.zeros((3, dim))))
            b_hh = jnp.asarray(pdict.get("bias_hh", jnp.zeros((3, dim))))
            # Cast to Any to allow tuple in list[jnp.ndarray]
            layer_params.append(cast(Any, (W_ih, W_hh, b_ih, b_hh)))
        elif kind == KIND_LSTM:
            # LSTM: [W_ih(4,D,D), W_hh(4,D,D), b_ih(4,D), b_hh(4,D)] as tuple for JAX
            # LSTM has 4 gates: input, forget, cell, output (i, f, g, o)
            W_ih = jnp.asarray(pdict.get("weight_ih", jnp.stack([jnp.eye(dim), jnp.eye(dim), jnp.eye(dim), jnp.eye(dim)], axis=0)))
            W_hh = jnp.asarray(pdict.get("weight_hh", jnp.stack([jnp.eye(dim), jnp.eye(dim), jnp.eye(dim), jnp.eye(dim)], axis=0)))
            b_ih = jnp.asarray(pdict.get("bias_ih", jnp.zeros((4, dim))))
            b_hh = jnp.asarray(pdict.get("bias_hh", jnp.zeros((4, dim))))
            # Cast to Any to allow tuple in list[jnp.ndarray]
            layer_params.append(cast(Any, (W_ih, W_hh, b_ih, b_hh)))
        else:
            layer_params.append(jnp.zeros((0, dim)))

    # Extract connection matrices in edge order
    conn_dict = param_tree.get("connections", {})
    E = len(topology.edge_from_layers)
    if E > 0:
        conn_matrices = []
        # Build from param tree using stored edge id mapping; fallback to original J
        has_ids = bool(getattr(topology, "edge_from_ids", ())) and bool(getattr(topology, "edge_to_ids", ()))
        for i in range(E):
            if has_ids:
                from_id = topology.edge_from_ids[i]
                to_id = topology.edge_to_ids[i]
                key = (from_id, to_id)
                J = conn_dict.get(key, None)
                if J is None:
                    # Fallback to original J
                    J = topology.edge_matrices[i]
                else:
                    base = topology.edge_matrices[i]
                    J = jnp.asarray(J)
                    if J.shape != base.shape:
                        padded = jnp.zeros_like(base)
                        padded = padded.at[: J.shape[0], : J.shape[1]].set(J)
                        J = padded
                J_eff = jnp.asarray(J)
                constraints = connection_constraints or getattr(topology, "connection_constraints", None)
                if constraints:
                    constraint_key = f"J_{from_id}_to_{to_id}"
                    constraint = constraints.get(constraint_key)
                    if constraint:
                        min_val = constraint.get("min")
                        max_val = constraint.get("max")
                        if min_val is not None:
                            J_eff = jnp.maximum(J_eff, min_val)
                        if max_val is not None:
                            J_eff = jnp.minimum(J_eff, max_val)
                conn_matrices.append(J_eff)
            else:
                conn_matrices.append(topology.edge_matrices[i])
        connection_params = jnp.stack(conn_matrices, axis=0)
    else:
        connection_params = jnp.zeros((0, 1, 1))

    return layer_params, connection_params


__all__ = ["Topology", "build_topology", "convert_params_to_arrays"]
