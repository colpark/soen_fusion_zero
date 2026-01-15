"""Unified forward pass for JAX models.

This module provides a single consolidated forward pass that handles:
- All solver types (layerwise, stepwise_jacobi, stepwise_gauss_seidel)
- Optional parameter overrides for training (connection, internal, layer params)
- Initial state overrides
- Noise and perturbation injection

The design eliminates duplication between __call__, _forward_layerwise, and apply_with_conn_params.

Usage:
    from .unified_forward import forward

    # Simple inference
    out, histories = forward(model, external_phi)

    # Training with overrides
    out, histories = forward(model, external_phi,
        conn_override=conn_params,
        layer_param_override=layer_params
    )

    # With noise
    from .noise_jax import build_noise_settings
    noise = build_noise_settings({"phi": 0.01, "s": 0.005})
    out, histories = forward(model, external_phi, noise_settings=noise, rng_key=key)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .jax_model import JAXModel
    from .noise_jax import NoiseSettings
    from .topology_arrays import TopologyArrays

from soen_toolkit.core.layer_registry import is_multiplier_nocc

from .forward_trace import ForwardTrace


class InputDimensionMismatchError(ValueError):
    """Raised when input data dimensions don't match the model's input layer."""

    pass


def _validate_input_dimensions(model: JAXModel, external_phi: jax.Array) -> None:
    """Validate that input data dimensions match the model's first layer.

    This check happens before JAX tracing to provide a clear error message
    instead of cryptic JAX scan shape mismatch errors.

    Raises:
        InputDimensionMismatchError: If dimensions don't match.
    """
    # Get first layer (input layer)
    layers_sorted = sorted(model.layers, key=lambda layer: layer.layer_id)
    if not layers_sorted:
        return  # No layers to validate against

    first_layer = layers_sorted[0]
    input_layer_dim = first_layer.dim

    # Get input data dimension
    if external_phi.ndim != 3:
        raise InputDimensionMismatchError(
            f"Input data must be 3D with shape [batch, time, features], "
            f"but got shape {external_phi.shape} (ndim={external_phi.ndim})"
        )

    data_input_dim = external_phi.shape[-1]

    # Check for mismatch
    if data_input_dim != input_layer_dim:
        raise InputDimensionMismatchError(
            f"\n{'='*70}\n"
            f"INPUT DIMENSION MISMATCH\n"
            f"{'='*70}\n\n"
            f"Your dataset has {data_input_dim} input features, but the model's\n"
            f"first layer (layer_id={first_layer.layer_id}, type={first_layer.kind}) "
            f"expects {input_layer_dim} features.\n\n"
            f"  Dataset input shape: {list(external_phi.shape)} -> {data_input_dim} features\n"
            f"  Model input layer:   {input_layer_dim} neurons\n\n"
            f"To fix this:\n"
            f"  1. Check your dataset's input dimensions\n"
            f"  2. Ensure the model's first layer 'dim' matches your data\n"
            f"  3. Or add a preprocessing layer to transform {data_input_dim} -> {input_layer_dim}\n"
            f"{'='*70}"
        )


def forward(
    model: JAXModel,
    external_phi: jax.Array,
    initial_states: dict[int, jax.Array] | None = None,
    s1_inits: dict[int, jax.Array] | None = None,
    s2_inits: dict[int, jax.Array] | None = None,
    conn_override: jax.Array | None = None,
    internal_conn_override: dict[int, jax.Array] | None = None,
    layer_param_override: tuple[jax.Array, ...] | None = None,
    noise_settings: NoiseSettings | dict | None = None,
    rng_key: jax.Array | None = None,
    *,
    return_trace: bool = False,
    track_phi: bool = False,
    track_g: bool = False,
) -> tuple[jax.Array, list[jax.Array]] | tuple[jax.Array, list[jax.Array], ForwardTrace]:
    """Unified forward pass that routes to appropriate solver.

    Args:
        model: The JAXModel instance
        external_phi: Input flux [B, T, D_in]
        initial_states: Optional dict mapping layer_id to initial state [B, D]
        s1_inits: Optional dict mapping layer_id to s1 initial state (MultiplierV2)
        s2_inits: Optional dict mapping layer_id to s2 initial state (MultiplierV2)
        conn_override: Optional connection override array [E, D_max, F_max] for training
        internal_conn_override: Optional internal connection overrides dict[layer_id -> internal_J]
        layer_param_override: Optional layer parameter overrides (tuple of arrays per layer)
        noise_settings: Optional noise configuration (NoiseSettings or dict)
        rng_key: JAX random key (required if noise_settings is provided)

    Returns:
        Tuple of (final_history [B, T+1, D_out], ordered_histories [list of [B, T+1, D] per layer])

    Raises:
        InputDimensionMismatchError: If input data dimensions don't match the model's input layer.
    """
    if track_g:
        # Keep this fail-fast: g tracing is not implemented yet and silently returning junk
        # would break criticality metrics.
        raise NotImplementedError(
            "JAX g-tracking is not implemented yet. "
            "Set track_g=False or use the Torch backend for g-based metrics for now."
        )
    if (track_phi or track_g) and not return_trace:
        raise ValueError(
            "track_phi/track_g requires return_trace=True (JAX does not store histories on the model)."
        )

    # Validate input dimensions early (before JAX tracing) for clear error messages
    _validate_input_dimensions(model, external_phi)

    # Convert dict config to NoiseSettings if needed
    effective_noise: Any = None
    if noise_settings is not None:
        if isinstance(noise_settings, dict):
            # Fail fast: users sometimes (incorrectly) pass connection-noise configs here.
            # `noise_settings` must be either a NoiseSettings instance or a dict of floats
            # (consumed by `build_noise_settings`). Connection noise belongs in
            # `model.connection_noise_settings` (dict[str, NoiseConfig]) and is applied
            # internally by the forward pass.
            from .noise_jax import NoiseConfig as _JaxNoiseConfig

            if any(isinstance(v, _JaxNoiseConfig) for v in noise_settings.values()):
                msg = (
                    "Invalid `noise_settings`: received a dict containing NoiseConfig values. "
                    "This looks like `model.connection_noise_settings` (connection weight noise) "
                    "and must NOT be passed as `noise_settings`. "
                    "Pass `noise_settings` as a dict of floats (e.g. {'phi': 0.01, 's': 0.005}) "
                    "or as a NoiseSettings instance; configure connection noise on the model via "
                    "`model.connection_noise_settings`."
                )
                raise ValueError(msg)

            from .noise_jax import build_noise_settings
            effective_noise = build_noise_settings(noise_settings)
        else:
            effective_noise = noise_settings
        # Validate rng_key provided
        if rng_key is None and effective_noise is not None and not effective_noise.is_trivial():
            msg = "rng_key must be provided when noise_settings is configured"
            raise ValueError(msg)
    # Ensure topology arrays built if using overrides
    if conn_override is not None and model._topology_arrays is None:
        from .topology_arrays import build_topology_arrays
        model._topology_arrays = build_topology_arrays(model)

    solver = str(model.network_evaluation_method).lower()

    if solver == "stepwise_jacobi":
        from .unified_stepwise import forward_stepwise_jacobi
        out = forward_stepwise_jacobi(
            model, external_phi,
            initial_states=initial_states,
            s1_inits=s1_inits,
            s2_inits=s2_inits,
            conn_override=conn_override,
            internal_conn_override=internal_conn_override,
            layer_param_override=layer_param_override,
            noise_settings=effective_noise,
            rng_key=rng_key,
            return_trace=return_trace,
            track_phi=track_phi,
            track_g=track_g,
        )
        return out

    if solver == "stepwise_gauss_seidel":
        from .unified_stepwise import forward_stepwise_gauss_seidel
        out = forward_stepwise_gauss_seidel(
            model, external_phi,
            initial_states=initial_states,
            s1_inits=s1_inits,
            s2_inits=s2_inits,
            conn_override=conn_override,
            internal_conn_override=internal_conn_override,
            layer_param_override=layer_param_override,
            noise_settings=effective_noise,
            rng_key=rng_key,
            return_trace=return_trace,
            track_phi=track_phi,
            track_g=track_g,
        )
        return out

    # Layerwise forward pass
    final_hist, histories, phi_tuple, s1_final_by_layer, s2_final_by_layer = _forward_layerwise(
        model, external_phi,
        initial_states=initial_states,
        s1_inits=s1_inits,
        s2_inits=s2_inits,
        conn_override=conn_override,
        internal_conn_override=internal_conn_override,
        layer_param_override=layer_param_override,
        noise_settings=effective_noise,
        rng_key=rng_key,
        collect_phi=bool(return_trace and track_phi),
    )
    if not return_trace:
        return final_hist, histories
    trace = ForwardTrace(
        layer_states=tuple(histories),
        phi_by_layer=phi_tuple if track_phi else None,
        g_by_layer=None,
        s1_final_by_layer=s1_final_by_layer,
        s2_final_by_layer=s2_final_by_layer,
    )
    return final_hist, histories, trace


def _forward_layerwise(
    model: JAXModel,
    external_phi: jax.Array,
    initial_states: dict[int, jax.Array] | None = None,
    s1_inits: dict[int, jax.Array] | None = None,
    s2_inits: dict[int, jax.Array] | None = None,
    conn_override: jax.Array | None = None,
    internal_conn_override: dict[int, jax.Array] | None = None,
    layer_param_override: tuple[jax.Array, ...] | None = None,
    noise_settings: Any = None,
    rng_key: jax.Array | None = None,
    *,
    collect_phi: bool = False,
) -> tuple[jax.Array, list[jax.Array], tuple[jax.Array, ...] | None, tuple[jax.Array | None, ...], tuple[jax.Array | None, ...]]:
    """Layerwise feedforward aggregation.

    This is the core layerwise implementation that supports:
    - Fast path for fixed connections (batched einsum)
    - Dynamic connections (WICC, NOCC) via batched kernels
    - Parameter overrides for training
    - Noise/perturbation injection via noise_settings
    """
    B, T, _ = external_phi.shape
    model._ensure_cache()

    # Precompute all noise offsets (consolidated logic)
    from .noise_jax import precompute_forward_noise
    rng_key, noise_ctx = precompute_forward_noise(
        rng_key, model, noise_settings, B, dtype=external_phi.dtype
    )
    perturbation_offsets = noise_ctx.layer_perturbations
    conn_perturbation_offsets = noise_ctx.conn_perturbations or {}

    # Use passed overrides or fall back to model's stored overrides
    effective_conn_override = conn_override if conn_override is not None else model._conn_override
    effective_internal_override = internal_conn_override if internal_conn_override is not None else model._internal_conn_override
    effective_layer_param_override = layer_param_override if layer_param_override is not None else model._layer_param_override
    has_override = effective_conn_override is not None and model._topology_arrays is not None

    layers_sorted = sorted(model.layers, key=lambda layer: layer.layer_id)
    inbound = model._cached_inbound or {layer_item.layer_id: [] for layer_item in layers_sorted}
    layer_idx_map = {spec.layer_id: idx for idx, spec in enumerate(layers_sorted)}

    histories: dict[int, jax.Array] = {}
    phi_by_layer: list[jax.Array] | None = [] if collect_phi else None
    # Optional: capture auxiliary final states per layer, aligned to layer order.
    aux_s1: dict[int, jax.Array] = {}
    aux_s2: dict[int, jax.Array] = {}

    # Process first layer
    first = layers_sorted[0]
    layer_impl = (model._cached_impls or [model._build_layer(first)])[0]
    params = model._layer_params(first, B, effective_internal_override, effective_layer_param_override)

    # Get initial state
    if initial_states is not None and first.layer_id in initial_states:
        s0 = initial_states[first.layer_id]
    else:
        s0 = jnp.zeros((B, first.dim), dtype=external_phi.dtype)

    # Handle first layer based on input type
    if str(model.input_type).lower() == "state":
        # Input is state, not flux
        ext = external_phi
        if ext.shape[-1] != first.dim:
            if ext.shape[-1] > first.dim:
                ext = ext[:, :, :first.dim]
            else:
                pad = first.dim - ext.shape[-1]
                ext = jnp.concatenate([ext, jnp.zeros((B, T, pad), dtype=ext.dtype)], axis=-1)
        h_first = jnp.concatenate([s0[:, None, :], ext], axis=1)
        if phi_by_layer is not None:
            phi_by_layer.append(ext)
    elif first.kind.lower() == "linear":
        h_first = layer_impl.forward(external_phi, s0)
        if phi_by_layer is not None:
            phi_by_layer.append(external_phi)
    else:
        # Get s1/s2 inits if available
        s1_init = s1_inits.get(first.layer_id) if s1_inits else None
        s2_init = s2_inits.get(first.layer_id) if s2_inits else None
        kind_l = str(first.kind).lower()
        if is_multiplier_nocc(kind_l) and hasattr(layer_impl, "forward_with_aux"):
            # Capture final s1/s2 for stateful training without mutating the layer/module.
            try:
                h_first, s1_f, s2_f = layer_impl.forward_with_aux(external_phi, params, s0, s1_init, s2_init)
            except TypeError:
                h_first, s1_f, s2_f = layer_impl.forward_with_aux(external_phi, params, s0)
            aux_s1[first.layer_id] = s1_f
            aux_s2[first.layer_id] = s2_f
        elif s1_init is not None or s2_init is not None:
            try:
                h_first = layer_impl.forward(external_phi, params, s0, s1_init, s2_init)
            except TypeError:
                h_first = layer_impl.forward(external_phi, params, s0)
        else:
            h_first = layer_impl.forward(external_phi, params, s0)
        if phi_by_layer is not None:
            phi_by_layer.append(external_phi)

    histories[first.layer_id] = h_first

    # Process subsequent layers
    for spec in layers_sorted[1:]:
        idx_layer = layer_idx_map[spec.layer_id]
        phi_t = None

        # Try fast path for fixed connections only
        if model._use_fast_layerwise and model._topology_arrays is not None:
            try:
                topo = cast("TopologyArrays", model._topology_arrays)
                conns = inbound[spec.layer_id]
                has_any_connections = len(conns) > 0

                # Check if all connections are fixed (fast path only supports fixed)
                has_dynamic_connections = False
                for c in conns:
                    cat = model._mode_category(getattr(c, "mode", "fixed"))
                    if cat in ("wicc", "nocc"):
                        has_dynamic_connections = True
                        break

                if has_any_connections and not has_dynamic_connections:
                    from .fast_kernels_fixed import layerwise_aggregate_fixed as _agg

                    ordered_hist = []
                    for layer in layers_sorted:
                        hist = histories.get(layer.layer_id)
                        if hist is None:
                            hist = jnp.zeros((B, T + 1, layer.dim), dtype=external_phi.dtype)
                        ordered_hist.append(hist)

                    start = topo.inbound_starts_py[idx_layer]
                    num_edges = topo.inbound_counts[idx_layer]
                    dst_dim = topo.layer_dims_py[idx_layer]
                    J_override = effective_conn_override if has_override else None
                    phi_t = _agg(topo, ordered_hist, start, num_edges, dst_dim, J_override)
            except Exception:
                phi_t = None

        # Slow path if fast path not used or failed
        if phi_t is None:
            conns = inbound[spec.layer_id]
            if len(conns) == 0:
                phi_t = jnp.zeros((B, T, spec.dim), dtype=external_phi.dtype)
            elif len(conns) == 1:
                c0 = conns[0]
                s_src = histories[c0.from_layer][:, 1:, :]
                cat = model._mode_category(getattr(c0, "mode", "fixed"))

                if cat == "wicc":
                    J_ovr = model._get_conn_override_func(c0, idx_layer, 0, effective_conn_override) if has_override else None
                    phi_t = model._multiplier_phi_layerwise(s_src, c0, external_phi.dtype, J_ovr)
                elif cat == "nocc":
                    J_ovr = model._get_conn_override_func(c0, idx_layer, 0, effective_conn_override) if has_override else None
                    phi_t = model._multiplier_v2_phi_layerwise(s_src, c0, external_phi.dtype, J_ovr)
                else:
                    # Fixed connection
                    if has_override and model._topology_arrays is not None:
                        topo = cast("TopologyArrays", model._topology_arrays)
                        start_i_val = jax.lax.index_in_dim(topo.inbound_starts, idx_layer, axis=0, keepdims=False)
                        J_ovr = jnp.take(cast(jax.Array, effective_conn_override), start_i_val, axis=0)
                        J_ovr = J_ovr[:c0.J.shape[0], :c0.J.shape[1]]
                        J_eff = J_ovr if c0.mask is None else (J_ovr * c0.mask)
                    else:
                        J_clamped = model._clamp_connections(c0)
                        J_eff = J_clamped if c0.mask is None else (J_clamped * c0.mask)

                    # Apply connection noise/perturbation if configured
                    conn_key = f"J_{c0.from_layer}_to_{c0.to_layer}"
                    conn_noise_cfg = model.connection_noise_settings.get(conn_key) if model.connection_noise_settings else None
                    perturb_offset = conn_perturbation_offsets.get(conn_key)
                    has_noise = conn_noise_cfg is not None and conn_noise_cfg.noise is not None and conn_noise_cfg.noise.std > 0.0

                    if has_noise and rng_key is not None:
                        # Per-timestep noise: expand J to [T, D, F]
                        from .noise_jax import apply_connection_noise_layerwise
                        rng_key, j_noise_key = jax.random.split(rng_key)
                        J_expanded = apply_connection_noise_layerwise(
                            j_noise_key, J_eff, conn_noise_cfg, T, perturb_offset
                        )
                        # phi_t = einsum('btf,tdf->btd', s_src, J_expanded)
                        phi_t = jnp.einsum("btf,tdf->btd", s_src, J_expanded)
                    else:
                        # No per-timestep noise, just apply perturbation if any
                        if perturb_offset is not None:
                            J_eff = J_eff + perturb_offset
                        phi_t = s_src @ J_eff.T
            else:
                # Multiple connections - separate by type
                fixed_conns, mult_v1_conns, mult_v2_conns = [], [], []
                conn_to_offset = {}  # Maps id(c) -> offset (ConnectionSpec is not hashable)
                for offset, c in enumerate(conns):
                    cat = model._mode_category(getattr(c, "mode", "fixed"))
                    conn_to_offset[id(c)] = offset
                    if cat == "wicc":
                        mult_v1_conns.append(c)
                    elif cat == "nocc":
                        mult_v2_conns.append(c)
                    else:
                        fixed_conns.append(c)

                phi_t = jnp.zeros((B, T, spec.dim), dtype=external_phi.dtype)

                # Process fixed connections
                if fixed_conns:
                    s_stack = jnp.stack([histories[c.from_layer][:, 1:, :] for c in fixed_conns], axis=0)

                    # Check if ANY connection has per-timestep noise
                    any_has_noise = False
                    for c in fixed_conns:
                        conn_key = f"J_{c.from_layer}_to_{c.to_layer}"
                        conn_noise_cfg = model.connection_noise_settings.get(conn_key) if model.connection_noise_settings else None
                        if conn_noise_cfg is not None and conn_noise_cfg.noise is not None and conn_noise_cfg.noise.std > 0.0:
                            any_has_noise = True
                            break

                    if any_has_noise and rng_key is not None:
                        # Need per-timestep noise - use einsum with [K, T, D, F] weights
                        from .noise_jax import apply_connection_noise_layerwise
                        j_expanded_list = []
                        for c in fixed_conns:
                            if has_override:
                                topo = cast("TopologyArrays", model._topology_arrays)
                                start_i_val = jax.lax.index_in_dim(topo.inbound_starts, idx_layer, axis=0, keepdims=False)
                                J_ovr = jnp.take(cast(jax.Array, effective_conn_override), start_i_val + conn_to_offset[id(c)], axis=0)
                                J_ovr = J_ovr[:c.J.shape[0], :c.J.shape[1]]
                                J_eff = J_ovr if c.mask is None else (J_ovr * c.mask)
                            else:
                                J_eff = model._clamp_connections(c)
                                if c.mask is not None:
                                    J_eff = J_eff * c.mask

                            conn_key = f"J_{c.from_layer}_to_{c.to_layer}"
                            conn_noise_cfg = model.connection_noise_settings.get(conn_key) if model.connection_noise_settings else None
                            perturb_offset = conn_perturbation_offsets.get(conn_key)

                            rng_key, j_noise_key = jax.random.split(rng_key)
                            J_expanded = apply_connection_noise_layerwise(
                                j_noise_key, J_eff, conn_noise_cfg, T, perturb_offset
                            )
                            j_expanded_list.append(J_expanded)

                        j_stack = jnp.stack(j_expanded_list, axis=0)  # [K, T, D, F]
                        phi_fixed = jnp.einsum("kbtf,ktdf->btd", s_stack, j_stack)
                    else:
                        # No per-timestep noise - use simpler [K, D, F] weights
                        j_list = []
                        for c in fixed_conns:
                            if has_override:
                                topo = cast("TopologyArrays", model._topology_arrays)
                                start_i_val = jax.lax.index_in_dim(topo.inbound_starts, idx_layer, axis=0, keepdims=False)
                                J_ovr = jnp.take(cast(jax.Array, effective_conn_override), start_i_val + conn_to_offset[id(c)], axis=0)
                                J_ovr = J_ovr[:c.J.shape[0], :c.J.shape[1]]
                                J_eff = J_ovr if c.mask is None else (J_ovr * c.mask)
                            else:
                                J_eff = model._clamp_connections(c)
                                if c.mask is not None:
                                    J_eff = J_eff * c.mask

                            # Apply perturbation only (no per-step noise)
                            conn_key = f"J_{c.from_layer}_to_{c.to_layer}"
                            perturb_offset = conn_perturbation_offsets.get(conn_key)
                            if perturb_offset is not None:
                                J_eff = J_eff + perturb_offset

                            j_list.append(J_eff)

                        j_stack = jnp.stack(j_list, axis=0)  # [K, D, F]
                        phi_fixed = jnp.einsum("kbtf,kdf->btd", s_stack, j_stack)

                    phi_t = phi_t + phi_fixed

                # Process WICC connections (batched - all edges in one scan)
                if mult_v1_conns:
                    from .fast_kernels_dynamic import wicc_aggregate_layerwise
                    # Build offset map using id() for hashability
                    offset_map = {id(c): conn_to_offset[id(c)] for c in mult_v1_conns}
                    phi_wicc, rng_key = wicc_aggregate_layerwise(
                        model, idx_layer, histories, mult_v1_conns,
                        effective_conn_override, offset_map,
                        rng_key=rng_key,
                        conn_perturbation_offsets=conn_perturbation_offsets,
                        T=T,
                    )
                    if phi_wicc is not None:
                        phi_t = phi_t + phi_wicc

                # Process NOCC connections (batched - all edges in one scan)
                if mult_v2_conns:
                    from .fast_kernels_dynamic import nocc_aggregate_layerwise
                    offset_map = {id(c): conn_to_offset[id(c)] for c in mult_v2_conns}
                    phi_nocc, rng_key = nocc_aggregate_layerwise(
                        model, idx_layer, histories, mult_v2_conns,
                        effective_conn_override, offset_map,
                        rng_key=rng_key,
                        conn_perturbation_offsets=conn_perturbation_offsets,
                        T=T,
                    )
                    if phi_nocc is not None:
                        phi_t = phi_t + phi_nocc

        # Apply noise to aggregated phi if configured
        if noise_settings is not None and rng_key is not None and phi_t is not None:
            if not noise_settings.is_trivial() and noise_settings.phi is not None:
                from .noise_jax import apply_noise_step
                # Split key for this layer's phi noise
                rng_key, phi_noise_key = jax.random.split(rng_key)
                # Get perturbation offset (broadcast across time)
                phi_offset = perturbation_offsets.phi if perturbation_offsets else None
                if phi_offset is not None:
                    # Expand [B, D] to [B, T, D]
                    phi_offset = jnp.expand_dims(phi_offset, axis=1)
                # Apply noise to each timestep
                # For layerwise, we apply the same noise strategy across time
                phi_t = apply_noise_step(phi_noise_key, phi_t, noise_settings.phi, phi_offset)

        if phi_by_layer is not None:
            # Record the φ that actually feeds the layer dynamics (post-aggregation + post-noise).
            phi_by_layer.append(phi_t)

        # Apply layer dynamics
        layer_impl = model._cached_impls[idx_layer] if model._cached_impls else model._build_layer(spec)
        params = model._layer_params(spec, B, effective_internal_override, effective_layer_param_override)

        # Get initial state
        if initial_states is not None and spec.layer_id in initial_states:
            s0 = initial_states[spec.layer_id]
        else:
            s0 = jnp.zeros((B, spec.dim), dtype=external_phi.dtype)

        kind_l = str(spec.kind).lower()
        if kind_l == "linear":
            h = layer_impl.forward(phi_t, s0)
        elif kind_l in ("nonlinear", "scalinglayer", "scaling"):
            h = layer_impl.forward(phi_t, params, s0)
        else:
            s1_init = s1_inits.get(spec.layer_id) if s1_inits else None
            s2_init = s2_inits.get(spec.layer_id) if s2_inits else None
            if is_multiplier_nocc(kind_l) and hasattr(layer_impl, "forward_with_aux"):
                try:
                    h, s1_f, s2_f = layer_impl.forward_with_aux(phi_t, params, s0, s1_init, s2_init)
                except TypeError:
                    h, s1_f, s2_f = layer_impl.forward_with_aux(phi_t, params, s0)
                aux_s1[spec.layer_id] = s1_f
                aux_s2[spec.layer_id] = s2_f
            elif s1_init is not None or s2_init is not None:
                try:
                    h = layer_impl.forward(phi_t, params, s0, s1_init, s2_init)
                except TypeError:
                    h = layer_impl.forward(phi_t, params, s0)
            else:
                h = layer_impl.forward(phi_t, params, s0)

        # Apply state noise if configured
        if noise_settings is not None and rng_key is not None:
            if not noise_settings.is_trivial() and noise_settings.s is not None:
                from .noise_jax import apply_noise_step
                rng_key, state_noise_key = jax.random.split(rng_key)
                # Get perturbation offset for this layer's state
                s_offset = None
                if perturbation_offsets and perturbation_offsets.layer_params:
                    layer_offsets = perturbation_offsets.layer_params.get(spec.layer_id, {})
                    s_offset = layer_offsets.get("s")
                    if s_offset is not None:
                        # Expand [B, D] to [B, T+1, D] for history
                        s_offset = jnp.expand_dims(s_offset, axis=1)
                # Apply noise to state history (skip t=0 which is initial)
                h_noisy = h.at[:, 1:, :].set(
                    apply_noise_step(state_noise_key, h[:, 1:, :], noise_settings.s, s_offset)
                )
                h = h_noisy

        histories[spec.layer_id] = h

    # Return results
    final_id = layers_sorted[-1].layer_id
    final_history = histories[final_id]
    ordered_histories = [histories[layer.layer_id] for layer in layers_sorted]
    phi_tuple = tuple(phi_by_layer) if phi_by_layer is not None else None
    s1_final_by_layer = tuple(aux_s1.get(layer.layer_id) if is_multiplier_nocc(str(layer.kind).lower()) else None for layer in layers_sorted)
    s2_final_by_layer = tuple(aux_s2.get(layer.layer_id) if is_multiplier_nocc(str(layer.kind).lower()) else None for layer in layers_sorted)
    return final_history, ordered_histories, phi_tuple, s1_final_by_layer, s2_final_by_layer


__all__ = ["forward", "_forward_layerwise", "InputDimensionMismatchError"]

