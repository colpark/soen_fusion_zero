"""Unified stepwise forward pass for JAX models.

This module provides a single implementation that handles both Jacobi and
Gauss-Seidel solvers, eliminating ~350 lines of duplicated code.

The key difference between solvers:
- Jacobi: All layers use previous timestep states for source computations
- Gauss-Seidel: Layers use updated states when available (freshest sources)

Usage:
    from .unified_stepwise import forward_stepwise

    # Jacobi mode
    out, histories = forward_stepwise(model, external_phi, use_gauss_seidel=False)

    # Gauss-Seidel mode
    out, histories = forward_stepwise(model, external_phi, use_gauss_seidel=True)

    # With noise
    from .noise_jax import build_noise_settings
    noise = build_noise_settings({"phi": 0.01, "s": 0.005})
    out, histories = forward_stepwise(model, external_phi, noise_settings=noise, rng_key=key)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from soen_toolkit.core.layer_registry import is_multiplier_nocc

if TYPE_CHECKING:
    from .jax_model import JAXModel
    from .noise_jax import NoiseSettings
    from .topology_arrays import TopologyArrays

from .forward_trace import ForwardTrace


def forward_stepwise(
    model: JAXModel,
    external_phi: jax.Array,
    use_gauss_seidel: bool = False,
    initial_states: dict[int, jax.Array] | None = None,
    s1_inits: dict[int, jax.Array] | None = None,
    s2_inits: dict[int, jax.Array] | None = None,
    conn_override: jax.Array | None = None,
    internal_conn_override: dict[int, jax.Array] | None = None,
    layer_param_override: tuple[jax.Array, ...] | None = None,
    noise_settings: NoiseSettings | None = None,
    rng_key: jax.Array | None = None,
    *,
    return_trace: bool = False,
    track_phi: bool = False,
    track_g: bool = False,
) -> tuple[jax.Array, list[jax.Array]] | tuple[jax.Array, list[jax.Array], ForwardTrace]:
    """Unified stepwise forward pass for both Jacobi and Gauss-Seidel solvers.

    Args:
        model: The JAXModel instance
        external_phi: Input flux [B, T, D_in]
        use_gauss_seidel: If True, use Gauss-Seidel (freshest states). If False, use Jacobi.
        initial_states: Optional dict mapping layer_id to initial state [B, D]
        s1_inits: Optional dict mapping layer_id to s1 initial state (MultiplierNOCC)
        s2_inits: Optional dict mapping layer_id to s2 initial state (MultiplierNOCC)
        conn_override: Optional connection override array [E, D_max, F_max]
        internal_conn_override: Optional internal connection overrides dict[layer_id -> internal_J]
        layer_param_override: Optional layer parameter overrides (tuple of arrays per layer)
        noise_settings: Optional noise configuration
        rng_key: JAX random key (required if noise_settings is provided)

    Returns:
        Tuple of (final_history [B, T+1, D_out], ordered_histories [list of [B, T+1, D] per layer])
    """
    B, T, _ = external_phi.shape
    if (track_phi or track_g) and not return_trace:
        raise ValueError("track_phi/track_g requires return_trace=True.")

    # Precompute all noise offsets (consolidated logic)
    from .noise_jax import precompute_forward_noise
    rng_key, noise_ctx = precompute_forward_noise(
        rng_key, model, noise_settings, B, dtype=external_phi.dtype
    )
    perturbation_offsets = noise_ctx.layer_perturbations
    conn_perturbation_offsets = noise_ctx.conn_perturbations or {}
    use_noise = rng_key is not None and (noise_ctx.has_layer_noise or noise_ctx.has_connection_noise)

    # Pre-split keys for each timestep (T keys) if any noise is used
    # Connection noise (GaussianNoise) should be time-varying, applied per timestep
    if use_noise and rng_key is not None:
        rng_key, *step_keys = jax.random.split(rng_key, T + 1)
        step_keys_array = jnp.stack(step_keys)  # [T, 2] key array
    else:
        step_keys_array = None

    # Use passed overrides or fall back to model's stored overrides
    effective_conn_override = conn_override if conn_override is not None else model._conn_override
    effective_internal_override = internal_conn_override if internal_conn_override is not None else model._internal_conn_override
    effective_layer_param_override = layer_param_override if layer_param_override is not None else model._layer_param_override

    # Ensure caches are built
    model._ensure_cache()
    layers_sorted = sorted(model.layers, key=lambda layer: layer.layer_id)

    # Use cached lookups
    idx_by_id = model._layer_id_to_pos
    inbound = model._cached_inbound
    layer_impls = model._cached_impls

    # Build layer params
    # NOTE: Do NOT cache onto model attributes here - this function runs inside JIT
    # and storing traced values onto self causes JAX tracer leaks
    layer_params = [
        model._layer_params(layer_item, B, effective_internal_override, effective_layer_param_override)
        for layer_item in layers_sorted
    ]

    # Initial states per layer [B,D]
    s0_list = []
    for layer_item in layers_sorted:
        if initial_states is not None and layer_item.layer_id in initial_states:
            s0_list.append(initial_states[layer_item.layer_id])
        else:
            s0_list.append(jnp.zeros((B, layer_item.dim), dtype=external_phi.dtype))

    # Track internal MultiplierNOCC layer states (s1, s2) for stepwise persistence
    layer_s1_states: list[jax.Array | None] = []
    layer_s2_states: list[jax.Array | None] = []
    for spec in layers_sorted:
        kind_l = str(spec.kind).lower()
        if is_multiplier_nocc(kind_l):
            if s1_inits is not None and spec.layer_id in s1_inits:
                layer_s1_states.append(s1_inits[spec.layer_id])
            else:
                layer_s1_states.append(jnp.zeros((B, spec.dim), dtype=external_phi.dtype))
            if s2_inits is not None and spec.layer_id in s2_inits:
                layer_s2_states.append(s2_inits[spec.layer_id])
            else:
                layer_s2_states.append(jnp.zeros((B, spec.dim), dtype=external_phi.dtype))
        else:
            layer_s1_states.append(None)
            layer_s2_states.append(None)

    # Initial edge states per connection
    edge_states0: list[Any] = []
    for c in model.connections:
        D, F = c.J.shape
        cat = model._mode_category(getattr(c, "mode", "fixed"))
        if cat == "nocc":
            # V2: tuple of three states
            s1_0 = jnp.zeros((B, D, F), dtype=external_phi.dtype)
            s2_0 = jnp.zeros((B, D, F), dtype=external_phi.dtype)
            m_0 = jnp.zeros((B, D), dtype=external_phi.dtype)
            edge_states0.append((s1_0, s2_0, m_0))
        else:
            # V1 (WICC) or fixed: single tensor
            edge_states0.append(jnp.zeros((B, D, F), dtype=external_phi.dtype))
    edge_states0 = tuple(edge_states0)

    # Precompute topology info for fast path
    layer_edge_ranges = []
    layer_edge_modes = []
    edge_from_layer_idx_concrete = None
    edge_src_dims_concrete = None

    # Select appropriate fast path flag
    use_fast_path = model._use_fast_gs if use_gauss_seidel else model._use_fast_jacobi
    fast_path_name = "GS" if use_gauss_seidel else "Jacobi"

    if use_fast_path and model._topology_arrays is not None:
        topo = cast("TopologyArrays", model._topology_arrays)
        inbound_starts_concrete = list(topo.inbound_starts_py)
        edge_from_layer_idx_concrete = topo.edge_from_layer_idx_py
        edge_src_dims_concrete = topo.edge_src_dims_py
        # edge_dst_dims_concrete not needed for fixed-only fast path

        for i in range(len(layers_sorted)):
            start_i = inbound_starts_concrete[i]
            end_i = inbound_starts_concrete[i + 1]
            layer_edge_ranges.append((start_i, end_i))
            if end_i > start_i:
                modes_slice = topo.edge_mode_py[start_i:end_i]
                layer_edge_modes.append(np.array(modes_slice, dtype=np.int32))
            else:
                layer_edge_modes.append(np.array([]))

    collect_phi = bool(return_trace and track_phi)

    def step_fn(carry, scan_input):
        prev_states_tuple, edge_states_tuple, layer_s1_tuple, layer_s2_tuple = carry

        # Unpack scan input - either just t_phi or (t_phi, step_key)
        if use_noise and step_keys_array is not None:
            t_phi, step_key = scan_input
        else:
            t_phi = scan_input
            step_key = None

        prev_states = list(prev_states_tuple)
        edge_states = list(edge_states_tuple)
        layer_s1_states_l = list(layer_s1_tuple)
        layer_s2_states_l = list(layer_s2_tuple)
        next_layer_s1 = list(layer_s1_states_l)
        next_layer_s2 = list(layer_s2_states_l)

        # State management differs between Jacobi and Gauss-Seidel
        if use_gauss_seidel:
            next_states = list(prev_states_tuple)  # Copy, update in-place
        else:
            next_states = []  # Build as we go

        def get_source_state(src_idx: int, layer_idx: int) -> jax.Array:
            """Get source state based on solver type."""
            if use_gauss_seidel:
                return next_states[src_idx] if src_idx < layer_idx else prev_states[src_idx]
            else:
                return prev_states[src_idx]

        phi_per_layer: list[jax.Array] | None = [] if collect_phi else None

        for i, spec in enumerate(layers_sorted):
            # Base phi from external only for first layer when input_type == 'flux'
            if i == 0 and str(model.input_type).lower() != "state":
                phi_sum = t_phi
            else:
                phi_sum = jnp.zeros((B, spec.dim), dtype=external_phi.dtype)

            # For φ tracking, record the effective φ that conceptually feeds the layer.
            # In input_type='state', the first layer does not consume φ for dynamics;
            # we still expose the external input as φ for consistency with tooling.
            if phi_per_layer is not None:
                if i == 0 and str(model.input_type).lower() == "state":
                    phi_per_layer.append(t_phi)
                else:
                    # Append a placeholder for now; we'll overwrite after aggregation.
                    phi_per_layer.append(phi_sum)

            # Fast path aggregation
            used_fast = False
            try:
                if use_fast_path and (model._topology_arrays is not None) and len(layer_edge_ranges) > 0:
                    topo = cast("TopologyArrays", model._topology_arrays)
                    start_i, end_i = layer_edge_ranges[i]

                    if end_i > start_i:
                        modes = layer_edge_modes[i]
                        F_max = int(topo.edge_J.shape[2])
                        # D_max only needed for WICC/NOCC fast path (currently disabled)

                        if bool(np.all(modes == 0)):
                            # All fixed connections - batched einsum
                            from .fast_kernels_fixed import stepwise_aggregate_fixed as stepwise_aggregate_fixed_edges

                            Ew = end_i - start_i
                            s_src_edges = jnp.zeros((Ew, B, F_max), dtype=external_phi.dtype)
                            for e_rel in range(Ew):
                                e_abs = start_i + e_rel
                                src_l_idx = int(edge_from_layer_idx_concrete[e_abs])
                                s_src_vec = get_source_state(src_l_idx, i)
                                src_dim = int(edge_src_dims_concrete[e_abs])
                                take = min(s_src_vec.shape[-1], src_dim, F_max)
                                s_src_edges = s_src_edges.at[e_rel, :, :take].set(s_src_vec[:, :take])
                            J_slice = None
                            if effective_conn_override is not None:
                                J_slice = effective_conn_override[start_i:end_i, :, :]
                            phi_add_sum = stepwise_aggregate_fixed_edges(topo, s_src_edges, start_i, end_i, spec.dim, J_slice)
                            phi_sum = phi_sum + phi_add_sum
                            used_fast = True

                        # WICC and NOCC fast paths disabled due to known bugs
                        # See: jax_model.py line 1058 comment
                        # TODO: Fix and re-enable when dynamic fast path is fixed

            except Exception as e:
                # Do not mutate `model` inside tracing/JIT. Logging once per trace is enough.
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"[JAX][{fast_path_name}] Fast path failed, using slow path. Error: {e}")
                used_fast = False

            if not used_fast:
                # Slow path: process connections one by one
                conn_offset = 0
                for c in inbound[spec.layer_id]:
                    src_idx = idx_by_id[c.from_layer]
                    s_src = get_source_state(src_idx, i)
                    cat = model._mode_category(getattr(c, "mode", "fixed"))

                    if cat == "wicc":
                        pos = model._conn_pos_map[(c.from_layer, c.to_layer, getattr(c, "index", 0))]
                        S_prev = edge_states[pos]
                        has_override = effective_conn_override is not None and model._topology_arrays is not None
                        J_override = model._get_conn_override_func(c, i, conn_offset, effective_conn_override) if has_override else None

                        # Get noise config
                        conn_key = f"J_{c.from_layer}_to_{c.to_layer}"
                        conn_noise_cfg = model.connection_noise_settings.get(conn_key) if model.connection_noise_settings else None
                        perturb_offset = conn_perturbation_offsets.get(conn_key)

                        if use_gauss_seidel:
                            # GS: Use phi from updated edge state (returned by _multiplier_step)
                            phi_add, S_next, step_key = model._multiplier_step(
                                s_src, S_prev, c, external_phi.dtype, J_override,
                                rng_key=step_key,
                                conn_noise_cfg=conn_noise_cfg,
                                perturb_offset=perturb_offset,
                            )
                        else:
                            # Jacobi: Use phi from previous edge state
                            phi_add = jnp.sum(S_prev, axis=-1)
                            j_out = jnp.asarray(getattr(c, "j_out", 1.0), dtype=external_phi.dtype)
                            phi_add = phi_add * j_out
                            _, S_next, step_key = model._multiplier_step(
                                s_src, S_prev, c, external_phi.dtype, J_override,
                                rng_key=step_key,
                                conn_noise_cfg=conn_noise_cfg,
                                perturb_offset=perturb_offset,
                            )

                        edge_states[pos] = S_next
                        phi_sum = phi_sum + phi_add

                    elif cat == "nocc":
                        pos = model._conn_pos_map[(c.from_layer, c.to_layer, getattr(c, "index", 0))]
                        state_prev = edge_states[pos]
                        has_override = effective_conn_override is not None and model._topology_arrays is not None
                        J_override = model._get_conn_override_func(c, i, conn_offset, effective_conn_override) if has_override else None

                        # Get noise config
                        conn_key = f"J_{c.from_layer}_to_{c.to_layer}"
                        conn_noise_cfg = model.connection_noise_settings.get(conn_key) if model.connection_noise_settings else None
                        perturb_offset = conn_perturbation_offsets.get(conn_key)

                        if use_gauss_seidel:
                            # GS: Use phi from updated edge state (returned by _multiplier_v2_step)
                            phi_add, state_next, step_key = model._multiplier_v2_step(
                                s_src, state_prev, c, external_phi.dtype, J_override,
                                rng_key=step_key,
                                conn_noise_cfg=conn_noise_cfg,
                                perturb_offset=perturb_offset,
                            )
                        else:
                            # Jacobi: Use phi from previous edge state (m_d is the third element)
                            phi_add = state_prev[2]
                            j_out = jnp.asarray(getattr(c, "j_out", 1.0), dtype=external_phi.dtype)
                            phi_add = phi_add * j_out
                            _, state_next, step_key = model._multiplier_v2_step(
                                s_src, state_prev, c, external_phi.dtype, J_override,
                                rng_key=step_key,
                                conn_noise_cfg=conn_noise_cfg,
                                perturb_offset=perturb_offset,
                            )

                        edge_states[pos] = state_next
                        phi_sum = phi_sum + phi_add

                    else:
                        # Fixed connection
                        has_override = effective_conn_override is not None and model._topology_arrays is not None
                        if has_override:
                            J_override = model._get_conn_override_func(c, i, conn_offset, effective_conn_override)
                            J_eff = J_override if (c.mask is None) else (J_override * c.mask)
                        else:
                            J_eff = model._clamp_connections(c)
                            if c.mask is not None:
                                J_eff = J_eff * c.mask

                        # Apply connection noise (per-timestep) and perturbation (fixed)
                        conn_key = f"J_{c.from_layer}_to_{c.to_layer}"
                        conn_noise_cfg = model.connection_noise_settings.get(conn_key) if model.connection_noise_settings else None
                        perturb_offset = conn_perturbation_offsets.get(conn_key)

                        # Apply perturbation (fixed per forward pass)
                        if perturb_offset is not None:
                            J_eff = J_eff + perturb_offset

                        # Apply per-timestep noise (GaussianNoise = time-varying)
                        if conn_noise_cfg is not None and step_key is not None:
                            if conn_noise_cfg.noise is not None and conn_noise_cfg.noise.std > 0.0:
                                from .noise_jax import apply_connection_noise
                                step_key, j_noise_key = jax.random.split(step_key)
                                # Pass None for perturb since we already applied it above
                                J_eff = apply_connection_noise(j_noise_key, J_eff, conn_noise_cfg, None)

                        phi_sum = phi_sum + (s_src @ J_eff.T)
                    conn_offset += 1

            # Apply phi noise if configured
            if use_noise and step_key is not None and noise_settings is not None and noise_settings.phi is not None:
                from .noise_jax import apply_noise_step
                step_key, phi_key = jax.random.split(step_key)
                phi_offset = perturbation_offsets.phi if perturbation_offsets else None
                phi_sum = apply_noise_step(phi_key, phi_sum, noise_settings.phi, phi_offset)

            if phi_per_layer is not None:
                # Overwrite with final φ used for this layer (post-aggregation + post-noise).
                if i == 0 and str(model.input_type).lower() == "state":
                    # keep external input as the φ-like trace for input-type state
                    pass
                else:
                    phi_per_layer[i] = phi_sum

            # Advance layer state
            impl = layer_impls[i]
            params = layer_params[i]

            if i == 0 and str(model.input_type).lower() == "state":
                s_next = t_phi
            else:
                phi_1 = phi_sum[:, None, :]
                kind_l = str(spec.kind).lower()
                use_forward_for_internal = False
                if is_multiplier_nocc(kind_l):
                    internal_mode = str(getattr(spec, "internal_mode", "fixed")).lower()
                    if internal_mode in ("wicc", "nocc"):
                        use_forward_for_internal = True

                if is_multiplier_nocc(kind_l) and layer_s1_states_l[i] is not None and not use_forward_for_internal:
                    s1_next_l, s2_next_l, m_next = impl.step(phi_sum, params, prev_states[i], layer_s1_states_l[i], layer_s2_states_l[i])
                    s_next = m_next
                    next_layer_s1[i] = s1_next_l
                    next_layer_s2[i] = s2_next_l
                elif is_multiplier_nocc(kind_l) and use_forward_for_internal:
                    if hasattr(impl, "forward_with_aux"):
                        s_hist, s1_final, s2_final = impl.forward_with_aux(phi_1, params, prev_states[i], layer_s1_states_l[i], layer_s2_states_l[i])
                        s_next = s_hist[:, -1, :]
                        next_layer_s1[i] = s1_final
                        next_layer_s2[i] = s2_final
                    else:
                        # Fail fast: NOCC internal dynamics require auxiliary state to be carried.
                        raise NotImplementedError(
                            "MultiplierNOCC layer implementation is missing forward_with_aux(). "
                            "This is required to carry s1/s2 states without mutating the module."
                        )
                else:
                    try:
                        s_hist = impl.forward(phi_1, params, prev_states[i])
                    except TypeError:
                        s_hist = impl.forward(phi_1, prev_states[i])
                    s_next = s_hist[:, -1, :]

            # Apply state noise if configured
            if use_noise and step_key is not None and noise_settings is not None and noise_settings.s is not None:
                from .noise_jax import apply_noise_step
                step_key, state_key = jax.random.split(step_key)
                s_offset = None
                if perturbation_offsets and perturbation_offsets.layer_params:
                    layer_offsets = perturbation_offsets.layer_params.get(spec.layer_id, {})
                    s_offset = layer_offsets.get("s")
                s_next = apply_noise_step(state_key, s_next, noise_settings.s, s_offset)

            # Update state storage based on solver type
            if use_gauss_seidel:
                next_states[i] = s_next
            else:
                next_states.append(s_next)

        if phi_per_layer is None:
            return (tuple(next_states), tuple(edge_states), tuple(next_layer_s1), tuple(next_layer_s2)), tuple(next_states)
        return (
            (tuple(next_states), tuple(edge_states), tuple(next_layer_s1), tuple(next_layer_s2)),
            (tuple(next_states), tuple(phi_per_layer)),
        )

    # Scan over time
    init_carry = (tuple(s0_list), edge_states0, tuple(layer_s1_states), tuple(layer_s2_states))

    # Prepare scan input - either just phi or (phi, step_keys)
    phi_transposed = external_phi.swapaxes(0, 1)  # [T, B, D]
    if use_noise and step_keys_array is not None:
        scan_input = (phi_transposed, step_keys_array)
    else:
        scan_input = phi_transposed

    if collect_phi:
        carry_final, (s_seq, phi_seq) = jax.lax.scan(step_fn, init_carry, scan_input)
    else:
        carry_final, s_seq = jax.lax.scan(step_fn, init_carry, scan_input)
        phi_seq = None

    histories: list[jax.Array] = []
    phi_histories: list[jax.Array] | None = [] if collect_phi else None
    for i, _layer_item in enumerate(layers_sorted):
        s_i_tbd = s_seq[i]  # [T,B,D]
        s_i_bt = s_i_tbd.swapaxes(0, 1)  # [B,T,D]
        s_i_hist = jnp.concatenate([s0_list[i][:, None, :], s_i_bt], axis=1)
        histories.append(s_i_hist)
        if phi_histories is not None and phi_seq is not None:
            phi_i_tbd = phi_seq[i]  # [T,B,D]
            phi_i_bt = phi_i_tbd.swapaxes(0, 1)  # [B,T,D]
            phi_histories.append(phi_i_bt)

    final_history = histories[-1]
    if not return_trace:
        return final_history, histories

    phi_tuple = tuple(phi_histories) if phi_histories is not None else None
    # Extract final s1/s2 auxiliary states for NOCC layers from final carry.
    _final_states, _final_edge_states, final_s1_tuple, final_s2_tuple = carry_final
    s1_final_by_layer: list[jax.Array | None] = []
    s2_final_by_layer: list[jax.Array | None] = []
    for i, spec in enumerate(layers_sorted):
        kind_l = str(spec.kind).lower()
        if is_multiplier_nocc(kind_l):
            s1_final_by_layer.append(final_s1_tuple[i])
            s2_final_by_layer.append(final_s2_tuple[i])
        else:
            s1_final_by_layer.append(None)
            s2_final_by_layer.append(None)
    trace = ForwardTrace(
        layer_states=tuple(histories),
        phi_by_layer=phi_tuple,
        g_by_layer=None,
        s1_final_by_layer=tuple(s1_final_by_layer),
        s2_final_by_layer=tuple(s2_final_by_layer),
    )
    return final_history, histories, trace


def forward_stepwise_jacobi(
    model: JAXModel,
    external_phi: jax.Array,
    initial_states: dict[int, jax.Array] | None = None,
    s1_inits: dict[int, jax.Array] | None = None,
    s2_inits: dict[int, jax.Array] | None = None,
    conn_override: jax.Array | None = None,
    internal_conn_override: dict[int, jax.Array] | None = None,
    layer_param_override: tuple[jax.Array, ...] | None = None,
    noise_settings: NoiseSettings | None = None,
    rng_key: jax.Array | None = None,
    *,
    return_trace: bool = False,
    track_phi: bool = False,
    track_g: bool = False,
) -> tuple[jax.Array, list[jax.Array]] | tuple[jax.Array, list[jax.Array], ForwardTrace]:
    """Stepwise Jacobi forward pass.

    Jacobi solver: All layers use previous timestep states for source computations.
    """
    return forward_stepwise(
        model,
        external_phi,
        use_gauss_seidel=False,
        initial_states=initial_states,
        s1_inits=s1_inits,
        s2_inits=s2_inits,
        conn_override=conn_override,
        internal_conn_override=internal_conn_override,
        layer_param_override=layer_param_override,
        noise_settings=noise_settings,
        rng_key=rng_key,
        return_trace=return_trace,
        track_phi=track_phi,
        track_g=track_g,
    )


def forward_stepwise_gauss_seidel(
    model: JAXModel,
    external_phi: jax.Array,
    initial_states: dict[int, jax.Array] | None = None,
    s1_inits: dict[int, jax.Array] | None = None,
    s2_inits: dict[int, jax.Array] | None = None,
    conn_override: jax.Array | None = None,
    internal_conn_override: dict[int, jax.Array] | None = None,
    layer_param_override: tuple[jax.Array, ...] | None = None,
    noise_settings: NoiseSettings | None = None,
    rng_key: jax.Array | None = None,
    *,
    return_trace: bool = False,
    track_phi: bool = False,
    track_g: bool = False,
) -> tuple[jax.Array, list[jax.Array]] | tuple[jax.Array, list[jax.Array], ForwardTrace]:
    """Stepwise Gauss-Seidel forward pass.

    Gauss-Seidel solver: Layers use freshest available states.
    """
    return forward_stepwise(
        model,
        external_phi,
        use_gauss_seidel=True,
        initial_states=initial_states,
        s1_inits=s1_inits,
        s2_inits=s2_inits,
        conn_override=conn_override,
        internal_conn_override=internal_conn_override,
        layer_param_override=layer_param_override,
        noise_settings=noise_settings,
        rng_key=rng_key,
        return_trace=return_trace,
        track_phi=track_phi,
        track_g=track_g,
    )


__all__ = [
    "forward_stepwise",
    "forward_stepwise_jacobi",
    "forward_stepwise_gauss_seidel",
]

