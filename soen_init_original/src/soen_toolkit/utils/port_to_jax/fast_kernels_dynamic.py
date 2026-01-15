"""Batched WICC/NOCC aggregation for unified fast path.

Conceptual insight from Multiplier_Equations_Reference.md:
- "Dynamic weights" are really hidden multiplier layers with fixed j_in/j_out
- φ_w (weight flux) is a layer parameter, not a weight
- All operations are tensor ops - no need for per-connection iteration

This module provides batched implementations that process ALL WICC/NOCC
connections to a destination layer in a single operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .jax_model import ConnectionSpec, JAXModel
    from .noise_jax import NoiseConfig


# Smoothing parameter for soft_abs
_SOFT_ABS_EPS = 0.01


def soft_abs_jax(x: jax.Array, eps: float = _SOFT_ABS_EPS) -> jax.Array:
    """Smooth approximation of abs(x) using sqrt(x² + ε).

    Unlike jnp.abs(), this function has a continuous derivative everywhere,
    which is important for numerical stability in Forward Euler integration.
    """
    return jnp.sqrt(x * x + eps)


# =============================================================================
# WICC Batched Aggregation
# =============================================================================

def wicc_aggregate_layerwise(
    model: JAXModel,
    dst_layer_idx: int,
    histories: dict[int, jax.Array],
    wicc_conns: list[ConnectionSpec],
    conn_override: jax.Array | None = None,
    conn_to_offset: dict | None = None,
    rng_key: jax.Array | None = None,
    conn_perturbation_offsets: dict[str, jax.Array] | None = None,
    T: int | None = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Aggregate all WICC connections to a destination layer.

    Processes ALL WICC edges in parallel using a single lax.scan operation.

    Args:
        model: JAX model
        dst_layer_idx: Index of destination layer
        histories: Dict mapping layer_id to history [B, T+1, D]
        wicc_conns: List of WICC ConnectionSpec objects to this layer
        conn_override: Optional connection override array [E, D_max, F_max]
        conn_to_offset: Dict mapping connection to its offset in the inbound list
        rng_key: JAX random key for noise generation (optional)
        conn_perturbation_offsets: Precomputed perturbation offsets dict (optional)
        T: Number of timesteps (optional, inferred from histories if not provided)

    Returns:
        Tuple of (aggregated phi [B, T, D_dst], updated rng_key)
    """
    if not wicc_conns:
        return None, rng_key

    # Get first connection for dimensions/parameters
    c0 = wicc_conns[0]
    B = next(iter(histories.values())).shape[0]
    D = c0.J.shape[0]  # Destination dimension
    dtype = next(iter(histories.values())).dtype
    dt = jnp.asarray(model.dt, dtype=dtype)

    # Infer T if not provided
    if T is None:
        T = next(iter(histories.values())).shape[1] - 1  # histories are [B, T+1, D]

    # Get source function (all WICC connections to same layer use same source)
    src = model._select_source_impl(c0.source_key)

    # If single connection, use existing optimized path
    if len(wicc_conns) == 1:
        s_src = histories[c0.from_layer][:, 1:, :]
        has_override = conn_override is not None and model._topology_arrays is not None
        # conn_to_offset uses id(c) as key for hashability
        offset = conn_to_offset.get(id(c0), 0) if conn_to_offset else 0
        J_ovr = model._get_conn_override_func(c0, dst_layer_idx, offset, conn_override) if has_override else None
        return model._multiplier_phi_layerwise(s_src, c0, dtype, J_ovr), rng_key

    # Multiple WICC connections - batch them all
    # Stack edge parameters across all connections
    K = len(wicc_conns)  # Number of connections

    # Find max source dimension for padding
    F_max = max(c.J.shape[1] for c in wicc_conns)

    # Stack source states: [K, B, T, F_max] with padding
    s_sources = []
    j_eff_list = []
    gamma_plus_list = []
    gamma_minus_list = []
    bias_current_list = []
    j_in_list = []
    j_out_list = []
    any_has_time_varying_noise = False

    has_override = conn_override is not None and model._topology_arrays is not None

    for c in wicc_conns:
        # Get source history
        s_src = histories[c.from_layer][:, 1:, :]  # [B, T, F]
        F_src = s_src.shape[2]

        # Pad to F_max if needed
        if F_src < F_max:
            s_src = jnp.pad(s_src, ((0, 0), (0, 0), (0, F_max - F_src)))
        s_sources.append(s_src)

        # Get effective J (with override and mask)
        if has_override:
            # conn_to_offset uses id(c) as key for hashability
            offset = conn_to_offset.get(id(c), 0) if conn_to_offset else 0
            J_ovr = model._get_conn_override_func(c, dst_layer_idx, offset, conn_override)
            J_base = J_ovr if J_ovr is not None else model._clamp_connections(c)
        else:
            J_base = model._clamp_connections(c)

        J_eff = J_base if c.mask is None else (J_base * c.mask)

        # Apply noise if configured
        conn_key = f"J_{c.from_layer}_to_{c.to_layer}"
        conn_noise_cfg = cast("NoiseConfig | None", model.connection_noise_settings.get(conn_key)) if model.connection_noise_settings else None
        perturb_offset = conn_perturbation_offsets.get(conn_key) if conn_perturbation_offsets else None
        has_noise = conn_noise_cfg is not None and conn_noise_cfg.noise is not None and conn_noise_cfg.noise.std > 0.0

        if has_noise and rng_key is not None:
            # Apply per-timestep noise (expands to [T, D, F])
            from .noise_jax import apply_connection_noise_layerwise
            rng_key, j_noise_key = jax.random.split(rng_key)
            J_eff_expanded = apply_connection_noise_layerwise(
                j_noise_key, J_eff, conn_noise_cfg, T, perturb_offset
            )
            # Pad to F_max after noise application
            if J_eff_expanded.shape[2] < F_max:
                J_eff_expanded = jnp.pad(J_eff_expanded, ((0, 0), (0, 0), (0, F_max - J_eff_expanded.shape[2])))
            j_eff_list.append(J_eff_expanded)  # [T, D, F_max]
            any_has_time_varying_noise = True
        else:
            # No per-timestep noise, just perturbation
            if perturb_offset is not None:
                J_eff = J_eff + perturb_offset
            # Pad to F_max
            if J_eff.shape[1] < F_max:
                J_eff = jnp.pad(J_eff, ((0, 0), (0, F_max - J_eff.shape[1])))
            j_eff_list.append(J_eff)  # [D, F_max]

        # Parameters (scalar per connection for now)
        gamma_plus_list.append(jnp.asarray(c.gamma_plus, dtype=dtype))
        gamma_minus_list.append(jnp.asarray(c.gamma_minus, dtype=dtype))
        bias_current_list.append(jnp.asarray(c.bias_current, dtype=dtype))
        j_in_list.append(jnp.asarray(c.j_in, dtype=dtype))
        j_out_list.append(jnp.asarray(c.j_out, dtype=dtype))

    # Stack everything: [K, ...]
    s_sources_stacked = jnp.stack(s_sources, axis=0)  # [K, B, T, F_max]
    gp = jnp.stack(gamma_plus_list)   # [K]
    gm = jnp.stack(gamma_minus_list)  # [K]
    bc = jnp.stack(bias_current_list) # [K]
    j_in = jnp.stack(j_in_list)       # [K]
    j_out = jnp.stack(j_out_list)     # [K]

    # Handle two cases: static J vs time-varying J
    if any_has_time_varying_noise:
        # Ensure ALL J are [T, D, F_max] by broadcasting static ones
        j_eff_final = []
        for j_eff_item in j_eff_list:
            if j_eff_item.ndim == 2:  # Static [D, F]
                j_eff_item = jnp.broadcast_to(j_eff_item, (T, j_eff_item.shape[0], j_eff_item.shape[1]))
            j_eff_final.append(j_eff_item)
        J_eff_stacked = jnp.stack(j_eff_final, axis=0)  # [K, T, D, F_max]
    else:
        # All static - original path
        J_eff_stacked = jnp.stack(j_eff_list, axis=0)  # [K, D, F_max]

    # Initial edge states: [K, B, D, F_max]
    S0 = jnp.zeros((K, B, D, F_max), dtype=dtype)

    # Define step function based on whether J is time-varying
    if any_has_time_varying_noise:
        def step_fn(S_prev, inputs):
            """Process one timestep for ALL WICC edges across ALL connections.

            S_prev: [K, B, D, F_max] - edge states for all connections
            inputs: tuple of (x_t, t_idx) where x_t is [K, B, F_max] and t_idx is scalar

            Returns: (S_next, phi_total)
            """
            x_t, t_idx = inputs
            # Input flux: φ_in = s_src * j_in, per connection
            phi_in = x_t[:, :, None, :] * j_in[:, None, None, None]

            # Weight flux: J_eff [K, T, D, F] indexed by timestep
            phi_w = J_eff_stacked[:, t_idx, :, :][:, None, :, :]  # [K, 1, D, F]

            # Check for half_flux_offset
            if getattr(wicc_conns[0], 'half_flux_offset', False):
                phi_w = phi_w + 0.5

            # Flux terms
            phi_a = phi_in + phi_w  # [K, B, D, F]
            phi_b = phi_in - phi_w  # [K, B, D, F]

            # SQUID currents (per-connection bias)
            bc_bc = bc[:, None, None, None]  # [K, 1, 1, 1]
            squid_a = bc_bc - S_prev  # [K, B, D, F]
            squid_b = bc_bc + S_prev  # [K, B, D, F]

            # Source function evaluations
            g_a = src.g(phi_a, squid_current=squid_a)
            g_b = src.g(phi_b, squid_current=squid_b)

            # ODE per connection
            gp_bc = gp[:, None, None, None]  # [K, 1, 1, 1]
            gm_bc = gm[:, None, None, None]
            ds_dt = gp_bc * (g_a - g_b) - gm_bc * S_prev

            # Update
            S_next = S_prev + dt * ds_dt

            # Output per connection: sum over source dim, scale by j_out
            j_out_bc = j_out[:, None, None]
            phi_per_conn = jnp.sum(S_next, axis=-1) * j_out_bc  # [K, B, D]

            # Sum across connections
            phi_total = jnp.sum(phi_per_conn, axis=0)  # [B, D]

            return S_next, phi_total

        # Transpose source for scan: [K, B, T, F] -> [T, K, B, F]
        s_sources_t = s_sources_stacked.transpose((2, 0, 1, 3))  # [T, K, B, F]
        # Create timestep indices
        t_indices = jnp.arange(T)
        # Pack inputs
        scan_inputs = (s_sources_t, t_indices)
        # Scan over time
        _, phi_seq = jax.lax.scan(step_fn, S0, scan_inputs)  # [T, B, D]
    else:
        def step_fn(S_prev, x_t):
            """Process one timestep for ALL WICC edges across ALL connections.

            S_prev: [K, B, D, F_max] - edge states for all connections
            x_t: [K, B, F_max] - source states at time t for all connections

            Returns: (S_next, phi_total)
            """
            # Input flux: φ_in = s_src * j_in, per connection
            phi_in = x_t[:, :, None, :] * j_in[:, None, None, None]

            # Weight flux: J_eff [K, D, F] -> [K, 1, D, F]
            phi_w = J_eff_stacked[:, None, :, :]  # [K, 1, D, F]

            # Check for half_flux_offset
            if getattr(wicc_conns[0], 'half_flux_offset', False):
                phi_w = phi_w + 0.5

            # Flux terms
            phi_a = phi_in + phi_w  # [K, B, D, F]
            phi_b = phi_in - phi_w  # [K, B, D, F]

            # SQUID currents (per-connection bias)
            bc_bc = bc[:, None, None, None]  # [K, 1, 1, 1]
            squid_a = bc_bc - S_prev  # [K, B, D, F]
            squid_b = bc_bc + S_prev  # [K, B, D, F]

            # Source function evaluations
            g_a = src.g(phi_a, squid_current=squid_a)
            g_b = src.g(phi_b, squid_current=squid_b)

            # ODE per connection
            gp_bc = gp[:, None, None, None]  # [K, 1, 1, 1]
            gm_bc = gm[:, None, None, None]
            ds_dt = gp_bc * (g_a - g_b) - gm_bc * S_prev

            # Update
            S_next = S_prev + dt * ds_dt

            # Output per connection: sum over source dim, scale by j_out
            j_out_bc = j_out[:, None, None]
            phi_per_conn = jnp.sum(S_next, axis=-1) * j_out_bc  # [K, B, D]

            # Sum across connections
            phi_total = jnp.sum(phi_per_conn, axis=0)  # [B, D]

            return S_next, phi_total

        # Transpose source for scan: [K, B, T, F] -> [T, K, B, F]
        s_sources_t = s_sources_stacked.transpose((2, 0, 1, 3))  # [T, K, B, F]

        # Scan over time
        _, phi_seq = jax.lax.scan(step_fn, S0, s_sources_t)  # [T, B, D]

    return phi_seq.swapaxes(0, 1), rng_key  # [B, T, D], updated_key


# =============================================================================
# NOCC Batched Aggregation
# =============================================================================

def nocc_aggregate_layerwise(
    model: JAXModel,
    dst_layer_idx: int,
    histories: dict[int, jax.Array],
    nocc_conns: list[ConnectionSpec],
    conn_override: jax.Array | None = None,
    conn_to_offset: dict | None = None,
    rng_key: jax.Array | None = None,
    conn_perturbation_offsets: dict[str, jax.Array] | None = None,
    T: int | None = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Aggregate all NOCC connections to a destination layer.

    NOCC has three coupled states per connection: s1, s2 (per-edge) and m (per-dest).
    This batches ALL NOCC edges across all connections.

    Args:
        model: JAX model
        dst_layer_idx: Index of destination layer
        histories: Dict mapping layer_id to history [B, T+1, D]
        nocc_conns: List of NOCC ConnectionSpec objects to this layer
        conn_override: Optional connection override array
        conn_to_offset: Dict mapping connection to its offset
        rng_key: JAX random key for noise generation (optional)
        conn_perturbation_offsets: Precomputed perturbation offsets dict (optional)
        T: Number of timesteps (optional, inferred from histories if not provided)

    Returns:
        Tuple of (aggregated phi [B, T, D_dst], updated rng_key)
    """
    if not nocc_conns:
        return None, rng_key

    c0 = nocc_conns[0]
    B = next(iter(histories.values())).shape[0]
    D = c0.J.shape[0]
    dtype = next(iter(histories.values())).dtype
    dt = jnp.asarray(model.dt, dtype=dtype)

    # Infer T if not provided
    if T is None:
        T = next(iter(histories.values())).shape[1] - 1  # histories are [B, T+1, D]

    src = model._select_source_impl(c0.source_key)

    # Single connection - use existing path
    if len(nocc_conns) == 1:
        s_src = histories[c0.from_layer][:, 1:, :]
        has_override = conn_override is not None and model._topology_arrays is not None
        # conn_to_offset uses id(c) as key for hashability
        offset = conn_to_offset.get(id(c0), 0) if conn_to_offset else 0
        J_ovr = model._get_conn_override_func(c0, dst_layer_idx, offset, conn_override) if has_override else None
        return model._multiplier_v2_phi_layerwise(s_src, c0, dtype, J_ovr), rng_key

    # Multiple NOCC connections - batch them
    K = len(nocc_conns)
    F_max = max(c.J.shape[1] for c in nocc_conns)

    # Collect all connection data
    s_sources = []
    j_eff_list = []
    mask_list = []
    alpha_list = []
    beta_list = []
    beta_out_list = []
    bias_current_list = []
    j_in_list = []
    j_out_list = []
    any_has_time_varying_noise = False

    has_override = conn_override is not None and model._topology_arrays is not None

    for c in nocc_conns:
        s_src = histories[c.from_layer][:, 1:, :]
        F_src = s_src.shape[2]

        if F_src < F_max:
            s_src = jnp.pad(s_src, ((0, 0), (0, 0), (0, F_max - F_src)))
        s_sources.append(s_src)

        if has_override:
            # conn_to_offset uses id(c) as key for hashability
            offset = conn_to_offset.get(id(c), 0) if conn_to_offset else 0
            J_ovr = model._get_conn_override_func(c, dst_layer_idx, offset, conn_override)
            J_base = J_ovr if J_ovr is not None else model._clamp_connections(c)
        else:
            J_base = model._clamp_connections(c)

        J_eff = J_base if c.mask is None else (J_base * c.mask)
        M = jnp.ones_like(J_eff) if c.mask is None else (c.mask != 0).astype(dtype)

        # Apply noise if configured
        conn_key = f"J_{c.from_layer}_to_{c.to_layer}"
        conn_noise_cfg = cast("NoiseConfig | None", model.connection_noise_settings.get(conn_key)) if model.connection_noise_settings else None
        perturb_offset = conn_perturbation_offsets.get(conn_key) if conn_perturbation_offsets else None
        has_noise = conn_noise_cfg is not None and conn_noise_cfg.noise is not None and conn_noise_cfg.noise.std > 0.0

        if has_noise and rng_key is not None:
            # Apply per-timestep noise (expands to [T, D, F])
            from .noise_jax import apply_connection_noise_layerwise
            rng_key, j_noise_key = jax.random.split(rng_key)
            J_eff_expanded = apply_connection_noise_layerwise(
                j_noise_key, J_eff, conn_noise_cfg, T, perturb_offset
            )
            # Pad to F_max after noise application
            if J_eff_expanded.shape[2] < F_max:
                J_eff_expanded = jnp.pad(J_eff_expanded, ((0, 0), (0, 0), (0, F_max - J_eff_expanded.shape[2])))
            j_eff_list.append(J_eff_expanded)  # [T, D, F_max]

            # Also expand mask to match
            M_expanded = jnp.broadcast_to(M, (T, M.shape[0], M.shape[1]))
            if M_expanded.shape[2] < F_max:
                M_expanded = jnp.pad(M_expanded, ((0, 0), (0, 0), (0, F_max - M_expanded.shape[2])))
            mask_list.append(M_expanded)  # [T, D, F_max]
            any_has_time_varying_noise = True
        else:
            # No per-timestep noise, just perturbation
            if perturb_offset is not None:
                J_eff = J_eff + perturb_offset
            # Pad to F_max
            if J_eff.shape[1] < F_max:
                J_eff = jnp.pad(J_eff, ((0, 0), (0, F_max - J_eff.shape[1])))
                M = jnp.pad(M, ((0, 0), (0, F_max - M.shape[1])))
            j_eff_list.append(J_eff)  # [D, F_max]
            mask_list.append(M)  # [D, F_max]

        alpha_list.append(jnp.asarray(c.alpha if c.alpha is not None else 1.64053, dtype=dtype))
        beta_list.append(jnp.asarray(c.beta if c.beta is not None else 303.85, dtype=dtype))
        beta_out_list.append(jnp.asarray(c.beta_out if c.beta_out is not None else 91.156, dtype=dtype))
        bias_current_list.append(jnp.asarray(c.bias_current if c.bias_current is not None else 2.1, dtype=dtype))
        j_in_list.append(jnp.asarray(c.j_in, dtype=dtype))
        j_out_list.append(jnp.asarray(c.j_out, dtype=dtype))

    # Stack
    s_sources_stacked = jnp.stack(s_sources, axis=0)  # [K, B, T, F]
    alpha = jnp.stack(alpha_list)
    beta = jnp.stack(beta_list)
    beta_out = jnp.stack(beta_out_list)
    bc = jnp.stack(bias_current_list)
    j_in = jnp.stack(j_in_list)
    j_out = jnp.stack(j_out_list)

    # Handle two cases: static J vs time-varying J
    if any_has_time_varying_noise:
        # Ensure ALL J and M are [T, D, F_max] by broadcasting static ones
        j_eff_final = []
        mask_final = []
        for j_eff_item, m_item in zip(j_eff_list, mask_list, strict=False):
            if j_eff_item.ndim == 2:  # Static [D, F]
                j_eff_item = jnp.broadcast_to(j_eff_item, (T, j_eff_item.shape[0], j_eff_item.shape[1]))
                m_item = jnp.broadcast_to(m_item, (T, m_item.shape[0], m_item.shape[1]))
            j_eff_final.append(j_eff_item)
            mask_final.append(m_item)
        J_eff_stacked = jnp.stack(j_eff_final, axis=0)  # [K, T, D, F_max]
        M_stacked = jnp.stack(mask_final, axis=0)  # [K, T, D, F_max]
    else:
        # All static - original path
        J_eff_stacked = jnp.stack(j_eff_list, axis=0)  # [K, D, F_max]
        M_stacked = jnp.stack(mask_list, axis=0)  # [K, D, F_max]

    # Compute fan-in and beta_eff per connection
    if any_has_time_varying_noise:
        # fan_in varies per timestep if mask varies
        fan_in = jnp.sum(M_stacked, axis=-1)  # [K, T, D]
        # beta_eff computation: broadcast beta to match
        beta_eff = beta[:, None, None] + 2 * fan_in * beta_out[:, None, None]  # [K, T, D]
    else:
        fan_in = jnp.sum(M_stacked, axis=-1)  # [K, D]
        beta_eff = beta[:, None] + 2 * fan_in * beta_out[:, None]  # [K, D]

    # Initial states
    s1_0 = jnp.zeros((K, B, D, F_max), dtype=dtype)
    s2_0 = jnp.zeros((K, B, D, F_max), dtype=dtype)
    m_0 = jnp.zeros((K, B, D), dtype=dtype)

    # Define step function based on whether J is time-varying
    if any_has_time_varying_noise:
        def step_fn(carry, inputs):
            """Process one timestep for ALL NOCC edges across ALL connections.

            carry: (s1, s2, m) with shapes [K, B, D, F], [K, B, D, F], [K, B, D]
            inputs: tuple of (x_t, t_idx) where x_t is [K, B, F] and t_idx is scalar
            """
            s1_e, s2_e, m_d = carry
            x_t, t_idx = inputs

            # Input flux
            phi_in = x_t[:, :, None, :] * j_in[:, None, None, None]  # [K, B, D, F]
            # Weight flux: J_eff [K, T, D, F] indexed by timestep
            phi_w = J_eff_stacked[:, t_idx, :, :][:, None, :, :]  # [K, 1, D, F]

            if getattr(nocc_conns[0], 'half_flux_offset', False):
                phi_w = phi_w + 0.5

            phi_a = phi_in + phi_w
            phi_b = phi_in - phi_w

            # SQUID currents (NOCC opposite orientation for branch 2)
            bc_bc = bc[:, None, None, None]
            squid_1 = bc_bc - s1_e
            squid_2 = -bc_bc + s2_e

            # Source functions - use soft_abs for smooth derivative
            g1 = src.g(phi_a, squid_current=soft_abs_jax(squid_1))
            g2 = src.g(phi_b, squid_current=soft_abs_jax(squid_2))

            # Aggregate g1 - g2, mask-gated (mask indexed by timestep)
            M_t = M_stacked[:, t_idx, :, :][:, None, :, :]  # [K, 1, D, F]
            g_diff = (g1 - g2) * M_t  # [K, B, D, F]
            g_sum = jnp.sum(g_diff, axis=-1)  # [K, B, D]

            # dm/dt (beta_eff indexed by timestep)
            alpha_bc = alpha[:, None, None]  # [K, 1, 1]
            beta_eff_t = beta_eff[:, t_idx, :][:, None, :]  # [K, 1, D]
            dm_dt = (g_sum - alpha_bc * m_d) / beta_eff_t  # [K, B, D]

            # ds1/dt, ds2/dt
            dm_dt_e = dm_dt[:, :, :, None]  # [K, B, D, 1]
            alpha_e = alpha[:, None, None, None]
            beta_e = beta[:, None, None, None]
            beta_out_e = beta_out[:, None, None, None]

            ds1_dt = ((g1 - beta_out_e * dm_dt_e - alpha_e * s1_e) / beta_e) * M_t
            ds2_dt = ((g2 - beta_out_e * dm_dt_e - alpha_e * s2_e) / beta_e) * M_t

            # Update
            s1_next = s1_e + dt * ds1_dt
            s2_next = s2_e + dt * ds2_dt
            m_next = m_d + dt * dm_dt

            # Output: m * j_out, summed across connections
            j_out_bc = j_out[:, None, None]
            phi_per_conn = m_next * j_out_bc  # [K, B, D]
            phi_total = jnp.sum(phi_per_conn, axis=0)  # [B, D]

            return (s1_next, s2_next, m_next), phi_total

        # Scan
        s_sources_t = s_sources_stacked.transpose((2, 0, 1, 3))  # [T, K, B, F]
        t_indices = jnp.arange(T)
        scan_inputs = (s_sources_t, t_indices)
        init_carry = (s1_0, s2_0, m_0)
        _, phi_seq = jax.lax.scan(step_fn, init_carry, scan_inputs)
    else:
        def step_fn(carry, x_t):
            """Process one timestep for ALL NOCC edges across ALL connections.

            carry: (s1, s2, m) with shapes [K, B, D, F], [K, B, D, F], [K, B, D]
            x_t: [K, B, F] source states at time t
            """
            s1_e, s2_e, m_d = carry

            # Input flux
            phi_in = x_t[:, :, None, :] * j_in[:, None, None, None]  # [K, B, D, F]
            phi_w = J_eff_stacked[:, None, :, :]  # [K, 1, D, F]

            if getattr(nocc_conns[0], 'half_flux_offset', False):
                phi_w = phi_w + 0.5

            phi_a = phi_in + phi_w
            phi_b = phi_in - phi_w

            # SQUID currents (NOCC opposite orientation for branch 2)
            bc_bc = bc[:, None, None, None]
            squid_1 = bc_bc - s1_e
            squid_2 = -bc_bc + s2_e

            # Source functions - use soft_abs for smooth derivative
            g1 = src.g(phi_a, squid_current=soft_abs_jax(squid_1))
            g2 = src.g(phi_b, squid_current=soft_abs_jax(squid_2))

            # Aggregate g1 - g2, mask-gated
            g_diff = (g1 - g2) * M_stacked[:, None, :, :]  # [K, B, D, F]
            g_sum = jnp.sum(g_diff, axis=-1)  # [K, B, D]

            # dm/dt
            alpha_bc = alpha[:, None, None]  # [K, 1, 1]
            dm_dt = (g_sum - alpha_bc * m_d) / beta_eff[:, None, :]  # [K, B, D]

            # ds1/dt, ds2/dt
            dm_dt_e = dm_dt[:, :, :, None]  # [K, B, D, 1]
            alpha_e = alpha[:, None, None, None]
            beta_e = beta[:, None, None, None]
            beta_out_e = beta_out[:, None, None, None]

            ds1_dt = ((g1 - beta_out_e * dm_dt_e - alpha_e * s1_e) / beta_e) * M_stacked[:, None, :, :]
            ds2_dt = ((g2 - beta_out_e * dm_dt_e - alpha_e * s2_e) / beta_e) * M_stacked[:, None, :, :]

            # Update
            s1_next = s1_e + dt * ds1_dt
            s2_next = s2_e + dt * ds2_dt
            m_next = m_d + dt * dm_dt

            # Output: m * j_out, summed across connections
            j_out_bc = j_out[:, None, None]
            phi_per_conn = m_next * j_out_bc  # [K, B, D]
            phi_total = jnp.sum(phi_per_conn, axis=0)  # [B, D]

            return (s1_next, s2_next, m_next), phi_total

        # Scan
        s_sources_t = s_sources_stacked.transpose((2, 0, 1, 3))  # [T, K, B, F]
        init_carry = (s1_0, s2_0, m_0)
        _, phi_seq = jax.lax.scan(step_fn, init_carry, s_sources_t)

    return phi_seq.swapaxes(0, 1), rng_key  # [B, T, D], updated_key


__all__ = ["wicc_aggregate_layerwise", "nocc_aggregate_layerwise"]

