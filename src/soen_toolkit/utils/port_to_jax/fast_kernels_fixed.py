"""Fast kernels for fixed connections only.

This module provides optimized batched operations for fixed connections.
Dynamic connections (WICC/NOCC) are handled by the slow path in the
forward pass modules since they require per-connection state evolution.

NOTE: This is a clean extraction of the working fixed-connection code
from fast_kernels_v2.py. The WICC/NOCC fast path code in that file has
known bugs (sign errors, state management issues) and is disabled.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .topology_arrays import TopologyArrays


def layerwise_aggregate_fixed(
    topo: TopologyArrays,
    histories: list[jnp.ndarray],
    start: int,
    num_edges: int,
    dst_dim: int,
    J_override: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute phi for destination layer via fixed connections.

    Uses batched einsum for efficient computation across multiple connections.

    Args:
        topo: TopologyArrays containing connection data
        histories: List of layer state histories, each [B, T+1, D]
        start: Starting index into topology arrays for this layer's connections
        num_edges: Number of connections to process
        dst_dim: Destination layer dimension
        J_override: Optional override weights [E, D_max, F_max] for training

    Returns:
        phi_t: [B, T, D_dst] contribution from fixed connections
    """
    if num_edges <= 0:
        B = histories[0].shape[0]
        T = histories[0].shape[1] - 1
        return jnp.zeros((B, T, dst_dim), dtype=histories[0].dtype)

    B = histories[0].shape[0]
    T = histories[0].shape[1] - 1
    D_max = int(topo.edge_J.shape[1])
    F_max = int(topo.edge_J.shape[2])

    # Collect source histories and J matrices for batching
    s_list = []
    j_list = []
    end = start + num_edges

    for e in range(start, end):
        src_layer_idx = topo.edge_from_layer_idx_py[e]
        src_hist = histories[src_layer_idx][:, 1:, :]

        D_actual = topo.edge_dst_dims_py[e]
        F_actual = topo.edge_src_dims_py[e]

        # Extract J with override support
        if J_override is not None:
            J_base = J_override[e, :D_actual, :F_actual]
        else:
            J_base = topo.edge_J[e, :D_actual, :F_actual]
        mask = topo.edge_mask[e, :D_actual, :F_actual]
        J_eff = J_base * mask

        # Pad source history to F_max for batching
        if src_hist.shape[-1] < F_max:
            padding = F_max - src_hist.shape[-1]
            src_hist = jnp.pad(src_hist, ((0, 0), (0, 0), (0, padding)), mode="constant")
        elif src_hist.shape[-1] > F_max:
            src_hist = src_hist[:, :, :F_max]

        # Pad J_eff to [D_max, F_max] for batching
        J_padded = jnp.zeros((D_max, F_max), dtype=J_eff.dtype)
        if D_actual <= D_max and F_actual <= F_max:
            J_padded = J_padded.at[:D_actual, :F_actual].set(J_eff)

        s_list.append(src_hist)
        j_list.append(J_padded)

    # Batch using einsum
    phi_sum = jnp.zeros((B, T, D_max), dtype=histories[0].dtype)
    if s_list:
        s_stack = jnp.stack(s_list, axis=0)
        j_stack = jnp.stack(j_list, axis=0)
        phi_sum = jnp.einsum("kbtf,kdf->btd", s_stack, j_stack)

    return phi_sum[:, :, :dst_dim]


def stepwise_aggregate_fixed(
    topo: TopologyArrays,
    s_t_edges: jnp.ndarray,
    start: int,
    end: int,
    dst_dim: int,
    J_override: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Batched aggregation of fixed connections for one timestep.

    Args:
        topo: TopologyArrays
        s_t_edges: [E, B, F_max] source states for each edge
        start, end: Edge range [start:end]
        dst_dim: Destination layer dimension
        J_override: Optional override weights [E, D_max, F_max]

    Returns:
        phi_t: [B, D_dst] aggregated phi contribution
    """
    E = end - start
    if E <= 0:
        B = s_t_edges.shape[1] if s_t_edges.shape[0] > 0 else 1
        dtype = s_t_edges.dtype if s_t_edges.shape[0] > 0 else jnp.float32
        return jnp.zeros((B, dst_dim), dtype=dtype)

    # Prepare J matrix
    if J_override is not None:
        J_base = J_override
    else:
        J_base = jax.lax.dynamic_slice_in_dim(topo.edge_J, start, E, axis=0)

    # Apply mask
    mask = jax.lax.dynamic_slice_in_dim(topo.edge_mask, start, E, axis=0)
    J_eff = J_base * mask

    # Einsum: [E,B,F] x [E,D,F] -> [E,B,D]
    phi_e = jnp.einsum("ebf,edf->ebd", s_t_edges, J_eff)

    # Sum over edges
    phi_sum = jnp.sum(phi_e, axis=0)

    return phi_sum[:, :dst_dim]


__all__ = [
    "layerwise_aggregate_fixed",
    "stepwise_aggregate_fixed",
]

