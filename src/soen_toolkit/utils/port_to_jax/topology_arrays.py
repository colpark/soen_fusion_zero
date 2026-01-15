from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

EdgeMode = Literal["fixed", "wicc", "nocc"]


@dataclass
class TopologyArrays:
    """Static, padded, batched view of a JAXModel graph.

    Shapes:
      - layer_order: [L]
      - inbound_starts: [L+1] CSR index into E for each layer's inbound edges
      - edge_from_layer_idx: [E] indices into layer_order for each edge's source layer
      - edge_J: [E, D_max, F_max] connection matrices, padded
      - edge_mask: [E, D_max, F_max] binary mask (1.0 = enabled)
      - edge_dst_dims: [E] true destination dims per edge
      - edge_src_dims: [E] true source dims per edge
      - edge_mode: [E] int code (0=fixed, 1=wicc, 2=nocc)
      - edge_j_in: [E, F_max] scaling for source states per edge
      - edge_j_out: [E, D_max] scaling for aggregated outputs per edge
      - edge_half_flux_offset: [E] bool flag
      - wicc_gamma_plus: [E, D_max, F_max] only valid where mode==wicc
      - wicc_gamma_minus: [E, D_max, F_max] only valid where mode==wicc
      - wicc_bias_current: [E, D_max, F_max] only valid where mode==wicc
      - nocc_alpha: [E, D_max, F_max] only valid where mode==nocc
      - nocc_beta: [E, D_max, F_max]
      - nocc_beta_out: [E, D_max, F_max]
      - nocc_bias_current: [E, D_max, F_max]
      - layer_dims: [L]
      - input_type: str ("flux"|"state")
      - network_evaluation_method: str
    """

    layer_order: tuple[int, ...]
    inbound_starts: jnp.ndarray
    inbound_starts_py: tuple[int, ...]
    inbound_counts: tuple[int, ...]
    edge_from_layer_idx: jnp.ndarray
    edge_from_layer_idx_py: tuple[int, ...]
    edge_J: jnp.ndarray
    edge_mask: jnp.ndarray
    edge_dst_dims: jnp.ndarray
    edge_dst_dims_py: tuple[int, ...]
    edge_src_dims: jnp.ndarray
    edge_src_dims_py: tuple[int, ...]
    edge_mode: jnp.ndarray
    edge_mode_py: tuple[int, ...]
    edge_j_in: jnp.ndarray
    edge_j_out: jnp.ndarray
    edge_half_flux_offset: jnp.ndarray
    wicc_gamma_plus: jnp.ndarray
    wicc_gamma_minus: jnp.ndarray
    wicc_bias_current: jnp.ndarray
    nocc_alpha: jnp.ndarray
    nocc_beta: jnp.ndarray
    nocc_beta_out: jnp.ndarray
    nocc_bias_current: jnp.ndarray
    layer_dims: jnp.ndarray
    layer_dims_py: tuple[int, ...]
    input_type: str
    network_evaluation_method: str
    dt: float


def _mode_category(mode_str: str | None) -> EdgeMode:
    m = str(mode_str or "fixed").lower()
    if m in {"wicc", "dynamic", "multiplier", "programmable", "dynamic_v1", "v1"}:
        return "wicc"
    if m in {"nocc", "dynamic_v2", "multiplier_v2", "v2"}:
        return "nocc"
    return "fixed"


def build_topology_arrays(jax_model) -> TopologyArrays:
    """Precompute a padded, array-friendly view of a JAXModel graph.

    This does not change behavior; it only materializes the static structure
    and per-edge parameters into padded arrays for fast batched kernels.
    """
    from .jax_model import ConnectionSpec

    layers_sorted = sorted(jax_model.layers, key=lambda s: s.layer_id)
    layer_order: list[int] = [int(s.layer_id) for s in layers_sorted]
    lid_to_idx = {lid: i for i, lid in enumerate(layer_order)}

    # Inbound edges per layer
    inbound_by_to: dict[int, list] = {s.layer_id: [] for s in layers_sorted}
    for c in jax_model.connections:
        # Validate that we have ConnectionSpec objects, not ConnectionConfig
        if not isinstance(c, ConnectionSpec):
            msg = f"Expected ConnectionSpec object, got {type(c).__name__}. This indicates a bug in model conversion."
            raise TypeError(msg)
        inbound_by_to[c.to_layer].append(c)

    # Flatten edges in CSR order
    inbound_starts: list[int] = [0]
    flat_edges: list = []
    for s in layers_sorted:
        edges = inbound_by_to[s.layer_id]
        flat_edges.extend(edges)
        inbound_starts.append(len(flat_edges))

    E = len(flat_edges)
    D_max = max([s.dim for s in layers_sorted] + [1])
    F_max = 1
    for e in flat_edges:
        d, f = e.J.shape
        D_max = max(D_max, d)
        F_max = max(F_max, f)

    # Allocate 2D parameter arrays
    edge_J = jnp.zeros((E, D_max, F_max), dtype=jnp.float32)
    edge_mask = jnp.ones((E, D_max, F_max), dtype=jnp.float32)
    edge_dst_dims = jnp.zeros((E,), dtype=jnp.int32)
    edge_src_dims = jnp.zeros((E,), dtype=jnp.int32)
    edge_from_layer_idx = jnp.zeros((E,), dtype=jnp.int32)

    edge_mode = jnp.zeros((E,), dtype=jnp.int32)  # 0=fixed, 1=wicc, 2=nocc
    # Parameters are now 3D: [E, D_max, F_max] for WICC/NOCC (per-edge parameters)
    # j_in/j_out are 2D: [E, F_max] and [E, D_max]
    edge_j_in = jnp.ones((E, F_max), dtype=jnp.float32)
    edge_j_out = jnp.ones((E, D_max), dtype=jnp.float32)
    edge_half_flux_offset = jnp.zeros((E,), dtype=jnp.int32)

    wicc_gamma_plus = jnp.zeros((E, D_max, F_max), dtype=jnp.float32)
    wicc_gamma_minus = jnp.zeros((E, D_max, F_max), dtype=jnp.float32)
    wicc_bias_current = jnp.zeros((E, D_max, F_max), dtype=jnp.float32)
    nocc_alpha = jnp.zeros((E, D_max, F_max), dtype=jnp.float32)
    nocc_beta = jnp.zeros((E, D_max, F_max), dtype=jnp.float32)
    nocc_beta_out = jnp.zeros((E, D_max, F_max), dtype=jnp.float32)
    nocc_bias_current = jnp.zeros((E, D_max, F_max), dtype=jnp.float32)

    def _broadcast_param(val, target_shape: tuple[int, ...], param_name: str) -> jnp.ndarray:
        """Broadcast a parameter value to target shape.

        Args:
            val: Scalar, 1D array, or 2D array
            target_shape: Target shape (D,) or (D, F)
            param_name: Parameter name for error messages

        Returns:
            Broadcasted array of target_shape
        """
        arr = jnp.asarray(val, dtype=jnp.float32)

        # Scalar case
        if arr.ndim == 0:
            return jnp.full(target_shape, float(arr), dtype=jnp.float32)

        # Already correct shape
        if arr.shape == target_shape:
            return arr

        # 1D array - broadcast to 2D if needed
        if arr.ndim == 1 and len(target_shape) == 1:
            if arr.shape[0] == target_shape[0]:
                return arr
            raise ValueError(f"Parameter {param_name} shape {arr.shape} doesn't match target {target_shape}")

        # 1D to 2D: broadcast along appropriate axis
        if arr.ndim == 1 and len(target_shape) == 2:
            d, f = target_shape
            if arr.shape[0] == d:
                # [D] -> [D, F]
                return jnp.broadcast_to(arr[:, None], (d, f))
            elif arr.shape[0] == f:
                # [F] -> [D, F]
                return jnp.broadcast_to(arr[None, :], (d, f))
            raise ValueError(f"Parameter {param_name} shape {arr.shape} doesn't match target {target_shape}")

        raise ValueError(f"Cannot broadcast {param_name} with shape {arr.shape} to {target_shape}")

    # Fill
    for i, c in enumerate(flat_edges):
        d, f = c.J.shape
        edge_J = edge_J.at[i, :d, :f].set(c.J)
        if getattr(c, "mask", None) is not None:
            m = (c.mask != 0).astype(c.J.dtype)
            edge_mask = edge_mask.at[i, :d, :f].set(m)
        edge_dst_dims = edge_dst_dims.at[i].set(d)
        edge_src_dims = edge_src_dims.at[i].set(f)
        edge_from_layer_idx = edge_from_layer_idx.at[i].set(lid_to_idx[int(c.from_layer)])

        mode = _mode_category(getattr(c, "mode", "fixed"))
        if mode == "wicc":
            edge_mode = edge_mode.at[i].set(1)
            # Broadcast WICC parameters to [D, F]
            gp = _broadcast_param(getattr(c, "gamma_plus", 1e-3), (d, f), "gamma_plus")
            gm = _broadcast_param(getattr(c, "gamma_minus", 1e-3), (d, f), "gamma_minus")
            bc = _broadcast_param(getattr(c, "bias_current", 2.0), (d, f), "bias_current")
            wicc_gamma_plus = wicc_gamma_plus.at[i, :d, :f].set(gp)
            wicc_gamma_minus = wicc_gamma_minus.at[i, :d, :f].set(gm)
            wicc_bias_current = wicc_bias_current.at[i, :d, :f].set(bc)
        elif mode == "nocc":
            edge_mode = edge_mode.at[i].set(2)
            # Broadcast NOCC parameters to [D, F]
            alpha = _broadcast_param(getattr(c, "alpha", 1.64053), (d, f), "alpha")
            beta = _broadcast_param(getattr(c, "beta", 303.85), (d, f), "beta")
            beta_out = _broadcast_param(getattr(c, "beta_out", 91.156), (d, f), "beta_out")
            bc = _broadcast_param(getattr(c, "bias_current", 2.1), (d, f), "bias_current")
            nocc_alpha = nocc_alpha.at[i, :d, :f].set(alpha)
            nocc_beta = nocc_beta.at[i, :d, :f].set(beta)
            nocc_beta_out = nocc_beta_out.at[i, :d, :f].set(beta_out)
            nocc_bias_current = nocc_bias_current.at[i, :d, :f].set(bc)

        # Broadcast j_in and j_out
        j_in_bc = _broadcast_param(getattr(c, "j_in", 1.0), (f,), "j_in")
        j_out_bc = _broadcast_param(getattr(c, "j_out", 1.0), (d,), "j_out")
        edge_j_in = edge_j_in.at[i, :f].set(j_in_bc)
        edge_j_out = edge_j_out.at[i, :d].set(j_out_bc)


    layer_dims = jnp.asarray([int(s.dim) for s in layers_sorted], dtype=jnp.int32)
    layer_dims_tuple = tuple(int(v) for v in layer_dims.tolist())
    inbound_starts_tuple = tuple(int(v) for v in inbound_starts)
    inbound_counts = tuple(inbound_starts[i + 1] - inbound_starts[i] for i in range(len(inbound_starts) - 1))
    edge_from_layer_idx_tuple = tuple(int(v) for v in edge_from_layer_idx.tolist())
    edge_dst_dims_tuple = tuple(int(v) for v in edge_dst_dims.tolist())
    edge_src_dims_tuple = tuple(int(v) for v in edge_src_dims.tolist())
    edge_mode_tuple = tuple(int(v) for v in edge_mode.tolist())

    topo = TopologyArrays(
        layer_order=tuple(layer_order),
        inbound_starts=jnp.asarray(inbound_starts, dtype=jnp.int32),
        inbound_starts_py=inbound_starts_tuple,
        inbound_counts=inbound_counts,
        edge_from_layer_idx=edge_from_layer_idx,
        edge_from_layer_idx_py=edge_from_layer_idx_tuple,
        edge_J=edge_J,
        edge_mask=edge_mask,
        edge_dst_dims=edge_dst_dims,
        edge_dst_dims_py=edge_dst_dims_tuple,
        edge_src_dims=edge_src_dims,
        edge_src_dims_py=edge_src_dims_tuple,
        edge_mode=edge_mode,
        edge_mode_py=edge_mode_tuple,
        edge_j_in=edge_j_in,
        edge_j_out=edge_j_out,
        edge_half_flux_offset=edge_half_flux_offset,
        wicc_gamma_plus=wicc_gamma_plus,
        wicc_gamma_minus=wicc_gamma_minus,
        wicc_bias_current=wicc_bias_current,
        nocc_alpha=nocc_alpha,
        nocc_beta=nocc_beta,
        nocc_beta_out=nocc_beta_out,
        nocc_bias_current=nocc_bias_current,
        layer_dims=layer_dims,
        layer_dims_py=layer_dims_tuple,
        input_type=str(getattr(jax_model, "input_type", "flux")),
        network_evaluation_method=str(getattr(jax_model, "network_evaluation_method", "layerwise")),
        dt=float(getattr(jax_model, "dt", 1.0)),
    )
    return topo


__all__ = ["TopologyArrays", "build_topology_arrays"]
