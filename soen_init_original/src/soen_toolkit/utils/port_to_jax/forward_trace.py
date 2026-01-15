from __future__ import annotations

from typing import NamedTuple

import jax


class ForwardTrace(NamedTuple):
    """Optional telemetry returned by JAX forward passes.

    This is designed to be JIT-safe (pytree) and **opt-in** so existing call
    sites that expect `(final_history, histories_per_layer)` remain unchanged.

    Notes:
    - `layer_states` mirrors `histories_per_layer` (kept for convenience).
    - `phi_by_layer` is per-layer input φ history, aligned with each layer's
      dynamics input (shape `[B, T, D_layer]`).
    - `g_by_layer` is reserved for future use. It is intentionally optional and
      currently may be `None`.
    """

    layer_states: tuple[jax.Array, ...]
    phi_by_layer: tuple[jax.Array, ...] | None
    g_by_layer: tuple[jax.Array, ...] | None
    # Optional auxiliary final states for layers that carry extra recurrent state
    # (e.g., MultiplierNOCC's s1/s2). These are aligned with `layer_states` by
    # layer order and contain per-layer tensors of shape [B, D] or None.
    s1_final_by_layer: tuple[jax.Array | None, ...] | None = None
    s2_final_by_layer: tuple[jax.Array | None, ...] | None = None


__all__ = ["ForwardTrace"]

