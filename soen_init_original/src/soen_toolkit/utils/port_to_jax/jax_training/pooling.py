from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def apply_time_pooling(
    sequence_logits: jnp.ndarray,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    range_start: int | None = None,
    range_end: int | None = None,
) -> jnp.ndarray:
    """Apply the configured time pooling strategy to a sequence of logits.

    Args:
        sequence_logits: Tensor shaped [B, T, C]
        method: Pooling method name (max, mean, rms, mean_range, final, mean_last_n, ewa)
        params: Optional dictionary of method-specific parameters
        range_start: Optional override for mean_range start index (matches Torch behaviour)
        range_end: Optional override for mean_range end index (matches Torch behaviour)

    Returns:
        Tensor shaped [B, C]

    """
    if sequence_logits.ndim != 3:
        # Note: Don't format shape directly as it may be a tracer during JIT
        msg = "Expected sequence_logits with 3 dims [B, T, C]; got different number of dimensions"
        raise ValueError(
            msg,
        )

    total_timesteps = sequence_logits.shape[1]
    if total_timesteps <= 0:
        msg = "sequence_logits must contain at least one timestep"
        raise ValueError(msg)

    cfg: dict[str, Any] = dict(params or {})
    scale = float(cfg.get("scale", 1.0))
    method_key = (method or "max").lower()

    if method_key == "max":
        pooled = jnp.max(sequence_logits, axis=1)

    elif method_key == "mean":
        pooled = jnp.mean(sequence_logits, axis=1)

    elif method_key == "rms":
        pooled = jnp.sqrt(jnp.mean(jnp.square(sequence_logits), axis=1) + 1e-8)

    elif method_key == "mean_range":
        default_points = int(min(50, total_timesteps))
        start_idx = int(range_start) if range_start is not None else max(0, total_timesteps - default_points)
        end_idx = int(range_end) if range_end is not None else int(total_timesteps)
        start_idx = max(0, min(start_idx, total_timesteps - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_timesteps))
        if start_idx >= end_idx:
            pooled = jnp.mean(sequence_logits, axis=1)
        else:
            pooled = jnp.mean(sequence_logits[:, start_idx:end_idx, :], axis=1)

    elif method_key == "final":
        pooled = sequence_logits[:, -1, :]

    elif method_key == "mean_last_n":
        n = int(cfg.get("n", 1))
        n = max(1, min(n, total_timesteps))
        pooled = jnp.mean(sequence_logits[:, -n:, :], axis=1)

    elif method_key == "ewa":
        min_weight = float(cfg.get("min_weight", 0.2))
        min_weight = float(jnp.clip(min_weight, 1e-6, 1.0))
        if total_timesteps == 1:
            pooled = sequence_logits[:, 0, :]
        else:
            t = jnp.linspace(0.0, 1.0, num=total_timesteps, dtype=sequence_logits.dtype)
            weights = min_weight ** (1.0 - t)
            weights = weights / jnp.sum(weights)
            pooled = jnp.einsum("btc,t->bc", sequence_logits, weights)

    else:
        msg = f"Unknown time pooling method '{method}'. Supported: max, mean, rms, mean_range, final, mean_last_n, ewa"
        raise ValueError(
            msg,
        )

    if scale != 1.0:
        pooled = pooled * scale

    return pooled


def max_over_time(sequence_logits: jnp.ndarray) -> jnp.ndarray:
    """Backward-compatible helper for legacy imports."""
    return apply_time_pooling(sequence_logits, "max")
