"""Utilities for autoregressive training with SOEN models (JAX backend).

This module provides JAX-compatible helper functions for:
- Multi-timestep autoregressive training
- Token-level time pooling
- Target construction for next-token prediction
"""

import logging
from typing import Any

import jax.numpy as jnp

logger = logging.getLogger(__name__)


def pool_token_timesteps_jax(
    outputs: jnp.ndarray,
    time_steps_per_token: int,
    pooling_method: str = "final",
    pooling_params: dict[str, Any] | None = None,
) -> jnp.ndarray:
    """Pool model outputs within each token's timesteps for multi-timestep AR (JAX).

    Args:
        outputs: Model outputs [batch, total_timesteps, vocab_size]
                 where total_timesteps = num_tokens * time_steps_per_token
        time_steps_per_token: Number of simulation timesteps per token
        pooling_method: How to pool timesteps within each token:
                       - "final": Use last timestep (default)
                       - "mean": Average all timesteps
                       - "max": Max over timesteps
                       - "mean_last_n": Average last N timesteps
        pooling_params: Method-specific parameters (e.g., {"n": 2} for mean_last_n)

    Returns:
        Pooled outputs [batch, num_tokens, vocab_size]
    """
    if pooling_params is None:
        pooling_params = {}

    batch_size, total_steps, vocab_size = outputs.shape

    if time_steps_per_token <= 0:
        raise ValueError(f"time_steps_per_token must be positive, got {time_steps_per_token}")

    if time_steps_per_token == 1:
        return outputs

    # Calculate number of complete tokens
    num_tokens = total_steps // time_steps_per_token

    if num_tokens == 0:
        raise ValueError(
            f"Not enough timesteps for even one token. "
            f"total_steps={total_steps}, time_steps_per_token={time_steps_per_token}"
        )

    # Truncate to complete tokens only
    usable_steps = num_tokens * time_steps_per_token
    if usable_steps < total_steps:
        # Note: JAX slice must be static for JIT if shapes change,
        # but here we assume shapes are consistent within a batch or handled by padding
        outputs = outputs[:, :usable_steps, :]

    # Reshape to [batch, num_tokens, time_steps_per_token, vocab_size]
    reshaped = outputs.reshape(batch_size, num_tokens, time_steps_per_token, vocab_size)

    # Apply pooling method
    if pooling_method == "final":
        # Use last timestep of each token
        pooled = reshaped[:, :, -1, :]

    elif pooling_method == "mean":
        # Average all timesteps
        pooled = jnp.mean(reshaped, axis=2)

    elif pooling_method == "max":
        # Max over timesteps
        pooled = jnp.max(reshaped, axis=2)

    elif pooling_method == "mean_last_n":
        # Average last N timesteps
        n = pooling_params.get("n", 2)
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"mean_last_n requires positive integer 'n', got {n}")
        if n > time_steps_per_token:
            raise ValueError(
                f"mean_last_n n={n} exceeds time_steps_per_token={time_steps_per_token}"
            )
        # Slice last n: [..., -n:, ...]
        # Note: dynamic slicing in JAX can be tricky, but negative indices usually work
        pooled = jnp.mean(reshaped[:, :, -n:, :], axis=2)

    else:
        raise ValueError(f"Unknown pooling method: '{pooling_method}'")

    return pooled


def build_ar_targets_jax(
    input_sequence: jnp.ndarray,
    time_steps_per_token: int = 1,
) -> jnp.ndarray:
    """Build targets for multi-timestep autoregressive training (JAX).

    Args:
        input_sequence: Token indices [batch, seq_len]
        time_steps_per_token: Unused, kept for API consistency

    Returns:
        Target sequence [batch, seq_len] where targets[i] = input_sequence[i+1]
    """
    # Shift: take tokens 1 through end
    shifted = input_sequence[:, 1:]  # [batch, seq_len-1]

    # For the last position, use the last token (or pad, but reusing last is common)
    last_token = input_sequence[:, -1:]  # [batch, 1]

    # Concatenate
    targets = jnp.concatenate([shifted, last_token], axis=1)

    return targets


def validate_ar_config_jax(
    time_steps_per_token: int,
    token_pooling_method: str,
    token_pooling_params: dict[str, Any] | None = None,
) -> None:
    """Validate AR config (JAX version)."""
    if time_steps_per_token <= 0:
        raise ValueError(f"time_steps_per_token must be positive, got {time_steps_per_token}")

    valid_methods = {"final", "mean", "max", "mean_last_n"}
    if token_pooling_method not in valid_methods:
        raise ValueError(f"Invalid token_pooling_method: '{token_pooling_method}'")
