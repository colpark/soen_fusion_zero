"""Utilities for autoregressive training with SOEN models.

This module provides helper functions for:
- Multi-timestep autoregressive training
- Token-level time pooling
- Dataset preparation for AR tasks
- Target construction for next-token prediction
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

logger = logging.getLogger(__name__)


def pool_token_timesteps(
    outputs: torch.Tensor,
    time_steps_per_token: int,
    pooling_method: str = "final",
    pooling_params: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Pool model outputs within each token's timesteps for multi-timestep AR.

    In multi-timestep autoregressive training, each token is processed over multiple
    simulation timesteps to allow the model's dynamics to settle. This function pools
    those timesteps to produce a single prediction per token.

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

    Example:
        >>> outputs = torch.randn(2, 256, 26)  # 64 tokens × 4 timesteps
        >>> pooled = pool_token_timesteps(outputs, 4, "final")
        >>> pooled.shape
        torch.Size([2, 64, 26])

    Raises:
        ValueError: If pooling_method is unknown or params are invalid
    """
    if pooling_params is None:
        pooling_params = {}

    batch_size, total_steps, vocab_size = outputs.shape

    # Validate time_steps_per_token
    if time_steps_per_token <= 0:
        raise ValueError(f"time_steps_per_token must be positive, got {time_steps_per_token}")

    if time_steps_per_token == 1:
        # No pooling needed
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
        logger.warning(
            f"Truncating outputs from {total_steps} to {usable_steps} timesteps "
            f"to fit {num_tokens} complete tokens with {time_steps_per_token} steps each"
        )
        outputs = outputs[:, :usable_steps, :]

    # Reshape to [batch, num_tokens, time_steps_per_token, vocab_size]
    reshaped = outputs.reshape(batch_size, num_tokens, time_steps_per_token, vocab_size)

    # Apply pooling method
    if pooling_method == "final":
        # Use last timestep of each token
        pooled = reshaped[:, :, -1, :]  # [batch, num_tokens, vocab]

    elif pooling_method == "mean":
        # Average over all timesteps
        pooled = reshaped.mean(dim=2)

    elif pooling_method == "max":
        # Max over timesteps
        pooled = reshaped.max(dim=2)[0]

    elif pooling_method == "mean_last_n":
        # Average last N timesteps
        n = pooling_params.get("n", 2)
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"mean_last_n requires positive integer 'n', got {n}")
        if n > time_steps_per_token:
            raise ValueError(
                f"mean_last_n n={n} exceeds time_steps_per_token={time_steps_per_token}"
            )
        pooled = reshaped[:, :, -n:, :].mean(dim=2)

    else:
        raise ValueError(
            f"Unknown pooling method: '{pooling_method}'. "
            f"Supported: 'final', 'mean', 'max', 'mean_last_n'"
        )

    return pooled


def prepare_multistep_ar_dataset(
    sequences: NDArray[Any],
    time_steps_per_token: int,
) -> NDArray[Any]:
    """Prepare dataset for multi-timestep autoregressive training.

    Duplicates each token's input across multiple timesteps to allow the model's
    dynamics to settle before making a prediction.

    Args:
        sequences: Input sequences [N, seq_len, vocab_size] (one-hot encoded)
                  or [N, seq_len] (token indices)
        time_steps_per_token: How many timesteps to duplicate each token

    Returns:
        Extended sequences with duplicated tokens:
        - If input is [N, seq_len, vocab_size]: returns [N, seq_len * time_steps_per_token, vocab_size]
        - If input is [N, seq_len]: returns [N, seq_len * time_steps_per_token]

    Example:
        >>> # One-hot encoded
        >>> seq = np.eye(26)[np.array([[7, 8]])]  # "hi" -> [1, 2, 26]
        >>> extended = prepare_multistep_ar_dataset(seq, 4)
        >>> extended.shape
        (1, 8, 26)  # Each token duplicated 4 times

        >>> # Token indices
        >>> seq = np.array([[7, 8]])  # [1, 2]
        >>> extended = prepare_multistep_ar_dataset(seq, 4)
        >>> extended.shape
        (1, 8)  # [7, 7, 7, 7, 8, 8, 8, 8]

    Raises:
        ValueError: If time_steps_per_token is not positive
    """
    if time_steps_per_token <= 0:
        raise ValueError(f"time_steps_per_token must be positive, got {time_steps_per_token}")

    if time_steps_per_token == 1:
        # No duplication needed
        return sequences

    # Repeat each token along the sequence dimension
    # np.repeat works on any axis
    if sequences.ndim == 3:
        # One-hot: [N, seq_len, vocab] -> [N, seq_len * k, vocab]
        extended = np.repeat(sequences, time_steps_per_token, axis=1)
    elif sequences.ndim == 2:
        # Token indices: [N, seq_len] -> [N, seq_len * k]
        extended = np.repeat(sequences, time_steps_per_token, axis=1)
    else:
        raise ValueError(
            f"Expected 2D or 3D sequences, got shape {sequences.shape}"
        )

    return extended


def build_multistep_ar_targets(
    input_sequence: torch.Tensor,
    time_steps_per_token: int = 1,
) -> torch.Tensor:
    """Build targets for multi-timestep autoregressive training.

    For multi-timestep AR, targets are still at the token level (not timestep level).
    This function is identical to standard AR target construction because we only
    need one target per token, regardless of how many timesteps we use.

    Args:
        input_sequence: Token indices [batch, seq_len] extracted from inputs
        time_steps_per_token: Timesteps per token (for documentation, not used)

    Returns:
        Target sequence [batch, seq_len] where targets[i] = input_sequence[i+1]

    Example:
        >>> tokens = torch.tensor([[7, 8, 11, 11, 14]])  # "hello"
        >>> targets = build_multistep_ar_targets(tokens)
        >>> targets
        tensor([[8, 11, 11, 14, 14]])  # Shifted by 1

    Note:
        This is a convenience wrapper. The actual target construction logic
        is the same as standard AR (shift by 1 token).
    """
    # Standard AR target construction (shift by 1)
    batch_size, seq_len = input_sequence.shape

    # Shift: take tokens 1 through end
    shifted = input_sequence[:, 1:]  # [batch, seq_len-1]

    # For the last position, use the last token from the input
    last_token = input_sequence[:, -1:]  # [batch, 1]

    # Concatenate to get full target sequence
    targets = torch.cat([shifted, last_token], dim=1)  # [batch, seq_len]

    return targets


def validate_ar_config(
    time_steps_per_token: int,
    token_pooling_method: str,
    token_pooling_params: dict[str, Any] | None = None,
) -> None:
    """Validate autoregressive configuration parameters.

    Args:
        time_steps_per_token: Number of timesteps per token
        token_pooling_method: Pooling method name
        token_pooling_params: Pooling parameters

    Raises:
        ValueError: If configuration is invalid
    """
    if time_steps_per_token <= 0:
        raise ValueError(f"time_steps_per_token must be positive, got {time_steps_per_token}")

    valid_methods = {"final", "mean", "max", "mean_last_n"}
    if token_pooling_method not in valid_methods:
        raise ValueError(
            f"Invalid token_pooling_method: '{token_pooling_method}'. "
            f"Valid options: {', '.join(sorted(valid_methods))}"
        )

    if token_pooling_params is None:
        token_pooling_params = {}

    # Validate method-specific params
    if token_pooling_method == "mean_last_n":
        n = token_pooling_params.get("n")
        if n is None:
            raise ValueError("mean_last_n requires 'n' parameter")
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"mean_last_n 'n' must be positive integer, got {n}")
        if n > time_steps_per_token:
            raise ValueError(
                f"mean_last_n n={n} cannot exceed time_steps_per_token={time_steps_per_token}"
            )


def get_ar_sequence_length(
    dataset_seq_len: int,
    time_steps_per_token: int,
) -> int:
    """Calculate the number of tokens from a multi-timestep sequence length.

    Args:
        dataset_seq_len: Sequence length in the dataset (total timesteps)
        time_steps_per_token: Timesteps per token

    Returns:
        Number of complete tokens

    Example:
        >>> get_ar_sequence_length(256, 4)
        64  # 256 timesteps / 4 per token = 64 tokens
    """
    if time_steps_per_token <= 0:
        raise ValueError(f"time_steps_per_token must be positive, got {time_steps_per_token}")

    return dataset_seq_len // time_steps_per_token
