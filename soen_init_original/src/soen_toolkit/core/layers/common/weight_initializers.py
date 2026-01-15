"""Custom weight loading from files.

Allows users to provide pre-trained or custom weights from .npy or .npz files.
Weights are applied directly during connection initialization.
"""

from __future__ import annotations

import os

import numpy as np
import torch


def init_custom_weights(
    from_nodes: int,
    to_nodes: int,
    mask: torch.Tensor,
    *,
    weights_file: str = "",
) -> torch.Tensor:
    """Load custom weights from .npy or .npz file.

    Args:
        from_nodes: Number of source nodes (expected weight columns)
        to_nodes: Number of target nodes (expected weight rows)
        mask: Binary connectivity mask [to_nodes, from_nodes]
        weights_file: Path to weights file (.npy or .npz)

    Returns:
        Weight tensor [to_nodes, from_nodes] with custom values applied to connected positions

    Raises:
        ValueError: If file not found, invalid format, wrong shape, or invalid values
    """
    file_path_str = weights_file
    if not file_path_str:
        raise ValueError("init_custom_weights requires 'weights_file' parameter")

    file_path_str = str(file_path_str)

    # Validate file exists
    if not os.path.exists(file_path_str):
        raise ValueError(f"Weights file not found: {file_path_str}")

    # Load array
    if file_path_str.lower().endswith(".npy"):
        try:
            weights_array = np.load(file_path_str)
        except Exception as e:
            raise ValueError(f"Failed to load .npy file '{file_path_str}': {e}") from e
    elif file_path_str.lower().endswith(".npz"):
        try:
            npz_data = np.load(file_path_str)
            npz_key = "weights"
            if npz_key not in npz_data:
                available_keys = list(npz_data.keys())
                raise ValueError(f"Key '{npz_key}' not found in .npz file. Available keys: {available_keys}")
            weights_array = npz_data[npz_key]
            npz_data.close()
        except Exception as e:
            raise ValueError(f"Failed to load .npz file '{file_path_str}': {e}") from e
    else:
        raise ValueError(f"Unsupported file format. Use .npy or .npz: {file_path_str}")

    # Validate shape
    if weights_array.ndim != 2:
        raise ValueError(f"Weights array must be 2D [to_nodes, from_nodes], got shape {weights_array.shape}")
    if weights_array.shape != (to_nodes, from_nodes):
        raise ValueError(f"Weights shape mismatch. Expected {(to_nodes, from_nodes)}, got {weights_array.shape}")

    # Apply mask
    weights_tensor = torch.from_numpy(weights_array).float()
    masked_weights = weights_tensor * mask.float()
    return masked_weights
