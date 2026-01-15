"""Utility functions for working with custom weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_weights_to_npy(weights: torch.Tensor | np.ndarray, filename: str | Path) -> None:
    """Save weights to a .npy file.

    Args:
        weights: Weight tensor or numpy array with shape [to_nodes, from_nodes]
        filename: Path to save the .npy file

    Raises:
        ValueError: If weights shape is invalid

    """
    filename = Path(filename)

    if isinstance(weights, torch.Tensor):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = np.asarray(weights)

    # Validate shape
    if weights_np.ndim != 2:
        msg = f"Weights must be 2D, got shape {weights_np.shape}"
        raise ValueError(msg)

    # Create parent directory if it doesn't exist
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(filename), weights_np)


def save_weights_to_npz(weights: torch.Tensor | np.ndarray, filename: str | Path, key: str = "weights") -> None:
    """Save weights to a .npz file.

    Args:
        weights: Weight tensor or numpy array with shape [to_nodes, from_nodes]
        filename: Path to save the .npz file
        key: Key name to store weights under (default: "weights")

    Raises:
        ValueError: If weights shape is invalid

    """
    filename = Path(filename)

    if isinstance(weights, torch.Tensor):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = np.asarray(weights)

    # Validate shape
    if weights_np.ndim != 2:
        msg = f"Weights must be 2D, got shape {weights_np.ndim}D"
        raise ValueError(msg)

    # Create parent directory if it doesn't exist
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savez(str(filename), **{key: weights_np})


def validate_weight_shape(
    weights: torch.Tensor | np.ndarray,
    from_nodes: int,
    to_nodes: int,
) -> bool:
    """Validate that weight matrix matches connection dimensions.

    Args:
        weights: Weight tensor or numpy array
        from_nodes: Expected number of source nodes (columns)
        to_nodes: Expected number of target nodes (rows)

    Returns:
        True if shape matches, False otherwise

    """
    if isinstance(weights, torch.Tensor):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = np.asarray(weights)

    if weights_np.ndim != 2:
        return False

    expected_shape = (to_nodes, from_nodes)
    return weights_np.shape == expected_shape


def load_weights_from_file(
    filename: str | Path,
    from_nodes: int | None = None,
    to_nodes: int | None = None,
) -> np.ndarray:
    """Load weights from .npy or .npz file.

    Args:
        filename: Path to .npy or .npz file
        from_nodes: Expected number of source nodes (for validation)
        to_nodes: Expected number of target nodes (for validation)

    Returns:
        Weight matrix as numpy array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported or validation fails

    """
    filename = Path(filename)

    if not filename.exists():
        msg = f"Weights file not found: {filename}"
        raise FileNotFoundError(msg)

    if filename.suffix.lower() == ".npy":
        weights = np.load(str(filename))
    elif filename.suffix.lower() == ".npz":
        data = np.load(str(filename))
        if "weights" not in data:
            msg = f"No 'weights' key found in {filename}. Available keys: {list(data.keys())}"
            raise ValueError(msg)
        weights = data["weights"]
        data.close()
    else:
        msg = f"Unsupported file format: {filename.suffix}. Use .npy or .npz"
        raise ValueError(msg)

    # Validate shape if dimensions provided
    if from_nodes is not None and to_nodes is not None:
        if not validate_weight_shape(weights, from_nodes, to_nodes):
            msg = f"Weight shape {weights.shape} doesn't match expected shape [{to_nodes}, {from_nodes}]"
            raise ValueError(msg)

    return weights


__all__ = [
    "save_weights_to_npy",
    "save_weights_to_npz",
    "validate_weight_shape",
    "load_weights_from_file",
]
