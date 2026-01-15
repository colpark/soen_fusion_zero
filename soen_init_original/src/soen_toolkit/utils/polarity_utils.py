"""Utilities for generating neuron polarity arrays.

Neuron polarity arrays enforce excitatory/inhibitory constraints on outgoing connections.
Values:
  - 1: Excitatory (all outgoing weights >= 0)
  - -1: Inhibitory (all outgoing weights <= 0)
  - 0: Normal (unrestricted, any sign)

Polarity enforcement methods control how weights are adjusted at initialization:
  - sign_flip: Preserve magnitude, flip sign to match polarity (abs for excitatory, -abs for inhibitory)
  - clip_to_zero: Clip violating weights to zero (original behavior)
"""

from __future__ import annotations

import numpy as np
import torch

# Polarity enforcement method constants
POLARITY_ENFORCEMENT_SIGN_FLIP = "sign_flip"
POLARITY_ENFORCEMENT_CLIP = "clip_to_zero"
POLARITY_ENFORCEMENT_DEFAULT = POLARITY_ENFORCEMENT_SIGN_FLIP


def generate_alternating_polarity(num_neurons: int) -> np.ndarray:
    """Generate 50:50 alternating excitatory/inhibitory polarity.

    Pattern: [1, -1, 1, -1, ...] (excitatory first)

    Args:
        num_neurons: Number of neurons

    Returns:
        Array of shape [num_neurons] with values alternating between 1 and -1

    Examples:
        >>> polarity = generate_alternating_polarity(4)
        >>> polarity
        array([ 1, -1,  1, -1], dtype=int8)
    """
    polarity = np.ones(num_neurons, dtype=np.int8)
    polarity[1::2] = -1  # Set odd indices to -1
    return polarity


def generate_excitatory_polarity(num_neurons: int) -> np.ndarray:
    """Generate pure excitatory polarity (all +1).

    Args:
        num_neurons: Number of neurons

    Returns:
        Array of shape [num_neurons] with all values = 1
    """
    return np.ones(num_neurons, dtype=np.int8)


def generate_inhibitory_polarity(num_neurons: int) -> np.ndarray:
    """Generate pure inhibitory polarity (all -1).

    Args:
        num_neurons: Number of neurons

    Returns:
        Array of shape [num_neurons] with all values = -1
    """
    return np.full(num_neurons, -1, dtype=np.int8)


def save_polarity(polarity: np.ndarray, filepath: str) -> None:
    """Save polarity array to .npy file.

    Args:
        polarity: Polarity array with values -1, 0, or 1
        filepath: Path to save the .npy file

    Examples:
        >>> polarity = generate_alternating_polarity(100)
        >>> save_polarity(polarity, "my_polarity.npy")
    """
    np.save(filepath, polarity.astype(np.int8))


def generate_random_polarity(
    num_neurons: int,
    excitatory_ratio: float = 0.8,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random polarity with specified excitatory ratio.

    Args:
        num_neurons: Number of neurons
        excitatory_ratio: Fraction of excitatory neurons (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Array of shape [num_neurons] with random polarity

    Examples:
        >>> polarity = generate_random_polarity(100, excitatory_ratio=0.8, seed=42)
        >>> (polarity == 1).sum()  # ~80 excitatory neurons
        80
    """
    if not 0.0 <= excitatory_ratio <= 1.0:
        msg = f"excitatory_ratio must be between 0.0 and 1.0, got {excitatory_ratio}"
        raise ValueError(msg)

    if seed is not None:
        np.random.seed(seed)

    polarity = np.ones(num_neurons, dtype=np.int8)
    n_inhibitory = int(num_neurons * (1 - excitatory_ratio))

    if n_inhibitory > 0:
        inhib_indices = np.random.choice(num_neurons, size=n_inhibitory, replace=False)
        polarity[inhib_indices] = -1

    return polarity


def apply_polarity_enforcement(
    weight: torch.Tensor,
    polarity: torch.Tensor,
    method: str = POLARITY_ENFORCEMENT_DEFAULT,
) -> torch.Tensor:
    """Apply polarity enforcement to weight matrix at initialization time.

    This function modifies weights based on the source neuron polarity. The weight
    matrix has shape [to_nodes, from_nodes], where each column corresponds to the
    outgoing weights from a source neuron.

    Args:
        weight: Weight matrix of shape [to_nodes, from_nodes]
        polarity: Polarity array of shape [from_nodes] with values -1, 0, or 1
        method: Enforcement method - either "sign_flip" or "clip_to_zero"

    Returns:
        Modified weight matrix with polarity enforcement applied

    Raises:
        ValueError: If method is not recognized or polarity length doesn't match

    Examples:
        >>> weight = torch.tensor([[-0.5, 0.3], [0.2, -0.4]])
        >>> polarity = torch.tensor([1, -1])  # first excitatory, second inhibitory
        >>> result = apply_polarity_enforcement(weight, polarity, "sign_flip")
        >>> result  # first column becomes abs, second becomes -abs
        tensor([[ 0.5000, -0.3000],
                [ 0.2000, -0.4000]])
    """
    if len(polarity) != weight.shape[1]:
        msg = f"Polarity length {len(polarity)} != weight columns {weight.shape[1]}"
        raise ValueError(msg)

    if method not in {POLARITY_ENFORCEMENT_SIGN_FLIP, POLARITY_ENFORCEMENT_CLIP}:
        msg = f"Unknown polarity enforcement method: {method}. Use '{POLARITY_ENFORCEMENT_SIGN_FLIP}' or '{POLARITY_ENFORCEMENT_CLIP}'"
        raise ValueError(msg)

    # Make a copy to avoid modifying in place
    result = weight.clone()

    if method == POLARITY_ENFORCEMENT_SIGN_FLIP:
        # Sign flip: preserve magnitude, adjust sign to match polarity
        for src in range(weight.shape[1]):
            if polarity[src] == 1:  # Excitatory: all outgoing weights >= 0
                result[:, src] = torch.abs(weight[:, src])
            elif polarity[src] == -1:  # Inhibitory: all outgoing weights <= 0
                result[:, src] = -torch.abs(weight[:, src])
            # polarity[src] == 0: unrestricted, no change

    elif method == POLARITY_ENFORCEMENT_CLIP:
        # Clip to zero: clip violating weights to zero
        for src in range(weight.shape[1]):
            if polarity[src] == 1:  # Excitatory: clip negative to zero
                result[:, src] = torch.clamp(weight[:, src], min=0.0)
            elif polarity[src] == -1:  # Inhibitory: clip positive to zero
                result[:, src] = torch.clamp(weight[:, src], max=0.0)
            # polarity[src] == 0: unrestricted, no change

    return result


__all__ = [
    "POLARITY_ENFORCEMENT_SIGN_FLIP",
    "POLARITY_ENFORCEMENT_CLIP",
    "POLARITY_ENFORCEMENT_DEFAULT",
    "apply_polarity_enforcement",
    "generate_alternating_polarity",
    "generate_excitatory_polarity",
    "generate_inhibitory_polarity",
    "generate_random_polarity",
    "save_polarity",
]

