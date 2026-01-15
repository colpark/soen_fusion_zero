"""Classification-specific functionality for state trajectory visualization.

This module handles multi-sample concatenation with state carryover for
seq2static classification tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClassificationResult:
    """Results from multi-sample classification run."""

    # Concatenated data
    input_features: torch.Tensor  # [1, total_T, D]
    metric_histories: list  # Per-layer histories
    raw_state_histories: list  # Per-layer state histories

    # Per-sample metadata
    sample_classes: list[int]  # Ground truth class for each sample
    sample_boundaries: list[int]  # Time indices where each sample starts
    predictions: list[int]  # Model prediction for each sample
    correct: list[bool]  # Whether each prediction was correct


def apply_time_pooling(states: torch.Tensor, method: str) -> torch.Tensor:
    """Apply time pooling to states to get single vector.

    Args:
        states: State tensor [B, T, D] or [T, D]
        method: "max" or "final"

    Returns:
        Pooled tensor [B, D] or [D]
    """
    if method == "max":
        # Max pooling over time dimension
        if states.ndim == 3:
            return torch.max(states, dim=1)[0]  # [B, D]
        else:
            return torch.max(states, dim=0)[0]  # [D]
    elif method == "final":
        # Take final timestep
        if states.ndim == 3:
            return states[:, -1, :]  # [B, D]
        else:
            return states[-1, :]  # [D]
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def get_prediction(output_states: torch.Tensor, pooling_method: str) -> int:
    """Get classification prediction from output layer states.

    Args:
        output_states: Output layer states [1, T+1, num_classes]
        pooling_method: "max" or "final"

    Returns:
        Predicted class index
    """
    # Apply time pooling
    pooled = apply_time_pooling(output_states, pooling_method)  # [1, num_classes]

    # Get predicted class (argmax)
    pred_class = torch.argmax(pooled, dim=-1).item()
    return pred_class


def concatenate_histories(histories_list: list[list]) -> list:
    """Concatenate histories from multiple samples along time dimension.

    Args:
        histories_list: List of per-sample histories, where each is a list of
                       per-layer tensors [1, T+1, D]

    Returns:
        List of concatenated per-layer histories [1, total_T+1, D]
    """
    if not histories_list:
        return []

    num_layers = len(histories_list[0])
    concatenated = []

    for layer_idx in range(num_layers):
        # Gather all samples for this layer
        layer_samples = [histories[layer_idx] for histories in histories_list]

        # Concatenate along time dimension (dim=1)
        # Each tensor is [1, T+1, D], result is [1, total_T+1, D]
        concat = torch.cat(layer_samples, dim=1)
        concatenated.append(concat)

    return concatenated


def extract_final_states(
    histories: list[torch.Tensor],
    layers_config: list | None = None,
) -> dict[int, torch.Tensor]:
    """Extract final states from all layers.

    Args:
        histories: Per-layer state histories [1, T+1, D], ordered by layers_config
        layers_config: Optional layer configurations to map index to layer_id.
                      If None, uses list index as layer_id (legacy behavior).

    Returns:
        Dict mapping layer_id to final state [1, 1, D]
    """
    final_states = {}
    for idx, hist in enumerate(histories):
        if hist is not None:
            # Get proper layer_id from config, or fall back to index
            if layers_config is not None and idx < len(layers_config):
                layer_id = layers_config[idx].layer_id
            else:
                layer_id = idx
            # Extract final timestep and clone to detach from original history
            final_states[layer_id] = hist[:, -1:, :].detach().clone()

    return final_states

