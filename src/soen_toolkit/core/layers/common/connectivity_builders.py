"""Connectivity builder utilities for layers.

- We need to add comments and docstrings here!

"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
import warnings

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _grid_coords(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D grid coordinates for nodes arranged in a square-ish grid.

    Layout matches visualize_grid_of_grids: nodes are arranged row-major with
    width = ceil(sqrt(n)), height = ceil(n/width).
    Node i is at position (col, row) where col = i % width, row = i // width.

    Args:
        num_nodes: Number of nodes to arrange

    Returns:
        xs: x-coordinates (columns, 0 to width-1)
        ys: y-coordinates (rows, 0 to height-1)

    """
    if num_nodes <= 0:
        return torch.empty(0), torch.empty(0)

    # Compute grid dimensions: width-first layout
    n_cols = math.ceil(math.sqrt(num_nodes))
    math.ceil(num_nodes / n_cols)

    # Simple vectorized computation: node i at (i % n_cols, i // n_cols)
    indices = torch.arange(num_nodes, dtype=torch.float32)
    xs = indices % n_cols
    ys = torch.floor(indices / n_cols)

    return xs, ys


def build_dense(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    return torch.ones(to_nodes, from_nodes, dtype=torch.float32)


def build_all_to_all(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    warnings.warn(
        "Connectivity type 'all_to_all' is deprecated and will be removed in a future release. Use 'dense' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_dense(from_nodes, to_nodes, params)


def build_one_to_one(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    """Build one-to-one (diagonal) connectivity.

    Creates diagonal connectivity mapping source nodes to target nodes.
    Optionally supports specifying a range of source nodes via source_start_node_id
    and source_end_node_id parameters.

    Args:
        from_nodes: Number of source nodes
        to_nodes: Number of target nodes
        params: Optional parameters:
            - source_start_node_id: Starting source node index (inclusive, default: 0)
            - source_end_node_id: Ending source node index (inclusive, default: from_nodes-1)

    Returns:
        Binary mask [to_nodes, from_nodes] with 1s on the diagonal mapping

    Raises:
        ValueError: If node range parameters are invalid

    Examples:
        Default behavior (diagonal from node 0):
        >>> build_one_to_one(5, 3, None)  # Connects nodes 0,1,2

        Specify source range:
        >>> build_one_to_one(10, 4, {"source_start_node_id": 0, "source_end_node_id": 3})
        >>> build_one_to_one(10, 6, {"source_start_node_id": 4, "source_end_node_id": 9})
    """
    matrix = torch.zeros(to_nodes, from_nodes, dtype=torch.float32)

    if params is None:
        params = {}

    source_start = params.get("source_start_node_id")
    source_end = params.get("source_end_node_id")

    # Check if range parameters are provided
    if source_start is not None or source_end is not None:
        # Both must be specified together
        if source_start is None or source_end is None:
            msg = "Both source_start_node_id and source_end_node_id must be specified together"
            raise ValueError(msg)

        source_start = int(source_start)
        source_end = int(source_end)

        # Validate range
        if source_start < 0:
            msg = f"source_start_node_id must be >= 0, got {source_start}"
            raise ValueError(msg)

        if source_end >= from_nodes:
            msg = f"source_end_node_id must be < from_nodes ({from_nodes}), got {source_end}"
            raise ValueError(msg)

        if source_start > source_end:
            msg = f"source_start_node_id ({source_start}) must be <= source_end_node_id ({source_end})"
            raise ValueError(msg)

        # Calculate range size
        range_size = source_end - source_start + 1

        # Validate target has enough nodes
        if range_size > to_nodes:
            msg = f"Source range size ({range_size}) exceeds target nodes ({to_nodes})"
            raise ValueError(msg)

        # Create diagonal mapping from source[start:end+1] to target[0:range_size]
        for i in range(range_size):
            matrix[i, source_start + i] = 1.0
    else:
        # Default behavior: diagonal from 0 up to min(from_nodes, to_nodes)
        diag = min(from_nodes, to_nodes)
        idx = torch.arange(diag)
        matrix[idx, idx] = 1.0

    return matrix


def build_inverse_one_to_one(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    """Build inverse one-to-one (all except diagonal) connectivity.

    Connects every source node to every target node EXCEPT the one with the matching index.
    Useful for lateral inhibition patterns.
    Currently requires from_nodes == to_nodes.

    Args:
        from_nodes: Number of source nodes
        to_nodes: Number of target nodes
        params: Optional parameters (none currently used)

    Returns:
        Binary mask [to_nodes, from_nodes] with 0s on the diagonal and 1s elsewhere.

    Raises:
        ValueError: If from_nodes != to_nodes
    """
    if from_nodes != to_nodes:
        msg = f"inverse_one_to_one requires equal input/output dimensions, got {from_nodes} -> {to_nodes}"
        raise ValueError(msg)

    matrix = torch.ones(to_nodes, from_nodes, dtype=torch.float32)
    matrix.fill_diagonal_(0.0)
    return matrix


def build_chain(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    matrix = torch.zeros(to_nodes, from_nodes, dtype=torch.float32)
    if from_nodes <= 1 or to_nodes == 0:
        return matrix

    sources = torch.arange(from_nodes - 1)
    targets = sources + 1
    valid = targets < to_nodes
    if torch.any(valid):
        matrix[targets[valid], sources[valid]] = 1.0
    return matrix


def build_sparse(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    if not params or "sparsity" not in params:
        msg = "sparse requires 'sparsity' parameter"
        raise ValueError(msg)
    p = float(params["sparsity"])
    if not (0 < p <= 1):
        msg = "sparsity must be in (0,1]"
        raise ValueError(msg)
    return (torch.rand(to_nodes, from_nodes) < p).float()


def create_deterministic_pattern(height: int, width: int, density: float) -> torch.Tensor:
    pattern = torch.zeros(height, width, dtype=torch.float32)
    density = float(max(0.0, min(1.0, density)))
    if density <= 0.0:
        return pattern

    total_connections = height * width
    target_connections = round(total_connections * density)
    if target_connections <= 0:
        return pattern

    stride = max(1, total_connections // target_connections)
    count = 0
    for i in range(height):
        for j in range(width):
            if (i * width + j) % stride == 0 and count < target_connections:
                pattern[i, j] = 1.0
                count += 1
    return pattern


def build_block_structure(from_nodes: int, to_nodes: int, params: Mapping[str, Any]) -> torch.Tensor:
    block_count = int(params["block_count"])
    connection_mode = str(params["connection_mode"])
    within_block_density = float(params["within_block_density"])
    cross_block_density = float(params["cross_block_density"])

    from_block_size = from_nodes // max(1, block_count)
    to_block_size = to_nodes // max(1, block_count)

    if from_block_size < 1 or to_block_size < 1:
        block_count = min(from_nodes, to_nodes)
        from_block_size = max(1, from_nodes // block_count)
        to_block_size = max(1, to_nodes // block_count)

    mask = torch.zeros(to_nodes, from_nodes, dtype=torch.float32)

    for i in range(block_count):
        to_start = i * to_block_size
        to_end = min((i + 1) * to_block_size, to_nodes)

        from_start = i * from_block_size
        from_end = min((i + 1) * from_block_size, from_nodes)

        if within_block_density > 0:
            block_mask = create_deterministic_pattern(
                to_end - to_start,
                from_end - from_start,
                within_block_density,
            )
            mask[to_start:to_end, from_start:from_end] = block_mask

        if cross_block_density > 0 and connection_mode != "none":
            if connection_mode == "full":
                targets = range(block_count)
            else:
                targets = range(i + 1, block_count)
            for j in targets:
                if j == i:
                    continue
                cross_to_start = j * to_block_size
                cross_to_end = min((j + 1) * to_block_size, to_nodes)
                cross_mask = create_deterministic_pattern(
                    cross_to_end - cross_to_start,
                    from_end - from_start,
                    cross_block_density,
                )
                mask[cross_to_start:cross_to_end, from_start:from_end] = cross_mask

                if connection_mode == "full":
                    cross_from_start = j * from_block_size
                    cross_from_end = min((j + 1) * from_block_size, from_nodes)
                    symmetric_mask = create_deterministic_pattern(
                        to_end - to_start,
                        cross_from_end - cross_from_start,
                        cross_block_density,
                    )
                    mask[to_start:to_end, cross_from_start:cross_from_end] = symmetric_mask

    return mask


def build_constant(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    if params is None:
        params = {}
    if "expected_fan_out" not in params:
        msg = "constant requires 'expected_fan_out' parameter"
        raise ValueError(msg)
    k = int(params.get("expected_fan_out", 0))
    k = max(0, k)
    allow_self = bool(params.get("allow_self_connections", True))

    mask = torch.zeros(to_nodes, from_nodes, dtype=torch.float32)
    if k == 0:
        return mask

    for j in range(from_nodes):
        available = list(range(to_nodes))
        if not allow_self and j < to_nodes:
            available.remove(j)
        if not available:
            continue
        chosen = torch.randperm(len(available))[: min(k, len(available))]
        tgt_idx = [available[i] for i in chosen]
        mask[tgt_idx, j] = 1.0
    return mask


def build_power_law(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    """Build power-law connectivity based on grid distance.

    Connection probability ~ 1 / (distance ** alpha)
    Samples expected_fan_out connections per source node based on distance-weighted probabilities.

    Args:
        from_nodes: Number of source nodes
        to_nodes: Number of target nodes
        params: Dict with keys:
            - expected_fan_out (required): Average number of connections per source node
            - alpha: Power-law exponent (default: 2.0). Higher = stronger distance bias
            - connection_scope: "internal" (intra-layer, use grid) or "external" (inter-layer)
            - allow_self_connections: Whether to allow i->i connections (default: True)

    Returns:
        Binary mask [to_nodes, from_nodes] where 1 indicates a connection

    """
    if params is None:
        params = {}
    if "expected_fan_out" not in params:
        msg = "power_law requires 'expected_fan_out' parameter"
        raise ValueError(msg)

    alpha = float(params.get("alpha", 2.0))
    k = max(0, int(params["expected_fan_out"]))
    scope = str(params.get("connection_scope", "internal")).lower()
    allow_self = bool(params.get("allow_self_connections", True))

    if k == 0:
        return torch.zeros(to_nodes, from_nodes, dtype=torch.float32)

    # Compute distance matrix based on scope
    if scope == "internal":
        # Intra-layer: both source and target use same grid layout
        sx, sy = _grid_coords(from_nodes)
        tx, ty = _grid_coords(to_nodes)
        dx = tx.view(-1, 1) - sx.view(1, -1)
        dy = ty.view(-1, 1) - sy.view(1, -1)
        D = torch.sqrt(dx * dx + dy * dy)
    else:
        # Inter-layer: arrange layers side-by-side (from on left, to on right)
        from_cols = math.ceil(math.sqrt(from_nodes))
        to_cols = math.ceil(math.sqrt(to_nodes))
        horizontal_offset = float(max(from_cols, to_cols))

        sx = torch.zeros(from_nodes)
        sy = torch.linspace(0.0, max(0.0, float(from_nodes - 1)), from_nodes)
        tx = torch.full((to_nodes,), horizontal_offset)
        ty = torch.linspace(0.0, max(0.0, float(to_nodes - 1)), to_nodes)

        dx = tx.view(-1, 1) - sx.view(1, -1)
        dy = torch.abs(ty.view(-1, 1) - sy.view(1, -1))
        D = dx.abs() + dy

    # Convert distances to connection weights: weight ~ 1 / distance^alpha
    W = torch.clamp(D, min=1e-6)  # Avoid division by zero
    W = 1.0 / (W**alpha)

    # Optionally exclude self-connections
    if (to_nodes == from_nodes) and (scope == "internal") and not allow_self:
        idx = torch.arange(to_nodes)
        W[idx, idx] = 0.0

    # Sample k connections per source node based on weight probabilities
    mask = torch.zeros(to_nodes, from_nodes, dtype=torch.float32)
    for j in range(from_nodes):
        col = W[:, j]
        positive = col > 0
        num_avail = int(positive.sum().item())
        if num_avail == 0:
            continue

        kj = min(k, num_avail)
        probs = col[positive]
        probs = probs / probs.sum()
        chosen = torch.multinomial(probs, num_samples=kj, replacement=False)
        tgt_idx = positive.nonzero(as_tuple=False).view(-1)[chosen]
        mask[tgt_idx, j] = 1.0

    return mask


def build_exponential(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    """Build exponential distance-decay connectivity based on grid distance.

    Connection probability ~ exp(-distance / d_0)
    Samples expected_fan_out connections per source node based on distance-weighted probabilities.

    Args:
        from_nodes: Number of source nodes
        to_nodes: Number of target nodes
        params: Dict with keys:
            - expected_fan_out (required): Average number of connections per source node
            - d_0: Decay length scale (default: 2.0). Larger = weaker distance bias
            - connection_scope: "internal" (intra-layer, use grid) or "external" (inter-layer)
            - allow_self_connections: Whether to allow i->i connections (default: True)

    Returns:
        Binary mask [to_nodes, from_nodes] where 1 indicates a connection

    """
    if params is None:
        params = {}
    if "expected_fan_out" not in params:
        msg = "exponential requires 'expected_fan_out' parameter"
        raise ValueError(msg)

    k = max(0, int(params["expected_fan_out"]))
    scope = str(params.get("connection_scope", "internal")).lower()
    d0 = float(params.get("d_0", 2.0))
    allow_self = bool(params.get("allow_self_connections", True))

    if k == 0:
        return torch.zeros(to_nodes, from_nodes, dtype=torch.float32)

    # Compute distance matrix based on scope
    if scope == "internal":
        # Intra-layer: both source and target use same grid layout
        sx, sy = _grid_coords(from_nodes)
        tx, ty = _grid_coords(to_nodes)
        dx = tx.view(-1, 1) - sx.view(1, -1)
        dy = ty.view(-1, 1) - sy.view(1, -1)
        D = torch.sqrt(dx * dx + dy * dy)
    else:
        # Inter-layer: arrange layers side-by-side (from on left, to on right)
        from_cols = math.ceil(math.sqrt(from_nodes))
        to_cols = math.ceil(math.sqrt(to_nodes))
        horizontal_offset = float(max(from_cols, to_cols))

        sx = torch.zeros(from_nodes)
        sy = torch.linspace(0.0, max(0.0, float(from_nodes - 1)), from_nodes)
        tx = torch.full((to_nodes,), horizontal_offset)
        ty = torch.linspace(0.0, max(0.0, float(to_nodes - 1)), to_nodes)

        dx = tx.view(-1, 1) - sx.view(1, -1)
        dy = torch.abs(ty.view(-1, 1) - sy.view(1, -1))
        D = dx.abs() + dy

    # Convert distances to connection weights: weight ~ exp(-distance / d_0)
    W = torch.exp(-D / max(d0, 1e-6))

    # Optionally exclude self-connections
    if (to_nodes == from_nodes) and (scope == "internal") and not allow_self:
        idx = torch.arange(to_nodes)
        W[idx, idx] = 0.0

    # Sample k connections per source node based on weight probabilities
    mask = torch.zeros(to_nodes, from_nodes, dtype=torch.float32)
    for j in range(from_nodes):
        col = W[:, j]
        positive = col > 0
        num_avail = int(positive.sum().item())
        if num_avail == 0:
            continue

        kj = min(k, num_avail)
        probs = col[positive]
        probs = probs / probs.sum()
        chosen = torch.multinomial(probs, num_samples=kj, replacement=False)
        tgt_idx = positive.nonzero(as_tuple=False).view(-1)[chosen]
        mask[tgt_idx, j] = 1.0

    return mask


def build_custom(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    """Load custom connectivity mask from .npz file.

    Allows users to provide their own connectivity patterns generated externally.
    The mask defines connectivity structure (0 = no connection, 1 = connection).
    Weights are initialized separately using the chosen initialization method.

    The mask must be stored with key "mask" in the .npz file.

    Args:
        from_nodes: Number of source nodes
        to_nodes: Number of target nodes
        params: Dict with keys:
            - mask_file (required): Path to .npz file containing mask array

    Returns:
        Binary mask tensor [to_nodes, from_nodes] with 0/1 values

    Raises:
        ValueError: If file not found, invalid format, wrong shape, or invalid values

    """
    import os

    import numpy as np

    if params is None:
        msg = "custom connectivity requires 'mask_file' parameter"
        raise ValueError(msg)

    # Get file path - support "mask_file" (and legacy "file_path" for backwards compatibility)
    file_path = params.get("mask_file") or params.get("file_path")
    if not file_path:
        msg = "custom connectivity requires 'mask_file' parameter"
        raise ValueError(msg)

    file_path = str(file_path)

    # Validate file exists
    if not os.path.exists(file_path):
        msg = f"Mask file not found: {file_path}"
        raise ValueError(msg)

    # Always use "mask" as the key
    npz_key = "mask"

    # Load .npz file
    try:
        npz_data = np.load(file_path)
    except Exception as e:
        msg = f"Failed to load .npz file '{file_path}': {e}"
        raise ValueError(msg) from e

    # Extract mask array
    if npz_key not in npz_data:
        available_keys = list(npz_data.keys())
        msg = f"Key '{npz_key}' not found in .npz file. Available keys: {available_keys}"
        raise ValueError(
            msg,
        )

    mask_array = npz_data[npz_key]
    npz_data.close()

    # Validate shape
    if mask_array.ndim != 2:
        msg = f"Mask array must be 2D, got shape {mask_array.shape} (ndim={mask_array.ndim})"
        raise ValueError(
            msg,
        )

    expected_shape = (to_nodes, from_nodes)
    if mask_array.shape != expected_shape:
        msg = f"Mask shape mismatch. Expected {expected_shape} (to_nodes, from_nodes), got {mask_array.shape}"
        raise ValueError(
            msg,
        )

    # Convert to tensor
    mask = torch.from_numpy(mask_array).float()

    # Validate values are binary (0 or 1)
    unique_vals = torch.unique(mask)
    if not torch.all((unique_vals == 0) | (unique_vals == 1)):
        msg = f"Mask must contain only 0 or 1 values. Found unique values: {unique_vals.tolist()}"
        raise ValueError(
            msg,
        )

    return mask


def build_hierarchical_blocks(from_nodes: int, to_nodes: int, params: Mapping[str, Any] | None = None) -> torch.Tensor:
    """Build hierarchical block structure connectivity.

    Creates nested multi-tier blocks with interneuron connectivity at each scale.
    Uses the hierarchical mask algorithm where neurons are organized into nested
    blocks and a fraction of neurons at each tier are selected as "interneurons"
    and fully connected.

    Args:
        from_nodes: Number of source nodes
        to_nodes: Number of target nodes
        params: Dict with keys:
            - levels (required): Number of hierarchical tiers (1-4)
            - base_size (required): Number of units per level in hierarchy (minimum 2)
            - tier_fractions (optional): List of fractions for each tier [tier0, tier1, ...]
                                        If not provided, uses [1.0, 0.5, 0.25, 0.125]

    Returns:
        Binary mask tensor [to_nodes, from_nodes] with symmetric hierarchical structure

    Raises:
        ValueError: If non-square, dimension mismatch, or invalid parameters

    """
    from soen_toolkit.core.utils.hierarchical_mask import create_hierarchical_mask

    if params is None:
        msg = "hierarchical_blocks requires 'levels' and 'base_size' parameters"
        raise ValueError(msg)

    # Validate square connectivity
    if from_nodes != to_nodes:
        msg = f"hierarchical_blocks requires square connectivity (from_nodes must equal to_nodes), got from_nodes={from_nodes}, to_nodes={to_nodes}"
        raise ValueError(msg)

    # Extract parameters
    levels = params.get("levels")
    base_size = params.get("base_size")
    tier_fractions = params.get("tier_fractions")

    if levels is None:
        msg = "hierarchical_blocks requires 'levels' parameter"
        raise ValueError(msg)
    if base_size is None:
        msg = "hierarchical_blocks requires 'base_size' parameter"
        raise ValueError(msg)

    levels = int(levels)
    base_size = int(base_size)

    # Validate and process tier_fractions
    if tier_fractions is not None:
        # Handle case where tier_fractions might be a single value or needs conversion
        if not isinstance(tier_fractions, list):
            # If it's a single float/int, convert to list
            # This can happen when loaded from YAML or passed from GUI
            tier_fractions = None  # Use default instead of trying to convert
        elif len(tier_fractions) != levels:
            msg = f"tier_fractions length ({len(tier_fractions)}) must equal levels ({levels})"
            raise ValueError(msg)

    # Validate dimension matches expected size
    expected_nodes = base_size ** levels
    if from_nodes != expected_nodes:
        msg = f"hierarchical_blocks requires from_nodes to equal base_size^levels (expected {expected_nodes} for base_size={base_size}, levels={levels}), got {from_nodes}"
        raise ValueError(msg)

    # Create hierarchical mask (this will validate levels, base_size, tier_fractions)
    mask_array = create_hierarchical_mask(
        levels=levels,
        base_size=base_size,
        tier_fractions=tier_fractions,
    )

    # Convert to torch tensor
    mask = torch.from_numpy(mask_array).float()

    return mask


CONNECTIVITY_BUILDERS: dict[str, Callable[[int, int, Mapping[str, Any] | None], torch.Tensor]] = {
    "dense": build_dense,
    "all_to_all": build_all_to_all,
    "one_to_one": build_one_to_one,
    "inverse_one_to_one": build_inverse_one_to_one,
    "chain": build_chain,
    "sparse": build_sparse,
    "block_structure": build_block_structure,  # type: ignore[dict-item]
    "constant": build_constant,
    "power_law": build_power_law,
    "exponential": build_exponential,
    "custom": build_custom,
    "hierarchical_blocks": build_hierarchical_blocks,
}


def build_connectivity(
    kind: str,
    *,
    from_nodes: int,
    to_nodes: int,
    params: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    try:
        builder = CONNECTIVITY_BUILDERS[kind]
    except KeyError as exc:
        msg = f"Unknown connectivity builder '{kind}'"
        raise ValueError(msg) from exc
    return builder(from_nodes, to_nodes, params)


__all__ = [
    "CONNECTIVITY_BUILDERS",
    "build_all_to_all",
    "build_chain",
    "build_connectivity",
    "build_constant",
    "build_custom",
    "build_dense",
    "build_exponential",
    "build_hierarchical_blocks",
    "build_one_to_one",
    "build_inverse_one_to_one",
    "build_power_law",
    "build_sparse",
]
