"""Utilities for creating hierarchical and fractal connectivity masks.

This module provides functions to generate custom connectivity patterns with
natural power-law properties through hierarchical block structures.

Based on the hierarchical blocks approach where neurons are organized into
nested tiers, and connections are created between selected "interneurons"
at each tier level.
"""

import math

import numpy as np


def create_hierarchical_mask(levels: int = 3, base_size: int = 5, tier_fractions: list[float] | None = None) -> np.ndarray:
    """Create a hierarchical fractal connectivity mask.

    Uses a multi-tier approach where neurons are organized into nested blocks:
    - Tier 0 (base): blocks of size `base_size`
    - Tier 1: blocks of size `base_size^2`
    - Tier 2: blocks of size `base_size^3`
    - etc.

    At each tier, a fraction of neurons within each block are selected as
    "interneurons" and fully connected (all-to-all). This creates a fractal
    pattern where small local groups are densely connected, with progressively
    sparser connections at larger scales.

    Args:
        levels: Number of hierarchical tiers (1-4). Default: 3
            - levels=1: base_size nodes
            - levels=2: base_size² nodes
            - levels=3: base_size³ nodes
            - levels=4: base_size⁴ nodes
        base_size: Number of units per level in the hierarchy. Default: 5
        tier_fractions: Fraction of neurons to select at each tier [tier0, tier1, ...].
            If None, uses [1.0, 0.5, 0.25, 0.125] (decreasing connectivity at higher tiers).
            Length must equal `levels`.

    Returns:
        np.ndarray: Symmetric mask array of shape (total_nodes, total_nodes).
            Values are 1.0 (connection exists) or 0.0 (no connection).
            The mask is symmetric: mask[i,j] == mask[j,i].

    Example:
        >>> # Create 125-node network (3 tiers with base_size=5)
        >>> mask = create_hierarchical_mask(levels=3, base_size=5)
        >>> print(mask.shape)  # (125, 125)
        >>> print(f"Sparsity: {1 - np.mean(mask):.2%}")
        >>> # Verify symmetry
        >>> print(np.allclose(mask, mask.T))  # True

    Notes:
        - The mask is always symmetric (reciprocal connections)
        - Diagonal self-connections are included
        - Higher tier fractions create denser connectivity at larger scales
    """

    if levels < 1 or levels > 4:
        raise ValueError(f"levels must be between 1 and 4, got {levels}")
    if base_size < 2:
        raise ValueError(f"base_size must be at least 2, got {base_size}")

    # Default tier fractions: dense at base, sparser at higher tiers
    if tier_fractions is None:
        tier_fractions = [1.0, 0.5, 0.25, 0.125][:levels]

    if len(tier_fractions) != levels:
        raise ValueError(f"tier_fractions length ({len(tier_fractions)}) must equal levels ({levels})")

    # Compute total size: base_size^levels
    N = base_size**levels

    # Compute block sizes at each tier
    # tier_sizes[t] = base_size^(t+1)
    tier_sizes = [base_size ** (t + 1) for t in range(levels)]

    # Initialize mask
    mask = np.zeros((N, N), dtype=np.float32)

    # Process each tier from base (tier 0) to top
    for tier_idx in range(levels):
        tier_size = tier_sizes[tier_idx]
        tier_frac = tier_fractions[tier_idx]

        # Number of blocks at this tier
        num_blocks = N // tier_size

        # For each block at this tier
        for block_idx in range(num_blocks):
            block_start = block_idx * tier_size

            # Select interneurons within this block
            # At tier 0, select uniformly from ascending indices
            # At higher tiers, distribute selections across child blocks
            interneurons = _select_interneurons(block_start, tier_size, tier_frac, base_size if tier_idx > 0 else None)

            if len(interneurons) > 0:
                # Create all-to-all connections among selected interneurons
                # This creates a symmetric clique
                for i in interneurons:
                    for j in interneurons:
                        mask[i, j] = 1.0

    return mask


def _select_interneurons(start: int, size: int, fraction: float, num_children: int | None) -> list[int]:
    """Select interneurons within a block using split-among-children scheme.

    Args:
        start: Starting index of block
        size: Size of block
        fraction: Fraction of neurons to select
        num_children: Number of child blocks (None for base tier)

    Returns:
        List of selected neuron indices
    """
    if fraction <= 0:
        return []

    n_select = math.ceil(fraction * size)
    n_select = max(0, min(n_select, size))

    if n_select == 0:
        return []

    if num_children is None:
        # Base tier: select from ascending indices
        return list(range(start, start + n_select))

    # Higher tiers: distribute selections across children using "split_among_children"
    child_size = size // num_children

    # Distribute selections across children
    q, r = divmod(n_select, num_children)

    selected: list[int] = []
    for child_idx in range(num_children):
        # Number to take from this child
        take = q + (1 if child_idx < r else 0)
        take = min(take, child_size)

        child_start = start + child_idx * child_size
        selected.extend(range(child_start, child_start + take))

    return selected


def reorder_mask_for_visualization(mask: np.ndarray, levels: int, base_size: int) -> np.ndarray:
    """Reorder mask so hierarchical blocks appear as spatial squares in visualization.

    By default, nodes are indexed linearly (0, 1, 2, ..., N-1), which doesn't map
    well to spatial grid layouts. This function reorders the mask so that hierarchical
    blocks appear as contiguous square regions when visualized.

    Args:
        mask: Connectivity mask array (shape: N × N)
        levels: Number of hierarchical tiers used to create the mask
        base_size: Base size used to create the mask

    Returns:
        np.ndarray: Reordered mask where hierarchical blocks are spatially grouped

    Example:
        >>> mask = create_hierarchical_mask(levels=3, base_size=5)
        >>> mask_viz = reorder_mask_for_visualization(mask, levels=3, base_size=5)
        >>> # Now hierarchical blocks appear as visible squares in grid-of-grids

    Notes:
        - This reorders nodes so that they fill space in a block-recursive pattern
        - The reordered mask maintains all connections but changes node IDs
        - Use this version for visualization; use the original for actual computation
    """
    N = mask.shape[0]
    expected_N = base_size**levels

    if N != expected_N:
        raise ValueError(f"Mask size ({N}) doesn't match expected size for {levels} levels of base_size {base_size} (expected {expected_N})")

    # Create block-recursive permutation
    # Strategy: arrange nodes so that hierarchical blocks form spatial squares
    # For levels=3, base_size=5: we want 5×5 arrangement of (5×5 arrangements of 5 nodes)

    grid_width = base_size ** ((levels + 1) // 2)

    def get_hierarchical_position(node_id: int, levels: int, base_size: int) -> tuple[int, int]:
        """Get 2D position for a node using hierarchical block layout."""
        if levels == 1:
            # Base case: arrange in a line/square
            return (node_id // base_size, node_id % base_size)

        # Determine which top-level block this node belongs to
        block_size = base_size ** (levels - 1)
        top_block_idx = node_id // block_size
        within_block_id = node_id % block_size

        # Position of the top-level block
        blocks_per_row = base_size ** (((levels - 1) + 1) // 2)
        top_block_row = top_block_idx // blocks_per_row
        top_block_col = top_block_idx % blocks_per_row

        # Recursive position within the block
        sub_row, sub_col = get_hierarchical_position(within_block_id, levels - 1, base_size)

        # Scale for this level
        sub_grid_width = base_size ** ((levels - 1 + 1) // 2)

        # Combine
        row = top_block_row * sub_grid_width + sub_row
        col = top_block_col * sub_grid_width + sub_col

        return (row, col)

    # Build permutation: for each row-major grid position, determine which node should be there
    perm = np.zeros(N, dtype=np.int32)
    grid_to_node = {}

    for node_id in range(N):
        row, col = get_hierarchical_position(node_id, levels, base_size)
        linear_pos = row * grid_width + col
        grid_to_node[linear_pos] = node_id

    # Fill permutation in row-major order
    for linear_pos in range(N):
        if linear_pos in grid_to_node:
            perm[linear_pos] = grid_to_node[linear_pos]
        else:
            # Shouldn't happen if layout is correct, but handle gracefully
            perm[linear_pos] = linear_pos

    # Reorder mask
    return mask[perm][:, perm]


def analyze_mask(mask: np.ndarray) -> dict:
    """Analyze connectivity properties of a mask.

    Args:
        mask: Connectivity mask array (shape: N × N)

    Returns:
        Dictionary with connectivity statistics including:
        - n_nodes: Total number of nodes
        - n_connections: Total number of connections
        - connection_density: Fraction of possible connections present
        - sparsity: 1 - connection_density
        - avg_fan_out: Average number of outgoing connections per node
        - min_fan_out: Minimum fan-out
        - max_fan_out: Maximum fan-out
        - is_symmetric: Whether mask is symmetric (reciprocal connections)
    """
    n_nodes = mask.shape[0]
    n_connections = np.sum(mask > 0)
    sparsity = 1.0 - (n_connections / (n_nodes * n_nodes))

    # Compute fan-out distribution
    fan_outs = np.sum(mask > 0, axis=0)

    # Check symmetry
    is_symmetric = np.allclose(mask, mask.T)

    return {
        "n_nodes": n_nodes,
        "n_connections": int(n_connections),
        "connection_density": float(n_connections / (n_nodes * n_nodes)),
        "sparsity": float(sparsity),
        "avg_fan_out": float(np.mean(fan_outs)),
        "min_fan_out": int(np.min(fan_outs)),
        "max_fan_out": int(np.max(fan_outs)),
        "is_symmetric": bool(is_symmetric),
    }
