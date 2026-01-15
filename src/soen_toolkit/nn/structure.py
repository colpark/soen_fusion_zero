"""Connection structure/topology helpers.

These functions return StructureSpec objects that can be passed to Graph.connect().
They map to the existing connectivity builders in the core toolkit.
"""

from .specs import StructureSpec


def dense(allow_self_connections: bool = True) -> StructureSpec:
    """Dense (all-to-all) connectivity.

    Args:
        allow_self_connections: Whether to allow self-connections (only for internal)

    Returns:
        StructureSpec for dense connectivity

    """
    return StructureSpec(
        type="dense",
        params={},
        allow_self_connections=allow_self_connections,
    )


def one_to_one(
    source_start_node_id: int | None = None,
    source_end_node_id: int | None = None,
) -> StructureSpec:
    """One-to-one (diagonal) connectivity.

    Creates diagonal connectivity mapping source nodes to target nodes.
    By default, connects from node 0. Optionally specify a source node range
    to connect a subset of source nodes.

    Args:
        source_start_node_id: Starting source node index (inclusive).
            If specified, source_end_node_id must also be specified.
        source_end_node_id: Ending source node index (inclusive).
            If specified, source_start_node_id must also be specified.

    Returns:
        StructureSpec for one-to-one connectivity

    Examples:
        Default behavior (connects from node 0):
        >>> structure.one_to_one()

        Connect nodes 0-3 to a 4-node target:
        >>> structure.one_to_one(source_start_node_id=0, source_end_node_id=3)

        Connect nodes 4-9 to a 6-node target:
        >>> structure.one_to_one(source_start_node_id=4, source_end_node_id=9)

    """
    params = {}
    if source_start_node_id is not None:
        params["source_start_node_id"] = source_start_node_id
    if source_end_node_id is not None:
        params["source_end_node_id"] = source_end_node_id

    return StructureSpec(type="one_to_one", params=params)


def inverse_one_to_one() -> StructureSpec:
    """Inverse one-to-one (all except diagonal) connectivity.

    Connects every source node to every target node EXCEPT the one with the matching index.
    Useful for lateral inhibition. Requires from_nodes == to_nodes.

    Returns:
        StructureSpec for inverse one-to-one connectivity

    Examples:
        >>> structure.inverse_one_to_one()
    """
    return StructureSpec(type="inverse_one_to_one", params={})


def sparse(sparsity: float, allow_self_connections: bool = True) -> StructureSpec:
    """Sparse connectivity with Bernoulli sampling.

    Args:
        sparsity: Connection probability between any two nodes (0-1)
        allow_self_connections: Whether to allow self-connections (only for internal)

    Returns:
        StructureSpec for sparse connectivity

    """
    return StructureSpec(
        type="sparse",
        params={"sparsity": sparsity},
        allow_self_connections=allow_self_connections,
    )


def block_structure(
    block_count: int,
    connection_mode: str = "diagonal",
    within_block_density: float = 1.0,
    cross_block_density: float = 0.0,
    allow_self_connections: bool = True,
) -> StructureSpec:
    """Block-based connectivity.

    Args:
        block_count: Number of blocks
        connection_mode: 'diagonal' (matching blocks) or 'full' (all block pairs)
        within_block_density: Fraction of edges within each block (0-1)
        cross_block_density: Fraction of edges across blocks (0-1)
        allow_self_connections: Whether to allow self-connections (only for internal)

    Returns:
        StructureSpec for block structure connectivity

    """
    return StructureSpec(
        type="block_structure",
        params={
            "block_count": block_count,
            "connection_mode": connection_mode,
            "within_block_density": within_block_density,
            "cross_block_density": cross_block_density,
        },
        allow_self_connections=allow_self_connections,
    )


def power_law(alpha: float, expected_fan_out: int, allow_self_connections: bool = True) -> StructureSpec:
    """Distance-biased connectivity with power-law decay.

    Nodes are arranged on a conceptual grid
    connection probability decreases
    with distance as d^(-alpha).

    Args:
        alpha: Decay exponent (higher = more local)
        expected_fan_out: Number of targets sampled per source node
        allow_self_connections: Whether to allow self-connections (only for internal)

    Returns:
        StructureSpec for power law connectivity

    """
    return StructureSpec(
        type="power_law",
        params={"alpha": alpha, "expected_fan_out": expected_fan_out},
        allow_self_connections=allow_self_connections,
    )


def exponential(d_0: float, expected_fan_out: int, allow_self_connections: bool = True) -> StructureSpec:
    """Distance-biased connectivity with exponential decay.

    Args:
        d_0: Characteristic length scale for decay
        expected_fan_out: Number of targets sampled per source node
        allow_self_connections: Whether to allow self-connections (only for internal)

    Returns:
        StructureSpec for exponential decay connectivity

    """
    return StructureSpec(
        type="exponential",
        params={"d_0": d_0, "expected_fan_out": expected_fan_out},
        allow_self_connections=allow_self_connections,
    )


def constant_fan_out(expected_fan_out: int, allow_self_connections: bool = True) -> StructureSpec:
    """Uniform sampling of a fixed number of targets per source node.

    Args:
        expected_fan_out: Number of targets sampled per source node
        allow_self_connections: Whether to allow self-connections (only for internal)

    Returns:
        StructureSpec for constant fan-out connectivity

    """
    return StructureSpec(
        type="constant",
        params={"expected_fan_out": expected_fan_out},
        allow_self_connections=allow_self_connections,
    )


def custom(
    mask_file: str,
    allow_self_connections: bool = True,
    visualization_metadata: dict | None = None,
) -> StructureSpec:
    """Load custom connectivity from .npz file.

    Allows you to provide connectivity patterns generated by your own code.
    The mask defines connectivity structure (0 = no connection, 1 = connection).
    Weights are initialized separately using the chosen initialization method.

    The mask must be stored with key "mask" in the .npz file.

    Args:
        mask_file: Path to .npz file containing binary mask array (0/1 values)
        allow_self_connections: Whether to allow self-connections (only for internal)
        visualization_metadata: Optional metadata dict for visualization purposes.
            This metadata is stored with the connection and can be used by visualization
            tools to customize rendering. Common uses:
            - Hierarchical structure: {"hierarchical": {"levels": 3, "base_size": 4}}
            - Custom layouts: {"layout": {"type": "custom", "params": {...}}}
            Any key-value pairs can be stored for future visualization features.

    Returns:
        StructureSpec for custom file-based connectivity

    Example:
        >>> import numpy as np
        >>> # Create custom mask
        >>> mask = np.zeros((5, 10), dtype=np.float32)
        >>> mask[0:3, 0] = 1.0  # Connect source 0 to destinations 0,1,2
        >>> np.savez("my_mask.npz", mask=mask)
        >>>
        >>> # Use in model
        >>> g.connect(0, 1,
        ...           structure=structure.custom("my_mask.npz"),
        ...           init=init.xavier_uniform())
        >>>
        >>> # With hierarchical structure metadata for visualization
        >>> from soen_toolkit.core.utils.hierarchical_mask import create_hierarchical_mask
        >>> mask = create_hierarchical_mask(levels=3, base_size=4, seed=42)
        >>> np.savez("hierarchical_mask.npz", mask=mask)
        >>> g.connect(0, 0,
        ...           structure=structure.custom(
        ...               "hierarchical_mask.npz",
        ...               visualization_metadata={"hierarchical": {"levels": 3, "base_size": 4}}
        ...           ),
        ...           init=init.constant(0.1))

    """
    return StructureSpec(
        type="custom",
        params={"mask_file": mask_file},
        allow_self_connections=allow_self_connections,
        visualization_metadata=visualization_metadata,
    )


def hierarchical_blocks(
    levels: int = 3,
    base_size: int = 4,
    tier_fractions: list[float] | None = None,
    allow_self_connections: bool = True,
) -> StructureSpec:
    """Hierarchical block structure with multi-tier interneuron connectivity.

    Creates nested multi-tier blocks where neurons are organized into hierarchical
    structures. At each tier, a fraction of neurons are selected as "interneurons"
    and fully connected. This creates a fractal pattern with dense local connectivity
    and progressively sparser connections at larger scales.

    The layer dimension must equal base_size^levels. This connectivity type only
    works for internal (recurrent) connections where from_nodes == to_nodes.

    Args:
        levels: Number of hierarchical tiers (1-4). Total nodes = base_size^levels.
            - levels=1: base_size nodes
            - levels=2: base_size² nodes
            - levels=3: base_size³ nodes
            - levels=4: base_size⁴ nodes
        base_size: Number of units per level in the hierarchy (minimum 2)
        tier_fractions: Fraction of neurons to select at each tier [tier0, tier1, ...].
            If None, uses defaults [1.0, 0.5, 0.25, 0.125] (decreasing connectivity
            at higher tiers). Length must equal levels.
        allow_self_connections: Whether to allow self-connections

    Returns:
        StructureSpec for hierarchical blocks connectivity

    Example:
        >>> # Create 64-node layer with 3-tier hierarchy
        >>> g = Graph(dt=37, network_evaluation_method="layerwise")
        >>> g.add_layer(0, layers.MultiplierWICC(dim=64))
        >>> g.connect(0, 0,
        ...           structure=structure.hierarchical_blocks(levels=3, base_size=4),
        ...           init=init.constant(0.1))
        >>>
        >>> # Custom tier fractions for sparser high-level connectivity
        >>> g.add_layer(1, layers.MultiplierWICC(dim=125))
        >>> g.connect(1, 1,
        ...           structure=structure.hierarchical_blocks(
        ...               levels=3, base_size=5,
        ...               tier_fractions=[1.0, 0.2, 0.1]
        ...           ),
        ...           init=init.uniform(-0.1, 0.1))

    """
    params: dict[str, int | list[float]] = {
        "levels": levels,
        "base_size": base_size,
    }
    if tier_fractions is not None:
        params["tier_fractions"] = tier_fractions

    return StructureSpec(
        type="hierarchical_blocks",
        params=params,
        allow_self_connections=allow_self_connections,
        visualization_metadata={"hierarchical": {"levels": levels, "base_size": base_size}},
    )
