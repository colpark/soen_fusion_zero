"""Parameter specification helpers for fine-grained layer control.

These helpers make it easier to specify distributions, constraints, and learnability
for layer parameters.
"""

from typing import Any


class ParamSpec:
    """Specification for a layer parameter with distribution, constraints, and learnability."""

    def __init__(
        self,
        distribution: str,
        params: dict[str, Any] | None = None,
        learnable: bool | None = None,
        constraints: dict[str, float] | None = None,
    ) -> None:
        """Create a parameter specification.

        Args:
            distribution: Distribution name ("uniform", "normal", "lognormal", etc.)
            params: Distribution parameters (e.g., {"min": 1.5, "max": 2.0}).
                    May include "block_size" to group nodes into blocks that share values.
            learnable: Whether this parameter should be trainable
            constraints: Min/max constraints {"min": ..., "max": ...}

        Example:
            >>> bias_current = ParamSpec(
            ...     distribution="uniform",
            ...     params={"min": 1.5, "max": 2.0},
            ...     learnable=True,
            ...     constraints={"min": 0.0, "max": 5.0}
            ... )

        """
        self.distribution = distribution
        self.params = params or {}
        self.learnable = learnable
        self.constraints = constraints

    def to_dict(self) -> dict[str, Any]:
        """Convert to LayerConfig params format."""
        result: dict[str, Any] = {
            "distribution": self.distribution,
            "params": self.params,
        }

        if self.learnable is not None:
            result["learnable"] = self.learnable

        if self.constraints is not None:
            result["constraints"] = self.constraints

        return result


def _build_params(
    base_params: dict[str, Any],
    block_size: int,
    block_mode: str,
) -> dict[str, Any]:
    """Build params dict, only including blocking params if not defaults."""
    result = dict(base_params)
    if block_size != 1:
        result["block_size"] = block_size
    if block_mode != "shared":
        result["block_mode"] = block_mode
    return result


# Distribution shortcuts
def uniform(
    min: float,
    max: float,
    block_size: int = 1,
    block_mode: str = "shared",
    learnable: bool | None = None,
    constraints: dict[str, float] | None = None,
) -> ParamSpec:
    """Uniform distribution parameter.

    Args:
        min: Minimum value
        max: Maximum value
        block_size: Number of nodes per block (default 1).
                   Layer width must be evenly divisible by block_size.
        block_mode: How blocking is applied (default "shared"):
                   - "shared": Nodes within a block share the same value.
                   - "tiled": A pattern of block_size values is tiled across blocks.
        learnable: Whether parameter is trainable
        constraints: Min/max constraints during training

    Returns:
        ParamSpec for uniform distribution

    """
    return ParamSpec("uniform", _build_params({"min": min, "max": max}, block_size, block_mode), learnable, constraints)


def normal(
    mean: float,
    std: float,
    block_size: int = 1,
    block_mode: str = "shared",
    learnable: bool | None = None,
    constraints: dict[str, float] | None = None,
) -> ParamSpec:
    """Normal (Gaussian) distribution parameter.

    Args:
        mean: Mean of distribution
        std: Standard deviation
        block_size: Number of nodes per block (default 1).
                   Layer width must be evenly divisible by block_size.
        block_mode: How blocking is applied (default "shared"):
                   - "shared": Nodes within a block share the same value.
                   - "tiled": A pattern of block_size values is tiled across blocks.
        learnable: Whether parameter is trainable
        constraints: Min/max constraints during training

    Returns:
        ParamSpec for normal distribution

    """
    return ParamSpec("normal", _build_params({"mean": mean, "std": std}, block_size, block_mode), learnable, constraints)


def lognormal(
    mean: float,
    std: float,
    block_size: int = 1,
    block_mode: str = "shared",
    learnable: bool | None = None,
    constraints: dict[str, float] | None = None,
) -> ParamSpec:
    """Log-normal distribution parameter (in log space).

    Args:
        mean: Mean in log space (e.g., -6.9 for exp(-6.9) ~ 1e-3)
        std: Standard deviation in log space
        block_size: Number of nodes per block (default 1).
                   Layer width must be evenly divisible by block_size.
        block_mode: How blocking is applied (default "shared"):
                   - "shared": Nodes within a block share the same value.
                   - "tiled": A pattern of block_size values is tiled across blocks.
        learnable: Whether parameter is trainable
        constraints: Min/max constraints during training

    Returns:
        ParamSpec for log-normal distribution

    Example:
        >>> gamma_plus = lognormal(mean=-6.9, std=0.2, learnable=True,
        ...                        constraints={"min": 0.0, "max": 0.01})

    """
    return ParamSpec("lognormal", _build_params({"mean": mean, "std": std}, block_size, block_mode), learnable, constraints)


def constant(
    value: float,
    block_size: int = 1,
    block_mode: str = "shared",
    learnable: bool | None = None,
    constraints: dict[str, float] | None = None,
) -> ParamSpec:
    """Constant value parameter.

    Args:
        value: Constant value for all nodes
        block_size: Number of nodes per block (default 1).
                   Layer width must be evenly divisible by block_size.
        block_mode: How blocking is applied (default "shared"):
                   - "shared": Nodes within a block share the same value.
                   - "tiled": A pattern of block_size values is tiled across blocks.
                   Note: For constant values, both modes produce the same result.
        learnable: Whether parameter is trainable
        constraints: Min/max constraints during training

    Returns:
        ParamSpec for constant value

    """
    return ParamSpec("constant", _build_params({"value": value}, block_size, block_mode), learnable, constraints)


def linear(
    min: float,
    max: float,
    block_size: int = 1,
    block_mode: str = "shared",
    learnable: bool | None = None,
    constraints: dict[str, float] | None = None,
) -> ParamSpec:
    """Linearly spaced values.

    Args:
        min: Minimum value
        max: Maximum value
        block_size: Number of nodes per block (default 1).
                   Layer width must be evenly divisible by block_size.
        block_mode: How blocking is applied (default "shared"):
                   - "shared": Linearly spaced values across blocks, repeated within.
                     Example: block_size=3, 12 nodes -> [v1,v1,v1, v2,v2,v2, v3,v3,v3, v4,v4,v4]
                   - "tiled": Linearly spaced values within blocks, tiled across.
                     Example: block_size=3, 12 nodes -> [v1,v2,v3, v1,v2,v3, v1,v2,v3, v1,v2,v3]
        learnable: Whether parameter is trainable
        constraints: Min/max constraints during training

    Returns:
        ParamSpec for linearly spaced values

    Example:
        >>> # 30 nodes with block_size=3, shared: 10 blocks with linearly spaced values
        >>> phi_offset = linear(min=0.0, max=0.5, block_size=3)
        >>> # 30 nodes with block_size=3, tiled: 3 values tiled 10 times
        >>> phi_offset = linear(min=0.0, max=0.5, block_size=3, block_mode="tiled")

    """
    return ParamSpec("linear", _build_params({"min": min, "max": max}, block_size, block_mode), learnable, constraints)


def loglinear(
    min: float,
    max: float,
    block_size: int = 1,
    block_mode: str = "shared",
    learnable: bool | None = None,
    constraints: dict[str, float] | None = None,
) -> ParamSpec:
    """Exponentially spaced values (in log space).

    Args:
        min: Minimum value in log space
        max: Maximum value in log space
        block_size: Number of nodes per block (default 1).
                   Layer width must be evenly divisible by block_size.
        block_mode: How blocking is applied (default "shared"):
                   - "shared": Nodes within a block share the same value.
                   - "tiled": A pattern of block_size values is tiled across blocks.
        learnable: Whether parameter is trainable
        constraints: Min/max constraints during training

    Returns:
        ParamSpec for exponentially spaced values

    """
    return ParamSpec("loglinear", _build_params({"min": min, "max": max}, block_size, block_mode), learnable, constraints)


def loguniform(
    min: float,
    max: float,
    block_size: int = 1,
    block_mode: str = "shared",
    learnable: bool | None = None,
    constraints: dict[str, float] | None = None,
) -> ParamSpec:
    """Log-uniform distribution (in log space).

    Args:
        min: Minimum value in log space
        max: Maximum value in log space
        block_size: Number of nodes per block (default 1).
                   Layer width must be evenly divisible by block_size.
        block_mode: How blocking is applied (default "shared"):
                   - "shared": Nodes within a block share the same value.
                   - "tiled": A pattern of block_size values is tiled across blocks.
        learnable: Whether parameter is trainable
        constraints: Min/max constraints during training

    Returns:
        ParamSpec for log-uniform distribution

    """
    return ParamSpec("loguniform", _build_params({"min": min, "max": max}, block_size, block_mode), learnable, constraints)
