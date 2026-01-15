"""Weight initialization helpers for connections.

These functions return InitSpec objects that can be passed to Graph.connect().
They map to the existing weight initializer registry in the core toolkit.
"""

from .specs import InitSpec


def normal(mean: float = 0.0, std: float = 0.1) -> InitSpec:
    """Normal (Gaussian) initialization.

    Args:
        mean: Mean of the distribution
        std: Standard deviation

    Returns:
        InitSpec for normal initialization

    """
    return InitSpec(name="normal", params={"mean": mean, "std": std})


def uniform(min: float = -0.24, max: float = 0.24) -> InitSpec:
    """Uniform initialization.

    Args:
        min: Minimum value
        max: Maximum value

    Returns:
        InitSpec for uniform initialization

    Note:
        Aliases `a` and `b` are also accepted for min/max.

    """
    return InitSpec(name="uniform", params={"min": min, "max": max})


def linear(min: float = 0.0, max: float = 1.0) -> InitSpec:
    """Linear spacing initialization.

    Linearly spaces weights between min and max across all connections.
    For example, with 2 connections you get [min, max]; with 3 you get [min, midpoint, max].

    Args:
        min: Minimum value
        max: Maximum value

    Returns:
        InitSpec for linear spacing initialization

    """
    return InitSpec(name="linear", params={"min": min, "max": max})


def xavier_normal(gain: float = 1.0) -> InitSpec:
    """Xavier normal initialization (Glorot normal).

    Args:
        gain: Scaling factor

    Returns:
        InitSpec for Xavier normal initialization

    """
    return InitSpec(name="xavier_normal", params={"gain": gain})


def xavier_uniform(gain: float = 1.0) -> InitSpec:
    """Xavier uniform initialization (Glorot uniform).

    Args:
        gain: Scaling factor

    Returns:
        InitSpec for Xavier uniform initialization

    """
    return InitSpec(name="xavier_uniform", params={"gain": gain})


def kaiming_normal(nonlinearity: str = "relu", a: float = 0.0) -> InitSpec:
    """Kaiming normal initialization (He normal).

    Args:
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', etc.)
        a: Negative slope for leaky_relu

    Returns:
        InitSpec for Kaiming normal initialization

    """
    return InitSpec(name="kaiming_normal", params={"nonlinearity": nonlinearity, "a": a})


def kaiming_uniform(nonlinearity: str = "relu", a: float = 0.0) -> InitSpec:
    """Kaiming uniform initialization (He uniform).

    Args:
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', etc.)
        a: Negative slope for leaky_relu

    Returns:
        InitSpec for Kaiming uniform initialization

    """
    return InitSpec(name="kaiming_uniform", params={"nonlinearity": nonlinearity, "a": a})


def orthogonal(gain: float = 1.0) -> InitSpec:
    """Orthogonal initialization.

    Args:
        gain: Scaling factor

    Returns:
        InitSpec for orthogonal initialization

    """
    return InitSpec(name="orthogonal", params={"gain": gain})


def constant(value: float = 1.0) -> InitSpec:
    """Constant initialization.

    Args:
        value: Constant value for all weights

    Returns:
        InitSpec for constant initialization

    """
    return InitSpec(name="constant", params={"value": value})


def custom_weights(weights_file: str) -> InitSpec:
    """Load custom weights from .npy or .npz file.

    Load pre-trained weights or custom weight matrices from file.
    Weights can be a standalone .npy file or stored as "weights" key in .npz.

    The weight matrix must have shape [to_nodes, from_nodes] matching the connection.

    Args:
        weights_file: Path to .npy or .npz file containing weights

    Returns:
        InitSpec for custom weights initialization

    Example:
        >>> import numpy as np
        >>> # Create and save custom weights
        >>> weights = np.random.randn(5, 10).astype(np.float32)
        >>> np.save("my_weights.npy", weights)
        >>>
        >>> # Use in model
        >>> g.connect(0, 1,
        ...           structure=structure.dense(),
        ...           init=init.custom_weights("my_weights.npy"))

    """
    return InitSpec(name="custom", params={"weights_file": weights_file})


def flux_balanced(
    phi_exc_target: float = 0.27,
    mean_state: float = 1.0,
    noise_std: float = 0.0,
) -> InitSpec:
    """Physics-informed initialization for SOEN networks.

    Sets weights so that when upstream neurons are at mean_state, the total
    external flux to each target node equals phi_exc_target. When multiple
    connections feed a layer, the flux is automatically split evenly.

    The source function (RateArray/Heaviside) is periodic with maximum response
    at phi_total = 0.5. With typical phi_offset = 0.23:
        phi_total = phi_exc + phi_offset = 0.27 + 0.23 = 0.5

    For each target node i with fan_in N_i from this connection:
        J[i] = phi_exc_target / (num_sources * N_i * mean_state)

    Where num_sources is auto-computed as the number of connections feeding
    the target layer (external + internal + feedback).

    This scales weights inversely with fan-in, so nodes with many inputs get
    smaller individual weights to maintain consistent total drive.

    Args:
        phi_exc_target: Target TOTAL external flux per node (default 0.27).
            Combined with phi_offset (~0.23), gives phi_total near 0.5.
        mean_state: Assumed mean state of upstream neurons (default 1.0).
            Since s is in [0,1] for RateArray/Heaviside, 1.0 represents
            fully active upstream neurons.
        noise_std: Standard deviation of noise to add for symmetry breaking
            (default 0.0 for deterministic initialization).

    Returns:
        InitSpec for flux-balanced initialization

    Example:
        >>> # Works automatically regardless of how many connections feed the layer
        >>> g.connect(0, 1, structure=structure.sparse(0.1), init=init.flux_balanced())
        >>> g.connect(1, 1, structure=structure.dense(), init=init.flux_balanced())  # internal
        >>> # Both connections auto-split the target flux evenly

    """
    return InitSpec(
        name="flux_balanced",
        params={
            "phi_exc_target": phi_exc_target,
            "mean_state": mean_state,
            "noise_std": noise_std,
        },
    )
