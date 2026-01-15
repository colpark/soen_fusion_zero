"""Connectivity metadata and initialization helpers for layers."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import torch

from .connectivity_builders import CONNECTIVITY_BUILDERS
from .weight_initializers import init_custom_weights

if TYPE_CHECKING:
    from collections.abc import Callable

CONNECTIVITY_ALIASES: dict[str, str] = {
    "all_to_all": "dense",
}


_DENSE_DESCRIPTION: dict[str, Any] = {
    "description": "Full connectivity between all nodes.",
    "params": {},
}


CONNECTIVITY_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "dense": _DENSE_DESCRIPTION,
    "one_to_one": {
        "description": "Diagonal connectivity between matching indices. Optionally specify source node range.",
        "params": {
            "source_start_node_id": "Starting source node index (inclusive, optional).",
            "source_end_node_id": "Ending source node index (inclusive, optional).",
        },
    },
    "inverse_one_to_one": {
        "description": "Fully connected except for diagonal elements (matching indices). Useful for lateral inhibition.",
        "params": {},
    },
    "chain": {
        "description": "Links each source node to the next target index, forming a forward chain.",
        "params": {},
    },
    "sparse": {
        "description": "Sparse connectivity with Bernoulli sampling per edge.",
        "params": {
            "sparsity": "Connection probability between any two nodes (0-1).",
        },
    },
    "block_structure": {
        "description": "Deterministic block-based connectivity with configurable densities.",
        "params": {
            "block_count": "Number of blocks used for both source and target layers.",
            "connection_mode": "'diagonal' connects matching blocks, 'full' connects all block pairs.",
            "within_block_density": "Fraction of edges populated inside each block (0-1).",
            "cross_block_density": "Fraction of edges populated across blocks (0-1).",
        },
    },
    "power_law": {
        "description": "Distance-biased sampling over deterministic grid coordinates using power-law decay.",
        "params": {
            "alpha": "Decay exponent for distance weighting.",
            "expected_fan_out": "Targets sampled per source node.",
        },
    },
    "exponential": {
        "description": "Distance-biased sampling with exponential decay over grid coordinates.",
        "params": {
            "d_0": "Characteristic length scale for exponential decay.",
            "expected_fan_out": "Targets sampled per source node.",
        },
    },
    "constant": {
        "description": "Uniform sampling of a fixed number of targets per source node.",
        "params": {
            "expected_fan_out": "Targets sampled per source node.",
        },
    },
    "custom": {
        "description": "Load custom connectivity mask from .npz file. Mask must be stored with key 'mask'.",
        "params": {
            "mask_file": "Path to .npz file containing the mask",
        },
    },
    "hierarchical_blocks": {
        "description": "Nested multi-tier blocks with interneuron connectivity at each scale. Uses defaults [1.0, 0.5, 0.25, 0.125] for tier fractions.",
        "params": {
            "levels": "Number of hierarchical tiers (1-4). Total nodes = base_size^levels.",
            "base_size": "Number of units per level in the hierarchy (minimum 2).",
        },
    },
}


CONNECTIVITY_PARAM_TYPES: dict[str, dict[str, Any]] = {
    "dense": {},
    "all_to_all": {},
    "one_to_one": {
        "source_start_node_id": {"type": "int", "min": 0, "default": 0, "optional": True},
        "source_end_node_id": {"type": "int", "min": 0, "default": 0, "optional": True},
    },
    "inverse_one_to_one": {},
    "chain": {},
    "sparse": {
        "sparsity": {"type": "float", "min": 0.0, "max": 1.0, "decimals": 2, "step": 0.05, "default": 0.5},
    },
    "block_structure": {
        "block_count": {"type": "int", "min": 1, "max": 100, "default": 4},
        "connection_mode": {"type": "enum", "options": ["diagonal", "full"], "default": "diagonal"},
        "within_block_density": {"type": "float", "min": 0.0, "max": 1.0, "decimals": 2, "step": 0.1, "default": 1.0},
        "cross_block_density": {"type": "float", "min": 0.0, "max": 1.0, "decimals": 2, "step": 0.1, "default": 0.0},
    },
    "power_law": {
        "alpha": {"type": "float", "min": 0.1, "max": 10.0, "decimals": 2, "step": 0.1, "default": 2.0},
        "expected_fan_out": {"type": "int", "min": 1, "max": 1_000_000, "default": 4},
    },
    "exponential": {
        "d_0": {"type": "float", "min": 0.1, "max": 20.0, "decimals": 2, "step": 0.1, "default": 2.0},
        "expected_fan_out": {"type": "int", "min": 1, "max": 1_000_000, "default": 4},
    },
    "constant": {
        "expected_fan_out": {"type": "int", "min": 1, "max": 1_000_000, "default": 4},
    },
    "custom": {
        "mask_file": {"type": "file", "extensions": [".npz"], "required": True},
    },
    "hierarchical_blocks": {
        "levels": {"type": "int", "min": 1, "max": 10, "default": 3},
        "base_size": {"type": "int", "min": 2, "max": 99, "default": 4},
        "tier_fractions": {
            "type": "dynamic_float_list",
            "depends_on": "levels",
            "defaults": [1.0, 0.5, 0.25, 0.125],
            "min": 0.0,
            "max": 1.0,
            "decimals": 2,
            "step": 0.05,
        },
    },
}


WEIGHT_INITIALIZER_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "normal": {
        "description": "Normal (Gaussian) distribution initialization.",
        "params": {
            "mean": "Mean of the distribution.",
            "std": "Standard deviation.",
        },
    },
    "uniform": {
        "description": "Uniform distribution initialization.",
        "params": {
            "min": "Minimum value.",
            "max": "Maximum value.",
        },
    },
    "linear": {
        "description": "Linearly spaced values across all weights.",
        "params": {
            "min": "Minimum value.",
            "max": "Maximum value.",
        },
    },
    "xavier_normal": {
        "description": "Xavier (Glorot) normal initialization.",
        "params": {
            "gain": "Scaling factor.",
        },
    },
    "xavier_uniform": {
        "description": "Xavier (Glorot) uniform initialization.",
        "params": {
            "gain": "Scaling factor.",
        },
    },
    "kaiming_normal": {
        "description": "Kaiming (He) normal initialization.",
        "params": {
            "nonlinearity": "Type of nonlinearity (e.g., 'relu', 'leaky_relu').",
            "a": "Negative slope for leaky_relu.",
        },
    },
    "kaiming_uniform": {
        "description": "Kaiming (He) uniform initialization.",
        "params": {
            "nonlinearity": "Type of nonlinearity (e.g., 'relu', 'leaky_relu').",
            "a": "Negative slope for leaky_relu.",
        },
    },
    "orthogonal": {
        "description": "Orthogonal matrix initialization.",
        "params": {
            "gain": "Scaling factor.",
        },
    },
    "constant": {
        "description": "Constant value initialization.",
        "params": {
            "value": "Constant value for all weights.",
        },
    },
    "custom": {
        "description": "Load custom weights from .npy or .npz file. Weights are stored as [to_nodes, from_nodes].",
        "params": {
            "weights_file": "Path to .npy or .npz file containing weights",
        },
    },
    "flux_balanced": {
        "description": (
            "Physics-informed initialization for SOEN networks. Sets weights so total external flux "
            "reaches phi_exc_target, evenly split across all input sources. Auto-computes the number "
            "of input sources when used via model builder."
        ),
        "params": {
            "phi_exc_target": "Target TOTAL external flux per node (default 0.27).",
            "mean_state": "Assumed mean state of upstream neurons in [0,1] (default 1.0).",
            "noise_std": "Standard deviation of noise to add for symmetry breaking (default 0.0).",
        },
    },
}


WEIGHT_INITIALIZER_PARAM_TYPES: dict[str, dict[str, Any]] = {
    "normal": {
        "mean": {"type": "float", "min": -10.0, "max": 10.0, "decimals": 3, "step": 0.1, "default": 0.0},
        "std": {"type": "float", "min": 0.01, "max": 10.0, "decimals": 3, "step": 0.1, "default": 0.1},
    },
    "uniform": {
        "min": {"type": "float", "min": -10.0, "max": 10.0, "decimals": 3, "step": 0.1, "default": -0.24},
        "max": {"type": "float", "min": -10.0, "max": 10.0, "decimals": 3, "step": 0.1, "default": 0.24},
    },
    "linear": {
        "min": {"type": "float", "min": -10.0, "max": 10.0, "decimals": 3, "step": 0.1, "default": 0.0},
        "max": {"type": "float", "min": -10.0, "max": 10.0, "decimals": 3, "step": 0.1, "default": 1.0},
    },
    "xavier_normal": {
        "gain": {"type": "float", "min": 0.1, "max": 10.0, "decimals": 3, "step": 0.1, "default": 1.0},
    },
    "xavier_uniform": {
        "gain": {"type": "float", "min": 0.1, "max": 10.0, "decimals": 3, "step": 0.1, "default": 1.0},
    },
    "kaiming_normal": {
        "nonlinearity": {"type": "enum", "options": ["relu", "leaky_relu", "tanh"], "default": "relu"},
        "a": {"type": "float", "min": 0.0, "max": 1.0, "decimals": 3, "step": 0.1, "default": 0.0},
    },
    "kaiming_uniform": {
        "nonlinearity": {"type": "enum", "options": ["relu", "leaky_relu", "tanh"], "default": "relu"},
        "a": {"type": "float", "min": 0.0, "max": 1.0, "decimals": 3, "step": 0.1, "default": 0.0},
    },
    "orthogonal": {
        "gain": {"type": "float", "min": 0.1, "max": 10.0, "decimals": 3, "step": 0.1, "default": 1.0},
    },
    "constant": {
        "value": {"type": "float", "min": -10.0, "max": 10.0, "decimals": 3, "step": 0.1, "default": 1.0},
    },
    "custom": {
        "weights_file": {"type": "file", "extensions": [".npy", ".npz"], "required": True},
    },
    "flux_balanced": {
        "phi_exc_target": {"type": "float", "min": 0.0, "max": 0.5, "decimals": 3, "step": 0.01, "default": 0.27},
        "mean_state": {"type": "float", "min": 0.01, "max": 1.0, "decimals": 2, "step": 0.1, "default": 1.0},
        "noise_std": {"type": "float", "min": 0.0, "max": 0.5, "decimals": 3, "step": 0.01, "default": 0.0},
    },
}


def remove_self_connections(mask: torch.Tensor, allow_self_connections: bool) -> torch.Tensor:
    if allow_self_connections or mask.shape[0] != mask.shape[1]:
        return mask
    eye = torch.eye(mask.shape[0], device=mask.device, dtype=mask.dtype)
    return mask * (1 - eye)


def normalize_connectivity_type(kind: str) -> str:
    target = CONNECTIVITY_ALIASES.get(kind, kind)
    if target != kind:
        warnings.warn(
            "Connectivity type 'all_to_all' is deprecated and will be removed in a future release. Use 'dense' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return target


def build_connectivity(connectivity_type: str, from_nodes: int, to_nodes: int, params: dict[str, Any] | None = None) -> torch.Tensor:
    params = params or {}
    normalized_type = normalize_connectivity_type(connectivity_type)
    builder = CONNECTIVITY_BUILDERS.get(normalized_type)
    if builder is None:
        msg = f"Unknown connectivity_type '{connectivity_type}'. Available: {list(CONNECTIVITY_BUILDERS.keys())}"
        raise ValueError(
            msg,
        )
    mask = builder(from_nodes, to_nodes, params)
    allow_self = bool(params.get("allow_self_connections", True))
    return remove_self_connections(mask, allow_self)


def init_normal(from_nodes: int, to_nodes: int, mask: torch.Tensor, *, mean: float = 0.0, std: float = 0.1) -> torch.Tensor:
    weights = torch.normal(mean, std, size=(to_nodes, from_nodes))
    return weights * mask


def init_uniform(from_nodes: int, to_nodes: int, mask: torch.Tensor, *, min: float = -0.24, max: float = 0.24) -> torch.Tensor:
    weights = torch.empty(to_nodes, from_nodes).uniform_(min, max)
    return weights * mask


def init_linear(from_nodes: int, to_nodes: int, mask: torch.Tensor, *, min: float = 0.0, max: float = 1.0) -> torch.Tensor:
    # Create linearly spaced values across all weights
    total_weights = to_nodes * from_nodes
    weights = torch.linspace(min, max, total_weights).reshape(to_nodes, from_nodes)
    return weights * mask


def _compute_per_node_fans(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-node fan_in and fan_out from connection mask.

    For proper Xavier/Kaiming initialization with heterogeneous connectivity,
    we compute the actual fan for each node individually.

    Args:
        mask: Connection mask tensor of shape [to_nodes, from_nodes]

    Returns:
        (fan_in_per_target, fan_out_per_source) tensors
        - fan_in_per_target: [to_nodes] number of incoming connections per target
        - fan_out_per_source: [from_nodes] number of outgoing connections per source
    """
    # fan_in[i]: number of incoming connections to target neuron i
    fan_in_per_target = mask.sum(dim=1).float()  # [to_nodes]

    # fan_out[j]: number of outgoing connections from source neuron j
    fan_out_per_source = mask.sum(dim=0).float()  # [from_nodes]

    # Clamp to minimum of 1 to avoid division by zero
    fan_in_per_target = torch.clamp(fan_in_per_target, min=1.0)
    fan_out_per_source = torch.clamp(fan_out_per_source, min=1.0)

    return fan_in_per_target, fan_out_per_source


def init_xavier_normal(from_nodes: int, to_nodes: int, mask: torch.Tensor, *, gain: float = 1.0) -> torch.Tensor:
    """Xavier normal initialization that accounts for connection mask node-wise.

    For each weight w[i,j], uses the specific fan_in of target i and fan_out of source j
    to compute the proper initialization scale. This handles heterogeneous connectivity
    where some nodes have many connections and others have few.
    """
    # Get per-node fan values
    fan_in_per_target, fan_out_per_source = _compute_per_node_fans(mask)

    # Build per-weight std matrix: std[i,j] = gain * sqrt(2 / (fan_in[i] + fan_out[j]))
    # fan_in_per_target is [to_nodes], fan_out_per_source is [from_nodes]
    # We need to broadcast to [to_nodes, from_nodes]
    fan_sum = fan_in_per_target.unsqueeze(1) + fan_out_per_source.unsqueeze(0)  # [to_nodes, from_nodes]
    std_matrix = gain * (2.0 / fan_sum).sqrt()

    # Sample from standard normal, then scale by per-weight std
    weights = torch.randn(to_nodes, from_nodes) * std_matrix
    return weights * mask


def init_xavier_uniform(from_nodes: int, to_nodes: int, mask: torch.Tensor, *, gain: float = 1.0) -> torch.Tensor:
    """Xavier uniform initialization that accounts for connection mask node-wise.

    For each weight w[i,j], uses the specific fan_in of target i and fan_out of source j
    to compute the proper initialization scale. This handles heterogeneous connectivity
    where some nodes have many connections and others have few.
    """
    # Get per-node fan values
    fan_in_per_target, fan_out_per_source = _compute_per_node_fans(mask)

    # Build per-weight limit matrix: limit[i,j] = gain * sqrt(6 / (fan_in[i] + fan_out[j]))
    fan_sum = fan_in_per_target.unsqueeze(1) + fan_out_per_source.unsqueeze(0)  # [to_nodes, from_nodes]
    limit_matrix = gain * (6.0 / fan_sum).sqrt()

    # Sample from uniform [-1, 1], then scale by per-weight limit
    weights = (torch.rand(to_nodes, from_nodes) * 2 - 1) * limit_matrix
    return weights * mask


def _get_kaiming_gain(nonlinearity: str, a: float) -> float:
    """Get the gain for Kaiming initialization based on nonlinearity.

    Args:
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', 'tanh', 'sigmoid', etc.)
        a: Negative slope for leaky_relu

    Returns:
        Recommended gain value
    """
    if nonlinearity == "relu":
        return 2.0 ** 0.5
    elif nonlinearity == "leaky_relu":
        return (2.0 / (1 + a ** 2)) ** 0.5
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "linear":
        return 1.0
    else:
        # Default gain for unknown nonlinearities
        return 1.0


def init_kaiming_normal(
    from_nodes: int,
    to_nodes: int,
    mask: torch.Tensor,
    *,
    nonlinearity: str = "relu",
    a: float = 0.0,
) -> torch.Tensor:
    """Kaiming normal initialization that accounts for connection mask node-wise.

    For each weight w[i,j], uses the specific fan_in of target i to compute
    the proper initialization scale. Kaiming uses fan_in to preserve forward
    pass variance through ReLU-like activations.
    """
    # Get per-node fan values (only need fan_in for Kaiming)
    fan_in_per_target, _ = _compute_per_node_fans(mask)

    # Build per-row std: std[i] = gain / sqrt(fan_in[i])
    # All weights going into target i share the same fan_in
    gain = _get_kaiming_gain(nonlinearity, a)
    std_per_target = gain / fan_in_per_target.sqrt()  # [to_nodes]

    # Broadcast to weight matrix shape
    std_matrix = std_per_target.unsqueeze(1).expand(to_nodes, from_nodes)

    # Sample from standard normal, then scale by per-weight std
    weights = torch.randn(to_nodes, from_nodes) * std_matrix
    return weights * mask


def init_kaiming_uniform(
    from_nodes: int,
    to_nodes: int,
    mask: torch.Tensor,
    *,
    nonlinearity: str = "relu",
    a: float = 0.0,
) -> torch.Tensor:
    """Kaiming uniform initialization that accounts for connection mask node-wise.

    For each weight w[i,j], uses the specific fan_in of target i to compute
    the proper initialization scale. Kaiming uses fan_in to preserve forward
    pass variance through ReLU-like activations.
    """
    # Get per-node fan values (only need fan_in for Kaiming)
    fan_in_per_target, _ = _compute_per_node_fans(mask)

    # Build per-row limit: limit[i] = gain * sqrt(3 / fan_in[i])
    # All weights going into target i share the same fan_in
    gain = _get_kaiming_gain(nonlinearity, a)
    limit_per_target = gain * (3.0 / fan_in_per_target).sqrt()  # [to_nodes]

    # Broadcast to weight matrix shape
    limit_matrix = limit_per_target.unsqueeze(1).expand(to_nodes, from_nodes)

    # Sample from uniform [-1, 1], then scale by per-weight limit
    weights = (torch.rand(to_nodes, from_nodes) * 2 - 1) * limit_matrix
    return weights * mask


def init_orthogonal(from_nodes: int, to_nodes: int, mask: torch.Tensor, *, gain: float = 1.0) -> torch.Tensor:
    weights = torch.empty(to_nodes, from_nodes)
    torch.nn.init.orthogonal_(weights, gain=gain)
    return weights * mask


def init_constant(from_nodes: int, to_nodes: int, mask: torch.Tensor, *, value: float = 1.0) -> torch.Tensor:
    weights = torch.full((to_nodes, from_nodes), float(value))
    return weights * mask


def init_flux_balanced(
    from_nodes: int,
    to_nodes: int,
    mask: torch.Tensor,
    *,
    phi_exc_target: float = 0.27,
    mean_state: float = 1.0,
    num_input_sources: int = 1,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Initialize weights to achieve a target external flux at each node.

    This physics-informed initialization sets weights so that when upstream
    neurons are at mean_state, the total external flux to each target node
    equals phi_exc_target (divided evenly among all input sources).
    Combined with phi_offset (~0.23), this places the total flux near 0.5
    where the source function (RateArray/Heaviside) has maximum response.

    The source function is periodic with max at phi_total = 0.5:
        phi_total = phi_exc + phi_offset

    For node i with fan_in N_i from this connection:
        phi_exc_i = sum_j(J_ij * s_j) = N_i * J_target * mean_state

    Solving for J_target (with even split across input sources):
        J_target[i] = phi_exc_target / (num_input_sources * N_i * mean_state)

    This scales weights inversely with fan-in, so nodes with many inputs
    get smaller individual weights to maintain the same total drive.

    NOTE: When used via the model builder, num_input_sources is automatically
    computed based on how many connections feed into the target layer.

    Args:
        from_nodes: Number of source neurons
        to_nodes: Number of target neurons
        mask: Connection mask [to_nodes, from_nodes]
        phi_exc_target: Target TOTAL external flux per node (default 0.27,
            which with typical phi_offset=0.23 gives phi_total=0.5)
        mean_state: Assumed mean state of upstream neurons (default 1.0,
            since s is in [0,1] for RateArray/Heaviside)
        num_input_sources: Number of connections feeding the target layer
            (default 1). Auto-computed when used via model builder.
        noise_std: Optional noise to add around the target weight (default 0.0
            for deterministic initialization, set >0 for small perturbations)

    Returns:
        Weight matrix with flux-balanced initialization
    """
    # Get fan-in per target node (from this connection only)
    fan_in_per_target, _ = _compute_per_node_fans(mask)

    # Compute target weight per row, with even split across input sources
    # J[i] = phi_exc_target / (num_input_sources * fan_in[i] * mean_state)
    j_target_per_row = phi_exc_target / (num_input_sources * fan_in_per_target * mean_state)

    # Broadcast to weight matrix shape
    weights = j_target_per_row.unsqueeze(1).expand(to_nodes, from_nodes).clone()

    # Add optional noise for breaking symmetry
    if noise_std > 0:
        noise = torch.randn(to_nodes, from_nodes) * noise_std
        weights = weights + noise

    return weights * mask


WEIGHT_INITIALIZERS: dict[str, Callable[..., torch.Tensor]] = {
    "normal": init_normal,
    "uniform": init_uniform,
    "linear": init_linear,
    "xavier_normal": init_xavier_normal,
    "xavier_uniform": init_xavier_uniform,
    "kaiming_normal": init_kaiming_normal,
    "kaiming_uniform": init_kaiming_uniform,
    "orthogonal": init_orthogonal,
    "constant": init_constant,
    "flux_balanced": init_flux_balanced,
    "custom": init_custom_weights,
}


def load_neuron_polarity(filepath: str) -> torch.Tensor:
    """Load neuron polarity from .npy file.

    Expected: 1D array with values -1, 0, or 1.

    Args:
        filepath: Path to .npy file containing polarity array

    Returns:
        torch.Tensor of dtype int8

    Raises:
        ValueError: If polarity values are not in {-1, 0, 1}
    """
    arr = np.load(filepath)
    tensor = torch.from_numpy(arr).to(torch.int8)
    # Validate values
    if not torch.all((tensor >= -1) & (tensor <= 1)):
        msg = "Polarity values must be -1, 0, or 1"
        raise ValueError(msg)
    return tensor


def build_weight(connectivity_type: str, from_nodes: int, to_nodes: int, params: dict[str, Any] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    params = params or {}
    mask = build_connectivity(connectivity_type, from_nodes, to_nodes, params)

    init_name = params.get("init", "normal")
    initializer = WEIGHT_INITIALIZERS.get(init_name)
    if initializer is None:
        msg = f"Unknown init method '{init_name}'. Available: {list(WEIGHT_INITIALIZERS.keys())}"
        raise ValueError(
            msg,
        )

    if init_name == "uniform":
        if ("a" in params and "min" not in params) or ("b" in params and "max" not in params):
            params = dict(params)
            if "a" in params and "min" not in params:
                params["min"] = params["a"]
            if "b" in params and "max" not in params:
                params["max"] = params["b"]

    sig = inspect.signature(initializer)
    init_kwargs: dict[str, Any] = {}
    for name in sig.parameters:
        if name in params:
            init_kwargs[name] = params[name]

    weights = initializer(from_nodes, to_nodes, mask, **init_kwargs)
    return weights, mask


__all__ = [
    "CONNECTIVITY_ALIASES",
    "CONNECTIVITY_DESCRIPTIONS",
    "CONNECTIVITY_PARAM_TYPES",
    "WEIGHT_INITIALIZERS",
    "WEIGHT_INITIALIZER_DESCRIPTIONS",
    "WEIGHT_INITIALIZER_PARAM_TYPES",
    "build_connectivity",
    "build_weight",
    "load_neuron_polarity",
    "normalize_connectivity_type",
    "remove_self_connections",
]
