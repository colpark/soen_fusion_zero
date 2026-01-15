"""Single source of truth for JAX layer parameter specifications.

This module defines required parameters and default values for all layer types.
It eliminates scattered defaults and ensures consistency between training,
simulation, and checkpointing code paths.

SOLID Principles:
- Single Responsibility: Owns all parameter definitions
- Fail-Fast Defense: Validates required params exist
- DRY: One registry used everywhere
- Encapsulation: Defaults hidden, exposed via functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp


@dataclass
class LayerParameterSpec:
    """Specification of required and optional parameters for a layer type.

    Attributes:
        layer_kind: Layer type name (e.g., "Multiplier", "SingleDendrite")
        required_params: Dict of parameter_name -> default_value (must exist)
        optional_params: Dict of parameter_name -> default_value (may be None/missing)
    """

    layer_kind: str
    required_params: dict[str, float] = field(default_factory=dict)
    optional_params: dict[str, float | None] = field(default_factory=dict)

    def all_param_names(self) -> set[str]:
        """Get all parameter names (required + optional)."""
        return set(self.required_params.keys()) | set(self.optional_params.keys())

    def is_required(self, param_name: str) -> bool:
        """Check if parameter is required."""
        return param_name in self.required_params

    def get_default(self, param_name: str) -> float | None:
        """Get default value for parameter."""
        if param_name in self.required_params:
            return self.required_params[param_name]
        return self.optional_params.get(param_name)


# Registry of all supported layer parameter specifications
# These defaults match the Torch layer implementations and are used consistently
# across all JAX code paths (convert, training, simulation, checkpointing)
LAYER_PARAM_SPECS = {
    # Legacy names (deprecated but needed for old checkpoints)
    "Multiplier": LayerParameterSpec(
        layer_kind="Multiplier",
        required_params={
            "phi_y": 0.0,
            "bias_current": 2.0,
            "gamma_plus": 1e-3,
            "gamma_minus": 1e-3,
        },
    ),
    "MultiplierV2": LayerParameterSpec(
        layer_kind="MultiplierV2",
        required_params={
            "phi_y": 0.1,
            "bias_current": 2.1,
            "alpha": 1.64053,
            "beta": 303.85,
            "beta_out": 91.156,
        },
    ),
    "MultiplierNOCC": LayerParameterSpec(
        layer_kind="MultiplierNOCC",
        required_params={
            "phi_y": 0.1,
            "bias_current": 2.1,
            "alpha": 1.64053,
            "beta": 303.85,
            "beta_out": 91.156,
        },
    ),
    # Modern names (WICC = v1, NOCC = v2)
    "WICC": LayerParameterSpec(
        layer_kind="WICC",
        required_params={
            "phi_y": 0.0,
            "bias_current": 2.0,
            "gamma_plus": 1e-3,
            "gamma_minus": 1e-3,
        },
    ),
    "NOCC": LayerParameterSpec(
        layer_kind="NOCC",
        required_params={
            "phi_y": 0.1,
            "bias_current": 2.1,
            "alpha": 1.64053,
            "beta": 303.85,
            "beta_out": 91.156,
        },
    ),
    "SingleDendrite": LayerParameterSpec(
        layer_kind="SingleDendrite",
        required_params={
            "phi_offset": 0.23,  # Flux offset
            "bias_current": 1.7,  # Bias current in μA
            "gamma_plus": 1e-3,  # Positive gain
            "gamma_minus": 1e-3,  # Negative gain (damping)
        },
    ),
    "Soma": LayerParameterSpec(
        layer_kind="Soma",
        required_params={
            "phi_offset": 0.23,
            "bias_current": 1.7,
            "gamma_plus": 1e-3,
            "gamma_minus": 1e-3,
            "threshold": 0.0,
        },
    ),
    "Dendrite": LayerParameterSpec(
        layer_kind="Dendrite",
        required_params={
            "phi_offset": 0.23,
            "bias_current": 1.7,
            "gamma_plus": 1e-3,
            "gamma_minus": 1e-3,
        },
    ),
    "NonLinear": LayerParameterSpec(
        layer_kind="NonLinear",
        required_params={},  # NonLinear params are truly optional
        optional_params={
            "phi_offset": 0.0,  # Optional flux offset
            "bias_current": 0.0,  # Optional bias current (used by some sources)
        },
    ),
    "ScalingLayer": LayerParameterSpec(
        layer_kind="ScalingLayer",
        required_params={
            "scale_factor": 1.0,  # Multiplicative scale factor
        },
    ),
    "Scaling": LayerParameterSpec(
        layer_kind="Scaling",
        required_params={
            "scale_factor": 1.0,
        },
    ),
    "Softmax": LayerParameterSpec(
        layer_kind="Softmax",
        required_params={
            "beta": 1.0,  # Temperature parameter (higher = sharper distribution)
        },
    ),
    "Synapse": LayerParameterSpec(
        layer_kind="Synapse",
        required_params={
            "alpha": 0.9,
        },
    ),
    "Linear": LayerParameterSpec(
        layer_kind="Linear",
        required_params={},  # Linear has no parameters
        optional_params={},
    ),
    "MinGRU": LayerParameterSpec(
        layer_kind="MinGRU",
        required_params={},  # MinGRU uses weight matrices, not vectors
        optional_params={
            "W_hidden": None,  # [D, D] matrix
            "W_gate": None,  # [D, D] matrix
        },
    ),
    "GRU": LayerParameterSpec(
        layer_kind="GRU",
        required_params={},  # GRU uses weight matrices
        optional_params={
            "weight_ih": None,  # Input-hidden weights
            "weight_hh": None,  # Hidden-hidden weights
            "bias_ih": None,  # Input-hidden bias
            "bias_hh": None,  # Hidden-hidden bias
        },
    ),
    "LSTM": LayerParameterSpec(
        layer_kind="LSTM",
        required_params={},  # LSTM uses weight matrices
        optional_params={
            "weight_ih": None,  # [4*D, D] Input-hidden weights for 4 gates
            "weight_hh": None,  # [4*D, D] Hidden-hidden weights for 4 gates
            "bias_ih": None,    # [4*D] Input-hidden bias
            "bias_hh": None,    # [4*D] Hidden-hidden bias
        },
    ),
}


def normalize_layer_kind(kind: str) -> str:
    """Normalize layer kind string to canonical form.

    Args:
        kind: Layer kind string (may be lowercase, alternate spelling)

    Returns:
        Canonical layer kind string
    """
    kind_lower = kind.lower()

    # Map variants to canonical names
    canonical_map = {
        # Modern names (preferred)
        "wicc": "WICC",
        "nocc": "NOCC",
        # Legacy/alternate names (for backward compatibility)
        "multiplier": "WICC",  # v1 = WICC
        "multiplierv2": "NOCC",  # v2 = NOCC
        "multiplier_v2": "NOCC",
        "multipliernocc": "NOCC",
        # Other layers
        "singledendrite": "SingleDendrite",
        "single_dendrite": "SingleDendrite",
        "soma": "Soma",
        "synapse": "Synapse",
        "dendrite": "Dendrite",
        "nonlinear": "NonLinear",
        "scalinglayer": "ScalingLayer",
        "scaling": "Scaling",
        "softmax": "Softmax",
        "linear": "Linear",
        "mingru": "MinGRU",
        "gru": "GRU",
        "lstm": "LSTM",
    }

    return canonical_map.get(kind_lower, kind)


def get_param_spec(layer_kind: str) -> LayerParameterSpec:
    """Get parameter specification for a layer type.

    Args:
        layer_kind: Layer type name

    Returns:
        LayerParameterSpec

    Raises:
        ValueError: If layer kind is not supported
    """
    canonical_kind = normalize_layer_kind(layer_kind)

    if canonical_kind not in LAYER_PARAM_SPECS:
        supported = ", ".join(sorted(LAYER_PARAM_SPECS.keys()))
        msg = f"Unsupported layer kind '{layer_kind}'. Supported: {supported}"
        raise ValueError(msg)

    return LAYER_PARAM_SPECS[canonical_kind]


def get_layer_defaults(layer_kind: str, dim: int) -> dict[str, jnp.ndarray]:
    """Get default parameter arrays for a layer type.

    Args:
        layer_kind: Layer type name
        dim: Layer dimension (for creating appropriately-sized arrays)

    Returns:
        Dictionary of parameter_name -> default JAX array
    """
    spec = get_param_spec(layer_kind)
    defaults = {}

    # Required parameters with defaults
    for name, default_value in spec.required_params.items():
        if name in ("W_hidden", "W_gate", "weight_ih", "weight_hh"):
            # Matrix parameters
            defaults[name] = jnp.eye(dim, dtype=jnp.float32)
        elif name in ("bias_ih", "bias_hh"):
            # Bias vectors (3D for GRU gates)
            defaults[name] = jnp.zeros((3, dim), dtype=jnp.float32)
        else:
            # Scalar parameters broadcast to vectors
            defaults[name] = jnp.ones((dim,), dtype=jnp.float32) * default_value

    # Optional parameters with defaults (if specified)
    for name, default_value in spec.optional_params.items():
        if default_value is not None:
            # Explicitly cast to float to satisfy mypy since default_value is float | None
            val_float = float(default_value)
            if name in ("W_hidden", "W_gate", "weight_ih", "weight_hh"):
                defaults[name] = jnp.eye(dim, dtype=jnp.float32)
            elif name in ("bias_ih", "bias_hh"):
                defaults[name] = jnp.zeros((3, dim), dtype=jnp.float32)
            else:
                defaults[name] = jnp.ones((dim,), dtype=jnp.float32) * float(val_float or 0.0)

    return defaults


def validate_layer_params(layer_kind: str, params: dict[str, Any] | None) -> None:
    """Validate that layer has all required parameters (fail-fast).

    Args:
        layer_kind: Layer type name
        params: Parameter dictionary to validate

    Raises:
        ValueError: If required parameters are missing
    """
    spec = get_param_spec(layer_kind)

    if not spec.required_params:
        # No required params for this layer type
        return

    if params is None:
        if spec.required_params:
            missing = list(spec.required_params.keys())
            msg = f"Layer type '{layer_kind}' requires parameters {missing}, but params dict is None"
            raise ValueError(msg)
        return

    # Check for missing required parameters
    missing = []
    for param_name in spec.required_params:
        if param_name not in params or params[param_name] is None:
            missing.append(param_name)

    if missing:
        msg = f"Layer type '{layer_kind}' missing required parameters: {missing}. Required: {list(spec.required_params.keys())}"
        raise ValueError(msg)


def fill_missing_params(
    layer_kind: str,
    dim: int,
    params: dict[str, Any] | None,
) -> dict[str, jnp.ndarray]:
    """Fill missing required parameters with defaults (ONLY for checkpoint backward compatibility).

    WARNING: This function should ONLY be used when deserializing old checkpoints
    that may be missing parameters. During normal conversion (Torch → JAX), all
    parameters MUST exist or the conversion should fail.

    Used ONLY by:
    - checkpointing.py when deserializing old checkpoints (backward compatibility)

    NEVER used by:
    - convert.py (should fail-fast if params missing)
    - Training initialization (should fail-fast if params missing)

    Args:
        layer_kind: Layer type name
        dim: Layer dimension
        params: Existing parameter dictionary (may be None or incomplete)

    Returns:
        Complete parameter dictionary with all required params filled

    Raises:
        ValueError: If layer kind is unsupported
    """
    defaults = get_layer_defaults(layer_kind, dim)

    if params is None:
        # No existing params, return all defaults (old checkpoint case)
        return defaults

    # Merge: existing params take precedence over defaults
    result = dict(defaults)  # Start with defaults

    for key, value in params.items():
        if value is not None:
            result[key] = value

    return result


def get_param_array_shape(layer_kind: str, dim: int) -> tuple[int, int]:
    """Get expected shape of flattened parameter array for layer.

    This is used by pure_forward.py to create the flattened arrays.

    Args:
        layer_kind: Layer type name
        dim: Layer dimension

    Returns:
        Tuple of (num_params, dim) or (0, dim) for layers with no vector params
    """
    spec = get_param_spec(layer_kind)

    # Count non-matrix parameters (matrices handled separately)
    vector_params = []
    for name in spec.required_params:
        if name not in ("W_hidden", "W_gate", "weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            vector_params.append(name)

    for name in spec.optional_params:
        if name not in ("W_hidden", "W_gate", "weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            vector_params.append(name)

    num_params = len(vector_params)
    return (num_params, dim)


__all__ = [
    "LayerParameterSpec",
    "LAYER_PARAM_SPECS",
    "normalize_layer_kind",
    "get_param_spec",
    "get_layer_defaults",
    "validate_layer_params",
    "fill_missing_params",
    "get_param_array_shape",
]
