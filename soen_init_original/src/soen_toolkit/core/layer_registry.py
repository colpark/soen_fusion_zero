# FILEPATH: src/soen_toolkit/core/layer_registry.py

from __future__ import annotations

# Central registry mapping layer type names to implementation classes.
# Keeping this mapping in one place avoids circular imports between builders
# and the core class, and makes it straightforward to extend with new layers.
from soen_toolkit.core.layers.physical import (
    MultiplierLayer,
    MultiplierNOCCLayer,
    ReadoutLayer,
    SingleDendriteLayer,
    SomaLayer,
)
from soen_toolkit.core.layers.virtual import (
    GRULayer,
    InputLayer,
    LeakyGRULayer,
    LinearLayer,
    LSTMLayer,
    MinGRULayer,
    NonLinearLayer,
    RNNLayer,
    ScalingLayer,
    SoftmaxLayer,
    SynapseLayer,
)

LAYER_TYPE_MAP = {
    "SingleDendrite": SingleDendriteLayer,
    "Soma": SomaLayer,
    "MultiplierWICC": MultiplierLayer,
    "MultiplierNOCC": MultiplierNOCCLayer,
    # Legacy aliases maintained for backward compatibility
    "Multiplier": MultiplierLayer,  # Legacy alias for MultiplierWICC
    "Readout": ReadoutLayer,
    "DendriteReadout": ReadoutLayer,
    "Linear": LinearLayer,
    "Input": InputLayer,
    "ScalingLayer": ScalingLayer,
    "NonLinear": NonLinearLayer,
    "RNN": RNNLayer,
    "GRU": GRULayer,
    "LSTM": LSTMLayer,
    "MinGRU": MinGRULayer,
    "LeakyGRU": LeakyGRULayer,
    # Backward-compatible alias (hidden in GUI picker)
    "leakyGRU": LeakyGRULayer,
    "Softmax": SoftmaxLayer,
    "Synapse": SynapseLayer,
}

# Canonical sets for layer type checking (lowercase for matching).
# Use these instead of hardcoding strings throughout the codebase.
MULTIPLIER_NOCC_TYPES: frozenset[str] = frozenset({
    "multipliernocc",
    "multiplier_nocc",
    "multiplierv2",  # Legacy alias
    "multiplier_v2",  # Legacy alias
})

MULTIPLIER_WICC_TYPES: frozenset[str] = frozenset({
    "multiplierwicc",
    "multiplier_wicc",
    "multiplier",  # Legacy alias
})


def is_multiplier_nocc(layer_type: str) -> bool:
    """Check if layer_type is a MultiplierNOCC layer (case-insensitive)."""
    return str(layer_type).lower() in MULTIPLIER_NOCC_TYPES


def is_multiplier_wicc(layer_type: str) -> bool:
    """Check if layer_type is a MultiplierWICC layer (case-insensitive)."""
    return str(layer_type).lower() in MULTIPLIER_WICC_TYPES


__all__ = [
    "LAYER_TYPE_MAP",
    "MULTIPLIER_NOCC_TYPES",
    "MULTIPLIER_WICC_TYPES",
    "is_multiplier_nocc",
    "is_multiplier_wicc",
]
