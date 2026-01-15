"""Layer factory functions for PyTorch-style API.

These functions return lightweight wrappers that capture layer configuration.
They are not actual nn.Module instances but specifications that Graph converts
to LayerConfig objects internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .param_specs import ParamSpec

# Type alias for parameters that can be float, ParamSpec, or dict
type ParamValue = float | "ParamSpec" | dict[str, Any]


class LayerFactory:
    """Base class for layer specifications."""

    def __init__(self, layer_type: str, **params: Any) -> None:
        self.layer_type = layer_type
        self.params = params

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.layer_type}({param_str})"


def SingleDendrite(
    dim: int,
    solver: str = "FE",
    source_func_type: str = "RateArray",
    bias_current: ParamValue = 1.7,
    phi_offset: ParamValue = 0.0,
    gamma_plus: ParamValue = 0.001,
    gamma_minus: ParamValue = 0.001,
    **kwargs: Any,
) -> LayerFactory:
    """SingleDendrite layer specification.

    The fundamental SOEN computational unit with temporal integration and
    nonlinear dynamics.

    Args:
        dim: Number of dendrites
        solver: Integration method ("FE" or "PS")
        source_func_type: Source function key (e.g., "RateArray")
        bias_current: Dimensionless bias current (threshold)
        phi_offset: Flux offset
        gamma_plus: Drive term gain
        gamma_minus: Leak rate
        **kwargs: Additional parameters (e.g., learnable_params, constraints)

    Returns:
        LayerFactory for SingleDendrite

    """
    params = {
        "dim": dim,
        "solver": solver,
        "source_func_type": source_func_type,
        "bias_current": bias_current,
        "phi_offset": phi_offset,
        "gamma_plus": gamma_plus,
        "gamma_minus": gamma_minus,
        **kwargs,
    }
    return LayerFactory("SingleDendrite", **params)


def MultiplierWICC(
    dim: int,
    solver: str = "FE",
    source_func_type: str = "RateArray",
    phi_y: ParamValue = 0.1,
    bias_current: ParamValue = 2.0,
    gamma_plus: ParamValue = 0.001,
    gamma_minus: ParamValue = 0.001,
    **kwargs: Any,
) -> LayerFactory:
    """MultiplierWICC (With Collection Coil) layer specification.

    Circuit that computes approximate analog multiplication of two input fluxes
    using V1 multiplier physics with collection coil.

    Args:
        dim: Number of multiplier circuits
        solver: Integration method (only "FE" supported)
        source_func_type: Source function key
        phi_y: Secondary input term
        bias_current: Bias current
        gamma_plus: Drive term gain
        gamma_minus: Leak/damping term
        **kwargs: Additional parameters

    Returns:
        LayerFactory for MultiplierWICC

    """
    params = {
        "dim": dim,
        "solver": solver,
        "source_func_type": source_func_type,
        "phi_y": phi_y,
        "bias_current": bias_current,
        "gamma_plus": gamma_plus,
        "gamma_minus": gamma_minus,
        **kwargs,
    }
    return LayerFactory("MultiplierWICC", **params)


# Legacy alias for backward compatibility
def Multiplier(
    dim: int,
    solver: str = "FE",
    source_func_type: str = "RateArray",
    phi_y: ParamValue = 0.1,
    bias_current: ParamValue = 2.0,
    gamma_plus: ParamValue = 0.001,
    gamma_minus: ParamValue = 0.001,
    **kwargs: Any,
) -> LayerFactory:
    """Legacy alias for MultiplierWICC.

    Deprecated: Use MultiplierWICC instead for clarity.
    This alias is maintained for backward compatibility.
    """
    return MultiplierWICC(
        dim=dim,
        solver=solver,
        source_func_type=source_func_type,
        phi_y=phi_y,
        bias_current=bias_current,
        gamma_plus=gamma_plus,
        gamma_minus=gamma_minus,
        **kwargs,
    )


def MultiplierNOCC(
    dim: int,
    solver: str = "FE",
    source_func_type: str = "RateArray",
    phi_y: ParamValue = 0.1,
    bias_current: ParamValue = 2.1,
    alpha: ParamValue = 1.64053,
    beta: ParamValue = 303.85,
    beta_out: ParamValue = 91.156,
    **kwargs: Any,
) -> LayerFactory:
    """Multiplier V2 layer specification.

    New multiplier circuit with dual SQUID states and aggregated output.
    Uses a different flux collection mechanism without collection coils.

    Recommended physical parameter mappings:
        - beta_1 ≈ 1nH → beta = 303.85
        - beta_out ≈ 300pH → beta_out = 91.156
        - i_b ≈ 210μA → ib = 2.1
        - R ≈ 2Ω → alpha = 1.64053

    Args:
        dim: Number of multiplier circuits
        solver: Integration method (only "FE" supported)
        source_func_type: Source function key
        phi_y: Secondary input/weight term
        bias_current: Bias current (default: 2.1)
        alpha: Dimensionless resistance (default: 1.64053)
        beta: Inductance of incoming branches (default: 303.85)
        beta_out: Inductance of output branch (default: 91.156)
        **kwargs: Additional parameters

    Returns:
        LayerFactory for MultiplierNOCC
    """
    params = {
        "dim": dim,
        "solver": solver,
        "source_func_type": source_func_type,
        "phi_y": phi_y,
        "bias_current": bias_current,
        "alpha": alpha,
        "beta": beta,
        "beta_out": beta_out,
        **kwargs,
    }
    return LayerFactory("MultiplierNOCC", **params)


def DendriteReadout(
    dim: int,
    source_func_type: str = "RateArray",
    bias_current: ParamValue = 1.7,
    phi_offset: ParamValue = 0.0,
    **kwargs: Any,
) -> LayerFactory:
    """DendriteReadout layer specification.

    Specialized readout that outputs source function values without integration.

    Args:
        dim: Number of readout circuits
        source_func_type: Source function key
        bias_current: Bias current
        phi_offset: Flux offset
        **kwargs: Additional parameters

    Returns:
        LayerFactory for DendriteReadout

    """
    params = {
        "dim": dim,
        "source_func_type": source_func_type,
        "bias_current": bias_current,
        "phi_offset": phi_offset,
        **kwargs,
    }
    return LayerFactory("DendriteReadout", **params)


def Linear(dim: int, **kwargs: Any) -> LayerFactory:
    """Linear (passthrough) layer specification.

    Simple layer with no dynamics. Commonly used as input layer (layer 0).

    Args:
        dim: Number of nodes
        **kwargs: Additional parameters

    Returns:
        LayerFactory for Linear

    """
    params = {"dim": dim, **kwargs}
    return LayerFactory("Linear", **params)


def NonLinear(
    dim: int,
    source_func_type: str = "Tanh",
    phi_offset: ParamValue = 0.0,
    bias_current: ParamValue = 1.7,
    **kwargs: Any,
) -> LayerFactory:
    """NonLinear layer specification.

    Applies a configurable nonlinearity without temporal dynamics.

    Args:
        dim: Number of nodes
        source_func_type: Activation function (e.g., "Tanh", "SimpleGELU")
        phi_offset: Input shift before nonlinearity
        bias_current: Passed to source function if applicable
        **kwargs: Additional parameters

    Returns:
        LayerFactory for NonLinear

    """
    params = {
        "dim": dim,
        "source_func_type": source_func_type,
        "phi_offset": phi_offset,
        "bias_current": bias_current,
        **kwargs,
    }
    return LayerFactory("NonLinear", **params)


def ScalingLayer(dim: int, scale_factor: ParamValue = 1.0, **kwargs: Any) -> LayerFactory:
    """ScalingLayer specification.

    Applies learnable per-feature scaling.

    Args:
        dim: Number of features
        scale_factor: Initial scaling value
        **kwargs: Additional parameters

    Returns:
        LayerFactory for ScalingLayer

    """
    params = {"dim": dim, "scale_factor": scale_factor, **kwargs}
    return LayerFactory("ScalingLayer", **params)


def RNN(dim: int, **kwargs: Any) -> LayerFactory:
    """RNN layer specification.

    Standard RNN wrapped for compatibility.

    Args:
        dim: Hidden size (also used as input size)
        **kwargs: Additional parameters

    Returns:
        LayerFactory for RNN

    """
    params = {"dim": dim, **kwargs}
    return LayerFactory("RNN", **params)


def LSTM(dim: int, **kwargs: Any) -> LayerFactory:
    """LSTM layer specification.

    Standard LSTM wrapped for compatibility.

    Args:
        dim: Hidden size (also used as input size)
        **kwargs: Additional parameters

    Returns:
        LayerFactory for LSTM

    """
    params = {"dim": dim, **kwargs}
    return LayerFactory("LSTM", **params)


def GRU(dim: int, **kwargs: Any) -> LayerFactory:
    """GRU layer specification.

    Standard GRU wrapped for compatibility.

    Args:
        dim: Hidden size (also used as input size)
        **kwargs: Additional parameters

    Returns:
        LayerFactory for GRU

    """
    params = {"dim": dim, **kwargs}
    return LayerFactory("GRU", **params)


def LeakyGRU(dim: int, **kwargs: Any) -> LayerFactory:
    """LeakyGRU layer specification.

    Leaky-integrator GRU-style recurrence with per-unit trainable time constants.
    Backed by a masked-gates fused GRU kernel under the hood.

    Args:
        dim: Hidden size (also used as input size)
        **kwargs: Additional parameters passed into the LeakyGRU layer constructor
            (e.g., tau_init, tau_spacing, candidate_diag, train_alpha).

    Returns:
        LayerFactory for LeakyGRU
    """

    params = {"dim": dim, **kwargs}
    return LayerFactory("LeakyGRU", **params)


def MinGRU(dim: int, **kwargs: Any) -> LayerFactory:
    """MinGRU layer specification.

    Lightweight, parallelizable GRU variant.

    Args:
        dim: Hidden size (also used as input size)
        **kwargs: Additional parameters

    Returns:
        LayerFactory for MinGRU

    """
    params = {"dim": dim, **kwargs}
    return LayerFactory("MinGRU", **params)
