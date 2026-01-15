"""Dynamic connection configuration.

DEPRECATED: Use mode parameter with connection_params dict instead.

Dynamic connections use multiplier circuits to create programmable weights
with temporal dynamics.
"""

import warnings

from .specs import DynamicSpec, DynamicV2Spec


def dynamic(
    source_func: str = "RateArray",
    gamma_plus: float = 0.001,
    bias_current: float = 2.0,
    j_in: float = 0.38,
    j_out: float = 0.38,
) -> DynamicSpec:
    """Create a dynamic connection specification.

    .. deprecated::
        Use mode="WICC" with connection_params dict instead:
        g.connect(..., mode="WICC", connection_params={"gamma_plus": 0.001, "bias_current": 2.0})

    Dynamic connections treat matrix entries as per-edge weight fluxes and
    internally integrate hidden multiplier states. This enables on-chip
    reprogrammable weights at the cost of additional computation.

    Args:
        source_func: Source function key (e.g., "RateArray")
        gamma_plus: Drive term gain for multiplier dynamics
        bias_current: Bias current for multiplier circuits
        j_in: Input coupling gain - scales upstream state (default: 0.38)
        j_out: Output coupling gain - scales edge state output (default: 0.38)

    Returns:
        DynamicSpec for use with Graph.connect(..., dynamic=...) [DEPRECATED]

    Example (NEW API - preferred):
        >>> from soen_toolkit.nn import Graph, layers, structure, init
        >>> g = Graph(dt=37, network_evaluation_method="layerwise")
        >>> g.add_layer(0, layers.Linear(dim=10))
        >>> g.add_layer(1, layers.SingleDendrite(dim=5, ...))
        >>> g.connect(
        ...     0, 1,
        ...     structure=structure.dense(),
        ...     init=init.uniform(-0.2, 0.2),
        ...     mode="WICC",
        ...     connection_params={"gamma_plus": 1e-3, "bias_current": 2.0}
        ... )

    """
    warnings.warn(
        "dynamic() is deprecated. Use mode='WICC' with connection_params dict instead. Example: mode='WICC', connection_params={'gamma_plus': 0.001, 'bias_current': 2.0}",
        DeprecationWarning,
        stacklevel=2,
    )
    return DynamicSpec(
        source_func=source_func,
        gamma_plus=gamma_plus,
        bias_current=bias_current,
        j_in=j_in,
        j_out=j_out,
    )


def dynamic_v2(
    source_func: str = "RateArray",
    alpha: float = 1.64053,
    beta: float = 303.85,
    beta_out: float = 91.156,
    bias_current: float = 2.1,
    j_in: float = 0.38,
    j_out: float = 0.38,
) -> DynamicV2Spec:
    """Create a dynamic v2 connection specification.

    .. deprecated::
        Use mode="NOCC" with connection_params dict instead:
        g.connect(..., mode="NOCC", connection_params={"alpha": 1.5, "beta": 303.85})

    Dynamic v2 connections use a multiplier circuit with dual SQUID
    states and aggregated output. This version supports a hardware design
    without collection coils.

    Physical parameter mappings:
        - beta_1 ≈ 1nH → beta = 303.85
        - beta_out ≈ 300pH → beta_out = 91.156
        - i_b ≈ 210μA → ib = 2.1
        - R ≈ 2Ω → alpha = 1.64053

    Args:
        source_func: Source function key (e.g., "RateArray")
        alpha: Dimensionless resistance (default: 1.64053)
        beta: Inductance of incoming branches (default: 303.85)
        beta_out: Inductance of output branch (default: 91.156)
        bias_current: Bias current (default: 2.1)
        j_in: Input coupling gain - scales upstream state (default: 0.38)
        j_out: Output coupling gain - scales aggregated m state (default: 0.38)

    Returns:
        DynamicV2Spec for use with Graph.connect(..., dynamic=...) [DEPRECATED]

    Example (NEW API - preferred):
        >>> from soen_toolkit.nn import Graph, layers, structure, init
        >>> g = Graph(dt=37, network_evaluation_method="layerwise")
        >>> g.add_layer(0, layers.Linear(dim=10))
        >>> g.add_layer(1, layers.SingleDendrite(dim=5, ...))
        >>> g.connect(
        ...     0, 1,
        ...     structure=structure.dense(),
        ...     init=init.uniform(-0.2, 0.2),
        ...     mode="NOCC",
        ...     connection_params={"alpha": 1.64, "beta": 303.85, "beta_out": 91.156, "bias_current": 2.1}
        ... )
    """
    warnings.warn(
        "dynamic_v2() is deprecated. Use mode='NOCC' with connection_params dict instead. Example: mode='NOCC', connection_params={'alpha': 1.5, 'beta': 303.85}",
        DeprecationWarning,
        stacklevel=2,
    )
    return DynamicV2Spec(
        source_func=source_func,
        alpha=alpha,
        beta=beta,
        beta_out=beta_out,
        bias_current=bias_current,
        j_in=j_in,
        j_out=j_out,
    )
