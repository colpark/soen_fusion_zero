"""PyTorch-style imperative API for building SOEN models.

This module provides Graph and Sequential wrappers that offer a more PyTorch-native
interface while maintaining full compatibility with the existing spec-based API.

Example:
    >>> from soen_toolkit.nn import Graph, layers, init, structure
    >>>
    >>> g = Graph(dt=37, network_evaluation_method="layerwise")
    >>> g.add_layer(0, layers.Linear(dim=10))
    >>> g.add_layer(1, layers.SingleDendrite(
    ...     dim=5, solver="FE", source_func_type="RateArray",
    ...     bias_current=1.7, gamma_plus=1e-3, gamma_minus=1e-3
    ... ))
    >>> g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
    >>>
    >>> output = g(input_tensor)

"""

from . import init, layers, param_specs, structure
from .dynamic import dynamic, dynamic_v2
from .graph import Graph
from .sequential import Sequential

__all__ = [
    "Graph",
    "Sequential",
    "dynamic",
    "dynamic_v2",
    "init",
    "layers",
    "param_specs",
    "structure",
]
