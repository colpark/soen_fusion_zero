"""Protocol definitions for type checking model interfaces."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    from soen_toolkit.core.configs import ConnectionConfig, LayerConfig, SimulationConfig


class ModelProtocol(Protocol):
    """Protocol defining the interface that mixins expect from a SOEN model.

    This protocol is used for type checking to ensure mixins can access
    the attributes and methods they need from the model class.
    """

    # Configuration attributes
    layers_config: list[LayerConfig]
    connections_config: list[ConnectionConfig]
    sim_config: SimulationConfig

    # Model components
    layers: nn.ModuleList
    connections: nn.ParameterDict
    connection_masks: dict[str, torch.Tensor]
    connection_constraints: dict[str, Any]
    connection_constraint_min_matrices: dict[str, torch.Tensor]
    connection_constraint_max_matrices: dict[str, torch.Tensor]

    # Simulation parameters
    dt: float | nn.Parameter
    num_layers: int

    # Optional attributes that may be present
    _load_missing_keys: list[str]
    _load_unexpected_keys: list[str]
    _load_filtered_keys: list[str]

    # Methods from nn.Module
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...
    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, nn.Parameter]]: ...
    def to(self, *args: Any, **kwargs: Any) -> ModelProtocol: ...
    def eval(self) -> ModelProtocol: ...
    def zero_grad(self, set_to_none: bool = True) -> None: ...
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any: ...

    # Model-specific methods
    def reset_stateful_components(self) -> None: ...
    def get_global_connection_mask(self) -> torch.Tensor: ...
