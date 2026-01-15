"""Spec definitions for PyTorch-style API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


@dataclass
class InitSpec:
    """Weight initialization specification."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to ConnectionConfig-compatible params dict."""
        return {"name": self.name, "params": self.params}


@dataclass
class StructureSpec:
    """Connection structure specification."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    allow_self_connections: bool = True
    visualization_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to ConnectionConfig-compatible params dict."""
        return {"type": self.type, "params": self.params}


@dataclass
class DynamicSpec:
    """Dynamic connection specification."""

    source_func: str = "RateArray"
    gamma_plus: float = 0.001
    bias_current: float = 2.0
    j_in: float = 0.38
    j_out: float = 0.38

    def to_dict(self) -> dict[str, Any]:
        """Convert to ConnectionConfig-compatible params dict."""
        return {
            "source_func": self.source_func,
            "gamma_plus": self.gamma_plus,
            "bias_current": self.bias_current,
            "j_in": self.j_in,
            "j_out": self.j_out,
        }


@dataclass
class DynamicV2Spec:
    """Dynamic connection specification for multiplier v2."""

    source_func: str = "RateArray"
    alpha: float = 1.64053
    beta: float = 303.85
    beta_out: float = 91.156
    bias_current: float = 2.1
    j_in: float = 0.38
    j_out: float = 0.38

    def to_dict(self) -> dict[str, Any]:
        """Convert to ConnectionConfig-compatible params dict."""
        return {
            "source_func": self.source_func,
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_out": self.beta_out,
            "bias_current": self.bias_current,
            "j_in": self.j_in,
            "j_out": self.j_out,
        }


@dataclass
class LayerSpec:
    """Internal layer specification for Graph."""

    layer_id: int
    module: nn.Module | None
    layer_type: str
    params: dict[str, Any]
    learnable_params: dict[str, bool] | None = None
    description: str = ""


@dataclass
class ConnectionSpec:
    """Internal connection specification for Graph."""

    from_layer: int
    to_layer: int
    structure: StructureSpec
    init: InitSpec
    mode: str = "fixed"
    connection_params: dict[str, Any] | None = None
    dynamic: DynamicSpec | DynamicV2Spec | None = None  # Deprecated - kept for backward compatibility
    constraints: dict[str, float] | None = None
    learnable: bool = True
    mask: Any | None = None
    allow_self_connections: bool = True
    visualization_metadata: dict[str, Any] | None = None
