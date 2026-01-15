"""Shared parameter definitions for next-gen SOEN layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import nn

from soen_toolkit.utils.physical_mappings.soen_conversion_utils import beta_from_inductance

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Constraint:
    """Simple box constraint description for a parameter."""

    min: float | None = None
    max: float | None = None


@dataclass(slots=True)
class InitializerSpec:
    """Describe how a parameter is initialised."""

    method: str = "constant"
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParameterDef:
    """Declarative description of a learnable or fixed parameter."""

    name: str
    default: float | torch.Tensor
    learnable: bool = True
    constraint: Constraint | None = None
    initializer: InitializerSpec | None = None
    transform: str | None = None  # e.g., "log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEF_EPS = 1e-10


def _linear_initializer(width: int, *, params: Mapping[str, Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    min_val = float(params.get("min", 0.0))
    max_val = float(params.get("max", 1.0))
    return torch.linspace(min_val, max_val, width, dtype=dtype, device=device)


def _loglinear_initializer(width: int, *, params: Mapping[str, Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    min_val = float(params.get("min", _DEF_EPS))
    max_val = float(params.get("max", 1.0))
    min_val = max(min_val, _DEF_EPS)
    if max_val <= min_val:
        max_val = min_val * (1.0 + 1e-9)
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    values = torch.linspace(log_min, log_max, width, dtype=dtype, device=device)
    return torch.exp(values)


def _constant_initializer(width: int, *, params: Mapping[str, Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    value = float(params.get("value", 0.0))
    return torch.full((width,), value, dtype=dtype, device=device)


def _normal_initializer(width: int, *, params: Mapping[str, Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mean = float(params.get("mean", 0.0))
    std = float(params.get("std", 0.01))
    return torch.normal(mean=mean, std=std, size=(width,), dtype=dtype, device=device)


def _uniform_initializer(width: int, *, params: Mapping[str, Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    min_val = float(params.get("min", 0.0))
    max_val = float(params.get("max", 1.0))
    return torch.empty(width, dtype=dtype, device=device).uniform_(min_val, max_val)


def _lognormal_initializer(width: int, *, params: Mapping[str, Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mean = float(params.get("mean", 0.0))
    std = float(params.get("std", 1.0))
    return torch.exp(torch.normal(mean=mean, std=std, size=(width,), dtype=dtype, device=device))


def _loguniform_initializer(width: int, *, params: Mapping[str, Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    min_val = float(params.get("min", _DEF_EPS))
    max_val = float(params.get("max", 1.0))
    min_val = max(min_val, _DEF_EPS)
    max_val = max(max_val, min_val + _DEF_EPS)
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    return torch.exp(torch.empty(width, dtype=dtype, device=device).uniform_(log_min, log_max))


def _fan_out_initializer(width: int, *, params: Mapping[str, Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Compute gamma_plus values based on fan-out and inductance.

    Args:
        width: Number of nodes to initialize.
        params: Initialization parameters:
            - inductance_per_fan: Inductance in Henries per fan-out (default 0.5nH)
            - node_fan_outs: List of fan-out counts per node
        dtype: Tensor dtype.
        device: Tensor device.

    The formula is: gamma_plus[i] = 1 / (beta_L * fan_out[i])
    where beta_L = beta_from_inductance(inductance_per_fan)
    """
    # Convert inductance (H) to dimensionless beta_L
    inductance_per_fan = float(params.get("inductance_per_fan", 5e-10))  # Default 0.5nH
    beta_L = beta_from_inductance(inductance_per_fan)

    node_fan_outs = params.get("node_fan_outs")
    if node_fan_outs is None:
        node_fan_outs = [1] * width
    if len(node_fan_outs) != width:
        node_fan_outs = [1] * width
    values = torch.empty(width, dtype=dtype, device=device)
    for i, fan_out in enumerate(node_fan_outs):
        fan = max(1, int(fan_out))
        values[i] = 1.0 / (beta_L * fan)
    return values


_INITIALISERS = {
    "constant": _constant_initializer,
    "normal": _normal_initializer,
    "uniform": _uniform_initializer,
    "lognormal": _lognormal_initializer,
    "loguniform": _loguniform_initializer,
    "linear": _linear_initializer,
    "loglinear": _loglinear_initializer,
    "fan_out": _fan_out_initializer,
}


def _init_with_blocking(
    width: int,
    method: str,
    params: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Run an initializer with optional block_size and block_mode support.

    Two blocking modes are available:
    - "shared" (default): Nodes within a block share the same value.
      Generates one value per block, then repeats to fill the width.
      Example: block_size=3, width=12 -> [v1,v1,v1, v2,v2,v2, v3,v3,v3, v4,v4,v4]

    - "tiled": A pattern of values is tiled (repeated) across blocks.
      Generates block_size values, then tiles them across all blocks.
      Example: block_size=3, width=12 -> [v1,v2,v3, v1,v2,v3, v1,v2,v3, v1,v2,v3]

    Args:
        width: Total number of nodes in the layer.
        method: Initializer method name (must be in _INITIALISERS).
        params: Initializer parameters. May include:
            - 'block_size' (default 1): Size of each block.
            - 'block_mode' (default "shared"): Either "shared" or "tiled".
        dtype: Tensor dtype.
        device: Tensor device.

    Returns:
        Tensor of shape (width,) with initialized values.

    Raises:
        ValueError: If width is not evenly divisible by block_size,
                    or if block_mode is invalid.
    """
    # Extract blocking parameters (don't pass them to the base initializer)
    block_size = int(params.pop("block_size", 1))
    block_mode = str(params.pop("block_mode", "shared"))

    if block_size < 1:
        msg = f"block_size must be >= 1, got {block_size}"
        raise ValueError(msg)

    if block_mode not in ("shared", "tiled"):
        msg = f"block_mode must be 'shared' or 'tiled', got '{block_mode}'"
        raise ValueError(msg)

    initializer = _INITIALISERS[method]

    if block_size == 1:
        # No blocking, standard behavior (mode is irrelevant)
        return initializer(width, params=params, dtype=dtype, device=device)

    # Validate divisibility
    if width % block_size != 0:
        msg = f"Layer width {width} is not evenly divisible by block_size {block_size}"
        raise ValueError(msg)

    num_blocks = width // block_size

    if block_mode == "shared":
        # Generate one value per block, repeat within each block
        block_values = initializer(num_blocks, params=params, dtype=dtype, device=device)
        return block_values.repeat_interleave(block_size)
    else:
        # block_mode == "tiled"
        # Generate block_size values (one per position within a block), tile across blocks
        pattern_values = initializer(block_size, params=params, dtype=dtype, device=device)
        return pattern_values.tile((num_blocks,))


class ParameterRegistry:
    """Registry that materialises parameter definitions onto an ``nn.Module``."""

    def __init__(
        self,
        module: nn.Module,
        *,
        width: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        init_context: Mapping[str, Any] | None = None,
    ) -> None:
        self._module = module
        self._width = int(width)
        self._dtype = dtype
        self._device = device
        self._defs: dict[str, ParameterDef] = {}
        self._attr_names: dict[str, str] = {}
        self._init_context = init_context or {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def add(self, definition: ParameterDef) -> None:
        name = definition.name
        if name in self._defs:
            msg = f"Parameter '{name}' registered twice"
            raise ValueError(msg)

        tensor = self._materialise(definition)
        attr_name = name
        if definition.transform == "log":
            tensor = torch.log(tensor.clamp_min(_DEF_EPS))
            attr_name = f"log_{name}"

        if definition.learnable:
            param = nn.Parameter(tensor)
            self._module.register_parameter(attr_name, param)
        else:
            self._module.register_buffer(attr_name, tensor, persistent=True)

        self._defs[name] = definition
        self._attr_names[name] = attr_name

        # Create property with getter AND setter for log-transformed parameters
        # This allows users to write `layer.gamma_minus = 0.005` instead of
        # manually accessing `layer.log_gamma_minus.data.copy_(torch.log(...))`
        if definition.transform == "log" and not hasattr(self._module.__class__, name):

            def _getter(instance, attr=name):
                raw = getattr(instance, f"log_{attr}")
                return torch.exp(raw)

            def _setter(instance, value, attr=name):
                log_attr = f"log_{attr}"
                raw = getattr(instance, log_attr)
                with torch.no_grad():
                    if isinstance(value, torch.Tensor):
                        new_val = torch.log(value.clamp_min(_DEF_EPS))
                        new_val = new_val.to(dtype=raw.dtype, device=raw.device)
                        if new_val.shape != raw.shape:
                            new_val = new_val.expand_as(raw)
                    else:
                        new_val = torch.full_like(raw, math.log(max(float(value), _DEF_EPS)))
                    raw.data.copy_(new_val)

            setattr(self._module.__class__, name, property(_getter, _setter))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def named_tensors(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, definition in self._defs.items():
            tensor = getattr(self._module, self._attr_names[name])
            if definition.transform == "log":
                tensor = torch.exp(tensor)
            out[name] = tensor
        return out

    def apply_constraints(self) -> None:
        for name, definition in self._defs.items():
            constraint = definition.constraint
            if constraint is None:
                continue
            tensor = getattr(self._module, self._attr_names[name])
            min_v = constraint.min
            max_v = constraint.max
            with torch.no_grad():
                if definition.transform == "log":
                    if min_v is not None:
                        tensor.clamp_(min=math.log(max(min_v, _DEF_EPS)))
                    if max_v is not None:
                        tensor.clamp_(max=math.log(max(max_v, _DEF_EPS)))
                else:
                    if min_v is not None:
                        tensor.clamp_(min=min_v)
                    if max_v is not None:
                        tensor.clamp_(max=max_v)

    def override_parameter(
        self,
        name: str,
        *,
        method: str | None = None,
        params: Mapping[str, Any] | None = None,
        value: float | torch.Tensor | None = None,
        learnable: bool | None = None,
        constraint: Constraint | Mapping[str, Any] | None = None,
    ) -> None:
        """Apply runtime overrides coming from GUI / spec metadata."""
        if name not in self._defs:
            msg = f"Unknown parameter '{name}'"
            raise ValueError(msg)

        definition = self._defs[name]
        attr_name = self._attr_names[name]
        tensor = getattr(self._module, attr_name)

        # Normalise constraint input (dict -> Constraint dataclass)
        if isinstance(constraint, Mapping):
            constraint = Constraint(
                min=constraint.get("min"),
                max=constraint.get("max"),
            )

        with torch.no_grad():
            target: torch.Tensor | None = None

            if method is not None:
                init_method = str(method).lower()
                init_params = dict(params or {})
                if init_method == "constant" and "value" not in init_params and value is not None:
                    init_params["value"] = value
                if init_method == "fan_out" and "node_fan_outs" not in init_params:
                    context_fan = self._init_context.get("node_fan_outs")
                    if context_fan is not None:
                        init_params["node_fan_outs"] = context_fan

                if init_method not in _INITIALISERS:
                    msg = f"Unknown initializer '{method}' for parameter '{name}'"
                    raise ValueError(msg)

                raw = _init_with_blocking(
                    tensor.numel(),
                    method=init_method,
                    params=init_params,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                target = raw.view_as(tensor)
            elif value is not None:
                if isinstance(value, torch.Tensor):
                    raw = value.to(dtype=tensor.dtype, device=tensor.device)
                    if raw.shape != tensor.shape:
                        raw = raw.expand_as(tensor)
                    target = raw.clone()
                else:
                    target = torch.full_like(tensor, float(value))

            if target is not None:
                if definition.transform == "log":
                    target = torch.log(target.clamp_min(_DEF_EPS))
                tensor.copy_(target)

            if isinstance(tensor, nn.Parameter) and learnable is not None:
                tensor.requires_grad = bool(learnable)

        if constraint is not None:
            self._defs[name].constraint = constraint

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _materialise(self, definition: ParameterDef) -> torch.Tensor:
        device = self._device if self._device is not None else torch.device("cpu")
        initializer = definition.initializer
        if initializer is None:
            initializer = InitializerSpec(method="constant", params={"value": definition.default})

        method = initializer.method.lower()
        params = dict(initializer.params)
        # allow defaults to influence constant initialiser
        if method == "constant" and "value" not in params:
            params["value"] = definition.default
        if method == "fan_out" and "node_fan_outs" not in params:
            params["node_fan_outs"] = self._init_context.get("node_fan_outs")

        if method not in _INITIALISERS:
            msg = f"Unknown initializer '{method}' for parameter '{definition.name}'"
            raise ValueError(msg)

        return _init_with_blocking(
            self._width,
            method=method,
            params=params,
            dtype=self._dtype,
            device=device,
        )


__all__ = ["Constraint", "InitializerSpec", "ParameterDef", "ParameterRegistry"]
