"""Shared parameter metadata and helpers for SOEN layers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from soen_toolkit.utils.physical_mappings.soen_conversion_utils import beta_from_inductance


@dataclass
class ParamConfig:
    """Declarative configuration describing a single scalar parameter."""

    name: str
    is_log_param: bool = False
    default_value: float = 0.0
    learnable: bool = True
    min_value: float | None = None
    max_value: float | None = None


LAYER_PARAM_CONFIGS: dict[str, list[ParamConfig]] = {
    "SingleDendrite": [
        ParamConfig(name="phi_offset", default_value=0.23),
        ParamConfig(name="bias_current", default_value=1.7, min_value=0.0),
        ParamConfig(name="gamma_plus", default_value=0.001, is_log_param=True),
        ParamConfig(name="gamma_minus", default_value=0.001, is_log_param=True),
    ],
    "Soma": [
        ParamConfig(name="phi_offset", default_value=0.23),
        ParamConfig(name="bias_current", default_value=1.7, min_value=0.0),
        ParamConfig(name="gamma_plus", default_value=0.001, is_log_param=True),
        ParamConfig(name="gamma_minus", default_value=0.001, is_log_param=True),
        ParamConfig(name="threshold", default_value=0.0),
    ],
    "Multiplier": [
        ParamConfig(name="phi_y", default_value=0.1),
        ParamConfig(name="bias_current", default_value=2.0, min_value=0.0),
        ParamConfig(name="gamma_plus", default_value=0.001, is_log_param=True),
        ParamConfig(name="gamma_minus", default_value=0.001, is_log_param=True),
    ],
    "MultiplierNOCC": [
        ParamConfig(name="phi_y", default_value=0.1),
        ParamConfig(name="bias_current", default_value=2.1, min_value=0.0),
        ParamConfig(name="alpha", default_value=1.64053, min_value=0.0),
        ParamConfig(name="beta", default_value=303.85, min_value=0.0),
        ParamConfig(name="beta_out", default_value=91.156, min_value=0.0),
    ],
    "Input": [],
    "MinGRU": [],
    "RNN": [],
    "LSTM": [],
    "GRU": [],
    "LeakyGRU": [
        ParamConfig(name="decay_bias", default_value=4.0, learnable=True),  # Maps to bias_z
        ParamConfig(name="candidate_bias", default_value=0.0, learnable=True),  # Maps to bias_n
    ],
    # Backward-compatible alias
    "leakyGRU": [],
    "Classifier": [ParamConfig(name="classifier_bias", default_value=0.0001, learnable=True)],
    "ScalingLayer": [ParamConfig(name="scale_factor", default_value=10.0, learnable=True)],
    "Softmax": [ParamConfig(name="beta", default_value=1.0, learnable=True, min_value=0.01)],
    "Synapse": [ParamConfig(name="alpha", default_value=0.9, learnable=True, min_value=0.0, max_value=1.0)],
    "DendriteReadout": [
        ParamConfig(name="phi_offset", default_value=0.0),
        ParamConfig(name="bias_current", default_value=1.7, min_value=0.0),
    ],
    # Legacy alias maintained for backward compatibility
    "Readout": [
        ParamConfig(name="phi_offset", default_value=0.0),
        ParamConfig(name="bias_current", default_value=1.7, min_value=0.0),
    ],
}


INIT_METHOD_PARAMS: dict[str, list[str]] = {
    "constant": ["value"],
    "linear": ["min", "max"],
    "fan_out": ["inductance_per_fan"],
    "normal": ["mean", "std"],
    "uniform": ["min", "max"],
    "lognormal": ["mean", "std"],
    "loguniform": ["min", "max"],
    "loglinear": ["min", "max"],
}

# Optional parameters that apply to all initialization methods
INIT_OPTIONAL_PARAMS: list[str] = ["block_size", "block_mode"]


def _validate_and_get_blocking_params(
    shape: torch.Size | tuple,
    params: dict[str, Any],
) -> tuple[int, int, str]:
    """Extract block_size, block_mode and validate divisibility.

    Two blocking modes are available:
    - "shared": Nodes within a block share the same value.
      Returns (block_size, num_blocks, "shared") where num_blocks values are generated.
    - "tiled": A pattern of values is tiled across blocks.
      Returns (block_size, num_blocks, "tiled") where block_size values are generated.

    Args:
        shape: Shape of the tensor to initialize (expects 1D shape).
        params: Init params dict. block_size and block_mode will be popped if present.

    Returns:
        Tuple of (block_size, num_blocks, block_mode).

    Raises:
        ValueError: If width is not evenly divisible by block_size,
                    or if block_mode is invalid.
    """
    block_size = int(params.pop("block_size", 1))
    block_mode = str(params.pop("block_mode", "shared"))

    if block_size < 1:
        msg = f"block_size must be >= 1, got {block_size}"
        raise ValueError(msg)

    if block_mode not in ("shared", "tiled"):
        msg = f"block_mode must be 'shared' or 'tiled', got '{block_mode}'"
        raise ValueError(msg)

    width = shape[0] if len(shape) == 1 else int(torch.tensor(shape).prod().item())

    if block_size == 1:
        return 1, width, block_mode

    if width % block_size != 0:
        msg = f"Layer width {width} is not evenly divisible by block_size {block_size}"
        raise ValueError(msg)

    return block_size, width // block_size, block_mode


def initialize_values_deterministic(
    shape: torch.Size | tuple,
    method: str,
    params: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    # Handle blocking: extract block_size, num_blocks, and block_mode
    params = dict(params)  # Don't mutate the original
    block_size, num_blocks, block_mode = _validate_and_get_blocking_params(shape, params)

    # Determine how many values to generate based on block_mode
    # - "shared": generate one value per block (num_blocks values)
    # - "tiled": generate one value per position within a block (block_size values)
    if block_mode == "tiled" and block_size > 1:
        gen_count = block_size
    else:
        gen_count = num_blocks

    # Generate values
    if method == "constant":
        value = params.get("value", 0.0)
        block_values = torch.full((gen_count,), value, device=device)

    elif method == "linear":
        min_val = params.get("min", 0.0)
        max_val = params.get("max", 1.0)
        block_values = torch.linspace(min_val, max_val, gen_count, device=device)

    elif method == "fan_out":
        # Convert inductance (H) to dimensionless beta_L
        inductance_per_fan = float(params.get("inductance_per_fan", 5e-10))  # Default 0.5nH
        beta_L = beta_from_inductance(inductance_per_fan)
        node_fan_outs = params.get("node_fan_outs")
        # For fan_out with blocking, use representative fan_out values
        if node_fan_outs is not None and len(node_fan_outs) >= gen_count:
            if block_mode == "tiled" and block_size > 1:
                # Use first block_size fan_outs (one per position in the pattern)
                fan_outs = node_fan_outs[:gen_count]
            else:
                # Use first fan_out from each block
                fan_outs = [node_fan_outs[i * block_size] for i in range(gen_count)]
        else:
            fan_outs = [1] * gen_count
        block_values = torch.empty(gen_count, device=device)
        for i, fan in enumerate(fan_outs):
            block_values[i] = 1.0 / (beta_L * max(1, fan))

    elif method == "loglinear":
        min_val = params.get("min", 0.0)
        max_val = params.get("max", 1.0)
        log_min = math.log(min_val) if min_val > 0 else 0.0
        log_max = math.log(max_val) if max_val > 0 else 0.0
        block_values = torch.exp(torch.linspace(log_min, log_max, gen_count, device=device))

    else:
        # Fallback
        block_values = torch.full((gen_count,), 0.0, device=device)

    # Apply blocking expansion
    if block_size > 1:
        if block_mode == "tiled":
            # Tile the pattern across all blocks
            values = block_values.tile((num_blocks,))
        else:
            # Repeat each value block_size times (shared mode)
            values = block_values.repeat_interleave(block_size)
    else:
        values = block_values

    # Reshape to original shape if needed
    if len(shape) == 1:
        return values
    return values.view(shape)


def initialize_values_stochastic(
    shape: torch.Size | tuple,
    method: str,
    params: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    # Handle blocking: extract block_size, num_blocks, and block_mode
    params = dict(params)  # Don't mutate the original
    block_size, num_blocks, block_mode = _validate_and_get_blocking_params(shape, params)

    # Determine how many values to generate based on block_mode
    # - "shared": generate one value per block (num_blocks values)
    # - "tiled": generate one value per position within a block (block_size values)
    if block_mode == "tiled" and block_size > 1:
        gen_count = block_size
    else:
        gen_count = num_blocks

    # Generate values
    if method == "normal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 0.01)
        block_values = torch.normal(mean=mean, std=std, size=(gen_count,), device=device)

    elif method == "uniform":
        min_val = params.get("min", 0.0)
        max_val = params.get("max", 1.0)
        block_values = torch.empty(gen_count, device=device).uniform_(min_val, max_val)

    elif method == "lognormal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        block_values = torch.exp(torch.normal(mean=mean, std=std, size=(gen_count,), device=device))

    elif method == "loguniform":
        eps = 1e-10
        min_val = max(params.get("min", 0.0), eps)
        max_val = max(params.get("max", 1.0), eps)
        log_min = math.log(min_val)
        log_max = math.log(max_val)
        if log_max < log_min:
            log_min, log_max = log_max, log_min
        block_values = torch.exp(torch.empty(gen_count, device=device).uniform_(log_min, log_max))

    else:
        # Fallback
        block_values = torch.normal(mean=0.0, std=0.01, size=(gen_count,), device=device)

    # Apply blocking expansion
    if block_size > 1:
        if block_mode == "tiled":
            # Tile the pattern across all blocks
            values = block_values.tile((num_blocks,))
        else:
            # Repeat each value block_size times (shared mode)
            values = block_values.repeat_interleave(block_size)
    else:
        values = block_values

    # Reshape to original shape if needed
    if len(shape) == 1:
        return values
    return values.view(shape)


def parse_param_config(
    param_name: str,
    config_value: Any,
    node_fan_outs: list[int] | None = None,
) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "init_method": "normal",
        "init_params": {},
        "min_value": None,
        "max_value": None,
        "learnable": True,
    }

    if param_name.startswith("gamma_plus") and config_value is None and node_fan_outs is not None:
        parsed["init_method"] = "fan_out"
        # Default inductance per fan-out: 0.5nH
        parsed["init_params"] = {"inductance_per_fan": 5e-10, "node_fan_outs": node_fan_outs}
        return parsed

    if isinstance(config_value, dict):
        distribution = config_value.get("distribution")
        if distribution:
            parsed["init_method"] = distribution
            parsed["init_params"] = dict(config_value.get("params", {}))
        elif "node_fan_outs" in config_value:
            parsed["init_method"] = "fan_out"
            parsed["init_params"] = {
                "inductance_per_fan": config_value.get("inductance_per_fan", 5e-10),
                "node_fan_outs": config_value["node_fan_outs"],
            }
        elif {"mean", "std"}.issubset(config_value.keys()):
            parsed["init_method"] = "normal"
            parsed["init_params"] = {
                "mean": config_value.get("mean", 0.0),
                "std": config_value.get("std", 0.01),
            }
        elif {"min", "max"}.issubset(config_value.keys()):
            parsed["init_method"] = "uniform"
            parsed["init_params"] = {
                "min": config_value.get("min"),
                "max": config_value.get("max"),
            }
        elif "value" in config_value:
            parsed["init_method"] = "constant"
            parsed["init_params"] = {"value": config_value.get("value")}

        # Preserve block_size if specified
        if "block_size" in config_value:
            parsed["init_params"]["block_size"] = int(config_value["block_size"])

        constraints = config_value.get("constraints")
        if isinstance(constraints, dict):
            parsed["min_value"] = constraints.get("min")
            parsed["max_value"] = constraints.get("max")
        else:
            if "min" in config_value:
                parsed["min_value"] = config_value["min"]
            if "max" in config_value:
                parsed["max_value"] = config_value["max"]

        if "learnable" in config_value:
            parsed["learnable"] = bool(config_value["learnable"])

    elif isinstance(config_value, (int, float)):
        parsed["init_method"] = "constant"
        parsed["init_params"] = {"value": float(config_value)}

    return parsed


def create_layer_parameters(
    layer: nn.Module,
    layer_type: str,
    dim: int,
    params: dict[str, Any] | None = None,
    node_fan_outs: list[int] | None = None,
) -> None:
    params = params or {}
    device = next(layer.parameters(), torch.empty(0)).device
    configs = LAYER_PARAM_CONFIGS.get(layer_type, [])

    for config in configs:
        param_name = config.name
        custom = params.get(param_name)
        parsed = parse_param_config(param_name, custom, node_fan_outs)

        if parsed["min_value"] is None:
            parsed["min_value"] = config.min_value
        if parsed["max_value"] is None:
            parsed["max_value"] = config.max_value

        shape = (dim,)
        method = parsed["init_method"]
        init_params = parsed["init_params"]
        if method in {"constant", "linear", "fan_out", "loglinear"}:
            values = initialize_values_deterministic(shape, method, init_params, device)
        else:
            values = initialize_values_stochastic(shape, method, init_params, device)

        if parsed["min_value"] is not None:
            values = torch.clamp(values, min=parsed["min_value"])
        if parsed["max_value"] is not None:
            values = torch.clamp(values, max=parsed["max_value"])

        if config.is_log_param:
            values = torch.clamp(values, min=1e-10)
            param_attr = f"log_{param_name}"
            values = torch.log(values)
        else:
            param_attr = param_name

        param = nn.Parameter(values, requires_grad=parsed["learnable"])
        setattr(layer, param_attr, param)

        if config.is_log_param and not hasattr(layer.__class__, param_name):

            def _getter(self, key=param_name):
                return torch.exp(getattr(self, f"log_{key}"))

            setattr(layer.__class__, param_name, property(_getter))


def apply_param_constraints(layer: nn.Module, layer_type: str) -> None:
    configs = LAYER_PARAM_CONFIGS.get(layer_type, [])

    def _user_bounds(name: str) -> tuple[float | None, float | None]:
        lp = getattr(layer, "params", None)
        if isinstance(lp, dict):
            entry = lp.get(name)
            if isinstance(entry, dict):
                constraints = entry.get("constraints")
                if isinstance(constraints, dict):
                    return constraints.get("min"), constraints.get("max")
        return None, None

    with torch.no_grad():
        for config in configs:
            base = config.name
            user_min, user_max = _user_bounds(base)
            eff_min = user_min if user_min is not None else config.min_value
            eff_max = user_max if user_max is not None else config.max_value

            if config.is_log_param:
                attr = f"log_{base}"
                if not hasattr(layer, attr):
                    continue
                param = getattr(layer, attr)
                if eff_min is not None and eff_min > 0:
                    param.data.clamp_(min=math.log(eff_min))
                if eff_max is not None and eff_max > 0 and eff_max < float("inf"):
                    param.data.clamp_(max=math.log(eff_max))
            else:
                attr = base
                if not hasattr(layer, attr):
                    continue
                param = getattr(layer, attr)
                if eff_min is not None:
                    param.data.clamp_(min=eff_min)
                if eff_max is not None:
                    param.data.clamp_(max=eff_max)


def noise_keys_for_layer(layer_type: str) -> list[str]:
    base_keys = ["s", "phi", "g"]
    param_keys = [p.name for p in LAYER_PARAM_CONFIGS.get(layer_type, [])]
    combined = list(dict.fromkeys(base_keys + param_keys))
    if layer_type == "Input":
        return ["s"]
    return combined


@dataclass
class LayerInfo:
    """High-level metadata for layer catalogues."""

    key: str
    title: str
    description: str
    category: str = "General"
    tags: list[str] | None = None


LAYER_CATALOG: dict[str, LayerInfo] = {
    "Linear": LayerInfo(
        key="Linear",
        title="Linear",
        description="Pass-through layer that output's it's own input.",
        category="Virtual",
    ),
    "Input": LayerInfo(
        key="Input",
        title="Input (legacy)",
        description="Legacy alias for the Linear layer maintained for backward compatibility.",
        category="Other",
    ),
    "ScalingLayer": LayerInfo(
        key="ScalingLayer",
        title="Scaling",
        description="Scales the input by a learnable factor; can be useful for normalisation.",
        category="Virtual",
    ),
    "RNN": LayerInfo(
        key="RNN",
        title="RNN",
        description="Single-layer RNN wrapper using PyTorch's implementation.",
        category="Virtual",
    ),
    "GRU": LayerInfo(
        key="GRU",
        title="GRU",
        description="Single-layer GRU wrapper.",
        category="Virtual",
    ),
    "LSTM": LayerInfo(
        key="LSTM",
        title="LSTM",
        description="Single-layer LSTM wrapper",
        category="Virtual",
    ),
    "MinGRU": LayerInfo(
        key="MinGRU",
        title="Min-GRU",
        description="Minimal GRU-style recurrent layer.",
        category="Virtual",
    ),
    "LeakyGRU": LayerInfo(
        key="LeakyGRU",
        title="Leaky-GRU",
        description="Leaky-integrator GRU variant with per-unit trainable time constants (fast fused kernel).",
        category="Virtual",
    ),
    "SingleDendrite": LayerInfo(
        key="SingleDendrite",
        title="Dendrite",
        description="Physical SOEN dendrite solving ODE dynamics.",
        category="Physical",
    ),
    "Multiplier": LayerInfo(
        key="Multiplier",
        title="Multiplier",
        description="Circuit that aims to multiply two signals physically.",
        category="Physical",
    ),
    "MultiplierNOCC": LayerInfo(
        key="MultiplierNOCC",
        title="Multiplier V2",
        description="New multiplier circuit with dual SQUID states for smaller circuit scale. Uses flux collection without collection coils.",
        category="Physical",
    ),
    "DendriteReadout": LayerInfo(
        key="DendriteReadout",
        title="Dendrite Readout",
        description="Used for measuring the supercurrent in the integration loops of dendrites.",
        category="Physical",
    ),
    # Legacy alias, hidden in picker but kept for metadata lookups
    "Readout": LayerInfo(
        key="Readout",
        title="Readout (legacy)",
        description="Alias of DendriteReadout. Used for measuring the supercurrent in the integration loops of dendrites.",
        category="Physical",
    ),
    "NonLinear": LayerInfo(
        key="NonLinear",
        title="Non-Linear",
        description="Applies a configurable source-function non-linearity to its inputs.",
        category="Virtual",
    ),
    "Softmax": LayerInfo(
        key="Softmax",
        title="Softmax",
        description="Applies temperature-scaled softmax normalization across features. Higher beta values produce sharper distributions. "
        "Tip: connect non-learnable one-to-one with the previous layer.",
        category="Normalisation",
    ),
}


LAYER_GROUP_ORDER: list[str] = ["Virtual", "Normalisation", "Physical", "Other"]


def layer_category_lookup() -> dict[str, str]:
    return {name: info.category for name, info in LAYER_CATALOG.items()}


def group_layers_by_category(layer_names: list[str]) -> dict[str, list[LayerInfo]]:
    catalog = LAYER_CATALOG
    grouped: dict[str, list[LayerInfo]] = {key: [] for key in LAYER_GROUP_ORDER}
    grouped.setdefault("Other", [])
    for name in layer_names:
        info = catalog.get(
            name,
            LayerInfo(
                key=name,
                title=name,
                description="Custom layer",
                category="Other",
            ),
        )
        grouped.setdefault(info.category, []).append(info)
    for group in grouped.values():
        group.sort(key=lambda item: item.title)
    return grouped


def iter_layer_groups(layer_names: list[str]):
    grouped = group_layers_by_category(layer_names)
    for category in LAYER_GROUP_ORDER:
        layers = grouped.get(category, [])
        if not layers:
            continue
        yield category, layers


def get_layer_catalog() -> dict[str, LayerInfo]:
    return dict(LAYER_CATALOG)


__all__ = [
    "INIT_METHOD_PARAMS",
    "INIT_OPTIONAL_PARAMS",
    "LAYER_CATALOG",
    "LAYER_GROUP_ORDER",
    "LAYER_PARAM_CONFIGS",
    "LayerInfo",
    "ParamConfig",
    "apply_param_constraints",
    "create_layer_parameters",
    "group_layers_by_category",
    "initialize_values_deterministic",
    "initialize_values_stochastic",
    "iter_layer_groups",
    "layer_category_lookup",
    "noise_keys_for_layer",
    "parse_param_config",
]
