import contextlib
from pathlib import Path
from typing import Any

import yaml

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    NoiseConfig,
    PerturbationConfig,
    SimulationConfig,
    SOENModelCore,
)

# Prefer C-accelerated SafeLoader when available
try:  # PyYAML provides C loaders if compiled
    from yaml import CSafeLoader as _SafeLoader
except Exception:  # pragma: no cover - fallback when C ext not available
    from yaml import SafeLoader as _SafeLoader


def _filter_kwargs(d: dict[str, Any], cls) -> dict[str, Any]:
    """Return only keys accepted by the dataclass constructor.

    Uses a cached lookup of __init__ parameter names to avoid repeated
    calls to inspect.signature during bulk parsing.
    """
    from functools import lru_cache
    import inspect

    @lru_cache(maxsize=64)
    def _init_param_names(target_cls: type) -> frozenset[str]:
        sig = inspect.signature(target_cls.__init__)
        names = set(sig.parameters.keys())
        names.discard("self")
        return frozenset(names)

    params = _init_param_names(cls)
    return {k: v for k, v in (d or {}).items() if k in params}


def _parse_simulation(sim_dict: dict[str, Any]) -> SimulationConfig:
    return SimulationConfig(**_filter_kwargs(sim_dict or {}, SimulationConfig))


def _parse_noise(obj: dict[str, Any] | None) -> NoiseConfig:
    if obj is None:
        return NoiseConfig()
    if isinstance(obj, NoiseConfig):
        return obj
    return NoiseConfig(**_filter_kwargs(obj, NoiseConfig))


def _parse_perturb(obj: dict[str, Any] | None) -> PerturbationConfig:
    if obj is None:
        return PerturbationConfig()
    if isinstance(obj, PerturbationConfig):
        return obj
    return PerturbationConfig(**_filter_kwargs(obj, PerturbationConfig))


def _parse_layers(layers_list: list[dict[str, Any]]) -> list[LayerConfig]:
    result: list[LayerConfig] = []
    seen_ids: set[int] = set()
    for raw in layers_list or []:
        # Required keys: layer_id, layer_type, params
        if not {"layer_id", "layer_type", "params"}.issubset(set(raw.keys())):
            missing = {"layer_id", "layer_type", "params"} - set(raw.keys())
            msg = f"Missing required layer keys: {missing}"
            raise ValueError(msg)
        lid = int(raw["layer_id"])  # ensure int-compatible
        if lid in seen_ids:
            msg = f"Duplicate layer_id detected: {lid}"
            raise ValueError(msg)
        seen_ids.add(lid)

        noise = _parse_noise(raw.get("noise"))
        perturb = _parse_perturb(raw.get("perturb"))

        filtered = _filter_kwargs(raw, LayerConfig)
        # Ensure params exists
        filtered.setdefault("params", {})
        # Pass explicit noise/perturb objects (constructor supplies defaults if omitted)
        filtered["noise"] = noise
        filtered["perturb"] = perturb
        result.append(LayerConfig(**filtered))
    # Sort by layer_id for deterministic construction
    result.sort(key=lambda c: c.layer_id)
    return result


def _parse_connections(conns_list: list[dict[str, Any]]) -> list[ConnectionConfig]:
    result: list[ConnectionConfig] = []
    for raw in conns_list or []:
        if not {"from_layer", "to_layer", "connection_type"}.issubset(set(raw.keys())):
            missing = {"from_layer", "to_layer", "connection_type"} - set(raw.keys())
            msg = f"Missing required connection keys: {missing}"
            raise ValueError(msg)
        # Support optional noise/perturb under the connection as well
        noise = _parse_noise(raw.get("noise")) if raw.get("noise") is not None else None
        perturb = _parse_perturb(raw.get("perturb")) if raw.get("perturb") is not None else None

        filtered = _filter_kwargs(raw, ConnectionConfig)
        # Keep optional params dict as-is (may carry init, constraints, learnable, etc.)
        if "params" in raw and filtered.get("params") is None:
            filtered["params"] = raw["params"]
        if noise is not None:
            filtered["noise"] = noise
        if perturb is not None:
            filtered["perturb"] = perturb
        result.append(ConnectionConfig(**filtered))
    return result


def parse_model_yaml(config: str | Path | dict[str, Any]) -> tuple[SimulationConfig, list[LayerConfig], list[ConnectionConfig]]:
    """Parse a YAML path or dict and return configuration objects without instantiating a model."""
    if isinstance(config, (str, Path)):
        path = Path(config)
        with path.open("r") as f:
            # Use C-accelerated loader if available for faster parsing
            data = yaml.load(f, Loader=_SafeLoader) or {}
    else:
        data = dict(config or {})

    sim_dict = data.get("simulation", data.get("sim_config", {})) or {}
    layers_list = data.get("layers", []) or []
    conns_list = data.get("connections", []) or []

    sim_config = _parse_simulation(sim_dict)
    layers_config = _parse_layers(layers_list)
    connections_config = _parse_connections(conns_list)

    return sim_config, layers_config, connections_config


def build_model_from_yaml(
    config: str | Path | dict[str, Any],
    *,
    honor_yaml_seed: bool = True,
    override_seed: int | None = None,
) -> SOENModelCore:
    """Build a SOENModelCore from a YAML/JSON spec.

    Seeding behavior:
      - If override_seed is provided, it is used for seeding.
      - Else if honor_yaml_seed is True, a seed embedded in the spec is used when present.
      - Else no seeding is performed here (caller is assumed to have seeded already).
    """
    # Load the raw mapping to inspect optional seed without altering parse API
    if isinstance(config, (str, Path)):
        path = Path(config)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = dict(config or {})

    # Determine seed to use
    seed_value: int | None
    if override_seed is not None:
        seed_value = int(override_seed)
    elif honor_yaml_seed:
        tmp_seed = None
        try:
            if isinstance(data, dict):
                tmp_seed = data.get("seed")
                if tmp_seed is None:
                    sim_block = data.get("simulation", data.get("sim_config", {})) or {}
                    tmp_seed = sim_block.get("seed")
                if tmp_seed is not None:
                    tmp_seed = int(tmp_seed)
        except Exception:
            tmp_seed = None
        seed_value = tmp_seed
    else:
        seed_value = None

    # Apply seeding if determined
    if seed_value is not None:
        try:
            import random as _random

            import numpy as np
            import torch as _torch

            _random.seed(seed_value)
            np.random.seed(seed_value)
            _torch.manual_seed(seed_value)
        except Exception:
            pass

    # Parse configs and instantiate model
    sim_config, layers_config, connections_config = parse_model_yaml(data)
    model = SOENModelCore(
        sim_config=sim_config,
        layers_config=layers_config,
        connections_config=connections_config,
    )
    # Remember the creation seed on the model for round-tripping/export
    if seed_value is not None:
        with contextlib.suppress(Exception):
            model._creation_seed = int(seed_value)
    return model


def load_model_from_yaml(config: str | Path | dict[str, Any]) -> SOENModelCore:
    """Deprecated: use build_model_from_yaml. Retained for backward compatibility."""
    return build_model_from_yaml(config)


def dump_model_to_yaml(model: SOENModelCore, path: str | Path) -> None:
    """Write the model configuration (not weights) to a YAML file."""
    from dataclasses import asdict

    # Note on connections naming:
    # Internal connections now use the unified key "J_<i>_to_<i>". Legacy
    # aliases "internal_<i>" remain available at runtime for backward compatibility,
    # but serialized configs and new code should adopt the unified naming.
    out = {
        "simulation": asdict(model.sim_config),
        "layers": [asdict(cfg) for cfg in model.layers_config],
        "connections": [asdict(cfg) for cfg in model.connections_config],
    }
    # Include seed used to build the model when available (top-level key)
    try:
        seed_val = getattr(model, "_creation_seed", None)
        if seed_val is not None:
            out["seed"] = int(seed_val)
    except Exception:
        pass
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
