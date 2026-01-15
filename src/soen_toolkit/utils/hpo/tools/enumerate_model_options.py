#!/usr/bin/env python3
"""Enumerate all tunable options for a given SOEN model spec by introspecting
soen_toolkit registries (layer params, connectivity types, weight initializers).

Usage:
  python Studies/Criticality/tools/enumerate_model_options.py --model path/to/model.yaml
  python Studies/Criticality/tools/enumerate_model_options.py --model path/to/model.yaml --as hpo-skeleton

Outputs YAML to stdout with either an 'option_schema' or an HPO-config-like
structure ('optimization_config') that can be used as a starting point.

Notes:
  - No GUI
  this is a CLI enumerator to help scaffold configs.

"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import TYPE_CHECKING, Any

import yaml
import yaml as _yaml

if TYPE_CHECKING:
    from types import ModuleType


def _load_module_from_toolkit(rel_path: str) -> ModuleType:
    """Load a single module file from the soen_toolkit source tree without importing the package.

    rel_path should be like 'layers/common/metadata.py'.
    Resolution order:
      1) SOEN_TOOLKIT_PATH env var (expected to point to .../soen-toolkit/src)
      2) Auto-detect from current file location
    """
    base = os.environ.get("SOEN_TOOLKIT_PATH")
    if not base or not os.path.isdir(base):
        # Find the src directory by walking up from this file
        here = os.path.dirname(os.path.abspath(__file__))
        current = here
        # Walk up until we find a directory that contains soen_toolkit
        for _ in range(10):  # Prevent infinite loops
            parent = os.path.dirname(current)
            if parent == current:  # Reached root
                break
            # Check if this is the src directory (contains soen_toolkit)
            if os.path.isdir(os.path.join(current, "soen_toolkit")):
                base = current
                break
            current = parent
    if not base or not os.path.isdir(base):
        msg = "Could not locate soen_toolkit source. Set SOEN_TOOLKIT_PATH to .../soen-toolkit/src"
        raise ImportError(msg)
    path = os.path.join(base, "soen_toolkit", *rel_path.split("/"))
    if not os.path.isfile(path):
        msg = f"soen_toolkit file not found: {path}"
        raise ImportError(msg)
    name = f"_soen_local_{rel_path.replace('/', '_').replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"Failed to create spec for {path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules temporarily to fix dataclass issues
    import sys

    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Clean up on failure
        sys.modules.pop(name, None)
        raise
    return mod


def _load_module_from_installed(base_dir: str, rel_path: str) -> ModuleType:
    """Load a module by file path under an installed soen_toolkit package directory."""
    path = os.path.join(base_dir, *rel_path.split("/"))
    if not os.path.isfile(path):
        msg = f"Installed soen_toolkit missing: {path}"
        raise ImportError(msg)
    name = f"_soen_installed_{rel_path.replace('/', '_').replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"Failed to load spec for {path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules temporarily to fix dataclass issues
    import sys

    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Clean up on failure
        sys.modules.pop(name, None)
        raise
    return mod


def _get_registries():
    """Get registries using standard imports.

    Returns tuple: (LAYER_PARAM_CONFIGS, INIT_METHOD_PARAMS, ParamConfig,
                    connectivity_builders, CONNECTIVITY_PARAM_TYPES,
                    weight_initializers, SOURCE_FUNCTIONS)
    """
    from soen_toolkit.core.layers.common.connectivity_metadata import (
        CONNECTIVITY_BUILDERS,
        CONNECTIVITY_PARAM_TYPES,
        WEIGHT_INITIALIZERS,
    )
    from soen_toolkit.core.layers.common.metadata import (
        INIT_METHOD_PARAMS,
        LAYER_PARAM_CONFIGS,
        ParamConfig,
    )
    from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS

    return LAYER_PARAM_CONFIGS, INIT_METHOD_PARAMS, ParamConfig, CONNECTIVITY_BUILDERS, CONNECTIVITY_PARAM_TYPES, WEIGHT_INITIALIZERS, SOURCE_FUNCTIONS


def layer_param_schema(layer_type: str, *, LAYER_PARAM_CONFIGS, INIT_METHOD_PARAMS, ParamConfig) -> dict[str, Any]:
    """Return schema for a layer type's parameters using LAYER_PARAM_CONFIGS.

    Includes per-parameter: default, bounds, learnable default, log flag,
    and allowed distributions + their required param keys.
    """
    out: dict[str, Any] = {}
    configs: list[ParamConfig] = LAYER_PARAM_CONFIGS.get(layer_type, [])
    for pc in configs:
        dists = list(INIT_METHOD_PARAMS.keys())
        # Only expose fan_out for gamma_plus* parameters
        if not pc.name.startswith("gamma_plus"):
            dists = [d for d in dists if d != "fan_out"]
        out[pc.name] = {
            "default_value": pc.default_value,
            "is_log_param": bool(pc.is_log_param),
            "learnable_default": bool(pc.learnable),
            "min_value_default": pc.min_value,
            "max_value_default": pc.max_value,
            "distributions": {name: {"required_keys": INIT_METHOD_PARAMS[name]} for name in dists},
        }
    return out


def connectivity_type_schema(*, connectivity_builders, CONNECTIVITY_PARAM_TYPES) -> dict[str, Any]:
    """Schema for all connectivity types and their parameter types.

    Uses CONNECTIVITY_PARAM_TYPES if available, plus common flags.
    """
    out: dict[str, Any] = {}
    for ctype in connectivity_builders:
        params_info = CONNECTIVITY_PARAM_TYPES.get(ctype, {})
        out[ctype] = {
            "params": params_info,
            "common": {
                "allow_self_connections": {"type": "bool", "default": True},
                "constraints": {
                    "min": {"type": "float", "optional": True},
                    "max": {"type": "float", "optional": True},
                },
                "learnable": {"type": "bool", "default": True},
            },
        }
    return out


def weight_init_schema(*, weight_initializers) -> dict[str, Any]:
    """List weight init methods and their accepted kwargs with defaults."""
    import inspect

    out: dict[str, Any] = {}
    for name, fn in weight_initializers.items():
        sig = inspect.signature(fn)
        # Skip the first three positional args (from_nodes, to_nodes, mask)
        params: dict[str, Any] = {}
        for p_name, p in sig.parameters.items():
            if p_name in ("from_nodes", "to_nodes", "mask"):
                continue
            entry = {"kind": str(p.kind)}
            if p.default is not inspect._empty:
                entry["default"] = p.default
            # quick type hint
            if isinstance(p.default, (int, float)):
                entry["type"] = "int" if isinstance(p.default, int) else "float"
            elif isinstance(p.default, str):
                entry["type"] = "str"
            params[p_name] = entry
        out[name] = {"params": params}
    return out


def _load_raw_model_yaml(model_path: str) -> dict[str, Any]:
    """Load model data, handling both YAML specs and binary .soen files."""
    path_lower = model_path.lower()

    # Handle binary model files (.soen, .pth, .pt, .json) by extracting spec
    if path_lower.endswith((".soen", ".pth", ".pt", ".json")):
        try:
            from dataclasses import asdict as _asdict

            from soen_toolkit.core import SOENModelCore as _Core

            _m = _Core.load(model_path, device=None, strict=False, verbose=False, show_logs=False)
            return {
                "simulation": _asdict(_m.sim_config),
                "layers": [_asdict(cfg) for cfg in _m.layers_config],
                "connections": [_asdict(cfg) for cfg in _m.connections_config],
            }
        except Exception as e:
            msg = f"Failed to extract spec from model file {model_path}: {e}"
            raise ValueError(msg)

    # Handle YAML/text files
    with open(model_path) as f:
        data = _yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = "Model spec must be a mapping (YAML/JSON)"
        raise ValueError(msg)
    return data


def build_option_schema(model_path: str, base_model_path: str | None = None) -> dict[str, Any]:
    """Construct a structured option schema by reading the raw YAML."""
    data = _load_raw_model_yaml(model_path)
    (
        LAYER_PARAM_CONFIGS,
        INIT_METHOD_PARAMS,
        ParamConfig,
        connectivity_builders,
        CONNECTIVITY_PARAM_TYPES,
        weight_initializers,
        SOURCE_FUNCTIONS,
    ) = _get_registries()

    sim_block = data.get("simulation", {}) or data.get("sim_config", {}) or {}
    sim_opts = {
        "dt": {"type": "float", "default": sim_block.get("dt", 37)},
        "dt_learnable": {"type": "bool", "default": bool(sim_block.get("dt_learnable", False))},
        "input_type": {"type": "enum", "options": ["state", "flux"], "default": sim_block.get("input_type", "state")},
        "track_power": {"type": "bool", "default": bool(sim_block.get("track_power", False))},
        "track_phi": {"type": "bool", "default": bool(sim_block.get("track_phi", False))},
        "track_g": {"type": "bool", "default": bool(sim_block.get("track_g", False))},
        "track_s": {"type": "bool", "default": bool(sim_block.get("track_s", False))},
    }

    layers_raw = data.get("layers", []) or []
    layers_opts: dict[int, Any] = {}
    # If this looks like a trial params dump (no layers/connections) and a base model is provided, use base model structure
    if (not layers_raw) and ("parameters" in data) and base_model_path:
        try:
            base_data = _load_raw_model_yaml(base_model_path)
            layers_raw = base_data.get("layers", []) or []
        except Exception:
            pass
    # Compute layer_types_used after any base-model fallback
    layer_types_used = sorted({(lr.get("layer_type") or "").strip() for lr in layers_raw if isinstance(lr, dict)})

    for lr in layers_raw:
        if not isinstance(lr, dict):
            continue
        try:
            lid = int(lr.get("layer_id"))
        except Exception:
            continue
        ltype = str(lr.get("layer_type", "")).strip()
        if not ltype:
            continue
        layers_opts[lid] = {
            "layer_type": ltype,
            "params_schema": layer_param_schema(ltype, LAYER_PARAM_CONFIGS=LAYER_PARAM_CONFIGS, INIT_METHOD_PARAMS=INIT_METHOD_PARAMS, ParamConfig=ParamConfig),
            "supports_solver": ltype in {"SingleDendrite", "MinGRU", "Multiplier", "DendriteReadout"},
            "solver_choices": ["FE", "PS"],
            "supports_source_func": ltype
            in {
                "SingleDendrite",
                "Multiplier",
                "DendriteReadout",
                "NonLinear",
            },
            "source_func_choices": list(SOURCE_FUNCTIONS.keys()),
        }

    conns_raw = data.get("connections", []) or []
    if (not conns_raw) and ("parameters" in data) and base_model_path:
        try:
            base_data = _load_raw_model_yaml(base_model_path)
            conns_raw = base_data.get("connections", []) or []
        except Exception:
            pass
    conn_types_all = list(connectivity_builders.keys())
    conns_opts: dict[str, Any] = {}
    for cr in conns_raw:
        if not isinstance(cr, dict):
            continue
        try:
            fl = int(cr.get("from_layer"))
            tl = int(cr.get("to_layer"))
        except Exception:
            continue
        name = f"J_{fl}_to_{tl}"
        conns_opts[name] = {
            "current_type": str(cr.get("connection_type", "")).strip(),
            "available_types": conn_types_all,
            "type_params": connectivity_type_schema(connectivity_builders=connectivity_builders, CONNECTIVITY_PARAM_TYPES=CONNECTIVITY_PARAM_TYPES),
            "weight_init": weight_init_schema(weight_initializers=weight_initializers),
        }

    return {
        "simulation": sim_opts,
        "layer_types_used": layer_types_used,
        "layers": layers_opts,
        "connections": conns_opts,
    }


def build_hpo_skeleton(model_path: str, base_model_path: str | None = None) -> dict[str, Any]:
    """Produce a minimal HPO optimization_config skeleton for a model."""
    data = _load_raw_model_yaml(model_path)
    layers_raw = data.get("layers", []) or []
    conns_raw = data.get("connections", []) or []
    # If called on a trial-params dump, use base model structure if provided
    if (not layers_raw or not conns_raw) and ("parameters" in data) and base_model_path:
        try:
            base_data = _load_raw_model_yaml(base_model_path)
            layers_raw = base_data.get("layers", layers_raw) or layers_raw
            conns_raw = base_data.get("connections", conns_raw) or conns_raw
        except Exception:
            pass
    (
        LAYER_PARAM_CONFIGS,
        INIT_METHOD_PARAMS,
        _ParamConfig,
        connectivity_builders,
        CONNECTIVITY_PARAM_TYPES,
        weight_initializers,
        _SOURCE_FUNCTIONS,
    ) = _get_registries()

    # Aggregate parameter names from the layer types used
    params_union: dict[str, None] = {}
    for lr in layers_raw:
        if not isinstance(lr, dict):
            continue
        ltype = str(lr.get("layer_type", "")).strip()
        for pc in LAYER_PARAM_CONFIGS.get(ltype, []):
            params_union[pc.name] = None

    layer_parameters: dict[str, Any] = {}
    for pname in sorted(params_union.keys()):
        # Suggest distributions based on INIT_METHOD_PARAMS; default to constant
        dists = list(INIT_METHOD_PARAMS.keys())
        if not pname.startswith("gamma_plus"):
            dists = [d for d in dists if d != "fan_out"]
        # Provide a very light default with constant enabled
        layer_parameters[pname] = {
            "enabled": True,
            "distributions": ["constant"],
            "per_distribution": {
                "constant": {"value_bounds": {"min": 0.0, "max": 1.0}},
            },
        }

    # Connection-level parameter scaffolding
    conn_param_block: dict[str, Any] = {
        "connection_type": {
            "enabled": True,
            "choices": list(connectivity_builders.keys()),
        },
    }

    # Merge all known type params (user can later set applies_to to narrow)
    seen_conn_params: dict[str, None] = {}
    for ctype, info in CONNECTIVITY_PARAM_TYPES.items():
        for p, meta in info.items():
            if p in seen_conn_params:
                continue
            seen_conn_params[p] = None
            if meta.get("type") == "int":
                conn_param_block[p] = {
                    "enabled": False,
                    "type": "int",
                    "bounds": {"min": int(meta.get("min", 0)), "max": int(meta.get("max", 100))},
                    "applies_to": [ctype],
                }
            elif meta.get("type") == "enum":
                conn_param_block[p] = {
                    "enabled": False,
                    "choices": list(meta.get("options", [])),
                    "applies_to": [ctype],
                }
            else:
                conn_param_block[p] = {
                    "enabled": False,
                    "type": "float",
                    "bounds": {"min": float(meta.get("min", 0.0)), "max": float(meta.get("max", 1.0))},
                    "applies_to": [ctype],
                }

    # Weight init scaffolding: default to normal with simple bounds
    weight_parameters: dict[str, Any] = {
        "init_method": {
            "enabled": True,
            "choices": list(weight_initializers.keys()),
        },
        "normal": {
            "mean": {"enabled": True, "bounds": {"min": -0.1, "max": 0.1}},
            "std": {"enabled": True, "bounds": {"min": 0.0, "max": 0.5}},
        },
        "uniform": {
            "min": {"enabled": True, "bounds": {"min": -1.0, "max": 0.0}},
            "max": {"enabled": True, "bounds": {"min": 0.0, "max": 1.0}},
        },
    }

    # Build a minimal optimization_config
    return {
        "enabled_components": {"layers": True, "connections": True, "weights": True},
        "target_layers": sorted(
            {int(lr.get("layer_id")) for lr in layers_raw if isinstance(lr, dict) and str(lr.get("layer_type", "")) not in {"Input", "Linear"}},
        )
        or [1],
        "target_connections": [f"J_{int(cr.get('from_layer'))}_to_{int(cr.get('to_layer'))}" for cr in conns_raw if isinstance(cr, dict) and bool(cr.get("learnable", True))],
        "layer_parameters": layer_parameters,
        "connection_parameters": conn_param_block,
        "weight_parameters": weight_parameters,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Enumerate tunable options from a SOEN model spec")
    ap.add_argument("--model", required=True, help="Path to model YAML/JSON")
    ap.add_argument("--as", dest="format_as", choices=["schema", "hpo-skeleton"], default="schema", help="Output as option schema (default) or HPO skeleton block")
    ap.add_argument("--base-model", dest="base_model", default=None, help="Optional base model spec to provide structure when --model is a trial params dump")
    args = ap.parse_args()

    try:
        if args.format_as == "schema":
            data = {"option_schema": build_option_schema(args.model, base_model_path=args.base_model)}
        else:
            data = {"optimization_config": build_hpo_skeleton(args.model, base_model_path=args.base_model)}
        yaml.safe_dump(data, sys.stdout, sort_keys=False)
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
