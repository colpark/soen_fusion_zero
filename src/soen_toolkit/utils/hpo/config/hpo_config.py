"""Utilities for robust HPO config creation, resolution, and validation.

Goals:
- Decouple HPO YAML handling from GUI/CLI business logic.
- Robustly resolve base model spec path (supports trained model files → spec extraction).
- Auto-populate/repair optimization_config targets/spaces from a model spec when missing.
- Provide simple load/save helpers used by both GUI and CLI callers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = "YAML root must be a mapping"
        raise ValueError(msg)
    return data


def write_yaml(obj: dict[str, Any], path: str) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _is_trained_model_file(p: str) -> bool:
    s = str(p).lower()
    return s.endswith((".soen", ".pth", ".pt", ".json")) and not s.endswith(("_spec.yaml", "_spec_for_hpo.yaml"))


def extract_spec_from_model(model_path: str, output_dir: str) -> str:
    """Load a trained model and write a YAML spec next to/outside it for HPO.

    Returns the path to the written spec YAML.
    """
    from dataclasses import asdict as _asdict

    from soen_toolkit.core import SOENModelCore as _Core

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    m = _Core.load(model_path, device=None, strict=False, verbose=False, show_logs=False)
    spec_dict = {
        "simulation": _asdict(m.sim_config),
        "layers": [_asdict(cfg) for cfg in m.layers_config],
        "connections": [_asdict(cfg) for cfg in m.connections_config],
    }
    stem = Path(model_path).stem
    out_path = str(Path(output_dir) / f"{stem}_spec_for_hpo.yaml")
    write_yaml(spec_dict, out_path)
    return out_path


def resolve_base_model_spec(paths_block: dict[str, Any], *, allow_extract: bool = True) -> tuple[str | None, dict[str, Any]]:
    """Ensure paths_block['base_model_spec'] points to a real YAML file.

    - If it's already an existing file, return it.
    - If it points to a trained model and allow_extract is True, extract a spec into output_dir and return that path.
    - Otherwise, search output_dir and the YAML dir for a '*_spec_for_hpo.yaml'.

    Returns (resolved_path_or_None, possibly_updated_paths_block).
    """
    base = (paths_block or {}).get("base_model_spec")
    out_dir = (paths_block or {}).get("output_dir")
    updated = dict(paths_block or {})

    if isinstance(out_dir, str) and out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(base, str) and base:
        if os.path.isabs(base):
            cand = base
        else:
            # Resolve relative to current working directory
            cand = os.path.abspath(base)
        if os.path.exists(cand):
            # If this is a trained model, optionally extract a spec
            if _is_trained_model_file(cand) and allow_extract:
                target_dir = out_dir or str(Path(cand).parent)
                spec_path = extract_spec_from_model(cand, target_dir)
                updated["base_model_spec"] = spec_path
                return spec_path, updated
            updated["base_model_spec"] = cand
            return cand, updated

    # Try to find a spec in output_dir
    search_dirs = []
    if isinstance(out_dir, str) and out_dir:
        search_dirs.append(out_dir)
    # Also search current dir
    search_dirs.append(os.getcwd())
    for d in search_dirs:
        try:
            for p in Path(d).glob("*_spec_for_hpo.yaml"):
                updated["base_model_spec"] = str(p)
                return str(p), updated
        except Exception:
            continue

    return None, updated


def populate_optimization_config(cfg_all: dict[str, Any], base_model_spec_path: str | None) -> dict[str, Any]:
    """Ensure cfg_all contains a robust 'optimization_config' block.

    - If missing or empty, build a skeleton from the model spec.
    - If present but targets are empty, fill from skeleton while preserving user-defined parameter blocks.
    """
    oc = (cfg_all.get("optimization_config") or {}) if isinstance(cfg_all.get("optimization_config"), dict) else {}
    if not base_model_spec_path or not os.path.exists(base_model_spec_path):
        # Nothing we can do if there's no model spec available.
        cfg_all["optimization_config"] = oc
        return cfg_all

    # Build a skeleton from the model
    try:
        from soen_toolkit.utils.hpo.tools.enumerate_model_options import (
            build_hpo_skeleton,
        )

        skeleton = build_hpo_skeleton(base_model_spec_path)
    except Exception:
        cfg_all["optimization_config"] = oc
        return cfg_all

    if not oc:
        cfg_all["optimization_config"] = skeleton
        return cfg_all

    # Merge: keep existing parameter blocks; backfill targets/enabled if missing
    oc.setdefault("enabled_components", skeleton.get("enabled_components", {}))
    for key in ("target_layers", "target_connections"):
        if not oc.get(key):
            oc[key] = skeleton.get(key)
    oc.setdefault("layer_parameters", oc.get("layer_parameters") or skeleton.get("layer_parameters", {}))
    oc.setdefault("connection_parameters", oc.get("connection_parameters") or skeleton.get("connection_parameters", {}))
    oc.setdefault("weight_parameters", oc.get("weight_parameters") or skeleton.get("weight_parameters", {}))
    cfg_all["optimization_config"] = oc
    return cfg_all


def normalize_paths(cfg_all: dict[str, Any], *, base_yaml_path: str | None = None) -> dict[str, Any]:
    """Normalize/absolutize important paths; ensure output_dir exists."""
    out = dict(cfg_all or {})
    paths = dict(out.get("paths") or {})
    if base_yaml_path:
        base_dir = os.path.dirname(os.path.abspath(base_yaml_path))
    else:
        base_dir = os.getcwd()

    bms = paths.get("base_model_spec")
    if isinstance(bms, str) and bms and not os.path.isabs(bms):
        paths["base_model_spec"] = os.path.abspath(os.path.join(base_dir, bms))
    od = paths.get("output_dir")
    if isinstance(od, str) and od and not os.path.isabs(od):
        paths["output_dir"] = os.path.abspath(os.path.join(base_dir, od))
    tc = paths.get("train_config")
    if isinstance(tc, str) and tc and not os.path.isabs(tc):
        paths["train_config"] = os.path.abspath(os.path.join(base_dir, tc))
    out["paths"] = paths
    if paths.get("output_dir"):
        Path(paths["output_dir"]).mkdir(parents=True, exist_ok=True)
    return out


def validate_and_normalize_run_config(cfg_all: dict[str, Any]) -> dict[str, Any]:
    """Validates HPO mode and associated training config path."""
    run_cfg = cfg_all.get("run", {})
    if not isinstance(run_cfg, dict):
        run_cfg = {}

    # Validate hpo_mode
    hpo_mode = run_cfg.get("hpo_mode", "forward")
    if hpo_mode not in ["forward", "epoch"]:
        msg = f"Invalid hpo_mode: '{hpo_mode}'. Must be 'forward' or 'epoch'."
        raise ValueError(msg)
    run_cfg["hpo_mode"] = hpo_mode

    # Validate train_config path if in epoch mode
    if hpo_mode == "epoch":
        paths_cfg = cfg_all.get("paths", {})
        train_config_path = paths_cfg.get("train_config")
        if not train_config_path or not isinstance(train_config_path, str):
            msg = "hpo_mode 'epoch' requires a valid 'train_config' path in the 'paths' section."
            raise ValueError(msg)

        # The path is already made absolute by normalize_paths
        if not os.path.exists(train_config_path):
            msg = f"Training config file not found for 'epoch' mode: {train_config_path}"
            raise FileNotFoundError(msg)

    cfg_all["run"] = run_cfg
    return cfg_all


def load_hpo_config(path: str, *, allow_extract: bool = True) -> dict[str, Any]:
    """Load an HPO YAML, resolve paths, extract spec if needed, and populate optimization_config.

    Returns the updated config dict.
    """
    cfg_all = read_yaml(path)
    cfg_all = normalize_paths(cfg_all, base_yaml_path=path)
    cfg_all = validate_and_normalize_run_config(cfg_all)
    paths = cfg_all.get("paths") or {}
    resolved, updated_paths = resolve_base_model_spec(paths, allow_extract=allow_extract)
    cfg_all["paths"] = updated_paths
    return populate_optimization_config(cfg_all, resolved)


def save_hpo_config(cfg_all: dict[str, Any], path: str) -> None:
    cfg_all = normalize_paths(cfg_all, base_yaml_path=path)
    write_yaml(cfg_all, path)
