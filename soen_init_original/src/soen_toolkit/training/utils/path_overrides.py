"""Resolve and patch training config paths for local runs.

Behavior:
- Locate project root (via pyproject.toml or .git)
- Read YAML and resolve any relative paths against the project root
- Optionally accept overrides via function args
- Write a sibling YAML with suffix .local.yaml (or a custom output name)

Usage (Python):
    from soen_toolkit.training.utils.path_overrides import patch_config_paths_for_local_run
    patched_cfg, resolved_project_dir, cfg_dict = patch_config_paths_for_local_run(".../config.yaml")

CLI:
    python -m soen_toolkit.training.utils.path_overrides --config path/to.yaml [--project-dir DIR]


To Do:
1. Either improve this tool, find a use for it, or remove it fully.

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _find_project_root(start: Path | None = None) -> Path:
    cur = Path(start) if start is not None else Path(__file__).resolve().parent
    for _ in range(len(cur.parts)):
        if (cur / "pyproject.toml").is_file() or (cur / ".git").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path.cwd().resolve()


def _ensure_absolute_path(value: str | Path | None, project_root: Path) -> Path | None:
    """Return absolute path if value is provided; resolve relatives against project_root.

    Expands ~ and environment variables. Returns None if value is falsy/None.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    expanded = os.path.expandvars(text)
    p = Path(expanded).expanduser()
    if not p.is_absolute():
        p = project_root / p
    try:
        # .resolve(strict=False) preserves non-existing paths but canonicalizes .. and symlinks
        return p.resolve(strict=False)
    except Exception:
        # Fallback without resolve
        return p


def patch_config_paths_for_local_run(
    base_config: str | Path,
    project_dir: str | Path | None = None,
    data_path: str | Path | None = None,
    base_model_path: str | Path | None = None,
    output_name: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Patch key paths in a training YAML so it runs on the local machine without edits.

    - Ensures absolute paths for data, base model, and logging.project_dir
    - Writes a sibling YAML with suffix .local.yaml (or output_name if given)

    Returns: (patched_config_path, resolved_project_dir, cfg_dict)
    """
    base_config = Path(base_config)
    if not base_config.is_file():
        msg = f"Config not found: {base_config}"
        raise FileNotFoundError(msg)

    project_root = _find_project_root(base_config.parent)

    # Load YAML first
    with open(base_config) as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("logging", {})

    # Resolve paths from arguments or YAML values (no tutorial-specific defaults)
    resolved_data = _ensure_absolute_path(data_path, project_root) if data_path is not None else _ensure_absolute_path(cfg.get("data", {}).get("data_path"), project_root)
    resolved_model = _ensure_absolute_path(base_model_path, project_root) if base_model_path is not None else _ensure_absolute_path(cfg.get("model", {}).get("base_model_path"), project_root)

    pd_value = project_dir if (project_dir is not None and str(project_dir).strip() != "") else cfg.get("logging", {}).get("project_dir")
    resolved_project_dir = _ensure_absolute_path(pd_value, project_root) or project_root

    if resolved_data is not None:
        cfg["data"]["data_path"] = str(resolved_data)
    if resolved_model is not None:
        cfg["model"]["base_model_path"] = str(resolved_model)
    # Always ensure project_dir is a usable absolute path
    cfg["logging"]["project_dir"] = str(resolved_project_dir)

    # Determine output path
    if output_name:
        patched_path = (base_config.parent / output_name).with_suffix(".yaml")
    else:
        patched_path = base_config.with_name(base_config.stem + ".local.yaml")

    with open(patched_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return patched_path, resolved_project_dir, cfg


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Patch training config paths for local run")
    parser.add_argument("--config", required=True, help="Path to base YAML config")
    parser.add_argument("--project-dir", default="", help="Optional output project_dir (logs/checkpoints)")
    args = parser.parse_args(argv)

    try:
        _patched, _proj_dir, _ = patch_config_paths_for_local_run(
            args.config,
            project_dir=args.project_dir or None,
        )
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
