"""HPOProject: a single abstraction to manage creating, loading, and saving
HPO optimization projects, unifying YAML, model spec, and trained model files.

Usage:
  - HPOProject.from_model(model_path, output_dir) -> new project with a resolved spec and skeleton opt config
  - HPOProject.from_yaml(yaml_path)             -> project loaded, with resolved spec and backfilled opt config
  - project.ensure_optimization_config()        -> populate targets/spaces if missing
  - project.save(yaml_path)                     -> save normalized YAML
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from soen_toolkit.utils.hpo.tools.enumerate_model_options import build_hpo_skeleton

from .hpo_config import (
    extract_spec_from_model,
    load_hpo_config,
    populate_optimization_config,
    save_hpo_config,
)


@dataclass
class HPOProject:
    cfg: dict[str, Any]
    yaml_path: str | None = None

    @property
    def paths(self) -> dict[str, Any]:
        return self.cfg.setdefault("paths", {})

    @property
    def base_model_spec(self) -> str | None:
        p = self.paths.get("base_model_spec")
        return str(p) if isinstance(p, str) else None

    @property
    def output_dir(self) -> str | None:
        p = self.paths.get("output_dir")
        return str(p) if isinstance(p, str) else None

    # ——— Constructors ———
    @classmethod
    def from_model(cls, model_path: str, output_dir: str | None = None) -> HPOProject:
        model_path = str(model_path)
        out_dir = output_dir or str(Path(model_path).parent)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        base_model_spec = model_path
        if model_path.lower().endswith((".soen", ".pth", ".pt", ".json")):
            base_model_spec = extract_spec_from_model(model_path, out_dir)
        base_model_spec = os.path.abspath(base_model_spec)
        # Build skeleton opt config
        opt_cfg = build_hpo_skeleton(base_model_spec)
        cfg: dict[str, Any] = {
            "paths": {"base_model_spec": base_model_spec, "output_dir": os.path.abspath(out_dir)},
            "optimization_config": opt_cfg,
        }
        return cls(cfg=cfg)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> HPOProject:
        cfg = load_hpo_config(yaml_path, allow_extract=True)
        return cls(cfg=cfg, yaml_path=yaml_path)

    # ——— Operations ———
    def ensure_optimization_config(self, *, replace: bool = False) -> None:
        bms = self.base_model_spec
        if not bms:
            return
        if replace or not self.cfg.get("optimization_config"):
            self.cfg["optimization_config"] = build_hpo_skeleton(bms)
        else:
            self.cfg = populate_optimization_config(self.cfg, bms)

    def save(self, path: str | None = None) -> str:
        target = path or self.yaml_path or str(Path(self.output_dir or ".") / "HPO_config.yaml")
        save_hpo_config(self.cfg, target)
        self.yaml_path = target
        return target
