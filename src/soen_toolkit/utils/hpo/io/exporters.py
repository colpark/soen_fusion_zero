#!/usr/bin/env python3
"""Export helpers for HPO studies.

Provides simple, testable functions to persist summary artifacts and best specs.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import TYPE_CHECKING, Any

import yaml

from .paths import get_study_paths

if TYPE_CHECKING:
    import optuna


def write_summary(study: optuna.Study, elapsed_sec: float, config: dict[str, Any], out_dir: str) -> str:
    """Write a single JSON summary for the study and return its path."""
    sp = get_study_paths(study.study_name, out_dir=out_dir)
    best = study.best_trial
    summary = {
        "study_name": study.study_name,
        "best_score": study.best_value,
        "best_trial_number": best.number,
        "elapsed_sec": elapsed_sec,
        "n_trials": len(study.trials),
        "config": config,
        "best_params": best.params,
        "best_metrics": best.user_attrs,
        "objective_weights": {
            "w_branch": config.get("w_branch"),
            "w_psd_t": config.get("w_psd_t"),
            "w_psd_spatial": config.get("w_psd_spatial"),
            "w_chi_inv": config.get("w_chi_inv"),
            "w_avalanche": config.get("w_avalanche"),
            "w_autocorr": config.get("w_autocorr"),
            "w_jacobian": config.get("w_jacobian"),
            "w_lyapunov": config.get("w_lyapunov"),
        },
    }
    with open(sp.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    return sp.summary_json


def write_best_spec_yaml(best_spec: dict[str, Any], study_name: str, out_dir: str) -> str:
    """Persist the best model spec YAML and return its path."""
    sp = get_study_paths(study_name, out_dir=out_dir)
    with open(sp.best_spec_yaml, "w") as f:
        yaml.safe_dump(best_spec, f, sort_keys=False)
    return sp.best_spec_yaml


def coerce_best_spec_with_seed_and_dt(best_spec: dict[str, Any], *, dt: Any, trial_user_attrs: dict[str, Any]) -> dict[str, Any]:
    """Ensure dt/seed are present in the saved spec for reproducibility.

    Returns a deep-copied spec.
    """
    result = deepcopy(best_spec)
    try:
        if dt is not None:
            result.setdefault("simulation", {})["dt"] = float(dt)
    except Exception:
        pass
    try:
        seed_in_spec = result.get("seed") if isinstance(result, dict) else None
        if seed_in_spec is None and isinstance(result, dict):
            sim_block = result.get("simulation") or result.get("sim_config") or {}
            if isinstance(sim_block, dict):
                seed_in_spec = sim_block.get("seed")
        if seed_in_spec is None:
            ms = trial_user_attrs.get("model_seed")
            if ms is not None:
                result["seed"] = int(ms)
    except Exception:
        pass
    return result


def append_trial_jsonl(study_name: str, out_dir: str, record: dict[str, Any]) -> None:
    """Append a single trial record to trials.jsonl for live progress and debugging."""
    sp = get_study_paths(study_name, out_dir=out_dir)
    line = json.dumps(record, separators=(",", ":"))
    with open(sp.trials_jsonl, "a") as f:
        f.write(line + "\n")
