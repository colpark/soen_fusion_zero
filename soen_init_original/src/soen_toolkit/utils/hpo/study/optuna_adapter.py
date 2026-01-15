#!/usr/bin/env python3
"""Thin adapter over Optuna: sampler/pruner builders, study creation, and callbacks.

Keeps Optuna-specific wiring out of the runner to simplify testing and swapping.
"""

from __future__ import annotations

import contextlib
import os
import re as _re
import time
from typing import TYPE_CHECKING, Any

import optuna
from optuna.pruners import MedianPruner, NopPruner

from soen_toolkit.utils.hpo.io.exporters import append_trial_jsonl
from soen_toolkit.utils.hpo.io.paths import get_study_paths

if TYPE_CHECKING:
    from collections.abc import Callable


def _is_fail_fast() -> bool:
    try:
        return str(os.environ.get("SOEN_HPO_FAIL_FAST", "")).strip() in ("1", "true", "True")
    except Exception:
        return False


def build_optuna_sampler_from_config(name: str, *, seed: int, kwargs: dict[str, Any] | None = None):
    from optuna.samplers import CmaEsSampler, NSGAIISampler, RandomSampler, TPESampler

    try:
        from optuna.integration.botorch import BoTorchSampler
    except Exception:
        BoTorchSampler = None

    import inspect as _inspect

    def _resolve_nested_sampler(v: Any, *, seed: int):
        """Coerce nested sampler config to an Optuna sampler instance.

        Accepts an existing sampler, a string name, or a dict {name|sampler, kwargs}.
        """
        try:
            # Already a sampler (duck-typed)
            if hasattr(v, "reseed_rng"):
                return v
            if isinstance(v, str):
                return build_optuna_sampler_from_config(v, seed=seed, kwargs={})
            if isinstance(v, dict):
                nm = v.get("name") or v.get("sampler") or v.get("type") or "tpe"
                kw = v.get("kwargs") or {k: v[k] for k in v if k not in ("name", "sampler", "type")}
                return build_optuna_sampler_from_config(str(nm), seed=seed, kwargs=kw)
        except Exception:
            pass
        # Fallback
        if _is_fail_fast():
            msg = f"Invalid nested sampler config: {v}"
            raise ValueError(msg)
        return build_optuna_sampler_from_config("random", seed=seed, kwargs={})

    def _safe_make(cls, *, seed: int, kwargs: dict[str, Any] | None = None):
        kw = dict(kwargs or {})
        # Normalize nested sampler arguments by convention
        if cls.__name__ in ("CmaEsSampler", "NSGAIISampler") and ("independent_sampler" in kw):
            kw["independent_sampler"] = _resolve_nested_sampler(kw.get("independent_sampler"), seed=seed)
        # Strip deprecated/unsupported options proactively (avoid noisy warnings)
        if cls.__name__ == "CmaEsSampler" and ("restart_strategy" in kw):
            if _is_fail_fast():
                msg = "CmaEsSampler.restart_strategy is deprecated in Optuna>=4.4; remove it from config"
                raise ValueError(msg)
            kw.pop("restart_strategy", None)
        try:
            sig = _inspect.signature(cls.__init__)
            allowed = {k: v for k, v in kw.items() if k in sig.parameters}
            if "seed" in sig.parameters:
                allowed["seed"] = seed
            return cls(**allowed)
        except TypeError:
            if _is_fail_fast():
                raise
            with contextlib.suppress(Exception):
                pass
            # Fallback to seed-only or default
            try:
                sig = _inspect.signature(cls.__init__)
                if "seed" in sig.parameters:
                    return cls(seed=seed)
                return cls()
            except Exception:
                return cls()

    name = str(name or "tpe").lower()
    kwargs = dict(kwargs or {})
    if name == "tpe":
        return _safe_make(TPESampler, seed=seed, kwargs=kwargs)
    if name in ("random", "rand"):
        return _safe_make(RandomSampler, seed=seed, kwargs=kwargs)
    if name in ("cmaes", "cma"):
        return _safe_make(CmaEsSampler, seed=seed, kwargs=kwargs)
    if name in ("nsga2", "nsgaii"):
        return _safe_make(NSGAIISampler, seed=seed, kwargs=kwargs)
    if name in ("botorch", "bo") and BoTorchSampler is not None:
        return _safe_make(BoTorchSampler, seed=seed, kwargs=kwargs)
    return _safe_make(TPESampler, seed=seed, kwargs=kwargs)


def build_pruner(cfg: dict[str, Any]) -> Any:
    cfg = cfg or {}
    if cfg.get("use", False):
        return MedianPruner(
            n_startup_trials=int(cfg.get("n_startup_trials", 20)),
            n_warmup_steps=int(cfg.get("n_warmup_steps", 5)),
        )
    return NopPruner()


def create_study(study_name: str, *, out_dir: str, sampler: Any, pruner: Any, resume: bool) -> optuna.Study:
    sp = get_study_paths(study_name, out_dir=out_dir)
    try:
        return optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{sp.db_path}",
            load_if_exists=bool(resume),
        )
    except optuna.exceptions.DuplicatedStudyError:
        # If a study with this name already exists, append a timestamp and retry.
        base = _re.sub(r"(_\d{8}_\d{6})+$", "", str(study_name))
        ts = time.strftime("%Y%m%d_%H%M%S")
        new_name = f"{base}_{ts}"
        sp2 = get_study_paths(new_name, out_dir=out_dir)
        return optuna.create_study(
            study_name=new_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{sp2.db_path}",
            load_if_exists=False,
        )


def make_progress_callbacks(study_name: str, *, out_dir: str) -> tuple[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]:
    """Return a tuple of Optuna callbacks for progress and JSONL logging."""

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        try:
            pass
        except Exception:
            with contextlib.suppress(Exception):
                pass
        # Append compact JSONL record for GUI/live tailing
        try:
            rec = {
                "ts": time.time(),
                "trial": int(trial.number),
                "state": str(trial.state.name if hasattr(trial, "state") else ""),
                "value": float(trial.value) if trial.value is not None else None,
                "best_value": float(study.best_value) if study.best_value is not None else None,
                "params": dict(trial.params or {}),
                "user_attrs": dict(trial.user_attrs or {}),
            }
            append_trial_jsonl(study_name, out_dir, rec)
        except Exception:
            pass

    return (_cb,)

    # Note: could add on_start/on_error hooks later if needed
