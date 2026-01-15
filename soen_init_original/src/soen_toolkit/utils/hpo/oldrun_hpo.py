#!/usr/bin/env python3
# Optuna-based criticality hyperparameter optimization with rich conditional parameter space

"""example model:
SOENModelCore(
  (layers): ModuleList(
    (0): LinearLayer()
    (1): SingleDendriteLayer()
    (2): LinearLayer()
  )
# and then:
  (connections): ParameterDict(
      (J_0_to_1): Parameter containing: [torch.FloatTensor of size 100x10]
      (J_1_to_1): Parameter containing: [torch.FloatTensor of size 100x100]
      (J_1_to_2): Parameter containing: [torch.FloatTensor of size 10x100]
  )
).
"""

import argparse
import contextlib
from copy import deepcopy
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import optuna
import torch
import yaml

from soen_toolkit.core import SOENModelCore

# BoTorchSampler is an optional dependency for Bayesian optimization
# It will be imported on-demand when the user selects botorch sampler
from soen_toolkit.utils.hpo.data.exporters import (
    coerce_best_spec_with_seed_and_dt as _coerce_best_spec,
    write_best_spec_yaml as _write_best_spec_yaml,
    write_summary as _write_summary,
)

# Import HPO configuration utilities
from soen_toolkit.utils.hpo.hpo_config import (
    load_hpo_config as _load_hpo_config,
    normalize_paths as _normalize_paths,
    populate_optimization_config as _populate_oc,
    resolve_base_model_spec as _resolve_bms,
)
from soen_toolkit.utils.hpo.inputs import create as create_input_provider
from soen_toolkit.utils.hpo.study import (
    Objective as _Objective,
    build_optuna_sampler_from_config as _build_sampler,
    build_pruner as _build_pruner,
    create_study as _create_study,
    make_progress_callbacks as _make_callbacks,
)

# ---------------------------------------------------------------------
# Config (paths are resolved from HPO YAML at runtime)
# ---------------------------------------------------------------------
BASE_MODEL_SPEC = os.environ.get("SOEN_BASE_SPEC", "")
# Default to a subdirectory in the current working directory
OUT_DIR = os.environ.get("SOEN_OUT_DIR", os.path.join(os.getcwd(), "criticality_runs"))

# Defer BASE_SPEC loading until after YAML is read so YAML can override paths
BASE_SPEC: dict[str, Any] = {}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT_PS = 1.28  # 1 dt unit = 1.28 ps

# Default settings are loaded from HPO YAML; CLI flags act as optional overrides.


def apply_input_block_to_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply config['input'] convenience block to canonical keys.

    Precedence: this block overwrites canonical input keys in 'config'.
    """
    inp = config.get("input")
    if not isinstance(inp, dict):
        return config
    kind_raw = str(inp.get("kind", "")).strip().lower()
    kind_map = {
        "white": "white_noise",
        "white_noise": "white_noise",
        "colored": "colored_noise",
        "colored_noise": "colored_noise",
        "log": "log_slope_noise",
        "log_slope": "log_slope_noise",
        "log_slope_noise": "log_slope_noise",
        "gp": "gp_rbf",
        "gp_rbf": "gp_rbf",
    }
    # Accept known names directly and legacy aliases via kind_map
    known = {"white_noise", "colored_noise", "log_slope_noise", "gp_rbf", "hdf5_dataset"}
    resolved_kind = kind_map.get(kind_raw, kind_raw if kind_raw in known else None)
    if resolved_kind:
        config["input_kind"] = resolved_kind
    params = inp.get("params") or {}
    if resolved_kind == "white_noise":
        dn = params.get("delta_n", params.get("variance"))
        if dn is not None:
            config["delta_n"] = float(dn)
    elif resolved_kind == "colored_noise":
        beta = params.get("beta")
        if beta is not None:
            config["colored_beta"] = float(beta)
    elif resolved_kind == "gp_rbf":
        sigma = params.get("sigma")
        if sigma is not None:
            config["input_sigma"] = float(sigma)
        ell_ns = params.get("ell_ns")
        if ell_ns is not None:
            config["input_ell_ns"] = float(ell_ns)
    elif resolved_kind == "log_slope_noise":
        s = params.get("slope_db_per_dec")
        if s is not None:
            config["log_slope_db_per_dec"] = float(s)
        for k_src, k_dst, _dft in (
            ("fmin_frac", "log_slope_fmin_frac", 0.01),
            ("fmax_frac", "log_slope_fmax_frac", 0.5),
        ):
            v = params.get(k_src)
            if v is not None:
                config[k_dst] = float(v)
    elif resolved_kind == "hdf5_dataset":
        p = params.get("path")
        if p:
            config["input_h5_path"] = str(p)
        s = params.get("split")
        if s:
            config["input_h5_split"] = str(s)
        dk = params.get("data_key")
        if dk:
            config["input_h5_data_key"] = str(dk)
    # Optional scaling (applies to any kind)
    if "scale_min" in params and "scale_max" in params:
        try:
            config["input_scale_min"] = float(params["scale_min"])
            config["input_scale_max"] = float(params["scale_max"])
        except Exception:
            pass
    return config


def dt_units_to_seconds(dt_units: float) -> float:
    return float(dt_units) * DT_PS * 1e-12


def get_study_dir(study_name: str) -> str:
    """Return per-study directory under OUT_DIR and ensure it exists."""
    study_dir = os.path.join(OUT_DIR, f"optuna_report_{study_name}")
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    return study_dir


def get_input_dim_from_spec(spec: dict[str, Any]) -> int:
    """Robustly extract input dimension from model spec.
    Prefers layer with layer_id == 0, falls back to first layer with dim.
    """
    layers = spec.get("layers", [])
    for layer in layers:
        try:
            if int(layer.get("layer_id", -1)) == 0:
                dim = layer.get("params", {}).get("dim")
                if isinstance(dim, int) and dim > 0:
                    return dim
        except Exception:
            continue
    for layer in layers:
        dim = layer.get("params", {}).get("dim")
        if isinstance(dim, int) and dim > 0:
            return dim
    msg = "Could not determine input dimension from BASE_SPEC"
    raise ValueError(msg)


def _conn_name_from_spec(conn: dict[str, Any]) -> str:
    """Return canonical connection name like 'J_1_to_2' from a spec entry."""
    return f"J_{int(conn.get('from_layer'))}_to_{int(conn.get('to_layer'))}"


# Simple adapter for the new modular system
class ConfigurableHyperparameterSampler:
    """Minimal adapter to use the new modular HPO system."""

    def __init__(self, config_path: str, preset: str | None = None) -> None:
        self.config_path = config_path
        self.config = _load_hpo_config(config_path, allow_extract=True).get("optimization_config", {})

    def build_model_config(self, trial: optuna.Trial, base_spec: dict[str, Any]) -> dict[str, Any]:
        """Build model config by sampling per optimization_config for the trial."""
        spec = deepcopy(base_spec)
        oc = dict(self.config or {})

        # Helpers -------------------------------------------------------------
        def _layers_by_id(spec_dict: dict[str, Any]) -> dict[int, dict[str, Any]]:
            out: dict[int, dict[str, Any]] = {}
            for lr in spec_dict.get("layers", []) or []:
                try:
                    out[int(lr.get("layer_id"))] = lr
                except Exception:
                    continue
            return out

        def _conns_by_name(spec_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for cr in spec_dict.get("connections", []) or []:
                try:
                    fl = int(cr.get("from_layer"))
                    tl = int(cr.get("to_layer"))
                except Exception:
                    continue
                out[f"J_{fl}_to_{tl}"] = cr
            return out

        def _suggest_float(name: str, lo: float, hi: float, *, log: bool = False) -> float:
            lo_f = float(lo)
            hi_f = float(hi)
            # guard invalid ranges
            if hi_f < lo_f:
                lo_f, hi_f = hi_f, lo_f
            if log and (lo_f <= 0.0):
                lo_f = max(1e-12, lo_f)
            return float(trial.suggest_float(name, lo_f, hi_f, log=bool(log)))

        def _suggest_int(name: str, lo: int, hi: int) -> int:
            a = int(min(lo, hi))
            b = int(max(lo, hi))
            return int(trial.suggest_int(name, a, b))

        def _suggest_choice(name: str, choices: list) -> Any:
            if not choices:
                return None
            return trial.suggest_categorical(name, list(choices))

        # Target scopes --------------------------------------------------------
        target_layers = list(oc.get("target_layers") or [])
        target_conns = list(oc.get("target_connections") or [])
        enabled = dict(oc.get("enabled_components") or {"layers": True, "connections": True, "weights": True})

        # Layers: sample per-parameter values and inject into spec
        if enabled.get("layers", True):
            layer_params_cfg = dict(oc.get("layer_parameters") or {})
            lids_to_edit = target_layers or [int(lr.get("layer_id")) for lr in (spec.get("layers", []) or []) if isinstance(lr, dict) and str(lr.get("layer_type", "")) not in {"Input", "Linear"}]
            by_id = _layers_by_id(spec)
            for lid in lids_to_edit:
                lr = by_id.get(int(lid))
                if not isinstance(lr, dict):
                    continue
                params = lr.setdefault("params", {})
                for pname, pcfg in layer_params_cfg.items():
                    if not isinstance(pcfg, dict) or not pcfg.get("enabled", False):
                        continue
                    # Decide sampling mode
                    log_scale = bool(pcfg.get("log_scale", False))
                    list(pcfg.get("distributions") or [])
                    pd = dict(pcfg.get("per_distribution") or {})
                    sampled: float | None = None
                    # Prefer explicit uniform-like bounds if given under any known key
                    bounds = None
                    # constant.value_bounds {min,max}
                    if "constant" in pd and isinstance(pd["constant"], dict):
                        vb = pd["constant"].get("value_bounds") or pd["constant"].get("bounds")
                        if isinstance(vb, dict) and ("min" in vb) and ("max" in vb):
                            bounds = (vb["min"], vb["max"])
                    # uniform-style
                    if (bounds is None) and ("uniform" in pd) and isinstance(pd["uniform"], dict):
                        u = pd["uniform"]
                        lo = u.get("min") or (u.get("min_bounds", {}) or {}).get("min")
                        hi = u.get("max") or (u.get("max_bounds", {}) or {}).get("max")
                        if (lo is not None) and (hi is not None):
                            bounds = (lo, hi)
                    if bounds is not None:
                        sampled = _suggest_float(f"lyr.{lid}.{pname}", float(bounds[0]), float(bounds[1]), log=log_scale)
                    else:
                        # Fallback: attempt simple value_bounds on top-level of pcfg
                        vb = pcfg.get("value_bounds")
                        if isinstance(vb, dict) and ("min" in vb) and ("max" in vb):
                            sampled = _suggest_float(f"lyr.{lid}.{pname}", float(vb["min"]), float(vb["max"]), log=log_scale)
                    if sampled is not None:
                        params[pname] = float(sampled)

        # Connections: connection type/params and weight init
        by_name = _conns_by_name(spec)
        if enabled.get("connections", True):
            conn_params_cfg = dict(oc.get("connection_parameters") or {})
            for cname in target_conns or list(by_name.keys()):
                cr = by_name.get(cname)
                if not isinstance(cr, dict):
                    continue
                # connection_type
                ctype_cfg = conn_params_cfg.get("connection_type") or {}
                if ctype_cfg.get("enabled", False):
                    choices = list(ctype_cfg.get("choices") or [])
                    if choices:
                        cr["connection_type"] = str(_suggest_choice(f"conn.{cname}.type", choices))
                # allow_self_connections
                allow_cfg = conn_params_cfg.get("allow_self_connections") or {}
                if allow_cfg.get("enabled", False):
                    choices = list(allow_cfg.get("choices") or [])
                    if choices:
                        pick = _suggest_choice(f"conn.{cname}.allow_self", choices)
                        try:
                            cr.setdefault("params", {})["allow_self_connections"] = str(pick).lower() == "true"
                        except Exception:
                            cr.setdefault("params", {})["allow_self_connections"] = bool(pick)
                # expected_fan_out (int)
                efo_cfg = conn_params_cfg.get("expected_fan_out") or {}
                if efo_cfg.get("enabled", False):
                    b = efo_cfg.get("bounds") or {}
                    if ("min" in b) and ("max" in b):
                        cr.setdefault("params", {})["expected_fan_out"] = _suggest_int(f"conn.{cname}.fan_out", int(b["min"]), int(b["max"]))
                # sparsity (float) for sparse
                sparsity_cfg = conn_params_cfg.get("sparsity") or {}
                if sparsity_cfg.get("enabled", False):
                    b = sparsity_cfg.get("bounds") or {}
                    applies_to = list(sparsity_cfg.get("applies_to") or [])
                    ctype = str(cr.get("connection_type", ""))
                    if (("min" in b) and ("max" in b)) and (not applies_to or (ctype in applies_to)):
                        val = _suggest_float(f"conn.{cname}.sparsity", float(b["min"]), float(b["max"]))
                        # Clamp to (0,1]
                        with contextlib.suppress(Exception):
                            val = max(1e-6, min(1.0, float(val)))
                        cr.setdefault("params", {})["sparsity"] = float(val)
                # Fallback: ensure sparse always has a sparsity
                try:
                    if str(cr.get("connection_type", "")) == "sparse":
                        cr.setdefault("params", {})
                        if "sparsity" not in cr["params"]:
                            cr["params"]["sparsity"] = 0.1
                except Exception:
                    pass

        if enabled.get("weights", True):
            wcfg = dict(oc.get("weight_parameters") or {})
            # init_method
            init_cfg = wcfg.get("init_method") or {}
            init_choices = list(init_cfg.get("choices") or [])
            for cname in target_conns or list(by_name.keys()):
                cr = by_name.get(cname)
                if not isinstance(cr, dict):
                    continue
                params = cr.setdefault("params", {})
                init_name = None
                if init_cfg.get("enabled", False) and init_choices:
                    init_name = str(_suggest_choice(f"conn.{cname}.init", init_choices))
                    params["init"] = init_name
                else:
                    init_name = params.get("init", "normal")
                # init-specific fields (support normal/uniform minimally)
                if init_name == "normal":
                    ncfg = wcfg.get("normal", {})
                    mean_cfg = ncfg.get("mean", {})
                    std_cfg = ncfg.get("std", {})
                    if mean_cfg.get("enabled", False):
                        b = mean_cfg.get("bounds") or {}
                        if ("min" in b) and ("max" in b):
                            params["mean"] = _suggest_float(f"conn.{cname}.normal.mean", float(b["min"]), float(b["max"]))
                    if std_cfg.get("enabled", False):
                        b = std_cfg.get("bounds") or {}
                        if ("min" in b) and ("max" in b):
                            params["std"] = _suggest_float(f"conn.{cname}.normal.std", float(b["min"]), float(b["max"]))
                elif init_name == "uniform":
                    ucfg = wcfg.get("uniform", {})
                    min_cfg = ucfg.get("min", {})
                    max_cfg = ucfg.get("max", {})
                    if min_cfg.get("enabled", False):
                        b = min_cfg.get("bounds") or {}
                        if ("min" in b) and ("max" in b):
                            vmin = _suggest_float(f"conn.{cname}.uniform.min", float(b["min"]), float(b["max"]))
                            params["min"] = float(vmin)
                    if max_cfg.get("enabled", False):
                        b = max_cfg.get("bounds") or {}
                        if ("min" in b) and ("max" in b):
                            vmax = _suggest_float(f"conn.{cname}.uniform.max", float(b["min"]), float(b["max"]))
                            params["max"] = float(vmax)
                    # Ensure valid ordering if both present
                    try:
                        if ("min" in params) and ("max" in params) and (float(params["min"]) > float(params["max"])):
                            params["min"], params["max"] = float(params["max"]), float(params["min"])  # swap
                    except Exception:
                        pass

        return spec


def create_sampler_from_config(config_path: str, preset: str | None = None) -> ConfigurableHyperparameterSampler:
    """Factory function to create configured sampler."""
    return ConfigurableHyperparameterSampler(config_path, preset)


def build_config_from_trial_v2(trial: optuna.Trial, base_spec: dict[str, Any], sampler: ConfigurableHyperparameterSampler) -> dict[str, Any]:
    """Build model config using configurable sampler and sanitize initializer ranges."""
    spec = sampler.build_model_config(trial, base_spec)
    # Sanitize connection params for valid initializer ranges and legacy keys
    try:
        for conn in spec.get("connections", []) or []:
            params = conn.get("params") or {}
            init_name = str(params.get("init", ""))
            if init_name == "uniform":
                # Map legacy a/b to min/max and ensure ordering
                if ("a" in params) and ("min" not in params):
                    params["min"] = params["a"]
                if ("b" in params) and ("max" not in params):
                    params["max"] = params["b"]
                if ("min" in params) and ("max" in params):
                    try:
                        vmin = float(params.get("min"))
                        vmax = float(params.get("max"))
                        if vmin > vmax:
                            params["min"], params["max"] = vmax, vmin
                    except Exception:
                        pass
            elif init_name == "normal":
                try:
                    if float(params.get("std", 0.0)) < 0:
                        params["std"] = abs(float(params["std"]))
                except Exception:
                    pass
            conn["params"] = params
    except Exception:
        pass
    return spec


# Optuna sampler building moved to study/optuna_adapter.py

# ---------------------------------------------------------------------
# Metrics moved to study/metrics/core.py
# ---------------------------------------------------------------------


def criticality_objective(
    model: SOENModelCore,
    s_histories: list[torch.Tensor],
    dt_s: float,
    *,
    target_layers: list[int] | None = None,
    w_branch: float = 1.0,
    w_psd_t: float = 0.5,
    w_psd_spatial: float = 0.0,
    w_chi_inv: float = 0.0,
    w_avalanche: float = 0.0,
    w_autocorr: float = 0.0,
    w_jacobian: float = 0.0,
    w_lyapunov: float = 0.0,
    target_beta_t: float = 2.0,
    target_beta_spatial: float = 2.0,
) -> dict[str, float]:
    """Legacy wrapper for the new modular objective system."""
    # Delegate to modular Objective while preserving signature and keys
    obj = _Objective(
        config={
            "w_branch": w_branch,
            "w_psd_t": w_psd_t,
            "w_psd_spatial": w_psd_spatial,
            "w_chi_inv": w_chi_inv,
            "w_avalanche": w_avalanche,
            "w_autocorr": w_autocorr,
            "w_jacobian": w_jacobian,
            "w_lyapunov": w_lyapunov,
            "target_layers": target_layers,
            "target_beta_t": target_beta_t,
            "target_beta_spatial": target_beta_spatial,
        }
    )
    context = {"dt_s": float(dt_s)}
    return obj.evaluate(context, model, s_histories)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_trial(trial, config: dict, x_in: torch.Tensor, hp_sampler: ConfigurableHyperparameterSampler) -> float:
    """Evaluate a single trial configuration."""
    try:
        # Determine model seed: prefer seed from spec (BASE_SPEC) if provided; otherwise per-trial seed
        try:
            base_seed = None
            if isinstance(BASE_SPEC, dict):
                base_seed = BASE_SPEC.get("seed")
                if base_seed is None:
                    sim_block = (BASE_SPEC.get("simulation") or BASE_SPEC.get("sim_config") or {}) or {}
                    if isinstance(sim_block, dict):
                        base_seed = sim_block.get("seed")
            model_seed = int(base_seed) if base_seed is not None else int(config.get("model_seed", trial.number + 42))
        except Exception:
            model_seed = int(config.get("model_seed", trial.number + 42))

        # Set deterministic seed for model construction and numpy/python
        import random as _random

        import numpy as _np

        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        _np.random.seed(model_seed)
        with contextlib.suppress(Exception):
            _random.seed(model_seed)

        # Build model from trial (configuration-driven)
        spec = build_config_from_trial_v2(trial, BASE_SPEC, hp_sampler)
        # Ensure the model seed is persisted into the spec if not already present
        try:
            if isinstance(spec, dict):
                seed_in_spec = spec.get("seed")
                if seed_in_spec is None:
                    sim_block = spec.get("simulation") or spec.get("sim_config") or {}
                    if isinstance(sim_block, dict):
                        seed_in_spec = sim_block.get("seed")
                if seed_in_spec is None:
                    spec["seed"] = int(model_seed)
        except Exception:
            pass
        model = SOENModelCore.build(spec).to(DEVICE)

        # Apply dt override if provided, otherwise use model's dt
        dt_override = config.get("dt")
        if dt_override is not None and hasattr(model, "set_dt"):
            with contextlib.suppress(Exception):
                model.set_dt(float(dt_override))
        dt_s = dt_units_to_seconds(float(model.dt))
        with torch.no_grad():
            _, s_histories = model(x_in)

        # Add diagnostic logging for branching ratio stability
        from soen_toolkit.utils.hpo.study.objective.core import (
            _layer_ids,
            compute_phi_in_all_layers_adjusted,
        )

        phi_in = compute_phi_in_all_layers_adjusted(model, s_histories, subtract_phi_offset=True)
        lids = _layer_ids(model)

        # Log phi_in statistics for debugging
        for lid in lids:
            if lid in phi_in:
                phi_vals = phi_in[lid].abs().reshape(-1)
                trial.set_user_attr(f"phi_in_abs_p01_layer_{lid}", float(torch.quantile(phi_vals, 0.01).item()))
                trial.set_user_attr(f"phi_in_abs_p10_layer_{lid}", float(torch.quantile(phi_vals, 0.10).item()))
                trial.set_user_attr(f"phi_in_abs_median_layer_{lid}", float(torch.median(phi_vals).item()))
                trial.set_user_attr(f"phi_in_abs_mean_layer_{lid}", float(torch.mean(phi_vals).item()))

        # Compute criticality metrics
        obj = criticality_objective(
            model,
            s_histories,
            dt_s,
            target_layers=config.get("target_layers"),
            w_branch=config.get("w_branch", 1.0),
            w_psd_t=config.get("w_psd_t", 0.0),
            w_psd_spatial=config.get("w_psd_spatial", 0.0),
            w_chi_inv=config.get("w_chi_inv", 0.0),
            w_avalanche=config.get("w_avalanche", 0.0),
            w_autocorr=config.get("w_autocorr", 0.0),
            w_jacobian=config.get("w_jacobian", 0.0),
            w_lyapunov=config.get("w_lyapunov", 0.0),
            target_beta_t=config.get("target_beta_t", 2.0),
            target_beta_spatial=config.get("target_beta_spatial", 2.0),
        )

        total_cost = obj["total_cost"]
        if math.isnan(total_cost):
            trial.set_user_attr("failure_reason", "nan_total_cost")
            return -1e2  # Bad score for invalid configurations

        score = -float(total_cost)  # Optuna maximizes, we want to minimize cost
        # Clamp to minimum -100 as requested
        if score < -100.0:
            score = -101.0
        # Classify very poor objective as failed-equivalent
        if score <= -100.0:
            trial.set_user_attr("failed_threshold", True)

        # Store metrics in trial user attributes
        for key, value in obj.items():
            trial.set_user_attr(key, value)

        # Log score for debugging
        trial.set_user_attr("raw_score", score)
        # Record model seed used for this trial for reproducibility/export
        try:
            trial.set_user_attr("model_seed", int(model_seed))
            trial.set_user_attr("py_seed", int(model_seed))
            trial.set_user_attr("np_seed", int(model_seed))
        except Exception:
            pass

        return float(score)

    except Exception as e:
        trial.set_user_attr("failure_reason", str(e))
        return -1e2  # Bad score for failed trials
    finally:
        with contextlib.suppress(Exception):
            del model
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------
def run_optimization(config: dict, hp_sampler: ConfigurableHyperparameterSampler) -> tuple[optuna.Study, float]:
    """Main optimization function using modular components."""
    # Create input sequence using modular input provider
    model_dt = float(BASE_SPEC.get("simulation", {}).get("dt", 1.0))
    dt_override = config.get("dt")
    effective_dt = float(dt_override) if dt_override is not None else model_dt
    dt_s = dt_units_to_seconds(effective_dt)
    D0 = get_input_dim_from_spec(BASE_SPEC)

    config = dict(config)
    config["dt_s"] = dt_s
    prov = create_input_provider(str(config.get("input_kind", "")), config, BASE_SPEC, DEVICE)
    if prov is None:
        msg = f"Unsupported input_kind: {config.get('input_kind')}"
        raise ValueError(msg)
    x_in = prov.get_batch(batch_size=int(config["batch_size"]), seq_len=int(config["seq_len"]), dim=int(D0))

    # Create study using modular adapter
    study_name = config.get("study_name") or f"criticality_{int(time.time())}"
    seed = int(config.get("seed", 42))
    sampler_algo = _build_sampler(str(config.get("optuna_sampler", "tpe")), seed=seed, kwargs=config.get("optuna_sampler_kwargs", {}))
    pruner = _build_pruner(config.get("pruner", {}))
    study = _create_study(study_name, out_dir=OUT_DIR, sampler=sampler_algo, pruner=pruner, resume=bool(config.get("resume", False)))

    # Update config if study name changed due to duplicates
    if study.study_name != study_name:
        config["study_name"] = study.study_name

    # Objective function
    def objective(trial):
        return evaluate_trial(trial, config, x_in, hp_sampler)

    # Run optimization with progress callbacks
    t0 = time.time()
    callbacks = list(_make_callbacks(study.study_name, out_dir=OUT_DIR))
    study.optimize(objective, n_trials=config["n_trials"], timeout=config.get("timeout"), n_jobs=(config.get("n_jobs") or 1), callbacks=callbacks)

    return study, time.time() - t0


# ---------------------------------------------------------------------
# Results analysis and saving
# ---------------------------------------------------------------------
def _is_fail_fast() -> bool:
    try:
        import os as _os

        return str(_os.environ.get("SOEN_HPO_FAIL_FAST", "")).strip() in ("1", "true", "True")
    except Exception:
        return False


def save_results(study: optuna.Study, elapsed_time: float, config: dict) -> None:
    """Save optimization results using modular exporters."""
    # Write summary and best spec using centralized helpers
    _write_summary(study, elapsed_time, config, OUT_DIR)

    # Rebuild the best spec from the best trial's parameters using the sampler
    try:
        from copy import deepcopy as _dc

        best_trial = study.best_trial
        # Base spec to patch
        base_spec_local = _dc(BASE_SPEC)
        # Use the same HPO config to create a sampler and reconstitute the trial spec
        hp_cfg_path = str(config.get("hp_config") or "")
        sampler_for_save = create_sampler_from_config(hp_cfg_path, preset=config.get("preset"))
        fixed = optuna.trial.FixedTrial(dict(best_trial.params))
        trial_spec = build_config_from_trial_v2(fixed, base_spec_local, sampler_for_save)
        # Apply dt override if present
        try:
            if config.get("dt") is not None:
                (trial_spec.setdefault("simulation", {}))["dt"] = float(config["dt"])
        except Exception:
            pass
        # Attach trial metadata for reference
        try:
            trial_spec["trial_parameters"] = dict(best_trial.params)
            trial_spec["trial_metrics"] = dict(best_trial.user_attrs)
            trial_spec["trial_info"] = {
                "number": int(best_trial.number),
                "study_name": study.study_name,
                "value": float(best_trial.value) if best_trial.value is not None else None,
                "state": best_trial.state.name,
            }
        except Exception:
            pass
        # Coerce additional fields if needed (dt/attrs)
        best_spec = _coerce_best_spec(trial_spec, dt=config.get("dt"), trial_user_attrs=(best_trial.user_attrs if hasattr(best_trial, "user_attrs") else {}))
    except Exception as _e:
        # Fallback to legacy behavior (base spec) if reconstruction fails
        from copy import deepcopy as _dc

        best_spec = _dc(BASE_SPEC)
        best_spec = _coerce_best_spec(best_spec, dt=config.get("dt"), trial_user_attrs=(study.best_trial.user_attrs if hasattr(study.best_trial, "user_attrs") else {}))
    # Persist best spec YAML
    spec_path = _write_best_spec_yaml(best_spec, study.study_name, OUT_DIR)
    # Also save a binary .soen sidecar of the best spec for immediate loading
    try:
        best_model = SOENModelCore.build(best_spec)
        soen_path = os.path.splitext(spec_path)[0] + ".soen"
        best_model.save(soen_path)
    except Exception as _e:
        pass


def _find_existing_study_dir_for_config(hp_config_path: str, out_dir: str, study_name: str | None = None) -> str:
    """Locate an existing study directory for the given HPO config.

    Strategy:
      1) If study_name is provided, use OUT_DIR/optuna_report_{study_name} and verify DB exists.
      2) Otherwise, scan OUT_DIR for optuna_report_* dirs, open their optuna_summary_*.json,
         and pick the most recent one whose stored config['hp_config'] matches hp_config_path.
    """
    out_dir = str(out_dir)
    abs_cfg = os.path.abspath(hp_config_path)
    if study_name:
        cand = os.path.join(out_dir, f"optuna_report_{study_name}")
        if not os.path.isdir(cand):
            msg = f"Study directory not found: {cand}"
            raise FileNotFoundError(msg)
        if not any(f.endswith(".db") for f in os.listdir(cand)):
            msg = f"No database (.db) found in study directory: {cand}"
            raise FileNotFoundError(msg)
        return cand

    # Scan OUT_DIR
    candidates: list[tuple[float, str]] = []
    try:
        for name in os.listdir(out_dir):
            if not name.startswith("optuna_report_"):
                continue
            sd = os.path.join(out_dir, name)
            if not os.path.isdir(sd):
                continue
            # quick db presence check
            if not any(f.endswith(".db") for f in os.listdir(sd)):
                continue
            # Try to match by summary config.hp_config
            try:
                summary_files = [f for f in os.listdir(sd) if f.startswith("optuna_summary_") and f.endswith(".json")]
                if not summary_files:
                    continue
                # Use the newest summary
                summary_files.sort(key=lambda fn: os.path.getmtime(os.path.join(sd, fn)), reverse=True)
                import json as _json

                with open(os.path.join(sd, summary_files[0])) as _f:
                    summary = _json.load(_f)
                cfg_in_summary = (summary.get("config") or {}).get("hp_config")
                if cfg_in_summary and os.path.abspath(str(cfg_in_summary)) == abs_cfg:
                    candidates.append((os.path.getmtime(os.path.join(sd, summary_files[0])), sd))
            except Exception:
                # ignore broken summaries
                continue
    except FileNotFoundError:
        pass

    if not candidates:
        msg = "No completed study found for this HPO config. Set run.study_name in the HPO config and run optimization first, or provide --study-name when launching the optimization."
        raise FileNotFoundError(
            msg,
        )
    # Pick the most recent
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# Dashboard generation removed - handled in GUI now


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna-based criticality hyperparameter optimization")
    # Minimal commonly-used flags; YAML is the source of truth for most settings
    parser.add_argument("--hp-config", type=str, default="", help="Path to HPO config YAML")
    parser.add_argument("--preset", type=str, default=None, help="Optional preset from HPO config")
    parser.add_argument("--resume", action="store_true", help="Resume optimization from existing study")
    parser.add_argument("--trials", type=int, default=None, help="Override number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Override optimization timeout (seconds)")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override number of parallel jobs")
    parser.add_argument("--study-name", type=str, default=None, help="Override study name")
    parser.add_argument("--seq-len", type=int, default=None, help="Override input sequence length")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for simulation")
    parser.add_argument("--target-connections", type=str, default=None, help="Comma-separated connections to optimize, e.g. J_0_to_1,J_1_to_1")
    # Extraction utility: export a specific trial's spec from an existing study and exit
    parser.add_argument("--extract-trial-num", type=int, default=None, help="Extract a specific trial's YAML spec and exit")
    parser.add_argument("--extract-output", type=str, default=None, help="Output YAML file path for extraction (defaults to study_dir/trial_<num>_spec.yaml)")
    parser.add_argument("--plot-example-inputs", type=bool, default=False)
    args = parser.parse_args()

    # Load HPO YAML settings (optional when extracting with study name)
    hp_config_path = args.hp_config
    hp_all: dict[str, Any] = {}
    if os.path.exists(hp_config_path):
        # Use robust loader to normalize/resolve/auto-extract spec and backfill opt config
        hp_all = _load_hpo_config(hp_config_path, allow_extract=True)
    else:
        msg = f"HPO config YAML not found: {hp_config_path}"
        raise FileNotFoundError(msg)

    # Resolve paths and load BASE_SPEC (skip strict requirement if extracting without YAML)
    global OUT_DIR, BASE_MODEL_SPEC, BASE_SPEC
    # Normalize relative paths in YAML
    if hp_all:
        hp_all = _normalize_paths(hp_all, base_yaml_path=hp_config_path)
    paths_cfg = hp_all.get("paths") or {}
    OUT_DIR = paths_cfg.get("output_dir", OUT_DIR)
    BASE_MODEL_SPEC = paths_cfg.get("base_model_spec", BASE_MODEL_SPEC)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    if os.path.exists(hp_config_path):
        # Prefer using the user's file as-is:
        # - If it's a YAML/JSON spec → read it directly
        # - If it's a trained model file → load model and build spec in memory (no temp file)
        def _is_trained_model_file(p: str) -> bool:
            s = str(p).lower()
            return s.endswith((".soen", ".pth", ".pt", ".json")) and not s.endswith(("_spec.yaml", "_spec_for_hpo.yaml"))

        bms_path = BASE_MODEL_SPEC
        if isinstance(bms_path, str) and os.path.exists(bms_path) and _is_trained_model_file(bms_path):
            # Load model and build spec dict in-memory
            try:
                from dataclasses import asdict as _asdict

                m = SOENModelCore.load(bms_path, device=None, strict=False, verbose=False, show_logs=False)
                BASE_SPEC = {
                    "simulation": _asdict(m.sim_config),
                    "layers": [_asdict(cfg) for cfg in m.layers_config],
                    "connections": [_asdict(cfg) for cfg in m.connections_config],
                }
            except Exception as _e:
                msg = f"Failed to load model file '{bms_path}' to extract configs: {_e}"
                raise RuntimeError(msg)
        else:
            # Fall back to resolver which may locate an existing spec file or extract one if available
            resolved_bms, updated_paths = _resolve_bms(paths_cfg, allow_extract=True)
            if resolved_bms:
                BASE_MODEL_SPEC = resolved_bms
                hp_all["paths"] = updated_paths
            if not os.path.exists(BASE_MODEL_SPEC):
                msg = f"Base model spec not found: {BASE_MODEL_SPEC}"
                raise FileNotFoundError(msg)
            with open(BASE_MODEL_SPEC) as f:
                BASE_SPEC = yaml.safe_load(f) or {}

    # Build runtime config from YAML
    run_cfg = hp_all.get("run", {}) or {}
    sim_cfg = hp_all.get("simulation", {}) or {}
    input_block = hp_all.get("input", {}) or {}
    objective_cfg = hp_all.get("objective", {}) or {}
    weights_cfg = objective_cfg.get("weights", {}) or {}
    targets_cfg = objective_cfg.get("targets", {}) or {}
    optuna_cfg = hp_all.get("optuna", {}) or {}
    pruner_cfg = hp_all.get("pruner", {}) or {}
    # Backfill optimization_config targets/spaces if missing
    hp_all = _populate_oc(hp_all, BASE_MODEL_SPEC if os.path.exists(str(BASE_MODEL_SPEC)) else None)
    opt_space_cfg = hp_all.get("optimization_config", {}) or {}

    config: dict[str, Any] = {
        "hp_config": hp_config_path,
        "preset": args.preset,
        # run
        "n_trials": run_cfg.get("n_trials"),
        "timeout": run_cfg.get("timeout"),
        "n_jobs": run_cfg.get("n_jobs"),
        "seed": run_cfg.get("seed", 42),
        "study_name": run_cfg.get("study_name"),
        "resume": bool(run_cfg.get("resume", False) or args.resume),
        # simulation
        "seq_len": sim_cfg.get("seq_len"),
        "batch_size": sim_cfg.get("batch_size"),
        "dt": sim_cfg.get("dt"),
        # input block (canonicalization applied below)
        "input": input_block,
        # objective weights
        "w_branch": weights_cfg.get("w_branch", 1.0),
        "w_psd_t": weights_cfg.get("w_psd_t", 0.0),
        "w_psd_spatial": weights_cfg.get("w_psd_spatial", 0.0),
        "w_chi_inv": weights_cfg.get("w_chi_inv", 0.0),
        "w_avalanche": weights_cfg.get("w_avalanche", 0.0),
        "w_autocorr": weights_cfg.get("w_autocorr", 0.0),
        "w_jacobian": weights_cfg.get("w_jacobian", 0.0),
        "w_lyapunov": weights_cfg.get("w_lyapunov", 0.0),
        # objective targets
        "target_beta_t": targets_cfg.get("target_beta_t", 2.0),
        "target_beta_spatial": targets_cfg.get("target_beta_spatial", 2.0),
        # optuna
        "optuna_sampler": optuna_cfg.get("sampler", "tpe"),
        "optuna_sampler_kwargs": optuna_cfg.get("sampler_kwargs", {}),
        # pruner
        "pruner": {
            "use": bool(pruner_cfg.get("use", False)),
            "n_startup_trials": pruner_cfg.get("n_startup_trials", 20),
            "n_warmup_steps": pruner_cfg.get("n_warmup_steps", 5),
        },
        # target scope
        "target_connections": opt_space_cfg.get("target_connections"),
        "target_layers": opt_space_cfg.get("target_layers"),
    }

    # Apply canonicalization from the HPO 'input' block
    config = apply_input_block_to_config(config)

    # Re-apply canonicalization on YAML 'input' block only
    config = apply_input_block_to_config(config)

    # (Extraction code removed; use src/soen_toolkit/utils/hpo/extract_trial_spec.py)

    # Misc overrides
    if args.trials is not None:
        config["n_trials"] = args.trials
    if args.timeout is not None:
        config["timeout"] = args.timeout
    if args.n_jobs is not None:
        config["n_jobs"] = args.n_jobs
    if args.study_name is not None:
        config["study_name"] = args.study_name
    if args.seq_len is not None:
        config["seq_len"] = args.seq_len
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.target_connections:
        config["target_connections"] = [s.strip() for s in args.target_connections.split(",")]

    # Finalize required simulation defaults if missing from YAML/overrides
    try:
        if config.get("seq_len") is None:
            config["seq_len"] = 200
        if config.get("batch_size") is None:
            config["batch_size"] = 1
        # Ensure ints
        config["seq_len"] = int(config["seq_len"])
        config["batch_size"] = int(config["batch_size"])
    except Exception:
        config["seq_len"] = 200
        config["batch_size"] = 1

    if args.plot_example_inputs:
        # Get model info for input generation from BASE_SPEC
        model_dt = float(BASE_SPEC.get("simulation", {}).get("dt", 1.0))
        # Respect optional dt override (previously ignored here)
        effective_dt = float(config.get("dt")) if config.get("dt") is not None else model_dt
        dt_s = dt_units_to_seconds(effective_dt)
        layers = BASE_SPEC.get("layers", [])
        D0 = next(
            (layer.get("params", {}).get("dim") for layer in layers if int(layer.get("layer_id", -1)) == 0),
            layers[0].get("params", {}).get("dim") if layers else 1,
        )

        # Generate 5 example input sequences using modular system
        n_examples = 5
        inputs = []
        # Use CPU device for plotting
        plot_config = dict(config)
        plot_config["dt_s"] = dt_s
        prov = create_input_provider(str(config.get("input_kind", "")), plot_config, BASE_SPEC, "cpu")
        if prov is None:
            msg = f"Unsupported input_kind for plotting: {config.get('input_kind')}"
            raise ValueError(msg)

        for i in range(n_examples):
            x_in = prov.get_batch(batch_size=1, seq_len=config["seq_len"], dim=D0)
            inputs.append(x_in[0].numpy())  # [seq_len, dim], remove batch dim

        # Create time axis (use effective dt, not model_dt)
        time_steps = np.arange(config["seq_len"]) * effective_dt
        time_ns = time_steps * DT_PS * 1e-3  # Convert to nanoseconds

        # Plot using matplotlib (fail fast if missing)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(n_examples, 1, figsize=(12, 2 * n_examples), sharex=True)
        if n_examples == 1:
            axes = [axes]

        for i, (ax, inp) in enumerate(zip(axes, inputs, strict=False)):
            # Plot first few dimensions for clarity
            n_dims_to_plot = min(5, D0)
            for d in range(n_dims_to_plot):
                ax.plot(time_ns, inp[:, d], alpha=0.7, linewidth=1, label=f"dim {d}")

            ax.set_ylabel(f"Input {i + 1}\nAmplitude")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                ax.set_title(f"Example Input Sequences (showing {n_dims_to_plot}/{D0} dimensions)")

        axes[-1].set_xlabel("Time (ns)")

        # Add timing info
        total_time_ns = time_ns[-1]
        if config["input_kind"] == "white_noise":
            fig.suptitle(f"White-noise Inputs: δn={config['delta_n']}, total_time={total_time_ns:.3f}ns, dt={dt_s * 1e9:.4f}ns")
        elif config["input_kind"] == "colored_noise":
            fig.suptitle(f"Colored-noise (1/f^β) Inputs: β={config['colored_beta']}, total_time={total_time_ns:.3f}ns, dt={dt_s * 1e9:.4f}ns")
        else:
            gp_length_scale_ns = config["input_ell_ns"]
            fig.suptitle(f"GP Input Sequences: σ²={config['input_sigma']}, ℓ={gp_length_scale_ns}ns, total_time={total_time_ns:.3f}ns, dt={dt_s * 1e9:.4f}ns")

        plt.tight_layout()

        # Save plot
        preview_name = config.get("study_name") or "preview"
        plot_path = os.path.join(get_study_dir(preview_name), "example_inputs.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")

        # Show plot
        plt.show()

        if config["input_kind"] == "white_noise" or config["input_kind"] == "colored_noise" or config["input_kind"] == "gp_rbf":
            pass
        elif config["input_kind"] == "hdf5_dataset":
            if (config.get("input_scale_min") is not None) and (config.get("input_scale_max") is not None):
                pass
        else:
            pass

        return

    # Show key settings and compute simulation timing info

    # Get timing info from BASE_SPEC, respecting optional dt override
    model_dt = float(BASE_SPEC.get("simulation", {}).get("dt", 1.0))  # dt units from sim_config
    effective_dt = float(config.get("dt")) if config.get("dt") is not None else model_dt
    dt_s = dt_units_to_seconds(effective_dt)  # convert to seconds
    layers = BASE_SPEC.get("layers", [])
    D0 = next(
        (layer.get("params", {}).get("dim") for layer in layers if int(layer.get("layer_id", -1)) == 0),
        layers[0].get("params", {}).get("dim") if layers else 1,
    )

    # Simulation timing
    total_sim_time_dt = config["seq_len"] * effective_dt
    total_sim_time_s = total_sim_time_dt * DT_PS * 1e-12
    total_sim_time_s * 1e9

    ik = config.get("input_kind")
    if ik in {"white_noise", "colored_noise"} or ik in {"log_slope_noise", "gp_rbf"}:
        pass
    elif ik == "hdf5_dataset":
        if (config.get("input_scale_min") is not None) and (config.get("input_scale_max") is not None):
            pass
    else:
        pass

    # Instantiate configuration-driven sampler
    sampler = create_sampler_from_config(args.hp_config, preset=args.preset)
    # If specific connections were provided via CLI, override sampler config
    if config.get("target_connections"):
        sampler.config["target_connections"] = config["target_connections"]

        study, dt = run_optimization(config, sampler)

    # Save results
    save_results(study, dt, config)

    # Dashboard generation handled in GUI


if __name__ == "__main__":
    main()
