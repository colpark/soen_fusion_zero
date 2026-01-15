from __future__ import annotations

import contextlib
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from soen_toolkit.utils.hpo.config.hpo_config import load_hpo_config as _load_hpo_config

if TYPE_CHECKING:
    import optuna


class ConfigurableHyperparameterSampler:
    """A sampler that builds a model configuration by sampling from a structured search space."""

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


def build_config_from_trial(trial: optuna.Trial, base_spec: dict[str, Any], sampler: ConfigurableHyperparameterSampler) -> dict[str, Any]:
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
