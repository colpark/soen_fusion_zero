from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from soen_toolkit.core import SOENModelCore

from .hpo_config import (
    load_hpo_config as _load_hpo_config,
    populate_optimization_config as _populate_oc,
    resolve_base_model_spec as _resolve_bms,
)

if TYPE_CHECKING:
    from argparse import Namespace


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


def setup_config_and_paths(args: Namespace) -> tuple[dict[str, Any], dict[str, Any], str]:
    # Load HPO YAML settings
    hp_config_path = args.hp_config
    hp_all: dict[str, Any] = {}
    if os.path.exists(hp_config_path):
        hp_all = _load_hpo_config(hp_config_path, allow_extract=True)
    else:
        msg = f"HPO config YAML not found: {hp_config_path}"
        raise FileNotFoundError(msg)

    # Resolve paths and load BASE_SPEC
    paths_cfg = hp_all.get("paths") or {}
    out_dir = paths_cfg.get("output_dir", os.path.join(os.getcwd(), "criticality_runs"))
    base_model_spec_path = paths_cfg.get("base_model_spec", "")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if os.path.exists(hp_config_path):

        def _is_trained_model_file(p: str) -> bool:
            s = str(p).lower()
            return s.endswith((".soen", ".pth", ".pt", ".json")) and not s.endswith(("_spec.yaml", "_spec_for_hpo.yaml"))

        bms_path = base_model_spec_path
        if isinstance(bms_path, str) and os.path.exists(bms_path) and _is_trained_model_file(bms_path):
            try:
                from dataclasses import asdict as _asdict

                m = SOENModelCore.load(bms_path, device=None, strict=False, verbose=False, show_logs=False)
                base_spec = {
                    "simulation": _asdict(m.sim_config),
                    "layers": [_asdict(cfg) for cfg in m.layers_config],
                    "connections": [_asdict(cfg) for cfg in m.connections_config],
                }
            except Exception as _e:
                msg = f"Failed to load model file '{bms_path}' to extract configs: {_e}"
                raise RuntimeError(msg)
        else:
            resolved_bms, updated_paths = _resolve_bms(paths_cfg, allow_extract=True)
            if resolved_bms:
                base_model_spec_path = resolved_bms
                hp_all["paths"] = updated_paths
            if not os.path.exists(base_model_spec_path):
                msg = f"Base model spec not found: {base_model_spec_path}"
                raise FileNotFoundError(msg)
            with open(base_model_spec_path) as f:
                base_spec = yaml.safe_load(f) or {}

    # Build runtime config from YAML
    run_cfg = hp_all.get("run", {}) or {}
    sim_cfg = hp_all.get("simulation", {}) or {}
    input_block = hp_all.get("input", {}) or {}
    objective_cfg = hp_all.get("objective", {}) or {}
    weights_cfg = objective_cfg.get("weights", {}) or {}
    targets_cfg = objective_cfg.get("targets", {}) or {}
    optuna_cfg = hp_all.get("optuna", {}) or {}
    pruner_cfg = hp_all.get("pruner", {}) or {}
    hp_all = _populate_oc(hp_all, base_model_spec_path if os.path.exists(str(base_model_spec_path)) else None)
    opt_space_cfg = hp_all.get("optimization_config", {}) or {}

    config: dict[str, Any] = {
        "hp_config": hp_config_path,
        "hpo_mode": run_cfg.get("hpo_mode", "forward"),
        "train_config": paths_cfg.get("train_config"),
        "preset": args.preset,
        "n_trials": run_cfg.get("n_trials"),
        "timeout": run_cfg.get("timeout"),
        "n_jobs": run_cfg.get("n_jobs"),
        "seed": run_cfg.get("seed", 42),
        "study_name": run_cfg.get("study_name"),
        "resume": bool(run_cfg.get("resume", False) or args.resume),
        "seq_len": sim_cfg.get("seq_len"),
        "batch_size": sim_cfg.get("batch_size"),
        "dt": sim_cfg.get("dt"),
        "input": input_block,
        "w_branch": weights_cfg.get("w_branch", 1.0),
        "w_psd_t": weights_cfg.get("w_psd_t", 0.0),
        "w_psd_spatial": weights_cfg.get("w_psd_spatial", 0.0),
        "w_chi_inv": weights_cfg.get("w_chi_inv", 0.0),
        "w_avalanche": weights_cfg.get("w_avalanche", 0.0),
        "w_autocorr": weights_cfg.get("w_autocorr", 0.0),
        "w_jacobian": weights_cfg.get("w_jacobian", 0.0),
        "w_lyapunov": weights_cfg.get("w_lyapunov", 0.0),
        "w_train_loss": weights_cfg.get("w_train_loss", 0.0),
        "target_beta_t": targets_cfg.get("target_beta_t", 2.0),
        "target_beta_spatial": targets_cfg.get("target_beta_spatial", 2.0),
        "optuna_sampler": optuna_cfg.get("sampler", "tpe"),
        "optuna_sampler_kwargs": optuna_cfg.get("sampler_kwargs", {}),
        "pruner": {
            "use": bool(pruner_cfg.get("use", False)),
            "n_startup_trials": pruner_cfg.get("n_startup_trials", 20),
            "n_warmup_steps": pruner_cfg.get("n_warmup_steps", 5),
        },
        "target_connections": opt_space_cfg.get("target_connections"),
        "target_layers": opt_space_cfg.get("target_layers"),
    }

    config = apply_input_block_to_config(config)

    # Overrides from CLI args
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

    try:
        if config.get("seq_len") is None:
            config["seq_len"] = 200
        if config.get("batch_size") is None:
            config["batch_size"] = 1
        config["seq_len"] = int(config["seq_len"])
        config["batch_size"] = int(config["batch_size"])
    except Exception:
        config["seq_len"] = 200
        config["batch_size"] = 1

    return config, base_spec, out_dir
