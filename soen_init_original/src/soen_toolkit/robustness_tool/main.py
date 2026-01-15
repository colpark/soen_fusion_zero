import argparse
import contextlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import yaml

# Import core toolkit
try:
    from soen_toolkit.core import SOENModelCore
    from soen_toolkit.training.data.dataloaders import create_data_loaders
except Exception:
    sys.exit(1)


# --------------------------------------
# Logging
# --------------------------------------


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# --------------------------------------
# Config
# --------------------------------------


@dataclass
class StudyConfig:
    model_path: str
    data_path: str
    results_dir: str

    device: str = "cpu"
    batch_size: int = 64
    num_workers: int = 0
    max_eval_batches: int | None = 50
    eval_split: str = "val"  # train|val|test
    monte_carlo_runs: int = 10

    # Optional overrides to match training
    model_dt: float | None = None
    time_pooling: str = "max"
    time_pooling_params: dict[str, Any] = field(default_factory=lambda: {"scale": 1.0})

    # Data options
    cache_data: bool = True
    scale_min: float | None = None
    scale_max: float | None = None
    target_seq_len: int | None = None
    input_encoding: str = "raw"
    vocab_size: int | None = None
    one_hot_dtype: str = "float32"

    # Experiments
    experiments: list[dict[str, Any]] = field(default_factory=list)

    log_level: str = "INFO"
    plot_format: str = "pdf"
    error_type: str = "std"

    def validate(self) -> None:
        p = Path(self.model_path)
        if not p.exists():
            msg = f"Model file not found: {p}"
            raise FileNotFoundError(msg)
        d = Path(self.data_path)
        if not d.exists():
            msg = f"Data file not found: {d}"
            raise FileNotFoundError(msg)
        if self.eval_split not in {"train", "val", "test"}:
            msg = "eval_split must be one of train|val|test"
            raise ValueError(msg)
        if self.input_encoding not in {"raw", "one_hot"}:
            msg = "input_encoding must be raw or one_hot"
            raise ValueError(msg)


# --------------------------------------
# Pooling (with scale)
# --------------------------------------


def pool_time(final_state: torch.Tensor, method: str, params: dict[str, Any]) -> torch.Tensor:
    history = final_state
    scale = float(params.get("scale", 1.0)) if isinstance(params, dict) else 1.0
    if method == "max":
        out = torch.max(history, dim=1)[0]
    elif method == "mean":
        out = torch.mean(history, dim=1)
    elif method == "rms":
        out = torch.sqrt(torch.mean(history**2, dim=1) + 1e-8)
    elif method == "final":
        out = history[:, -1, :]
    elif method == "mean_last_n":
        n = int(params.get("n", 1))
        n = max(1, min(n, history.size(1)))
        out = torch.mean(history[:, -n:, :], dim=1)
    elif method == "mean_range":
        total_timesteps = history.size(1)
        default_points = min(50, total_timesteps)
        start_idx = int(params.get("range_start", max(0, total_timesteps - default_points)))
        end_idx = int(params.get("range_end", total_timesteps))
        start_idx = max(0, min(start_idx, total_timesteps - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_timesteps))
        out = torch.mean(history[:, start_idx:end_idx, :], dim=1)
    elif method == "ewa":
        total_timesteps = history.size(1)
        min_weight = float(params.get("min_weight", 0.2))
        min_weight = max(1e-6, min(min_weight, 1.0))
        t = torch.linspace(0.0, 1.0, steps=total_timesteps, device=history.device)
        weights = min_weight ** (1.0 - t)
        weights = weights / weights.sum()
        out = torch.einsum("btf,t->bf", history, weights)
    else:
        out = torch.max(history, dim=1)[0]
    return out * scale if scale != 1.0 else out


# --------------------------------------
# V1 compatibility helpers implemented locally to avoid circular imports
# --------------------------------------


class ParameterAnalyzer:
    """Lightweight analyzer for layer/internal parameters.

    Provides just enough functionality for v2 robustness runs:
    - discover_parameter_info(param_name): report nominal value and log-space flag
    - apply_perturbation_inplace(cfg, level): apply additive or multiplicative shifts
    - restore_original_params(): reset mutated parameters to original snapshot

    Notes:
    - Targets the provided layer IDs only. For param_name == 'internal_J', uses the
      model's connection key 'J_<layer_id>_to_<layer_id>' (legacy 'internal_<layer_id>' supported).
    - For log parameters (stored as 'log_<name>'), we operate in real space by exp/log.
    - Distributions supported: 'relative' (default), 'absolute'. Other values fall back
      to 'relative'.

    """

    def __init__(self, model: SOENModelCore, target_layers: list[int]) -> None:
        self.model = model
        self.target_layers = list(target_layers or [])

        # Map layer_id -> module index
        self._layer_id_to_index: dict[int, int] = {cfg.layer_id: idx for idx, cfg in enumerate(self.model.layers_config)}

        # Snapshot original parameters for the target layers
        self._param_snapshots: dict[tuple[int, str], torch.Tensor] = {}
        for layer_id in self.target_layers:
            idx = self._layer_id_to_index.get(layer_id)
            if idx is None:
                continue
            layer = self.model.layers[idx]
            for name, param in layer.named_parameters(recurse=True):
                with contextlib.suppress(Exception):
                    self._param_snapshots[(idx, name)] = param.detach().clone()

        # Also snapshot internal_J for each targeted layer if present
        for layer_id in self.target_layers:
            new_key = f"J_{layer_id}_to_{layer_id}"
            old_key = f"internal_{layer_id}"
            key_to_use = new_key if new_key in self.model.connections else old_key
            if key_to_use in self.model.connections:
                with contextlib.suppress(Exception):
                    self._param_snapshots[(-1, key_to_use)] = self.model.connections[key_to_use].detach().clone()

    def restore_original_params(self) -> None:
        with torch.no_grad():
            for (idx, name), tensor in self._param_snapshots.items():
                if idx == -1:
                    # connection tensor like J_i_to_i (or legacy internal_i)
                    if name in self.model.connections:
                        self.model.connections[name].data.copy_(tensor)
                    continue
                if 0 <= idx < len(self.model.layers):
                    layer = self.model.layers[idx]
                    # traverse to the parameter by name path
                    target_param = None
                    for n, p in layer.named_parameters(recurse=True):
                        if n == name:
                            target_param = p
                            break
                    if isinstance(target_param, torch.nn.Parameter):
                        target_param.data.copy_(tensor)

    def _find_param_tensors(self, param_name: str) -> list[tuple[int, str, torch.nn.Parameter, bool]]:
        """Return list of (layer_idx, name, tensor, is_log) for matching params.

        Matching rules:
        - exact match (e.g., 'bias_current')
        - numbered suffixes (e.g., 'gamma_plus_1', 'gamma_plus_2') when base provided
        - log parameters recorded as 'log_<name>'
        """
        matches: list[tuple[int, str, torch.nn.Parameter, bool]] = []
        base = param_name

        for layer_id in self.target_layers:
            idx = self._layer_id_to_index.get(layer_id)
            if idx is None:
                continue
            layer = self.model.layers[idx]

            # First try exact and log_ exact
            cand: dict[str, torch.nn.Parameter] = dict(layer.named_parameters(recurse=True))

            # exact real-space
            if base in cand and isinstance(cand[base], torch.nn.Parameter):
                matches.append((idx, base, cand[base], False))
                continue

            # exact log-space
            log_name = f"log_{base}"
            if log_name in cand and isinstance(cand[log_name], torch.nn.Parameter):
                matches.append((idx, log_name, cand[log_name], True))
                continue

            # Fallback: any parameter that startswith the base (handles *_1, *_2, ...)
            for name, p in cand.items():
                if not isinstance(p, torch.nn.Parameter):
                    continue
                if name == base or name.startswith(f"{base}_"):
                    matches.append((idx, name, p, False))
                elif name == f"log_{base}" or name.startswith(f"log_{base}_"):
                    matches.append((idx, name, p, True))

        return matches

    def discover_parameter_info(self, param_name: str) -> dict:
        if param_name == "internal_J":
            # Aggregate across targeted layers' internal connections, if present
            vals = []
            for layer_id in self.target_layers:
                key = f"internal_{layer_id}"
                if key in self.model.connections:
                    t = self.model.connections[key].detach()
                    if t.numel() > 0:
                        vals.append(float(t.mean().item()))
            nominal = float(sum(vals) / len(vals)) if vals else 0.0
            return {"nominal_value": nominal, "is_log_param": False}

        matches = self._find_param_tensors(param_name)
        if not matches:
            return {"nominal_value": 0.0, "is_log_param": False}

        # Compute mean in real space for the first match group
        _idx, _name, tensor, is_log = matches[0]
        data = tensor.detach()
        if is_log:
            try:
                data = data.exp()
            except Exception:
                data = torch.exp(data)
        nominal = float(data.mean().item()) if data.numel() > 0 else 0.0
        return {"nominal_value": nominal, "is_log_param": bool(is_log)}

    def apply_perturbation_inplace(self, cfg: object, level: float) -> None:
        """Apply an additive or multiplicative shift based on cfg.distribution.

        - For 'absolute': new = nominal + level
        - For 'relative' (default): new = nominal * (1 + level)
        - For log params we operate in real space and write back to log-space tensors.
        """
        param_name = getattr(cfg, "param_name", None)
        distribution = str(getattr(cfg, "distribution", "relative")).lower()
        hard_limits = bool(getattr(cfg, "hard_limits", False))
        nominal = float(getattr(cfg, "nominal_value", 0.0))

        if param_name == "internal_J":
            # Adjust internal connections for targeted layers
            with torch.no_grad():
                for layer_id in self.target_layers:
                    new_key = f"J_{layer_id}_to_{layer_id}"
                    old_key = f"internal_{layer_id}"
                    key_to_use = new_key if new_key in self.model.connections else old_key
                    if key_to_use not in self.model.connections:
                        continue
                    J = self.model.connections[key_to_use]
                    if distribution == "absolute":
                        J.add_(float(level))
                    else:
                        J.mul_(1.0 + float(level))
            return

        # Layer parameters
        if param_name is None:
            return
        matches = self._find_param_tensors(param_name)
        if not matches:
            return

        with torch.no_grad():
            for _idx, _name, tensor, is_log in matches:
                # Compute target real value
                if distribution == "absolute":
                    real_target = nominal + float(level)
                else:
                    real_target = nominal * (1.0 + float(level))

                # Write back (log vs real)
                if is_log:
                    try:
                        tensor.copy_(torch.log(torch.full_like(tensor, real_target)))
                    except Exception:
                        tensor.data = torch.log(torch.full_like(tensor, real_target))
                else:
                    try:
                        tensor.copy_(torch.full_like(tensor, real_target))
                    except Exception:
                        tensor.data = torch.full_like(tensor, real_target)

        # Optional clamp via existing constraint mechanism (forward() also clamps)
        if hard_limits:
            with contextlib.suppress(Exception):
                self.model.enforce_param_constraints()


class WeightQuantizer:
    """In-place weight quantization utilities with snapshot/restore.

    Matches semantics used throughout training/utilities via utils.quantization.
    """

    def __init__(self, model: SOENModelCore) -> None:
        self.model = model
        self._snapshot: dict[str, torch.Tensor] | None = None

    def restore_original_weights(self) -> None:
        if not self._snapshot:
            return
        from soen_toolkit.utils.quantization import restore_connection_tensors

        restore_connection_tensors(self.model, self._snapshot)
        self._snapshot = None

    def _ensure_snapshot(self, names: list[str]) -> None:
        from soen_toolkit.utils.quantization import snapshot_connection_tensors

        if self._snapshot is None:
            self._snapshot = snapshot_connection_tensors(self.model, names)

    def _resolve_target_names(self, names: list[str] | None) -> list[str]:
        if names is not None:
            return list(names)
        # Default to all quantizable connections (learnable by default)
        from soen_toolkit.utils.quantization import list_quantizable_connection_names

        return list_quantizable_connection_names(self.model)

    def quantize_connection_weights(self, connection_names: list[str] | None, min_value: float, max_value: float, num_levels: int) -> None:
        from soen_toolkit.utils.quantization import (
            generate_uniform_codebook,
            quantize_connections_in_place,
        )

        names = self._resolve_target_names(connection_names)
        self._ensure_snapshot(names)

        # Build codebook on a sane device/dtype
        try:
            first = next(iter(self.model.connections.values()))
            device = first.device
            dtype = first.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32
        codebook = generate_uniform_codebook(float(min_value), float(max_value), int(num_levels), device=device, dtype=dtype)
        quantize_connections_in_place(self.model, codebook, connections=names)

    def quantize_connection_weights_stochastic(self, connection_names: list[str] | None, min_value: float, max_value: float, num_levels: int, temperature: float = 1.0) -> None:
        """Stochastic snapping: soft assignment with temperature before hard round.

        This is a lightweight approximation sufficient for robustness sweeps.
        """
        names = self._resolve_target_names(connection_names)
        self._ensure_snapshot(names)

        # Build codebook
        try:
            first = next(iter(self.model.connections.values()))
            device = first.device
            dtype = first.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32

        from soen_toolkit.utils.quantization import generate_uniform_codebook

        codebook = generate_uniform_codebook(float(min_value), float(max_value), int(num_levels), device=device, dtype=dtype)

        # Soft nearest with temperature then take argmax
        with torch.no_grad():
            for name in names:
                if name not in self.model.connections:
                    continue
                W = self.model.connections[name]
                flat = W.view(-1, 1)
                cb = codebook.view(1, -1).to(W.device)
                # Negative distance; scale by temperature
                logits = -((flat - cb).abs()) / max(1e-8, float(temperature))
                probs = torch.softmax(logits, dim=1)
                idx = probs.argmax(dim=1)
                snapped = cb[0, idx].view_as(W)
                W.copy_(snapped)


# Aliases for local v1 parity
V1ParameterAnalyzer = ParameterAnalyzer
V1WeightQuantizer = WeightQuantizer

# --------------------------------------
# Data & Model setup
# --------------------------------------


@dataclass
class StudyState:
    cfg: StudyConfig
    device: torch.device
    model: SOENModelCore
    cached_batches: list[tuple[torch.Tensor, torch.Tensor]]


def setup_study(cfg: StudyConfig) -> StudyState:
    cfg.validate()
    device = torch.device(cfg.device)
    model = SOENModelCore.load(cfg.model_path, device=device, show_logs=False)
    model.eval()
    if cfg.model_dt is not None:
        with contextlib.suppress(Exception):
            model.set_dt(float(cfg.model_dt))

    train_loader, val_loader, test_loader = create_data_loaders(
        data_path=cfg.data_path,
        batch_size=cfg.batch_size,
        val_split=0.2,
        test_split=0.1,
        num_workers=cfg.num_workers,
        cache_data=cfg.cache_data,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
        target_seq_len=cfg.target_seq_len,
        input_encoding=cfg.input_encoding,
        vocab_size=cfg.vocab_size,
        one_hot_dtype=cfg.one_hot_dtype,
    )

    if cfg.eval_split == "train":
        loader = train_loader
    elif cfg.eval_split == "test":
        loader = test_loader
    else:
        loader = val_loader

    cached = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if cfg.max_eval_batches and i >= cfg.max_eval_batches:
                break
            cached.append((x, y.long()))
    logging.info(f"Cached {len(cached)} evaluation batches.")
    return StudyState(cfg=cfg, device=device, model=model, cached_batches=cached)


# --------------------------------------
# Evaluation
# --------------------------------------


@torch.no_grad()
def evaluate_model(
    model: SOENModelCore, batches: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device, method: str, params: dict[str, Any]
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds_all, logits_all, confs_all = [], [], []
    correct, total = 0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        state_hist, _ = model(x)
        pooled = pool_time(state_hist, method, params)
        probs = torch.softmax(pooled, dim=1)
        preds = probs.argmax(1)
        confs = probs.max(1)[0]
        preds_all.append(preds.cpu())
        logits_all.append(pooled.cpu())
        confs_all.append(confs.cpu())
        correct += int((preds == y).sum().item())
        total += y.size(0)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return (
        acc,
        torch.cat(preds_all).numpy(),
        torch.cat(logits_all).numpy(),
        torch.cat(confs_all).numpy(),
    )


# --------------------------------------
# Readout perturbations (vectorized across K replicas)
# --------------------------------------


@torch.no_grad()
def eval_readout_quantization(
    model: SOENModelCore, batches: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device, method: str, params: dict[str, Any], step_size: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    correct, total = 0, 0
    preds_all, logits_all, confs_all = [], [], []
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        state_hist, _ = model(x)
        pooled = pool_time(state_hist, method, params)
        pert = torch.round(pooled / step_size) * step_size
        probs = torch.softmax(pert, dim=1)
        preds = probs.argmax(1)
        confs = probs.max(1)[0]
        preds_all.append(preds.cpu())
        logits_all.append(pert.cpu())
        confs_all.append(confs.cpu())
        correct += int((preds == y).sum().item())
        total += y.size(0)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, torch.cat(preds_all).numpy(), torch.cat(logits_all).numpy(), torch.cat(confs_all).numpy()


@torch.no_grad()
def eval_readout_noise(
    model: SOENModelCore, batches: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device, method: str, params: dict[str, Any], sigma: float, repeats: int
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    results: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(repeats):
        correct, total = 0, 0
        preds_all, logits_all, confs_all = [], [], []
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            state_hist, _ = model(x)
            pooled = pool_time(state_hist, method, params)
            noise = torch.randn_like(pooled) * sigma
            pert = pooled + noise
            probs = torch.softmax(pert, dim=1)
            preds = probs.argmax(1)
            confs = probs.max(1)[0]
            preds_all.append(preds.cpu())
            logits_all.append(pert.cpu())
            confs_all.append(confs.cpu())
            correct += int((preds == y).sum().item())
            total += y.size(0)
        acc = 100.0 * correct / total if total > 0 else 0.0
        results.append((acc, torch.cat(preds_all).numpy(), torch.cat(logits_all).numpy(), torch.cat(confs_all).numpy()))
    return results


# --------------------------------------
# Main entry
# --------------------------------------


# Results structure matching v1 for compatibility with analyse.py
@dataclass
class RobustnessResults:
    baseline_accuracy: float
    baseline_predictions: np.ndarray
    baseline_readouts: np.ndarray
    baseline_confidences: np.ndarray
    perturbation_results: dict[str, dict[str, Any]]
    config: dict
    timestamp: str
    num_samples_evaluated: int


# Analyzers/quantizer are provided locally to avoid circular imports


def _run_parameter_experiment(state: StudyState, exp_def: dict[str, Any], method: str, params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Run parameter perturbations with local Monte Carlo (fast path)."""
    exp_def.get("name", "param_exp")
    param_name = exp_def["parameter"]
    distribution = exp_def["distribution"]
    target_layers = exp_def.get("target_layers", [1])
    levels: list[float] = exp_def["levels"]
    hard_limits = bool(exp_def.get("hard_limits", False))

    analyzer = V1ParameterAnalyzer(state.model, target_layers)
    param_info = analyzer.discover_parameter_info(param_name)

    # Build a light config dict compatible with v1 PerturbationConfig fields used by apply_perturbation_inplace
    p_config = type(
        "_MiniPerturb",
        (),
        {
            "param_name": param_name,
            "distribution": distribution,
            "levels": levels,
            "hard_limits": hard_limits,
            "nominal_value": param_info["nominal_value"],
            "is_log_param": param_info["is_log_param"],
        },
    )()

    results_by_level: dict[str, dict[str, Any]] = {}

    for level in levels:
        raw_runs: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(state.cfg.monte_carlo_runs):
            # Restore, perturb, evaluate
            analyzer.restore_original_params()
            analyzer.apply_perturbation_inplace(p_config, float(level))
            res = evaluate_model(state.model, state.cached_batches, state.device, method, params)
            raw_runs.append(res)
        # Restore after level
        analyzer.restore_original_params()
        results_by_level[str(level)] = {"raw_results": raw_runs}

    return results_by_level


def _calc_levels_from_bits_or_levels(bits: int | None, levels: int | None) -> int:
    if bits is not None and levels is not None:
        msg = "Specify either 'bits' or 'levels', not both"
        raise ValueError(msg)
    if bits is None and levels is None:
        msg = "Must specify either 'bits' or 'levels'"
        raise ValueError(msg)
    if bits is not None:
        if bits < 0:
            msg = "bits must be non-negative"
            raise ValueError(msg)
        return (2 ** int(bits)) + 1
    assert levels is not None  # guaranteed by check above
    return int(levels)


def _run_weight_quantization_experiment(state: StudyState, exp_def: dict[str, Any], method: str, params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    connection_spec = exp_def.get("connection")
    if isinstance(connection_spec, str):
        conn_names = [connection_spec]
    elif isinstance(connection_spec, list):
        conn_names = connection_spec
    else:
        msg = "connection must be a string or list of strings"
        raise ValueError(msg)

    min_value = float(exp_def["min_value"])  # required
    max_value = float(exp_def["max_value"])  # required
    bits = exp_def.get("bits")
    levels = exp_def.get("levels")
    stochastic = bool(exp_def.get("stochastic", False))
    temperature = float(exp_def.get("temperature", 1.0))

    # Choose quantization values list for reporting
    if isinstance(bits, list):
        q_values = bits
        q_type = "bits"
    elif isinstance(levels, list):
        q_values = levels
        q_type = "levels"
    else:
        q_values = [bits] if bits is not None else [levels]
        q_type = "bits" if bits is not None else "levels"

    results_by_value: dict[str, dict[str, Any]] = {}
    quantizer = V1WeightQuantizer(state.model)

    for value in q_values:
        num_levels = _calc_levels_from_bits_or_levels(value if q_type == "bits" else None, value if q_type == "levels" else None)
        raw_runs: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        try:
            if stochastic:
                for _ in range(state.cfg.monte_carlo_runs):
                    # Restore before each run
                    quantizer.restore_original_weights()
                    quantizer.quantize_connection_weights_stochastic(conn_names, min_value, max_value, num_levels, temperature)
                    res = evaluate_model(state.model, state.cached_batches, state.device, method, params)
                    raw_runs.append(res)
                # Restore after block
                quantizer.restore_original_weights()
            else:
                quantizer.restore_original_weights()
                quantizer.quantize_connection_weights(conn_names, min_value, max_value, num_levels)
                res = evaluate_model(state.model, state.cached_batches, state.device, method, params)
                raw_runs.append(res)
                quantizer.restore_original_weights()
        except Exception as e:
            logging.exception(f"Weight quantization failed for {value} ({q_type}) on {conn_names}: {e}")
            quantizer.restore_original_weights()
            continue
        results_by_value[str(value)] = {"raw_results": raw_runs}

    return results_by_value


def run(cfg: StudyConfig) -> None:
    setup_logging(cfg.log_level)
    state = setup_study(cfg)
    method = cfg.time_pooling
    params = cfg.time_pooling_params or {}

    # Baseline
    base = evaluate_model(state.model, state.cached_batches, state.device, method, params)
    logging.info(f"Baseline accuracy: {base[0]:.2f}%")

    # Experiments
    all_experiment_results: dict[str, dict[str, Any]] = {}
    for exp in cfg.experiments:
        name = exp.get("name", "experiment")
        typ = exp.get("type", "readout")
        logging.info(f"\n--- Running {name} ({typ}) ---")
        if typ == "readout":
            perturb = exp.get("perturbation")
            if perturb == "quantization":
                res_ua = exp["resolution_levels_ua"]
                scale = float(exp["scaling_factor_ua_per_unit"]) * 1e-6
                readout_results: dict[str, dict[str, Any]] = {}
                for r in res_ua:
                    step = (float(r) * 1e-6) / scale
                    res = eval_readout_quantization(state.model, state.cached_batches, state.device, method, params, step)
                    logging.info(f"  {r} µA → {res[0]:.2f}%")
                    readout_results[str(r)] = {"raw_results": [res]}
                all_experiment_results[name] = readout_results
            elif perturb == "noise":
                fractions = exp["noise_fractions"]
                # Estimate RMS on pooled outputs
                with torch.no_grad():
                    outs_list: list[torch.Tensor] = []
                    for x, _ in state.cached_batches:
                        x = x.to(state.device)
                        hist, _ = state.model(x)
                        pooled = pool_time(hist, method, params)
                        outs_list.append(pooled)
                    outs_tensor = torch.cat(outs_list, dim=0)
                    rms = torch.sqrt(torch.mean(outs_tensor**2)).item()
                repeats = int(exp.get("repeats", cfg.monte_carlo_runs))
                noise_results: dict[str, dict[str, Any]] = {}
                for f in fractions:
                    sigma = float(f) * rms
                    res_list = eval_readout_noise(state.model, state.cached_batches, state.device, method, params, sigma, repeats)
                    logging.info(f"  {f}×RMS → {float(np.mean([r[0] for r in res_list])):.2f}% (n={repeats})")
                    noise_results[str(f)] = {"raw_results": res_list}
                all_experiment_results[name] = noise_results
        elif typ == "parameter":
            all_experiment_results[name] = _run_parameter_experiment(state, exp, method, params)
        elif typ == "weight_quantization":
            all_experiment_results[name] = _run_weight_quantization_experiment(state, exp, method, params)
        else:
            logging.warning(f"  Unknown experiment type '{typ}', skipping")

    # Package results compatible with v1 analyse.py
    results = RobustnessResults(
        baseline_accuracy=base[0],
        baseline_predictions=base[1],
        baseline_readouts=base[2],
        baseline_confidences=base[3],
        perturbation_results=all_experiment_results,
        config=asdict(cfg),
        timestamp=datetime.now().isoformat(),
        num_samples_evaluated=len(base[1]),
    )

    # Save results to timestamped directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path(cfg.results_dir) / f"study_v2_{timestamp_str}"
    study_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = study_dir / "robustness_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    # Save YAML config
    with open(study_dir / "study_config.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, indent=2)
    logging.info(f"Saved results to {pkl_path}")

    # Invoke analysis for identical plots (search common locations incl. old/)
    try:
        here = Path(__file__).resolve()
        candidates = [
            here.parent / "analyse.py",  # same package dir
            here.parent.parent / "robustness_tool" / "analyse.py",  # sibling robustness_tool
            here.parents[3] / "old" / "robustness_tool" / "analyse.py",  # repo-root/old fallback
            Path.cwd() / "old" / "robustness_tool" / "analyse.py",  # CWD-based fallback
        ]
        analysis_script = next((p for p in candidates if p.exists()), None)
        if analysis_script is None:
            logging.error("✗ Analysis script not found in expected locations; skipping analysis step.")
            logging.error("Checked: %s", ", ".join(str(p) for p in candidates))
        else:
            cmd = [
                sys.executable,
                str(analysis_script),
                str(pkl_path),
                "--plot-format",
                cfg.plot_format,
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=analysis_script.parent)
            if result.returncode == 0:
                logging.info("✓ Analysis completed successfully")
            else:
                logging.error("✗ Analysis failed")
                logging.error(result.stdout)
                logging.error(result.stderr)
    except Exception as e:
        logging.exception(f"Failed to run analysis: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast Robustness Tool v2 (readout-focused)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    cfg = StudyConfig(**raw)
    run(cfg)


if __name__ == "__main__":
    main()
