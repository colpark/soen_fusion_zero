# FILEPATH: src/soen_toolkit/core/mixins/io.py

from __future__ import annotations

import contextlib
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, NoReturn, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterator
    from soen_toolkit.core.configs import ConnectionConfig, LayerConfig, SimulationConfig


class IOMixin:
    """Mixin providing I/O operations for SOEN models."""

    if TYPE_CHECKING:
        # Attributes expected from the composed class
        sim_config: SimulationConfig
        layers_config: list[LayerConfig]
        connections_config: list[ConnectionConfig]
        connections: nn.ParameterDict
        connection_masks: dict[str, torch.Tensor]
        dt: float | nn.Parameter
        _load_missing_keys: list[str]
        _load_unexpected_keys: list[str]
        _load_filtered_keys: list[str]

        def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...
        def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
        def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any: ...
        def to(self, *args: Any, **kwargs: Any) -> Any: ...
    def save(self, file_path: str, include_metadata: bool = True) -> None:
        from pathlib import Path

        path_str = str(file_path)
        suffixes = [s.lower() for s in Path(path_str).suffixes]

        if ".json" in suffixes:
            from soen_toolkit.utils.model_tools import export_model_to_json

            # This mixin is always part of SOENModelCore in practice
            from soen_toolkit.core.soen_model_core import SOENModelCore
            s = export_model_to_json(cast(SOENModelCore, self), filename=None)
            with open(path_str, "w") as f:
                f.write(s)
            return None

        if (".soen" in suffixes) or (".pth" in suffixes):
            return self._save_binary(path_str, include_metadata=include_metadata)
        msg = f"Unsupported file extension for '{file_path}'. Supported: '.json' (JSON), '.soen' or '.pth' (binary)."
        raise ValueError(
            msg,
        )

    @classmethod
    def load(
        cls,
        file_path: str,
        device: torch.device | None = None,
        strict: bool = True,
        verbose: bool = True,
        show_logs: bool = False,
    ):
        path_l = str(file_path).lower()
        if path_l.endswith(".json"):
            from soen_toolkit.utils.model_tools import model_from_json

            model = model_from_json(str(file_path))
            return model.to(device or torch.device("cpu"))
        if path_l.endswith((".soen", ".pth")):
            return cls._load_binary(
                str(file_path),
                device=device,
                strict=strict,
                verbose=verbose,
                show_logs=show_logs,
            )
        if path_l.endswith(".pkl"):
            # Auto-detect JAX checkpoint and convert to PyTorch
            return cls._load_from_jax_checkpoint(
                str(file_path),
                device=device,
                verbose=verbose,
                show_logs=show_logs,
            )
        msg = f"Unsupported file extension for '{file_path}'. Supported: '.json' (JSON), '.soen'/'.pth' (binary), '.pkl' (JAX)."
        raise ValueError(
            msg,
        )

    def _save_binary(self, file_path: str, include_metadata: bool = True) -> None:
        from datetime import datetime
        import logging

        logger = logging.getLogger(__name__)

        sim_config_dict = asdict(self.sim_config)
        layers_config_dicts = [asdict(cfg) for cfg in self.layers_config]
        connections_config_dicts = [asdict(cfg) for cfg in self.connections_config]

        # Serialize masks explicitly (not part of state_dict)
        conn_masks_serialized = {}
        try:
            for key, mask in getattr(self, "connection_masks", {}).items():
                with contextlib.suppress(Exception):
                    conn_masks_serialized[key] = mask.detach().cpu()
        except Exception:
            conn_masks_serialized = {}

        save_data = {
            "state_dict": cast(Any, self).state_dict(),
            "sim_config": sim_config_dict,
            "layers_config": layers_config_dicts,
            "connections_config": connections_config_dicts,
            "connection_masks": conn_masks_serialized,
            "dt": float(self.dt.detach().item()) if isinstance(self.dt, torch.nn.Parameter) else self.dt,
            "dt_learnable": isinstance(self.dt, torch.nn.Parameter) and self.dt.requires_grad,
            "model_type": "SOENModelCore",
        }

        if include_metadata:
            metadata = {
                "save_timestamp": datetime.now().isoformat(),
                "model_hash": hash(str(cast(Any, self).state_dict())),
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "layer_types": [cfg.get("layer_type", "Unknown") for cfg in layers_config_dicts],
                "layer_dimensions": [cfg.get("params", {}).get("dim", 0) for cfg in layers_config_dicts],
            }
            save_data["metadata"] = metadata
            logger.debug("Saving model with metadata: %s", metadata)

        try:
            torch.save(save_data, file_path)
            logger.debug("Model successfully saved to %s", file_path)
        except Exception as e:
            logger.exception("Failed to save model to %s: %s", file_path, str(e))
            msg = f"Failed to save model to {file_path}: {e!s}"
            raise RuntimeError(msg)

    @classmethod
    def _load_from_jax_checkpoint(
        cls,
        file_path: str,
        device: torch.device | None = None,
        verbose: bool = True,
        show_logs: bool = False,
    ):
        """Load model from JAX .pkl checkpoint (auto-converts to PyTorch).

        Args:
            file_path: Path to JAX .pkl checkpoint
            device: Target device
            verbose: Print loading info
            show_logs: Show detailed logs

        Returns:
            SOENModelCore instance
        """
        import logging

        logger = logging.getLogger(__name__)

        if verbose:
            logger.info(f"Loading JAX checkpoint: {file_path}")

        try:
            from soen_toolkit.utils.port_to_jax.convert import convert_jax_to_core_model
            from soen_toolkit.utils.port_to_jax.jax_training.callbacks.checkpointing import (
                apply_params_to_jax_model,
                load_checkpoint,
                reconstruct_jax_model,
            )
        except ImportError as e:
            msg = "JAX checkpoint loading requires JAX dependencies. Install with: pip install jax jaxlib"
            raise RuntimeError(msg) from e

        # Load JAX checkpoint
        try:
            checkpoint = load_checkpoint(file_path)
            jax_model = reconstruct_jax_model(checkpoint)
            apply_params_to_jax_model(jax_model, checkpoint.get("params"))
        except Exception as e:
            msg = f"Failed to load JAX checkpoint from {file_path}: {e}"
            raise RuntimeError(msg) from e

        # Convert to PyTorch
        try:
            torch_model = convert_jax_to_core_model(jax_model)
        except Exception as e:
            msg = f"Failed to convert JAX model to PyTorch: {e}"
            raise RuntimeError(msg) from e

        if verbose:
            logger.info(f"Converted JAX checkpoint to PyTorch model with {len(torch_model.layers_config)} layers")

        # Move to device
        if device is None:
            device = torch.device("cpu")
        torch_model = torch_model.to(device)

        return torch_model

    @classmethod
    def _load_binary(
        cls,
        file_path: str,
        device: torch.device | None = None,
        strict: bool = True,
        verbose: bool = True,
        show_logs: bool = False,
    ):
        import logging

        logger = logging.getLogger(__name__)

        if device is None:
            device = torch.device("cpu")

        try:
            import warnings as _warnings

            from soen_toolkit.core import (
                ConnectionConfig,
                LayerConfig,
                NoiseConfig,
                SimulationConfig,
            )

            def _load_with_warnings_suppressed():
                if not show_logs:
                    with _warnings.catch_warnings():
                        _warnings.filterwarnings(
                            "ignore",
                            message=r".*one_to_one connectivity with mismatched dimensions.*",
                            category=UserWarning,
                            module=r"soen_toolkit\.layers\.connectivity",
                        )
                        _warnings.filterwarnings(
                            "ignore",
                            category=UserWarning,
                            module=r"soen_toolkit\.layers\.connectivity",
                        )
                        return torch.load(file_path, map_location=device)
                else:
                    return torch.load(file_path, map_location=device)

            # Newer PyTorch (>=2.1): prefer safe_load with trusted modules
            if hasattr(torch.serialization, "safe_load"):
                try:
                    with torch.serialization.safe_load(trusted_modules=["soen_toolkit"]):
                        save_data = _load_with_warnings_suppressed()
                except Exception as _e1:
                    # Fallback to best-effort unsafe load paths below
                    save_data = None
            # Older PyTorch that provides safe_globals context manager
            elif hasattr(torch.serialization, "safe_globals"):
                try:
                    with torch.serialization.safe_globals(
                        [SimulationConfig, LayerConfig, ConnectionConfig, NoiseConfig],
                    ):
                        save_data = _load_with_warnings_suppressed()
                except Exception:
                    save_data = None
            else:
                save_data = None

            # If both guarded approaches failed or are unavailable, try standard load
            if save_data is None:
                try:
                    save_data = _load_with_warnings_suppressed()
                except Exception as _e2:
                    # Last resort for PyTorch >=2.0: weights_only to extract tensors
                    try:
                        save_data = torch.load(file_path, map_location=device, weights_only=True)
                    except Exception as _e3:
                        # Ultimate fallback (unsafe): disable pickle checks; only for trusted files
                        try:
                            save_data = torch.load(file_path, map_location=device, pickle_module=None)
                        except Exception as e:
                            logger.exception("Failed to load model from %s: %s", file_path, str(e))
                            msg = f"Failed to load model from {file_path}: {e!s}"
                            raise RuntimeError(msg)

            if show_logs:
                logger.info("File %s loaded successfully.", file_path)
        except Exception as e:
            logger.exception("Failed to load model from %s: %s", file_path, str(e))
            msg = f"Failed to load model from {file_path}: {e!s}"
            raise RuntimeError(msg)

        from soen_toolkit.core import (
            ConnectionConfig,
            LayerConfig,
            NoiseConfig,
            PerturbationConfig,
            SimulationConfig,
        )

        try:
            raw_sim_config = save_data.get("sim_config", {})
            if "noise_train" in raw_sim_config or "noise_eval" in raw_sim_config:
                if show_logs:
                    logger.warning(
                        "Legacy 'noise_train'/'noise_eval' found in saved sim_config. Ignoring them.",
                    )
                raw_sim_config.pop("noise_train", None)
                raw_sim_config.pop("noise_eval", None)
            import inspect as _inspect

            sim_config_fields = set(_inspect.signature(SimulationConfig.__init__).parameters.keys())
            sim_config_fields.discard("self")
            filtered_config = {}
            ignored_fields = []
            for key, value in raw_sim_config.items():
                if key in sim_config_fields:
                    filtered_config[key] = value
                else:
                    ignored_fields.append(key)
            if ignored_fields and show_logs:
                logger.warning(
                    f"Ignoring unrecognized SimulationConfig fields from saved model: {ignored_fields}",
                )
            sim_config = SimulationConfig(**filtered_config)
        except Exception as e:
            msg = f"Failed to reconstruct SimulationConfig: {e!s}"
            raise RuntimeError(msg)

        raw_layers_config = save_data.get("layers_config", [])
        if not raw_layers_config:
            msg = "No layers_config found in saved data!"
            raise ValueError(msg)
        try:
            layers_config = []
            for cfg_dict in raw_layers_config:
                noise_dict = cfg_dict.pop("noise", None)
                noise_config_obj = None
                if noise_dict is not None and isinstance(noise_dict, dict):
                    try:
                        noise_config_obj = NoiseConfig(**noise_dict)
                    except Exception:
                        noise_config_obj = NoiseConfig()
                else:
                    noise_config_obj = NoiseConfig()

                if "params" not in cfg_dict:
                    cfg_dict["params"] = {}

                required_layer_keys = {"layer_id", "layer_type", "params"}
                missing_keys = required_layer_keys - set(cfg_dict.keys())
                if missing_keys:
                    msg = f"Missing required keys in layer config dict: {missing_keys}"
                    raise ValueError(
                        msg,
                    )

                import inspect as _inspect

                layer_config_fields = set(_inspect.signature(LayerConfig.__init__).parameters.keys())
                layer_config_fields.discard("self")

                filtered_layer_config = {}
                for key, value in cfg_dict.items():
                    if key in layer_config_fields:
                        filtered_layer_config[key] = value

                filtered_layer_config.setdefault("model_id", 0)
                layer_cfg = LayerConfig(noise=noise_config_obj, **filtered_layer_config)
                layers_config.append(layer_cfg)
        except Exception as e:
            msg = f"Failed to reconstruct layers_config: {e!s}"
            raise RuntimeError(msg)

        raw_connections_config = save_data.get("connections_config", [])
        try:
            connections_config = []
            for cfg_dict in raw_connections_config:
                required_conn_keys = {"from_layer", "to_layer", "connection_type"}
                missing_keys = required_conn_keys - set(cfg_dict.keys())
                if missing_keys:
                    msg = f"Missing required keys in connection config dict: {missing_keys}"
                    raise ValueError(
                        msg,
                    )

                noise_dict = cfg_dict.pop("noise", None)
                noise_cfg = NoiseConfig(**(noise_dict or {})) if isinstance(noise_dict, dict) else NoiseConfig()

                perturb_dict = cfg_dict.pop("perturb", None)
                perturb_cfg = PerturbationConfig(**(perturb_dict or {})) if isinstance(perturb_dict, dict) else PerturbationConfig()

                connection = ConnectionConfig(
                    from_layer=cfg_dict["from_layer"],
                    to_layer=cfg_dict["to_layer"],
                    connection_type=cfg_dict["connection_type"],
                    params=cfg_dict.get("params"),
                    learnable=cfg_dict.get("learnable", True),
                    noise=noise_cfg,
                    perturb=perturb_cfg,
                )
                connections_config.append(connection)
        except Exception as e:
            msg = f"Failed to reconstruct connections_config: {e!s}"
            raise RuntimeError(msg)

        required_keys_for_build = ["sim_config", "layers_config", "connections_config", "state_dict"]
        is_soen_format = all(key in save_data for key in required_keys_for_build)

        if is_soen_format:
            try:
                import warnings as _warnings

                if not show_logs:
                    with _warnings.catch_warnings():
                        _warnings.filterwarnings("ignore", category=Warning)
                        model = cls(  # type: ignore[call-arg]
                            sim_config=sim_config,
                            layers_config=layers_config,
                            connections_config=connections_config,
                        )
                else:
                    model = cls(  # type: ignore[call-arg]
                        sim_config=sim_config,
                        layers_config=layers_config,
                        connections_config=connections_config,
                    )

                def _coerce_legacy_state_dict(target_model, state_dict) -> None:
                    layer_index_map = {cfg.layer_id: idx for idx, cfg in enumerate(target_model.layers_config)}
                    for layer_id, layer_idx in layer_index_map.items():
                        dt_key = f"layers.{layer_idx}._dt"
                        layer_module = target_model.layers[layer_idx]
                        legacy_internal_key = f"layers.{layer_idx}.internal_J"
                        connection_key = f"J_{layer_id}_to_{layer_id}"
                        conn_state_key = f"connections.{connection_key}"
                        # Legacy saved key for internal self-connection parameter
                        legacy_conn_state_key = f"connections.internal_{layer_id}"

                        if hasattr(layer_module, "_dt") and dt_key not in state_dict:
                            state_dict[dt_key] = layer_module._dt.detach().clone()

                        if hasattr(layer_module, "connectivity") and getattr(layer_module, "connectivity", None) is not None:
                            connectivity_weight_key = f"layers.{layer_idx}.connectivity.weight"
                            internal_module_weight_key = f"layers.{layer_idx}.internal_connectivity.weight"

                            internal_tensor = None
                            if conn_state_key in state_dict:
                                internal_tensor = state_dict[conn_state_key]
                            elif legacy_internal_key in state_dict:
                                internal_tensor = state_dict[legacy_internal_key]
                            elif legacy_conn_state_key in state_dict:
                                internal_tensor = state_dict[legacy_conn_state_key]

                            if internal_tensor is not None:
                                if connectivity_weight_key not in state_dict:
                                    state_dict[connectivity_weight_key] = internal_tensor
                                if internal_module_weight_key not in state_dict:
                                    state_dict[internal_module_weight_key] = internal_tensor
                                if conn_state_key not in state_dict:
                                    state_dict[conn_state_key] = internal_tensor

                        elif legacy_internal_key not in state_dict and conn_state_key in state_dict:
                            state_dict[legacy_internal_key] = state_dict[conn_state_key]

                        state_dict.pop(legacy_internal_key, None)
                        # Map and remove legacy ParameterDict key if present
                        if legacy_conn_state_key in state_dict:
                            if conn_state_key not in state_dict:
                                state_dict[conn_state_key] = state_dict[legacy_conn_state_key]
                            state_dict.pop(legacy_conn_state_key, None)

                def _normalize_state_dict(raw_state_dict):
                    # Create a copy so that legacy adjustments do not mutate the persisted object
                    state_dict_local = dict(raw_state_dict)

                    # 1) Map legacy connection ParameterDict keys: connections.internal_<id>
                    #    -> connections.J_<id>_to_<id>
                    try:
                        import re as _re

                        for key in list(state_dict_local.keys()):
                            m = _re.match(r"^connections\.internal_(\d+)$", key)
                            if m:
                                lid = m.group(1)
                                new_key = f"connections.J_{lid}_to_{lid}"
                                if new_key not in state_dict_local:
                                    state_dict_local[new_key] = state_dict_local[key]
                                # remove legacy key to avoid unexpected_keys in strict mode
                                state_dict_local.pop(key, None)
                    except Exception:
                        pass

                    # 2) Upgrade legacy attribute paths (e.g., source_function -> _source_function)
                    legacy_token_map = {
                        "source_function": "_source_function",
                        "dynamics": "_dynamics",
                        "solver": "_solver",
                    }
                    keys_to_delete: list[str] = []
                    for key, value in list(state_dict_local.items()):
                        if not key.startswith("layers."):
                            continue
                        parts = key.split(".")
                        replaced = False
                        converted_parts = []
                        for part in parts:
                            if part in legacy_token_map:
                                converted_parts.append(legacy_token_map[part])
                                replaced = True
                            else:
                                converted_parts.append(part)
                        if not replaced:
                            continue
                        new_key = ".".join(converted_parts)
                        if new_key not in state_dict_local:
                            state_dict_local[new_key] = value
                        keys_to_delete.append(key)
                    for legacy_key in keys_to_delete:
                        state_dict_local.pop(legacy_key, None)

                    # 3) If a legacy direct source_function tensor exists, mirror it into the
                    # modern dynamics/solver submodules to prevent missing keys when loading
                    for key, value in list(state_dict_local.items()):
                        if not key.startswith("layers."):
                            continue
                        if "_source_function" not in key:
                            continue
                        if not key.endswith("_source_function.g_table"):
                            continue
                        base_prefix = key.rsplit("._source_function.g_table", 1)[0]
                        # Avoid re-nesting if this key already belongs under _dynamics or _solver
                        if ".__dynamics" in base_prefix or "._dynamics" in base_prefix or ".__solver" in base_prefix or "._solver" in base_prefix:
                            continue
                        dyn_key = f"{base_prefix}._dynamics._source_function.g_table"
                        solver_key = f"{base_prefix}._solver._dynamics._source_function.g_table"
                        if dyn_key not in state_dict_local:
                            state_dict_local[dyn_key] = value
                        if solver_key not in state_dict_local:
                            state_dict_local[solver_key] = value
                        # Keep the original key too, in case current architecture expects it

                    # 4) Final coercions that require an instantiated model
                    _coerce_legacy_state_dict(model, state_dict_local)
                    return state_dict_local

                try:
                    if not show_logs:
                        with _warnings.catch_warnings():
                            _warnings.filterwarnings("ignore", category=Warning)
                            state_dict = _normalize_state_dict(save_data["state_dict"])
                            _load_ret = model.load_state_dict(state_dict, strict=strict)
                    else:
                        state_dict = _normalize_state_dict(save_data["state_dict"])
                        _load_ret = model.load_state_dict(state_dict, strict=strict)
                    # If tolerant load, attach a diff report for callers/GUI
                    try:
                        if strict is False and _load_ret is not None:
                            miss = list(getattr(_load_ret, "missing_keys", []) or [])
                            unexp = list(getattr(_load_ret, "unexpected_keys", []) or [])
                            # Store on the model for later inspection
                            model._load_missing_keys = miss
                            model._load_unexpected_keys = unexp
                            if show_logs and (miss or unexp):
                                logger.warning(
                                    "Tolerant load: %d missing and %d unexpected keys.",
                                    len(miss),
                                    len(unexp),
                                )
                    except Exception:
                        pass
                except RuntimeError as e:
                    # Check if failure is due to source function buffer keys (g_table)
                    # This can happen when solver structure changes (e.g. FE -> ParaRNN)
                    error_msg = str(e)
                    if "g_table" in error_msg and "Unexpected key" in error_msg:
                        # Filter out g_table keys and retry with non-strict loading
                        logger.warning(
                            "Detected source function buffer keys mismatch (g_table). "
                            "This can happen when the solver type differs between save and load. "
                            "Retrying with tolerant loading..."
                        )
                        # Filter out problematic keys
                        filtered_state_dict = {
                            k: v for k, v in state_dict.items()
                            if not k.endswith("._source_function.g_table")
                        }
                        try:
                            _load_ret = model.load_state_dict(filtered_state_dict, strict=False)
                            # Store info about filtered keys
                            model._load_filtered_keys = [
                                k for k in state_dict.keys()
                                if k.endswith("._source_function.g_table")
                            ]
                            if show_logs and model._load_filtered_keys:
                                logger.info(
                                    "Filtered %d source function buffer keys during load.",
                                    len(model._load_filtered_keys),
                                )
                        except Exception as retry_e:
                            msg = f"Failed to load state dict even after filtering g_table keys: {retry_e!s}"
                            raise RuntimeError(msg) from retry_e
                    else:
                        msg = f"Failed to load state dict: {e!s}"
                        raise RuntimeError(msg) from e

                for conn_cfg in connections_config:
                    if conn_cfg.from_layer == conn_cfg.to_layer:
                        key = f"internal_{conn_cfg.from_layer}"
                    else:
                        key = f"J_{conn_cfg.from_layer}_to_{conn_cfg.to_layer}"
                    if key in model.connections:
                        model.connections[key].requires_grad_(conn_cfg.learnable)

                if "dt" in save_data:
                    loaded_dt = save_data["dt"]
                    dt_learn = save_data.get("dt_learnable", False)
                    if dt_learn:
                        model.dt = nn.Parameter(torch.tensor(float(loaded_dt), dtype=torch.float32), requires_grad=True)
                    else:
                        model.dt = float(loaded_dt)
                        model.sim_config.dt = float(loaded_dt)
                        model.sim_config.dt_learnable = dt_learn

                # Restore connection masks if present
                try:
                    cm = save_data.get("connection_masks", {}) or {}
                    if isinstance(cm, dict):
                        with torch.no_grad():
                            for key, tensor in cm.items():
                                try:
                                    t = tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor)
                                    if key in model.connections:
                                        p = model.connections[key]
                                        if tuple(t.shape) == tuple(p.shape):
                                            model.connection_masks[key] = t.to(device=p.device, dtype=p.dtype)
                                except Exception:
                                    continue
                except Exception:
                    pass

            except Exception as e:
                import traceback

                logger.exception(traceback.format_exc())
                msg = f"Failed to reconstruct model: {e!s}"
                raise RuntimeError(msg)
        else:
            first_val = next(iter(save_data.values()), None)
            if isinstance(first_val, torch.Tensor):
                if strict:
                    msg = "Strict loading enabled and file seems to contain only state_dict. Use strict=False to load with default configurations, or provide a full save file."
                    raise ValueError(
                        msg,
                    )
                model = cls()
                model.load_state_dict(save_data, strict=strict)
            else:
                msg = "Unable to determine the format of the saved model file. Expected a dictionary matching save() binary format or a raw state_dict."
                raise ValueError(
                    msg,
                )

        model = model.to(device)
        return model

    def save_soen(self, file_path: str, include_metadata: bool = True) -> None:
        msg = "save_soen has been removed. Use model.save('<path>.soen' or '<path>.pth') instead."
        raise RuntimeError(
            msg,
        )

    @classmethod
    def load_soen(
        cls,
        file_path: str,
        device: torch.device | None = None,
        strict: bool = True,
        verbose: bool = True,
        show_logs: bool = False,
    ) -> NoReturn:
        msg = "load_soen has been removed. Use SOENModelCore.load('<path>.soen' or '<path>.pth', device=..., strict=..., verbose=..., show_logs=...) instead."
        raise RuntimeError(
            msg,
        )
