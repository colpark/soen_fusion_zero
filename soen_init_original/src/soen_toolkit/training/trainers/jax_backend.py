from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import jax
    import jax.numpy as jnp
    import optax
except Exception:  # pragma: no cover
    jax = None
    jnp = None
    optax = None


from soen_toolkit.utils.port_to_jax.jax_training.callbacks.loss_weight_scheduler import (
    LossWeightSchedulerJAX,
)
from soen_toolkit.utils.port_to_jax.jax_training.callbacks.metrics import METRICS_REGISTRY
from soen_toolkit.utils.port_to_jax.jax_training.callbacks.qat import (
    QATStraightThroughJAX,
)
from soen_toolkit.utils.port_to_jax.jax_training.callbacks.schedulers import (
    make_schedule_from_callbacks_config as _make_jax_schedule,
)
from soen_toolkit.utils.port_to_jax.jax_training.callbacks.seq_len_scheduler import (
    TargetSeqLenSchedulerJAX,
)
from soen_toolkit.utils.port_to_jax.jax_training.callbacks.time_pooling_scale_scheduler import (
    TimePoolingScaleSchedulerJAX,
)
from soen_toolkit.utils.port_to_jax.jax_training.data_jax import (
    JAXBatchIterator,
    load_hdf5_splits,
)
from soen_toolkit.utils.port_to_jax.jax_training.pooling import apply_time_pooling
from soen_toolkit.utils.port_to_jax.jax_training.trainer import (
    DataConfigJAX,
    ExperimentConfigJAX,
    JaxTrainer as _LowLevelJaxTrainer,
    TrainingConfigJAX,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from soen_toolkit.training.configs import ExperimentConfig
    from soen_toolkit.training.data import SOENDataModule


@dataclass
class _LoggerAdapter:
    """Minimal logger adapter that writes scalar metrics to TensorBoardLogger
    and optional MLflow logger instances that the Lightning path already sets up.
    Accepts floats
    callers must convert jnp to float before calling.
    """

    tb_logger: Any
    mlflow_logger: Any | None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        # TensorBoard
        for k, v in metrics.items():
            self.tb_logger.experiment.add_scalar(str(k), float(v), global_step=step or 0)

        # MLflow (optional)
        if self.mlflow_logger is not None:
            # Sanitize keys the same way SafeMLFlowLogger does (replace '/')
            safe = {str(k).replace("/", "_"): float(v) for k, v in metrics.items()}
            self.mlflow_logger.log_metrics(safe, step=step)


class JaxRunner:
    """High-level JAX training runner that plugs into the existing ExperimentRunner.

    - Uses the same `SOENDataModule` for dataset creation/splitting
    - Wraps the low-level JAX trainer for fully-jitted step/epoch compute
    - Logs to the same TensorBoard and optional MLflow loggers
    - Saves a simple JAX checkpoint (params + opt_state) in the repeat checkpoint directory

    Logging Config Support:
    - log_freq: [YES] Supported (controls step-level logging frequency)
    - log_batch_metrics: [YES] Supported (controls whether batch metrics are logged)
    - log_level: [YES] Supported (standard Python logging)
    - track_layer_params: [NO] Not yet implemented in JAX backend
    - track_connections: [NO] Not yet implemented in JAX backend
    - log_gradients: [NO] Not yet implemented in JAX backend
    """

    def __init__(self, config: ExperimentConfig, *, repeat_log_dir: Path, repeat_ckpt_dir: Path, loggers: list[Any]) -> None:
        self._log = logging.getLogger(__name__)

        # Suppress verbose JAX compilation debug logs unless explicitly enabled
        import os

        if os.environ.get("JAX_DEBUG", "").lower() not in {"1", "true", "yes"}:
            try:
                # Suppress JAX internal DEBUG logs (compilation cache, XLA compilation, etc.)
                import logging as std_logging

                # Suppress all JAX-related loggers
                for logger_name in ["jax", "jax._src.compiler", "jax._src.dispatch", "jax._src.cache_key", "jax._src.compilation_cache", "jax._src.interpreters.pxla", "jax._src.xla_bridge"]:
                    jax_logger = std_logging.getLogger(logger_name)
                    jax_logger.setLevel(std_logging.WARNING)
            except Exception:
                pass  # JAX not available or logging setup failed
        self.config = config
        self.repeat_log_dir = repeat_log_dir
        self.repeat_ckpt_dir = repeat_ckpt_dir
        self.loggers = loggers or []
        self.is_classification = False  # Track task type for metric logging
        self.profiler_results: dict[str, float] = {}
        self.do_profile = getattr(self.config.profiler, "active", False)

        self.losses = [
            {
                "loss_function": item.name,
                "weight": jnp.asarray(getattr(item, "weight", 0.5), dtype=jnp.float32),
                "params": (getattr(item, "params", item.get("params", {}) if isinstance(item, dict) else {}) or {}),
            }
            for item in (getattr(self.config.training.loss, "losses", []) or [])
        ]
        # Parse metrics from config (mirror PyTorch backend)
        self.metric_names: list[str] = []
        raw_metrics = getattr(getattr(config, "logging", object()), "metrics", [])
        if isinstance(raw_metrics, (list, tuple)):
            for item in raw_metrics:
                if isinstance(item, str):
                    self.metric_names.append(item)
                elif isinstance(item, dict) and "name" in item:
                    self.metric_names.append(item["name"])
        elif isinstance(raw_metrics, str):
            self.metric_names = [raw_metrics]

        # Checkpoint tracking (mirror Lightning's ModelCheckpoint)
        self.best_k_models: dict[Path, float] = {}  # {checkpoint_path: metric_value}
        self.save_top_k = getattr(getattr(config, "training", object()), "checkpoint_save_top_k", 3)
        self.save_last = getattr(getattr(config, "training", object()), "checkpoint_save_last", True)
        self.best_metric = float("inf")
        self.best_ckpt_path = ""
        # Background checkpoint saving (avoid races on exit/tests)
        self._checkpoint_thread = None
        self._checkpoint_thread_error: Exception | None = None

        # Pick first TB and optional MLflow logger from provided list
        self.tb_logger = next((lg for lg in self.loggers if hasattr(lg, "experiment") and hasattr(lg.experiment, "add_scalar")), None)
        self.ml_logger = next((lg for lg in self.loggers if hasattr(lg, "experiment") and hasattr(lg, "run_id")), None)

        if self.tb_logger is None:
            from torch.utils.tensorboard import SummaryWriter

            class _TBWrap:
                def __init__(self, writer) -> None:
                    self.experiment = writer

            self.tb_logger = _TBWrap(SummaryWriter(log_dir=str(self.repeat_log_dir)))

        self.logger_adapter = _LoggerAdapter(tb_logger=self.tb_logger, mlflow_logger=self.ml_logger)

        # Async artifact uploader for MLflow
        self.artifact_uploader = None
        if self.ml_logger is not None:
            try:
                from soen_toolkit.utils.io_utils import AsyncArtifactUploader

                # Try to get experiment and run_id (might be properties or attributes)
                exp = getattr(self.ml_logger, "experiment", None)
                run_id = getattr(self.ml_logger, "run_id", None)

                if exp and run_id:
                    self.artifact_uploader = AsyncArtifactUploader(exp, run_id)
                    self._log.info(f"[JAX] Initialized async MLflow artifact uploader for run {run_id}")
            except Exception as e:
                self._log.warning(f"[JAX] Failed to initialize async artifact uploader: {e}")

        # Stateful training configuration
        stateful_cfg = {}
        try:
            if hasattr(config, "callbacks") and isinstance(config.callbacks, dict):
                stateful_cfg = config.callbacks.get("stateful_training", {}) or {}
        except Exception:
            stateful_cfg = {}

        self.stateful_enable_train = bool(stateful_cfg.get("enable_for_training", False))
        self.stateful_enable_val = bool(stateful_cfg.get("enable_for_validation", False))
        self.stateful_sample_selection = str(stateful_cfg.get("sample_selection", "random"))

        if self.stateful_sample_selection not in {"random", "first", "last"}:
            self._log.warning(
                f"Invalid stateful training sample_selection '{self.stateful_sample_selection}', defaulting to 'random'"
            )
            self.stateful_sample_selection = "random"

        # State storage for stateful training
        self._train_states: dict[int, jnp.ndarray] | None = None
        self._train_s1_states: dict[int, jnp.ndarray] | None = None
        self._train_s2_states: dict[int, jnp.ndarray] | None = None
        self._val_states: dict[int, jnp.ndarray] | None = None
        self._val_s1_states: dict[int, jnp.ndarray] | None = None
        self._val_s2_states: dict[int, jnp.ndarray] | None = None

        if self.stateful_enable_train or self.stateful_enable_val:
            self._log.info(
                f"[JAX] Stateful training enabled: train={self.stateful_enable_train}, "
                f"val={self.stateful_enable_val}, sample_selection={self.stateful_sample_selection}"
            )

    @staticmethod
    def _params_to_dict(params: Any) -> dict:
        """Normalize training params to the legacy dict shape.

        Stage-2 Equinox training uses `SoenEqxModel` as the params pytree; many
        checkpointing and conversion utilities still expect:
        {"layer_params": ..., "connection_params": ..., "internal_connections": ...}.
        """
        if hasattr(params, "as_params_dict"):
            return params.as_params_dict()
        if isinstance(params, dict):
            return params
        raise TypeError(f"Unsupported params type: {type(params)}")

    def _pick_sample_index(self, batch_size: int) -> int:
        """Select which sample from batch to use for state carryover."""
        if self.stateful_sample_selection == "first":
            return 0
        elif self.stateful_sample_selection == "last":
            return batch_size - 1
        else:  # random
            import random
            return random.randint(0, batch_size - 1)

    def _extract_states_from_histories(
        self,
        trainer: _LowLevelJaxTrainer,
        histories: tuple,
        sample_idx: int,
        *,
        s1_final_by_layer: tuple[jnp.ndarray | None, ...] | None = None,
        s2_final_by_layer: tuple[jnp.ndarray | None, ...] | None = None,
    ) -> tuple[dict[int, jnp.ndarray], dict[int, jnp.ndarray], dict[int, jnp.ndarray]]:
        """Extract final states from histories for a specific sample.

        Returns:
            Tuple of (main_states, s1_states, s2_states) dicts mapping layer_id to states.
        """
        # Use the JAXModel's extract_final_states method (pure; NOCC aux must be provided explicitly)
        main_states, s1_states, s2_states = trainer.jax_model.extract_final_states(
            list(histories),
            s1_final_by_layer=s1_final_by_layer,
            s2_final_by_layer=s2_final_by_layer,
        )

        # Extract the specific sample from each state
        main_states_sample = {lid: state[sample_idx:sample_idx+1] for lid, state in main_states.items()}
        s1_states_sample = {lid: state[sample_idx:sample_idx+1] for lid, state in s1_states.items()}
        s2_states_sample = {lid: state[sample_idx:sample_idx+1] for lid, state in s2_states.items()}

        return main_states_sample, s1_states_sample, s2_states_sample

    def _compute_metrics(self, preds_jax: jnp.ndarray, targets_jax: jnp.ndarray, trainer: _LowLevelJaxTrainer | None = None) -> dict[str, float]:
        """Compute metrics using METRICS_REGISTRY (mirrors PyTorch backend).

        Args:
            preds_jax: Predictions, shape [batch, time, features] for sequences or [batch, features] for pooled
            targets_jax: Targets, typically shape [batch] for classification
            trainer: Optional trainer to access config for time pooling
        """
        metrics_dict = {}
        if not self.metric_names:
            return metrics_dict

        # Apply time pooling if needed for seq2static classification
        # This ensures predictions match the shape expected by metrics
        if trainer is not None and preds_jax.ndim == 3:
            mapping = getattr(trainer.cfg.training, "mapping", "seq2static")
            if mapping == "seq2static" and self.is_classification:
                # Apply same time pooling as used in loss computation
                method = getattr(trainer.cfg.training, "time_pooling_method", "max")
                params = getattr(trainer.cfg.training, "time_pooling_params", {})
                range_start = getattr(trainer.cfg.training, "time_pooling_range_start", None)
                range_end = getattr(trainer.cfg.training, "time_pooling_range_end", None)
                preds_jax = apply_time_pooling(preds_jax, method, params, range_start=range_start, range_end=range_end)

        # Targets for classification should be long integers (class indices)
        if self.is_classification:
            # Handle one-hot encoded targets by converting them to class indices
            if targets_jax.ndim > 1 and targets_jax.shape[-1] > 1:
                targets_jax = jnp.argmax(targets_jax, axis=-1)
            targets_jax = jnp.asarray(targets_jax, dtype=jnp.int32)
        else:
            targets_jax = jnp.asarray(targets_jax, dtype=jnp.float32)

        for metric_name in self.metric_names:
            if metric_name in {"loss", "input_seq_len"}:
                continue
            metric_fn = METRICS_REGISTRY.get(metric_name)
            metric_val = metric_fn(preds_jax, targets_jax)
            metrics_dict[metric_name] = float(metric_val.item())

        return metrics_dict

    def _validate_model_output_dimension(self, model_path: str, is_classification: bool, data_module: SOENDataModule | None = None) -> None:
        """Validate that model output dimension matches dataset num_classes for classification tasks.

        Args:
            model_path: Path to the model file (.soen, .pth, or .json)
            is_classification: Whether this is a classification task
            data_module: Optional data module to infer num_classes from dataset if not in config

        Raises:
            ValueError: If model output dimension doesn't match dataset num_classes
        """
        # Only validate for classification tasks
        if not is_classification:
            return

        # Get dataset num_classes from config, or infer from dataset if not set
        num_classes = getattr(self.config.data, "num_classes", None)
        if num_classes is None:
            # Try to infer from dataset if data_module is available
            if data_module is not None:
                try:
                    num_classes = data_module.infer_num_classes()
                    if num_classes > 0:
                        # Store inferred value back in config for consistency
                        self.config.data.num_classes = num_classes
                        self._log.info(f"[JAX] Auto-detected num_classes={num_classes} from dataset")
                    else:
                        self._log.warning("[JAX] Could not infer num_classes from dataset. Skipping validation.")
                        return
                except Exception as e:
                    self._log.warning(f"[JAX] Failed to infer num_classes from dataset: {e}. Skipping validation.")
                    return
            else:
                # No data_module and no config value - skip validation
                self._log.warning("[JAX] num_classes not set in config and data_module not available. Skipping validation.")
                return

        # Load the model to get its output dimension
        try:
            from soen_toolkit.core import SOENModelCore

            soen_model = SOENModelCore.load(model_path, show_logs=False)
        except Exception as e:
            self._log.warning(f"Cannot validate output dimension: failed to load model: {e}")
            return

        # Get model output dimension (last layer's dimension)
        if not hasattr(soen_model, "layer_nodes") or not soen_model.layer_nodes:
            self._log.warning("Cannot validate output dimension: model.layer_nodes not available")
            return

        # Get the last layer's dimension
        sorted_layer_ids = sorted(soen_model.layer_nodes.keys())
        if not sorted_layer_ids:
            self._log.warning("Cannot validate output dimension: no layers found")
            return

        last_layer_id = sorted_layer_ids[-1]
        model_output_dim = soen_model.layer_nodes[last_layer_id]

        # Compare with dataset num_classes
        if model_output_dim != num_classes:
            paradigm = getattr(self.config.training, "paradigm", "supervised")
            mapping = getattr(self.config.training, "mapping", "seq2static")

            error_msg = [
                "=" * 80,
                "MODEL OUTPUT DIMENSION MISMATCH DETECTED (JAX Backend)",
                "=" * 80,
                "",
                f"Your model's output layer has dimension {model_output_dim}, but your dataset",
                f"expects {num_classes} classes.",
                "",
                "This mismatch will cause training to fail with errors like:",
                f"  - JAX: NaN losses or 'Target {num_classes - 1} is out of bounds'",
                "  - Invalid predictions and incorrect metrics",
                "",
                "DETAILS:",
                f"  • Model output layer (layer_id={last_layer_id}): {model_output_dim} dimensions",
                f"  • Dataset num_classes (config.data.num_classes): {num_classes}",
                f"  • Task: {paradigm} classification ({mapping})",
                f"  • Model path: {model_path}",
                "",
                "HOW TO FIX:",
                "",
                "Option 1: Rebuild your model with the correct output dimension",
                f"  - Modify your model YAML/architecture to set the last layer's 'dim' to {num_classes}",
                f"  - Or rebuild from model_creation_gui with output_dim={num_classes}",
                "",
                "Option 2: Regenerate your dataset to match the model",
                f"  - If your dataset generator supports it, set num_classes={model_output_dim}",
                "  - Or exclude classes that don't fit (e.g., set include_space=False if model has 26 outputs)",
                "",
                "Option 3: Check your dataset generation script",
                "  - Ensure dataset labels use values in range [0, num_classes-1]",
                "  - For seq2seq tasks: padding should use -100 (already handled by framework)",
                "",
                "=" * 80,
            ]

            self._log.error("\n".join(error_msg))
            raise ValueError(f"Model output dimension ({model_output_dim}) does not match dataset num_classes ({num_classes}). See error message above for detailed fix instructions.")

    def _log_internal_connection_stats(self, trainer: _LowLevelJaxTrainer, params: Any, epoch: int) -> None:
        """Log statistics about internal connections in the current model and params."""
        import jax.numpy as jnp

        params_dict = self._params_to_dict(params)
        internal_conns_params = params_dict.get("internal_connections", {})

        # Check JAX model specs
        internal_conns_model = {}
        for spec in trainer.jax_model.layers:
            if getattr(spec, "internal_J", None) is not None:
                internal_conns_model[spec.layer_id] = spec.internal_J

        self._log.info(f"[Epoch {epoch}] Internal Connection Stats:")
        self._log.info(f"  Found in params dict: {len(internal_conns_params)} layers")
        self._log.info(f"  Found in JAX model: {len(internal_conns_model)} layers")

        # Compare and log details
        all_layer_ids = set(list(internal_conns_params.keys()) + list(internal_conns_model.keys()))
        for layer_id in sorted(all_layer_ids):
            in_params = layer_id in internal_conns_params
            in_model = layer_id in internal_conns_model

            if in_params:
                params_J = internal_conns_params[layer_id]
                if hasattr(params_J, "shape"):
                    params_mean = float(jnp.mean(params_J))
                    params_std = float(jnp.std(params_J))
                    params_min = float(jnp.min(params_J))
                    params_max = float(jnp.max(params_J))
                    self._log.info(f"    Layer {layer_id} in params: shape={params_J.shape}, mean={params_mean:.6f}, std={params_std:.6f}, range=[{params_min:.6f}, {params_max:.6f}]")
                else:
                    self._log.info(f"    Layer {layer_id} in params: type={type(params_J)}")

            if in_model:
                model_J = internal_conns_model[layer_id]
                if hasattr(model_J, "shape"):
                    model_mean = float(jnp.mean(model_J))
                    model_std = float(jnp.std(model_J))
                    model_min = float(jnp.min(model_J))
                    model_max = float(jnp.max(model_J))
                    self._log.info(f"    Layer {layer_id} in model: shape={model_J.shape}, mean={model_mean:.6f}, std={model_std:.6f}, range=[{model_min:.6f}, {model_max:.6f}]")
                else:
                    self._log.info(f"    Layer {layer_id} in model: type={type(model_J)}")

            # Compare if both exist
            if in_params and in_model:
                params_J = internal_conns_params[layer_id]
                model_J = internal_conns_model[layer_id]
                try:
                    params_arr = jnp.asarray(params_J)
                    model_arr = jnp.asarray(model_J)
                    if params_arr.shape == model_arr.shape:
                        diff = jnp.abs(params_arr - model_arr)
                        max_diff = float(jnp.max(diff))
                        mean_diff = float(jnp.mean(diff))
                        are_same = jnp.allclose(params_arr, model_arr, rtol=1e-5, atol=1e-6)
                        self._log.info(f"    Layer {layer_id} comparison: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, identical={are_same}")
                    else:
                        self._log.warning(f"    Layer {layer_id} shape mismatch: params={params_arr.shape}, model={model_arr.shape}")
                except Exception as e:
                    self._log.warning(f"    Layer {layer_id} comparison failed: {e}")

            if not in_params and not in_model:
                self._log.info(f"    Layer {layer_id}: no internal connection")

        if not all_layer_ids:
            self._log.info("  No internal connections found in this model")

    def _save_soen_checkpoint(self, trainer: _LowLevelJaxTrainer, params: Any, soen_path: Path, epoch: int, val_loss: float) -> None:
        """Convert JAX params back to SOENModelCore and save as .soen."""
        from soen_toolkit.core import SOENModelCore
        from soen_toolkit.utils.port_to_jax.convert import convert_jax_to_core_model
        from soen_toolkit.utils.port_to_jax.jax_training.callbacks.checkpointing import (
            apply_params_to_jax_model,
        )

        # Load the base model to get architecture
        # This can be slow, so do it here or cache it if possible
        base_core = SOENModelCore.load(trainer.cfg.model_path, show_logs=False)

        # Update JAX model with trained params
        apply_params_to_jax_model(trainer.jax_model, self._params_to_dict(params))

        # Convert JAX model back to PyTorch core (updates base_core in-place)
        updated_core = convert_jax_to_core_model(trainer.jax_model, base_core=base_core)

        # Save as .soen
        updated_core.save(str(soen_path))
        self._log.info(f"[Checkpoint Save Epoch {epoch}] Saved SOEN checkpoint: {soen_path}")

    def _log_artifacts_to_mlflow(self) -> None:
        """Log checkpoint artifacts to MLflow (mirrors PyTorch Lightning behavior)."""
        if not self.ml_logger:
            return

        # Check if MLflow is active (required for any artifact uploads)
        mlflow_active = getattr(getattr(self.config, "logging", object()), "mlflow_active", False)
        mlflow_log_artifacts = getattr(getattr(self.config, "logging", object()), "mlflow_log_artifacts", True)

        if not mlflow_active:
            return

        # Get MLflow experiment and run_id
        exp = getattr(self.ml_logger, "experiment", None)
        run_id = getattr(self.ml_logger, "run_id", None)
        if exp is None or run_id is None:
            return

        # Collect checkpoint files and log files separately
        ckpt_dir = Path(self.repeat_ckpt_dir)
        log_dir = Path(self.repeat_log_dir)

        # Checkpoints respect mlflow_log_artifacts setting
        checkpoint_artifacts = []
        if mlflow_log_artifacts:
            checkpoint_artifacts.extend(sorted(p for p in ckpt_dir.glob("*.soen") if p.is_file()))
            checkpoint_artifacts.extend(sorted(p for p in ckpt_dir.glob("*.pkl") if p.is_file()))

        # Log files are ALWAYS uploaded (regardless of mlflow_log_artifacts setting)
        log_artifacts = sorted(p for p in log_dir.glob("*.log") if p.is_file())

        # Combine: log files first, then checkpoints
        artifacts = log_artifacts + checkpoint_artifacts

        # If no artifacts to upload, return early
        if not artifacts:
            return

        # Import exception types for error handling (optional dependencies)
        try:
            import botocore.exceptions as boto_exceptions
        except ImportError:
            boto_exceptions = None
        try:
            import urllib3.exceptions as urllib3_exceptions
        except ImportError:
            urllib3_exceptions = None

        # Size threshold for warnings (100MB)
        SIZE_WARNING_THRESHOLD_MB = 100
        SIZE_WARNING_THRESHOLD_BYTES = SIZE_WARNING_THRESHOLD_MB * 1024 * 1024

        # Log each artifact to MLflow
        for artifact_path in artifacts:
            try:
                # Check file size before uploading
                file_size_bytes = artifact_path.stat().st_size
                file_size_mb = file_size_bytes / (1024 * 1024)

                # Log file size for debugging
                self._log.debug(f"Attempting to upload artifact to MLflow: {artifact_path.name} ({file_size_mb:.2f}MB)")

                # Warn if file is large
                if file_size_bytes > SIZE_WARNING_THRESHOLD_BYTES:
                    self._log.warning(
                        f"Large artifact detected: {artifact_path.name} ({file_size_mb:.2f}MB). "
                        f"Upload may fail or timeout. Consider setting mlflow_log_artifacts: false "
                        f"in your logging config for large models."
                    )

                # Attempt upload
                exp.log_artifact(run_id, str(artifact_path))
                self._log.debug(f"Logged artifact to MLflow: {artifact_path.name}")

            except Exception as e:
                # Check if this is an S3/network-related error
                is_s3_error = False
                error_type_name = type(e).__name__

                if boto_exceptions and isinstance(e, (boto_exceptions.ConnectionClosedError, boto_exceptions.ClientError)):
                    is_s3_error = True
                elif urllib3_exceptions and isinstance(e, (urllib3_exceptions.ProtocolError,)):
                    is_s3_error = True
                elif isinstance(e, (BrokenPipeError, ConnectionError, OSError)):
                    is_s3_error = True
                elif "Connection" in error_type_name or "S3" in error_type_name or "upload" in error_type_name.lower():
                    is_s3_error = True

                # Get file size for error message
                try:
                    file_size_bytes = artifact_path.stat().st_size
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    size_str = f"{file_size_mb:.2f}MB"
                except Exception:
                    size_str = "unknown size"

                # Log appropriate error message
                if is_s3_error:
                    self._log.warning(
                        f"Failed to upload artifact to MLflow: {artifact_path.name} ({size_str}). "
                        f"This may occur with large model checkpoints or network issues. "
                        f"Consider setting mlflow_log_artifacts: false in your logging config "
                        f"for large models. Original error: {error_type_name}: {e}"
                    )
                else:
                    self._log.warning(
                        f"Failed to upload artifact to MLflow: {artifact_path.name} ({size_str}). "
                        f"Original error: {error_type_name}: {e}"
                    )
                # Continue training even if artifact upload fails (non-fatal)

    def _log_param_means(self, params: Any, *, tag: str) -> None:
        """Log mean values of connection and layer parameters to verify updates.

        This executes on the host and does not affect any jitted functions.
        """
        params_dict = self._params_to_dict(params)
        # Connection params
        conn = params_dict.get("connection_params", None)
        if conn is not None:
            cm = float(np.asarray(jnp.mean(conn)))
            self._log.info(f"[JAX] {tag} conn_params mean={cm:.6e}")
        # Layer params (sequence per layer; entries may be arrays or tuples)
        lps = params_dict.get("layer_params", ())
        if isinstance(lps, (list, tuple)):
            means: list[float] = []
            for i, item in enumerate(lps):
                if isinstance(item, (list, tuple)):
                    parts = [jnp.ravel(jnp.asarray(a)) for a in item]
                    if len(parts) == 0:
                        means.append(float("nan"))  # No params (e.g., Linear layer)
                    else:
                        vec = jnp.concatenate(parts, axis=0)
                        if vec.size == 0:
                            means.append(float("nan"))
                        else:
                            means.append(float(np.asarray(jnp.mean(vec))))
                else:
                    arr = jnp.asarray(item)
                    if arr.size == 0:
                        means.append(float("nan"))
                    else:
                        means.append(float(np.asarray(jnp.mean(arr))))
            # Only log layer params if at least one has valid values
            if any(not np.isnan(m) for m in means):
                self._log.info(f"[JAX] {tag} layer_params mean per-layer={means}")
            # Connection params shape info
            if conn is not None:
                num_connections = conn.shape[0]
                max_dims = conn.shape[1:] if len(conn.shape) > 1 else "N/A"
                self._log.info(f"[JAX] {tag} connection_params shape={conn.shape} ({num_connections} connections, max_dims={max_dims})")

    def _get_dataloaders(self, data_module: SOENDataModule) -> tuple[Iterable, Iterable]:
        """Use JAX-native HDF5 loader if configured, else fall back to PyTorch."""
        data_path = getattr(self.config.data, "data_path", "")
        if str(data_path).lower().endswith((".hdf5", ".h5")):
            self._log.info("Using JAX-native HDF5 data loader.")

            # Determine if labels should be int (classification) or float (distillation/regression)
            paradigm = str(getattr(self.config.training, "paradigm", "supervised")).lower()
            labels_as_int = paradigm not in ["distillation", "regression", "unsupervised", "self_supervised"]

            if not labels_as_int:
                self._log.info(f"Paradigm '{paradigm}' detected: loading labels as float32 for regression/distillation")

            (Xtr, ytr), (Xva, yva) = load_hdf5_splits(
                str(data_path),
                target_seq_len=getattr(self.config.data, "target_seq_len", None),
                scale_min=getattr(self.config.data, "min_scale", None),
                scale_max=getattr(self.config.data, "max_scale", None),
                labels_as_int=labels_as_int,
            )

            # Log data statistics for debugging
            self._log.info(f"Loaded data: X_train shape={Xtr.shape} dtype={Xtr.dtype}, y_train shape={ytr.shape} dtype={ytr.dtype}")
            self._log.info(f"            X_val shape={Xva.shape} dtype={Xva.dtype}, y_val shape={yva.shape} dtype={yva.dtype}")
            self._log.info(f"y_train stats: min={ytr.min():.6f}, max={ytr.max():.6f}, mean={ytr.mean():.6f}")

            train_loader = JAXBatchIterator(Xtr, ytr, batch_size=self.config.training.batch_size, shuffle=True)
            val_loader = JAXBatchIterator(Xva, yva, batch_size=self.config.training.batch_size, shuffle=False)
            return train_loader, val_loader
        # Fallback to wrapping the PyTorch datamodule
        self._log.info("Using PyTorch data loader with JAX wrapper.")
        return data_module.train_dataloader(), data_module.val_dataloader()

    def _batch_to_jax(self, xb: Any, yb: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Convert a batch of data (torch tensor or numpy array) to JAX arrays."""
        import torch as _torch

        if hasattr(xb, "detach") and isinstance(xb, _torch.Tensor):
            xb = xb.detach().cpu().numpy()
        if hasattr(yb, "detach") and isinstance(yb, _torch.Tensor):
            yb = yb.detach().cpu().numpy()
        if not isinstance(xb, np.ndarray):
            xb = np.asarray(xb)
        if not isinstance(yb, np.ndarray):
            yb = np.asarray(yb)
        return jnp.asarray(xb), jnp.asarray(yb)

    def _wrap_dataloader_to_jnp(self, loader: Iterable[tuple[Any, Any]]) -> Iterable[tuple[jnp.ndarray, jnp.ndarray]]:
        for xb, yb in loader:
            yield self._batch_to_jax(xb, yb)

    def _build_low_level_trainer(self, data_module: SOENDataModule) -> _LowLevelJaxTrainer:
        # Decide platform for JAX
        platform = "cpu"
        acc = str(getattr(self.config.training, "accelerator", "auto")).lower()
        if acc in {"gpu", "cuda"}:
            platform = "cuda"
        elif acc in {"mps", "metal"}:
            platform = "metal"
        elif acc == "cpu":
            platform = "cpu"
        else:
            # For "auto" or other values, check if GPU is available
            try:
                import jax
                devices = jax.devices()
                gpu_devices = [d for d in devices if d.platform == "gpu"]
                if gpu_devices:
                    platform = "cuda"
                    self._log.info(
                        "[JAX] Auto-detected GPU platform: found %d GPU device(s): %s",
                        len(gpu_devices),
                        [str(d) for d in gpu_devices],
                    )
                else:
                    platform = "cpu"
                    self._log.info(
                        "[JAX] Auto-detected CPU platform: available devices: %s",
                        [str(d) for d in devices],
                    )
            except Exception as e:
                self._log.warning(
                    "[JAX] Failed to detect platform, defaulting to CPU: %s",
                    e,
                )
                platform = "cpu"
        self._log.info(
            "[JAX] Selected platform based on config.accelerator='%s' -> '%s'",
            str(getattr(self.config.training, "accelerator", "auto")),
            platform,
        )

        mapping = str(getattr(self.config.training, "mapping", "seq2static")).lower()
        # Use the migrated loss configuration
        losses_cfg = getattr(self.config.training.loss, "losses", [])
        loss_names = {getattr(loss, "name", None) for loss in losses_cfg if getattr(loss, "name", None)}
        classification_losses = {"cross_entropy"}
        # [Fix later]
        # Autoregressive support enabled
        # if "autoregressive_cross_entropy" in loss_names or bool(getattr(self.config.training, "autoregressive", False)):
        #     msg = "JAX backend does not yet support autoregressive cross-entropy training"
        #     raise NotImplementedError(msg)
        is_classification = bool(loss_names & classification_losses)
        if not loss_names:
            # Try to infer num_classes from config or dataset
            num_classes = getattr(self.config.data, "num_classes", None)
            if num_classes is None:
                # Try to infer from dataset
                try:
                    num_classes = data_module.infer_num_classes()
                    if num_classes > 0:
                        self.config.data.num_classes = num_classes
                        self._log.info(f"[JAX] Auto-detected num_classes={num_classes} from dataset")
                    else:
                        num_classes = 0
                except Exception as e:
                    self._log.warning(f"[JAX] Failed to infer num_classes from dataset: {e}")
                    num_classes = 0
            else:
                num_classes = int(num_classes)
            is_classification = num_classes > 1
        self.is_classification = is_classification  # Store for metric logging
        self._log.info("[JAX] Task classification=%s mapping=%s", is_classification, mapping)

        method, params, range_start, range_end = None, None, None, None
        model_cfg = getattr(self.config, "model", None)
        if model_cfg is not None:
            method = getattr(model_cfg, "parsed_time_pooling_name", None)
            params = getattr(model_cfg, "parsed_time_pooling_params", None)
            range_start = getattr(model_cfg, "range_start", None)
            range_end = getattr(model_cfg, "range_end", None)

        # Extract AR settings
        ar_cfg = getattr(self.config.training, "ar", None)
        ar_enabled = False
        ar_tspt = 1
        ar_pooling_method = "final"
        ar_pooling_params = {}

        if ar_cfg is not None:
            # New config structure
            if isinstance(ar_cfg, dict):
                ar_enabled = bool(ar_cfg.get("enabled", False))
                ar_tspt = int(ar_cfg.get("time_steps_per_token", 1))
                token_pooling = ar_cfg.get("token_pooling", {})
                if isinstance(token_pooling, dict):
                    ar_pooling_method = token_pooling.get("method", "final")
                    ar_pooling_params = token_pooling.get("params", {})
            else:
                # AutoregressiveConfig object
                ar_enabled = bool(ar_cfg.enabled)
                ar_tspt = int(ar_cfg.time_steps_per_token)
                token_pooling = ar_cfg.token_pooling
                if isinstance(token_pooling, dict):
                    ar_pooling_method = token_pooling.get("method", "final")
                    ar_pooling_params = token_pooling.get("params", {})
        else:
            # Old config structure (backward compatibility)
            ar_enabled = bool(getattr(self.config.training, "autoregressive", False))
            ar_tspt = int(getattr(self.config.training, "time_steps_per_token", 1))
            # Old structure didn't have explicit pooling config, assumed default
            ar_pooling_method = "final"
            ar_pooling_params = {}

        # Extract optimizer configuration for LR and optimizer setup
        optimizer_config = getattr(self.config.training, "optimizer", None)
        if optimizer_config is None:
            optimizer_lr = 0.001
        elif isinstance(optimizer_config, dict):
            optimizer_lr = float(optimizer_config.get("lr", 0.001))
        else:
            optimizer_lr = float(getattr(optimizer_config, "lr", 0.001))

        # Check if precision override is requested
        precision = str(getattr(self.config.training, "precision", "32-true")).lower()
        if precision in {"64", "64-true", "float64"}:
            import jax
            jax.config.update("jax_enable_x64", True)
            self._log.info("[JAX] Precision: Enabled 64-bit precision (jax_enable_x64=True)")
        else:
            self._log.info(f"[JAX] Precision: Default (usually 32-bit). Config value: {precision}")

        # Get paradigm for distillation support
        paradigm = str(getattr(self.config.training, "paradigm", "supervised")).lower()

        # Fail-fast metric validation:
        # In distillation/regression runs, the dataset "labels" are teacher trajectories (float),
        # not ground-truth class ids. Computing classification metrics (e.g. accuracy) against
        # those trajectories is at best "pseudo-accuracy" and often meaningless.
        if not is_classification:
            # If the user did not specify metrics, the global default is ["accuracy"].
            # That default makes sense for classification but not for distillation/regression.
            # Avoid failing users who didn't set logging.metrics explicitly.
            if self.metric_names == ["accuracy"]:
                self._log.info(
                    "[JAX] logging.metrics defaulted to ['accuracy'], but this run is non-classification "
                    f"(paradigm='{paradigm}'). Clearing metrics. Set logging.metrics explicitly to track "
                    "non-classification metrics."
                )
                self.metric_names = []

            classification_metrics = {
                "accuracy",
                "precision",
                "recall",
                "f1",
                "topk",
                "top_k_accuracy",
                "autoregressive_accuracy",
                "perplexity",
                "bits_per_character",
            }
            requested = {m for m in self.metric_names if m not in {"loss", "input_seq_len"}}
            bad = sorted(requested & classification_metrics)
            if bad:
                raise ValueError(
                    "Non-classification run requested classification metrics "
                    f"{bad}. For distillation, the dataset labels are teacher trajectories "
                    "(regression targets), not class ids. Remove these metrics or run a "
                    "supervised classification setup with ground-truth labels."
                )

        tcfg = TrainingConfigJAX(
            lr=optimizer_lr,
            max_epochs=int(self.config.training.max_epochs),
            log_every_n_steps=int(self.config.logging.log_freq),
            seed=int(self.config.seed),
            platform=platform,
            classification=is_classification,
            mapping=mapping,
            paradigm=paradigm,
            profile=self.do_profile,
            profile_samples=int(getattr(self.config.profiler, "num_train_batches", 0) or 0) if self.do_profile else 0,
            trace_dir=str(self.repeat_log_dir) if self.do_profile else None,
            time_pooling_method=str(method or "max"),
            time_pooling_params=dict(params or {}),
            time_pooling_range_start=range_start,
            time_pooling_range_end=range_end,
            # AR settings
            autoregressive=ar_enabled,
            ar_time_steps_per_token=ar_tspt,
            ar_token_pooling_method=ar_pooling_method,
            ar_token_pooling_params=ar_pooling_params,
        )

        # Enforce platform choice if possible
        # Note: This usually needs to happen before JAX ops, but we do our best
        if platform == "cpu":
            import os
            # If JAX is already initialized, this might warn but not work
            # But setting the environment variable helps for sub-processes or if JAX isn't init'd yet
            os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
            try:
                import jax
                # This will work if JAX hasn't been initialized yet
                jax.config.update("jax_platform_name", "cpu")
            except Exception as e:
                self._log.warning(f"[JAX] Could not force CPU platform: {e}")

        self._log.info(f"[JAX] Selected platform: {platform} (config accelerator: {acc})")

        train_iter = self._wrap_dataloader_to_jnp(data_module.train_dataloader())
        val_iter = self._wrap_dataloader_to_jnp(data_module.val_dataloader())
        dcfg = DataConfigJAX(train_loader=train_iter, val_loader=val_iter)

        # Prefer building from YAML architecture if present; else require base_model_path
        model_path = None
        base = getattr(self.config.model, "base_model_path", None)
        if base is not None:
            base_path = Path(str(base))
            suffix = base_path.suffix.lower()
            if suffix in {".yaml", ".yml"}:
                # Treat YAML as an architecture spec; build and save a temporary .soen
                self._log.warning(
                    "model.base_model_path points to a YAML spec. Building from YAML. Consider using model.architecture_yaml instead.",
                )
                from soen_toolkit.core.model_yaml import build_model_from_yaml

                # Training seed already set globally by ExperimentRunner; disable YAML seed
                core = build_model_from_yaml(base_path, honor_yaml_seed=False)
                tmp_path = Path(self.repeat_ckpt_dir) / "initial_model_for_jax.soen"
                core.save(str(tmp_path))
                model_path = str(tmp_path)
            elif suffix in {".json", ".soen", ".pth"}:
                # Check load_exact_model_state to match PyTorch Lightning behavior
                load_exact = getattr(self.config.model, "load_exact_model_state", False)
                if load_exact:
                    # Load exact state (weights and config) from the file
                    model_path = str(base_path)
                    self._log.info(f"[JAX] load_exact_model_state=True: loading exact state from {base_path}")
                else:
                    # Load config from model file and rebuild with fresh weights
                    self._log.info(f"[JAX] load_exact_model_state=False: loading config from {base_path} and rebuilding with fresh weights")
                    from soen_toolkit.core import SOENModelCore
                    base_model_for_config = SOENModelCore.load(str(base_path))
                    # Rebuild model from extracted config (fresh weights using training seed)
                    core = SOENModelCore(
                        sim_config=base_model_for_config.sim_config,
                        layers_config=base_model_for_config.layers_config,
                        connections_config=base_model_for_config.connections_config,
                    )
                    tmp_path = Path(self.repeat_ckpt_dir) / "initial_model_for_jax.soen"
                    core.save(str(tmp_path))
                    model_path = str(tmp_path)
                    self._log.info(f"[JAX] Rebuilt model saved to {tmp_path}")
            elif suffix == ".ckpt":
                msg = "base_model_path points to a .ckpt. Use training.train_from_checkpoint to resume; do not set base_model_path to .ckpt."
                raise ValueError(
                    msg,
                )
            else:
                msg = f"Unsupported model.base_model_path extension '{suffix}'. Use .soen/.pth/.json, or provide a YAML via model.architecture_yaml."
                raise ValueError(
                    msg,
                )
        if model_path is None:
            # When training from YAML, build a temporary SOEN core and save it to a temp .soen
            from soen_toolkit.core.model_yaml import build_model_from_yaml

            arch_yaml = getattr(self.config.model, "architecture_yaml", None)
            arch_inline = getattr(self.config.model, "architecture", None)
            if arch_yaml is None and arch_inline is None:
                msg = "JAX backend requires 'model.base_model_path' or 'model.architecture_yaml' / 'model.architecture'."
                raise RuntimeError(msg)
            core = build_model_from_yaml(
                arch_inline if arch_inline is not None else arch_yaml,
                honor_yaml_seed=False,
            )
            tmp_path = Path(self.repeat_ckpt_dir) / "initial_model_for_jax.soen"
            core.save(str(tmp_path))
            model_path = str(tmp_path)

        # Validate model output dimension matches dataset num_classes (for classification tasks)
        self._validate_model_output_dimension(model_path, is_classification, data_module=data_module)

        # Extract dt and dt_learnable from config (for parity with Lightning wrapper)
        model_dt = getattr(self.config.model, "dt", None)
        model_dt_learnable = getattr(self.config.model, "dt_learnable", None)

        ecfg = ExperimentConfigJAX(
            model_path=model_path,
            data=dcfg,
            training=tcfg,
            model_dt=float(model_dt) if model_dt is not None else None,
            model_dt_learnable=bool(model_dt_learnable) if model_dt_learnable is not None else None,
        )

        # Build optional LR schedule from config callbacks
        # Steps per epoch is required for proper LR scheduling
        try:
            steps_per_epoch = len(data_module.train_dataloader())
            if steps_per_epoch == 0:
                raise ValueError("Train dataloader has zero batches")
        except Exception as e:
            # Fail-fast: steps_per_epoch is required for LR scheduling
            self._log.error(f"[JAX] Failed to compute steps_per_epoch from dataloader: {e}. This is required for learning rate scheduling.")
            raise RuntimeError(f"Cannot determine steps per epoch for LR scheduling. Dataloader length calculation failed: {e}") from e
        lr_schedule = _make_jax_schedule(self.config, steps_per_epoch=int(steps_per_epoch), max_epochs=int(self.config.training.max_epochs))

        # Build metric names list for JAX trainer (skip loss/internal names)
        metric_names = [m for m in self.metric_names if m not in {"loss", "input_seq_len"}]

        # Extract gradient clipping and accumulation settings
        gradient_clip_val = getattr(self.config.training, "gradient_clip_val", None)
        gradient_clip_algorithm = str(getattr(self.config.training, "gradient_clip_algorithm", "norm"))
        accumulate_grad_batches = int(getattr(self.config.training, "accumulate_grad_batches", 1))

        # Extract optimizer name and kwargs (optimizer_config already extracted above)
        if optimizer_config is None:
            optimizer_name = "adamw"
            optimizer_kwargs = {}
        elif isinstance(optimizer_config, dict):
            optimizer_name = str(optimizer_config.get("name", "adamw")).lower()
            optimizer_kwargs = dict(optimizer_config.get("kwargs", {}))
            # Also extract params if present (for backward compatibility)
            if "params" in optimizer_config:
                optimizer_kwargs.update(optimizer_config["params"])
        else:
            optimizer_name = str(getattr(optimizer_config, "name", "adamw")).lower()
            optimizer_kwargs = dict(getattr(optimizer_config, "kwargs", {}))

        # Convert optimizer kwargs to ensure all numeric values are Python scalars
        # JAX arrays or other types can break optax's numeric parameter handling
        from soen_toolkit.training.utils.helpers import safe_convert_optimizer_kwargs

        optimizer_kwargs = safe_convert_optimizer_kwargs(optimizer_kwargs)

        # Filter out weight_decay for optimizers that don't support it (adam, rmsprop, etc.)
        # This prevents warnings when default weight_decay is inherited from OptimizerConfig defaults
        optimizer_name_lower = optimizer_name.lower()
        if optimizer_name_lower in ("adam", "rmsprop"):
            # These optimizers don't support weight_decay in optax
            if "weight_decay" in optimizer_kwargs:
                optimizer_kwargs.pop("weight_decay")

        return _LowLevelJaxTrainer(
            ecfg,
            losses=self.losses,
            lr_schedule=lr_schedule,
            metric_names=metric_names,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            accumulate_grad_batches=accumulate_grad_batches,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _iter_with_heartbeat(self, iterator: Any, tag: str, heartbeat_seconds: int = 10) -> Iterable[tuple[Any, Any]]:
        """Iterate over an iterator, emitting a heartbeat log if fetching the next item
        takes longer than heartbeat_seconds. Helps diagnose DataLoader stalls.
        """
        # Use simple iteration if threading issues suspected or for simplicity
        # But for now, keep it robust
        # Note: JAX on GPU can sometimes deadlock if data loading happens in threads that fight for GIL/resources
        # If you see stalls, try disabling this wrapper

        # Simple direct iteration for now to avoid thread overhead and potential deadlocks with JAX
        # The threading logic was causing overhead and complexity without much benefit if JAX is properly async
        # Let's trust the data loader
        yield from iterator

    def _warmup_jax_functions(self, loader: Iterable, params: dict, opt_state: dict | None, train_step: callable, trainer: _LowLevelJaxTrainer) -> None:
        """Warm-up and cache compiled functions for training."""
        warm_xb, warm_yb = next(iter(loader))
        warm_xb_j, warm_yb_j = self._batch_to_jax(warm_xb, warm_yb)
        msg = f"Compiling JAX train_step for batch {tuple(warm_xb_j.shape)}..."
        self._log.info(msg)
        t_compile0 = time.perf_counter()

        # Get a warmup RNG key
        warmup_rng_key = trainer.get_batch_rng_key()

        if trainer.accumulate_grad_batches > 1:
            # Warm up gradient accumulation functions
            _l, _m, _g = trainer._accumulate_grads_step(params, warm_xb_j, warm_yb_j, None, None, None, warmup_rng_key)
            _l.block_until_ready()
            # Warm up apply accumulated grads
            _p, _s = trainer._apply_accumulated_grads(params, opt_state, _g, 1)
            jax.tree_map(lambda x: x.block_until_ready(), _p)
        else:
            # Standard single-step warmup
            _p, _s, _l, _m, _g = train_step(params, opt_state, warm_xb_j, warm_yb_j, None, None, None, warmup_rng_key)
            _l.block_until_ready()

        t_compile1 = time.perf_counter()
        msg2 = f"JAX compile completed in {(t_compile1 - t_compile0):.2f}s"
        self._log.info(msg2)

    def _run_one_epoch(
        self,
        *,
        trainer: _LowLevelJaxTrainer,
        loader: Iterable,
        step_fn: callable,
        params: dict,
        opt_state: dict | None,
        limit: int | None,
        epoch: int,
        phase: str,
        global_step: int,
        lw_schedulers: list | None = None,
        steps_per_epoch: int | None = None,
    ) -> dict:
        """Runs a single epoch of training or evaluation.

        Uses deferred logging to avoid device-to-host transfers in hot loop.
        Metrics are kept as JAX arrays until actually needed for logging.
        """
        # Reset states at the start of each epoch for stateful training
        if phase == "train" and self.stateful_enable_train:
            self._train_states = None
            self._train_s1_states = None
            self._train_s2_states = None
        elif phase == "val" and self.stateful_enable_val:
            self._val_states = None
            self._val_s1_states = None
            self._val_s2_states = None

        # Keep metrics as JAX arrays during accumulation to avoid device-to-host transfers
        loss_sum = jnp.array(0.0, dtype=jnp.float32)
        acc_sum = jnp.array(0.0, dtype=jnp.float32)
        num_batches = 0
        metric_sums = {name: jnp.array(0.0, dtype=jnp.float32) for name in self.metric_names}
        # Accumulate loss component terms for logging (match PyTorch naming)
        loss_terms_sum: dict[str, jnp.ndarray] = {}  # loss_name -> sum
        loss_terms_w_sum: dict[str, jnp.ndarray] = {}  # loss_name -> weighted sum
        batch_data_loading_times = []
        batch_conversion_times = []
        lw_callback_times = []
        step_times = []  # Per-step timing for detailed analysis
        step_compute_times = []  # JAX computation time (excluding data loading/conversion)
        t_end_of_step = time.perf_counter()

        # Deferred logging: store previous step's metrics to log while current step runs
        prev_step_metrics: dict[str, float] | None = None
        prev_step_log_step: int | None = None

        trace_started = False
        if self.do_profile and phase == "train" and epoch == 0:
            start_batch = int(getattr(self.config.profiler, "trace_start_batch", 5))
            duration = int(getattr(self.config.profiler, "trace_duration_batches", 5))
            stop_batch = start_batch + duration

        # Try to get batch count for progress bar (non-critical, so we handle gracefully)
        n_batches = None
        try:
            n_batches = len(loader)
        except Exception as e:
            # Non-critical: just for progress bar display
            self._log.debug(f"[JAX] Could not determine batch count for progress bar: {e}")
            n_batches = None

        pbar = None
        if tqdm is not None:
            total = limit if limit is not None else n_batches
            pbar = tqdm(
                total=total,
                desc=f"Epoch {epoch + 1} [{phase}]",
                leave=False,
                dynamic_ncols=True,
                unit="batch",
            )

        data_iter = iter(loader)
        for xb, yb in self._iter_with_heartbeat(data_iter, tag=phase, heartbeat_seconds=10):
            t_start_of_step = time.perf_counter()
            if self.do_profile:
                batch_data_loading_times.append((t_start_of_step - t_end_of_step) * 1000)

            if self.do_profile:
                t_start_conv = time.perf_counter()
            # Convert batch to jnp
            xb_j, yb_j = self._batch_to_jax(xb, yb)
            if self.do_profile:
                t_end_conv = time.perf_counter()
                batch_conversion_times.append((t_end_conv - t_start_conv) * 1000)

            # Log first-batch statistics for debugging (epoch 0, batch 0)
            if epoch == 0 and num_batches == 0 and phase == "train":
                self._log.info("=" * 60)
                self._log.info("FIRST BATCH STATISTICS (Epoch 0, Batch 0) [JAX]")
                self._log.info("=" * 60)
                self._log.info(f"Inputs shape: {xb_j.shape}, dtype: {xb_j.dtype}")
                self._log.info(f"Inputs range: [{float(jnp.min(xb_j)):.6f}, {float(jnp.max(xb_j)):.6f}], mean: {float(jnp.mean(xb_j)):.6f}")
                self._log.info(f"Targets shape: {yb_j.shape}, dtype: {yb_j.dtype}")
                self._log.info(f"Targets range: [{float(jnp.min(yb_j)):.6f}, {float(jnp.max(yb_j)):.6f}], mean: {float(jnp.mean(yb_j)):.6f}")
                paradigm = getattr(trainer.cfg.training, "paradigm", "supervised")
                self._log.info(f"Training paradigm: {paradigm}")
                self._log.info("=" * 60)

            # Start timing JAX computation (after data conversion)
            t_step_start = time.perf_counter() if self.do_profile else None

            # Prepare initial states for stateful training
            initial_states = None
            s1_inits = None
            s2_inits = None
            if (phase == "train" and self.stateful_enable_train) or (phase == "val" and self.stateful_enable_val):
                if phase == "train":
                    stored_states = self._train_states
                    stored_s1 = self._train_s1_states
                    stored_s2 = self._train_s2_states
                else:
                    stored_states = self._val_states
                    stored_s1 = self._val_s1_states
                    stored_s2 = self._val_s2_states

                # Broadcast stored states to match current batch size
                if stored_states is not None:
                    batch_size = int(xb_j.shape[0])
                    initial_states = {
                        lid: jnp.repeat(state, batch_size, axis=0)
                        for lid, state in stored_states.items()
                    }
                    if stored_s1 is not None:
                        s1_inits = {
                            lid: jnp.repeat(state, batch_size, axis=0)
                            for lid, state in stored_s1.items()
                        }
                    if stored_s2 is not None:
                        s2_inits = {
                            lid: jnp.repeat(state, batch_size, axis=0)
                            for lid, state in stored_s2.items()
                        }

            # --- JAX Step ---
            if phase == "train":
                assert opt_state is not None
                if num_batches == 0 and epoch == 0:
                    self._log.info("[JAX] Running first train step (may trigger recompilation if shapes differ)")
                    if trainer.accumulate_grad_batches > 1:
                        self._log.info(f"[JAX] Gradient accumulation active: accumulating over {trainer.accumulate_grad_batches} batches")

                # Apply loss weight schedulers before computing the batch loss
                if lw_schedulers:
                    if self.do_profile:
                        t_start_lw = time.perf_counter()
                    sp = int(steps_per_epoch) if steps_per_epoch is not None else (int(n_batches) if n_batches is not None else 1)
                    for _sch in lw_schedulers:
                        _sch.apply(trainer=trainer, epoch=epoch, step=global_step, steps_per_epoch=sp, max_epochs=int(trainer.cfg.training.max_epochs))
                    if self.do_profile:
                        t_end_lw = time.perf_counter()
                        lw_callback_times.append((t_end_lw - t_start_lw) * 1000)

                # Get noise RNG key for this batch
                batch_rng_key = trainer.get_batch_rng_key()

                # Handle gradient accumulation
                if trainer.accumulate_grad_batches > 1:
                    # Compute gradients only (no parameter update)
                    if not hasattr(trainer, "_accumulate_grads_step_jit"):
                        trainer._accumulate_grads_step_jit = jax.jit(trainer._accumulate_grads_step)
                    if not hasattr(trainer, "_apply_accumulated_grads_jit"):
                        trainer._apply_accumulated_grads_jit = jax.jit(trainer._apply_accumulated_grads)

                    loss, metrics, grads = trainer._accumulate_grads_step_jit(params, xb_j, yb_j, initial_states, s1_inits, s2_inits, batch_rng_key)

                    # Accumulate gradients
                    if not hasattr(trainer, "accumulated_grads") or trainer.accum_steps == 0:
                        trainer.accumulated_grads = grads
                        trainer.accum_steps = 1
                    else:
                        trainer.accumulated_grads = jax.tree_map(lambda acc, g: acc + g, trainer.accumulated_grads, grads)
                        trainer.accum_steps += 1

                    # Apply accumulated gradients when we've accumulated enough
                    if trainer.accum_steps >= trainer.accumulate_grad_batches:
                        params, opt_state = trainer._apply_accumulated_grads_jit(params, opt_state, trainer.accumulated_grads, trainer.accum_steps)
                        # Reset accumulation
                        trainer.accumulated_grads = jax.tree_map(jnp.zeros_like, params)
                        trainer.accum_steps = 0
                        global_step += 1  # Only increment global step when params are actually updated
                else:
                    # No accumulation: standard single-step update
                    params, opt_state, loss, metrics, grads = step_fn(params, opt_state, xb_j, yb_j, initial_states, s1_inits, s2_inits, batch_rng_key)
                    global_step += 1
            else:  # eval
                # Get noise RNG key for this batch
                batch_rng_key = trainer.get_batch_rng_key()
                if num_batches == 0 and epoch == 0:
                    self._log.info("[JAX] Running first eval step")
                loss, metrics = step_fn(params, xb_j, yb_j, initial_states, s1_inits, s2_inits, batch_rng_key)

            loss.block_until_ready()

            # Extract and store states for stateful training
            if (phase == "train" and self.stateful_enable_train) or (phase == "val" and self.stateful_enable_val):
                # Extract histories from metrics
                if "_histories" in metrics:
                    histories = metrics["_histories"]
                    batch_size = int(xb_j.shape[0])
                    sample_idx = self._pick_sample_index(batch_size)

                    # Extract final states from histories
                    main_states, s1_states, s2_states = self._extract_states_from_histories(
                        trainer,
                        histories,
                        sample_idx,
                        s1_final_by_layer=metrics.get("_s1_final_by_layer"),
                        s2_final_by_layer=metrics.get("_s2_final_by_layer"),
                    )

                    # Store states for next batch
                    if phase == "train":
                        self._train_states = main_states
                        self._train_s1_states = s1_states
                        self._train_s2_states = s2_states
                    else:
                        self._val_states = main_states
                        self._val_s1_states = s1_states
                        self._val_s2_states = s2_states

            # Track step compute time (JAX computation only)
            if self.do_profile and t_step_start is not None:
                t_step_end = time.perf_counter()
                step_compute_times.append((t_step_end - t_step_start) * 1000)

            # Accumulate metrics as JAX arrays (no device-to-host transfer)
            loss_sum = loss_sum + loss
            if self.is_classification:
                acc_val = metrics.get("acc", jnp.array(0.0, dtype=jnp.float32))
                acc_sum = acc_sum + acc_val

            # Metrics are computed inside the step; aggregate as JAX arrays
            if self.metric_names and isinstance(metrics, dict):
                for name in self.metric_names:
                    if name in metrics:
                        metric_sums[name] = metric_sums[name] + metrics[name]

            # Accumulate loss component terms if available (for matching PyTorch logging)
            if isinstance(metrics, dict) and "loss_terms" in metrics and "loss_terms_w" in metrics:
                loss_terms_batch = metrics.get("loss_terms")
                loss_terms_w_batch = metrics.get("loss_terms_w")
                if hasattr(trainer, "losses") and trainer.losses:
                    for idx, loss_spec in enumerate(trainer.losses):
                        # Loss name can be in "name" or "loss_function" key (check loss_function first as that's what trainer uses)
                        loss_name = loss_spec.get("loss_function") or loss_spec.get("name", "unknown")
                        if loss_name == "unknown":
                            continue
                        # Normalize loss name (e.g., "cross_entropy" -> "cross_entropy")
                        loss_key = loss_name.replace("-", "_")

                        if loss_terms_batch is not None and idx < loss_terms_batch.shape[0]:
                            if loss_key not in loss_terms_sum:
                                loss_terms_sum[loss_key] = jnp.array(0.0, dtype=jnp.float32)
                            loss_terms_sum[loss_key] = loss_terms_sum[loss_key] + loss_terms_batch[idx]

                        if loss_terms_w_batch is not None and idx < loss_terms_w_batch.shape[0]:
                            if loss_key not in loss_terms_w_sum:
                                loss_terms_w_sum[loss_key] = jnp.array(0.0, dtype=jnp.float32)
                            loss_terms_w_sum[loss_key] = loss_terms_w_sum[loss_key] + loss_terms_w_batch[idx]

            num_batches += 1

            # Log previous step's metrics while current step computation happens (async dispatch)
            if prev_step_metrics is not None and prev_step_log_step is not None:
                # Convert to float only when actually logging (previous step's metrics)
                self.logger_adapter.log_metrics(prev_step_metrics, step=prev_step_log_step)

            # Defer logging of current step's metrics until next iteration
            if phase == "train" and trainer.cfg.training.log_every_n_steps and (global_step % trainer.cfg.training.log_every_n_steps == 0):
                # Store metrics as JAX arrays; convert to float in next iteration
                prev_step_metrics_raw = {"train/loss_step": loss}  # Keep as JAX array
                prev_step_log_step = global_step
                # Convert to float for next iteration's logging (async dispatch)
                prev_step_metrics = {k: float(v) for k, v in prev_step_metrics_raw.items()}
            else:
                prev_step_metrics = None
                prev_step_log_step = None

            # Update progress bar (minor device-to-host transfer, but acceptable for UX)
            # Note: Conversion happens after step completes, so doesn't block next step
            if pbar:
                pbar.update(1)
                # Convert loss to float for display (happens after step completes)
                loss_val = float(loss)
                pbar.set_postfix({"loss": f"{loss_val:.5f}"}, refresh=False)

            t_end_of_step = time.perf_counter()

            # Track total step time (including data loading, conversion, computation, logging)
            if self.do_profile:
                step_times.append((t_end_of_step - t_start_of_step) * 1000)

            if self.do_profile and phase == "train" and epoch == 0:
                if num_batches == start_batch:
                    self._log.info(f"Starting JAX profiler trace for {duration} batches.")
                    try:
                        import jax.profiler

                        jax.profiler.start_trace(str(self.repeat_log_dir))
                        trace_started = True
                    except Exception as e:
                        if "tensorflow.python.profiler.trace" in str(e) or "tensorflow" in str(e).lower():
                            self._log.warning(
                                "JAX profiler trace started but TensorFlow not available. "
                                "Traces will be created but cannot be viewed in TensorBoard. "
                                "Install 'tensorflow>=2.14' or install with 'uv pip install -e .[profiling]' "
                                "to enable trace viewing. JSON summary is still available."
                            )
                        else:
                            self._log.warning(f"Failed to start JAX profiler trace: {e}")
                        trace_started = False
                if trace_started and num_batches >= stop_batch:
                    self._log.info("Stopping JAX profiler trace.")
                    try:
                        import jax.profiler

                        jax.profiler.stop_trace()
                        trace_started = False
                    except Exception as e:
                        self._log.warning(f"Failed to stop JAX profiler trace: {e}")

            if limit is not None and num_batches >= limit:
                self._log.info(f"[JAX] {phase.capitalize()} loop limit reached ({limit} batches)")
                break

        # Flush any remaining accumulated gradients at epoch end (train phase only)
        if phase == "train" and trainer.accumulate_grad_batches > 1:
            if hasattr(trainer, "accumulated_grads") and trainer.accum_steps > 0:
                self._log.info(f"[JAX] Flushing {trainer.accum_steps} accumulated gradient(s) at epoch end")
                if not hasattr(trainer, "_apply_accumulated_grads_jit"):
                    trainer._apply_accumulated_grads_jit = jax.jit(trainer._apply_accumulated_grads)
                params, opt_state = trainer._apply_accumulated_grads_jit(params, opt_state, trainer.accumulated_grads, trainer.accum_steps)
                # Reset accumulation
                trainer.accumulated_grads = jax.tree_map(jnp.zeros_like, params)
                trainer.accum_steps = 0
                global_step += 1

        # Log final pending metrics if any
        if prev_step_metrics is not None and prev_step_log_step is not None:
            self.logger_adapter.log_metrics(prev_step_metrics, step=prev_step_log_step)

        if pbar:
            pbar.close()

        if trace_started:
            self._log.warning("JAX profiler trace was started but not stopped within the epoch. Stopping now.")
            try:
                import jax.profiler

                jax.profiler.stop_trace()
            except Exception as e:
                self._log.warning(f"Failed to stop JAX profiler trace at end of epoch: {e}")

        if self.do_profile:
            if batch_data_loading_times:
                avg_load_time = np.mean(batch_data_loading_times[1:])  # Skip first
                self.profiler_results[f"epoch_{epoch}_{phase}_avg_batch_load_ms"] = avg_load_time
                if len(batch_data_loading_times) > 1:
                    self.profiler_results[f"epoch_{epoch}_{phase}_min_batch_load_ms"] = np.min(batch_data_loading_times[1:])
                    self.profiler_results[f"epoch_{epoch}_{phase}_max_batch_load_ms"] = np.max(batch_data_loading_times[1:])
                    self.profiler_results[f"epoch_{epoch}_{phase}_std_batch_load_ms"] = np.std(batch_data_loading_times[1:])
            if batch_conversion_times:
                self.profiler_results[f"epoch_{epoch}_{phase}_avg_batch_convert_ms"] = np.mean(batch_conversion_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_min_batch_convert_ms"] = np.min(batch_conversion_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_max_batch_convert_ms"] = np.max(batch_conversion_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_std_batch_convert_ms"] = np.std(batch_conversion_times)
            if lw_callback_times:
                self.profiler_results[f"epoch_{epoch}_{phase}_avg_lw_callback_ms"] = np.mean(lw_callback_times)
            if step_times:
                self.profiler_results[f"epoch_{epoch}_{phase}_avg_step_ms"] = np.mean(step_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_min_step_ms"] = np.min(step_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_max_step_ms"] = np.max(step_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_std_step_ms"] = np.std(step_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_p50_step_ms"] = np.percentile(step_times, 50)
                self.profiler_results[f"epoch_{epoch}_{phase}_p95_step_ms"] = np.percentile(step_times, 95)
                self.profiler_results[f"epoch_{epoch}_{phase}_p99_step_ms"] = np.percentile(step_times, 99)
                self.profiler_results[f"epoch_{epoch}_{phase}_total_steps"] = len(step_times)
            if step_compute_times:
                self.profiler_results[f"epoch_{epoch}_{phase}_avg_compute_ms"] = np.mean(step_compute_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_min_compute_ms"] = np.min(step_compute_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_max_compute_ms"] = np.max(step_compute_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_std_compute_ms"] = np.std(step_compute_times)
                self.profiler_results[f"epoch_{epoch}_{phase}_p50_compute_ms"] = np.percentile(step_compute_times, 50)
                self.profiler_results[f"epoch_{epoch}_{phase}_p95_compute_ms"] = np.percentile(step_compute_times, 95)
                self.profiler_results[f"epoch_{epoch}_{phase}_p99_compute_ms"] = np.percentile(step_compute_times, 99)

        # Sync internal connections from params back to JAX model specs after training
        # This ensures spec.internal_J reflects the trained values, not just during checkpointing
        internal_conns = self._params_to_dict(params).get("internal_connections", {})
        if internal_conns:
            # Use module-level jnp (already imported at top of file)
            for spec in trainer.jax_model.layers:
                layer_id = int(spec.layer_id)
                if layer_id in internal_conns:
                    spec.internal_J = jnp.asarray(internal_conns[layer_id])

        # Convert accumulated JAX arrays to Python floats only at end of epoch
        avg_loss = float(loss_sum / max(1, num_batches))
        avg_acc = float(acc_sum / max(1, num_batches)) if self.is_classification else 0.0
        avg_metrics = {name: float(metric_sums[name] / max(1, num_batches)) for name in self.metric_names}
        # Average loss component terms
        avg_loss_terms = {name: float(loss_terms_sum[name] / max(1, num_batches)) for name in loss_terms_sum}
        avg_loss_terms_w = {name: float(loss_terms_w_sum[name] / max(1, num_batches)) for name in loss_terms_w_sum}

        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "metrics": avg_metrics,
            "loss_terms": avg_loss_terms,
            "loss_terms_w": avg_loss_terms_w,
            "params": params,
            "opt_state": opt_state,
            "global_step": global_step,
            "num_batches": num_batches,
        }

    def _manage_checkpoints(
        self,
        trainer: _LowLevelJaxTrainer,
        params: Any,
        opt_state: dict | None,
        epoch: int,
        global_step: int,
        vl: float,
    ) -> None:
        """Save last and top-k checkpoints with enhanced format (includes architecture)."""
        # Use asynchronous saving to avoid blocking training loop
        # Copy params to host/CPU first to avoid race conditions with JAX async execution
        # This ensures the saved params are from this specific epoch
        import jax

        from soen_toolkit.utils.port_to_jax.jax_training.callbacks.checkpointing import (
            save_last_checkpoint,
            save_topk_checkpoint,
        )

        def _to_cpu(x):
            if hasattr(x, "device_buffer"):
                return jax.device_get(x)
            return x

        # Deep copy params structure with values moved to CPU
        # Note: params is a dict of trees, opt_state is complex
        # For safety, we just rely on JAX's device_get which handles trees
        cpu_params_obj = jax.device_get(params)
        cpu_params = self._params_to_dict(cpu_params_obj)

        # opt_state can be large, but saving it is required for resumption
        # If disk IO is the bottleneck, async thread helps
        # If CPU transfer is bottleneck, this won't help much but is safer
        cpu_opt_state = jax.device_get(opt_state)

        # Ensure at most one background save at a time.
        # If a prior save failed, surface it immediately (fail-fast).
        prev = getattr(self, "_checkpoint_thread", None)
        if prev is not None:
            try:
                if prev.is_alive():
                    prev.join()
            finally:
                self._checkpoint_thread = None
                if self._checkpoint_thread_error is not None:
                    err = self._checkpoint_thread_error
                    self._checkpoint_thread_error = None
                    raise RuntimeError(f"Previous checkpoint save failed: {err}") from err

        # Define the heavy lifting save function
        def _save_task(params_in, opt_state_in, epoch_in, step_in, val_loss_in):
            # Save last checkpoint with enhanced format
            if self.save_last:
                # Save enhanced .pkl with architecture
                _last_pkl_path, last_soen_path = save_last_checkpoint(
                    ckpt_dir=Path(self.repeat_ckpt_dir),
                    params=params_in,
                    opt_state=opt_state_in,
                    epoch=epoch_in,
                    global_step=step_in,
                    val_loss=val_loss_in,
                    jax_model=trainer.jax_model,
                    topology=trainer.topology,
                    source_model_path=trainer.cfg.model_path,
                )

                # Save .soen checkpoint (conversion happens here)
                self._save_soen_checkpoint(trainer, params_in, last_soen_path, epoch_in, val_loss_in)

            # Save top-k best checkpoints with enhanced format
            if self.save_top_k != 0:
                if val_loss_in < self.best_metric or self.save_top_k == -1:
                    # Save enhanced .pkl with architecture
                    _best_pkl_path, best_soen_path = save_topk_checkpoint(
                        ckpt_dir=Path(self.repeat_ckpt_dir),
                        params=params_in,
                        opt_state=opt_state_in,
                        epoch=epoch_in,
                        global_step=step_in,
                        val_loss=val_loss_in,
                        jax_model=trainer.jax_model,
                        topology=trainer.topology,
                        source_model_path=trainer.cfg.model_path,
                    )

                    # Save .soen checkpoint
                    self._save_soen_checkpoint(trainer, params_in, best_soen_path, epoch_in, val_loss_in)

                    # Track this checkpoint
                    self.best_k_models[best_soen_path] = val_loss_in
                    self.best_ckpt_path = str(best_soen_path)
                    self.best_metric = min(self.best_metric, val_loss_in)

                    # Clean up old checkpoints
                    if self.save_top_k > 0 and len(self.best_k_models) > self.save_top_k:
                        sorted_models = sorted(self.best_k_models.items(), key=lambda x: x[1])
                        for ckpt_path, _ in sorted_models[self.save_top_k :]:
                            if ckpt_path.exists():
                                ckpt_path.unlink()
                                self._log.debug(f"Deleted old checkpoint: {ckpt_path}")
                            pkl_path = ckpt_path.with_suffix(".pkl")
                            if pkl_path.exists():
                                pkl_path.unlink()
                            del self.best_k_models[ckpt_path]

        # Launch in background thread
        if self.do_profile:
             # During profiling, run synchronously to capture timing
             t_start_save = time.perf_counter()
             _save_task(cpu_params, cpu_opt_state, epoch, global_step, vl)
             t_end_save = time.perf_counter()
             self.profiler_results[f"epoch_{epoch}_checkpoint_total_ms"] = (t_end_save - t_start_save) * 1000
        else:
             # Normal training: use background thread
             # Wait for any previous save to complete to avoid OOM?
             # For now, fire and forget but log start
             import threading
             def _thread_entry():
                 try:
                     _save_task(cpu_params, cpu_opt_state, epoch, global_step, vl)
                 except Exception as e:
                     self._checkpoint_thread_error = e
             t = threading.Thread(target=_thread_entry, daemon=True)
             t.start()
             self._checkpoint_thread = t
             self._log.debug(f"[JAX] Launched background checkpoint save for epoch {epoch}")

    def _collect_memory_and_gpu_stats(self) -> None:
        """Collect memory and GPU utilization statistics."""
        if not self.do_profile:
            return

        # Memory profiling
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            self.profiler_results["memory_rss_gb"] = mem_info.rss / (1024**3)  # Resident Set Size
            self.profiler_results["memory_vms_gb"] = mem_info.vms / (1024**3)  # Virtual Memory Size
            if hasattr(process, "memory_full_info"):
                mem_full = process.memory_full_info()
                self.profiler_results["memory_uss_gb"] = mem_full.uss / (1024**3)  # Unique Set Size
        except ImportError:
            self._log.debug("psutil not available for memory profiling")
        except Exception as e:
            self._log.debug(f"Failed to collect memory stats: {e}")

        # GPU memory profiling (JAX)
        try:
            if jax is not None:
                devices = jax.devices()
                gpu_devices = [d for d in devices if d.device_kind == "gpu"]
                if gpu_devices:
                    # Get memory usage from JAX
                    for i, device in enumerate(gpu_devices):
                        try:
                            # JAX doesn't provide direct memory stats, but we can check device memory
                            # This is a placeholder - actual GPU memory tracking requires device-specific APIs
                            device_id = device.id if hasattr(device, "id") else i
                            self.profiler_results[f"gpu_{device_id}_device_kind"] = device.device_kind
                        except Exception:
                            pass

                    # Try to get memory info if available
                    try:
                        from jax.lib import xla_bridge

                        if hasattr(xla_bridge, "get_backend"):
                            backend = xla_bridge.get_backend()
                            if hasattr(backend, "memory_stats"):
                                stats = backend.memory_stats()
                                if stats:
                                    for key, value in stats.items():
                                        if isinstance(value, (int, float)):
                                            # Convert bytes to GB if memory-related
                                            if "bytes" in key.lower() or "memory" in key.lower():
                                                self.profiler_results[f"gpu_{key}"] = value / (1024**3)
                                            else:
                                                self.profiler_results[f"gpu_{key}"] = value
                    except Exception:
                        pass
        except Exception as e:
            self._log.debug(f"Failed to collect GPU stats: {e}")

        # System CPU info
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.profiler_results["cpu_percent"] = cpu_percent
            cpu_count = psutil.cpu_count()
            self.profiler_results["cpu_count"] = cpu_count
        except ImportError:
            pass
        except Exception as e:
            self._log.debug(f"Failed to collect CPU stats: {e}")

    def _log_gpu_info_to_mlflow(self) -> None:
        """Log GPU availability and device info to MLflow for debugging."""
        if not self.ml_logger:
            return

        try:
            import jax
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == "gpu"]
            cpu_devices = [d for d in devices if d.platform == "cpu"]

            # Log to file
            self._log.info(f"[JAX] Device detection: {len(devices)} total device(s)")
            self._log.info(f"[JAX] GPU devices: {len(gpu_devices)}")
            self._log.info(f"[JAX] CPU devices: {len(cpu_devices)}")
            for i, d in enumerate(devices):
                self._log.info(f"[JAX]   Device {i}: {d} (platform={d.platform})")

            # Log metrics to MLflow
            exp = getattr(self.ml_logger, "experiment", None)
            run_id = getattr(self.ml_logger, "run_id", None)
            if exp is None or run_id is None:
                return

            metrics = {
                "system/jax_devices_total": len(devices),
                "system/jax_gpu_devices": len(gpu_devices),
                "system/jax_cpu_devices": len(cpu_devices),
                "system/using_gpu": 1 if len(gpu_devices) > 0 else 0,
            }

            # Log device names as parameters
            params = {
                "system/jax_backend": "jax",
                "system/jax_device_platforms": ",".join([d.platform for d in devices]),
            }
            if gpu_devices:
                params["system/jax_gpu_names"] = ",".join([str(d) for d in gpu_devices])

            try:
                exp.log_metrics(run_id, metrics, step=0)
                exp.log_params(run_id, params)
                self._log.info(f"[JAX] Logged GPU detection metrics to MLflow: {metrics}")
            except Exception as e:
                self._log.debug(f"Failed to log GPU info to MLflow: {e}")
        except Exception as e:
            self._log.debug(f"Failed to detect JAX devices: {e}")

    def fit(self, data_module: SOENDataModule) -> str:
        if self.do_profile:
            t_fit_start = time.perf_counter()

        # Log GPU availability and device info to MLflow
        self._log_gpu_info_to_mlflow()

        trainer = self._build_low_level_trainer(data_module)
        # Log optimizer info after trainer creation - all logs go to file
        optimizer_name = getattr(trainer, "_optimizer_name", "unknown")
        self._log.info("=" * 80)
        self._log.info("[JAX] Training configuration:")
        self._log.info(f"  Optimizer: {optimizer_name}")
        self._log.info(f"  Learning rate: {trainer.cfg.training.lr}")
        self._log.info(f"  Max epochs: {trainer.cfg.training.max_epochs}")
        self._log.info(f"  Batch size: {self.config.training.batch_size}")
        self._log.info("=" * 80)

        self.best_metric = float("inf")
        self.best_ckpt_path = str(Path(self.repeat_ckpt_dir) / "last.soen")

        # Get data loaders (either JAX-native HDF5 or wrapped PyTorch)
        if self.do_profile:
            t0_dl = time.perf_counter()
        train_loader, val_loader = self._get_dataloaders(data_module)
        if self.do_profile:
            t1_dl = time.perf_counter()
            self.profiler_results["initial_dataloader_creation_ms"] = (t1_dl - t0_dl) * 1000


        params = trainer.params
        opt_state = trainer.opt_state
        train_step = trainer.train_step
        eval_step = trainer.eval_step
        # In jax_backend.py, after trainer is built (around line 964):
        self._log.info("=== CONNECTION ANALYSIS ===")
        for i, conn in enumerate(trainer.jax_model.connections):
            from_lid = conn.from_layer
            to_lid = conn.to_layer
            # Find layer indices
            from_idx = next((idx for idx, spec in enumerate(trainer.jax_model.layers) if spec.layer_id == from_lid), -1)
            to_idx = next((idx for idx, spec in enumerate(trainer.jax_model.layers) if spec.layer_id == to_lid), -1)
            self._log.info(f"  Conn[{i}]: L{from_lid}(idx={from_idx}) -> L{to_lid}(idx={to_idx}), J.shape={conn.J.shape}")

        # Also check layer order:
        self._log.info("=== LAYER ORDER ===")
        for idx, spec in enumerate(trainer.jax_model.layers):
            self._log.info(f"  Layer[{idx}]: id={spec.layer_id}, kind={spec.kind}, dim={spec.dim}")
        # --- One-time warmup compile with timing to avoid "hang" perception ---
        if self.do_profile:
            t0_warmup = time.perf_counter()
        self._warmup_jax_functions(train_loader, params, opt_state, train_step, trainer)
        if self.do_profile:
            t1_warmup = time.perf_counter()
            self.profiler_results["jax_warmup_ms"] = (t1_warmup - t0_warmup) * 1000

        # Save initial state if configured
        if getattr(self.config.training, "save_initial_state", False):
            from soen_toolkit.utils.port_to_jax.jax_training.callbacks.checkpointing import (
                save_initial_checkpoint,
            )

            self._log.info("[JAX] Saving initial model state...")
            init_pkl, init_soen = save_initial_checkpoint(
                ckpt_dir=Path(self.repeat_ckpt_dir),
                params=trainer.params,
                opt_state=trainer.opt_state,
                jax_model=trainer.jax_model,
                topology=trainer.topology,
                source_model_path=trainer.cfg.model_path,
            )

            # Also save the .soen file if configured (defaults to True if not set, to match behavior)
            if getattr(self.config.training, "save_soen_core_in_checkpoint", True):
                # Using epoch=0 and val_loss=inf for initial state
                self._save_soen_checkpoint(trainer, trainer.params, init_soen, epoch=0, val_loss=float("inf"))

            self._log.info(f"[JAX] Initial state saved to {init_pkl}")

        global_step = 0
        # Optional QAT enable (mirrors Torch callback semantics)
        qat_cfg = {}
        try:
            if isinstance(getattr(self.config, "callbacks", {}), dict):
                qat_cfg = self.config.callbacks.get("qat", {}) or {}
        except Exception:
            qat_cfg = {}
        qat_helper = None
        if isinstance(qat_cfg, dict) and bool(qat_cfg.get("active", False)):
            try:
                qat_helper = QATStraightThroughJAX(
                    min_val=float(qat_cfg["min_val"]),
                    max_val=float(qat_cfg["max_val"]),
                    bits=(int(qat_cfg["bits"]) if qat_cfg.get("bits") is not None else None),
                    levels=(int(qat_cfg["levels"]) if qat_cfg.get("levels") is not None else None),
                    update_on_train_epoch_start=bool(qat_cfg.get("update_on_train_epoch_start", False)),
                    stochastic_rounding=bool(qat_cfg.get("stochastic_rounding", False)),
                )
                if self.do_profile:
                    t_start_qat = time.perf_counter()
                qat_helper.enable(trainer)
                if self.do_profile:
                    t_end_qat = time.perf_counter()
                    self.profiler_results["qat_initial_enable_ms"] = (t_end_qat - t_start_qat) * 1000
                self._log.info("[JAX] QAT-STE enabled (uniform codebook)")
            except Exception as e:
                self._log.warning("[JAX] Failed to enable QAT-STE: %s", e)

        # Optional time pooling scale scheduler
        tp_cfg = {}
        try:
            if isinstance(getattr(self.config, "callbacks", {}), dict):
                tp_cfg = self.config.callbacks.get("time_pooling_scale_scheduler", {}) or {}
        except Exception:
            tp_cfg = {}
        tp_sched = None
        if isinstance(tp_cfg, dict) and tp_cfg:
            try:
                tp_sched = TimePoolingScaleSchedulerJAX(
                    start_scale=float(tp_cfg["start_scale"]),
                    end_scale=float(tp_cfg["end_scale"]),
                    start_epoch=int(tp_cfg.get("start_epoch", 0)),
                    end_epoch=(int(tp_cfg["end_epoch"]) if tp_cfg.get("end_epoch") is not None else None),
                )
            except Exception as e:
                self._log.warning("[JAX] Failed to set up time_pooling_scale_scheduler: %s", e)

        # Optional loss weight schedulers (list)
        lw_cfgs = []
        try:
            if isinstance(getattr(self.config, "callbacks", {}), dict):
                lw_cfgs = list(self.config.callbacks.get("loss_weight_schedulers", []) or [])
        except Exception:
            lw_cfgs = []
        lw_schedulers: list[LossWeightSchedulerJAX] = []
        for spec in lw_cfgs:
            try:
                if not isinstance(spec, dict):
                    continue
                if "loss_name" in spec and "scheduler_type" in spec:
                    lw_schedulers.append(
                        LossWeightSchedulerJAX(
                            loss_name=str(spec["loss_name"]),
                            scheduler_type=str(spec["scheduler_type"]),
                            params=dict(spec.get("params", {}) or {}),
                            start_epoch=int(spec.get("start_epoch", 0)),
                            end_epoch=(int(spec["end_epoch"]) if spec.get("end_epoch") is not None else None),
                            per_step=bool(spec.get("per_step", True)),
                        ),
                    )
            except Exception as e:
                self._log.warning("[JAX] Skipping invalid loss_weight_scheduler: %s", e)

        # Early stopping state (mirrors Torch EarlyStopping on val_loss)
        early_patience = None
        try:
            if isinstance(getattr(self.config.training, "early_stopping_patience", None), int):
                early_patience = int(self.config.training.early_stopping_patience)
        except Exception:
            early_patience = None
        best_val = float("inf")
        bad_epochs = 0

        # Pre-build sequence length scheduler and JIT cache if dt scaling is enabled
        seq_len_sched = None
        jit_cache: dict[tuple[int, float], tuple[object, object, object]] = {}  # (seq_len, dt) -> (params, train_step, eval_step)

        seq_cfg = {}
        try:
            if isinstance(getattr(self.config, "callbacks", {}), dict):
                seq_cfg = self.config.callbacks.get("seq_len_scheduler", {}) or {}
        except Exception:
            seq_cfg = {}

        if bool(seq_cfg.get("active", False)):
            try:
                start = int(seq_cfg.get("start_len", getattr(self.config.data, "target_seq_len", 0) or 1))
                end = int(seq_cfg.get("end_len", start))
                start_epoch = int(seq_cfg.get("start_epoch", 0))
                end_epoch = seq_cfg.get("end_epoch", None)
                if end_epoch is not None:
                    end_epoch = int(end_epoch)
                scale_dt = bool(seq_cfg.get("scale_dt", False))

                # Extract base_dt from the model if scaling is enabled
                base_dt = None
                if scale_dt:
                    try:
                        # SOENModelCore stores dt directly as an attribute (Parameter or float)
                        dt_value = trainer.core.dt
                        if hasattr(dt_value, "item"):
                            # It's a torch Parameter/Tensor
                            base_dt = float(dt_value.detach().cpu().item())
                        else:
                            # It's already a float
                            base_dt = float(dt_value)
                        self._log.info(f"[JAX] Extracted base_dt={base_dt} from model for dt scaling")
                    except Exception as e:
                        self._log.warning(f"[JAX] Failed to extract base_dt from model: {e}. Disabling dt scaling.")
                        scale_dt = False

                seq_len_sched = TargetSeqLenSchedulerJAX(
                    data_module=data_module,
                    start_len=start,
                    end_len=end,
                    max_epochs=int(self.config.training.max_epochs),
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    scale_dt=scale_dt,
                    base_dt=base_dt,
                )

                # Pre-compile jitted functions for all unique configurations if dt scaling is active
                if scale_dt:
                    unique_configs = seq_len_sched.get_all_unique_configs()
                    msg = f"[JAX] Pre-compiling jitted functions for {len(unique_configs)} unique (seq_len, dt) configurations. This may take a few minutes..."
                    self._log.info(msg)

                    t_precompile_start = time.perf_counter()
                    config_compile_times = []

                    for idx, (cfg_seq_len, cfg_dt) in enumerate(unique_configs):
                        t_config_start = time.perf_counter()
                        msg = f"[JAX] Building config {idx + 1}/{len(unique_configs)}: seq_len={cfg_seq_len}, dt={cfg_dt:.6f}"
                        self._log.info(msg)

                        # Temporarily update model dt (save original value for restoration)
                        if hasattr(trainer.core.dt, "item"):
                            # It's a torch Parameter/Tensor - extract the value
                            original_dt = float(trainer.core.dt.detach().cpu().item())
                        else:
                            # It's already a float
                            original_dt = float(trainer.core.dt)

                        # Update dt in both the core and sim_config
                        if hasattr(trainer.core.dt, "data"):
                            # It's a Parameter - update the data
                            import torch

                            trainer.core.dt.data = torch.tensor(float(cfg_dt), dtype=trainer.core.dt.dtype, device=trainer.core.dt.device)
                        else:
                            # It's a float - assign directly
                            trainer.core.dt = float(cfg_dt)
                        trainer.core.sim_config.dt = float(cfg_dt)

                        # Rebuild topology and jitted functions with new dt
                        from soen_toolkit.utils.port_to_jax.pure_forward import build_topology, convert_params_to_arrays

                        cfg_topology = build_topology(trainer.jax_model)
                        cfg_layer_params, cfg_connection_params = convert_params_to_arrays(
                            {"layers": {spec.layer_id: spec.params for spec in trainer.jax_model.layers}, "connections": {(c.from_layer, c.to_layer): c.J for c in trainer.jax_model.connections}},
                            cfg_topology,
                            connection_constraints=cfg_topology.connection_constraints,
                        )
                        # Extract internal connections from model (same as in trainer._build_param_tree)
                        cfg_internal_conns = {}
                        for spec in trainer.jax_model.layers:
                            if getattr(spec, "internal_J", None) is not None:
                                cfg_internal_conns[spec.layer_id] = spec.internal_J
                        # Ensure layer_params is a tuple to match params structure (optax requirement)
                        cfg_params = {"layer_params": tuple(cfg_layer_params), "connection_params": cfg_connection_params, "internal_connections": cfg_internal_conns}

                        # Build temporary trainer with this configuration
                        import jax

                        from soen_toolkit.utils.port_to_jax.jax_training.trainer import JaxTrainer as _TempTrainer

                        temp_trainer = _TempTrainer.__new__(_TempTrainer)
                        temp_trainer._log = self._log
                        temp_trainer.cfg = trainer.cfg
                        temp_trainer.losses = trainer.losses
                        temp_trainer.topology = cfg_topology
                        temp_trainer.params = cfg_params
                        temp_trainer.edge_masks = trainer.edge_masks
                        temp_trainer._qat_active = trainer._qat_active
                        temp_trainer._qat_codebook = trainer._qat_codebook
                        temp_trainer.metric_names = trainer.metric_names
                        temp_trainer.tx = trainer.tx  # Add optimizer for _train_step
                        temp_trainer.jax_model = trainer.jax_model  # Add for _forward_with_params

                        # Copy missing attributes required for training step and gradient masking (Polarity/Connectivity support)
                        temp_trainer.edge_learnable_mask = trainer.edge_learnable_mask
                        temp_trainer.edge_constraint_mins = trainer.edge_constraint_mins
                        temp_trainer.edge_constraint_maxs = trainer.edge_constraint_maxs
                        temp_trainer.internal_learnable = trainer.internal_learnable
                        temp_trainer._internal_masks = trainer._internal_masks
                        temp_trainer._internal_constraint_mins = trainer._internal_constraint_mins
                        temp_trainer._internal_constraint_maxs = trainer._internal_constraint_maxs
                        temp_trainer._original_internal_conns = trainer._original_internal_conns

                        # JIT compile steps for this configuration
                        cfg_train_step = jax.jit(temp_trainer._train_step)
                        cfg_eval_step = jax.jit(temp_trainer._eval_step)

                        # Warm up to trigger compilation
                        warm_xb, warm_yb = next(iter(train_loader))
                        warm_xb_j, warm_yb_j = self._batch_to_jax(warm_xb, warm_yb)
                        warmup_rng_key = temp_trainer.get_batch_rng_key()
                        _p, _s, _l, _m, _g = cfg_train_step(cfg_params, opt_state, warm_xb_j, warm_yb_j, None, None, None, warmup_rng_key)
                        _l.block_until_ready()

                        # Store in cache
                        jit_cache[(cfg_seq_len, cfg_dt)] = (cfg_params, cfg_train_step, cfg_eval_step)

                        # Restore original dt
                        if hasattr(trainer.core.dt, "data"):
                            import torch

                            trainer.core.dt.data = torch.tensor(float(original_dt), dtype=trainer.core.dt.dtype, device=trainer.core.dt.device)
                        else:
                            trainer.core.dt = float(original_dt)
                        trainer.core.sim_config.dt = float(original_dt)

                        # Record timing for this configuration
                        t_config_end = time.perf_counter()
                        config_time_ms = (t_config_end - t_config_start) * 1000
                        config_compile_times.append(config_time_ms)
                        self._log.info(f"[JAX] Config {idx + 1}/{len(unique_configs)} compiled in {config_time_ms:.1f}ms")

                    # Log summary statistics
                    t_precompile_end = time.perf_counter()
                    total_precompile_ms = (t_precompile_end - t_precompile_start) * 1000
                    avg_config_ms = np.mean(config_compile_times) if config_compile_times else 0.0
                    min_config_ms = np.min(config_compile_times) if config_compile_times else 0.0
                    max_config_ms = np.max(config_compile_times) if config_compile_times else 0.0

                    self._log.info(f"[JAX] Pre-compilation complete. Cache contains {len(jit_cache)} configurations.")
                    self._log.info(f"[JAX] Total pre-compilation time: {total_precompile_ms:.1f}ms ({total_precompile_ms / 1000:.2f}s)")
                    self._log.info(f"[JAX] Per-config times: avg={avg_config_ms:.1f}ms, min={min_config_ms:.1f}ms, max={max_config_ms:.1f}ms")

                    # Store in profiler results if profiling is active
                    if self.do_profile:
                        self.profiler_results["dt_scaling_precompile_total_ms"] = total_precompile_ms
                        self.profiler_results["dt_scaling_precompile_avg_per_config_ms"] = avg_config_ms
                        self.profiler_results["dt_scaling_num_configs"] = len(unique_configs)
            except Exception as e:
                self._log.warning(f"[JAX] Failed to set up seq_len scheduler with dt scaling: {e}", exc_info=True)
                seq_len_sched = None
                jit_cache = {}

        for epoch in range(trainer.cfg.training.max_epochs):
            t0 = time.perf_counter()
            # Eagerly resolve loaders and log sizes (non-critical, just for logging)
            n_train, n_val = None, None
            try:
                n_train = len(train_loader)
            except Exception as e:
                self._log.debug(f"[JAX] Could not determine train batch count: {e}")
            try:
                n_val = len(val_loader)
            except Exception as e:
                self._log.debug(f"[JAX] Could not determine val batch count: {e}")
            self._log.info(f"[JAX] Epoch {epoch + 1} starting: train_batches={n_train} val_batches={n_val}")

            # Limits from profiler (mirrors Torch path semantics for quick runs)
            train_limit, val_limit = None, None
            if self.do_profile:
                tl = getattr(self.config.profiler, "num_train_batches", None)
                vl = getattr(self.config.profiler, "num_val_batches", None)
                train_limit = int(tl) if tl is not None else None
                val_limit = int(vl) if vl is not None else None

            # Apply sequence length scheduler if configured
            if seq_len_sched is not None:
                try:
                    if self.do_profile:
                        t0_rebuild = time.perf_counter()

                    _new_len = seq_len_sched.apply_on_epoch_start(epoch=epoch)
                    cfg_seq_len, cfg_dt = seq_len_sched.get_config_for_epoch(epoch)

                    # Switch to cached jitted functions if dt scaling is active
                    if jit_cache and (cfg_seq_len, cfg_dt) in jit_cache:
                        cached_params, cached_train_step, cached_eval_step = jit_cache[(cfg_seq_len, cfg_dt)]
                        # Update current params from cache (preserves trained weights)
                        # Only update topology-dependent structures, keep trained parameters
                        train_step = cached_train_step
                        eval_step = cached_eval_step
                        self._log.info(f"[JAX] Switched to cached JIT functions for seq_len={cfg_seq_len}, dt={cfg_dt:.6f}")

                    # Dataloader may need rebuilding; get fresh iterators
                    # This path is only valid for SOENDataModule; JAX loader is in-memory
                    if not isinstance(train_loader, JAXBatchIterator):
                        train_loader = data_module.train_dataloader()
                        val_loader = data_module.val_dataloader()

                    if self.do_profile:
                        t1_rebuild = time.perf_counter()
                        self.profiler_results[f"epoch_{epoch}_dataloader_rebuild_ms"] = (t1_rebuild - t0_rebuild) * 1000

                    self._log.info("[JAX] SeqLenScheduler set target_seq_len=%d, dt=%.6f for epoch %d", _new_len, cfg_dt, epoch)
                except Exception as e:
                    self._log.warning("[JAX] Failed to apply seq_len scheduler: %s", e)

            # Apply time pooling scale scheduler at epoch start
            if tp_sched is not None:
                try:
                    if self.do_profile:
                        t_start_tp = time.perf_counter()
                    new_scale = tp_sched.apply_on_epoch_start(trainer=trainer, epoch=epoch)
                    if self.do_profile:
                        t_end_tp = time.perf_counter()
                        self.profiler_results[f"epoch_{epoch}_time_pooling_callback_ms"] = (t_end_tp - t_start_tp) * 1000
                    self._log.info("[JAX] TimePoolingScaleScheduler scale=%.6f (epoch %d)", new_scale, epoch)
                except Exception as e:
                    self._log.warning("[JAX] Failed to apply time_pooling_scale_scheduler: %s", e)

            # Train phase
            if self.do_profile:
                t0_train = time.perf_counter()
            train_results = self._run_one_epoch(
                trainer=trainer,
                loader=train_loader,
                step_fn=train_step,
                params=params,
                opt_state=opt_state,
                limit=train_limit,
                epoch=epoch,
                phase="train",
                global_step=global_step,
                lw_schedulers=lw_schedulers,
                steps_per_epoch=(int(n_train) if isinstance(n_train, int) else None),
            )
            if self.do_profile:
                t1_train = time.perf_counter()
                self.profiler_results[f"epoch_{epoch}_train_ms"] = (t1_train - t0_train) * 1000
            params = train_results["params"]
            opt_state = train_results["opt_state"]
            global_step = train_results["global_step"]
            tl, ta, _ = train_results["loss"], train_results["acc"], train_results["num_batches"]
            train_metrics_avg = train_results["metrics"]

            # Eval phase
            if self.do_profile:
                t0_val = time.perf_counter()
            val_results = self._run_one_epoch(
                trainer=trainer,
                loader=val_loader,
                step_fn=eval_step,
                params=params,
                opt_state=None,
                limit=val_limit,
                epoch=epoch,
                phase="val",
                global_step=global_step,
            )
            if self.do_profile:
                t1_val = time.perf_counter()
                self.profiler_results[f"epoch_{epoch}_val_ms"] = (t1_val - t0_val) * 1000
            vl, va, _ = val_results["loss"], val_results["acc"], val_results["num_batches"]
            val_metrics_avg = val_results["metrics"]

            t1 = time.perf_counter()

            # Log epoch metrics (match PyTorch Lightning naming convention)
            step_for_log = max(1, global_step)
            log_dict = {
                "epoch": float(epoch),
                "train_loss_total_epoch": tl,  # Match PyTorch: training losses have _epoch suffix
                "val_loss_total": vl,  # Match PyTorch: validation losses have no _epoch suffix
                "epoch_time_ms": (t1 - t0) * 1000.0,
            }
            # Log learning rate (schedule-aware)
            current_lr = trainer.eval_lr(step_for_log)
            log_dict["callbacks_learning_rate"] = current_lr  # Match PyTorch: no slashes
            if self.is_classification:
                log_dict["train_accuracy"] = ta
                log_dict["val_accuracy"] = va
            # Add dynamic metrics (match PyTorch naming: no slashes, direct underscores)
            for name, value in train_metrics_avg.items():
                log_dict[f"train_{name}"] = value  # Changed from train/{name} to train_{name}
            for name, value in val_metrics_avg.items():
                log_dict[f"val_{name}"] = value  # Changed from val/{name} to val_{name}

            # Extract and log loss components to match PyTorch naming
            # PyTorch logs: train_loss_cross_entropy_epoch, train_loss_cross_entropy_w_epoch, etc.
            assert "loss_terms" in train_results, "train_results must contain 'loss_terms' key"
            assert "loss_terms_w" in train_results, "train_results must contain 'loss_terms_w' key"
            assert "loss_terms" in val_results, "val_results must contain 'loss_terms' key"
            assert "loss_terms_w" in val_results, "val_results must contain 'loss_terms_w' key"

            train_loss_terms = train_results.get("loss_terms", {})
            train_loss_terms_w = train_results.get("loss_terms_w", {})
            val_loss_terms = val_results.get("loss_terms", {})
            val_loss_terms_w = val_results.get("loss_terms_w", {})

            # Log training loss components with _epoch suffix (unweighted)
            for loss_name, loss_val in train_loss_terms.items():
                log_dict[f"train_loss_{loss_name}_epoch"] = loss_val

            # Log training loss components with _w_epoch suffix (weighted)
            for loss_name, loss_val in train_loss_terms_w.items():
                log_dict[f"train_loss_{loss_name}_w_epoch"] = loss_val

            # Log validation loss components without _epoch suffix (unweighted)
            for loss_name, loss_val in val_loss_terms.items():
                log_dict[f"val_loss_{loss_name}"] = loss_val

            # Log validation loss components with _w suffix (weighted, no _epoch for validation)
            for loss_name, loss_val in val_loss_terms_w.items():
                log_dict[f"val_loss_{loss_name}_w"] = loss_val

            self.logger_adapter.log_metrics(log_dict, step=step_for_log)

            # Build log message
            parts = [f"train_loss={tl:.6f}"]
            if self.is_classification:
                parts.append(f"train_accuracy={ta:.6f}")
            for name, value in train_metrics_avg.items():
                parts.append(f"train_{name}={value:.6f}")
            parts.append(f"val_loss={vl:.6f}")
            if self.is_classification:
                parts.append(f"val_accuracy={va:.6f}")
            for name, value in val_metrics_avg.items():
                parts.append(f"val_{name}={value:.6f}")
            parts.append(f"time_ms={(t1 - t0) * 1000.0:.1f}")
            self._log.info(f"[JAX] Epoch {epoch + 1} done: {' '.join(parts)}")

            # Checkpoint management
            if self.do_profile:
                t0_ckpt = time.perf_counter()
            # Log internal connection stats before checkpointing
            self._log_internal_connection_stats(trainer, params, epoch)

            self._manage_checkpoints(trainer, params, opt_state, epoch, global_step, vl)
            if self.do_profile:
                t1_ckpt = time.perf_counter()
                self.profiler_results[f"epoch_{epoch}_checkpoint_ms"] = (t1_ckpt - t0_ckpt) * 1000

            # Log artifacts to MLflow (async)
            if self.artifact_uploader is not None:
                # Enqueue checkpoints if they exist
                # Note: .pkl is saved by save_last/save_topk, .soen by _save_soen_checkpoint
                # We need to glob because names vary (epoch=XX-val_loss=YY.soen)

                # Log files are always safe to upload
                log_dir = Path(self.repeat_log_dir)
                for log_file in log_dir.glob("*.log"):
                    self.artifact_uploader.enqueue(log_file)

                # Checkpoints - rely on what's currently in directory
                # Only enqueue if mlflow_log_artifacts is True
                mlflow_log_artifacts = getattr(getattr(self.config, "logging", object()), "mlflow_log_artifacts", True)
                if mlflow_log_artifacts:
                    ckpt_dir = Path(self.repeat_ckpt_dir)
                    # Enqueue .soen and .pkl files
                    # The uploader will skip if file was deleted before processing
                    for p in ckpt_dir.glob("*.soen"):
                        self.artifact_uploader.enqueue(p)
                    for p in ckpt_dir.glob("*.pkl"):
                        self.artifact_uploader.enqueue(p)

            # Re-apply QAT at epoch start if configured
            if qat_helper is not None and bool(qat_helper.update_on_train_epoch_start):
                try:
                    if self.do_profile:
                        t_start_qat_epoch = time.perf_counter()
                    qat_helper.enable(trainer)
                    if self.do_profile:
                        t_end_qat_epoch = time.perf_counter()
                        self.profiler_results[f"epoch_{epoch}_qat_callback_ms"] = (t_end_qat_epoch - t_start_qat_epoch) * 1000
                except Exception as e:
                    # QAT enable failure should be logged but training can continue
                    self._log.warning(f"[JAX] Failed to re-enable QAT at epoch {epoch}: {e}")

            # Early stopping check
            if early_patience is not None:
                if vl < best_val:
                    best_val = vl
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= max(1, early_patience):
                    self._log.info("[JAX] Early stopping triggered: no val_loss improvement for %d epochs", early_patience)
                    break

        # Disable QAT at end
        if qat_helper is not None:
            try:
                qat_helper.disable(trainer)
            except Exception as e:
                # QAT disable failure should be logged but not block training completion
                self._log.warning(f"[JAX] Failed to disable QAT at end of training: {e}")

        # --- Testing phase (mirror Torch path) ---
        has_test_attr = hasattr(data_module, "test_dataloader")
        test_loader_obj = data_module.test_dataloader() if has_test_attr else None
        if test_loader_obj is not None:
            # Ensure test dataloaders are set up (Torch Lightning usually calls setup("test"))
            if hasattr(data_module, "setup"):
                data_module.setup(stage="test")
                test_loader_obj = data_module.test_dataloader()
            if self.do_profile:
                t0_test = time.perf_counter()
            test_results = self._run_one_epoch(
                trainer=trainer,
                loader=test_loader_obj,
                step_fn=eval_step,
                params=params,
                opt_state=None,
                limit=None,
                epoch=trainer.cfg.training.max_epochs,
                phase="test",
                global_step=global_step,
            )
            if self.do_profile:
                t1_test = time.perf_counter()
                self.profiler_results["test_epoch_ms"] = (t1_test - t0_test) * 1000
            tloss, tacc = test_results["loss"], test_results["acc"]
            tmetrics = test_results["metrics"]
            step_for_log = max(1, global_step)
            log_dict = {"test_loss": float(tloss)}  # Match PyTorch naming (no slashes)
            if self.is_classification:
                log_dict["test_accuracy"] = float(tacc)  # Match PyTorch: test_accuracy not test_acc
            for name, value in tmetrics.items():
                log_dict[f"test_{name}"] = float(value)  # Match PyTorch: test_{name} not test/{name}
            self.logger_adapter.log_metrics(log_dict, step=step_for_log)

            parts = [f"test_loss={tloss:.6f}"]
            for name, value in tmetrics.items():
                parts.append(f"test_{name}={value:.6f}")
            self._log.info(f"[JAX] Test: {' '.join(parts)}")

        if self.do_profile:
            t_fit_end = time.perf_counter()
            self.profiler_results["total_fit_ms"] = (t_fit_end - t_fit_start) * 1000

            # Collect memory and GPU stats
            self._collect_memory_and_gpu_stats()

            # Log summary report
            self._log.info("\n--- JAX Runner Profiling Report ---")
            for key, value in sorted(self.profiler_results.items()):
                if isinstance(value, (int, float)):
                    unit = "ms" if "ms" in key or "_step" in key or "_compute" in key else ""
                    unit = "GB" if "memory" in key.lower() else unit
                    unit = "%" if "utilization" in key.lower() else unit
                    self._log.info(f"  {key}: {value:.2f}{unit}")
                else:
                    self._log.info(f"  {key}: {value}")
            self._log.info("------------------------------------")

            # Export to JSON for easy analysis
            try:
                import json

                profile_json_path = self.repeat_log_dir / "profiler_results.json"
                # Convert to dict with float values (JSON serializable)
                profile_dict = {k: float(v) for k, v in self.profiler_results.items()}
                with open(profile_json_path, "w") as f:
                    json.dump(profile_dict, f, indent=2)
                self._log.info(f"Profiling results saved to: {profile_json_path}")
            except Exception as e:
                self._log.warning(f"Failed to save profiling results to JSON: {e}")

        # Ensure any background checkpoint save finished before returning.
        # This avoids races where callers/tests check for `last.soen` immediately.
        t = getattr(self, "_checkpoint_thread", None)
        if t is not None:
            if t.is_alive():
                t.join()
            self._checkpoint_thread = None
            if self._checkpoint_thread_error is not None:
                err = self._checkpoint_thread_error
                self._checkpoint_thread_error = None
                raise RuntimeError(f"Checkpoint save failed: {err}") from err

        return self.best_ckpt_path


__all__ = ["JaxRunner"]
