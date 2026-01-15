"""Metrics tracking callback for SOEN model training.

This module provides a callback for tracking and logging metrics during training.
"""

import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn  # type: ignore[attr-defined]
import torch

# Import the registry containing metric implementations
from soen_toolkit.analysis import GradientStatsCollector
from soen_toolkit.training.callbacks.metrics import METRICS_REGISTRY

logger = logging.getLogger(__name__)


class MetricsTracker(Callback):
    """Callback for tracking and logging metrics during training.

    This callback logs metrics to TensorBoard and maintains history of important metrics.

    Attributes:
        config: Logging configuration
        log_freq: Frequency of logging metrics within each epoch
        track_gradients: Whether to track gradients of model parameters
        batch_metrics: Metrics to track at batch level
        metrics_history: History of metrics for analysis

    """

    def __init__(self, config, debug: bool = False) -> None:
        """Initialize MetricsTracker.

        Args:
            config: Configuration object containing logging settings
            debug: Enable verbose logging

        """
        super().__init__()
        self.config = config
        self.log_freq = config.logging.log_freq
        self.debug = debug
        self.track_gradients = config.logging.log_gradients
        self.track_layer_params = config.logging.track_layer_params
        self.layer_params_log_freq = config.logging.log_freq  # Use single log_freq for consistency
        self._gradient_stats_cfg = getattr(config.logging, "gradient_stats", None)
        self._gradient_stats_collector: GradientStatsCollector | None = None
        self._gradient_stats_log_every_n = 1
        if self._gradient_stats_cfg and getattr(self._gradient_stats_cfg, "active", False):
            self._gradient_stats_collector = GradientStatsCollector(
                track_per_step=not bool(getattr(self._gradient_stats_cfg, "summary_only", False)),
                max_steps_per_param=getattr(self._gradient_stats_cfg, "max_steps_per_param", None),
                include_patterns=list(getattr(self._gradient_stats_cfg, "include", []) or []),
                exclude_patterns=list(getattr(self._gradient_stats_cfg, "exclude", []) or []),
            )
            self._gradient_stats_log_every_n = max(1, int(getattr(self._gradient_stats_cfg, "log_every_n_steps", 1)))

        self.batch_metrics = config.logging.log_batch_metrics
        self._layer_params_logged_once = False  # Flag to ensure layer info is only logged once

        # Metrics configured by the user in the YAML file. Accept strings or dicts.
        # Dicts are normalized to a metric name if possible; otherwise skipped with a warning.
        raw_metrics = config.logging.metrics
        normalized: list[str] = []
        if isinstance(raw_metrics, (list, tuple)):
            for i, item in enumerate(raw_metrics):
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    # Try common shapes: {name: params} or {name: True}
                    if len(item) == 1:
                        key = next(iter(item.keys()))
                        if isinstance(key, str):
                            normalized.append(key)
                            continue
                    # Or explicit {name: 'metric_name', ...}
                    name = item.get("name") if isinstance(item, dict) else None
                    if isinstance(name, str):
                        normalized.append(name)
                    else:
                        rank_zero_warn(f"[MetricsTracker] Skipping unsupported metric spec at index {i}: {item}")
                else:
                    rank_zero_warn(f"[MetricsTracker] Skipping unsupported metric entry at index {i}: type {type(item)}")
        elif isinstance(raw_metrics, str):
            normalized = [raw_metrics]
        else:
            rank_zero_warn(f"[MetricsTracker] 'logging.metrics' should be a list or string; got {type(raw_metrics)}. Using empty list.")

        self.metrics_to_track: list[str] = normalized

        # History containers – automatically extended based on configured metrics
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch": [],
        }

        # Add history keys for each requested metric (train/val/test)
        for metric in self.metrics_to_track:
            if metric in {"loss", "input_seq_len"}:
                continue
            self.metrics_history[f"train_{metric}"] = []
            self.metrics_history[f"val_{metric}"] = []
            self.metrics_history[f"test_{metric}"] = []

        if self.debug:
            rank_zero_info(
                f"[MetricsTracker] Initialized with log_freq={self.log_freq}, track_gradients={self.track_gradients}, track_layer_params={self.track_layer_params}, metrics={self.metrics_to_track}",
            )

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the beginning of training.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module

        """
        # Log GPU availability and device info to MLflow
        if trainer.is_global_zero:
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                device_count = torch.cuda.device_count() if cuda_available else 0

                # Log to console/file
                rank_zero_info(f"[PyTorch] CUDA available: {cuda_available}")
                if cuda_available:
                    rank_zero_info(f"[PyTorch] GPU devices: {device_count}")
                    for i in range(device_count):
                        rank_zero_info(f"[PyTorch]   GPU {i}: {torch.cuda.get_device_name(i)}")

                # Log to MLflow if available
                if bool(getattr(self.config.logging, "mlflow_active", False)):
                    logger_objs = []
                    if hasattr(trainer, "loggers") and trainer.loggers:
                        logger_objs = list(trainer.loggers)
                    elif hasattr(trainer, "logger") and trainer.logger is not None:
                        logger_objs = [trainer.logger]

                    for lg in logger_objs:
                        try:
                            if hasattr(lg, "experiment") and hasattr(lg, "run_id"):
                                exp = lg.experiment
                                run_id = lg.run_id
                                metrics = {
                                    "system/pytorch_cuda_available": 1 if cuda_available else 0,
                                    "system/pytorch_gpu_count": device_count,
                                    "system/using_gpu": 1 if cuda_available else 0,
                                }
                                params = {
                                    "system/pytorch_backend": "pytorch",
                                }
                                if cuda_available and device_count > 0:
                                    gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
                                    params["system/pytorch_gpu_names"] = ",".join(gpu_names)

                                exp.log_metrics(run_id, metrics, step=0)
                                exp.log_params(run_id, params)
                        except Exception as e:
                            logger.debug(f"Failed to log GPU info to MLflow: {e}")
            except Exception as e:
                logger.debug(f"Failed to detect PyTorch GPU: {e}")

        # Log model architecture and hyperparameters
        if trainer.is_global_zero and self.config.logging.log_model:
            params_dict = {}

            total_params = sum(p.numel() for p in pl_module.parameters())
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            params_dict["total_parameters"] = total_params
            params_dict["trainable_parameters"] = trainable_params

            if self.debug:
                rank_zero_info(
                    f"[MetricsTracker] Model parameters: {trainable_params:,}/{total_params:,} trainable/total",
                )

            logger_exp = getattr(trainer.logger, "experiment", None)
            if logger_exp is not None:
                for k, v in params_dict.items():
                    logger_exp.add_text(f"model/{k}", str(v), global_step=0)

    def on_train_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.track_layer_params and batch_idx % self.layer_params_log_freq == 0:
            self._log_layer_params(pl_module, trainer.global_step)

        if not self.batch_metrics:
            return

        if batch_idx % self.log_freq != 0:
            return

        lr = self._get_lr(pl_module)
        metrics_dict: dict[str, float] = {"learning_rate": lr}

        try:
            # Prefer reusing outputs from training_step to avoid an extra forward
            preds = None
            targets_for_metrics = None
            if isinstance(outputs, dict):
                preds = outputs.get("preds")
                # Prefer sequence targets if present (AR mode might stash them)
                targets_for_metrics = outputs.get("targets")
            if preds is None or targets_for_metrics is None:
                # Fallback to a lightweight forward if outputs are missing (rare)
                x, y = batch
                with torch.no_grad():
                    preds, _, _ = pl_module(x)
                targets_for_metrics = y

            batch_metrics = self._batch_calculate_metrics(preds, targets_for_metrics, prefix="train")
            metrics_dict.update(batch_metrics)
        except Exception as e:
            if self.debug:
                rank_zero_warn(f"[MetricsTracker] Could not compute additional train metrics: {e}")

        if isinstance(outputs, dict) and "loss" in outputs:
            metrics_dict["train_loss_step"] = outputs["loss"].item()

        # Log gradients if configured - NOW WITH STEP
        if self.track_gradients:
            self._log_gradients(pl_module, trainer.global_step)  # Pass global_step

        if trainer.is_global_zero:
            try:
                if metrics_dict and trainer.logger:
                    trainer.logger.log_metrics(metrics_dict, step=trainer.global_step)
            except Exception as e:
                if self.debug:
                    rank_zero_warn(f"[MetricsTracker] Warning: Failed to log batch metrics: {e}")

    def _log_gradients(self, pl_module: pl.LightningModule, step: int) -> None:
        """Log gradients of model parameters to TensorBoard."""
        try:
            logger_exp = getattr(pl_module.logger, "experiment", None)
            if logger_exp is None:
                return

            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    logger_exp.add_histogram(
                        f"gradients/{name}",
                        param.grad.detach().cpu(),
                        global_step=step,
                    )
                    if (
                        self._gradient_stats_collector is not None
                        and step % self._gradient_stats_log_every_n == 0
                    ):
                        self._gradient_stats_collector.record(name, param.grad, step)
        except Exception as e:
            if self.debug:
                rank_zero_info(f"[MetricsTracker] Warning: Failed to log gradients: {e}")

    def _log_layer_params(self, pl_module: pl.LightningModule, step: int) -> None:
        """Log histograms of layer-specific parameters and their gradients (excluding connections)."""
        try:
            logger_exp = getattr(pl_module.logger, "experiment", None)
            if logger_exp is None:
                return

            soen_wrapper = getattr(pl_module, "model", None)
            if soen_wrapper is None:
                return

            soen_model = soen_wrapper

            layers = getattr(soen_model, "layers", [])
            layer_cfgs = getattr(soen_model, "layers_config", [])

            # Only log layer structure info once at the beginning
            first_time_logging = not self._layer_params_logged_once
            if first_time_logging:
                if self.debug:
                    rank_zero_info(f"[MetricsTracker] Found {len(layers)} layers to track parameters for")
                self._layer_params_logged_once = True

            logged_params_count = 0
            for idx, layer in enumerate(layers):
                layer_id = idx
                if idx < len(layer_cfgs):
                    layer_id = layer_cfgs[idx].layer_id

                layer_params = list(layer.named_parameters())

                # Only log layer parameter names once at the beginning
                if first_time_logging and self.debug:
                    param_names = [name for name, _ in layer_params]
                    rank_zero_info(f"[MetricsTracker] Layer {layer_id} has parameters: {param_names}")

                for name, param in layer_params:
                    # Skip internal_J parameters as they're covered by track_connections
                    if name.startswith("internal_J") or name == "internal_J":
                        # Only log this skip message once at the beginning
                        if first_time_logging and self.debug:
                            rank_zero_info(f"[MetricsTracker] Skipping {name} (covered by track_connections)")
                        continue

                    if param.requires_grad:
                        logger_exp.add_histogram(
                            f"layer_params/layer{layer_id}/{name}",
                            param.detach().cpu(),
                            global_step=step,
                        )
                        logged_params_count += 1

                        if param.grad is not None:
                            logger_exp.add_histogram(
                                f"layer_gradients/layer{layer_id}/{name}",
                                param.grad.detach().cpu(),
                                global_step=step,
                            )

            # Only show the warning once at the beginning
            if logged_params_count == 0 and first_time_logging:
                rank_zero_info("[MetricsTracker] Warning: No layer parameters found to log. This might indicate an issue with the model structure or parameter names.")

        except Exception as e:
            rank_zero_info(f"[MetricsTracker] Warning: Failed to log layer parameters: {e}")
            if self.debug:
                import traceback

                rank_zero_info(f"[MetricsTracker] Full traceback: {traceback.format_exc()}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of each training epoch.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module

        """
        # Track metrics
        lr = self._get_lr(pl_module)

        # Store in history
        self.metrics_history["learning_rate"].append(lr)
        self.metrics_history["epoch"].append(trainer.current_epoch)

        if "train_loss" in trainer.callback_metrics:
            self.metrics_history["train_loss"].append(trainer.callback_metrics["train_loss"].item())

        # Log learning rate to TensorBoard under callbacks section
        if lr is not None:
            pl_module.log(
                "callbacks/learning_rate",
                lr,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # Log learning rate to console
        if self.debug:
            rank_zero_info(f"[MetricsTracker] Epoch {trainer.current_epoch}: LR = {lr:.2e}")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of each validation epoch.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module

        """
        metrics_to_log = {}

        # Explicitly log the current epoch number using the global_step
        metrics_to_log["epoch"] = float(trainer.current_epoch)

        # Store val_loss in history if it exists (for internal tracking if needed)
        if "val_loss" in trainer.callback_metrics:
            try:
                self.metrics_history["val_loss"].append(trainer.callback_metrics["val_loss"].item())
            except Exception as e:
                if self.debug:
                    rank_zero_warn(f"[MetricsTracker] Could not store val_loss in history: {e}")

        # The LightningModule is expected to handle logging of its own validation metrics (e.g., val_loss, val_accuracy)
        # via `self.log(..., on_epoch=True)`. This callback will now primarily log the 'epoch' scalar.
        # If there are other metrics this callback uniquely computes and should log at epoch end, add them to metrics_to_log here.

        if trainer.is_global_zero and trainer.logger and metrics_to_log:
            try:
                # Log only the metrics explicitly prepared in this callback (e.g., 'epoch')
                trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
            except Exception as e:
                if self.debug:
                    rank_zero_warn(f"[MetricsTracker] Failed to log epoch-level metrics: {e}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Export collected gradient statistics once training concludes."""
        if self._gradient_stats_collector is None or not getattr(self._gradient_stats_cfg, "active", False):
            return
        try:
            destination = self._export_gradient_stats(trainer)
            if destination:
                rank_zero_info(f"[MetricsTracker] Gradient stats saved to {destination}")
        except Exception as exc:
            if self.debug:
                rank_zero_warn(f"[MetricsTracker] Failed to export gradient stats: {exc}")

    def _export_gradient_stats(self, trainer: pl.Trainer) -> Path | None:
        cfg = self._gradient_stats_cfg
        if cfg is None or self._gradient_stats_collector is None:
            return None
        log_dir = self._resolve_log_dir(trainer)
        if log_dir is None:
            return None
        filename = getattr(cfg, "output_filename", None) or "gradient_stats.json"
        destination = log_dir / filename
        self._gradient_stats_collector.save(destination)
        return destination

    @staticmethod
    def _resolve_log_dir(trainer: pl.Trainer) -> Path | None:
        candidates: list[Any] = []
        if hasattr(trainer, "loggers") and trainer.loggers:
            candidates.extend(trainer.loggers)
        elif hasattr(trainer, "logger") and trainer.logger is not None:
            candidates.append(trainer.logger)
        for logger_obj in candidates:
            for attr in ("log_dir", "save_dir"):
                value = getattr(logger_obj, attr, None)
                if value:
                    return Path(value)
            experiment = getattr(logger_obj, "experiment", None)
            if experiment is not None:
                for attr in ("log_dir",):
                    value = getattr(experiment, attr, None)
                    if value:
                        return Path(value)
        default_dir = getattr(trainer, "default_root_dir", None)
        return Path(default_dir) if default_dir else None

    def _get_lr(self, pl_module: pl.LightningModule) -> float:
        """Get current learning rate from optimizer.

        Args:
            pl_module: PyTorch Lightning module

        Returns:
            Current learning rate

        """
        optimizers = pl_module.optimizers()
        if not optimizers:
            return 0.0

        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers

        if not hasattr(optimizer, "param_groups") or len(optimizer.param_groups) == 0:
            return 0.0

        return optimizer.param_groups[0].get("lr", 0.0)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _batch_calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, prefix: str) -> dict[str, float]:
        """Compute all configured metrics for a single batch.

        Parameters
        ----------
        outputs : torch.Tensor
            Raw outputs of the model (logits or probabilities).
        targets : torch.Tensor
            Ground-truth class indices.
        prefix : str
            One of ``"train"``, ``"val"``, ``"test"`` – prepended to metric name.

        """
        metric_values: dict[str, float] = {}

        # Determine AR mode primarily from config; fall back to tensor rank only if unavailable
        try:
            # Prefer the explicit training config flag which is robust to output processing
            is_autoregressive = bool(getattr(self.config.training, "autoregressive", False))
        except Exception:
            # Fallback heuristic: sequence logits are typically 3D
            is_autoregressive = outputs.dim() == 3
        universal_metrics = {"perplexity", "bits_per_character"}

        for metric_name_from_config in self.metrics_to_track:
            # Loss is already handled by Lightning, skip here
            if metric_name_from_config in {"loss", "input_seq_len"}:
                continue

            metric_to_compute = metric_name_from_config

            # --- START: Smart metric selection logic ---

            # If in AR mode and user asks for 'accuracy', swap to 'autoregressive_accuracy'
            if is_autoregressive and metric_name_from_config == "accuracy":
                metric_to_compute = "autoregressive_accuracy"

            # Skip non-AR metrics in AR mode, unless they are universal
            if is_autoregressive and not (metric_to_compute.startswith("autoregressive") or metric_to_compute in universal_metrics):
                if self.debug:
                    # Silence noisy console messages in normal runs; rely on logs
                    pass
                continue

            # Skip AR-specific metrics in non-AR mode
            if not is_autoregressive and metric_to_compute.startswith("autoregressive"):
                if self.debug:
                    pass
                continue

            # --- END: Smart metric selection logic ---

            try:
                # Handle top-k metrics specified as e.g. "top_5" or "topk"
                if metric_to_compute.startswith("top_") and metric_to_compute.split("_")[1].isdigit():
                    k_val = int(metric_to_compute.split("_")[1])
                    metric_fn = METRICS_REGISTRY.get("top_k_accuracy")
                    if metric_fn is None:
                        rank_zero_warn(f"Metric implementation for top_k_accuracy not found in registry – skipping {metric_to_compute}.")
                        continue
                    metric_value = metric_fn(outputs, targets, k=k_val)
                else:
                    metric_fn = METRICS_REGISTRY.get(metric_to_compute)
                    if metric_fn is None:
                        rank_zero_warn(f"Metric '{metric_to_compute}' not found in registry – skipping.")
                        continue
                    metric_value = metric_fn(outputs, targets)

                # Detach to avoid autograd graph leaks and move to CPU for logging
                metric_value_scalar = metric_value.detach().cpu().item()
                metric_key = f"{prefix}_{metric_name_from_config}"  # Log with the original name from config
                metric_values[metric_key] = metric_value_scalar

                # Save to history
                if metric_key in self.metrics_history:
                    self.metrics_history[metric_key].append(metric_value_scalar)

            except Exception as e:
                rank_zero_warn(f"Failed to compute metric '{metric_to_compute}' on batch: {e}")

        return metric_values

    # ---------------------------------------------------------------------
    # Validation / Test batch hooks – similar logic as training
    # ---------------------------------------------------------------------

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: dict[str, Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:  # type: ignore
        # if not self.batch_metrics:
        #     return

        # # Compute metrics for this validation batch
        # x, y = batch  # type: ignore
        # try:
        #     with torch.no_grad():
        #         y_hat, _, _ = pl_module(x)
        #     metrics_dict = self._batch_calculate_metrics(y_hat, y, prefix='val')
        #     if metrics_dict and trainer.is_global_zero and trainer.logger:
        #         trainer.logger.log_metrics(metrics_dict, step=trainer.global_step)
        # except Exception as e:
        #     if self.debug:
        #         rank_zero_warn(f"[MetricsTracker] Failed to compute/ log validation batch metrics: {e}")
        pass

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: dict[str, Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:  # type: ignore[override]
        if not self.batch_metrics:
            return

        # Compute metrics for this test batch
        x, y = batch
        try:
            with torch.no_grad():
                y_hat, _, _ = pl_module(x)
            metrics_dict = self._batch_calculate_metrics(y_hat, y, prefix="test")
            if metrics_dict and trainer.is_global_zero and trainer.logger:
                trainer.logger.log_metrics(metrics_dict, step=trainer.global_step)
        except Exception as e:
            if self.debug:
                rank_zero_warn(f"[MetricsTracker] Failed to compute/ log test batch metrics: {e}")
