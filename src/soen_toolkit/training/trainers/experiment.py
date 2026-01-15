# FILEPATH: src/soen_toolkit/training/trainers/experiment.py

"""Experiment runner for SOEN model training.

This module provides an experiment runner for training SOEN models with PyTorch Lightning.
"""

import contextlib
import datetime
import logging
import os
from pathlib import Path
import random
import threading
from typing import Any
import warnings

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# Import MLFlowLogger optionally; code paths guard usage by config
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
import torch

from soen_toolkit.training.callbacks import SCHEDULER_REGISTRY, MetricsTracker
from soen_toolkit.training.callbacks.checkpointing import InitialModelSaver, SOENModelCheckpoint


class SimpleProgressCallback(pl.Callback):
    """Simple text-based progress callback that prints to stdout without widgets."""

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._train_batch_count = 0
        self._epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        import time
        self._train_batch_count = 0
        self._epoch_start_time = time.time()
        total_batches = len(trainer.train_dataloader) if trainer.train_dataloader else "?"
        print(f"\n[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] Starting training ({total_batches} batches)...")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._train_batch_count += 1
        if self._train_batch_count % self.log_every_n_steps == 0:
            loss = outputs.get("loss", None) if isinstance(outputs, dict) else outputs
            loss_str = f"{loss.item():.4f}" if loss is not None and hasattr(loss, "item") else "N/A"
            print(f"  [Step {self._train_batch_count}] loss: {loss_str}")

    def on_train_epoch_end(self, trainer, pl_module):
        import time
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss/total", metrics.get("train_loss", "N/A"))
        if hasattr(train_loss, "item"):
            train_loss = f"{train_loss.item():.4f}"
        print(f"  [Epoch {trainer.current_epoch + 1}] Train complete in {elapsed:.1f}s | train_loss: {train_loss}")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss/total", metrics.get("val_loss", "N/A"))
        val_acc = metrics.get("val_accuracy", "N/A")
        if hasattr(val_loss, "item"):
            val_loss = f"{val_loss.item():.4f}"
        if hasattr(val_acc, "item"):
            val_acc = f"{val_acc.item():.4f}"
        print(f"  [Epoch {trainer.current_epoch + 1}] Validation | val_loss: {val_loss} | val_accuracy: {val_acc}")
from soen_toolkit.training.configs import ExperimentConfig, load_config
from soen_toolkit.training.data import SOENDataModule
from soen_toolkit.training.models import SOENLightningModule
from soen_toolkit.training.utils.s3_uri import (
    is_s3_uri as _is_s3_uri,
    looks_like_file_uri as _s3_looks_file,
    split_s3_uri as _split_s3_uri,
)

logger = logging.getLogger(__name__)


class SafeMLFlowLogger(MLFlowLogger):
    """MLflow logger that sanitizes metric keys to avoid filesystem issues.

    Replaces '/' with '_' in metric names so that MLflow file store does not
    create nested directories like metrics/val_loss/total which can conflict
    with a plain 'val_loss' metric later.

    Also handles deleted MLflow runs gracefully by logging a warning and skipping
    metrics logging instead of crashing.
    """

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:  # type: ignore[override]
        try:
            safe_metrics = {}
            for k, v in metrics.items():
                # MLflow treats '/' as path separators in file store
                safe_key = str(k).replace("/", "_")
                safe_metrics[safe_key] = v
            super().log_metrics(safe_metrics, step=step)
        except Exception as e:
            import logging

            _logger = logging.getLogger(__name__)
            # Check if this is a deleted run error
            error_msg = str(e)
            if "deleted" in error_msg.lower() or "must be in the 'active' state" in error_msg:
                _logger.warning(f"MLflow run {getattr(self, 'run_id', 'unknown')} has been deleted. Skipping MLflow metric logging for this run. Consider creating a new run.")
                return
            # Fallback without transformation if anything unexpected happens
            try:
                super().log_metrics(metrics, step=step)
            except Exception:
                # If even fallback fails, log warning and continue
                _logger.warning(f"Failed to log metrics to MLflow: {e}. Continuing training without MLflow logging.")


class ExperimentRunner:
    """Runner for SOEN model training experiments.

    This class handles the setup and execution of SOEN model training experiments,
    including model creation, data loading, and training workflow.

    Attributes:
        config: Experiment configuration
        repeat: Current repeat number
        seed: Random seed for reproducibility
        checkpoints_dir: Directory for checkpoints
        logs_dir: Directory for logs

    """

    def _resolve_base_dir(
        self,
        user_specified_path: Path | None,
        script_dir: Path,
        default_parent_stem: Path,
        default_leaf_name: str,
        dir_type_for_log: str,
    ) -> Path:
        """Resolve base directory for logs or checkpoints without timestamping."""
        if user_specified_path is not None:
            base_path = user_specified_path
            if not base_path.is_absolute():
                if self.project_root_dir:
                    base_path = self.project_root_dir / base_path
                    logger.info(f"Resolving user-specified relative path for {dir_type_for_log} against project root: {self.project_root_dir}")
                else:
                    base_path = script_dir / base_path
                    logger.warning(f"Project root not specified. Resolving user-specified relative path for {dir_type_for_log} against script directory: {script_dir}")

            project_name_str = self.config.logging.project_name or "default_project"
            path_to_use = base_path / f"project_{project_name_str}"
            logger.info(f"Using user-specified path with project structure for {dir_type_for_log} directory: {path_to_use}")
        else:
            if default_leaf_name:
                path_to_use = script_dir / default_parent_stem / default_leaf_name
            else:
                path_to_use = script_dir / default_parent_stem
            logger.info(f"Using default path for {dir_type_for_log} directory: {path_to_use}")

        try:
            path_to_use.mkdir(parents=True, exist_ok=True)
            return path_to_use
        except Exception as e:
            # Fallback: current working directory to ensure a valid, writable location
            fallback_root = Path.cwd().resolve()
            project_name_str = self.config.logging.project_name or "default_project"
            fallback = fallback_root / "experiments" / f"project_{project_name_str}"
            logger.warning(f"Failed to create {dir_type_for_log} directory at {path_to_use} ({e}). Falling back to {fallback}")
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _add_experiment_subdir(self, base_dir: Path) -> Path:
        """Return directory for the experiment under the base path."""
        exp_name = self.config.logging.experiment_name or "default_experiment"
        final_dir = base_dir / f"experiment_{exp_name}"
        final_dir.mkdir(parents=True, exist_ok=True)
        return final_dir

    def _add_group_subdir(self, base_dir: Path) -> Path:
        """Return directory for the experiment group under the base path."""
        group_name = self.config.logging.group_name or "default_group"
        final_dir = base_dir / f"group_{group_name}"
        final_dir.mkdir(parents=True, exist_ok=True)
        return final_dir

    def _prepare_repeat_subdirs(self, repeat: int) -> tuple[Path, Path, str]:
        """Create unique subdirectories for this repeat and return them."""
        base_name = f"repeat_{repeat}"
        log_dir = self.logs_dir / base_name
        ckpt_dir = self.checkpoints_dir / base_name
        if log_dir.exists() or ckpt_dir.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp}"
            log_dir = self.logs_dir / base_name
            ckpt_dir = self.checkpoints_dir / base_name
            logger.info(f"Repeat directory exists, using timestamped name: {base_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return log_dir, ckpt_dir, base_name

    def __init__(self, config: ExperimentConfig, script_dir: Path, project_root_dir: Path | None = None) -> None:
        """Initialize ExperimentRunner.

        Args:
            config: Experiment configuration
            script_dir: The absolute path to the directory containing the executing script.
            project_root_dir: Optional. The absolute path to the project's root directory.
                              Used for resolving relative paths in the config if provided.

        """
        self.config = config
        self.repeat = 0
        self.base_seed = config.seed
        self.project_root_dir = project_root_dir  # Store project_root_dir
        # Placeholder for instance state if needed in future

        # Helper to compute S3-relative keys starting at experiment/group/... by
        # stripping any absolute path and the local project_*/ prefix if present.
        def _rel_from_project_base(path: Path) -> Path:
            parts = list(path.parts)
            try:
                # Prefer exact local project base
                if hasattr(self, "project_base_dir") and self.project_base_dir is not None:
                    return path.relative_to(self.project_base_dir)
            except Exception:
                pass
            # Heuristic fallback: strip up to and including the first folder that starts with 'project_'
            try:
                for idx, part in enumerate(parts):
                    if part.startswith("project_"):
                        suffix = parts[idx + 1 :]
                        if suffix:
                            from pathlib import Path as _P

                            return _P(*suffix)
                        break
            except Exception:
                pass
            # Last resort: use only the leaf directory name
            return Path(path.name)

        # Bind helper to instance for reuse
        self._rel_from_project_base = _rel_from_project_base

        # Helper: determine if S3 should mirror-delete by default.
        # If logging.s3_mirror_delete is explicitly set, respect it.
        # Otherwise: prune when not saving all epochs (i.e., top-k mode), keep when saving all.
        def _should_mirror_delete() -> bool:
            try:
                explicit = getattr(self.config.logging, "s3_mirror_delete", None)
                if explicit is not None:
                    return bool(explicit)
            except Exception:
                pass
            save_all = bool(getattr(self.config.training, "checkpoint_save_all_epochs", False))
            return not save_all

        self._should_mirror_delete = _should_mirror_delete

        # --- Determine experiment parent directory stem ---
        project_name_str = self.config.logging.project_name or "default_project"
        default_parent_stem_for_paths = Path("experiments") / f"project_{project_name_str}"

        # --- Resolve base project directory ---
        project_dir = self.config.logging.project_dir
        project_dir_path = Path(project_dir) if isinstance(project_dir, str) else project_dir
        project_base = self._resolve_base_dir(
            user_specified_path=project_dir_path,
            script_dir=script_dir,
            default_parent_stem=default_parent_stem_for_paths,
            default_leaf_name="",  # No additional leaf name needed
            dir_type_for_log="Project",
        )
        # Treat this as the local "project root" for uploads so S3 keys begin at
        # project_{name}/experiment_{name}/group_{name}/...
        self.project_base_dir = project_base
        # Cache project folder name for S3 key prefixing
        self.project_dirname = f"project_{project_name_str}"

        # --- Create Experiment and Group directories, then place logs & checkpoints UNDER group ---
        exp_dir = self._add_experiment_subdir(project_base)
        group_dir = self._add_group_subdir(exp_dir)

        # Sibling directories under the group
        logs_group = group_dir / "logs"
        checkpoints_group = group_dir / "checkpoints"
        logs_group.mkdir(parents=True, exist_ok=True)
        checkpoints_group.mkdir(parents=True, exist_ok=True)

        self.logs_dir = logs_group
        self.checkpoints_dir = checkpoints_group
        # Remember the base directory to mirror for S3 uploads (the user-specified project_dir if provided)
        # This ensures S3 keys start at `project_{project_name}/...` and avoids duplicating absolute prefixes.
        self._s3_mirror_root = project_base.parent

        # --- Logging configuration ---
        # Application-level logging should already be set up by the entry point.
        # We don't reconfigure it here to avoid destroying the application log file.
        logger.info("ExperimentRunner initialized. Per-repeat file logging will be set when each repeat starts.")
        # --- End logging note ---

        # Note: self.config.name is NO LONGER mutated or used for primary directory naming.
        # The original config.name (from YAML, if present) is preserved.

        config_name_display = self.config.name if self.config.name is not None else "[Not Set In Config]"
        logger.info(f"Initialized ExperimentRunner for experiment config name: '{config_name_display}'")
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Checkpoints directory set to: {self.checkpoints_dir}")
        logger.info(f"Logs directory set to: {self.logs_dir}")

    def _upload_directory_to_s3(self, local_path: Path) -> None:
        """Upload a directory to the configured S3 location if enabled."""
        if not self.config.logging.upload_logs_and_checkpoints:
            return
        s3_url = self._normalize_s3_url(self.config.logging.s3_upload_url)
        if not s3_url:
            logger.warning("upload_logs_and_checkpoints is True but no s3_upload_url provided")
            return

        if not local_path.exists():
            logger.warning(f"Directory {local_path} does not exist; skipping S3 upload")
            return

        try:
            import s3fs

            fs = s3fs.S3FileSystem()
            # Compute relative path from the local project base (project_{name})
            # Compute relative path starting at experiment/group/... (skip local project folder)
            rel = self._rel_from_project_base(local_path)
            # Destination root includes the repeat directory explicitly
            dest_root = f"{s3_url.rstrip('/')}/{self.project_dirname}/{rel.as_posix()}".rstrip("/")
            # Upload files individually to avoid s3fs ambiguity about whether to include the leaf dir
            uploaded = 0
            for dirpath, _dirnames, filenames in os.walk(local_path):
                dir_path = Path(dirpath)
                inside = dir_path.relative_to(local_path)
                # Base remote directory for this inside path
                inside_posix = inside.as_posix()
                if inside_posix and inside_posix != ".":
                    base_remote = f"{dest_root}/{inside_posix}"
                else:
                    base_remote = dest_root
                for fname in filenames:
                    lfile = dir_path / fname
                    rfile = f"{base_remote}/{fname}"
                    try:
                        fs.put(str(lfile), rfile, recursive=False)
                        uploaded += 1
                    except FileNotFoundError:
                        # File disappeared between walk and upload (e.g., pruned top-k)
                        continue
                    except Exception as e_put:
                        logger.warning(f"Failed to upload {lfile} -> {rfile}: {e_put}")
            logger.info(f"Uploaded {uploaded} files from {local_path} to {dest_root}")

            # Optional prune: mirror-delete remote files that no longer exist locally
            try:
                should_prune = bool(self._should_mirror_delete())
            except Exception:
                should_prune = False
            if should_prune:
                # Remote prefix corresponding to this local directory after upload
                remote_prefix = dest_root
                try:
                    # Walk remote and delete files not present locally
                    for remote_path, _, remote_files in fs.walk(remote_prefix):
                        # Map remote prefix to local path
                        # remote_path like s3://bucket/prefix/... -> derive relative suffix after remote_prefix
                        suffix = remote_path[len(remote_prefix) :].lstrip("/")
                        local_dir = local_path / suffix
                        for filename in remote_files:
                            local_file = local_dir / filename
                            remote_file = f"{remote_path.rstrip('/')}/{filename}"
                            if not local_file.exists():
                                try:
                                    fs.rm(remote_file)
                                    logger.info(f"Pruned remote stale file: {remote_file}")
                                except Exception as e_del:
                                    logger.warning(f"Failed to prune remote file {remote_file}: {e_del}")
                except Exception as e_walk:
                    logger.warning(f"Failed walking remote for prune at {remote_prefix}: {e_walk}")
        except Exception as e:
            logger.exception(f"Failed to upload {local_path} to S3: {e}")

    # ------------------------ Continuous uploader support ------------------------
    class _S3SyncUploader(threading.Thread):
        def __init__(self, fs: Any, local_paths: list[Path], dest_base: str, project_base: Path | None, project_dirname: str | None, interval_sec: int = 30, mirror_delete: bool = False) -> None:
            super().__init__(daemon=True)
            self.fs = fs
            self.local_paths = local_paths
            self.dest_base = dest_base.rstrip("/")
            # Use the local project base (project_{name}) for relative key computation
            self.project_base = project_base
            # S3 key prefix: project folder name
            self.project_dirname = project_dirname or "project_default_project"
            self.interval_sec = max(5, int(interval_sec))
            self.mirror_delete = bool(mirror_delete)
            self._stop_event = threading.Event()

        def stop(self) -> None:
            self._stop_event.set()

        def run(self) -> None:
            while not self._stop_event.is_set():
                try:
                    for p in self.local_paths:
                        if not p.exists():
                            continue
                        # Compute relative to project base; never allow absolute path segments into S3 keys
                        # Compute relative starting at experiment/group/... and avoid abs paths
                        rel = None
                        if self.project_base is not None:
                            try:
                                rel = p.relative_to(self.project_base)
                            except Exception:
                                rel = None
                        if rel is None:
                            # Heuristic: strip up to 'project_*' if present
                            parts = list(p.parts)
                            rel_candidate = None
                            for idx, part in enumerate(parts):
                                if part.startswith("project_"):
                                    from pathlib import Path as _P

                                    rel_candidate = _P(*parts[idx + 1 :])
                                    break
                            rel = rel_candidate if rel_candidate is not None else Path(p.name)
                        # Ensure rel is a directory path (not '.')
                        if rel.as_posix() == ".":
                            rel = Path(p.name)
                        # Build destination root including the repeat directory explicitly
                        dest_root = f"{self.dest_base}/{self.project_dirname}/{rel.as_posix()}".rstrip("/")
                        # Upload files individually to keep the repeat folder as a real directory
                        for dirpath, _dirnames, filenames in os.walk(p):
                            dir_path = Path(dirpath)
                            inside = dir_path.relative_to(p)
                            inside_posix = inside.as_posix()
                            if inside_posix and inside_posix != ".":
                                base_remote = f"{dest_root}/{inside_posix}"
                            else:
                                base_remote = dest_root
                            for fname in filenames:
                                lfile = dir_path / fname
                                rfile = f"{base_remote}/{fname}"
                                try:
                                    self.fs.put(str(lfile), rfile, recursive=False)
                                except FileNotFoundError:
                                    # File may have been removed due to top-k pruning between walk and put
                                    continue
                                except Exception as e_put:
                                    logger.warning(f"Failed to upload {lfile} -> {rfile}: {e_put}")
                        # Mirror-delete: remove remote files that are no longer present locally
                        if self.mirror_delete:
                            try:
                                for remote_path, _dirs, remote_files in self.fs.walk(dest_root):
                                    # Compute relative suffix under dest_root
                                    suffix = remote_path[len(dest_root) :].lstrip("/")
                                    local_dir = p / suffix if suffix else p
                                    for filename in remote_files:
                                        local_file = local_dir / filename
                                        remote_file = f"{remote_path.rstrip('/')}/{filename}"
                                        if not local_file.exists():
                                            try:
                                                self.fs.rm(remote_file)
                                                logger.info(f"Pruned remote stale file: {remote_file}")
                                            except Exception as e_del:
                                                logger.warning(f"Failed to prune remote file {remote_file}: {e_del}")
                            except Exception as e_walk:
                                logger.warning(f"Failed walking remote for prune at {dest_root}: {e_walk}")
                except Exception as e:
                    logger.warning(f"Continuous S3 sync error: {e}")
                self._stop_event.wait(self.interval_sec)

    def _normalize_s3_url(self, url: str | None) -> str | None:
        """Accept common S3 URL forms and return s3://bucket/prefix or None.
        Supports:
        - s3://bucket/prefix
        - https://bucket.s3.<region>.amazonaws.com/prefix
        - https://s3.<region>.amazonaws.com/bucket/prefix
        - AWS console bucket URL with prefix param.
        """
        if not url:
            return None
        url = str(url).strip()
        if url.startswith("s3://"):
            return url
        try:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            host = parsed.netloc or ""
            path = parsed.path or ""
            # Console URL: .../s3/buckets/<bucket>?prefix=...
            if "console.aws.amazon.com" in host and "/s3/buckets/" in path:
                parts = path.split("/s3/buckets/")[-1]
                bucket = parts.split("/")[0]
                qs = parse_qs(parsed.query)
                prefix = qs.get("prefix", [""])[0]
                return f"s3://{bucket}/{prefix}" if bucket else None
            # Virtual-hosted style: bucket.s3.region.amazonaws.com
            if ".s3." in host and host.endswith("amazonaws.com"):
                bucket = host.split(".s3.")[0]
                prefix = path.lstrip("/")
                return f"s3://{bucket}/{prefix}"
            # Path-style: s3.region.amazonaws.com/bucket/prefix
            if host.startswith("s3.") and host.endswith("amazonaws.com"):
                segs = path.lstrip("/").split("/", 1)
                bucket = segs[0] if segs else ""
                prefix = segs[1] if len(segs) > 1 else ""
                return f"s3://{bucket}/{prefix}" if bucket else None
        except Exception:
            pass
        return None

    def setup_tensorboard(self, repeat_log_dir: Path, repeat_name: str) -> TensorBoardLogger:
        """Set up TensorBoard logging.

        Args:
            repeat_log_dir: Directory for this specific repeat's logs
            repeat_name: Name for this repeat (e.g. 'repeat_0' or with timestamp)

        Returns:
            TensorBoardLogger configured for this run.

        """
        # Ensure TensorBoard writes directly into the single repeat directory without
        # creating a nested duplicate repeat folder. We set save_dir to the parent
        # logs directory and use the repeat name as the version folder.
        parent_logs_dir = repeat_log_dir.parent
        parent_logs_dir.mkdir(parents=True, exist_ok=True)
        repeat_log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorBoardLogger(
            save_dir=str(parent_logs_dir),
            name="",  # no extra name level
            version=repeat_name,  # use repeat folder as the version level
            default_hp_metric=False,
        )
        # Guard against external cleaners removing the directory after creation.
        try:
            Path(tb_logger.log_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort: TensorBoard writer will attempt again if this fails
            pass

        logger.info(
            f"TensorBoard logger initialised for {repeat_name} in {repeat_log_dir}",
        )

        return tb_logger

    def _create_callbacks(self, repeat_name: str, ckpt_dir: Path, data_module: "SOENDataModule") -> list[pl.Callback]:
        """Create callbacks for training.

        Args:
            repeat_name: The directory/version identifier for this repeat.
            ckpt_dir: Directory where checkpoints for this repeat are saved.
            data_module: The data module for the experiment.

        Returns:
            List of PyTorch Lightning callbacks.

        """
        callbacks: list[pl.Callback] = []
        repeat_dir = ckpt_dir

        # Add simple text-based progress callback when progress bar is disabled
        if os.environ.get("SOEN_NO_PROGRESS_BAR", ""):
            callbacks.append(SimpleProgressCallback(log_every_n_steps=self.config.logging.log_freq))

        # Save the initial model state before any training
        if self.config.training.save_initial_state:
            callbacks.append(
                InitialModelSaver(
                    config=self.config,
                    dirpath=repeat_dir,
                ),
            )

        # Create SOEN checkpoint callback
        # Disable periodic epoch-based saving if non-positive
        _every_n_raw = self.config.training.checkpoint_every_n_epochs
        _every_n: int | None = None if (isinstance(_every_n_raw, int) and _every_n_raw <= 0) else _every_n_raw

        # Configure checkpointing policy: either top-k tracking or save-every-epoch
        if getattr(self.config.training, "checkpoint_save_all_epochs", False):
            checkpoint_callback = SOENModelCheckpoint(
                config=self.config,
                dirpath=str(repeat_dir),
                filename="epoch-{epoch:02d}",
                save_top_k=-1,  # keep all
                save_last=self.config.training.checkpoint_save_last,
                every_n_epochs=_every_n or 1,
            )
        else:
            checkpoint_callback = SOENModelCheckpoint(
                config=self.config,
                dirpath=str(repeat_dir),
                filename="model-{epoch:02d}-{val_loss/total:.4f}",  # PyTorch Lightning metric name
                monitor="val_loss/total",  # PyTorch Lightning metric name (MLflow sanitizes to val_loss_total)
                mode="min",
                save_top_k=self.config.training.checkpoint_save_top_k,
                save_last=self.config.training.checkpoint_save_last,
                every_n_epochs=None,
            )
        callbacks.append(checkpoint_callback)

        # Early stopping if configured
        if self.config.training.early_stopping_patience is not None:
            early_stopping = EarlyStopping(
                monitor="val_loss/total",  # PyTorch Lightning metric name (MLflow sanitizes to val_loss_total)
                mode="min",
                patience=self.config.training.early_stopping_patience,
                verbose=True,
            )
            callbacks.append(early_stopping)

        # Learning rate monitor (configured to log to file, not print to stdout)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Metrics tracker
        metrics_tracker = MetricsTracker(self.config, debug=False)
        callbacks.append(metrics_tracker)

        # ----------------------- Probing Callbacks -----------------------
        # Add ConnectionParameterProbeCallback if configured
        track_connections = getattr(self.config.logging, "track_connections", False)
        if track_connections:
            from soen_toolkit.training.callbacks.probing import (
                ConnectionParameterProbeCallback,
            )

            connection_probe = ConnectionParameterProbeCallback()
            callbacks.append(connection_probe)
            logger.info("Added ConnectionParameterProbeCallback for connection weight histograms")

        # Add StateTrajectoryLoggerCallback if configured
        try:
            st_cfg = getattr(self.config.logging, "state_trajectories", None)
            if st_cfg and bool(getattr(st_cfg, "active", False)):
                from soen_toolkit.training.callbacks.state_trajectories import (
                    StateTrajectoryLoggerCallback,
                )

                cb = StateTrajectoryLoggerCallback(
                    mode=str(getattr(st_cfg, "mode", "val")),
                    layer_id=getattr(st_cfg, "layer_id", None),
                    num_samples=int(getattr(st_cfg, "num_samples", 4)),
                    class_ids=getattr(st_cfg, "class_ids", None),
                    max_neurons_per_sample=int(getattr(st_cfg, "max_neurons_per_sample", 4)),
                    neuron_indices=getattr(st_cfg, "neuron_indices", None),
                    tag_prefix=str(getattr(st_cfg, "tag_prefix", "callbacks/state_trajectories")),
                )
                callbacks.append(cb)
                logger.info("Added StateTrajectoryLoggerCallback (mode=%s, layer_id=%s, samples=%d)", cb.mode, str(cb.layer_id), cb.num_samples)
        except Exception as e:
            logger.error("Failed to create StateTrajectoryLoggerCallback: %s", e, exc_info=True)

        # ----------------------- Dynamic callbacks -----------------------
        cbs_cfg = self.config.callbacks or {}
        seq_cfg = cbs_cfg.get("seq_len_scheduler", {}) if isinstance(cbs_cfg, dict) else {}
        if seq_cfg.get("active", False):
            try:
                from soen_toolkit.training.callbacks.seq_len_scheduler import (
                    TargetSeqLenScheduler,
                )

                start = int(seq_cfg.get("start_len", self.config.data.target_seq_len or 100))
                end = int(seq_cfg.get("end_len", start))
                start_epoch = int(seq_cfg.get("start_epoch", 0))
                end_epoch = seq_cfg.get("end_epoch", None)
                if end_epoch is not None:
                    end_epoch = int(end_epoch)
                scale_dt = bool(seq_cfg.get("scale_dt", False))
                sched_cb = TargetSeqLenScheduler(
                    data_module=data_module,
                    start_len=start,
                    end_len=end,
                    max_epochs=self.config.training.max_epochs,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    scale_dt=scale_dt,
                )
                callbacks.append(sched_cb)
                logger.info(
                    "TargetSeqLenScheduler added: %d→%d, epochs %s→%s (scale_dt=%s, max_epochs=%d)",
                    start,
                    end,
                    start_epoch,
                    end_epoch if end_epoch is not None else "max",
                    scale_dt,
                    self.config.training.max_epochs,
                )
            except Exception as e:
                logger.error("Failed to create seq_len_scheduler callback: %s", e, exc_info=True)

        # Scheduler callback (LR) - Updated to use new callbacks structure
        lr_scheduler_config = self.config.callbacks.get("lr_scheduler", {}) if isinstance(self.config.callbacks, dict) else {}

        if lr_scheduler_config:
            scheduler_type = lr_scheduler_config.get("type", "constant")

            if scheduler_type in SCHEDULER_REGISTRY:
                # Extract parameters from the scheduler config (copy to avoid modifying original)
                scheduler_params = lr_scheduler_config.copy()
                # Remove the 'type' field since it's not a parameter for the scheduler class
                if "type" in scheduler_params:
                    del scheduler_params["type"]

                # Safely convert scheduler parameters to appropriate types
                from soen_toolkit.training.utils.helpers import (
                    safe_convert_scheduler_params,
                )

                scheduler_params = safe_convert_scheduler_params(scheduler_params)

                # Create scheduler callback
                scheduler_callback = SCHEDULER_REGISTRY[scheduler_type](**scheduler_params)
                callbacks.append(scheduler_callback)

                logger.info(f"Created {scheduler_type} scheduler")
            else:
                logger.warning(f"Unknown scheduler type: {scheduler_type}. Available types: {list(SCHEDULER_REGISTRY.keys())}")
        else:
            logger.info("No lr_scheduler configuration found, skipping scheduler callback")

        # Add LossWeightScheduler(s) if configured
        loss_weight_schedulers_config = self.config.callbacks.get("loss_weight_schedulers", []) if isinstance(self.config.callbacks, dict) else {}
        if isinstance(loss_weight_schedulers_config, list):
            from soen_toolkit.training.callbacks import LossWeightScheduler

            for scheduler_config in loss_weight_schedulers_config:
                if "loss_name" in scheduler_config and "scheduler_type" in scheduler_config:
                    callbacks.append(LossWeightScheduler(**scheduler_config))
                    logger.info(f"Created LossWeightScheduler for '{scheduler_config['loss_name']}' of type '{scheduler_config['scheduler_type']}'")
                else:
                    logger.warning("Skipping loss weight scheduler due to missing 'loss_name' or 'scheduler_type'")

        # Time pooling scale scheduler callback
        time_pooling_scheduler_config = self.config.callbacks.get("time_pooling_scale_scheduler", {}) if isinstance(self.config.callbacks, dict) else {}

        if time_pooling_scheduler_config:
            try:
                from soen_toolkit.training.callbacks.time_pooling_scale_scheduler import (
                    TimePoolingScaleScheduler,
                )

                # Extract required parameters
                start_scale = float(time_pooling_scheduler_config.get("start_scale"))
                end_scale = float(time_pooling_scheduler_config.get("end_scale"))

                # Extract optional parameters
                start_epoch = int(time_pooling_scheduler_config.get("start_epoch", 0))
                end_epoch = time_pooling_scheduler_config.get("end_epoch", None)  # Will default to max_epochs
                if end_epoch is not None:
                    end_epoch = int(end_epoch)
                verbose = bool(time_pooling_scheduler_config.get("verbose", True))

                # Create the scheduler callback
                time_pooling_scheduler = TimePoolingScaleScheduler(
                    start_scale=start_scale,
                    end_scale=end_scale,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    verbose=verbose,
                )
                callbacks.append(time_pooling_scheduler)

                logger.info(f"Created TimePoolingScaleScheduler: {start_scale} → {end_scale} over epochs {start_epoch} to {end_epoch or 'max_epochs'}")

            except KeyError as e:
                logger.exception(f"Missing required parameter for time_pooling_scale_scheduler: {e}")
            except Exception as e:
                logger.error("Failed to create time_pooling_scale_scheduler callback: %s", e, exc_info=True)

        # ----------------------- Noise annealers -----------------------
        # Callbacks to anneal per-layer noise/perturb over time (e.g., bias_current perturb std)
        try:
            noise_annealers_cfg = self.config.callbacks.get("noise_annealers", []) if isinstance(self.config.callbacks, dict) else []
            if isinstance(noise_annealers_cfg, list) and len(noise_annealers_cfg) > 0:
                from soen_toolkit.training.callbacks import NoiseAnnealingCallback

                for na in noise_annealers_cfg:
                    if not isinstance(na, dict):
                        logger.warning("Skipping noise_annealer entry that is not a dict: %s", type(na))
                        continue
                    required = {"key", "target", "start_value", "end_value"}
                    if not required.issubset(na.keys()):
                        logger.warning("Skipping noise_annealer due to missing required fields. Required: %s, Provided keys: %s", required, list(na.keys()))
                        continue
                    # Instantiate with provided kwargs
                    cb = NoiseAnnealingCallback(**na)  # type: ignore[assignment]
                    callbacks.append(cb)
                logger.info("Created %d NoiseAnnealingCallback(s)", sum(isinstance(cb, NoiseAnnealingCallback) for cb in callbacks))
        except Exception as e:
            logger.error("Failed to create noise_annealers callbacks: %s", e, exc_info=True)

        # ----------------------- Stateful training (state carryover) -----------------------
        try:
            stateful_cfg = self.config.callbacks.get("stateful_training", {}) if isinstance(self.config.callbacks, dict) else {}
            if isinstance(stateful_cfg, dict) and (stateful_cfg.get("enable_for_training") or stateful_cfg.get("enable_for_validation")):
                from soen_toolkit.training.callbacks import StatefulTrainingCallback

                # Create callback with provided configuration
                stateful_callback = StatefulTrainingCallback(**stateful_cfg)
                callbacks.append(stateful_callback)
                logger.info(
                    "Created StatefulTrainingCallback (training=%s, validation=%s, sample_selection=%s)",
                    stateful_cfg.get("enable_for_training", False),
                    stateful_cfg.get("enable_for_validation", False),
                    stateful_cfg.get("sample_selection", "random")
                )
        except Exception as e:
            logger.error("Failed to create stateful_training callback: %s", e, exc_info=True)

        # ----------------------- Quantized accuracy metric -----------------------
        try:
            # Prefer callbacks.metrics.quantized_accuracy; fallback to callbacks.quantized_accuracy for backwards compatibility
            qacc_cfg = {}
            if isinstance(self.config.callbacks, dict):
                metrics_block = self.config.callbacks.get("metrics", {})
                if isinstance(metrics_block, dict):
                    qacc_cfg = metrics_block.get("quantized_accuracy", {})
                if not qacc_cfg:
                    qacc_cfg = self.config.callbacks.get("quantized_accuracy", {})
            if isinstance(qacc_cfg, dict) and qacc_cfg.get("active", False):
                from soen_toolkit.training.callbacks import QuantizedAccuracyCallback

                # Required
                min_val = float(qacc_cfg["min_val"])
                max_val = float(qacc_cfg["max_val"])
                # Either bits or levels
                bits = qacc_cfg.get("bits", None)
                if bits is not None:
                    bits = int(bits)
                levels = qacc_cfg.get("levels", None)
                if levels is not None:
                    levels = int(levels)
                # Optional
                connections = qacc_cfg.get("connections", None)
                # Default behavior: if eval_every_n_steps is not provided, evaluate once per epoch
                eval_every = qacc_cfg.get("eval_every_n_steps", None)
                if eval_every is not None:
                    try:
                        eval_every = int(eval_every)
                    except Exception:
                        eval_every = None
                # New: allow limiting number of validation batches, or None for full set
                max_eval_batches = qacc_cfg.get("max_eval_batches", None)
                if max_eval_batches is not None:
                    try:
                        max_eval_batches = int(max_eval_batches)
                    except Exception:
                        max_eval_batches = None
                cb = QuantizedAccuracyCallback(  # type: ignore[assignment]
                    min_val=min_val,
                    max_val=max_val,
                    bits=bits,
                    levels=levels,
                    connections=connections,
                    eval_every_n_steps=eval_every,
                    max_eval_batches=max_eval_batches,
                )
                callbacks.append(cb)
                logger.info("Added QuantizedAccuracyCallback (min=%.4f, max=%.4f, bits=%s, levels=%s)", min_val, max_val, str(bits), str(levels))
        except KeyError as e:
            logger.exception("Missing required field for quantized_accuracy: %s", e)
        except Exception as e:
            logger.error("Failed to create QuantizedAccuracyCallback: %s", e, exc_info=True)

        # ----------------------- QAT‑STE (Quantization Aware Training) -----------------------
        try:
            qat_cfg = {}
            if isinstance(self.config.callbacks, dict):
                qat_cfg = self.config.callbacks.get("qat", {})
            if isinstance(qat_cfg, dict) and qat_cfg.get("active", False):
                from soen_toolkit.training.callbacks import QATStraightThroughCallback

                min_val = float(qat_cfg["min_val"])
                max_val = float(qat_cfg["max_val"])
                bits = qat_cfg.get("bits", None)
                if bits is not None:
                    bits = int(bits)
                levels = qat_cfg.get("levels", None)
                if levels is not None:
                    levels = int(levels)
                connections = qat_cfg.get("connections", None)
                update_on_epoch = bool(qat_cfg.get("update_on_train_epoch_start", False))
                cb_qat = QATStraightThroughCallback(
                    min_val=min_val,
                    max_val=max_val,
                    bits=bits,
                    levels=levels,
                    connections=connections,
                    update_on_train_epoch_start=update_on_epoch,
                    stochastic_rounding=bool(qat_cfg.get("stochastic_rounding", False)),
                )
                callbacks.append(cb_qat)
                logger.info("Added QATStraightThroughCallback (min=%.4f, max=%.4f, bits=%s, levels=%s)", min_val, max_val, str(bits), str(levels))
        except KeyError as e:
            logger.exception("Missing required field for qat: %s", e)
        except Exception as e:
            logger.error("Failed to create QATStraightThroughCallback: %s", e, exc_info=True)

        # ----------------------- Connection noise callback -----------------------
        try:
            conn_noise_cfg: dict[str, Any] = {}
            if isinstance(self.config.callbacks, dict):
                conn_noise_cfg = self.config.callbacks.get("connection_noise", {}) or {}
            if isinstance(conn_noise_cfg, dict) and conn_noise_cfg.get("connections"):
                from soen_toolkit.training.callbacks import ConnectionNoiseCallback

                # Required
                connections = conn_noise_cfg.get("connections", [])
                std = float(conn_noise_cfg.get("std", 0.0))
                # Optional
                relative = bool(conn_noise_cfg.get("relative", False))
                per_step = bool(conn_noise_cfg.get("per_step", True))
                log_every_n_steps = int(conn_noise_cfg.get("log_every_n_steps", self.config.logging.log_freq))
                seed = conn_noise_cfg.get("seed", None)
                if seed is not None:
                    try:
                        seed = int(seed)
                    except Exception:
                        seed = None
                conn_cb = ConnectionNoiseCallback(
                    connections=connections,
                    std=std,
                    relative=relative,
                    per_step=per_step,
                    log_every_n_steps=log_every_n_steps,
                    seed=seed,
                )
                callbacks.append(conn_cb)
                logger.info(
                    "Added ConnectionNoiseCallback for %d connections (std=%.6f, relative=%s, per_step=%s, log_every_n_steps=%d)",
                    len(connections),
                    std,
                    relative,
                    per_step,
                    log_every_n_steps,
                )
        except Exception as e:
            logger.error("Failed to create ConnectionNoiseCallback: %s", e, exc_info=True)

        return callbacks

    def run_experiment(self) -> None:
        """Run the experiment with configured repeats.

        This method runs the experiment for the specified number of repeats,
        setting different random seeds for each repeat.
        """
        # When profiling is active, force a single short run to avoid nested profiler issues across repeats
        try:
            if getattr(self.config, "profiler", None) and bool(getattr(self.config.profiler, "active", False)):
                num_repeats = 1
            else:
                num_repeats = self.config.training.num_repeats
        except Exception:
            num_repeats = self.config.training.num_repeats

        for repeat in range(num_repeats):
            self.repeat = repeat
            seed = self.base_seed + repeat

            logger.info(f"Starting experiment '{self.config.name}' - Repeat {repeat + 1}/{num_repeats} (Seed: {seed})")

            try:
                self.run_single_repeat(repeat, seed)
                logger.info(f"Completed repeat {repeat + 1}/{num_repeats}")
            except Exception as e:
                logger.error(f"Error in repeat {repeat + 1}: {e}", exc_info=True)
                raise  # Fail fast - do not silently continue

        logger.info(f"Experiment '{self.config.name}' completed")

    def run_single_repeat(self, repeat: int, seed: int) -> str | None:
        """Run a single repeat of the experiment.

        Args:
            repeat: Numerical index of the repeat.
            seed: Random seed for this repeat.

        """
        # --- Setup ---
        pl.seed_everything(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        repeat_log_dir, repeat_ckpt_dir, repeat_name = self._prepare_repeat_subdirs(repeat)

        # Robust logging setup - writes to file only (terminal kept clean)
        from soen_toolkit.training.utils.robust_logging import setup_robust_logging

        repeat_log_file = repeat_log_dir / f"experiment_run_{repeat_name}.log"
        log_level = getattr(self.config.logging, "log_level", "INFO").upper()

        # Remove all existing handlers from root logger to prevent duplication or interference
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set up robust logging - file only, no console clutter
        setup_robust_logging(
            log_file=repeat_log_file,
            log_level=log_level,
            console=False,  # No console output - all logs go to file
            console_level=log_level,
        )

        # Log confirmation to file only
        logger.info(f"{'=' * 80}")
        logger.info("LOGGING INITIALIZED")
        logger.info(f"  Log file: {repeat_log_file}")
        logger.info(f"  Log level: {log_level}")
        logger.info(f"{'=' * 80}")
        logger.info(f"Per-repeat experiment logging configured: {repeat_log_file}")
        logger.info(f"All logs will be written to: {repeat_log_file}")

        # Minimal terminal output - just the log file location
        print(f"\nAll logs are being written to: {repeat_log_file}\n", flush=True)

        tensorboard_logger = self.setup_tensorboard(repeat_log_dir, repeat_name)
        # Prepare list of loggers (TensorBoard is always present)
        loggers: list[Any] = [tensorboard_logger]
        # Optionally add MLflow logger if enabled in config and available
        try:
            # Suppress MLflow's own URL prints so we can print links once, ourselves
            if bool(getattr(self.config.logging, "mlflow_active", False)):
                with contextlib.suppress(Exception):
                    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "1")
            if bool(getattr(self.config.logging, "mlflow_active", False)):
                # Validation: Check for common misconfigurations
                tracking_uri = self._resolve_mlflow_tracking_uri()
                if not tracking_uri:
                    msg = "MLflow is enabled (mlflow_active=true) but mlflow_tracking_uri is not set. Please set it to 'file:./mlruns' for local tracking or your server URL."
                    raise RuntimeError(
                        msg,
                    )

                # Check for authentication requirements
                import os as _os

                needs_auth = tracking_uri.startswith(("http://", "https://")) and "@" not in tracking_uri
                env_username = _os.environ.get("MLFLOW_TRACKING_USERNAME")
                env_password = _os.environ.get("MLFLOW_TRACKING_PASSWORD")
                has_env_auth = bool(env_username) and bool(env_password)
                cfg_username = getattr(self.config.logging, "mlflow_username", None)
                cfg_password = getattr(self.config.logging, "mlflow_password", None)
                has_config_auth = bool(cfg_username) and bool(cfg_password)

                if needs_auth and not has_env_auth and not has_config_auth:
                    msg = (
                        f"MLflow server '{tracking_uri}' requires authentication. "
                        "Add to your config:\n"
                        "  mlflow_username: admin\n"
                        "  mlflow_password: your_password\n"
                        "Or set MLFLOW_TRACKING_USERNAME/MLFLOW_TRACKING_PASSWORD env vars."
                    )
                    raise RuntimeError(
                        msg,
                    )

                # Derive MLflow experiment from logging.experiment_name when provided,
                # otherwise fall back to logging.project_name. If a legacy
                # mlflow_experiment_name is provided, ignore it and warn for clarity.
                try:
                    if getattr(self.config.logging, "mlflow_experiment_name", None):
                        logger.warning(
                            "'logging.mlflow_experiment_name' is deprecated and ignored. MLflow experiment now derives from logging.experiment_name (if set) or logging.project_name.",
                        )
                except Exception:
                    pass
                experiment_name = getattr(self.config.logging, "experiment_name", None) or getattr(self.config.logging, "project_name", None) or "soen_training"
                run_name = getattr(self.config.logging, "mlflow_run_name", None) or repeat_name
                tags = dict(getattr(self.config.logging, "mlflow_tags", {}) or {})
                # Force canonical tags from logging config to avoid drift with YAML overrides
                tags["project"] = str(getattr(self.config.logging, "project_name", "soen_training") or "soen_training")
                tags["experiment"] = str(
                    getattr(self.config.logging, "experiment_name", None) or getattr(self.config.logging, "project_name", "soen_training"),
                )
                tags["group"] = str(getattr(self.config.logging, "group_name", "default_group") or "default_group")
                # Add run context
                tags["repeat"] = str(repeat)
                tags["seed"] = str(seed)

                # Ensure all tag values are strings (MLflow requirement)
                tags = {str(k): str(v) for k, v in tags.items()}

                # If explicit username/password configured and env vars not set, set them for this process
                try:
                    if not _os.environ.get("MLFLOW_TRACKING_USERNAME") and not _os.environ.get("MLFLOW_TRACKING_PASSWORD") and cfg_username and cfg_password:
                        _os.environ["MLFLOW_TRACKING_USERNAME"] = str(cfg_username)
                        _os.environ["MLFLOW_TRACKING_PASSWORD"] = str(cfg_password)
                except Exception:
                    pass

                # Use the safe wrapper to sanitize metric keys for MLflow only
                logger_cls = SafeMLFlowLogger if "SafeMLFlowLogger" in globals() and SafeMLFlowLogger is not None else MLFlowLogger
                mlf_logger = logger_cls(
                    experiment_name=experiment_name,
                    tracking_uri=tracking_uri,
                    run_name=run_name,
                    tags=tags,
                    # Disable PL's auto checkpoint scanning to avoid MLflow artifact_path issues;
                    # our checkpoint callback logs artifacts explicitly.
                    log_model=False,
                )
                loggers.append(mlf_logger)
                logger.info(f"MLflow logger initialised (experiment='{experiment_name}', run='{run_name}', uri='{tracking_uri}')")
        except Exception as e:
            logger.exception(f"MLflow logger setup failed; continuing with TensorBoard only: {e}")
            logger.exception("To fix MLflow issues, check: tracking URI, credentials, network connectivity, and that 'uv sync --extra tracking' was run")

        # --- Distillation: Generate teacher trajectories if paradigm is 'distillation' ---
        if self.config.training.paradigm == "distillation":
            distillation_data_path = self._prepare_distillation_data(repeat_log_dir)
            # Update config to use generated distillation dataset
            self.config.data.data_path = distillation_data_path
            logger.info(f"Distillation mode: using generated dataset at {distillation_data_path}")

        data_module = SOENDataModule(self.config)

        # Optional continuous S3 sync (logs + checkpoints) if enabled
        s3_sync_thread = None
        if bool(getattr(self.config.logging, "upload_logs_and_checkpoints", False)) and getattr(self.config.logging, "s3_upload_url", None):
            try:
                import s3fs

                fs = s3fs.S3FileSystem()
                dest_base = self._normalize_s3_url(self.config.logging.s3_upload_url)
                to_sync = [repeat_log_dir, repeat_ckpt_dir]
                # Choose interval (default 5s; uploader enforces a minimum of 5s)
                try:
                    sync_interval = int(getattr(self.config.logging, "s3_sync_interval_seconds", 5))
                except Exception:
                    sync_interval = 5
                self._s3_sync_interval_seconds = sync_interval
                s3_sync_thread = ExperimentRunner._S3SyncUploader(
                    fs=fs,
                    local_paths=to_sync,
                    dest_base=dest_base or "",
                    project_base=getattr(self, "project_base_dir", None),
                    project_dirname=getattr(self, "project_dirname", None),
                    interval_sec=sync_interval,
                    mirror_delete=bool(self._should_mirror_delete()),
                )
                s3_sync_thread.start()
                logger.info(f"Started continuous S3 sync thread (interval {self._s3_sync_interval_seconds}s)")
            except Exception as e:
                logger.warning(f"Could not start continuous S3 sync: {e}")
        # Infer num_classes from data if not provided or mismatched
        try:
            # Distillation datasets store teacher trajectories as labels (regression targets),
            # so "num_classes" is not a meaningful concept and can lead to downstream
            # classification assumptions. Skip inference in distillation mode.
            if str(getattr(self.config.training, "paradigm", "supervised")).lower() != "distillation":
                inferred_classes = data_module.infer_num_classes()
                current_classes = int(getattr(self.config.data, "num_classes", 0) or 0)
                # Update config if: (1) num_classes not set (current_classes == 0) and we inferred something, OR
                #                    (2) inferred value differs from current value
                if inferred_classes > 0 and (current_classes == 0 or inferred_classes != current_classes):
                    logger.info(
                        f"Inferred num_classes={inferred_classes} from dataset (was {current_classes if current_classes > 0 else 'not set'}). Updating config.",
                    )
                    self.config.data.num_classes = inferred_classes
        except Exception as e:
            logger.warning(f"Failed to infer num_classes from dataset: {e}")

        # --- Backend selection: Torch (default) vs JAX ---
        # Ensure datamodule is prepared for non-Lightning runs
        with contextlib.suppress(Exception):
            data_module.prepare_data()
        try:
            data_module.setup(stage="fit")
        except Exception:
            with contextlib.suppress(Exception):
                data_module.setup()

        backend = str(getattr(self.config.model, "backend", "torch")).lower()
        if backend == "jax":
            try:
                from soen_toolkit.training.trainers.jax_backend import JaxRunner
            except Exception as exc:
                msg = "JAX backend requested but unavailable. Install JAX dependencies or set model.backend: 'torch'."
                logger.exception("%s (%s)", msg, exc)
                raise RuntimeError(msg) from exc

            # Print a minimal summary and validate logging targets before training
            self._print_training_summary(
                repeat,
                seed,
                repeat_log_dir,
                repeat_ckpt_dir,
                ckpt_path_to_resume=None,
                callbacks=[],
            )
            self._validate_mlflow_config()

            jax_runner = JaxRunner(
                self.config,
                repeat_log_dir=repeat_log_dir,
                repeat_ckpt_dir=repeat_ckpt_dir,
                loggers=loggers,
            )
            logger.info("Starting JAX runner fit loop")
            try:
                best_ckpt_path: str | None = jax_runner.fit(data_module)
            except Exception as e:
                logger.error(f"JAX training failed: {e}", exc_info=True)
                raise  # Fail fast - do not continue to next repeat

            if s3_sync_thread is not None:
                with contextlib.suppress(Exception):
                    s3_sync_thread.stop()
            self._upload_directory_to_s3(Path(tensorboard_logger.log_dir))
            self._upload_directory_to_s3(repeat_ckpt_dir)

            with contextlib.suppress(Exception):
                self._print_mlflow_links(loggers, repeat_name)

            return best_ckpt_path
        callbacks = self._create_callbacks(repeat_name, repeat_ckpt_dir, data_module)
        # --- Model Initialization & Checkpoint Handling (Torch) ---
        ckpt_path_to_resume = self.config.training.train_from_checkpoint
        model = None

        if ckpt_path_to_resume:
            ckpt_path = Path(ckpt_path_to_resume)
            if not ckpt_path.exists():
                msg = f"train_from_checkpoint path does not exist: {ckpt_path}"
                raise FileNotFoundError(msg)
            if not ckpt_path.is_file():
                msg = f"train_from_checkpoint path is not a file: {ckpt_path}"
                raise FileNotFoundError(msg)
            if ckpt_path.suffix != ".ckpt":
                msg = f"train_from_checkpoint path must be a .ckpt file, got: {ckpt_path}"
                raise ValueError(msg)

            logger.info(f"Resuming training from checkpoint: {ckpt_path}")
            logger.info(f"Checkpoint file size: {ckpt_path.stat().st_size / (1024 * 1024):.1f} MB")

            # Load hparams from checkpoint to build the model structure correctly
            try:
                logger.info(f"Loading checkpoint data from {ckpt_path}...")
                # Load the full checkpoint dictionary, allowing arbitrary objects
                checkpoint_data = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)

                # If a sidecar .soen exists, prefer using it to reconstruct the SOEN core
                sidecar_path = ckpt_path.with_suffix(".soen")
                soen_core_loaded = None
                if sidecar_path.exists():
                    try:
                        from soen_toolkit.core import SOENModelCore

                        logger.info(f"Found sidecar SOEN core next to checkpoint: {sidecar_path}")
                        soen_core_loaded = SOENModelCore.load(str(sidecar_path))
                        logger.info("Loaded SOEN core from sidecar successfully.")
                    except Exception as e:
                        logger.warning(f"Failed to load sidecar SOEN core {sidecar_path}: {e}")

                # Check if hyperparameters exist in the checkpoint
                if "hyper_parameters" in checkpoint_data:
                    logger.info("Found hyperparameters in checkpoint - using them to build model structure.")
                    hparams = checkpoint_data["hyper_parameters"]

                    # Log some key config info for debugging
                    if isinstance(hparams, dict):
                        logger.info(f"Checkpoint hyperparameters keys: {list(hparams.keys())}")
                        if "name" in hparams:
                            logger.info(f"Checkpoint experiment name: {hparams['name']}")

                    # Create a config object from loaded hparams
                    logger.info("Creating model configuration from checkpoint hyperparameters.")
                    try:
                        resume_config = ExperimentConfig.from_dict(hparams)
                        logger.info("Successfully created config from checkpoint hyperparameters.")
                    except Exception as config_error:
                        logger.exception(f"Failed to create config from checkpoint hyperparameters: {config_error}")
                        logger.warning("Falling back to current config file to build model structure.")
                        model = self._create_model(seed)
                        logger.info("Model created using current config as fallback.")
                    else:
                        # Build the model using the configuration loaded FROM THE CHECKPOINT
                        logger.info("Instantiating model using configuration from checkpoint.")
                        model = SOENLightningModule(resume_config)
                        logger.info("Model structure instantiated successfully from checkpoint config.")
                        # If we managed to load a sidecar core, inject it into the wrapper
                        if soen_core_loaded is not None and hasattr(model, "model"):
                            try:
                                model.model = soen_core_loaded
                                logger.info("Injected SOEN core from sidecar into Lightning model (replaced core).")
                            except Exception as e:
                                logger.warning(f"Failed to inject sidecar SOEN core into model: {e}")
                else:
                    logger.warning("No hyperparameters found in checkpoint - using current config file to build model structure.")
                    logger.info("This may indicate the checkpoint was saved without hyperparameters enabled.")
                    # Use the current configuration file settings instead
                    model = self._create_model(seed)

            except Exception as e:
                logger.error(f"Failed to load hyperparameters or build model from checkpoint {ckpt_path}: {e}", exc_info=True)
                msg = f"Could not prepare model for resumption from checkpoint: {ckpt_path}"
                raise RuntimeError(msg) from e
        else:
            logger.info("Starting new training run (not resuming from checkpoint).")
            # Create model using the current configuration file settings
            model = self._create_model(seed)

        if model is None:
            msg = "Model initialization failed."
            raise RuntimeError(msg)

        # --- Trainer Setup & Execution ---
        # If dynamic sequence-length scheduler is active we must reload
        # dataloaders each epoch so that the new cached dataset is picked up.
        reload_every_epoch = 1 if (self.config.callbacks.get("seq_len_scheduler", {}).get("active", False) if isinstance(self.config.callbacks, dict) else False) else 0

        # Configure Lightning's TBPTT in a version-safe way
        trainer_kwargs: dict[str, Any] = {
            "max_epochs": self.config.training.max_epochs,
            "callbacks": callbacks,
            "logger": loggers,
            "default_root_dir": str(repeat_log_dir),
            "accelerator": self.config.training.accelerator,
            "devices": self.config.training.devices,
            "precision": self.config.training.precision,
            "deterministic": self.config.training.deterministic,
            "accumulate_grad_batches": self.config.training.accumulate_grad_batches,
            "gradient_clip_val": self.config.training.gradient_clip_val,
            "gradient_clip_algorithm": self.config.training.gradient_clip_algorithm,
            "log_every_n_steps": self.config.logging.log_freq,
            "reload_dataloaders_every_n_epochs": reload_every_epoch,
            # Disable progress bar if SOEN_NO_PROGRESS_BAR is set (avoids widget errors in notebooks)
            "enable_progress_bar": not bool(os.environ.get("SOEN_NO_PROGRESS_BAR", "")),
        }
        # Optional distributed settings from config: num_nodes, strategy
        try:
            num_nodes = getattr(self.config.training, "num_nodes", None)
            if num_nodes is not None:
                trainer_kwargs["num_nodes"] = int(num_nodes)
        except Exception:
            pass
        try:
            if getattr(self.config.training, "strategy", None):
                trainer_kwargs["strategy"] = self.config.training.strategy
        except Exception:
            pass
        # Only pass truncated_bptt_steps if this PL version supports it
        try:
            import inspect as _inspect

            if getattr(self.config.training, "use_tbptt", False):
                sig = _inspect.signature(pl.Trainer.__init__)
                if "truncated_bptt_steps" in sig.parameters:
                    tbptt_steps = self.config.training.tbptt_steps
                    if tbptt_steps is not None:
                        trainer_kwargs["truncated_bptt_steps"] = int(tbptt_steps)
        except Exception:
            # Fall back silently; the LightningModule path will handle TBPTT
            pass

        # Optional profiler

        profiler_cfg = getattr(self.config, "profiler", None)
        if profiler_cfg and getattr(profiler_cfg, "active", False):
            try:
                ptype = getattr(profiler_cfg, "type", "simple")
                if ptype == "pytorch":
                    # Use dirpath + filename. Do NOT pass an absolute path as filename.
                    profiler = pl.profilers.PyTorchProfiler(
                        dirpath=str(repeat_log_dir),
                        filename=(getattr(profiler_cfg, "output_filename", None) or "pytorch_profiler"),
                        record_shapes=bool(getattr(profiler_cfg, "record_shapes", False)),
                        profile_memory=bool(getattr(profiler_cfg, "profile_memory", False)),
                        with_stack=bool(getattr(profiler_cfg, "with_stack", False)),
                    )
                elif ptype == "advanced":
                    # Advanced profiler disabled for this project; use PyTorch profiler instead
                    logger.warning("Advanced profiler is disabled; using PyTorch profiler instead.")
                    profiler = pl.profilers.PyTorchProfiler(
                        dirpath=str(repeat_log_dir),
                        filename=(getattr(profiler_cfg, "output_filename", None) or "pytorch_profiler"),
                        record_shapes=bool(getattr(profiler_cfg, "record_shapes", False)),
                        profile_memory=bool(getattr(profiler_cfg, "profile_memory", False)),
                        with_stack=bool(getattr(profiler_cfg, "with_stack", False)),
                    )
                else:
                    profiler = "simple"  # type: ignore[assignment]
                trainer_kwargs["profiler"] = profiler
                # Optionally limit batches during profiling (None = all batches)
                tlim = getattr(profiler_cfg, "num_train_batches", None)
                vlim = getattr(profiler_cfg, "num_val_batches", None)
                if tlim is not None:
                    trainer_kwargs["limit_train_batches"] = int(tlim)
                if vlim is not None:
                    trainer_kwargs["limit_val_batches"] = int(vlim)
                # Speed up & make interrupt responsive during profiling
                trainer_kwargs["num_sanity_val_steps"] = 0
                trainer_kwargs["enable_progress_bar"] = False
                logger.info("Profiler enabled (type=%s) with batch limits train=%s, val=%s", ptype, str(tlim), str(vlim))
            except Exception as e:
                # Fail fast with a clear message; do not silently downgrade
                logger.exception("Profiler configuration failed: %s", e)
                raise

        # Suppress unnecessary PyTorch Lightning warnings for cleaner output
        import warnings

        warnings.filterwarnings("ignore", message="GPU available but not used.*")
        trainer = pl.Trainer(**trainer_kwargs)

        # Print training summary before starting
        self._print_training_summary(repeat, seed, repeat_log_dir, repeat_ckpt_dir, ckpt_path_to_resume, callbacks)

        # Validate MLflow configuration before starting training
        self._validate_mlflow_config()

        logger.info(f"Starting trainer.fit. Checkpoint path (if resuming): {ckpt_path_to_resume}")
        # Pass ckpt_path_to_resume, which is None if not resuming
        # IMPORTANT: pass datamodule by keyword so PL does not treat it as train_dataloaders

        import time

        train_start_time = time.time()

        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path_to_resume)

        train_end_time = time.time()
        train_duration = train_end_time - train_start_time

        # Print post-training summary with final metrics
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

        # Display final metrics from trainer
        if hasattr(trainer, "callback_metrics") and trainer.callback_metrics:
            print("\nFinal Metrics:")
            metrics_dict = trainer.callback_metrics

            # Group metrics by type
            train_metrics = {}
            val_metrics = {}
            other_metrics = {}

            for key, value in metrics_dict.items():
                key_str = str(key)
                try:
                    val_float = float(value.item() if hasattr(value, "item") else value)
                    if key_str.startswith("train_"):
                        train_metrics[key_str] = val_float
                    elif key_str.startswith("val_"):
                        val_metrics[key_str] = val_float
                    else:
                        other_metrics[key_str] = val_float
                except Exception:
                    pass

            if train_metrics:
                print("   Training:")
                for k, v in train_metrics.items():
                    print(f"      {k}: {v:.6f}")

            if val_metrics:
                print("   Validation:")
                for k, v in val_metrics.items():
                    print(f"      {k}: {v:.6f}")

            if other_metrics:
                print("   Other:")
                for k, v in other_metrics.items():
                    print(f"      {k}: {v:.6f}")

        # Display training time
        hours = int(train_duration // 3600)
        minutes = int((train_duration % 3600) // 60)
        seconds = int(train_duration % 60)
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        print(f"\nTotal Training Time: {time_str}")

        # Find and display best checkpoint
        best_ckpt_path = None
        for cb in callbacks:
            if isinstance(cb, SOENModelCheckpoint):
                best_ckpt_path = cb.best_model_path
                break
        if best_ckpt_path:
            print(f"\nBest Checkpoint: {best_ckpt_path}")

        print("=" * 80 + "\n")

        # --- Testing & Cleanup ---
        if hasattr(data_module, "test_dataloader") and data_module.test_dataloader() is not None:
            logger.info("Running test")
            trainer.test(model, datamodule=data_module)

        # TensorBoardLogger does not require explicit closing

        # Stop continuous sync if running, then perform a final sync
        if s3_sync_thread is not None:
            with contextlib.suppress(Exception):
                s3_sync_thread.stop()
        self._upload_directory_to_s3(Path(tensorboard_logger.log_dir))
        self._upload_directory_to_s3(repeat_ckpt_dir)

        # best_ckpt_path already found and printed in post-training summary above
        logger.info(f"Best model checkpoint path from this run: {best_ckpt_path}")

        # (Final upload already performed above)

        # Print MLflow links once per repeat (after test completes)
        with contextlib.suppress(Exception):
            self._print_mlflow_links(loggers, repeat_name)

        return best_ckpt_path

    def _prepare_distillation_data(self, repeat_log_dir: Path) -> Path:
        """Generate teacher trajectories for distillation training.

        This method runs the teacher model on the source dataset and saves
        the output state trajectories as regression targets for the student.
        If a pre-existing distillation dataset is specified, it will be used instead.

        Args:
            repeat_log_dir: Directory for this repeat's logs (distillation data saved here).

        Returns:
            Path to the distillation HDF5 dataset (existing or generated).
        """
        from soen_toolkit.training.distillation import generate_teacher_trajectories

        dist_cfg = self.config.training.distillation
        if dist_cfg is None:
            msg = "Distillation config is required when paradigm='distillation'"
            raise ValueError(msg)

        # Check if user specified an existing distillation dataset
        existing_distill_path = getattr(dist_cfg, "distillation_data_path", None)
        if existing_distill_path is not None:
            existing_distill_path = Path(existing_distill_path)
            if existing_distill_path.exists():
                logger.info("=" * 60)
                logger.info("USING EXISTING DISTILLATION DATASET")
                logger.info("=" * 60)
                logger.info(f"  Distillation data: {existing_distill_path}")
                logger.info("  Skipping teacher trajectory generation")
                logger.info("=" * 60)
                return existing_distill_path
            else:
                logger.warning(f"Specified distillation_data_path does not exist: {existing_distill_path}")
                logger.warning("Will generate new distillation dataset instead")

        # Output path for distillation data
        output_path = repeat_log_dir / "distillation_data.hdf5"

        # Teacher model path (may be None if using existing distillation data)
        teacher_path = Path(dist_cfg.teacher_model_path) if dist_cfg.teacher_model_path else None
        if teacher_path and not teacher_path.exists():
            msg = f"Teacher model not found: {teacher_path}"
            raise FileNotFoundError(msg)

        # Source data path
        source_data_path = Path(self.config.data.data_path)
        if not source_data_path.exists():
            msg = f"Source dataset not found: {source_data_path}"
            raise FileNotFoundError(msg)

        # Get data preprocessing settings to apply same transforms
        target_seq_len = getattr(self.config.data, "target_seq_len", None)
        scale_min = getattr(self.config.data, "min_scale", None)
        scale_max = getattr(self.config.data, "max_scale", None)

        # Determine backend for teacher trajectory generation
        backend = str(getattr(self.config.model, "backend", "torch")).lower()
        logger.info(f"  Backend: {backend}")

        logger.info("=" * 60)
        logger.info("DISTILLATION DATA GENERATION")
        logger.info("=" * 60)
        logger.info(f"  Teacher model: {teacher_path}")
        logger.info(f"  Source dataset: {source_data_path}")
        logger.info(f"  Subset fraction: {dist_cfg.subset_fraction}")
        logger.info(f"  Max samples: {dist_cfg.max_samples}")
        logger.info(f"  Target seq len: {target_seq_len}")
        logger.info(f"  Scale range: [{scale_min}, {scale_max}]")
        logger.info(f"  Output path: {output_path}")

        # Generate teacher trajectories with same preprocessing as training
        if teacher_path is None:
            msg = "Teacher model path must be provided for trajectory generation when no existing dataset is found."
            raise ValueError(msg)

        generate_teacher_trajectories(
            teacher_model_path=teacher_path,
            source_data_path=source_data_path,
            output_path=output_path,
            subset_fraction=dist_cfg.subset_fraction,
            max_samples=dist_cfg.max_samples,
            batch_size=dist_cfg.batch_size,
            target_seq_len=target_seq_len,
            scale_min=scale_min,
            scale_max=scale_max,
            device=self.config.training.accelerator,
        )

        logger.info("=" * 60)
        logger.info("Distillation data generation complete")
        logger.info("=" * 60)

        return output_path

    def _create_model(self, seed: int) -> SOENLightningModule:
        """Create and initialize model based on the current config file.
        This is called when not resuming from a checkpoint.

        Args:
            seed: Random seed (may not be used directly here but good practice)

        Returns:
            Initialized SOENLightningModule

        """
        logger.info("Creating new model instance based on configuration file.")
        # Logic using self.config (including base_model_path and load_exact_model_state)
        if self.config.model.base_model_path is None and not self.config.model.load_exact_model_state:
            logger.warning("base_model_path is None and load_exact_model_state is False. Ensure SOENLightningModule handles this or define a default structure.")
            # Depending on SOENLightningModule implementation, might need default config here

        # _build_model in SOENLightningModule now handles the logic based on self.config
        model = SOENLightningModule(self.config)

        log_msg = "Created model structure."
        if self.config.model.base_model_path:
            log_msg += f" Source: {self.config.model.base_model_path}, Exact Load: {self.config.model.load_exact_model_state}"
        logger.info(log_msg)
        return model

    def _save_training_config_summary(
        self,
        repeat: int,
        seed: int,
        repeat_log_dir: Path,
        repeat_ckpt_dir: Path,
        ckpt_path_to_resume: str | None = None,
        callbacks: list[pl.Callback] | None = None,
    ) -> Path:
        """Save training configuration summary to a clean YAML file.

        Returns:
            Path to the saved configuration summary file
        """
        from datetime import datetime

        import yaml

        summary_path = repeat_log_dir / "training_config_summary.yaml"

        # Build summary dictionary with explicit type annotations
        losses_list: list[dict[str, Any]] = []
        metrics_list: list[Any] = []
        optimizer_dict: dict[str, Any] = {}
        data_dict: dict[str, Any] = {}
        callbacks_dict: dict[str, Any] = {}

        summary: dict[str, Any] = {
            "experiment": {
                "repeat": repeat + 1,
                "total_repeats": getattr(self.config.training, "num_repeats", 1),
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
            },
            "model": {
                "backend": str(getattr(self.config.model, "backend", "torch")).lower(),
                "base_model_path": str(self.config.model.base_model_path) if self.config.model.base_model_path else None,
                "load_exact_model_state": getattr(self.config.model, "load_exact_model_state", False),
                "dt": getattr(self.config.model, "dt", None),
                "dt_learnable": getattr(self.config.model, "dt_learnable", False),
                "time_pooling": getattr(self.config.model, "time_pooling", "max"),
                "architecture_yaml": str(self.config.model.architecture_yaml) if getattr(self.config.model, "architecture_yaml", None) else None,
            },
            "training": {
                "max_epochs": self.config.training.max_epochs,
                "batch_size": self.config.training.batch_size,
                "accumulate_grad_batches": getattr(self.config.training, "accumulate_grad_batches", 1),
                "gradient_clip_val": getattr(self.config.training, "gradient_clip_val", None),
                "gradient_clip_algorithm": getattr(self.config.training, "gradient_clip_algorithm", "norm"),
                "mapping": getattr(self.config.training, "mapping", "seq2static"),
                "paradigm": getattr(self.config.training, "paradigm", "supervised"),
                "accelerator": getattr(self.config.training, "accelerator", "auto"),
                "precision": getattr(self.config.training, "precision", "32-true"),
                "deterministic": getattr(self.config.training, "deterministic", True),
                "devices": getattr(self.config.training, "devices", "auto"),
                "strategy": getattr(self.config.training, "strategy", None),
                "early_stopping_patience": getattr(self.config.training, "early_stopping_patience", None),
                "checkpoint_every_n_epochs": getattr(self.config.training, "checkpoint_every_n_epochs", 1),
                "checkpoint_save_top_k": getattr(self.config.training, "checkpoint_save_top_k", 3),
                "checkpoint_save_last": getattr(self.config.training, "checkpoint_save_last", True),
                "checkpoint_save_all_epochs": getattr(self.config.training, "checkpoint_save_all_epochs", False),
                "save_initial_state": getattr(self.config.training, "save_initial_state", True),
                "use_tbptt": getattr(self.config.training, "use_tbptt", False),
                "tbptt_steps": getattr(self.config.training, "tbptt_steps", None),
                "tbptt_stride": getattr(self.config.training, "tbptt_stride", None),
            },
            "optimizer": optimizer_dict,
            "losses": losses_list,
            "metrics": metrics_list,
            "data": data_dict,
            "callbacks": callbacks_dict,
            "logging": {
                "project_dir": str(self.config.logging.project_dir),
                "project_name": self.config.logging.project_name,
                "experiment_name": self.config.logging.experiment_name,
                "group_name": self.config.logging.group_name,
                "log_level": getattr(self.config.logging, "log_level", "INFO"),
            },
            "paths": {
                "log_dir": str(repeat_log_dir),
                "checkpoint_dir": str(repeat_ckpt_dir),
                "resume_from_checkpoint": str(ckpt_path_to_resume) if ckpt_path_to_resume else None,
            },
        }

        # Optimizer info
        try:
            opt_config = self.config.training.optimizer
            if isinstance(opt_config, dict):
                optimizer_dict["name"] = opt_config.get("name", "unknown")
                optimizer_dict["lr"] = opt_config.get("lr", "unknown")
                optimizer_dict["kwargs"] = opt_config.get("kwargs", {})
            else:
                optimizer_dict["name"] = getattr(opt_config, "name", "unknown")
                optimizer_dict["lr"] = getattr(opt_config, "lr", "unknown")
                optimizer_dict["kwargs"] = getattr(opt_config, "kwargs", {})
        except Exception:
            optimizer_dict["name"] = "not_configured"

        # Loss configuration
        try:
            losses = None
            if hasattr(self.config.training, "losses"):
                losses = self.config.training.losses
            elif hasattr(self.config.training, "loss") and hasattr(self.config.training.loss, "losses"):
                losses = self.config.training.loss.losses

            if losses:
                loss_list = losses if isinstance(losses, (list, tuple)) else [losses]
                for loss_config in loss_list:
                    if isinstance(loss_config, dict):
                        losses_list.append({"name": loss_config.get("name", "unknown"), "weight": loss_config.get("weight", 1.0), "params": loss_config.get("params", {})})
                    else:
                        losses_list.append({"name": getattr(loss_config, "name", "unknown"), "weight": getattr(loss_config, "weight", 1.0), "params": getattr(loss_config, "params", {})})
        except Exception as e:
            losses_list.clear()
            losses_list.append({"error": str(e)})

        # Metrics
        try:
            if self.config.logging.metrics:
                metrics_config = self.config.logging.metrics
                if isinstance(metrics_config, list):
                    for item in metrics_config:
                        if isinstance(item, str):
                            metrics_list.append(item)
                        elif isinstance(item, dict):
                            metrics_list.append(item.get("name", "unknown"))
                elif isinstance(metrics_config, str):
                    metrics_list.append(metrics_config)
        except Exception:
            pass

        # Data configuration
        try:
            data_dict["data_path"] = str(getattr(self.config.data, "data_path", "not_specified"))
            data_dict["target_seq_len"] = getattr(self.config.data, "target_seq_len", None)
            data_dict["cache_data"] = getattr(self.config.data, "cache_data", False)
            data_dict["num_classes"] = getattr(self.config.data, "num_classes", None)
            data_dict["val_split"] = getattr(self.config.data, "val_split", None)
            data_dict["test_split"] = getattr(self.config.data, "test_split", None)
            data_dict["sequence_length"] = getattr(self.config.data, "sequence_length", None)
            data_dict["input_encoding"] = getattr(self.config.data, "input_encoding", "raw")
        except Exception:
            pass

        # Autoregressive configuration
        try:
            ar_config = getattr(self.config.training, "ar", None)
            if ar_config:
                if isinstance(ar_config, dict):
                    ar_enabled = ar_config.get("enabled", False)
                else:
                    ar_enabled = getattr(ar_config, "enabled", False)

                if ar_enabled:
                    summary["autoregressive"] = {}
                    if isinstance(ar_config, dict):
                        summary["autoregressive"]["enabled"] = True
                        summary["autoregressive"]["mode"] = ar_config.get("mode", "next_token")
                        summary["autoregressive"]["time_steps_per_token"] = ar_config.get("time_steps_per_token", 1)
                        summary["autoregressive"]["start_timestep"] = ar_config.get("start_timestep", 0)
                        summary["autoregressive"]["token_pooling"] = ar_config.get("token_pooling", {"method": "final"})
                        summary["autoregressive"]["loss"] = ar_config.get("loss", "autoregressive_cross_entropy")
                        summary["autoregressive"]["loss_weight"] = ar_config.get("loss_weight", 1.0)
                    else:
                        summary["autoregressive"]["enabled"] = True
                        summary["autoregressive"]["mode"] = getattr(ar_config, "mode", "next_token")
                        summary["autoregressive"]["time_steps_per_token"] = getattr(ar_config, "time_steps_per_token", 1)
                        summary["autoregressive"]["start_timestep"] = getattr(ar_config, "start_timestep", 0)
                        summary["autoregressive"]["token_pooling"] = getattr(ar_config, "token_pooling", {"method": "final"})
                        summary["autoregressive"]["loss"] = getattr(ar_config, "loss", "autoregressive_cross_entropy")
                        summary["autoregressive"]["loss_weight"] = getattr(ar_config, "loss_weight", 1.0)
            elif getattr(self.config.training, "autoregressive", False):
                # Legacy autoregressive config
                summary["autoregressive"] = {
                    "enabled": True,
                    "mode": getattr(self.config.training, "autoregressive_mode", "next_token"),
                    "time_steps_per_token": getattr(self.config.training, "time_steps_per_token", 1),
                    "start_timestep": getattr(self.config.training, "autoregressive_start_timestep", 0),
                }
        except Exception:
            pass

        # Callbacks
        backend = str(getattr(self.config.model, "backend", "torch")).lower()
        if callbacks and isinstance(callbacks, list) and len(callbacks) > 0:
            active_callbacks: list[dict[str, Any]] = []
            for cb in callbacks:
                try:
                    cb_info: dict[str, Any] = {"name": cb.__class__.__name__}
                    # Add callback-specific details
                    if hasattr(cb, "scheduler_type"):
                        cb_info["scheduler_type"] = str(getattr(cb, "scheduler_type", None)) if getattr(cb, "scheduler_type", None) is not None else None
                    if hasattr(cb, "loss_name"):
                        cb_info["loss_name"] = str(getattr(cb, "loss_name", None)) if getattr(cb, "loss_name", None) is not None else None
                    if hasattr(cb, "start_scale") and hasattr(cb, "end_scale"):
                        cb_info["start_scale"] = getattr(cb, "start_scale", None)
                        cb_info["end_scale"] = getattr(cb, "end_scale", None)
                    if hasattr(cb, "connections"):
                        conns = getattr(cb, "connections", None)
                        if isinstance(conns, (list, tuple)):
                            cb_info["num_connections"] = len(conns)
                    if hasattr(cb, "bits"):
                        cb_info["bits"] = getattr(cb, "bits", None)
                    if hasattr(cb, "key"):
                        cb_info["key"] = str(getattr(cb, "key", None)) if getattr(cb, "key", None) is not None else None
                    active_callbacks.append(cb_info)
                except Exception:
                    pass
            callbacks_dict["active"] = active_callbacks
        elif backend == "jax" and hasattr(self.config, "callbacks"):
            try:
                cb_config = self.config.callbacks
                if cb_config:
                    configured: list[dict[str, Any]] = []
                    if hasattr(cb_config, "lr_scheduler") and cb_config.lr_scheduler:
                        lr_sched = cb_config.lr_scheduler
                        configured.append({"name": "LR Scheduler", "type": str(getattr(lr_sched, "type", "unknown"))})
                    if hasattr(cb_config, "qat") and getattr(cb_config.qat, "active", False):
                        configured.append({"name": "QAT", "bits": str(getattr(cb_config.qat, "bits", "?"))})
                    if configured:
                        callbacks_dict["configured"] = configured
            except Exception:
                pass

        # MLflow info
        try:
            summary["mlflow"] = {
                "active": bool(getattr(self.config.logging, "mlflow_active", False)),
                "tracking_uri": getattr(self.config.logging, "mlflow_tracking_uri", None),
            }
        except Exception:
            summary["mlflow"] = {"active": False}

        # Save to YAML file
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Saved training configuration summary to: {summary_path}")
        return summary_path

    def _print_training_summary(self, repeat: int, seed: int, repeat_log_dir: Path, repeat_ckpt_dir: Path, ckpt_path_to_resume: str | None = None, callbacks: list[pl.Callback] | None = None) -> None:
        """Print a formatted training summary to the terminal (not log file)."""

        # Save to clean YAML file (logged)
        summary_path = self._save_training_config_summary(repeat, seed, repeat_log_dir, repeat_ckpt_dir, ckpt_path_to_resume, callbacks)
        logger.info(f"Training configuration summary saved to: {summary_path}")

        # Print formatted summary to terminal (bypasses logger)
        print("\n" + "=" * 80, flush=True)
        print("🚀 SOEN TRAINING SUMMARY", flush=True)
        print("=" * 80, flush=True)

        # Basic experiment info
        experiment_name = self.config.logging.experiment_name or "default_experiment"
        project_name = self.config.logging.project_name or "default_project"
        group_name = self.config.logging.group_name or "default_group"

        print(f"📋 Project Name:      {project_name}", flush=True)
        print(f"🧪 Experiment Name:   {experiment_name}", flush=True)
        print(f"👥 Group Name:        {group_name}", flush=True)
        print(f"🔢 Repeat:            {repeat + 1}/{self.config.training.num_repeats}", flush=True)
        print(f"🎲 Seed:              {seed}", flush=True)

        # Training configuration
        print("\n⚙️  TRAINING CONFIGURATION", flush=True)
        print(f"   • Max Epochs:      {self.config.training.max_epochs}", flush=True)
        print(f"   • Batch Size:      {self.config.training.batch_size}", flush=True)

        # Optimizer (handle both dict and object formats)
        try:
            opt_config = self.config.training.optimizer
            if isinstance(opt_config, dict):
                opt_name = opt_config.get("name", "unknown")
                opt_lr = opt_config.get("lr", "unknown")
            else:
                opt_name = getattr(opt_config, "name", "unknown")
                opt_lr = getattr(opt_config, "lr", "unknown")
            print(f"   • Optimizer:       {opt_name}", flush=True)
            print(f"   • Learning Rate:   {opt_lr}", flush=True)
        except Exception:
            print("   • Optimizer:       not configured", flush=True)

        print(f"   • Accelerator:     {self.config.training.accelerator}", flush=True)
        print(f"   • Devices:         {self.config.training.devices}", flush=True)
        print(f"   • Precision:       {self.config.training.precision}", flush=True)

        # Data configuration
        print("\n📊 DATA CONFIGURATION", flush=True)
        print(f"   • Dataset Path:    {self.config.data.data_path}", flush=True)
        print(f"   • Num Classes:     {self.config.data.num_classes}", flush=True)
        print(f"   • Val Split:       {getattr(self.config.data, 'val_split', 'N/A')}", flush=True)
        print(f"   • Test Split:      {getattr(self.config.data, 'test_split', 'N/A')}", flush=True)

        # Model configuration
        print("\n🧠 MODEL CONFIGURATION", flush=True)
        if self.config.model.base_model_path:
            print(f"   • Base Model:      {self.config.model.base_model_path}", flush=True)
        backend = str(getattr(self.config.model, "backend", "torch")).lower()
        print(f"   • Backend:         {backend}", flush=True)
        time_pooling = getattr(self.config.model, "time_pooling", "N/A")
        print(f"   • Time Pooling:    {time_pooling}", flush=True)

        # Paths and logging
        print("\n📁 PATHS & LOGGING", flush=True)
        print(f"   • Checkpoints:     {repeat_ckpt_dir}", flush=True)
        print(f"   • Logs:            {repeat_log_dir}", flush=True)
        print(f"   • Log Frequency:   Every {self.config.logging.log_freq} steps", flush=True)

        # Resume information
        if ckpt_path_to_resume:
            print("\n🔄 RESUMING FROM CHECKPOINT", flush=True)
            print(f"   • Checkpoint:      {ckpt_path_to_resume}", flush=True)

        print("=" * 80, flush=True)
        print("\n", flush=True)

    def _resolve_mlflow_tracking_uri(self) -> str | None:
        """Ensure mlflow_tracking_uri is populated when MLflow is active."""
        if not bool(getattr(self.config.logging, "mlflow_active", False)):
            return None

        tracking_uri = getattr(self.config.logging, "mlflow_tracking_uri", None)
        if tracking_uri:
            return tracking_uri

        import os as _os_env

        candidates = [
            _os_env.environ.get("MLFLOW_TRACKING_URI"),
            _os_env.environ.get("SOEN_MLFLOW_TRACKING_URI"),
            _os_env.environ.get("SOEN_TOOLKIT_DEFAULT_MLFLOW_URI"),
        ]
        for cand in candidates:
            if cand:
                tracking_uri = cand
                break
        if not tracking_uri:
            tracking_uri = "https://mlflow-greatsky.duckdns.org"

        with contextlib.suppress(Exception):
            self.config.logging.mlflow_tracking_uri = tracking_uri
        return tracking_uri

    def _validate_mlflow_config(self) -> None:
        """Validate MLflow configuration and fail fast if misconfigured."""
        if not bool(getattr(self.config.logging, "mlflow_active", False)):
            return  # MLflow disabled, nothing to validate

        tracking_uri = self._resolve_mlflow_tracking_uri()
        if not tracking_uri:
            msg = "[ERROR] MLflow is enabled but mlflow_tracking_uri is not set. Set it to 'file:./mlruns' for local or your server URL."
            raise RuntimeError(
                msg,
            )

        # Check authentication for HTTP(S) servers
        import os as _os_val

        needs_auth = tracking_uri.startswith(("http://", "https://")) and "@" not in tracking_uri
        env_username = _os_val.environ.get("MLFLOW_TRACKING_USERNAME")
        env_password = _os_val.environ.get("MLFLOW_TRACKING_PASSWORD")
        has_env_auth = bool(env_username) and bool(env_password)
        cfg_username = getattr(self.config.logging, "mlflow_username", None)
        cfg_password = getattr(self.config.logging, "mlflow_password", None)
        has_config_auth = bool(cfg_username) and bool(cfg_password)

        if needs_auth and not has_env_auth and not has_config_auth:
            msg = (
                f"[ERROR] MLflow server '{tracking_uri}' requires authentication. "
                "Add to your config:\n"
                "  mlflow_password: your_password\n"
                "Or set MLFLOW_TRACKING_PASSWORD env var (username defaults to 'admin')."
            )
            raise RuntimeError(
                msg,
            )

        logger.info("[OK] MLflow configuration validated successfully")

    def _print_mlflow_links(self, loggers: list[Any], repeat_name: str) -> None:
        """Print MLflow run and experiment links once per repeat.

        Uses the MLflow logger and client to construct the correct URLs.
        """
        if not bool(getattr(self.config.logging, "mlflow_active", False)):
            return

        # Identify an MLflow logger
        ml_logger = None
        try:
            for lg in loggers or []:
                if hasattr(lg, "experiment") and hasattr(lg, "run_id"):
                    ml_logger = lg
                    break
        except Exception:
            ml_logger = None

        if ml_logger is None:
            return

        # Build tracking URI
        tracking_uri = None
        try:
            tracking_uri = getattr(self.config.logging, "mlflow_tracking_uri", None)
        except Exception:
            tracking_uri = None
        if not tracking_uri:
            try:
                tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            except Exception:
                tracking_uri = None
        if not tracking_uri:
            return

        # Resolve experiment id from run id via MlflowClient
        run_id = None
        try:
            run_id = getattr(ml_logger, "run_id", None)
        except Exception:
            run_id = None
        if not run_id:
            return

        exp_id = None
        try:
            from mlflow.tracking import MlflowClient

            _client = MlflowClient()
            _run = _client.get_run(run_id)
            exp_id = _run.info.experiment_id
        except Exception:
            exp_id = None

        if not exp_id:
            return


def run_from_config(config_path: str | Path, script_dir: Path) -> None:
    """Run experiment from configuration file.

    Args:
        config_path: Path to configuration YAML file.
        script_dir: Path to the directory of the script calling this function.

    """
    # Suppress PL checkpoint directory warnings that clutter the console
    warnings.filterwarnings(
        "ignore",
        message=r"Checkpoint directory .* exists and is not empty.*",
        module="pytorch_lightning.callbacks.model_checkpoint",
        category=UserWarning,
    )

    # Load configuration
    # Note: Logging should already be configured by the entry point
    # We don't reconfigure it here to avoid destroying the application log file
    config = load_config(config_path)

    # Guard: S3 paths provided while not in cloud mode (local run)
    # Provide actionable AWS CLI commands to download locally
    try:

        def _in_sm_container() -> bool:
            try:
                return bool(
                    os.environ.get("SM_TRAINING_ENV") or os.environ.get("SM_CURRENT_HOST") or os.environ.get("TRAINING_JOB_NAME"),
                )
            except Exception:
                return False

        cloud_active = False
        try:
            cloud_active = bool(getattr(getattr(config, "cloud", object()), "active", False))
        except Exception:
            cloud_active = False
        legacy_toggle = False
        try:
            legacy_toggle = bool(getattr(config.training, "use_cloud", False))
        except Exception:
            legacy_toggle = False


        if not _in_sm_container() and (cloud_active or legacy_toggle):
            logger.warning(
                "\n" + "!" * 80 + "\n"
                "WARNING: Cloud execution requested (cloud.active=True) but running LOCALLY!\n"
                "This likely means you are calling 'run_from_config' directly without using the 'maybe_launch_on_sagemaker' launcher.\n"
                "If you intended to run on SageMaker, use the CLI: 'python -m soen_toolkit.training ...'\n"
                "or ensure your custom script handles cloud submission.\n"
                + "!" * 80 + "\n"
            )

        if not _in_sm_container() and not (cloud_active or legacy_toggle):
            s3_items: list[tuple[str, str]] = []
            try:
                ds_path = getattr(config.data, "data_path", None)
                if isinstance(ds_path, str) and _is_s3_uri(ds_path):
                    s3_items.append(("data_path", ds_path))
            except Exception:
                pass
            try:
                model_path = getattr(config.model, "base_model_path", None)
                if isinstance(model_path, str) and _is_s3_uri(model_path):
                    s3_items.append(("base_model_path", model_path))
            except Exception:
                pass

            if s3_items:
                for _field, uri in s3_items:
                    try:
                        _bucket, key = _split_s3_uri(uri)
                        base = key.split("/")[-1] if key else ""
                        if _s3_looks_file(uri) and base:
                            pass
                        else:
                            pass
                    except Exception:
                        pass
                raise SystemExit(2)
    except SystemExit:
        raise
    except Exception:
        # Do not block training for unexpected guard errors; proceed
        pass

    # Make deterministic mode non-fatal when CUDA lacks deterministic kernels
    try:
        import torch  # local import to avoid hard dependency at module load time

        det = bool(getattr(config.training, "deterministic", False))
        if det:
            # Prefer warn_only=True when available (PyTorch >= 2.3)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older versions: best-effort; avoid crashing later by disabling strict mode
                torch.use_deterministic_algorithms(False)
                logger.warning("Deterministic mode requested but warn_only unsupported; proceeding non-deterministically to avoid runtime errors.")
    except Exception:
        # If torch not available or any unexpected error occurs, continue
        pass

    # === Determine project_root_dir ===
    # Prefer an upward search for a known marker (pyproject.toml or .git). If none found,
    # default to the current working directory to ensure we have a writable, existing path.
    current_path = script_dir.resolve()
    project_root_dir_found = None
    for _ in range(len(current_path.parts) - 1):  # Limit upward search
        if (current_path / ".git").is_dir() or (current_path / "pyproject.toml").is_file():
            project_root_dir_found = current_path
            logger.info(f"Auto-detected project root at: {project_root_dir_found}")
            break
        if current_path.parent == current_path:  # Reached filesystem root
            break
        current_path = current_path.parent

    if not project_root_dir_found:
        project_root_dir_found = Path.cwd().resolve()
        logger.warning(
            f"Could not auto-detect project root. Defaulting to current working directory: {project_root_dir_found}",
        )

    # Create and run experiment, passing the script directory and detected project_root_dir
    runner = ExperimentRunner(config, script_dir=script_dir, project_root_dir=project_root_dir_found)
    runner.run_experiment()
