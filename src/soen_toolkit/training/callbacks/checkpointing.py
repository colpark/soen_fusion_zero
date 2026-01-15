# FILEPATH: src/soen_toolkit/training/callbacks/checkpointing.py

"""Custom checkpointing callbacks for SOEN training."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from soen_toolkit.training.configs import ExperimentConfig
from soen_toolkit.training.models import SOENLightningModule

if TYPE_CHECKING:
    import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class SOENModelCheckpoint(ModelCheckpoint):
    """Checkpoint callback that maintains a 1:1 sidecar .soen file for each .ckpt.

    When enabled via config.training.save_soen_core_in_checkpoint, a sidecar file
    with the same basename as the checkpoint (but with .soen extension) is created
    and kept in sync with retained checkpoints (including last.ckpt).
    """

    def __init__(self, config: ExperimentConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    def _save_sidecar(self, ckpt_path: Path, pl_module: "pl.LightningModule") -> None:
        if not isinstance(pl_module, SOENLightningModule):
            logger.warning("SOEN sidecar save skipped: LightningModule is not SOENLightningModule.")
            return
        if not hasattr(pl_module, "model") or not hasattr(pl_module.model, "save"):
            logger.warning("SOEN sidecar save skipped: Expected pl_module.model.save not found.")
            return

        sidecar_path = ckpt_path.with_suffix(".soen")
        try:
            tmp_path = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
            pl_module.model.save(str(tmp_path))
            os.replace(tmp_path, sidecar_path)
            logger.info(f"Wrote SOEN sidecar for checkpoint: {sidecar_path}")
        except Exception as e:
            logger.error(f"Failed to write SOEN sidecar {sidecar_path}: {e}", exc_info=True)
        finally:
            try:
                if "tmp_path" in locals() and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _sync_sidecars(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Ensure 1:1 mapping of .ckpt <-> .soen in the checkpoint directory.

        - Create a .soen sidecar for any .ckpt missing one
        - Refresh the sidecar if the .ckpt is newer, or if it's the rolling 'last.ckpt'
        - Delete any .soen that has no corresponding .ckpt
        """
        if not self.config.training.save_soen_core_in_checkpoint:
            return

        # Resolve the directory where checkpoints are being written
        ckpt_dir = Path(self.dirpath) if self.dirpath is not None else Path(trainer.default_root_dir)
        if not ckpt_dir.exists():
            return

        ckpts = sorted(p for p in ckpt_dir.glob("*.ckpt") if p.is_file())
        sidecars = sorted(p for p in ckpt_dir.glob("*.soen") if p.is_file())

        ckpt_basenames = {p.with_suffix("").name for p in ckpts}
        {p.with_suffix("").name for p in sidecars}
        sidecar_by_base = {p.with_suffix("").name: p for p in sidecars}

        # Create missing sidecars and refresh outdated ones
        for ckpt in ckpts:
            base = ckpt.with_suffix("").name
            sidecar = sidecar_by_base.get(base)
            if sidecar is None:
                # No sidecar yet -> create it
                self._save_sidecar(ckpt, pl_module)
            else:
                # Refresh policy:
                # - Always refresh for rolling last.ckpt
                # - Or if the ckpt is newer than the sidecar
                try:
                    ckpt_mtime = ckpt.stat().st_mtime
                    sc_mtime = sidecar.stat().st_mtime
                except Exception:
                    ckpt_mtime = sc_mtime = 0.0
                if base == "last" or ckpt_mtime > sc_mtime:
                    self._save_sidecar(ckpt, pl_module)

        # Delete orphaned sidecars
        for sc in sidecars:
            base = sc.with_suffix("").name
            if base not in ckpt_basenames:
                try:
                    sc.unlink()
                    logger.info(f"Removed orphaned SOEN sidecar: {sc}")
                except Exception as e:
                    logger.warning(f"Failed to remove orphaned sidecar {sc}: {e}")

    # Hooks
    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict[str, Any]) -> None:
        # Let ModelCheckpoint handle file writing; we'll sync sidecars afterwards
        try:
            super().on_save_checkpoint(trainer, pl_module, checkpoint)
        except Exception:
            # Some PL versions may not require/allow calling super here; ignore
            pass

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # After validation ends, checkpoints (top-k/last) are materialized; sync sidecars
        try:
            try:
                super().on_validation_end(trainer, pl_module)
            except Exception:
                pass
            self._sync_sidecars(trainer, pl_module)
            # Optionally push artifacts to MLflow - logs are always uploaded, checkpoints respect mlflow_log_artifacts
            try:
                mlflow_active = bool(getattr(self.config.logging, "mlflow_active", False))
                mlflow_log_artifacts = bool(getattr(self.config.logging, "mlflow_log_artifacts", True))

                if not mlflow_active:
                    return

                # Support single or multiple loggers
                logger_objs = []
                if hasattr(trainer, "loggers") and trainer.loggers:
                    logger_objs = list(trainer.loggers)
                elif hasattr(trainer, "logger") and trainer.logger is not None:
                    logger_objs = [trainer.logger]

                # Collect artifacts to log from checkpoint directory and log directory
                ckpt_dir = Path(self.dirpath) if self.dirpath is not None else Path(trainer.default_root_dir)
                log_dir = Path(trainer.default_root_dir)

                # Checkpoints respect mlflow_log_artifacts setting
                checkpoint_artifacts: list[Path] = []
                if mlflow_log_artifacts:
                    try:
                        checkpoint_artifacts.extend(sorted(p for p in ckpt_dir.glob("*.ckpt") if p.is_file()))
                        checkpoint_artifacts.extend(sorted(p for p in ckpt_dir.glob("*.soen") if p.is_file()))
                    except Exception:
                        pass

                # Log files are ALWAYS uploaded (regardless of mlflow_log_artifacts setting)
                log_artifacts: list[Path] = []
                try:
                    log_artifacts.extend(sorted(p for p in log_dir.glob("*.log") if p.is_file()))
                except Exception:
                    pass

                # Combine: log files first, then checkpoints
                artifacts = log_artifacts + checkpoint_artifacts

                for lg in logger_objs:
                    try:
                        # Detect MLFlowLogger by attribute presence to avoid hard dependency
                        if hasattr(lg, "experiment") and hasattr(lg, "run_id"):
                            exp = lg.experiment
                            run_id = lg.run_id
                            for f in artifacts:
                                try:
                                    exp.log_artifact(run_id, str(f))
                                except Exception as e_file:
                                    logger.debug(f"MLflow artifact log skipped for {f}: {e_file}")
                    except Exception:
                        continue
            except Exception as e_art:
                logger.debug(f"MLflow artifact logging skipped: {e_art}")
        except Exception as e:
            logger.error(f"Error syncing SOEN sidecars with checkpoints: {e}", exc_info=True)


class InitialModelSaver(Callback):
    """Save an initial checkpoint and matching SOEN sidecar before training starts."""

    def __init__(self, config: ExperimentConfig, dirpath: Path) -> None:
        super().__init__()
        self.config = config
        self.dirpath = Path(dirpath)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ckpt_path = self.dirpath / "initial.ckpt"
        logger.info(f"Saving initial checkpoint to {ckpt_path}")
        trainer.save_checkpoint(str(ckpt_path))

        if not self.config.training.save_soen_core_in_checkpoint:
            return

        if isinstance(pl_module, SOENLightningModule) and hasattr(pl_module.model, "save"):
            sidecar_path = ckpt_path.with_suffix(".soen")
            try:
                pl_module.model.save(str(sidecar_path))
                logger.info(f"Saved initial SOEN sidecar to: {sidecar_path}")
            except Exception as e:
                logger.error(f"Failed to save initial SOEN sidecar: {e}", exc_info=True)
        else:
            logger.warning("Could not save initial SOEN sidecar: LightningModule structure not as expected.")
