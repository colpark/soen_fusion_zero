# FILEPATH: src/soen_toolkit/training/callbacks/time_pooling_scale_scheduler.py

"""Time pooling scale scheduler callback for SOEN training.

This callback linearly interpolates the scale parameter in the time_pooling configuration
of the LightningModule's time pooling during training.
"""

import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


class TimePoolingScaleScheduler(Callback):
    """A callback to schedule the time_pooling scale parameter during training.

    This scheduler linearly interpolates between a start_scale and end_scale value
    over the course of training epochs.

    Attributes:
        start_scale (float): Initial scale value at the beginning of training.
        end_scale (float): Final scale value at the end of training.
        start_epoch (int): Epoch to start changing the scale (default: 0).
        end_epoch (int): Epoch to finish changing the scale (default: max_epochs).
        verbose (bool): If True, logs scale updates.

    """

    def __init__(
        self,
        start_scale: float,
        end_scale: float,
        start_epoch: int = 0,
        end_epoch: int | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.start_scale = float(start_scale)
        self.end_scale = float(end_scale)
        self.start_epoch = int(start_epoch)
        self.end_epoch = end_epoch  # Will be set in setup if None
        self.verbose = verbose

        if self.start_scale <= 0:
            msg = "start_scale must be positive"
            raise ValueError(msg)
        if self.end_scale <= 0:
            msg = "end_scale must be positive"
            raise ValueError(msg)
        if self.start_epoch < 0:
            msg = "start_epoch must be non-negative"
            raise ValueError(msg)

        if self.verbose:
            rank_zero_info(
                f"[TimePoolingScaleScheduler] Initialized: {self.start_scale} → {self.end_scale} from epoch {self.start_epoch} to {self.end_epoch or 'max_epochs'}",
            )

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Setup callback - set end_epoch if not provided."""
        if self.end_epoch is None:
            self.end_epoch = trainer.max_epochs
            if self.verbose:
                rank_zero_info(f"[TimePoolingScaleScheduler] Set end_epoch to {self.end_epoch}")

        if self.start_epoch >= self.end_epoch:
            rank_zero_warn(
                f"[TimePoolingScaleScheduler] start_epoch ({self.start_epoch}) >= end_epoch ({self.end_epoch}). Scale will not change during training.",
            )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update the time_pooling scale at the start of each epoch."""
        current_epoch = trainer.current_epoch

        # Calculate the new scale value
        new_scale = self._calculate_scale(current_epoch)

        # Update the model's time_pooling scale
        self._update_model_scale(pl_module, new_scale)

        # Log to TensorBoard under callbacks section
        pl_module.log(
            "callbacks/time_pooling_scale",
            new_scale,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        if self.verbose and current_epoch >= self.start_epoch:
            rank_zero_info(
                f"[TimePoolingScaleScheduler] Epoch {current_epoch}: Updated time_pooling scale to {new_scale:.6f}",
            )

    def _calculate_scale(self, current_epoch: int) -> float:
        """Calculate the scale value for the current epoch using linear interpolation."""
        if current_epoch < self.start_epoch:
            return self.start_scale
        if current_epoch >= self.end_epoch:
            return self.end_scale
        # Linear interpolation
        progress = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.start_scale + progress * (self.end_scale - self.start_scale)

    def _update_model_scale(self, pl_module: pl.LightningModule, new_scale: float) -> None:
        """Update the time_pooling scale parameter in the model."""
        try:
            # Access the LightningModule's time_pooling_params
            if hasattr(pl_module, "time_pooling_params"):
                # Update the scale parameter in the time_pooling_params dictionary
                pl_module.time_pooling_params["scale"] = new_scale

                if self.verbose and hasattr(pl_module, "time_pooling_method_name"):
                    method_name = pl_module.time_pooling_method_name
                    logger.debug(f"Updated scale to {new_scale:.6f} for {method_name} time pooling")
            else:
                rank_zero_warn(
                    "[TimePoolingScaleScheduler] Could not find time_pooling_params on LightningModule.",
                )
        except Exception as e:
            rank_zero_warn(f"[TimePoolingScaleScheduler] Failed to update model scale: {e}")

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "start_scale": self.start_scale,
            "end_scale": self.end_scale,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "verbose": self.verbose,
        }

    def load_state_dict(self, state_dict) -> None:
        """Load state dict from checkpoint."""
        self.start_scale = state_dict.get("start_scale", self.start_scale)
        self.end_scale = state_dict.get("end_scale", self.end_scale)
        self.start_epoch = state_dict.get("start_epoch", self.start_epoch)
        self.end_epoch = state_dict.get("end_epoch", self.end_epoch)
        self.verbose = state_dict.get("verbose", self.verbose)
