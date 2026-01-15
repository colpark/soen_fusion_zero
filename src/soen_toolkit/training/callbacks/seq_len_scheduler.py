from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from pytorch_lightning.callbacks import Callback

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from soen_toolkit.training.configs.data import SOENDataModule

logger = logging.getLogger(__name__)


class TargetSeqLenScheduler(Callback):
    """Adjust `data.target_seq_len` over epochs and rebuild cached dataset.

    Supports optional scheduling window [start_epoch, end_epoch].
    Before start_epoch the length stays at start_len
    after end_epoch it stays at end_len.
    """

    def __init__(
        self,
        data_module: SOENDataModule,
        *,
        start_len: int,
        end_len: int,
        max_epochs: int,
        start_epoch: int = 0,
        end_epoch: int | None = None,
        scale_dt: bool = False,
    ) -> None:
        super().__init__()
        self.dm = data_module
        self.start = start_len
        self.end = end_len
        self.max_epochs = max_epochs
        self.start_epoch = max(0, int(start_epoch))
        self.end_epoch = int(end_epoch) if end_epoch is not None else max_epochs
        self.end_epoch = max(self.end_epoch, self.start_epoch)
        window = max(1, self.end_epoch - self.start_epoch)
        self.delta = (end_len - start_len) / window if window > 0 else 0

        # dt scaling settings
        self.scale_dt = bool(scale_dt)
        self._last_len: int | None = None
        self._last_dt: float | None = None

        # Ensure datamodule uses caching so that each epoch we rebuild fresh
        # (cached) batches for the new sequence length.
        self.dm.config.data.cache_data = True

        # --- Apply the starting sequence length immediately so that the very
        # first dataloader built by Lightning uses the correct length.
        self.dm.config.data.target_seq_len = start_len

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        current_epoch = int(trainer.current_epoch)
        if current_epoch <= self.start_epoch:
            new_len = int(self.start)
        elif current_epoch >= self.end_epoch:
            new_len = int(self.end)
        else:
            progress = (current_epoch - self.start_epoch) / max(1, (self.end_epoch - self.start_epoch))
            new_len = round(self.start + progress * (self.end - self.start))

        prev_len = int(self.dm.config.data.target_seq_len)

        # If requested, scale dt to preserve total simulated time
        if self.scale_dt and new_len > 0:
            try:
                soen_model = getattr(pl_module, "model", None)
                if soen_model is not None and hasattr(soen_model, "dt") and hasattr(soen_model, "set_dt"):
                    current_dt = float(soen_model.dt)
                    # Initialize last values on first use
                    if self._last_len is None:
                        self._last_len = new_len
                        self._last_dt = current_dt
                    # Scale dt inversely with new length
                    if self._last_len > 0:
                        new_dt = current_dt * (self._last_len / float(new_len))
                        soen_model.set_dt(new_dt, propagate_to_layers=True)
                        # Log scaled dt
                        with contextlib.suppress(Exception):
                            pl_module.log("callbacks/seq_len_scheduler/dt", new_dt, on_step=False, on_epoch=True)
                        self._last_len = new_len
                        self._last_dt = new_dt
            except Exception:
                pass

        if new_len != prev_len:
            self.dm.update_target_seq_len(new_len)
            # Trigger dataloader rebuild if the API is available; otherwise rely on
            # Trainer(reload_dataloaders_every_n_epochs=1) configured by ExperimentRunner.
            try:
                if hasattr(trainer, "reset_train_dataloader"):
                    trainer.reset_train_dataloader(pl_module)
                if hasattr(trainer, "reset_val_dataloader"):
                    trainer.reset_val_dataloader(pl_module)
                if not (hasattr(trainer, "reset_train_dataloader") or hasattr(trainer, "reset_val_dataloader")):
                    logger.debug(
                        "reset_*_dataloader methods not available; relying on reload_dataloaders_every_n_epochs",
                    )
            except Exception as e:
                logger.debug("Non-fatal: could not reset dataloaders immediately: %s", e)
            # Log to TensorBoard under callbacks section
            pl_module.log(
                "callbacks/target_seq_len",
                float(new_len),
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

            # Backward compatibility: still log to old metric names if requested
            metrics_to_log = self.dm.config.logging.metrics
            if "train_target_seq_len" in metrics_to_log:
                pl_module.log("train_target_seq_len", float(new_len), prog_bar=False, logger=True)

            if "input_seq_len" in metrics_to_log:
                pl_module.log("input_seq_len", float(new_len), prog_bar=False, logger=True)
            logger.info("[SeqLenScheduler] Updated target_seq_len to %d (epoch %d)", new_len, trainer.current_epoch)
