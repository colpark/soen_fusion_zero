"""Output state statistics callback for SOEN model training."""

import contextlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch


class OutputStateStatsCallback(Callback):
    """Log distributions and summary stats of the output layer states.

    - Post time-pooling (processed, after optional scaling)
    - Pre-scale (undo the user-defined scale from time pooling)
    - Optional simple trajectory stats (min/mean/max over time for the last layer),
      if the model tracked state histories (sim_config.track_s=True)

    Parameters
    ----------
    mode: str
        'train' to log during training, 'val' to log during validation.
    every_n_steps: int
        Log every N steps (train mode) or on the final validation batch of the epoch (val mode).
    include_pre_scale: bool
        Log histogram of pre-scale (raw) processed outputs.
    include_post_scale: bool
        Log histogram of post-scale processed outputs.
    include_trajectory_stats: bool
        If True and state histories are tracked, log min/mean/max over time distributions for
        the last layer states (unscaled). If histories aren't available, this is skipped.
    hist_tag_prefix: str
        Prefix for TensorBoard tags. Default: 'callbacks/output_states'.

    """

    def __init__(
        self,
        *,
        mode: str = "train",
        every_n_steps: int = 200,
        include_pre_scale: bool = True,
        include_post_scale: bool = True,
        include_trajectory_stats: bool = False,
        hist_tag_prefix: str = "callbacks/output_states",
    ) -> None:
        super().__init__()
        mode = str(mode).lower().strip()
        if mode not in {"train", "val"}:
            msg = "mode must be 'train' or 'val'"
            raise ValueError(msg)
        self.mode = mode
        self.every_n_steps = int(every_n_steps)
        self.include_pre_scale = bool(include_pre_scale)
        self.include_post_scale = bool(include_post_scale)
        self.include_trajectory_stats = bool(include_trajectory_stats)
        self.hist_tag_prefix = hist_tag_prefix.rstrip("/")

    # --------------------------- helpers ---------------------------
    def _log_hist(self, pl_module: pl.LightningModule, tag: str, values: torch.Tensor) -> None:
        if pl_module.logger and hasattr(pl_module.logger, "experiment"):
            with contextlib.suppress(Exception):
                pl_module.logger.experiment.add_histogram(
                    tag,
                    values.detach().cpu().numpy().flatten(),
                    global_step=pl_module.trainer.global_step,
                )

    def _log_scalar(self, pl_module: pl.LightningModule, tag: str, value: float) -> None:
        pl_module.log(tag, value, on_step=(self.mode == "train"), on_epoch=(self.mode == "val"), prog_bar=False, logger=True)

    def _collect_processed_states(self, pl_module: pl.LightningModule) -> torch.Tensor | None:
        # Latest processed (pooled, scaled) states from the LightningModule
        proc = getattr(pl_module, "latest_processed_state", None)
        if proc is not None and torch.is_tensor(proc):
            return proc
        return None

    def _get_scale(self, pl_module: pl.LightningModule) -> float:
        scale = 1.0
        if hasattr(pl_module, "time_pooling_params"):
            try:
                scale = float(pl_module.time_pooling_params.get("scale", 1.0))
            except Exception:
                scale = 1.0
        return scale

    def _log_processed_distributions(self, pl_module: pl.LightningModule) -> None:
        proc = self._collect_processed_states(pl_module)
        if proc is None:
            return
        prefix = self.hist_tag_prefix

        # Post-scale
        if self.include_post_scale:
            self._log_hist(pl_module, f"{prefix}/post_scale_hist", proc)
            self._log_scalar(pl_module, f"{prefix}/post_scale_mean", float(proc.mean().item()))
            self._log_scalar(pl_module, f"{prefix}/post_scale_std", float(proc.std().item()))
            self._log_scalar(pl_module, f"{prefix}/post_scale_min", float(proc.min().item()))
            self._log_scalar(pl_module, f"{prefix}/post_scale_max", float(proc.max().item()))

        # Pre-scale (raw)
        if self.include_pre_scale:
            scale = self._get_scale(pl_module)
            raw = proc / scale if scale != 0.0 else proc
            self._log_hist(pl_module, f"{prefix}/pre_scale_hist", raw)
            self._log_scalar(pl_module, f"{prefix}/pre_scale_mean", float(raw.mean().item()))
            self._log_scalar(pl_module, f"{prefix}/pre_scale_std", float(raw.std().item()))
            self._log_scalar(pl_module, f"{prefix}/pre_scale_min", float(raw.min().item()))
            self._log_scalar(pl_module, f"{prefix}/pre_scale_max", float(raw.max().item()))

    def _log_trajectory_stats(self, pl_module: pl.LightningModule) -> None:
        if not hasattr(pl_module.model, "get_state_history"):
            # The SOEN core is stored under pl_module.model and exposes state histories
            return
        soen = pl_module.model
        if not getattr(soen.sim_config, "track_s", False):
            return
        try:
            # Get last layer state history: [B, T+1, D]
            histories = soen.get_state_history()
            last_hist = histories[-1]
            if last_hist is None or not torch.is_tensor(last_hist):
                return
            # Skip initial s0; compute across time dimension
            s = last_hist[:, 1:, :]
            s_min = s.amin(dim=1)
            s_max = s.amax(dim=1)
            s_mean = s.mean(dim=1)

            prefix = self.hist_tag_prefix + "/trajectory"
            self._log_hist(pl_module, f"{prefix}/min_over_time_hist", s_min)
            self._log_hist(pl_module, f"{prefix}/mean_over_time_hist", s_mean)
            self._log_hist(pl_module, f"{prefix}/max_over_time_hist", s_max)

            # Also scalar summary of distributions
            for name, tensor in [("min_over_time", s_min), ("mean_over_time", s_mean), ("max_over_time", s_max)]:
                self._log_scalar(pl_module, f"{prefix}/{name}_mean", float(tensor.mean().item()))
                self._log_scalar(pl_module, f"{prefix}/{name}_std", float(tensor.std().item()))
                self._log_scalar(pl_module, f"{prefix}/{name}_global_min", float(tensor.min().item()))
                self._log_scalar(pl_module, f"{prefix}/{name}_global_max", float(tensor.max().item()))
        except Exception:
            # Be silent if histories missing/mismatched; this is an optional feature
            pass

    # --------------------------- hooks ---------------------------
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if self.mode != "train":
            return
        if self.every_n_steps <= 0 or (trainer.global_step % self.every_n_steps) != 0:
            return
        self._log_processed_distributions(pl_module)
        if self.include_trajectory_stats:
            self._log_trajectory_stats(pl_module)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.mode != "val":
            return
        # Only log on last val batch to avoid spam
        val_dl = trainer.datamodule.val_dataloader() if hasattr(trainer, "datamodule") else None
        last_batch = False
        try:
            if val_dl is not None:
                total = len(val_dl)
                last_batch = batch_idx == total - 1
        except Exception:
            pass
        if not last_batch:
            return
        self._log_processed_distributions(pl_module)
        if self.include_trajectory_stats:
            self._log_trajectory_stats(pl_module)
