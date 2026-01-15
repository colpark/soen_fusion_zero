# FILEPATH: src/soen_toolkit/training/callbacks/state_trajectories.py

from collections.abc import Sequence
import logging

import matplotlib as mpl

mpl.use("Agg")
import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch

logger = logging.getLogger(__name__)


class StateTrajectoryLoggerCallback(Callback):
    """Log example state trajectories to TensorBoard at epoch end.

    Features:
    - Select layer by `layer_id` (default: last/output layer)
    - Select number of sample inputs to visualize
    - Optional class filtering for classification tasks (choose specific class IDs)
    - Plot up to `max_neurons_per_sample` neuron traces per sample (first dims by default or user-provided indices)

    Notes:
    - Requires temporarily enabling `track_s` for the model to collect per-timestep state histories.
    - Uses a small cached batch of examples from the validation dataloader (default) or training, based on `mode`.

    """

    def __init__(
        self,
        *,
        mode: str = "val",
        layer_id: int | None = None,
        num_samples: int = 4,
        class_ids: Sequence[int] | None = None,
        max_neurons_per_sample: int = 4,
        neuron_indices: Sequence[int] | None = None,
        tag_prefix: str = "callbacks/state_trajectories",
    ) -> None:
        super().__init__()
        assert mode in {"train", "val"}
        self.mode = mode
        self.layer_id = layer_id
        self.num_samples = max(1, int(num_samples))
        self.class_ids = {int(c) for c in class_ids} if class_ids is not None else None
        self.max_neurons_per_sample = max(1, int(max_neurons_per_sample))
        self.neuron_indices = [int(i) for i in neuron_indices] if neuron_indices is not None else None
        self.tag_prefix = tag_prefix.rstrip("/")

        # Cached example tensors and labels
        self._cached_inputs: torch.Tensor | None = None  # [N, T, C]
        self._cached_labels: torch.Tensor | None = None  # [N]
        self._cached_layer_index: int | None = None

    # --------------------------- helpers ---------------------------
    def _resolve_layer_index(self, pl_module: pl.LightningModule) -> int:
        model = getattr(pl_module, "model", None)
        if model is None:
            msg = "StateTrajectoryLogger expects LightningModule to expose `.model` (SOENModelCore)"
            raise RuntimeError(msg)
        if self.layer_id is None:
            # Default to last/output layer
            return len(model.layers_config) - 1
        try:
            return next(i for i, cfg in enumerate(model.layers_config) if cfg.layer_id == self.layer_id)
        except StopIteration:
            msg = f"Layer id {self.layer_id} not found in model layers_config"
            raise ValueError(msg)

    def _prepare_examples(self, trainer: pl.Trainer, *, refresh: bool = False) -> None:
        # Always refresh when requested so we use the latest dataset (e.g., after seq-len scheduler updates)
        if not refresh and self._cached_inputs is not None:
            return
        dl = None
        # Prefer datamodule if present
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            if self.mode == "val" and hasattr(trainer.datamodule, "val_dataloader"):
                dl = trainer.datamodule.val_dataloader()
            elif self.mode == "train" and hasattr(trainer.datamodule, "train_dataloader"):
                dl = trainer.datamodule.train_dataloader()

        if dl is None:
            logger.warning("StateTrajectoryLogger could not access a dataloader; skipping example preparation.")
            return

        selected_x: list[torch.Tensor] = []
        selected_y: list[torch.Tensor] = []

        try:
            for batch in dl:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    # If labels are not available, treat as regression/no labels
                    x, y = batch, None

                # Flatten batch dimension and iterate until enough samples are collected
                if x is None:
                    continue
                # Accept shapes [B, T, C] or [T, C]; normalize to [B, T, C]
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                if x.dim() != 3:
                    # Unexpected format; skip batch
                    continue

                if y is not None:
                    if y.dim() == 0:
                        y = y.unsqueeze(0)
                    y = y.long()

                for i in range(x.shape[0]):
                    if self.class_ids is not None and y is not None:
                        label_val = int(y[i].item())
                        if label_val not in self.class_ids:
                            continue
                    selected_x.append(x[i].cpu())
                    if y is not None:
                        selected_y.append(y[i].cpu())
                    if len(selected_x) >= self.num_samples:
                        break
                if len(selected_x) >= self.num_samples:
                    break
        except Exception as e:
            logger.warning(f"Error while sampling examples for StateTrajectoryLogger: {e}")
            return

        # If class filtering yielded fewer than requested, allow any remaining
        if len(selected_x) < self.num_samples and self.class_ids is not None:
            try:
                for batch in dl:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        x, y = batch[0], batch[1]
                    else:
                        x, y = batch, None
                    if x is None:
                        continue
                    if x.dim() == 2:
                        x = x.unsqueeze(0)
                    if x.dim() != 3:
                        continue
                    if y is not None:
                        if y.dim() == 0:
                            y = y.unsqueeze(0)
                        y = y.long()
                    for i in range(x.shape[0]):
                        selected_x.append(x[i].cpu())
                        if y is not None:
                            selected_y.append(y[i].cpu())
                        if len(selected_x) >= self.num_samples:
                            break
                    if len(selected_x) >= self.num_samples:
                        break
            except Exception:
                pass

        if not selected_x:
            logger.info("StateTrajectoryLogger did not find any examples to cache.")
            return

        x_cat = torch.stack(selected_x, dim=0)  # [N, T, C]
        y_cat = torch.stack(selected_y, dim=0) if selected_y else None

        self._cached_inputs = x_cat
        self._cached_labels = y_cat

    @staticmethod
    def _plot_sample(
        s_history_btc: torch.Tensor,
        sample_index: int,
        label: int | None,
        neuron_indices: Sequence[int] | None,
        max_neurons: int,
        title_prefix: str,
        dt: float,
        epoch: int,
    ) -> plt.Figure:
        # s_history_btc: [B, T, C]
        sample = s_history_btc[sample_index]  # [T, C]
        T, C = sample.shape

        if neuron_indices is None:
            k = min(C, max_neurons)
            idxs = list(range(k))
        else:
            idxs = [i for i in neuron_indices if 0 <= i < C]
            if not idxs:
                k = min(C, max_neurons)
                idxs = list(range(k))

        fig, ax = plt.subplots(figsize=(6, 3))
        # Use discrete step index (1..T) as x-axis to reflect current seq length
        t = torch.arange(1, T + 1, dtype=torch.float32)
        for j in idxs:
            ax.plot(t.cpu().numpy(), sample[:, j].detach().cpu().numpy(), label=f"n{j}")
        ttl = f"{title_prefix} | epoch {epoch} | sample {sample_index}"
        if label is not None:
            ttl += f" | class {int(label)}"
        ax.set_title(ttl)
        ax.set_xlabel("timestep")
        ax.set_ylabel("s")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        fig.tight_layout()
        return fig

    def _log_figures(
        self,
        trainer: pl.Trainer,
        s_history_btc: torch.Tensor,
        labels: torch.Tensor | None,
        layer_id_for_tag: int,
        seq_len: int,
    ) -> None:
        if not trainer.logger or not hasattr(trainer.logger, "experiment"):
            return
        experiment = trainer.logger.experiment
        for i in range(s_history_btc.shape[0]):
            fig = self._plot_sample(
                s_history_btc,
                sample_index=i,
                label=int(labels[i].item()) if labels is not None else None,
                neuron_indices=self.neuron_indices,
                max_neurons=self.max_neurons_per_sample,
                title_prefix=f"Layer {layer_id_for_tag} trajectories",
                dt=1.0,  # unused when using discrete steps
                epoch=int(trainer.current_epoch),
            )
            tag = f"{self.tag_prefix}/layer_{layer_id_for_tag}/sample_{i}"
            try:
                # Always rasterize and add as image for maximum reliability
                fig.canvas.draw()
                try:
                    # Fast path for Agg backends
                    buffer, (w, h) = fig.canvas.print_to_buffer()  # type: ignore[attr-defined]
                    if w == 0 or h == 0:
                        msg = "print_to_buffer returned zero size"
                        raise RuntimeError(msg)
                    arr = np.frombuffer(buffer, dtype=np.uint8).copy().reshape((h, w, 4))
                    arr = arr[:, :, :3]
                except Exception:
                    # Portable fallback via PNG round-trip
                    bytes_io = io.BytesIO()
                    fig.savefig(bytes_io, format="png", dpi=150, bbox_inches="tight")
                    bytes_io.seek(0)
                    arr = plt.imread(bytes_io, format="png")
                    if arr.dtype != np.uint8:
                        arr = (arr * 255.0).clip(0, 255).astype(np.uint8, copy=True)
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    if arr.shape[2] == 4:
                        arr = arr[:, :, :3]
                # Ensure writable, contiguous array for safe from_numpy
                arr = np.ascontiguousarray(arr)
                if not arr.flags.writeable:
                    arr = arr.copy()
                img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8
                experiment.add_image(tag, img, global_step=trainer.global_step, dataformats="CHW")
            except Exception as e_img:
                logger.warning(f"Failed to log state trajectory image (sample {i}): {e_img}")
                try:
                    # As a last resort, log a small text note for debugging
                    experiment.add_text(f"{tag}/_error", str(e_img), global_step=trainer.global_step)
                except Exception:
                    pass
            finally:
                plt.close(fig)
        # Ensure data is flushed to disk so TensorBoard can pick it up immediately
        try:
            if hasattr(experiment, "flush"):
                experiment.flush()
        except Exception:
            pass

    def _run_and_collect(self, pl_module: pl.LightningModule) -> tuple[torch.Tensor | None, int | None]:
        # Inputs cached on CPU; move to device and run model forward with track_s enabled
        if self._cached_inputs is None:
            return None, None

        device = pl_module.device
        x = self._cached_inputs.to(device=device)

        model = pl_module.model
        # Resolve layer index lazily (model is available now)
        if self._cached_layer_index is None:
            self._cached_layer_index = self._resolve_layer_index(pl_module)

        # Save original tracking flags
        orig_track_s = getattr(model.sim_config, "track_s", False)  # type: ignore[union-attr]
        # Temporarily enable state tracking and clear any old histories
        model.set_tracking(track_s=True)  # type: ignore[union-attr, operator]
        for layer in model.layers:  # type: ignore[union-attr]
            if hasattr(layer, "_clear_state_history"):
                layer._clear_state_history()

        try:
            model.eval()  # type: ignore[union-attr]
            with torch.no_grad():
                _y, _all = model(x)  # type: ignore[operator]
                # Pull per-layer histories
                histories = model.get_state_history()  # type: ignore[union-attr, operator]
                layer_hist = histories[self._cached_layer_index]
                if layer_hist is None:
                    return None, None
                # layer_hist: [B, T, C]
                # Use the current input sequence length for the x-axis
                try:
                    seq_len = int(self._cached_inputs.shape[1])
                except Exception:
                    seq_len = layer_hist.shape[1]
                return layer_hist.detach().clone().cpu(), seq_len
        finally:
            # Restore tracking flag
            model.set_tracking(track_s=orig_track_s)  # type: ignore[union-attr, operator]

    # --------------------------- hooks ---------------------------
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.mode != "train":
            return
        if not trainer.is_global_zero:
            return
        # Always refresh examples each epoch so we use the latest dataset/seq length
        self._prepare_examples(trainer, refresh=True)
        s_hist, seq_len = self._run_and_collect(pl_module)
        if s_hist is None or seq_len is None:
            return
        layer_id_for_tag = self.layer_id if self.layer_id is not None else pl_module.model.layers_config[-1].layer_id  # type: ignore[union-attr, index]
        self._log_figures(trainer, s_hist, self._cached_labels, layer_id_for_tag, seq_len)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.mode != "val":
            return
        if not trainer.is_global_zero:
            return
        # Always refresh examples each epoch so we use the latest dataset/seq length
        self._prepare_examples(trainer, refresh=True)
        s_hist, seq_len = self._run_and_collect(pl_module)
        if s_hist is None or seq_len is None:
            return
        layer_id_for_tag = self.layer_id if self.layer_id is not None else pl_module.model.layers_config[-1].layer_id  # type: ignore[union-attr, index]
        self._log_figures(trainer, s_hist, self._cached_labels, layer_id_for_tag, seq_len)
