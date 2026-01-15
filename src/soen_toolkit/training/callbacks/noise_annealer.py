# FILEPATH: src/soen_toolkit/training/callbacks/noise_annealer.py

"""Noise annealing callback for SOEN model training."""

import contextlib
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class NoiseAnnealingCallback(Callback):
    """Anneal layer-wise noise or perturbation magnitudes over training.

    Supports both per-timestep stochastic noise ("noise") and per-forward fixed
    offsets ("perturb") for keys like 'bias_current', 'phi', 'g', 's', 'j', or
    extra parameter names like 'phi_offset'.

    Parameters
    ----------
    key: str
        Which quantity to anneal. Core keys: 'phi', 'g', 's', 'bias_current', 'j'.
        Any other key is treated as an "extra" (e.g., 'phi_offset', 'alpha', etc.).
    target: str
        Either 'noise' (GaussianNoise, applied each timestep) or 'perturb'
        (GaussianPerturbation, sampled once per forward and fixed over the sequence).
    start_value: float
        Starting magnitude (std for Gaussian). For 'perturb', this sets *_std.
    end_value: float
        Final magnitude (std) at the end of training.
    start_epoch: int
        Epoch to start annealing (inclusive). Defaults to 0.
    end_epoch: Optional[int]
        Epoch to end annealing (inclusive). If None, uses trainer.max_epochs - 1.
    per_step: bool
        If True, anneal per training step
        otherwise per epoch.
    relative: Optional[bool]
        Only for target='noise'. When True, scales GaussianNoise by |tensor| (relative mode).
        Cannot be combined with perturbations by design of the noise builder.
    percent_of_param: bool
        Only for target='perturb' and for keys that correspond to actual parameters
        (e.g., 'bias_current'). If True, interprets start/end values as fractions
        of the current parameter magnitude. Uses mean(abs(param)) as baseline.
    layer_ids: Optional[List[int]]
        If provided, apply only to these layers
        otherwise apply to all layers.
    verbose: bool
        Log updates periodically.

    """

    def __init__(
        self,
        *,
        key: str,
        target: str = "perturb",
        start_value: float = 0.0,
        end_value: float = 0.0,
        start_epoch: int = 0,
        end_epoch: int | None = None,
        per_step: bool = True,
        relative: bool | None = None,
        percent_of_param: bool = False,
        layer_ids: list[int] | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        if target not in {"noise", "perturb"}:
            msg = "target must be 'noise' or 'perturb'"
            raise ValueError(msg)
        if target == "perturb" and relative:
            # The builder forbids mixing relative scaling with perturbations
            msg = "relative=True is only valid for target='noise'"
            raise ValueError(msg)

        self.key = key
        self.target = target
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_epoch = int(start_epoch)
        self.end_epoch = end_epoch
        self.per_step = bool(per_step)
        self.relative = bool(relative) if relative is not None else None
        self.percent_of_param = bool(percent_of_param)
        # Treat an empty list the same as "apply to all layers"
        if layer_ids is None or (isinstance(layer_ids, list) and len(layer_ids) == 0):
            self.layer_ids = None
        else:
            self.layer_ids = set(layer_ids)
        self.verbose = bool(verbose)

    def _progress_fraction(self, trainer: pl.Trainer) -> float:
        # Clamp to [0, 1]
        if self.per_step:
            # Compute step window corresponding to epoch range
            max_epochs = trainer.max_epochs if trainer.max_epochs is not None else max(trainer.current_epoch + 1, 1)
            end_epoch = self.end_epoch if self.end_epoch is not None else (max_epochs - 1)
            start_epoch = max(self.start_epoch, 0)

            # Steps per epoch: prefer trainer.num_training_batches, else derive
            steps_per_epoch = getattr(trainer, "num_training_batches", None)
            # Normalize/guard value from Trainer
            if isinstance(steps_per_epoch, (list, tuple)):
                try:
                    steps_per_epoch = int(sum(int(x) for x in steps_per_epoch))
                except Exception:
                    steps_per_epoch = None
            elif steps_per_epoch is not None:
                try:
                    steps_per_epoch = int(steps_per_epoch)
                except Exception:
                    steps_per_epoch = None

            if steps_per_epoch is None or steps_per_epoch <= 0:
                est_total = getattr(trainer, "estimated_stepping_batches", None)
                if est_total is not None and max_epochs > 0:
                    try:
                        steps_per_epoch = max(1, round(float(est_total) / max_epochs))
                    except Exception:
                        try:
                            steps_per_epoch = max(1, int(est_total))
                        except Exception:
                            steps_per_epoch = 1
                else:
                    steps_per_epoch = 1
            start_step = start_epoch * steps_per_epoch
            end_step = max(end_epoch, start_epoch) * steps_per_epoch + (steps_per_epoch - 1)

            step = trainer.global_step
            if step <= start_step:
                return 0.0
            if step >= end_step:
                return 1.0
            return (step - start_step) / max(1, (end_step - start_step))
        max_epochs = trainer.max_epochs if trainer.max_epochs is not None else max(trainer.current_epoch + 1, 1)
        end_epoch = self.end_epoch if self.end_epoch is not None else (max_epochs - 1)
        start_epoch = max(self.start_epoch, 0)

        e = trainer.current_epoch
        if e <= start_epoch:
            return 0.0
        if e >= end_epoch:
            return 1.0
        return (e - start_epoch) / max(1, (end_epoch - start_epoch))

    def _interp_value(self, frac: float) -> float:
        frac = max(0.0, min(1.0, float(frac)))
        return self.start_value + frac * (self.end_value - self.start_value)

    def _selected_layers(self, soen_model) -> list[int]:
        if self.layer_ids is None:
            return [cfg.layer_id for cfg in soen_model.layers_config]
        return [cfg.layer_id for cfg in soen_model.layers_config if cfg.layer_id in self.layer_ids]

    def _apply_to_layer_cfg(self, cfg, value: float, pl_module: pl.LightningModule) -> float:
        # Mutate the layer's Noise/Perturbation config in-place; SOENModelCore.forward
        # rebuilds NoiseSettings from these configs on every call.
        key = self.key

        if self.target == "noise":
            # Core keys live as attributes; extras go into cfg.noise.extras
            core_keys = {"phi", "g", "s", "bias_current", "j"}
            if hasattr(cfg, "noise") and cfg.noise is not None:
                n = cfg.noise
                if isinstance(n, dict):
                    if key in core_keys:
                        n[key] = float(value)
                        if self.relative is not None:
                            n["relative"] = bool(self.relative)
                    else:
                        extras = n.get("extras")
                        if not isinstance(extras, dict):
                            n["extras"] = {key: float(value)}
                        else:
                            extras[key] = float(value)
                elif key in core_keys:
                    setattr(n, key, float(value))
                    if self.relative is not None:
                        n.relative = bool(self.relative)
                else:
                    extras = getattr(n, "extras", None)
                    if extras is None:
                        n.extra = {key: float(value)} if hasattr(n, "extra") else None
                        # Prefer the documented 'extras' container
                        if hasattr(n, "extra"):
                            # already set above; also set 'extras' if present
                            pass
                        if hasattr(n, "extra") and not hasattr(n, "extras"):
                            # nothing else to do
                            pass
                    else:
                        extras[key] = float(value)
            else:
                logger.warning("LayerConfig.noise is None; cannot apply noise update.")
            return float(value)

        # perturb
        core_std_field = f"{key}_std"

        # If percent_of_param, rescale value by current parameter magnitude
        scaled_value = float(value)
        if self.percent_of_param:
            try:
                # Find the actual nn.Module for this layer to inspect current parameter
                soen_model = pl_module.model if hasattr(pl_module, "model") else None
                if soen_model is not None:
                    # Locate index of this layer
                    idx = next(i for i, c in enumerate(soen_model.layers_config) if c.layer_id == cfg.layer_id)
                    layer_module = soen_model.layers[idx]
                    if hasattr(layer_module, key):
                        param_tensor = getattr(layer_module, key)
                        return float(param_tensor.detach().cpu().mean().item())
            except Exception:
                pass

        if hasattr(cfg, "perturb") and cfg.perturb is not None:
            core_keys = {"phi", "g", "s", "bias_current", "j"}
            p = cfg.perturb
            if isinstance(p, dict):
                if key in core_keys:
                    p[core_std_field] = float(scaled_value)
                else:
                    extras_std = p.get("extras_std")
                    if not isinstance(extras_std, dict):
                        p["extras_std"] = {key: float(scaled_value)}
                    else:
                        extras_std[key] = float(scaled_value)
            elif key in core_keys:
                setattr(p, core_std_field, float(scaled_value))
            else:
                extras_std = getattr(p, "extras_std", None)
                if extras_std is None:
                    p.extras_std = {key: float(scaled_value)}
                else:
                    extras_std[key] = float(scaled_value)
        else:
            logger.warning("LayerConfig.perturb is None; cannot apply perturbation update.")
        return float(scaled_value)

    def _maybe_log(self, trainer: pl.Trainer, value: float) -> None:
        if self.verbose and trainer.global_step % 100 == 0:
            logger.info(
                f"[NoiseAnnealingCallback] step={trainer.global_step} epoch={trainer.current_epoch} key={self.key} target={self.target} value={value:.6f}",
            )

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx: int) -> None:
        if not self.per_step:
            return
        # Locate the core model
        if not hasattr(pl_module, "model"):
            return
        soen_model = pl_module.model
        frac = self._progress_fraction(trainer)
        value = self._interp_value(frac)
        # Apply and log (every 10 steps to reduce spam)
        for layer_id in self._selected_layers(soen_model):
            cfg = next(c for c in soen_model.layers_config if c.layer_id == layer_id)
            # Only act/log if the target key exists for this layer type
            layer_type = cfg.layer_type
            from soen_toolkit.core.layers.common.metadata import noise_keys_for_layer

            valid_keys = set(noise_keys_for_layer(layer_type))
            # For perturb on core noise keys, allow 'bias_current' as valid if present in param keys
            if self.key not in valid_keys:
                continue
            applied = self._apply_to_layer_cfg(cfg, value, pl_module)
            if trainer.global_step % 10 == 0:
                pl_module.log(
                    f"callbacks/noise/{self.key}_{self.target}/layer_{layer_id}",
                    applied,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                )
        self._maybe_log(trainer, value)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.per_step:
            return
        if not hasattr(pl_module, "model"):
            return
        soen_model = pl_module.model
        frac = self._progress_fraction(trainer)
        value = self._interp_value(frac)
        for layer_id in self._selected_layers(soen_model):
            cfg = next(c for c in soen_model.layers_config if c.layer_id == layer_id)
            from soen_toolkit.core.layers.common.metadata import noise_keys_for_layer

            valid_keys = set(noise_keys_for_layer(cfg.layer_type))
            if self.key not in valid_keys:
                continue
            applied = self._apply_to_layer_cfg(cfg, value, pl_module)
            # Log per-epoch value
            pl_module.log(
                f"callbacks/noise/{self.key}_{self.target}/layer_{layer_id}",
                applied,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
        self._maybe_log(trainer, value)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Always emit an epoch-level log so the curve continues after end_epoch.

        This runs for both per_step and per_epoch modes to keep a steady
        callbacks/noise/* epoch series.
        """
        if not hasattr(pl_module, "model"):
            return
        soen_model = pl_module.model
        frac = self._progress_fraction(trainer)
        value = self._interp_value(frac)
        from soen_toolkit.core.layers.common.metadata import noise_keys_for_layer

        for layer_id in self._selected_layers(soen_model):
            cfg = next(c for c in soen_model.layers_config if c.layer_id == layer_id)
            valid_keys = set(noise_keys_for_layer(cfg.layer_type))
            if self.key not in valid_keys:
                continue
            current_value = self._apply_to_layer_cfg(cfg, value, pl_module)
            with contextlib.suppress(Exception):
                pl_module.log(f"callbacks/noise/{self.key}_layer{layer_id}", float(current_value), on_step=False, on_epoch=True)
