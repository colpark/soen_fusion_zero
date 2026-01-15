import logging

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


class LossWeightScheduler(Callback):
    """A general-purpose callback to schedule the weight of a specified loss component.

    Supports various scheduling functions like linear, exponential decay, and sinusoidal.

    The scheduler follows the same parameter structure as loss functions, with a 'params'
    dictionary containing scheduler-specific parameters.

    Attributes:
        loss_name (str): The name of the loss component to schedule.
        scheduler_type (str): The type of scheduler ("linear", "exponential_decay", "cosine_decay", "sinusoidal").
        params (dict): Dictionary containing scheduler-specific parameters.
        verbose (bool): If True, logs weight updates.

    """

    def __init__(
        self,
        loss_name: str,
        scheduler_type: str,
        params: dict | None = None,
        verbose: bool = True,
        # New optional schedule window controls
        start_epoch: int = 0,
        end_epoch: int | None = None,
        per_step: bool = True,
        # Optional UX: category grouping (e.g., 'periodic', 'decay')
        scheduler_category: str | None = None,
    ) -> None:
        super().__init__()
        self.loss_name = loss_name
        # Normalise scheduler type and allow common aliases/misspellings
        alias_map = {
            "exp_decay": "exponential_decay",
            "exponential": "exponential_decay",
            "cosine": "cosine_decay",
            "cosine_decay": "cosine_decay",
            "sinisoidal": "sinusoidal",  # common misspelling
        }
        self.scheduler_type = alias_map.get(str(scheduler_type).lower(), str(scheduler_type).lower())
        self.params = params or {}
        self.verbose = verbose
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch) if end_epoch is not None else None
        self.per_step = bool(per_step)
        self.scheduler_category = None
        if scheduler_category is not None:
            try:
                cat = str(scheduler_category).lower()
                if cat in {"periodic", "decay"}:
                    self.scheduler_category = cat
                else:
                    rank_zero_warn(f"[LossWeightScheduler] Unknown scheduler_category '{scheduler_category}'. Ignoring.")
            except Exception:
                pass

        if not loss_name:
            msg = "loss_name must be specified."
            raise ValueError(msg)
        if self.scheduler_type not in ["linear", "exponential_decay", "cosine_decay", "sinusoidal"]:
            msg = f"Unknown scheduler_type: {self.scheduler_type}"
            raise ValueError(msg)

        # Coerce common numeric params from strings if needed (robust YAML handling)
        self._coerce_params_types()

        # Validate scheduler-specific parameters
        self._validate_params()

        if self.verbose:
            cat_str = f", category='{self.scheduler_category}'" if self.scheduler_category else ""
            rank_zero_info(
                f"[LossWeightScheduler] Initialized for '{self.loss_name}' with type '{self.scheduler_type}'{cat_str}. Params: {self.params}",
            )

    def _coerce_params_types(self) -> None:
        """Coerce known parameter fields to appropriate numeric/boolean types if possible."""
        numeric_float_keys = {
            "min_weight",
            "max_weight",
            "initial_weight",
            "final_weight",
            "decay_rate",
            "amplitude_decay",
            "period_decay",
        }
        numeric_int_keys = {"period_steps", "period_epochs"}
        for key in list(self.params.keys()):
            val = self.params[key]
            try:
                if key in numeric_float_keys:
                    if isinstance(val, (str, int, float)):
                        self.params[key] = float(val)
                elif key in numeric_int_keys:
                    if isinstance(val, str):
                        self.params[key] = int(float(val))
                    elif isinstance(val, (int, float)):
                        self.params[key] = int(val)
                elif key == "scale":
                    self.params[key] = str(val)
            except Exception:
                # Leave as-is if conversion fails; validation step will catch issues
                pass

    def _validate_params(self) -> None:
        """Validate that required parameters are present for each scheduler type."""
        if self.scheduler_type == "linear":
            required = ["min_weight", "max_weight"]
            for param in required:
                if param not in self.params:
                    msg = f"Linear scheduler requires '{param}' in params"
                    raise ValueError(msg)

        elif self.scheduler_type == "exponential_decay":
            # Accept either (initial_weight & final_weight) or (initial_weight & decay_rate), both is fine too
            if "initial_weight" not in self.params:
                msg = "Exponential decay scheduler requires 'initial_weight'"
                raise ValueError(msg)
            if ("final_weight" not in self.params) and ("decay_rate" not in self.params):
                msg = "Exponential decay scheduler requires either 'final_weight' or 'decay_rate'"
                raise ValueError(msg)
            if "final_weight" in self.params:
                if self.params["initial_weight"] <= 0 or self.params["final_weight"] <= 0:
                    msg = "Exponential decay requires positive initial_weight and final_weight when using 'final_weight'"
                    raise ValueError(msg)

        elif self.scheduler_type == "cosine_decay":
            # Standard cosine decay between initial and final
            required = ["initial_weight", "final_weight"]
            for param in required:
                if param not in self.params:
                    msg = f"Cosine decay scheduler requires '{param}' in params"
                    raise ValueError(msg)

        elif self.scheduler_type == "sinusoidal":
            # Sinusoidal oscillation between min and max. Period can be specified in steps or epochs.
            required = ["min_weight", "max_weight"]
            for param in required:
                if param not in self.params:
                    msg = f"Sinusoidal scheduler requires '{param}' in params"
                    raise ValueError(msg)

            scale = self.params.get("scale", "linear")
            if scale not in ["linear", "log"]:
                msg = "For sinusoidal scheduler, scale must be 'linear' or 'log'"
                raise ValueError(msg)
            if scale == "log" and self.params["min_weight"] <= 0:
                msg = "For log scale, min_weight must be positive"
                raise ValueError(msg)
            # Optional UX params for periodic behavior
            amp_d = float(self.params.get("amplitude_decay", 0.0) or 0.0)
            per_d = float(self.params.get("period_decay", 1.0) or 1.0)
            if amp_d < 0:
                msg = "amplitude_decay must be non-negative"
                raise ValueError(msg)
            if per_d <= 0:
                msg = "period_decay must be > 0 and represents a multiplicative factor per cycle (e.g., 1.2)"
                raise ValueError(msg)

    def _progress_fraction(self, trainer: pl.Trainer) -> float:
        """Compute normalised progress in [0, 1] using start/end epoch and optionally per-step granularity."""
        # Determine schedule window in epochs
        max_epochs = getattr(trainer, "max_epochs", None)
        try:
            max_epochs = int(max_epochs) if max_epochs is not None else max(trainer.current_epoch + 1, 1)
        except Exception:
            max_epochs = max(trainer.current_epoch + 1, 1)

        end_epoch = self.end_epoch if self.end_epoch is not None else (max_epochs - 1)
        start_epoch = max(self.start_epoch, 0)

        if self.per_step:
            # Resolve steps/epoch robustly
            steps_per_epoch = getattr(trainer, "num_training_batches", None)
            if isinstance(steps_per_epoch, (list, tuple)):
                try:
                    steps_per_epoch = int(sum(int(x) for x in steps_per_epoch))
                except Exception:
                    steps_per_epoch = None
            else:
                try:
                    steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else None
                except Exception:
                    steps_per_epoch = None

            if not steps_per_epoch or steps_per_epoch <= 0:
                est_total = getattr(trainer, "estimated_stepping_batches", None)
                try:
                    est_total = int(est_total) if est_total is not None else None
                except Exception:
                    est_total = None
                if est_total and max_epochs > 0:
                    steps_per_epoch = max(1, round(est_total / max_epochs))
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

        # Per-epoch mode
        e = trainer.current_epoch
        if e <= start_epoch:
            return 0.0
        if e >= end_epoch:
            return 1.0
        return (e - start_epoch) / max(1, (end_epoch - start_epoch))

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: any,
        batch_idx: int,
    ) -> None:
        if not hasattr(pl_module, "active_loss_components"):
            return

        progress = self._progress_fraction(trainer)
        new_weight = self._calculate_weight(trainer, progress)

        for loss_component in pl_module.active_loss_components:
            if loss_component.get("name") == self.loss_name:
                old_weight = loss_component.get("weight", 0)
                loss_component["weight"] = new_weight

                # Log to TensorBoard under callbacks section (every 10 steps to avoid spam)
                if trainer.global_step % 10 == 0:
                    pl_module.log(
                        f"callbacks/loss_weight_{self.loss_name}",
                        new_weight,
                        prog_bar=False,
                        logger=True,
                        on_step=True,
                        on_epoch=False,
                    )

                if self.verbose and trainer.global_step % 100 == 0:
                    rank_zero_info(
                        f"[{self.__class__.__name__}] Step {trainer.global_step}: Updated {self.loss_name} weight from {old_weight:.6f} to {new_weight:.6f} (type={self.scheduler_type})",
                    )
                break

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Apply epoch-level updates when per_step=False."""
        if self.per_step or not hasattr(pl_module, "active_loss_components"):
            return
        progress = self._progress_fraction(trainer)
        new_weight = self._calculate_weight(trainer, progress)
        for loss_component in pl_module.active_loss_components:
            if loss_component.get("name") == self.loss_name:
                loss_component["weight"] = new_weight
                # Epoch-level log
                pl_module.log(
                    f"callbacks/loss_weight_{self.loss_name}",
                    new_weight,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                )
                break

    def _calculate_weight(self, trainer: pl.Trainer, progress: float) -> float:
        """Calculate the weight based on the scheduler type and normalised progress in [0, 1].

        For sinusoidal schedules, if 'period_steps' (per-step mode) or 'period_epochs' (per-epoch mode)
        are provided, multiple oscillation cycles are performed within the schedule window.
        """
        progress = max(0.0, min(1.0, float(progress)))

        if self.scheduler_type == "linear":
            min_weight = self.params["min_weight"]
            max_weight = self.params["max_weight"]
            return min_weight + progress * (max_weight - min_weight)

        if self.scheduler_type == "exponential_decay":
            initial_weight = float(self.params["initial_weight"])
            decay_rate = self.params.get("decay_rate", None)
            final_weight = self.params.get("final_weight", None)
            if decay_rate is None and final_weight is not None:
                # Compute rate so that progress=1 yields final_weight
                initial_safe = max(initial_weight, 1e-12)
                final_safe = max(float(final_weight), 1e-12)
                decay_rate = -np.log(final_safe / initial_safe)
            if decay_rate is None:
                # Should not happen due to validation, but guard anyway
                decay_rate = 0.0
            return float(initial_weight) * float(np.exp(-float(decay_rate) * float(progress)))

        if self.scheduler_type == "cosine_decay":
            initial_weight = float(self.params["initial_weight"])
            final_weight = float(self.params["final_weight"])
            # Standard cosine decay from initial to final over progress in [0,1]
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return float(final_weight + (initial_weight - final_weight) * cosine)

        if self.scheduler_type == "sinusoidal":
            min_weight = self.params["min_weight"]
            max_weight = self.params["max_weight"]
            scale = self.params.get("scale", "linear")

            # Determine number of oscillation cycles across the schedule window
            try:
                if self.per_step:
                    # Compute total steps in the schedule window
                    max_epochs = getattr(trainer, "max_epochs", None)
                    try:
                        max_epochs = int(max_epochs) if max_epochs is not None else max(trainer.current_epoch + 1, 1)
                    except Exception:
                        max_epochs = max(trainer.current_epoch + 1, 1)
                    end_epoch = self.end_epoch if self.end_epoch is not None else (max_epochs - 1)
                    start_epoch = max(self.start_epoch, 0)

                    steps_per_epoch = getattr(trainer, "num_training_batches", None)
                    if isinstance(steps_per_epoch, (list, tuple)):
                        try:
                            steps_per_epoch = int(sum(int(x) for x in steps_per_epoch))
                        except Exception:
                            steps_per_epoch = None
                    else:
                        try:
                            steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else None
                        except Exception:
                            steps_per_epoch = None
                    if not steps_per_epoch or steps_per_epoch <= 0:
                        est_total = getattr(trainer, "estimated_stepping_batches", None)
                        try:
                            est_total = int(est_total) if est_total is not None else None
                        except Exception:
                            est_total = None
                        if est_total and max_epochs > 0:
                            steps_per_epoch = max(1, round(est_total / max_epochs))
                        else:
                            steps_per_epoch = 1

                    total_steps = max(1, (max(end_epoch, start_epoch) - start_epoch + 1) * steps_per_epoch)
                    period_steps = int(self.params.get("period_steps", total_steps))
                    period_steps = max(1, period_steps)
                    float(total_steps) / float(period_steps)
                else:
                    # Epoch-based oscillation
                    max_epochs = getattr(trainer, "max_epochs", None)
                    try:
                        max_epochs = int(max_epochs) if max_epochs is not None else max(trainer.current_epoch + 1, 1)
                    except Exception:
                        max_epochs = max(trainer.current_epoch + 1, 1)
                    end_epoch = self.end_epoch if self.end_epoch is not None else (max_epochs - 1)
                    start_epoch = max(self.start_epoch, 0)
                    total_epochs = max(1, (end_epoch - start_epoch + 1))
                    period_epochs = int(self.params.get("period_epochs", total_epochs))
                    period_epochs = max(1, period_epochs)
                    float(total_epochs) / float(period_epochs)
            except Exception:
                pass

            # Compute angle using geometric period growth/shrink per completed cycle
            # period_decay r: >1 → each subsequent cycle is longer by factor r; <1 → shorter
            r = float(self.params.get("period_decay", 1.0) or 1.0)
            if r <= 0:
                r = 1.0

            # Determine elapsed units (steps or epochs) since schedule window start
            # Use progress fraction scaled by total units to avoid hidden dependencies
            if self.per_step:
                total_units = float(max(1, int(total_steps)))
                elapsed = float(progress) * total_units
                base_period = float(int(self.params.get("period_steps", max(1, int(total_steps)))))
            else:
                total_units = float(max(1, int(total_epochs)))
                elapsed = float(progress) * total_units
                base_period = float(int(self.params.get("period_epochs", max(1, int(total_epochs)))))

            base_period = max(1.0, base_period)

            if abs(r - 1.0) < 1e-12:
                # Constant period
                cycles_completed = elapsed / base_period
                angle = 2.0 * np.pi * cycles_completed
            else:
                # Invert cumulative time for n cycles: S(n) = P0 * (r^n - 1) / (r - 1)
                try:
                    n_float = np.log1p((r - 1.0) * (elapsed / base_period)) / np.log(r)
                except Exception:
                    n_float = elapsed / base_period
                if not np.isfinite(n_float):
                    n_float = elapsed / base_period
                k = int(np.floor(max(0.0, n_float)))
                # Cumulative time for k full cycles
                try:
                    cum_k = base_period * (pow(r, k) - 1.0) / (r - 1.0)
                except Exception:
                    cum_k = k * base_period
                # Current period at cycle k
                try:
                    current_period = base_period * pow(r, k)
                except Exception:
                    current_period = base_period
                rem = max(0.0, elapsed - max(0.0, float(cum_k)))
                frac = rem / max(1e-12, float(current_period))
                cycles_completed = float(k) + float(frac)
                angle = 2.0 * np.pi * cycles_completed

            cos_wave = np.cos(angle)

            # Amplitude with optional exponential decay over cycles completed
            base_amplitude = (max_weight - min_weight) / 2
            midpoint = (max_weight + min_weight) / 2
            amplitude_decay = float(self.params.get("amplitude_decay", 0.0) or 0.0)
            if amplitude_decay > 0:
                try:
                    amp = base_amplitude * np.exp(-amplitude_decay * cycles_completed)
                except Exception:
                    amp = base_amplitude
            else:
                amp = base_amplitude

            if scale == "linear":
                return midpoint + amp * cos_wave

            if scale == "log":
                log_min = np.log(min_weight)
                log_max = np.log(max_weight)
                progress_wave = (cos_wave + 1) / 2
                # Apply amplitude decay in log-space by narrowing the span
                span = log_max - log_min
                if base_amplitude > 0:
                    # scale span proportionally to amp/base_amplitude
                    span = span * float(amp / base_amplitude)
                log_weight = log_min + progress_wave * span
                return np.exp(log_weight)

        return 1.0  # Default fallback
