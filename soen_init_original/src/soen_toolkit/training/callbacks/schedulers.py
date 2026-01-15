# FILEPATH: src/soen_toolkit/training/callbacks/schedulers.py

"""Learning rate scheduler implementations.

This module contains implementations of various learning rate schedulers
as PyTorch Lightning callbacks. Each scheduler is responsible for adjusting
the learning rate based on training progress.
"""

import contextlib
import math
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn

from soen_toolkit.training.callbacks.scheduler_core import (
    BaseLRScheduler,
    register_scheduler,
)


@register_scheduler("cosine")
class CosineAnnealingScheduler(BaseLRScheduler):
    """PyTorch Lightning Callback for Cosine Annealing with Warmup and Restarts.

    Args:
        max_lr: Upper learning rate boundary after warmup/at restart.
        min_lr: Lower learning rate boundary. Also starting LR during warmup if warmup_epochs > 0.
        warmup_epochs: Number of epochs for linear warmup from min_lr to max_lr.
        cycle_epochs: Number of epochs for the first cosine cycle (after warmup).
        enable_restarts: If True, restarts the cycle every `cycle_epochs`.
        restart_decay: Factor to decay `max_lr` after each restart (0.0 to 1.0). Default 1.0 (no decay).
        period_decay: Factor to modify cycle duration after each restart (1.0 = constant). Default 1.0.
        amplitude_decay: Factor to decay the oscillation amplitude (max_lr - min_lr) after each restart (1.0 = no decay). Default 1.0.
        adjust_on_batch: If True, adjust LR every batch
        otherwise, every epoch. Default True.
        batches_per_adjustment: How many batches between LR adjustments if `adjust_on_batch` is True. Default 1.
        soft_restart: If True, use smoother restart (like full cycle cosine)
        if False, use hard restart (half cycle). Default False.
        debug: Enable verbose logging for debugging. Default False.

    """

    def __init__(
        self,
        max_lr: float = 1e-3,
        min_lr: float = 1e-6,
        warmup_epochs: int = 5,
        cycle_epochs: int = 50,
        enable_restarts: bool = True,
        restart_decay: float = 1.0,
        period_decay: float = 1.0,
        amplitude_decay: float = 1.0,
        adjust_on_batch: bool = True,
        batches_per_adjustment: int = 1,
        soft_restart: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = max(0, warmup_epochs)
        self.cycle_epochs: int | float = max(1, cycle_epochs)
        self.initial_cycle_epochs: int | float = self.cycle_epochs
        self.enable_restarts = enable_restarts
        self.restart_decay = max(0.0, min(1.0, restart_decay))
        self.period_decay = max(0.0, period_decay)
        self.amplitude_decay = max(0.0, min(1.0, amplitude_decay))
        self.adjust_on_batch = adjust_on_batch
        self.batches_per_adjustment = max(1, batches_per_adjustment)
        self.soft_restart = soft_restart

        # Cosine state variables
        self.current_max_lr = max_lr
        self.initial_amplitude = max(0, max_lr - min_lr)
        self.current_lr = min_lr if self.warmup_epochs > 0 else max_lr
        self._num_restarts = 0
        self._total_batches_seen = 0
        self._batches_in_current_adjustment_period = 0
        self._current_cycle_progress_units: int | float = 0
        self._batches_per_epoch: int | None = None

        # Log initialization settings
        if self.debug:
            rank_zero_info("Initializing CosineAnnealingScheduler:")
            rank_zero_info(f"- max_lr: {self.max_lr:.2e}, min_lr: {self.min_lr:.2e}")
            rank_zero_info(f"- warmup_epochs: {self.warmup_epochs}")
            rank_zero_info(f"- initial_cycle_epochs: {self.initial_cycle_epochs}")
            rank_zero_info(f"- enable_restarts: {self.enable_restarts}")
            if self.enable_restarts:
                rank_zero_info(f"  - restart_decay: {self.restart_decay:.2f}")
                rank_zero_info(f"  - period_decay: {self.period_decay:.2f}")
                rank_zero_info(f"  - amplitude_decay: {self.amplitude_decay:.2f}")
            rank_zero_info(f"- adjust_on_batch: {self.adjust_on_batch}")
            if self.adjust_on_batch:
                rank_zero_info(f"  - batches_per_adjustment: {self.batches_per_adjustment}")
            rank_zero_info(f"- soft_restart: {self.soft_restart}")

        # Initialize state
        self.reset()

    def reset(self) -> None:
        """Reset Cosine-specific scheduler state."""
        self.current_max_lr = self.max_lr
        self.initial_amplitude = max(0, self.max_lr - self.min_lr)
        self.current_lr = self.min_lr if self.warmup_epochs > 0 else self.max_lr
        self._num_restarts = 0
        self._total_batches_seen = 0
        self._batches_in_current_adjustment_period = 0
        self._current_cycle_progress_units = 0
        if self.debug:
            rank_zero_info("[Cosine] CosineAnnealingScheduler state reset.")

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Initialize internal state based on trainer."""
        if stage != "fit":
            return

        # Determine batches per epoch only once
        if self._batches_per_epoch is None:
            estimated_batches = None
            # Prefer estimated_stepping_batches if available and finite
            if (
                hasattr(trainer, "estimated_stepping_batches")
                and isinstance(trainer.estimated_stepping_batches, (int, float))
                and math.isfinite(trainer.estimated_stepping_batches)
                and trainer.estimated_stepping_batches > 0
            ):
                estimated_batches = int(trainer.estimated_stepping_batches / max(1, trainer.max_epochs))  # type: ignore[operator,type-var]
            elif (
                hasattr(trainer, "num_training_batches") and isinstance(trainer.num_training_batches, (int, float)) and math.isfinite(trainer.num_training_batches) and trainer.num_training_batches > 0
            ):
                estimated_batches = int(trainer.num_training_batches)
            elif hasattr(trainer, "limit_train_batches"):
                limit = trainer.limit_train_batches
                full_batches = None
                # Try to get length from dataloader
                if hasattr(trainer, "train_dataloader") and trainer.train_dataloader is not None:
                    with contextlib.suppress(TypeError):
                        full_batches = len(trainer.train_dataloader)
                elif hasattr(trainer, "datamodule") and hasattr(trainer.datamodule, "train_dataloader"):
                    with contextlib.suppress(TypeError):
                        full_batches = len(trainer.datamodule.train_dataloader())

                if isinstance(limit, int) and limit > 0:
                    estimated_batches = limit
                elif isinstance(limit, float) and 0 < limit <= 1.0 and full_batches is not None:
                    estimated_batches = int(full_batches * limit)

            if estimated_batches is not None and estimated_batches > 0:
                self._batches_per_epoch = estimated_batches
            else:
                self._batches_per_epoch = 1000  # Fallback guess
                rank_zero_warn("[Cosine] Could not reliably determine batches_per_epoch. Using fallback: 1000.")

            self._batches_per_epoch = max(1, self._batches_per_epoch)
            rank_zero_info(f"[Cosine] Determined batches_per_epoch: {self._batches_per_epoch}")

        # Apply the current LR if starting fresh
        if trainer.current_epoch == 0 and self._total_batches_seen == 0:
            self._set_optimizer_lr(pl_module, self.current_lr)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Log initial state and ensure LR is set correctly."""
        self._set_optimizer_lr(pl_module, self.current_lr)  # Ensure LR is correct
        if self.debug:
            rank_zero_info(f"[Cosine] Training started. Initial LR set to {self.current_lr:.2e}. Batches/Epoch: {self._batches_per_epoch}")
            rank_zero_info(f"[Cosine] Initialized. MaxLR: {self.max_lr:.2e}, MinLR: {self.min_lr:.2e}, Warmup: {self.warmup_epochs} epochs, Initial Cycle: {self.initial_cycle_epochs} epochs")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Adjust LR at the start of each epoch if not adjusting per batch."""
        if not self.adjust_on_batch:
            # Equivalent epoch progress is just the current epoch number
            self._adjust_lr(trainer, pl_module, trainer.current_epoch)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch: Any, batch_idx: int) -> None:
        """Increment total batch counter and adjust LR if needed."""
        self._total_batches_seen += 1
        self._batches_in_current_adjustment_period += 1

        # Check if adjustment is due
        if self.adjust_on_batch and self._batches_in_current_adjustment_period >= self.batches_per_adjustment:
            if self._batches_per_epoch and self._batches_per_epoch > 0:
                # Equivalent epoch progress for batch-based adjustment
                equivalent_epoch = self._total_batches_seen / self._batches_per_epoch
                self._adjust_lr(trainer, pl_module, equivalent_epoch)
                self._batches_in_current_adjustment_period = 0  # Reset counter
            else:
                # This should not happen if setup is correct
                rank_zero_warn("[Cosine] Cannot adjust LR per batch: _batches_per_epoch is not valid.")

    def _adjust_lr(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", current_progress_metric: float) -> None:
        """Calculates and sets the learning rate based on the current progress metric
        (which can be epochs or equivalent epochs based on batches).
        """
        # --- Handle Warmup Phase ---
        warmup_batches = self.warmup_epochs * self._batches_per_epoch  # type: ignore[operator]
        is_in_warmup = self.warmup_epochs > 0 and self._total_batches_seen < warmup_batches

        if is_in_warmup:
            # Linear warmup from min_lr to max_lr based on batch progress
            warmup_progress = self._total_batches_seen / max(1, warmup_batches)  # Avoid div by zero
            lr = self.min_lr + (self.max_lr - self.min_lr) * warmup_progress
            # Reset cycle progress during warmup *only if just exiting warmup*
            if self._total_batches_seen >= warmup_batches - self.batches_per_adjustment:  # Check if near end
                self._current_cycle_progress_units = 0
        else:
            # --- Post-Warmup Phase (Cosine Annealing) ---
            # If just finished warmup, reset cycle progress units
            if self.warmup_epochs > 0 and self._total_batches_seen - self.batches_per_adjustment < warmup_batches <= self._total_batches_seen:
                self._current_cycle_progress_units = self._total_batches_seen - warmup_batches  # Start progress from batches after warmup
                if self.debug:
                    rank_zero_info(f"[Cosine] Warmup complete at batch {self._total_batches_seen}. Starting cosine cycle.")

            # Determine cycle length and max LR for the *current* restart cycle
            current_cycle_num = self._num_restarts
            current_cycle_len_epochs = self.initial_cycle_epochs * (self.period_decay**current_cycle_num)
            current_cycle_len_batches = current_cycle_len_epochs * self._batches_per_epoch  # type: ignore[operator]

            current_max_lr_decayed = self.max_lr * (self.restart_decay**current_cycle_num)
            current_amplitude = (current_max_lr_decayed - self.min_lr) * (self.amplitude_decay**current_cycle_num)
            # Ensure amplitude is not negative
            current_amplitude = max(0, current_amplitude)

            # Calculate progress within the current cycle (in batches)
            if current_cycle_len_batches <= 1e-6:  # Avoid division by zero/NaN
                progress_in_cycle = 1.0
            else:
                progress_in_cycle = self._current_cycle_progress_units / current_cycle_len_batches

            # Apply cosine formula (ensure progress is capped at 1.0)
            progress_in_cycle = min(1.0, max(0.0, progress_in_cycle))  # Clamp progress

            effective_amplitude = current_amplitude  # Default to current

            if self.soft_restart:
                # Full cycle (0..2pi)
                cosine_factor = (math.cos(2 * math.pi * progress_in_cycle) + 1) / 2

                # --- Amplitude Blending for Continuity with Decay ---
                # If decay is active, interpolate amplitude towards the next cycle's start amplitude
                if self.amplitude_decay < 1.0 and self.enable_restarts:
                    # Calculate the starting amplitude for the *next* cycle
                    next_cycle_num = current_cycle_num + 1
                    next_max_lr_decayed = self.max_lr * (self.restart_decay**next_cycle_num)
                    next_amplitude = (next_max_lr_decayed - self.min_lr) * (self.amplitude_decay**next_cycle_num)
                    next_amplitude = max(0, next_amplitude)

                    # Interpolate between current and next amplitude
                    effective_amplitude = current_amplitude * (1.0 - progress_in_cycle) + next_amplitude * progress_in_cycle
            else:
                # Hard restart (Half cycle 0 to pi)
                cosine_factor = (math.cos(math.pi * progress_in_cycle) + 1) / 2

            # Calculate LR using the potentially blended amplitude
            lr = self.min_lr + effective_amplitude * cosine_factor

            # Increment progress counter (always in batches) *before* checking for restart
            # Use the actual number of batches processed in this adjustment step
            increment_amount = self.batches_per_adjustment if self.adjust_on_batch else self._batches_per_epoch
            self._current_cycle_progress_units += increment_amount  # type: ignore[operator]

            # Check for cycle completion and handle restarts
            # Need >= check as progress might overshoot slightly
            if self.enable_restarts and self._current_cycle_progress_units >= current_cycle_len_batches and current_cycle_len_batches > 0:
                self._num_restarts += 1
                # Carry over the remainder progress into the next cycle
                self._current_cycle_progress_units = self._current_cycle_progress_units % current_cycle_len_batches

                # Update state for the *next* cycle calculation
                next_cycle_len_epochs = self.initial_cycle_epochs * (self.period_decay**self._num_restarts)
                next_max_lr = self.max_lr * (self.restart_decay**self._num_restarts)
                self.cycle_epochs = next_cycle_len_epochs  # Update internal state for saving/logging
                self.current_max_lr = next_max_lr  # Update internal state

                if self.debug:
                    rank_zero_info(f"[Cosine] Restart {self._num_restarts} at batch {self._total_batches_seen}. Next cycle: {next_cycle_len_epochs:.2f} epochs, MaxLR: {next_max_lr:.2e}")

        # Apply calculated LR
        final_lr = max(lr, self.min_lr)  # Ensure LR doesn't go below min_lr
        if abs(final_lr - self.current_lr) > 1e-9:  # Only set if changed significantly
            self._set_optimizer_lr(pl_module, final_lr)
            self.current_lr = final_lr


@register_scheduler("rex")
class RexScheduler(BaseLRScheduler):
    """REX (Rational EXponential) scheduler.

    Implements warmup (optional) followed by a smooth, principled decay profile.
    The scheduler uses a principled decay that is proportional to 1/sqrt(remaining_steps).

    Args:
        warmup_epochs: Number of epochs for linear warmup
        warmup_start_lr: Starting learning rate for warmup
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate (after warmup)
        debug: Enable verbose logging

    """

    def __init__(
        self,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 1e-6,
        min_lr: float = 0.0,
        max_lr: float | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.reset()

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.total_steps: int | float = 0
        self.step: int | float = 0
        self.warmup_steps: int | float = 0
        if self.debug:
            rank_zero_info("[REX] State reset.")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize scheduler at the start of training."""
        self.total_steps = trainer.max_epochs * trainer.num_training_batches  # type: ignore[operator, assignment]
        self.warmup_steps = self.warmup_epochs * trainer.num_training_batches  # type: ignore[operator, assignment]

        # Set initial LR to warmup_start_lr if using warmup
        if self.warmup_steps > 0:
            self._set_optimizer_lr(pl_module, self.warmup_start_lr)

        # Determine max_lr to use
        if self.max_lr is not None:
            optimizer = pl_module.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            for g in optimizer.param_groups:
                g["initial_lr"] = self.max_lr

        # Log settings
        if self.debug:
            msg = f"[REX] total_steps={self.total_steps}, warmup_steps={self.warmup_steps}"
            if self.warmup_steps > 0:
                optimizer = pl_module.optimizers()
                if isinstance(optimizer, list):
                    optimizer = optimizer[0]
                msg += f"\n[REX] Starting with lr={self.warmup_start_lr:.7f}, will increase to {optimizer.param_groups[0]['initial_lr']:.5f}"
            rank_zero_info(msg)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs) -> None:
        """Update learning rate at the end of each training batch."""
        self.step += 1
        optimizer = pl_module.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        initial_lr = optimizer.param_groups[0]["initial_lr"]
        min_lr = self.min_lr

        # Apply warmup if in warmup phase
        if self.step <= self.warmup_steps and self.warmup_steps > 0:
            progress = self.step / self.warmup_steps
            cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
            lr = self.warmup_start_lr + (initial_lr - self.warmup_start_lr) * cosine_factor
        else:
            # Decay from initial_lr to min_lr using REX formula
            # η_t = η_0 * (1 - t/T) / (1/2 + 1/2*(1 - t/T))
            t_norm = (self.step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            t_norm = min(1.0, max(0.0, t_norm))  # Clamp to [0, 1]
            term = 1.0 - t_norm
            rex_factor = term / (0.5 + 0.5 * term)
            lr = min_lr + (initial_lr - min_lr) * rex_factor

        # Apply new learning rate
        self._set_optimizer_lr(pl_module, lr)

        # Debug logging
        if self.debug and (self.step in (1, self.warmup_steps) or self.step % (self.total_steps // 10) == 0):
            rank_zero_info(f"[REX] step={self.step}/{self.total_steps} lr={lr:.7f}")


@register_scheduler("constant")
class ConstantLRScheduler(BaseLRScheduler):
    """Constant learning rate scheduler.

    This scheduler maintains a constant learning rate throughout training.
    It's useful as a baseline for comparison with other schedulers.

    Args:
        lr: Optional fixed learning rate (if None, uses optimizer's initial lr)
        debug: Enable verbose logging

    """

    def __init__(self, lr: float | None = None, debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.lr = lr

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Set fixed learning rate at the start of training if specified."""
        if self.lr is not None:
            self._set_optimizer_lr(pl_module, self.lr)

        # Always print initial LR
        if self.debug:
            current_lr = self._get_optimizer_lr(pl_module)
            rank_zero_info(f"[Constant] lr={current_lr:.5f}")


@register_scheduler("linear")
class LinearDecayScheduler(BaseLRScheduler):
    """Linear Decay Learning Rate scheduler.

    Linearly decreases the learning rate from max_lr to min_lr over all training steps.
    Can optionally perform the decay linearly in log space.

    Args:
        max_lr: Maximum learning rate (at the beginning)
        min_lr: Minimum learning rate (at the end)
        log_space: If True, decay linearly in log space. Default False.
        debug: Enable verbose logging

    """

    def __init__(
        self,
        max_lr: float = 1e-3,
        min_lr: float = 1e-6,
        log_space: bool = False,  # Add log_space parameter
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.log_space = log_space

        # Validate min_lr for log space
        if self.log_space and self.min_lr <= 0:
            msg = "min_lr must be positive when using log_space=True"
            raise ValueError(msg)

        self.reset()

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.total_steps: int | float = 0
        self.step: int | float = 0
        if self.debug:
            log_mode = " (log space)" if self.log_space else ""
            rank_zero_info(f"[Linear{log_mode}] State reset.")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize scheduler at the start of training."""
        # Calculate total steps from trainer info
        self.total_steps = trainer.max_epochs * trainer.num_training_batches  # type: ignore[operator, assignment]

        # Start with max_lr
        self._set_optimizer_lr(pl_module, self.max_lr)

        if self.debug:
            log_mode = " (log space)" if self.log_space else ""
            rank_zero_info(f"[Linear{log_mode}] Starting LR={self.max_lr:.6f}, will decrease to {self.min_lr:.6f} over {self.total_steps} steps")

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs) -> None:
        """Update learning rate at each training step."""
        self.step += 1

        if self.total_steps > 1:  # Avoid division by zero
            progress = min(1.0, self.step / self.total_steps)  # Ensure progress doesn't exceed 1.0

            if self.log_space:
                # Log-linear interpolation
                log_max_lr = math.log(self.max_lr)
                log_min_lr = math.log(self.min_lr)
                log_lr = log_max_lr - progress * (log_max_lr - log_min_lr)
                lr = math.exp(log_lr)
            else:
                # Simple linear interpolation
                lr = self.max_lr - progress * (self.max_lr - self.min_lr)
        else:
            lr = self.min_lr

        # Ensure lr doesn't go below min_lr (can happen due to float precision)
        lr = max(lr, self.min_lr)

        # Apply new learning rate
        self._set_optimizer_lr(pl_module, lr)

        # Debug logging
        if self.debug and (self.step == 1 or self.step % (self.total_steps // 10) == 0 or self.step == self.total_steps):
            log_mode = " (log space)" if self.log_space else ""
            rank_zero_info(f"[Linear{log_mode}] step={self.step}/{self.total_steps} progress={progress:.4f} lr={lr:.7f}")


@register_scheduler("greedy")
class GreedyScheduler(BaseLRScheduler):
    """Greedy learning rate scheduler that adaptively adjusts LR based on validation loss trends.

    This scheduler increases the learning rate by factor_increase when validation loss improves,
    and decreases it by factor_decrease when validation loss worsens. It can optionally include
    a warmup phase at the beginning of training.

    Args:
        factor_increase: Factor to multiply LR by when validation loss improves
        factor_decrease: Factor to multiply LR by when validation loss worsens
        patience: Number of validation epochs to wait before adjusting LR
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        warmup_epochs: Number of epochs for linear warmup
        warmup_start_lr: Starting learning rate for warmup
        intra_epoch: Whether to adjust LR within epochs based on training loss
        ema_beta: Beta value for EMA of training loss (only used if intra_epoch=True)
        debug: Enable verbose logging

    """

    def __init__(
        self,
        factor_increase: float = 1.1,
        factor_decrease: float = 0.9,
        patience: int = 3,
        min_lr: float = 1e-6,
        max_lr: float = 0.01,
        warmup: dict[str, Any] | None = None,
        intra_epoch: bool = False,
        adjustment_frequency: int = 100,  # Added parameter
        ema_beta: float = 0.9,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.factor_increase = factor_increase
        self.factor_decrease = factor_decrease
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adjustment_frequency = adjustment_frequency  # Store the parameter

        # Extract warmup parameters from dictionary
        if warmup is not None and isinstance(warmup, dict):
            self.warmup_epochs = warmup.get("epochs", 0)
            self.warmup_start_lr = warmup.get("start_lr", 1e-6)
            self.warmup_enabled = warmup.get("enabled", True)
        else:
            self.warmup_epochs = 0
            self.warmup_start_lr = 1e-6
            self.warmup_enabled = False

        self.intra_epoch = intra_epoch
        self.ema_beta = ema_beta

        # Initialize state variables
        self.reset()

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.current_lr = self.warmup_start_lr if self.warmup_epochs > 0 else self.max_lr
        self.warmup_steps = 0
        self.current_step = 0
        self.current_epoch = 0
        self.steps_per_epoch: int | None = None

        # For intra-epoch adjustments
        self.ema_loss: float | None = None
        self.prev_ema_loss: float | None = None
        self.batches_since_adjustment = 0

        if self.debug:
            rank_zero_info(f"[Greedy] Scheduler state reset. Initial LR: {self.current_lr:.6e}")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize scheduler at the start of training."""
        # Calculate steps per epoch from trainer
        if hasattr(trainer, "num_training_batches"):
            self.steps_per_epoch = int(trainer.num_training_batches)
        else:
            self.steps_per_epoch = 100  # Fallback estimate
            rank_zero_warn("[Greedy] Could not determine steps_per_epoch. Using fallback value: 100")

        # Calculate total warmup steps
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch

        # Set initial learning rate
        self._set_optimizer_lr(pl_module, self.current_lr)

        if self.debug:
            rank_zero_info(f"[Greedy] Training started. Initial LR: {self.current_lr:.6e}")
            rank_zero_info(f"[Greedy] Warmup steps: {self.warmup_steps}, Steps per epoch: {self.steps_per_epoch}")

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Handle per-batch updates during training."""
        self.current_step += 1

        # Handle warmup phase
        if self.current_step <= self.warmup_steps and self.warmup_steps > 0:
            # Linear warmup from warmup_start_lr to max_lr
            progress = self.current_step / self.warmup_steps
            new_lr = self.warmup_start_lr + progress * (self.max_lr - self.warmup_start_lr)
            self.current_lr = new_lr
            self._set_optimizer_lr(pl_module, new_lr)

            if self.debug and (self.current_step == 1 or self.current_step % (self.warmup_steps // 10) == 0):
                rank_zero_info(f"[Greedy] Warmup progress: {progress:.2f}, LR: {new_lr:.6e}")

            return

        # Intra-epoch LR adaptation based on training loss (if enabled)
        if self.intra_epoch and isinstance(outputs, dict) and "loss" in outputs:
            self.batches_since_adjustment += 1
            current_loss = outputs["loss"].item()

            # Initialize EMA loss on first batch
            if self.ema_loss is None:
                self.ema_loss = current_loss
            else:
                # Update exponential moving average
                self.ema_loss = self.ema_beta * self.ema_loss + (1 - self.ema_beta) * current_loss

            # Use the configured adjustment frequency
            adjustment_frequency = self.adjustment_frequency

            if self.batches_since_adjustment >= adjustment_frequency:
                # Compare with previous EMA loss to see if improving
                if hasattr(self, "prev_ema_loss"):
                    if self.ema_loss < self.prev_ema_loss * 0.99:  # 1% improvement threshold
                        # Training is improving, increase LR slightly
                        new_lr = min(self.current_lr * (self.factor_increase**0.2), self.max_lr)
                        if new_lr != self.current_lr:
                            self.current_lr = new_lr
                            self._set_optimizer_lr(pl_module, new_lr)
                            if self.debug:
                                rank_zero_info(f"[Greedy] Intra-epoch increase - Loss improved: {self.prev_ema_loss:.4f}->{self.ema_loss:.4f}, New LR: {new_lr:.6e}")
                    elif self.ema_loss > self.prev_ema_loss * 1.05:  # 5% worsening threshold
                        # Training is worsening, decrease LR
                        new_lr = max(self.current_lr * (self.factor_decrease**0.3), self.min_lr)
                        if new_lr != self.current_lr:
                            self.current_lr = new_lr
                            self._set_optimizer_lr(pl_module, new_lr)
                            if self.debug:
                                rank_zero_info(f"[Greedy] Intra-epoch decrease - Loss worsened: {self.prev_ema_loss:.4f}->{self.ema_loss:.4f}, New LR: {new_lr:.6e}")

                # Save current EMA loss for next comparison
                self.prev_ema_loss = self.ema_loss
                self.batches_since_adjustment = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Adjust learning rate based on validation loss improvement."""
        # Skip during warmup phase
        if self.current_step <= self.warmup_steps:
            return

        # Get current validation loss
        if "val_loss" not in trainer.callback_metrics:
            return

        val_loss = trainer.callback_metrics["val_loss"].item()
        self.current_epoch += 1

        # Check if validation loss improved
        if val_loss < self.best_val_loss:
            # Validation improved
            improvement = (self.best_val_loss - val_loss) / self.best_val_loss
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0

            # Only increase LR if significant improvement (> 1%)
            if improvement > 0.01 and self.current_epoch % self.patience == 0:
                new_lr = min(self.current_lr * self.factor_increase, self.max_lr)
                if new_lr != self.current_lr:
                    self.current_lr = new_lr
                    self._set_optimizer_lr(pl_module, new_lr)
                    if self.debug:
                        rank_zero_info(f"[Greedy] Epoch {self.current_epoch}: Val loss improved by {improvement:.2%}. Increasing LR to {new_lr:.6e}")
            elif self.debug:
                rank_zero_info(f"[Greedy] Epoch {self.current_epoch}: Val loss improved by {improvement:.2%} to {val_loss:.4f}. LR unchanged at {self.current_lr:.6e}")
        else:
            # Validation worsened
            self.epochs_without_improvement += 1

            # Decrease LR if no improvement for patience epochs
            if self.epochs_without_improvement >= self.patience:
                new_lr = max(self.current_lr * self.factor_decrease, self.min_lr)
                if new_lr != self.current_lr:
                    self.current_lr = new_lr
                    self._set_optimizer_lr(pl_module, new_lr)
                    self.epochs_without_improvement = 0  # Reset counter
                    if self.debug:
                        rank_zero_info(f"[Greedy] Epoch {self.current_epoch}: No improvement for {self.patience} epochs. Decreasing LR to {new_lr:.6e}")
            elif self.debug:
                rank_zero_info(
                    f"[Greedy] Epoch {self.current_epoch}: Val loss worsened to {val_loss:.4f} from best {self.best_val_loss:.4f}. "
                    f"Epochs without improvement: {self.epochs_without_improvement}/{self.patience}"
                )


@register_scheduler("adaptive")
class AdaptiveScheduler(BaseLRScheduler):
    """Adaptive learning rate scheduler that monitors a metric and adjusts LR accordingly.

    This scheduler increases or decreases learning rate based on whether a monitored
    metric is improving or worsening, with different patience settings for each direction.
    It also supports an optional warmup phase.

    Args:
        monitor_metric: Metric to monitor (typically 'val_loss')
        max_lr: Maximum learning rate (after warmup)
        min_lr: Minimum learning rate
        warmup_epochs: Number of epochs for warmup
        warmup_start_lr: Starting learning rate for warmup
        increase_factor: Factor to multiply LR by when metric improves
        decrease_factor: Factor to multiply LR by when metric worsens
        patience_increase: Number of epochs to wait before increasing LR
        patience_decrease: Number of epochs to wait before decreasing LR
        threshold: Minimum change in monitored metric to qualify as improvement
        threshold_mode: How to interpret threshold ('rel' or 'abs')
        cooldown: Number of epochs to wait after LR change before allowing another change
        debug: Enable verbose logging

    """

    def __init__(
        self,
        monitor_metric: str = "val_loss",
        max_lr: float = 1e-3,
        min_lr: float = 1e-6,
        warmup_epochs: int = 3,
        warmup_start_lr: float = 1e-7,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.7,
        patience_increase: int = 3,
        patience_decrease: int = 5,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.monitor_metric = monitor_metric
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.patience_increase = patience_increase
        self.patience_decrease = patience_decrease
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown

        # Initialize state
        self.reset()

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.current_lr = self.warmup_start_lr if self.warmup_epochs > 0 else self.max_lr
        self.best_metric = float("inf") if self._is_better_inverted() else float("-inf")
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.cooldown_counter = 0
        self.current_epoch = 0
        self.warmup_steps: int | float = 0  # Initialize to 0
        self.current_step = 0

        if self.debug:
            rank_zero_info(f"[Adaptive] State reset. Initial LR: {self.current_lr:.6e}")

    def _is_better_inverted(self) -> bool:
        """Return True if lower metric values are better (e.g., for loss)."""
        return "loss" in self.monitor_metric.lower() or "error" in self.monitor_metric.lower()

    def _is_better(self, current: float, best: float) -> bool:
        """Determine if current metric value is better than best so far.

        Args:
            current: Current metric value
            best: Best metric value so far

        Returns:
            True if current is better than best

        """
        if self._is_better_inverted():
            # For metrics like loss where lower is better
            if self.threshold_mode == "rel":
                return current < best * (1 - self.threshold)
            # 'abs'
            return current < best - self.threshold
        # For metrics like accuracy where higher is better
        if self.threshold_mode == "rel":
            return current > best * (1 + self.threshold)
        # 'abs'
        return current > best + self.threshold

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize scheduler at the start of training."""
        # Calculate total warmup steps
        self.warmup_steps = self.warmup_epochs * trainer.num_training_batches

        # Set initial learning rate
        self._set_optimizer_lr(pl_module, self.current_lr)

        if self.debug:
            rank_zero_info(f"[Adaptive] Training started. Initial LR: {self.current_lr:.6e}")
            rank_zero_info(f"[Adaptive] Warmup steps: {self.warmup_steps}")

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Handle warmup during training batches."""
        self.current_step += 1

        # Handle warmup phase
        if self.current_step <= self.warmup_steps and self.warmup_steps > 0:
            # Linear warmup from warmup_start_lr to max_lr
            progress = self.current_step / self.warmup_steps
            new_lr = self.warmup_start_lr + progress * (self.max_lr - self.warmup_start_lr)
            self.current_lr = new_lr
            self._set_optimizer_lr(pl_module, new_lr)

            if self.debug and (self.current_step == 1 or self.current_step % (self.warmup_steps // 10) == 0):
                rank_zero_info(f"[Adaptive] Warmup progress: {progress:.2f}, LR: {new_lr:.6e}")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Adjust learning rate based on validation metric.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module

        """
        # Skip during warmup phase
        if self.current_step <= self.warmup_steps:
            return

        # Get monitored metric
        if self.monitor_metric not in trainer.callback_metrics:
            if self.debug:
                rank_zero_info(f"[Adaptive] Warning: Monitor metric '{self.monitor_metric}' not found in callback_metrics.")
            return

        current_metric = trainer.callback_metrics[self.monitor_metric].item()
        self.current_epoch += 1

        # Skip if in cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.debug:
                rank_zero_info(f"[Adaptive] Epoch {self.current_epoch}: In cooldown period ({self.cooldown_counter} epochs left)")
            return

        # Check if metric is better
        is_better = self._is_better(current_metric, self.best_metric)

        if is_better:
            # Update best metric
            self.best_metric = current_metric
            self.num_bad_epochs = 0
            self.num_good_epochs += 1

            # Check if we should increase learning rate
            if self.num_good_epochs >= self.patience_increase:
                new_lr = min(self.current_lr * self.increase_factor, self.max_lr)
                if new_lr != self.current_lr:
                    self.current_lr = new_lr
                    self._set_optimizer_lr(pl_module, new_lr)
                    self.num_good_epochs = 0
                    self.cooldown_counter = self.cooldown

                    if self.debug:
                        rank_zero_info(f"[Adaptive] Epoch {self.current_epoch}: Metric improved to {current_metric:.6f}. Increasing LR to {new_lr:.6e}")
            elif self.debug:
                rank_zero_info(f"[Adaptive] Epoch {self.current_epoch}: Metric improved to {current_metric:.6f}. Consecutive improvements: {self.num_good_epochs}/{self.patience_increase}")
        else:
            # Metric worsened
            self.num_good_epochs = 0
            self.num_bad_epochs += 1

            # Check if we should decrease learning rate
            if self.num_bad_epochs >= self.patience_decrease:
                new_lr = max(self.current_lr * self.decrease_factor, self.min_lr)
                if new_lr != self.current_lr:
                    self.current_lr = new_lr
                    self._set_optimizer_lr(pl_module, new_lr)
                    self.num_bad_epochs = 0
                    self.cooldown_counter = self.cooldown

                    if self.debug:
                        rank_zero_info(f"[Adaptive] Epoch {self.current_epoch}: No improvement for {self.patience_decrease} epochs. Decreasing LR to {new_lr:.6e}")
            elif self.debug:
                rank_zero_info(
                    f"[Adaptive] Epoch {self.current_epoch}: Metric not improved ({current_metric:.6f} vs best {self.best_metric:.6f}). "
                    f"Epochs without improvement: {self.num_bad_epochs}/{self.patience_decrease}"
                )


# ----------------------------------------------------------------------------------------------------------------------
