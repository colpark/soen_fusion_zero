# FILEPATH: src/soen_toolkit/training/callbacks/scheduler_core.py

"""Core components for the learning rate scheduler framework.

Defines the base scheduler class, registry, and registration decorator.
"""

from collections.abc import Callable
import logging
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn  # type: ignore[attr-defined]
import torch

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Registry patterns for schedulers, models, and optimizers
# -----------------------------------------------------------------------------
SCHEDULER_REGISTRY: dict[str, Any] = {}
OPTIMIZER_REGISTRY: dict[str, Callable] = {}


def register_scheduler(name: str):
    """Decorator to register a new scheduler class."""

    def decorator(cls):
        if name in SCHEDULER_REGISTRY:
            logger.warning(f"Scheduler '{name}' already registered. Overwriting.")
        SCHEDULER_REGISTRY[name] = cls
        return cls

    return decorator


def register_optimizer(name: str) -> Callable:
    """Register an optimizer factory function with the registry."""

    def decorator(fn: Callable) -> Callable:
        OPTIMIZER_REGISTRY[name] = fn
        return fn

    return decorator


# Register common optimizers
@register_optimizer("adamw")
def adamw_optimizer(params, lr, **kwargs):
    return torch.optim.AdamW(params, lr=lr, **kwargs)


@register_optimizer("adam")
def adam_optimizer(params, lr, **kwargs):
    return torch.optim.Adam(params, lr=lr, **kwargs)


@register_optimizer("sgd")
def sgd_optimizer(params, lr, **kwargs):
    return torch.optim.SGD(params, lr=lr, **kwargs)


@register_optimizer("lion")
def lion_optimizer(params, lr, **kwargs):
    """Lion optimizer from the `lion-pytorch` package.

    This requires `pip install lion-pytorch`. If the package is not available,
    an informative error will be raised prompting the user to install it.
    """
    try:
        from lion_pytorch import Lion
    except ModuleNotFoundError as exc:
        msg = "Lion optimizer requested but the 'lion-pytorch' package is not installed. Install it with `pip install lion-pytorch` to proceed."
        raise ModuleNotFoundError(
            msg,
        ) from exc

    return Lion(params, lr=lr, **kwargs)


# -----------------------------------------------------------------------------
# Base class for all LR scheduler callbacks
# -----------------------------------------------------------------------------
class BaseLRScheduler(Callback):
    """Base class for custom learning rate scheduler callbacks.

    Handles debug flag and provides common helper methods.
    """

    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug

    def reset(self) -> None:
        """Reset internal state of the scheduler (if any). Base implementation does nothing."""
        if self.debug:
            rank_zero_info(f"[{self.__class__.__name__}] State reset method called.")
        # Subclasses should override if they have state to reset

    # Helper to get LR from optimizer
    def _get_optimizer_lr(self, pl_module: pl.LightningModule) -> float | None:
        """Reads the LR from the first param group of the first optimizer."""
        try:
            optimizers = pl_module.optimizers()
            if not optimizers:
                if self.debug:
                    rank_zero_info("[_get_optimizer_lr] No optimizers found.")
                return None

            # Handle both single optimizer and list of optimizers
            optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers

            if optimizer and hasattr(optimizer, "param_groups") and optimizer.param_groups:
                # Return LR from the first parameter group
                return optimizer.param_groups[0].get("lr")
            if self.debug:
                rank_zero_info("[_get_optimizer_lr] Optimizer has no param_groups.")
            return None
        except Exception as e:
            rank_zero_warn(f"[_get_optimizer_lr] Error getting LR: {e}. Returning None.")
            return None

    # Helper to set LR in optimizer
    def _set_optimizer_lr(self, pl_module: pl.LightningModule, new_lr: float) -> None:
        """Sets LR for all param groups, respecting per-group scaling.

        If a param group contains 'lr_scale', we set group['lr'] = new_lr * lr_scale.
        Otherwise, we try to preserve the relative ratio based on 'initial_lr'
        compared to the first group's 'initial_lr'. If neither is available, we
        set group['lr'] = new_lr.
        """
        try:
            optimizers = pl_module.optimizers()
            if not optimizers:
                if self.debug:
                    rank_zero_info("[_set_optimizer_lr] No optimizers found to set LR.")
                return

            applied_to_at_least_one = False
            optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]

            for opt in optimizer_list:
                if not (opt and hasattr(opt, "param_groups") and opt.param_groups):
                    rank_zero_warn(f"[_set_optimizer_lr] Could not set LR for optimizer {opt}. Invalid structure.")
                    continue

                # Determine base initial LR from the first group if present
                try:
                    base_initial_lr = float(opt.param_groups[0].get("initial_lr", new_lr))
                except Exception:
                    base_initial_lr = float(new_lr)

                for group in opt.param_groups:
                    # Prefer explicit lr_scale
                    lr_scale = group.get("lr_scale", None)
                    if lr_scale is not None:
                        try:
                            lr_scale = float(lr_scale)
                        except Exception:
                            lr_scale = 1.0
                        group["lr"] = float(new_lr) * lr_scale
                        applied_to_at_least_one = True
                        continue

                    # Fallback: preserve ratio using initial_lr if available
                    try:
                        g_init = float(group.get("initial_lr", base_initial_lr))
                        ratio = (g_init / base_initial_lr) if base_initial_lr != 0 else 1.0
                    except Exception:
                        ratio = 1.0

                    group["lr"] = float(new_lr) * ratio
                    applied_to_at_least_one = True

            if applied_to_at_least_one and self.debug:
                step = pl_module.global_step if hasattr(pl_module, "global_step") else "N/A"
                rank_zero_info(f"[{self.__class__.__name__} Step: {step}] Set optimizer LR base={new_lr:.6e} (with per-group scaling)")

        except Exception as e:
            rank_zero_warn(f"[_set_optimizer_lr] Failed to set LR: {e}")

    # PTL Hooks (Subclasses override as needed)
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook called after train batch. Base implementation does nothing."""

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Hook called at the start of training. Base implementation does nothing."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Hook called at the start of each training epoch. Base implementation does nothing."""
