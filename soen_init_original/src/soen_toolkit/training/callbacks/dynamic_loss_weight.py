"""Dynamic loss weight adjustment callback for SOEN training.

This callback monitors validation metrics and dynamically adjusts the weight
of specific loss components based on performance. Useful for quantization losses
that should be reduced when validation performance degrades.
"""

import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import (  # type: ignore[attr-defined]  # rank zero means only print to the rank 0 process, which means if we're on a multi-node setup, only the first node will print
    rank_zero_info,
    rank_zero_warn,
)
import torch

logger = logging.getLogger(__name__)


class DynamicLossWeightCallback(Callback):
    """Callback that dynamically adjusts loss component weights based on validation metrics.

    This callback monitors a specified validation metric (e.g., 'val_accuracy', 'val_loss') and
    adjusts the weight of target loss components when performance degrades or improves.

    Attributes:
        target_loss_names: List of loss component names to adjust
        monitor_metric: Name of metric to monitor (e.g., 'val_accuracy', 'val_loss')
        mode: 'min' for metrics where lower is better, 'max' for metrics where higher is better
        patience: Number of epochs to wait before adjusting weights
        degradation_threshold: Threshold for considering metric as degraded
        weight_reduction_factor: Factor to multiply weight by when performance degrades
        weight_increase_factor: Factor to multiply weight by when performance improves
        min_weight: Minimum allowed weight value
        max_weight: Maximum allowed weight value
        improvement_threshold: Threshold for considering metric as improved

    """

    def __init__(
        self,
        target_loss_names: str | list[str],
        monitor_metric: str = "val_accuracy",
        mode: str = "max",
        patience: int = 3,
        degradation_threshold: float = 0.02,
        improvement_threshold: float = 0.01,
        weight_reduction_factor: float = 0.7,
        weight_increase_factor: float = 1.05,
        min_weight: float = 0.001,
        max_weight: float = 10.0,
        verbose: bool = True,
    ) -> None:
        """Initialize DynamicLossWeightCallback.

        Args:
            target_loss_names: Name(s) of loss components to adjust weights for
            monitor_metric: Metric to monitor for performance changes
            mode: 'min' for metrics where lower is better, 'max' for higher is better
            patience: Number of epochs to wait before making weight adjustments
            degradation_threshold: Relative threshold for detecting performance degradation
            improvement_threshold: Relative threshold for detecting performance improvement
            weight_reduction_factor: Factor to reduce weights by when performance degrades
            weight_increase_factor: Factor to increase weights by when performance improves
            min_weight: Minimum allowed weight value
            max_weight: Maximum allowed weight value
            verbose: Whether to log weight adjustments

        """
        super().__init__()

        # Convert single loss name to list
        if isinstance(target_loss_names, str):
            target_loss_names = [target_loss_names]

        self.target_loss_names = target_loss_names
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.patience = patience
        self.degradation_threshold = degradation_threshold
        self.improvement_threshold = improvement_threshold
        self.weight_reduction_factor = weight_reduction_factor
        self.weight_increase_factor = weight_increase_factor
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.verbose = verbose

        # Internal state
        self.metric_history = []
        self.best_metric = None
        self.epochs_since_improvement = 0
        self.epochs_since_degradation = 0

        # Validate mode
        if mode not in ["min", "max"]:
            msg = f"mode must be 'min' or 'max', got {mode}"
            raise ValueError(msg)

        if self.verbose:
            rank_zero_info(
                f"[DynamicLossWeight] Initialized to monitor '{monitor_metric}' for loss components {target_loss_names} (mode={mode})",
            )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of each validation epoch to check metric and adjust weights."""
        # Get current metric value
        current_metric = trainer.callback_metrics.get(self.monitor_metric)

        if current_metric is None:
            if self.verbose:
                rank_zero_warn(f"[DynamicLossWeight] Metric '{self.monitor_metric}' not found in callback_metrics")
            return

        # Convert to scalar if tensor
        if isinstance(current_metric, torch.Tensor):
            current_metric = current_metric.item()

        # Store metric history
        self.metric_history.append(current_metric)

        # Initialize best metric on first epoch
        if self.best_metric is None:
            self.best_metric = current_metric
            return

        # Determine if performance has improved or degraded
        if self.mode == "max":
            # Higher is better
            relative_change = (current_metric - self.best_metric) / abs(self.best_metric) if self.best_metric != 0 else 0
            improved = relative_change > self.improvement_threshold
            degraded = relative_change < -self.degradation_threshold
        else:
            # Lower is better
            relative_change = (self.best_metric - current_metric) / abs(self.best_metric) if self.best_metric != 0 else 0
            improved = relative_change > self.improvement_threshold
            degraded = relative_change < -self.degradation_threshold

        # Update counters
        if improved:
            self.best_metric = current_metric
            self.epochs_since_improvement = 0
            self.epochs_since_degradation += 1
        elif degraded:
            self.epochs_since_improvement += 1
            self.epochs_since_degradation = 0
        else:
            self.epochs_since_improvement += 1
            self.epochs_since_degradation += 1

        # Check if we should adjust weights
        should_reduce_weight = degraded and self.epochs_since_improvement >= self.patience

        should_increase_weight = not degraded and not improved and self.epochs_since_degradation >= self.patience and self.epochs_since_improvement >= self.patience

        # Adjust weights if needed
        if should_reduce_weight:
            self._adjust_loss_weights(pl_module, self.weight_reduction_factor, "reduced")
            self.epochs_since_improvement = 0  # Reset counter

        elif should_increase_weight:
            self._adjust_loss_weights(pl_module, self.weight_increase_factor, "increased")
            self.epochs_since_degradation = 0  # Reset counter

    def _adjust_loss_weights(self, pl_module: pl.LightningModule, factor: float, action: str) -> None:
        """Adjust the weights of target loss components.

        Args:
            pl_module: The lightning module containing loss components
            factor: Factor to multiply current weights by
            action: Description of action for logging ("reduced", "increased")

        """
        if not hasattr(pl_module, "active_loss_components"):
            if self.verbose:
                rank_zero_warn("[DynamicLossWeight] Module does not have 'active_loss_components' attribute")
            return

        adjusted_losses = []

        for loss_component in pl_module.active_loss_components:
            loss_name = loss_component.get("name", "")

            if loss_name in self.target_loss_names:
                old_weight = loss_component["weight"]
                new_weight = old_weight * factor

                # Clamp to min/max bounds
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))

                loss_component["weight"] = new_weight
                adjusted_losses.append((loss_name, old_weight, new_weight))

        # Log adjustments
        if adjusted_losses and self.verbose:
            for loss_name, old_weight, new_weight in adjusted_losses:
                rank_zero_info(
                    f"[DynamicLossWeight] {action.capitalize()} weight for '{loss_name}': {old_weight:.6f} -> {new_weight:.6f} (factor={factor:.3f})",
                )

    def get_metric_history(self) -> list[float]:
        """Get the history of monitored metric values."""
        return self.metric_history.copy()

    def reset_state(self) -> None:
        """Reset the internal state of the callback."""
        self.metric_history = []
        self.best_metric = None
        self.epochs_since_improvement = 0
        self.epochs_since_degradation = 0

        if self.verbose:
            rank_zero_info("[DynamicLossWeight] Reset internal state")
