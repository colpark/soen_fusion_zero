# FILEPATH: src/soen_toolkit/training/callbacks/probing.py

"""Model probing callbacks for SOEN training.

These callbacks allow visualization of model internals like connection parameters and
processed output state values during training, providing insights into the model's
behavior and evolution over time.
"""

import logging
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import nn

logger = logging.getLogger(__name__)


class ConnectionParameterProbeCallback(Callback):
    """Callback to log histograms of connection parameters during validation.

    This callback takes snapshots of the model's connection weights at validation end,
    using TensorBoard's native histogram functionality for efficient logging and
    interactive visualization.
    """

    def __init__(self) -> None:
        """Initialize the ConnectionParameterProbeCallback.
        TensorBoard's native histograms automatically determine optimal binning.
        """
        super().__init__()
        self._validated_connections = False

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log histograms of connection parameters at the end of a validation epoch.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module (expected to wrap a SOEN core model)

        """
        # Skip if logger does not support histogram logging
        if not self._is_tb_available(trainer):
            if not self._validated_connections:
                logger.info("TensorBoard logger not found. Skipping connection parameter histograms.")
                self._validated_connections = True  # Only log this once
            return

        # Look for the SOEN model and its connections
        soen_model = self._get_soen_model(pl_module)
        if soen_model is None:
            return

        connections_dict = getattr(soen_model, "connections", None)
        if connections_dict is None:
            if not self._validated_connections:
                logger.warning("SOENModelCore.connections not found. Cannot log connection histograms.")
                self._validated_connections = True  # Only log this once
            return

        # Validate connections type - it should be a dict-like object
        if not hasattr(connections_dict, "items"):
            if not self._validated_connections:
                logger.warning(f"connections does not have 'items()' method, found {type(connections_dict)}. Skipping histograms.")
                self._validated_connections = True  # Only log this once
            return

        # Mark as validated so we don't keep logging warnings
        self._validated_connections = True

        # Log histograms for each connection parameter using TensorBoard's native functionality
        for name, param in connections_dict.items():
            if isinstance(param, nn.Parameter) and param.requires_grad:
                self._log_parameter_histogram(trainer, param, name)

    def _is_tb_available(self, trainer: pl.Trainer) -> bool:
        """Check if TensorBoard logger is available."""
        return bool(trainer.logger and hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_histogram"))

    def _get_soen_model(self, pl_module: pl.LightningModule) -> Any | None:
        """Extract the SOEN model core from the PyTorch Lightning module."""
        if not hasattr(pl_module, "model"):
            if not self._validated_connections:
                logger.warning("LightningModule has no 'model' attribute. Cannot log connection histograms.")
            return None

        # In the new design, pl_module.model is the SOEN core
        return pl_module.model

    def _log_parameter_histogram(self, trainer: pl.Trainer, param: nn.Parameter, name: str) -> None:
        """Log a histogram for a single parameter using TensorBoard's native histogram functionality.

        Args:
            trainer: PyTorch Lightning trainer
            param: The parameter tensor to visualize
            name: Name of the parameter for logging

        """
        try:
            # Use TensorBoard's native histogram logging - no need to create matplotlib figures
            trainer.logger.experiment.add_histogram(
                f"connection_histograms/{name}",
                param.detach().cpu(),
                global_step=trainer.global_step,
            )

            # Also log some basic statistics
            param_data = param.detach().cpu()
            trainer.logger.experiment.add_scalar(
                f"connection_stats/{name}/mean",
                param_data.mean().item(),
                global_step=trainer.global_step,
            )
            trainer.logger.experiment.add_scalar(
                f"connection_stats/{name}/std",
                param_data.std().item(),
                global_step=trainer.global_step,
            )
            trainer.logger.experiment.add_scalar(
                f"connection_stats/{name}/abs_max",
                param_data.abs().max().item(),
                global_step=trainer.global_step,
            )

        except Exception as e:
            logger.warning(f"Error logging histogram for connection '{name}': {e}")
