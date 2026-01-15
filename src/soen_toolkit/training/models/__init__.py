# FILEPATH: src/soen_toolkit/training/models/__init__.py

"""Model wrappers for SOEN training.

This module provides PyTorch Lightning wrappers for SOEN models.
"""

from .lightning_wrapper import SOENLightningModule

# from .soen_classifier import SOENWithClassifier
from .model_factory import (
    create_model_with_configs,
    load_and_modify_model,
    modify_connection_params,
    modify_layer_params,
    modify_simulation_config,
)

__all__ = [
    "SOENLightningModule",
    "create_model_with_configs",
    # 'SOENWithClassifier',
    "load_and_modify_model",
    "modify_connection_params",
    "modify_layer_params",
    "modify_simulation_config",
]
