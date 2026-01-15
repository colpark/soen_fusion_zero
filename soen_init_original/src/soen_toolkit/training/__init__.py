# FILEPATH: soen_toolkit/training/__init__.py

"""SOEN Model Training Framework (Version 3).

This package provides a framework for training SOEN (Structured ODE Neural) models
using PyTorch Lightning and TensorBoard.

To keep lightweight environments usable (e.g., for CLI utilities that don't need
PyTorch), heavy submodules are imported lazily on attribute access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

# Public surface (names preserved for backward compatibility)
__all__ = [
    "OPTIMIZER_REGISTRY",
    # Callbacks
    "SCHEDULER_REGISTRY",
    "ConstantLRScheduler",
    "CosineAnnealingScheduler",
    "DataConfig",
    # Configs
    "ExperimentConfig",
    # Trainers
    "ExperimentRunner",
    "LinearDecayScheduler",
    "LoggingConfig",
    "MetricsTracker",
    "ModelConfig",
    "RexScheduler",
    # Data
    "SOENDataModule",
    # Models
    "SOENLightningModule",
    "TrainingConfig",
    "create_default_config",
    "load_and_modify_model",
    "load_config",
    "run_from_config",
    "save_config",
    # Utils
    "setup_logger",
]


def __getattr__(name: str) -> Any:
    """Lazily import heavy modules only when attributes are accessed.

    This avoids importing PyTorch and Lightning when the caller only needs
    lightweight tools (e.g., S3 log streaming).
    """
    # Configs
    if name in {
        "ExperimentConfig",
        "TrainingConfig",
        "DataConfig",
        "ModelConfig",
        "LoggingConfig",
        "load_config",
        "save_config",
        "create_default_config",
    }:
        mod = import_module(".configs", __name__)
        return getattr(mod, name)

    # Models (PyTorch/Lightning heavy)
    if name in {"SOENLightningModule", "load_and_modify_model"}:
        mod = import_module(".models", __name__)
        return getattr(mod, name)

    # Trainers (PyTorch/Lightning heavy)
    if name in {"ExperimentRunner", "run_from_config"}:
        mod = import_module(".trainers", __name__)
        return getattr(mod, name)

    # Data
    if name == "SOENDataModule":
        mod = import_module(".data", __name__)
        return getattr(mod, name)

    # Callbacks
    if name in {
        "SCHEDULER_REGISTRY",
        "OPTIMIZER_REGISTRY",
        "CosineAnnealingScheduler",
        "RexScheduler",
        "ConstantLRScheduler",
        "LinearDecayScheduler",
        "MetricsTracker",
    }:
        mod = import_module(".callbacks", __name__)
        return getattr(mod, name)

    # Utils (lightweight)
    if name == "setup_logger":
        mod = import_module(".utils", __name__)
        return getattr(mod, name)

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
