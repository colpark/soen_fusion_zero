# FILEPATH: src/soen_toolkit/training/configs/__init__.py

"""Configuration module for SOEN model training.

This module provides dataclasses and utilities for configuration management.
"""

from .config_classes import (
    AdaptiveConfig,
    # Scheduler configs
    ConstantConfig,
    CosineConfig,
    DataConfig,
    # Distillation config
    DistillationConfig,
    # Overall config
    ExperimentConfig,
    GreedyConfig,
    LinearConfig,
    LoggingConfig,
    ModelConfig,
    # Base configs
    NoiseConfig,
    OptimizerConfig,
    PerturbationConfig,
    RexConfig,
    # Main configs
    TrainingConfig,
    WarmupConfig,
)
from .experiment_config import (
    create_default_config,
    create_derived_config,
    load_config,
    modify_config,
    save_config,
)

__all__ = [
    "AdaptiveConfig",
    # Scheduler configs
    "ConstantConfig",
    "CosineConfig",
    "DataConfig",
    # Distillation config
    "DistillationConfig",
    # Overall config
    "ExperimentConfig",
    "GreedyConfig",
    "LinearConfig",
    "LoggingConfig",
    "ModelConfig",
    # Base configs
    "NoiseConfig",
    "OptimizerConfig",
    "PerturbationConfig",
    "RexConfig",
    # Main configs
    "TrainingConfig",
    "WarmupConfig",
    "create_default_config",
    "create_derived_config",
    # Utility functions
    "load_config",
    "modify_config",
    "save_config",
]
