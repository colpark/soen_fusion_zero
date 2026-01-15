# FILEPATH: src/soen_toolkit/training/configs/experiment_config.py

"""Utility functions for loading and saving experiment configurations.

This module provides functions to load configuration from YAML files,
save configurations to YAML, and create default configurations.

These are mostly not yet used or tested.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from .config_classes import (
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path, validate: bool = True, auto_detect: bool = False) -> ExperimentConfig:
    """Load configuration from a YAML file with optional validation and auto-detection.

    Args:
        config_path: Path to the YAML configuration file
        validate: Whether to perform configuration validation
        auto_detect: Whether to auto-detect task type from data and suggest corrections

    Returns:
        ExperimentConfig: Loaded configuration object

    """
    from .config_validation import auto_detect_task_type, validate_config

    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = ExperimentConfig.from_dict(config_dict)
    logger.info(f"Loaded configuration from {config_path}")

    # Auto-detection if requested
    if auto_detect and hasattr(config.data, "data_path") and config.data.data_path:
        suggestions = auto_detect_task_type(str(config.data.data_path))
        if suggestions["confidence"] in ["medium", "high"]:
            logger.info("Auto-detected task type from data:")
            if suggestions["paradigm"]:
                logger.info(f"  • Suggested paradigm: {suggestions['paradigm']}")
            if suggestions["mapping"]:
                logger.info(f"  • Suggested mapping: {suggestions['mapping']}")
            if suggestions["losses"]:
                logger.info(f"  • Suggested losses: {[loss_fn['name'] for loss_fn in suggestions['losses']]}")
            if suggestions["num_classes"]:
                logger.info(f"  • Detected classes: {suggestions['num_classes']}")
            logger.info(f"  • Detection confidence: {suggestions['confidence']}")

    # Validation if requested
    if validate:
        try:
            warnings, _errors = validate_config(config, raise_on_error=True)
            if warnings:
                logger.info(f"Configuration loaded with {len(warnings)} warnings")
            else:
                logger.info("Configuration validation passed")
        except Exception as e:
            logger.exception(f"Configuration validation failed: {e}")
            raise

    return config


def save_config(config: ExperimentConfig, save_path: str | Path) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration object to save
        save_path: Path where to save the YAML file

    """
    save_path = Path(save_path)

    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config to serializable dictionary
    config_dict = config.to_dict()

    # Save to YAML
    with open(save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to {save_path}")


def create_default_config() -> ExperimentConfig:
    """Create a default configuration with reasonable parameters.

    Returns:
        ExperimentConfig: Default configuration object

    """
    # Define a default project name to be used if specific project name isn't available elsewhere
    # This also helps in creating default paths.
    default_project_identifier = "default_experiment"

    return ExperimentConfig(
        name=None,  # Name is now optional and not used for critical pathing
        description="Default configuration",
        seed=42,
        training=TrainingConfig(
            batch_size=32,
            max_epochs=100,
            num_repeats=1,
        ),
        data=DataConfig(
            sequence_length=100,
            data_path="./data",
            num_classes=10,
        ),
        model=ModelConfig(
            time_pooling="max",
        ),
        logging=LoggingConfig(
            project_name="soen_training",  # Default project name
            group_name=default_project_identifier,
            project_dir=f"./{default_project_identifier}",
            upload_logs_and_checkpoints=False,
            s3_upload_url=None,
        ),
        callbacks={
            "lr_scheduler": {
                "type": "cosine",
                "max_lr": 1e-3,
                "min_lr": 1e-6,
                "warmup_epochs": 5,
                "cycle_epochs": 50,
                "enable_restarts": True,
            },
        },
    )


def modify_config(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply overrides to an existing configuration.

    Args:
        config: Original configuration object
        overrides: Dictionary of override values with dot notation keys
                  (e.g. 'training.batch_size: 64')

    Returns:
        ExperimentConfig: Modified configuration object

    """
    # Convert config to dictionary for easier manipulation
    config_dict = config.to_dict()

    # Apply overrides
    for key, value in overrides.items():
        # Handle nested attributes with dot notation
        if "." in key:
            parts = key.split(".")
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config_dict[key] = value

    # Convert back to ExperimentConfig
    return ExperimentConfig.from_dict(config_dict)


def create_derived_config(base_config: ExperimentConfig, changes: dict[str, Any], new_name: str | None = None) -> ExperimentConfig:
    """Create a new configuration derived from a base configuration with changes.

    Args:
        base_config: Base configuration to derive from
        changes: Dictionary of changes to apply
        new_name: New name for the derived configuration (optional)

    Returns:
        ExperimentConfig: New configuration object

    """
    # Create a copy of the base config
    base_config.to_dict()

    # Apply changes
    modified_config = modify_config(base_config, changes)

    # Update name if provided (name is Optional[str])
    if new_name is not None:
        modified_config.name = new_name

    # Directory paths are now primarily handled by ExperimentRunner based on project_name.
    # Removing automatic path updates based on 'new_name' here to avoid conflicts
    # and because 'name' is no longer the primary identifier for directory structures.
    # if new_name:
    # modified_config.training.checkpoint_dir = Path(f"./checkpoints/{new_name}")
    # modified_config.training.log_dir = Path(f"./logs/{new_name}")
    # modified_config.logging.group_name = new_name

    return modified_config
