# FILEPATH: src/soen_toolkit/training/models/model_factory.py

"""Model factory functions for creating and modifying SOEN models.

This module provides utility functions for creating, loading, and modifying SOEN models.
"""

from collections.abc import Callable
import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.core.model_yaml import build_model_from_yaml

logger = logging.getLogger(__name__)


def load_and_modify_model(
    model_path: Path,
    modifier_fn: Callable[..., Any] | None = None,
    seed: int | None = None,
) -> tuple[SOENModelCore, SimulationConfig, list[LayerConfig], list[ConnectionConfig]]:
    """Load a SOEN model and apply optional modifications.

    Args:
        model_path: Path to the SOEN model file
        modifier_fn: Optional function to modify model configurations
        seed: Optional random seed to set before loading/modifying

    Returns:
        Tuple containing:
            - Loaded and potentially modified SOEN model
            - SimulationConfig object
            - List of LayerConfig objects
            - List of ConnectionConfig objects

    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Load the model
    logger.info(f"Loading SOEN model from {model_path}")
    model = SOENModelCore.load(str(model_path))

    # Extract configurations
    sim_config = model.sim_config
    layers_config = model.layers_config.copy()
    connections_config = model.connections_config.copy()

    # Apply modifier function if provided
    if modifier_fn is not None:
        logger.info("Applying model modification function")
        sim_config, layers_config, connections_config = modifier_fn(
            sim_config,
            layers_config,
            connections_config,
        )

        # Create new model with modified configs
        logger.info("Creating new model with modified configurations")
        model = SOENModelCore(
            sim_config=sim_config,
            layers_config=layers_config,
            connections_config=connections_config,
        )

    return model, sim_config, layers_config, connections_config


def create_model_with_configs(
    sim_config: SimulationConfig,
    layers_config: list[LayerConfig],
    connections_config: list[ConnectionConfig],
    seed: int | None = None,
) -> SOENModelCore:
    """Create a new SOEN model with the given configurations.

    Args:
        sim_config: Simulation configuration
        layers_config: List of layer configurations
        connections_config: List of connection configurations
        seed: Optional random seed to set before creating the model

    Returns:
        SOENModelCore: New SOEN model

    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create new model
    logger.info("Creating new SOEN model with provided configurations")
    return SOENModelCore(
        sim_config=sim_config,
        layers_config=layers_config,
        connections_config=connections_config,
    )


def create_model_from_yaml(yaml_path_or_dict: Any, seed: int | None = None) -> SOENModelCore:
    """Convenience helper to build a model from a YAML architecture path or inline dict."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    return build_model_from_yaml(yaml_path_or_dict)


def modify_layer_params(
    layers_config: list[LayerConfig],
    layer_id: int,
    param_name: str,
    param_value: Any,
) -> list[LayerConfig]:
    """Modify parameter for a specific layer in the layers_config.

    Args:
        layers_config: List of layer configurations
        layer_id: ID of the layer to modify
        param_name: Name of the parameter to modify
        param_value: New value for the parameter

    Returns:
        List[LayerConfig]: Modified list of layer configurations

    """
    # Make a deep copy to avoid modifying the original
    modified_layers = copy.deepcopy(layers_config)

    # Find the layer with the specified ID
    for layer in modified_layers:
        if layer.layer_id == layer_id:
            # Check if the parameter is in the params dictionary
            if param_name in layer.params:
                logger.info(f"Modifying layer {layer_id} parameter '{param_name}': {layer.params[param_name]} -> {param_value}")
                layer.params[param_name] = param_value
                return modified_layers

            # Special case for noise configuration
            if param_name.startswith("noise."):
                noise_param = param_name.split(".", 1)[1]
                if hasattr(layer.noise, noise_param):
                    logger.info(f"Modifying layer {layer_id} noise parameter '{noise_param}': {getattr(layer.noise, noise_param)} -> {param_value}")
                    setattr(layer.noise, noise_param, param_value)
                    return modified_layers

            msg = f"Parameter '{param_name}' not found in layer {layer_id}"
            raise ValueError(msg)

    msg = f"Layer with ID {layer_id} not found"
    raise ValueError(msg)


def modify_connection_params(
    connections_config: list[ConnectionConfig],
    from_layer: int,
    to_layer: int,
    param_name: str,
    param_value: Any,
) -> list[ConnectionConfig]:
    """Modify parameter for a specific connection in the connections_config.

    Args:
        connections_config: List of connection configurations
        from_layer: ID of the source layer
        to_layer: ID of the target layer
        param_name: Name of the parameter to modify
        param_value: New value for the parameter

    Returns:
        List[ConnectionConfig]: Modified list of connection configurations

    """
    # Make a deep copy to avoid modifying the original
    modified_connections = copy.deepcopy(connections_config)

    # Find the connection with the specified source and target layers
    for connection in modified_connections:
        if connection.from_layer == from_layer and connection.to_layer == to_layer:
            # Check if the parameter is directly in the ConnectionConfig
            if hasattr(connection, param_name):
                logger.info(f"Modifying connection {from_layer}->{to_layer} parameter '{param_name}': {getattr(connection, param_name)} -> {param_value}")
                setattr(connection, param_name, param_value)
                return modified_connections

            # Check if the parameter is in the params dictionary
            if connection.params is not None and param_name in connection.params:
                logger.info(f"Modifying connection {from_layer}->{to_layer} parameter '{param_name}': {connection.params[param_name]} -> {param_value}")
                connection.params[param_name] = param_value
                return modified_connections

            # The parameter might be nested within params
            if connection.params is not None:
                for key, value in connection.params.items():
                    if isinstance(value, dict) and param_name in value:
                        logger.info(f"Modifying connection {from_layer}->{to_layer} nested parameter '{key}.{param_name}': {value[param_name]} -> {param_value}")
                        value[param_name] = param_value
                        return modified_connections

            msg = f"Parameter '{param_name}' not found in connection {from_layer}->{to_layer}"
            raise ValueError(msg)

    msg = f"Connection from layer {from_layer} to layer {to_layer} not found"
    raise ValueError(msg)


def modify_simulation_config(
    sim_config: SimulationConfig,
    param_name: str,
    param_value: Any,
) -> SimulationConfig:
    """Modify parameter in the simulation configuration.

    Args:
        sim_config: Simulation configuration
        param_name: Name of the parameter to modify
        param_value: New value for the parameter

    Returns:
        SimulationConfig: Modified simulation configuration

    """
    # Make a deep copy to avoid modifying the original
    modified_sim_config = copy.deepcopy(sim_config)

    # Check if the parameter exists
    if hasattr(modified_sim_config, param_name):
        logger.info(f"Modifying simulation parameter '{param_name}': {getattr(modified_sim_config, param_name)} -> {param_value}")
        setattr(modified_sim_config, param_name, param_value)
        return modified_sim_config

    msg = f"Parameter '{param_name}' not found in simulation configuration"
    raise ValueError(msg)
