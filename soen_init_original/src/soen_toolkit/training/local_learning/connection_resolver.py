"""Connection resolution for local learning.

This module handles the logic of determining which connections to train,
with proper validation and error handling.
"""

from dataclasses import dataclass
import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class ConnectionKey(NamedTuple):
    """Type-safe connection identifier.

    Attributes:
        from_layer: Source layer ID
        to_layer: Target layer ID
    """
    from_layer: int
    to_layer: int

    def to_parameter_name(self) -> str:
        """Convert to model parameter name format.

        Returns:
            String key for accessing model.connections ParameterDict
        """
        return f"J_{self.from_layer}_to_{self.to_layer}"

    def __str__(self) -> str:
        return f"{self.from_layer}→{self.to_layer}"


@dataclass(frozen=True)
class ConnectionInfo:
    """Metadata about a model connection.

    Attributes:
        key: Type-safe connection identifier
        param_name: Parameter name in model.connections
        from_dim: Pre-synaptic dimension
        to_dim: Post-synaptic dimension
    """
    key: ConnectionKey
    param_name: str
    from_dim: int
    to_dim: int

    def __str__(self) -> str:
        return f"{self.key} [{self.from_dim}×{self.to_dim}]"


class ConnectionNotFoundError(ValueError):
    """Raised when a requested connection doesn't exist in the model."""
    pass


class InvalidLayerError(ValueError):
    """Raised when layer ID is invalid."""
    pass


class ConnectionResolver:
    """Resolves which connections to train based on user specification.

    Responsibilities:
    - Convert user specification (layers, connections) to ConnectionKey list
    - Validate that connections exist in the model
    - Provide clear error messages for invalid specifications

    Example:
        >>> resolver = ConnectionResolver(model)
        >>> connections = resolver.resolve_from_layers([1, 2])
        >>> for conn_info in connections:
        ...     print(f"Training {conn_info}")
    """

    def __init__(self, model):
        """Initialize resolver with a SOEN model.

        Args:
            model: SOENModelCore instance
        """
        self.model = model
        self._validate_model()

    def _validate_model(self) -> None:
        """Validate that model has required attributes."""
        if not hasattr(self.model, 'connections_config'):
            raise AttributeError(
                "Model must have 'connections_config' attribute. "
                "Ensure model is a SOENModelCore instance."
            )
        if not hasattr(self.model, 'connections'):
            raise AttributeError(
                "Model must have 'connections' ParameterDict. "
                "Ensure model is properly initialized."
            )

    def resolve_from_layers(
        self,
        layers: list[int] | None = None
    ) -> list[ConnectionInfo]:
        """Resolve connections incoming to specified layers.

        Args:
            layers: List of target layer IDs. If None, returns all connections.

        Returns:
            List of ConnectionInfo objects for training

        Raises:
            InvalidLayerError: If layer ID is invalid
            ConnectionNotFoundError: If expected connection is missing
        """
        if layers is not None:
            self._validate_layer_ids(layers)

        connections = []

        for conn_config in self.model.connections_config:
            # Filter by target layers if specified
            if layers is not None and conn_config.to_layer not in layers:
                continue

            # Create connection key
            key = ConnectionKey(
                from_layer=conn_config.from_layer,
                to_layer=conn_config.to_layer
            )

            # Validate connection exists in model
            conn_info = self._get_connection_info(key)
            connections.append(conn_info)

        if not connections:
            if layers is not None:
                raise ConnectionNotFoundError(
                    f"No connections found incoming to layers {layers}"
                )
            else:
                raise ConnectionNotFoundError("No connections found in model")

        logger.info(
            f"Resolved {len(connections)} connection(s) "
            f"{'incoming to layers ' + str(layers) if layers else 'total'}"
        )

        return connections

    def resolve_from_connection_list(
        self,
        connections: list[tuple[int, int]]
    ) -> list[ConnectionInfo]:
        """Resolve specific connections from (from_layer, to_layer) tuples.

        Args:
            connections: List of (from_layer, to_layer) tuples

        Returns:
            List of ConnectionInfo objects for training

        Raises:
            ConnectionNotFoundError: If any connection doesn't exist
        """
        if not connections:
            raise ValueError("connections list cannot be empty")

        conn_infos = []

        for from_layer, to_layer in connections:
            key = ConnectionKey(from_layer=from_layer, to_layer=to_layer)
            conn_info = self._get_connection_info(key)
            conn_infos.append(conn_info)

        logger.info(f"Resolved {len(conn_infos)} specific connection(s)")

        return conn_infos

    def _get_connection_info(self, key: ConnectionKey) -> ConnectionInfo:
        """Get connection info for a given key.

        Args:
            key: ConnectionKey to look up

        Returns:
            ConnectionInfo with metadata

        Raises:
            ConnectionNotFoundError: If connection doesn't exist
        """
        param_name = key.to_parameter_name()

        if param_name not in self.model.connections:
            available = list(self.model.connections.keys())
            raise ConnectionNotFoundError(
                f"Connection '{param_name}' not found in model. "
                f"Available connections: {available}"
            )

        # Get weight tensor to extract dimensions
        weights = self.model.connections[param_name]

        # PyTorch convention: weights are [post_dim, pre_dim]
        post_dim, pre_dim = weights.shape

        return ConnectionInfo(
            key=key,
            param_name=param_name,
            from_dim=pre_dim,
            to_dim=post_dim
        )

    def _validate_layer_ids(self, layers: list[int]) -> None:
        """Validate layer IDs are valid integers within model bounds.

        Args:
            layers: List of layer IDs to validate

        Raises:
            InvalidLayerError: If any layer ID is invalid
        """
        if not isinstance(layers, list):
            raise TypeError(f"layers must be a list, got {type(layers)}")

        if not layers:
            raise ValueError("layers list cannot be empty")

        for layer_id in layers:
            if not isinstance(layer_id, int):
                raise TypeError(
                    f"layer ID must be int, got {type(layer_id)} for {layer_id}"
                )

            # Check if layer exists in model
            if not hasattr(self.model, 'layers'):
                # Can't validate without layers list, skip
                continue

            if layer_id < 0 or layer_id >= len(self.model.layers):
                raise InvalidLayerError(
                    f"Layer ID {layer_id} out of bounds. "
                    f"Model has {len(self.model.layers)} layers (0-{len(self.model.layers)-1})"
                )

    def get_all_connections(self) -> list[ConnectionInfo]:
        """Get all connections in the model.

        Returns:
            List of all ConnectionInfo objects
        """
        return self.resolve_from_layers(layers=None)
