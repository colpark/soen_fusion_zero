"""State collection from SOEN models.

This module handles extracting layer states from model forward passes,
with proper validation and error handling.
"""

from dataclasses import dataclass
import logging

import torch

logger = logging.getLogger(__name__)


class StateCollectionError(RuntimeError):
    """Raised when state collection fails."""
    pass


class InvalidStateFormatError(ValueError):
    """Raised when model returns states in unexpected format."""
    pass


@dataclass
class ForwardResult:
    """Result from a forward pass with state collection.

    Attributes:
        outputs: Final model outputs [batch, output_dim]
        layer_states: Dict mapping layer_id -> state [batch, dim]
        layer_trajectories: Optional dict mapping layer_id -> trajectory [batch, seq_len, dim]
        full_output: Full output including all timesteps [batch, seq_len+1, output_dim]
    """
    outputs: torch.Tensor
    layer_states: dict[int, torch.Tensor]
    full_output: torch.Tensor
    layer_trajectories: dict[int, torch.Tensor] | None = None

    def has_state_for_layer(self, layer_id: int) -> bool:
        """Check if state is available for a given layer."""
        return layer_id in self.layer_states

    def has_trajectory_for_layer(self, layer_id: int) -> bool:
        """Check if trajectory is available for a given layer."""
        return self.layer_trajectories is not None and layer_id in self.layer_trajectories

    def get_state_or_raise(self, layer_id: int) -> torch.Tensor:
        """Get state for layer, raising error if not available.

        Args:
            layer_id: Layer ID to get state for

        Returns:
            State tensor [batch, dim]

        Raises:
            StateCollectionError: If state not available
        """
        if layer_id not in self.layer_states:
            available = list(self.layer_states.keys())
            raise StateCollectionError(
                f"State for layer {layer_id} not available. "
                f"Available layers: {available}"
            )
        return self.layer_states[layer_id]


class StateCollector:
    """Collects layer states from SOEN model forward passes.

    Responsibilities:
    - Run forward pass through model
    - Extract layer states from model output
    - Validate state format and shapes
    - Handle edge cases (2D inputs, missing states, etc.)

    Example:
        >>> collector = StateCollector(model)
        >>> result = collector.collect(inputs)
        >>> pre_state = result.get_state_or_raise(0)
        >>> post_state = result.get_state_or_raise(1)
    """

    def __init__(self, model, track_phi: bool = False, collect_trajectories: bool = False):
        """Initialize state collector.

        Args:
            model: SOENModelCore instance
            track_phi: If True, track phi (flux) states. If False, track s (current) states.
            collect_trajectories: If True, also collect full trajectories for each layer.
        """
        self.model = model
        self.track_phi = track_phi
        self.collect_trajectories = collect_trajectories

    def collect(self, inputs: torch.Tensor) -> ForwardResult:
        """Collect states from a forward pass.

        Args:
            inputs: Input tensor
                   - [batch, seq_len, input_dim] for sequences
                   - [batch, input_dim] for static inputs (will add time dim)

        Returns:
            ForwardResult containing outputs and layer states

        Raises:
            InvalidStateFormatError: If model output format is unexpected
            StateCollectionError: If state collection fails
        """
        # Validate inputs
        self._validate_inputs(inputs)

        # Ensure inputs have time dimension
        inputs = self._ensure_time_dimension(inputs)

        # Run model forward
        with torch.no_grad():
            model_output = self.model(inputs)

        # Parse model output
        full_output, state_history = self._parse_model_output(model_output)

        # Extract final outputs (last timestep)
        outputs = self._extract_final_outputs(full_output)

        # Collect layer states and optionally trajectories
        layer_states, layer_trajectories = self._collect_states_and_trajectories(state_history)

        return ForwardResult(
            outputs=outputs,
            layer_states=layer_states,
            full_output=full_output,
            layer_trajectories=layer_trajectories if self.collect_trajectories else None
        )

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """Validate input tensor shape and type.

        Args:
            inputs: Input tensor to validate

        Raises:
            TypeError: If inputs is not a tensor
            ValueError: If inputs has invalid shape
        """
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"inputs must be torch.Tensor, got {type(inputs)}")

        if inputs.dim() not in (2, 3):
            raise ValueError(
                f"inputs must be 2D [batch, dim] or 3D [batch, seq, dim], "
                f"got {inputs.dim()}D with shape {inputs.shape}"
            )

        if inputs.size(0) == 0:
            raise ValueError("inputs cannot have batch size of 0")

    def _ensure_time_dimension(self, inputs: torch.Tensor) -> torch.Tensor:
        """Ensure inputs have time dimension.

        Args:
            inputs: Input tensor [batch, dim] or [batch, seq, dim]

        Returns:
            Tensor with shape [batch, seq, dim]
        """
        if inputs.dim() == 2:
            # Add time dimension: [batch, dim] -> [batch, 1, dim]
            inputs = inputs.unsqueeze(1)
            logger.debug(f"Added time dimension: {inputs.shape}")

        return inputs

    def _parse_model_output(
        self,
        model_output
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Parse model output into (output, state_history).

        Args:
            model_output: Raw output from model forward pass

        Returns:
            Tuple of (full_output, state_history)

        Raises:
            InvalidStateFormatError: If output format is unexpected
        """
        # Handle tuple output: (output, state_history)
        if isinstance(model_output, tuple):
            if len(model_output) != 2:
                raise InvalidStateFormatError(
                    f"Expected model to return (output, state_history), "
                    f"got tuple of length {len(model_output)}"
                )

            full_output, state_history = model_output

            if not isinstance(full_output, torch.Tensor):
                raise InvalidStateFormatError(
                    f"Expected output to be Tensor, got {type(full_output)}"
                )

            if not isinstance(state_history, list):
                raise InvalidStateFormatError(
                    f"Expected state_history to be list, got {type(state_history)}"
                )

            return full_output, state_history

        # Handle single tensor output (no state history)
        elif isinstance(model_output, torch.Tensor):
            logger.warning(
                "Model returned single tensor without state_history. "
                "State collection may be incomplete."
            )
            return model_output, []

        else:
            raise InvalidStateFormatError(
                f"Unexpected model output type: {type(model_output)}"
            )

    def _extract_final_outputs(self, full_output: torch.Tensor) -> torch.Tensor:
        """Extract final timestep outputs.

        Args:
            full_output: Full output tensor [batch, seq_len, output_dim]

        Returns:
            Final outputs [batch, output_dim]

        Raises:
            ValueError: If output shape is invalid
        """
        if full_output.dim() != 3:
            raise ValueError(
                f"Expected 3D output [batch, seq, dim], got {full_output.dim()}D"
            )

        # Get last timestep
        outputs = full_output[:, -1, :]

        logger.debug(
            f"Extracted final outputs: {outputs.shape} from {full_output.shape}"
        )

        return outputs

    def _collect_states_and_trajectories(
        self,
        state_history: list[torch.Tensor | None]
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Collect layer states and trajectories from state history.

        Args:
            state_history: List where state_history[layer_idx] is
                          [batch, seq_len, dim] or None

        Returns:
            Tuple of (layer_states, layer_trajectories)
        """
        layer_states = {}
        layer_trajectories = {}

        for layer_idx, layer_trajectory in enumerate(state_history):
            if layer_trajectory is None:
                continue

            # Validate trajectory shape
            if not isinstance(layer_trajectory, torch.Tensor):
                continue

            if layer_trajectory.dim() != 3:
                continue

            # Detach to prevent gradients
            trajectory = layer_trajectory.detach()

            # Store full trajectory if requested
            if self.collect_trajectories:
                layer_trajectories[layer_idx] = trajectory

            # Extract final timestep state
            layer_states[layer_idx] = trajectory[:, -1, :]

            logger.debug(
                f"Layer {layer_idx}: Collected state {layer_states[layer_idx].shape}"
            )

        if not layer_states:
            logger.warning(
                "No layer states collected. Model may not be tracking states correctly."
            )

        return layer_states, layer_trajectories

    def validate_states_for_connections(
        self,
        result: ForwardResult,
        connection_keys: list
    ) -> None:
        """Validate that required states are available for given connections.

        Args:
            result: ForwardResult from collect()
            connection_keys: List of ConnectionKey objects

        Raises:
            StateCollectionError: If required states are missing
        """
        missing_layers = []

        for conn_key in connection_keys:
            if not result.has_state_for_layer(conn_key.from_layer):
                missing_layers.append(conn_key.from_layer)

            if not result.has_state_for_layer(conn_key.to_layer):
                missing_layers.append(conn_key.to_layer)

        if missing_layers:
            missing_layers = sorted(set(missing_layers))
            available = sorted(result.layer_states.keys())
            raise StateCollectionError(
                f"Required layer states missing: {missing_layers}. "
                f"Available layers: {available}. "
                f"Ensure model is configured to track states for these layers."
            )
