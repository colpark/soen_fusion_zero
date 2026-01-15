"""Local learning trainer for SOEN models."""

import logging

import torch
import torch.nn.functional as F

from soen_toolkit.core import SOENModelCore
from soen_toolkit.training.local_learning.modulators.base import AbstractModulator
from soen_toolkit.training.local_learning.rules.base import AbstractLocalRule

logger = logging.getLogger(__name__)


class LocalTrainer:
    """Trainer for local learning rules on SOEN models.

    This trainer applies local learning rules (2-factor or 3-factor) to update
    weights based on layer activations rather than global backpropagation.

    Example:
        ```python
        from soen_toolkit.training.local_learning import LocalTrainer, HebbianRule

        trainer = LocalTrainer(
            model=my_model,
            rule=HebbianRule(lr=0.01),
            layers=[0, 1],  # Which layers to train
        )

        for inputs, targets in dataloader:
            metrics = trainer.step(inputs, targets)
            print(f"Loss: {metrics['loss']:.4f}")
        ```
    """

    def __init__(
        self,
        model: SOENModelCore,
        rule: AbstractLocalRule,
        layers: list[int] | None = None,
        connections: list[tuple[int, int]] | None = None,
        modulator: AbstractModulator | None = None,
        readout_loss: str = "mse",
        track_phi: bool = False,
    ):
        """Initialize local learning trainer.

        Args:
            model: SOEN model to train
            rule: Local learning rule to apply
            layers: List of layer IDs to train. If None, trains all layers.
                   Note: This trains connections INCOMING to these layers.
            connections: Alternatively, specify specific connections as
                        list of (from_layer, to_layer) tuples.
                        If provided, overrides `layers`.
            modulator: Modulator for 3-factor rules (optional)
            readout_loss: Loss function for computing error/modulator
                         Options: "mse", "cross_entropy"
            track_phi: If True, track phi states. If False, use s states.
                      (phi is flux, s is current/state)
        """
        self.model = model
        self.rule = rule
        self.modulator = modulator
        self.readout_loss = readout_loss
        self.track_phi = track_phi

        # Determine which connections to train
        if connections is not None:
            self.target_connections = connections
            logger.info(
                f"Local learning on {len(connections)} specific connections"
            )
        elif layers is not None:
            # Train all connections incoming to specified layers
            self.target_connections = []
            for conn_config in self.model.connections_config:
                if conn_config.to_layer in layers:
                    conn_tuple = (conn_config.from_layer, conn_config.to_layer)
                    self.target_connections.append(conn_tuple)
            logger.info(
                f"Local learning on {len(self.target_connections)} connections "
                f"incoming to layers {layers}"
            )
        else:
            # Train all connections
            self.target_connections = []
            for conn_config in self.model.connections_config:
                conn_tuple = (conn_config.from_layer, conn_config.to_layer)
                self.target_connections.append(conn_tuple)
            logger.info(
                f"Local learning on all {len(self.target_connections)} connections"
            )

        logger.info(f"Using rule: {self.rule}")
        if self.modulator:
            logger.info(f"Using modulator: {self.modulator}")

    def step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Perform one training step with local learning.

        Args:
            inputs: Input tensor
                   - [batch, seq_len, input_dim] for sequences
                   - [batch, input_dim] for static inputs
            targets: Target tensor (optional, for 3-factor rules and loss)
                    - [batch, output_dim] for regression
                    - [batch] for classification (class indices)

        Returns:
            Dictionary with metrics:
                - 'loss': Readout loss (for monitoring)
                - 'total_update_norm': Total norm of weight updates
        """
        # 1. Forward pass - collect states
        with torch.no_grad():
            outputs, layer_states = self._forward_and_collect_states(inputs)

        # 2. Compute modulator if 3-factor rule
        modulator_signal = None
        loss_value = 0.0

        if targets is not None:
            # Compute loss for monitoring
            loss_value = self._compute_loss(outputs, targets)

            # Compute modulator if using 3-factor rule
            if self.modulator is not None:
                modulator_signal = self.modulator.compute(outputs, targets)

        # 3. Apply local updates to target connections
        total_update_norm = 0.0
        for from_idx, to_idx in self.target_connections:
            update_norm = self._update_connection(
                from_idx, to_idx, layer_states, modulator_signal
            )
            total_update_norm += update_norm

        return {
            "loss": loss_value,
            "total_update_norm": total_update_norm,
        }

    def _forward_and_collect_states(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run forward pass and collect layer states.

        Args:
            inputs: Input tensor [batch, seq_len, input_dim] or [batch, input_dim]

        Returns:
            outputs: Final output [batch, output_dim]
            layer_states: Dict mapping layer_id -> state tensor [batch, dim]
        """
        # Ensure inputs have time dimension
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)  # [batch, 1, input_dim]

        # Run model forward
        # SOENModelCore.forward returns tuple: (output, list_of_layer_states)
        model_output = self.model(inputs)

        # Handle tuple output
        if isinstance(model_output, tuple) and len(model_output) == 2:
            full_output, state_history = model_output
        else:
            full_output = model_output
            state_history = []

        # Get final output (last timestep)
        outputs = full_output[:, -1, :]  # [batch, output_dim]

        # Collect layer states from the returned history
        # state_history is a list where state_history[layer_idx] has shape [batch, seq_len, dim]
        layer_states = {}

        for layer_idx, layer_state_trajectory in enumerate(state_history):
            if layer_state_trajectory is not None:
                # Use final timestep for local learning
                # layer_state_trajectory shape: [batch, seq_len, dim]
                layer_states[layer_idx] = layer_state_trajectory[:, -1, :].detach()

        return outputs, layer_states

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Compute loss for monitoring.

        Args:
            outputs: Model outputs [batch, output_dim]
            targets: Targets [batch, output_dim] or [batch]

        Returns:
            Scalar loss value
        """
        with torch.no_grad():
            if self.readout_loss == "mse":
                loss = F.mse_loss(outputs, targets)
            elif self.readout_loss == "cross_entropy":
                # Handle one-hot targets
                if targets.dim() == 2:
                    targets = targets.argmax(dim=1)
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = torch.tensor(0.0)

        return loss.item()

    def _update_connection(
        self,
        from_idx: int,
        to_idx: int,
        layer_states: dict[int, torch.Tensor],
        modulator: torch.Tensor | None,
    ) -> float:
        """Apply local learning rule to a specific connection.

        Args:
            from_idx: Source layer index
            to_idx: Target layer index
            layer_states: Dictionary of layer states
            modulator: Optional modulating signal

        Returns:
            Norm of the weight update (for monitoring)
        """
        # Get pre and post activities
        if from_idx not in layer_states or to_idx not in layer_states:
            logger.warning(
                f"Cannot update connection {from_idx}->{to_idx}: "
                "states not available"
            )
            return 0.0

        pre_activity = layer_states[from_idx]  # [batch, pre_dim]
        post_activity = layer_states[to_idx]  # [batch, post_dim]

        # Get weight parameter from model.connections ParameterDict
        # Connection keys are formatted as "J_{from}_to_{to}"
        conn_key = f"J_{from_idx}_to_{to_idx}"

        if conn_key not in self.model.connections:
            logger.warning(
                f"Connection {conn_key} not found in model.connections"
            )
            return 0.0

        weights = self.model.connections[conn_key]  # [pre_dim, post_dim]

        # Compute update using learning rule
        delta_w = self.rule.compute_update(
            pre_activity, post_activity, weights, modulator
        )

        # Apply update (in-place, no gradient tracking)
        with torch.no_grad():
            weights.data += delta_w

        # Return update norm for monitoring
        update_norm = torch.norm(delta_w).item()
        return update_norm

    def reset(self):
        """Reset trainer state (e.g., for BCM threshold)."""
        if hasattr(self.rule, 'reset_threshold'):
            self.rule.reset_threshold()

    def __repr__(self) -> str:
        return (
            f"LocalTrainer(\n"
            f"  rule={self.rule},\n"
            f"  modulator={self.modulator},\n"
            f"  connections={len(self.target_connections)},\n"
            f"  readout_loss={self.readout_loss}\n"
            f")"
        )
