"""Refactored local learning trainer (v2) with clean separation of concerns.

This is the new recommended trainer that uses the modular components:
- ConnectionResolver: Determines which connections to train
- StateCollector: Extracts layer states from model
- LossStrategy: Computes loss (pluggable)
- WeightUpdater: Applies weight updates with validation

The old trainer.py is kept for backward compatibility.
"""

import logging
from typing import Any

import torch

from soen_toolkit.core import SOENModelCore
from soen_toolkit.training.local_learning.connection_resolver import (
    ConnectionInfo,
    ConnectionResolver,
)
from soen_toolkit.training.local_learning.constraints import (
    RangeConstraint,
    SignPreservationConstraint,
)
from soen_toolkit.training.local_learning.loss_strategy import (
    LossFactory,
    LossStrategy,
)
from soen_toolkit.training.local_learning.modulators.base import AbstractModulator
from soen_toolkit.training.local_learning.rules.base import AbstractLocalRule
from soen_toolkit.training.local_learning.state_collector import StateCollector
from soen_toolkit.training.local_learning.weight_updater import (
    WeightUpdater,
)

logger = logging.getLogger(__name__)


class LocalTrainer:
    """Clean, modular trainer for local learning on SOEN models.

    This trainer orchestrates local learning by delegating to specialized components:
    - ConnectionResolver: Finds connections to train
    - StateCollector: Extracts layer states
    - LossStrategy: Computes training loss
    - WeightUpdater: Applies weight updates

    Benefits of this design:
    - Single Responsibility: Each component has one job
    - Open/Closed: Easy to add new loss types, rules, etc.
    - Testable: Each component can be unit tested independently
    - Fail-fast: Clear error messages when things go wrong

    Example:
        ```python
        from soen_toolkit.training.local_learning import LocalTrainer, HebbianRule

        trainer = LocalTrainer(
            model=my_model,
            rule=HebbianRule(lr=0.01),
            layers=[1, 2],  # Train connections to layers 1 and 2
            loss="mse"      # Use MSE loss for monitoring
        )

        for inputs, targets in dataloader:
            metrics = trainer.step(inputs, targets)
            print(f"Loss: {metrics['loss']:.4f}, Updates: {metrics['total_update_norm']:.6f}")
        ```
    """

    def __init__(
        self,
        model: SOENModelCore,
        rule: AbstractLocalRule,
        layers: list[int] | None = None,
        connections: list[tuple[int, int]] | None = None,
        modulator: AbstractModulator | None = None,
        loss: str | LossStrategy | None = None,
        readout_loss: str | None = None,  # Backward compatibility
        track_phi: bool = False,
        check_finite: bool = True,
        preserve_sign: bool = False,
        weight_range: tuple[float | None, float | None] | None = None,
    ):
        """Initialize local learning trainer.

        Args:
            model: SOEN model to train
            rule: Local learning rule (Hebbian, Oja, etc.)
            layers: List of layer IDs. Trains connections INCOMING to these layers.
                   If None and connections=None, trains all connections.
            connections: Alternatively, specify connections as list of (from, to) tuples.
                        Overrides `layers` if provided.
            modulator: Modulator for 3-factor rules (optional)
            loss: Loss function for monitoring. Either:
                 - String: "mse", "cross_entropy", "mae", "huber", "none"
                 - LossStrategy instance for custom losses
            readout_loss: (Deprecated) Alias for `loss` parameter. Use `loss` instead.
            track_phi: If True, track phi (flux) states. If False, track s (current).
            check_finite: If True, raise error on NaN/Inf in updates.
            preserve_sign: If True, ensures weights maintain their initial polarity.
            weight_range: Optional (min, max) tuple for weight clipping.

        Raises:
            ValueError: If both layers and connections are None (ambiguous)
            ConnectionNotFoundError: If specified connections don't exist
            InvalidLayerError: If layer IDs are invalid
        """
        # Handle backward compatibility: readout_loss -> loss
        if readout_loss is not None and loss is None:
            logger.warning(
                "Parameter 'readout_loss' is deprecated. Use 'loss' instead."
            )
            loss = readout_loss
        elif readout_loss is not None and loss is not None:
            raise ValueError(
                "Cannot specify both 'loss' and 'readout_loss'. "
                "Use 'loss' only (readout_loss is deprecated)."
            )

        # Default to MSE if neither specified
        if loss is None:
            loss = "mse"
        self.model = model
        self.rule = rule
        self.modulator = modulator
        self.check_finite = check_finite

        # Initialize components
        self._connection_resolver = ConnectionResolver(model)

        # Resolve connections to train
        if connections is not None:
            self._connections = self._connection_resolver.resolve_from_connection_list(
                connections
            )
        elif layers is not None:
            self._connections = self._connection_resolver.resolve_from_layers(layers)
        else:
            # Train all connections
            self._connections = self._connection_resolver.get_all_connections()

        # Resolve loss strategy
        if isinstance(loss, str):
            self._loss_strategy = LossFactory.create(loss)
        elif isinstance(loss, LossStrategy):
            self._loss_strategy = loss
        else:
            raise TypeError(
                f"loss must be str or LossStrategy, got {type(loss)}"
            )

        # Initialize constraints
        self._constraints = {}
        for conn_info in self._connections:
            conn_constraints = []

            # Add range constraint if specified
            if weight_range is not None:
                conn_constraints.append(
                    RangeConstraint(min_val=weight_range[0], max_val=weight_range[1])
                )

            # Add sign preservation if specified
            if preserve_sign:
                initial_weights = self.model.connections[conn_info.param_name]
                conn_constraints.append(SignPreservationConstraint(initial_weights))

            if conn_constraints:
                self._constraints[conn_info.key] = conn_constraints

        self._state_collector = StateCollector(
            model,
            track_phi=track_phi,
            collect_trajectories=rule.requires_trajectory
        )
        self._weight_updater = WeightUpdater(
            model,
            rule,
            check_finite=check_finite,
            constraints=self._constraints
        )

        # Log configuration
        logger.info("LocalTrainer initialized:")
        logger.info(f"  Rule: {self.rule}")
        logger.info(f"  Loss: {self._loss_strategy.name()}")
        logger.info(f"  Connections: {len(self._connections)}")
        if self.modulator:
            logger.info(f"  Modulator: {self.modulator}")

        for conn_info in self._connections:
            logger.debug(f"    {conn_info}")

    def step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Perform one training step with local learning.

        Args:
            inputs: Input tensor
                   - [batch, seq_len, input_dim] for sequences
                   - [batch, input_dim] for static (time dim added automatically)
            targets: Target tensor (optional, for loss and 3-factor rules)
                    - [batch, output_dim] for regression
                    - [batch] for classification (class indices)

        Returns:
            Dictionary with metrics:
                - 'loss': Readout loss value (float)
                - 'total_update_norm': Sum of weight update norms (float)
                - 'weight_norm_change': Total change in weight norms (float)
                - 'update_metrics': List of UpdateMetrics for each connection

        Raises:
            StateCollectionError: If required states are missing
            DimensionMismatchError: If activity dimensions don't match
            ValueError: If update contains NaN/Inf (when check_finite=True)
        """
        # 1. Collect states via forward pass
        forward_result = self._state_collector.collect(inputs)

        # 2. Validate that required states are available
        conn_keys = [conn.key for conn in self._connections]
        self._state_collector.validate_states_for_connections(
            forward_result, conn_keys
        )

        # 3. Compute loss (for monitoring)
        loss_value = 0.0
        if targets is not None:
            with torch.no_grad():
                loss_tensor = self._loss_strategy.compute(
                    forward_result.outputs, targets
                )
                loss_value = loss_tensor.item()

        # 4. Compute modulator signal (for 3-factor rules)
        modulator_signal = None
        if targets is not None and self.modulator is not None:
            modulator_signal = self.modulator.compute(
                forward_result.outputs, targets
            )

        # 5. Apply updates to all connections
        update_metrics_list = self._weight_updater.update_all(
            self._connections, forward_result, modulator_signal
        )

        # 6. Compute aggregate statistics
        total_update_norm = self._weight_updater.get_total_update_norm(
            update_metrics_list
        )
        weight_norm_change = self._weight_updater.get_weight_norm_change(
            update_metrics_list
        )

        return {
            "loss": loss_value,
            "total_update_norm": total_update_norm,
            "weight_norm_change": weight_norm_change,
            "update_metrics": update_metrics_list,
        }

    def state_dict(self) -> dict[str, Any]:
        """Get trainer state dictionary.

        Returns:
            Dictionary containing trainer state (e.g., BCM thresholds)
        """
        state = {}

        # Rule state (e.g., BCM threshold)
        if hasattr(self.rule, 'threshold') and self.rule.threshold is not None:
            state['rule_threshold'] = self.rule.threshold

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load trainer state from dictionary.

        Args:
            state_dict: State dictionary to load
        """
        if 'rule_threshold' in state_dict:
            self.rule.threshold = state_dict['rule_threshold']
            logger.debug("Loaded rule threshold from state_dict")

    def reset(self) -> None:
        """Reset trainer state (e.g., BCM threshold, eligibility traces).

        Call this between episodes or when changing tasks.
        """
        if hasattr(self.rule, 'reset_threshold'):
            self.rule.reset_threshold()
            logger.debug("Reset learning rule state")

    def get_connections(self) -> list[ConnectionInfo]:
        """Get list of connections being trained.

        Returns:
            List of ConnectionInfo objects
        """
        return self._connections

    def get_loss_strategy(self) -> LossStrategy:
        """Get the current loss strategy.

        Returns:
            LossStrategy instance
        """
        return self._loss_strategy

    def set_loss_strategy(self, loss: str | LossStrategy) -> None:
        """Change the loss strategy.

        Args:
            loss: New loss strategy (string name or LossStrategy instance)
        """
        if isinstance(loss, str):
            self._loss_strategy = LossFactory.create(loss)
        elif isinstance(loss, LossStrategy):
            self._loss_strategy = loss
        else:
            raise TypeError(f"loss must be str or LossStrategy, got {type(loss)}")

        logger.info(f"Loss strategy changed to: {self._loss_strategy.name()}")

    def __repr__(self) -> str:
        return (
            f"LocalTrainer(\n"
            f"  rule={self.rule},\n"
            f"  loss={self._loss_strategy.name()},\n"
            f"  modulator={self.modulator},\n"
            f"  connections={len(self._connections)}\n"
            f")"
        )
