"""Weight update logic for local learning.

This module handles applying weight updates to model connections,
with proper validation and error handling.
"""

from dataclasses import dataclass
import logging

import torch

from soen_toolkit.training.local_learning.connection_resolver import (
    ConnectionInfo,
    ConnectionKey,
)
from soen_toolkit.training.local_learning.constraints import AbstractConstraint
from soen_toolkit.training.local_learning.rules.base import AbstractLocalRule
from soen_toolkit.training.local_learning.state_collector import ForwardResult, StateCollectionError

logger = logging.getLogger(__name__)


class DimensionMismatchError(ValueError):
    """Raised when activity dimensions don't match connection dimensions."""
    pass


@dataclass
class UpdateMetrics:
    """Metrics from a weight update operation.

    Attributes:
        connection: Connection that was updated
        update_norm: L2 norm of the weight update
        weight_norm_before: Weight norm before update
        weight_norm_after: Weight norm after update
        pre_activity_stats: Stats about pre-synaptic activity (mean, std)
        post_activity_stats: Stats about post-synaptic activity (mean, std)
    """
    connection: ConnectionKey
    update_norm: float
    weight_norm_before: float
    weight_norm_after: float
    pre_activity_stats: tuple[float, float]  # (mean, std)
    post_activity_stats: tuple[float, float]  # (mean, std)

    def __str__(self) -> str:
        return (
            f"{self.connection}: "
            f"Δ||W||={self.update_norm:.6f}, "
            f"||W||: {self.weight_norm_before:.3f}→{self.weight_norm_after:.3f}"
        )


class WeightUpdater:
    """Applies local learning updates to model weights.

    Responsibilities:
    - Extract pre/post activities from states
    - Validate activity dimensions match connection dimensions
    - Compute updates via learning rule
    - Apply updates to model weights
    - Track update statistics

    Example:
        >>> updater = WeightUpdater(model, rule)
        >>> metrics = updater.update(conn_info, forward_result, modulator)
        >>> print(f"Update norm: {metrics.update_norm:.6f}")
    """

    def __init__(
        self,
        model,
        rule: AbstractLocalRule,
        check_finite: bool = True,
        constraints: dict[ConnectionKey, list[AbstractConstraint]] | None = None
    ):
        """Initialize weight updater.

        Args:
            model: SOENModelCore instance
            rule: Local learning rule to use
            check_finite: If True, check for NaN/Inf in updates
            constraints: Dictionary mapping ConnectionKey to list of constraints
        """
        self.model = model
        self.rule = rule
        self.check_finite = check_finite
        self.constraints = constraints or {}

    def update(
        self,
        conn_info: ConnectionInfo,
        forward_result: ForwardResult,
        modulator: torch.Tensor | None = None
    ) -> UpdateMetrics:
        """Apply local learning update to a connection.

        Args:
            conn_info: Connection to update
            forward_result: Forward pass result with layer states
            modulator: Optional modulating signal for 3-factor rules

        Returns:
            UpdateMetrics with statistics about the update

        Raises:
            DimensionMismatchError: If activity dimensions don't match connection
            ValueError: If update contains NaN/Inf (when check_finite=True)
        """
        # Get pre and post activities
        if self.rule.requires_trajectory:
            if not forward_result.has_trajectory_for_layer(conn_info.key.from_layer):
                raise StateCollectionError(
                    f"Rule {self.rule} requires trajectories, but trajectory for "
                    f"layer {conn_info.key.from_layer} is missing."
                )
            pre_activity = forward_result.layer_trajectories[conn_info.key.from_layer]
            post_activity = forward_result.layer_trajectories[conn_info.key.to_layer]
        else:
            pre_activity = forward_result.get_state_or_raise(conn_info.key.from_layer)
            post_activity = forward_result.get_state_or_raise(conn_info.key.to_layer)

        # Validate dimensions
        self._validate_dimensions(conn_info, pre_activity, post_activity)

        # Get current weights
        weights = self.model.connections[conn_info.param_name]
        weight_norm_before = torch.norm(weights).item()

        # Compute statistics before update
        pre_stats = self._compute_activity_stats(pre_activity)
        post_stats = self._compute_activity_stats(post_activity)

        # Compute update using learning rule
        delta_w = self.rule.compute_update(
            pre_activity, post_activity, weights, modulator
        )

        # Validate update
        if self.check_finite:
            self._validate_update(delta_w, conn_info)

        # Apply update (in-place, no gradient)
        with torch.no_grad():
            weights.data.add_(delta_w)

            # Apply constraints if any
            if conn_info.key in self.constraints:
                for constraint in self.constraints[conn_info.key]:
                    constraint(weights)

        # Compute post-update statistics
        update_norm = torch.norm(delta_w).item()
        weight_norm_after = torch.norm(weights).item()

        # Create metrics
        metrics = UpdateMetrics(
            connection=conn_info.key,
            update_norm=update_norm,
            weight_norm_before=weight_norm_before,
            weight_norm_after=weight_norm_after,
            pre_activity_stats=pre_stats,
            post_activity_stats=post_stats
        )

        logger.debug(f"Updated {metrics}")

        return metrics

    def update_all(
        self,
        connections: list[ConnectionInfo],
        forward_result: ForwardResult,
        modulator: torch.Tensor | None = None
    ) -> list[UpdateMetrics]:
        """Apply updates to multiple connections.

        Args:
            connections: List of connections to update
            forward_result: Forward pass result with layer states
            modulator: Optional modulating signal

        Returns:
            List of UpdateMetrics for each connection
        """
        all_metrics = []

        for conn_info in connections:
            try:
                metrics = self.update(conn_info, forward_result, modulator)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(
                    f"Failed to update connection {conn_info.key}: {e}"
                )
                # Re-raise to fail fast
                raise

        return all_metrics

    def _validate_dimensions(
        self,
        conn_info: ConnectionInfo,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor
    ) -> None:
        """Validate that activity dimensions match connection dimensions.

        Args:
            conn_info: Connection metadata
            pre_activity: Pre-synaptic activity [batch, pre_dim]
            post_activity: Post-synaptic activity [batch, post_dim]

        Raises:
            DimensionMismatchError: If dimensions don't match
        """
        # Check pre-synaptic dimension
        # Trajectory has shape [batch, seq, dim], state has [batch, dim]
        feat_dim = 2 if self.rule.requires_trajectory else 1

        if pre_activity.size(feat_dim) != conn_info.from_dim:
            raise DimensionMismatchError(
                f"Pre-synaptic activity dimension mismatch for {conn_info.key}. "
                f"Expected {conn_info.from_dim}, got {pre_activity.size(feat_dim)}. "
                f"Activity shape: {pre_activity.shape}"
            )

        # Check post-synaptic dimension
        if post_activity.size(feat_dim) != conn_info.to_dim:
            raise DimensionMismatchError(
                f"Post-synaptic activity dimension mismatch for {conn_info.key}. "
                f"Expected {conn_info.to_dim}, got {post_activity.size(feat_dim)}. "
                f"Activity shape: {post_activity.shape}"
            )

        # Check batch size consistency
        if pre_activity.size(0) != post_activity.size(0):
            raise DimensionMismatchError(
                f"Batch size mismatch for {conn_info.key}. "
                f"Pre: {pre_activity.size(0)}, Post: {post_activity.size(0)}"
            )

    def _validate_update(
        self,
        delta_w: torch.Tensor,
        conn_info: ConnectionInfo
    ) -> None:
        """Validate that update is finite (no NaN/Inf).

        Args:
            delta_w: Weight update tensor
            conn_info: Connection metadata

        Raises:
            ValueError: If update contains NaN or Inf
        """
        if not torch.isfinite(delta_w).all():
            nan_count = torch.isnan(delta_w).sum().item()
            inf_count = torch.isinf(delta_w).sum().item()

            raise ValueError(
                f"Non-finite values in weight update for {conn_info.key}. "
                f"NaN count: {nan_count}, Inf count: {inf_count}. "
                f"This may indicate numerical instability. "
                f"Try reducing learning rate or using adaptive LR."
            )

    def _compute_activity_stats(
        self,
        activity: torch.Tensor
    ) -> tuple[float, float]:
        """Compute mean and std of activity.

        Args:
            activity: Activity tensor [batch, dim]

        Returns:
            Tuple of (mean, std)
        """
        mean = activity.mean().item()
        std = activity.std().item()
        return (mean, std)

    def get_total_update_norm(
        self,
        metrics_list: list[UpdateMetrics]
    ) -> float:
        """Compute total update norm across all connections.

        Args:
            metrics_list: List of UpdateMetrics

        Returns:
            Sum of update norms
        """
        return sum(m.update_norm for m in metrics_list)

    def get_weight_norm_change(
        self,
        metrics_list: list[UpdateMetrics]
    ) -> float:
        """Compute total change in weight norms.

        Args:
            metrics_list: List of UpdateMetrics

        Returns:
            Sum of (weight_norm_after - weight_norm_before)
        """
        return sum(
            m.weight_norm_after - m.weight_norm_before
            for m in metrics_list
        )
