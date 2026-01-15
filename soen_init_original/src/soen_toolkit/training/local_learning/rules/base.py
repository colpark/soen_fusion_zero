"""Base class for local learning rules."""

from abc import ABC, abstractmethod

import torch


class AbstractLocalRule(ABC):
    """Base class for all local learning rules.

    Local learning rules compute weight updates based on local information
    (pre-synaptic and post-synaptic activity) rather than global error signals.

    Subclasses should implement the `compute_update` method.
    """

    def __init__(self, lr: float = 0.01):
        """Initialize the learning rule.

        Args:
            lr: Learning rate (step size for weight updates)
        """
        self.lr = lr

    @property
    def requires_trajectory(self) -> bool:
        """Return True if this rule requires full trajectories rather than final states."""
        return False

    @abstractmethod
    def compute_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weight update ΔW based on local activity.

        Args:
            pre_activity: Pre-synaptic activity.
                         Shape: [batch, dim] if requires_trajectory is False.
                         Shape: [batch, seq, dim] if requires_trajectory is True.
            post_activity: Post-synaptic activity.
                          Shape: [batch, dim] if requires_trajectory is False.
                          Shape: [batch, seq, dim] if requires_trajectory is True.
            weights: Current weight matrix [post_dim, pre_dim] (PyTorch convention)
            modulator: Optional modulating signal for 3-factor rules
                      Can be scalar or [batch] tensor

        Returns:
            weight_update: ΔW with shape [pre_dim, post_dim]

        Note:
            Implementations should detach inputs to prevent backpropagation.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr})"
