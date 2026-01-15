"""Weight constraints for local learning.

This module provides constraints that can be applied to weights after a local update,
useful for enforcing physical hardware limits or biological principles.
"""

from abc import ABC, abstractmethod

import torch


class AbstractConstraint(ABC):
    """Base class for weight constraints."""

    @abstractmethod
    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply the constraint to the weights in-place.

        Args:
            weights: Weight tensor to constrain

        Returns:
            The constrained weight tensor
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SignPreservationConstraint(AbstractConstraint):
    """Ensures weights maintain their initial polarity (Excitatory/Inhibitory).

    If a weight was initially positive, it will be clamped to [0, inf).
    If a weight was initially negative, it will be clamped to (-inf, 0].
    """

    def __init__(self, initial_weights: torch.Tensor):
        """Initialize with a reference to initial weights to determine polarity.

        Args:
            initial_weights: Weight tensor used to determine the sign mask
        """
        self.mask = torch.sign(initial_weights)

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        """Clamp weights to preserve sign."""
        with torch.no_grad():
            positive_mask = self.mask > 0
            negative_mask = self.mask < 0

            # For positive synapses, clamp to [0, inf)
            if positive_mask.any():
                weights.data[positive_mask] = weights.data[positive_mask].clamp(min=0)

            # For negative synapses, clamp to (-inf, 0]
            if negative_mask.any():
                weights.data[negative_mask] = weights.data[negative_mask].clamp(max=0)

        return weights


class RangeConstraint(AbstractConstraint):
    """Clamps weights to a specific range."""

    def __init__(self, min_val: float | None = None, max_val: float | None = None):
        """Initialize with min/max values.

        Args:
            min_val: Minimum weight value
            max_val: Maximum weight value
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        """Clamp weights to range."""
        with torch.no_grad():
            weights.data.clamp_(min=self.min_val, max=self.max_val)
        return weights


class MaxNormConstraint(AbstractConstraint):
    """Constrains the norm of weights (e.g., L2 norm) to a maximum value."""

    def __init__(self, max_norm: float = 1.0, dim: int = 1):
        """Initialize with max norm.

        Args:
            max_norm: Maximum allowed norm
            dim: Dimension along which to compute the norm (default: 1, columns)
        """
        self.max_norm = max_norm
        self.dim = dim

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        """Renormalize weights if norm exceeds max_norm."""
        with torch.no_grad():
            norm = weights.norm(p=2, dim=self.dim, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm)
            weights.data.mul_(desired / (1e-7 + norm))
        return weights
