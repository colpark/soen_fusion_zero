"""Loss computation strategies for local learning.

This module provides pluggable loss functions using the Strategy pattern,
making it easy to add new loss types without modifying trainer code.
"""

from abc import ABC, abstractmethod
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LossStrategy(ABC):
    """Abstract base class for loss computation strategies.

    Subclasses implement specific loss functions (MSE, CrossEntropy, etc.)
    with consistent interface for the trainer.
    """

    @abstractmethod
    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between outputs and targets.

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            Scalar loss tensor
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of loss function."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MSELoss(LossStrategy):
    """Mean Squared Error loss for regression tasks.

    Formula: L = (1/N) Σ(y_pred - y_true)²
    """

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            outputs: Model outputs [batch, output_dim]
            targets: Target values [batch, output_dim]

        Returns:
            Scalar MSE loss
        """
        return F.mse_loss(outputs, targets, reduction='mean')

    def name(self) -> str:
        return "MSE"


class CrossEntropyLoss(LossStrategy):
    """Cross-entropy loss for classification tasks.

    Handles both:
    - One-hot encoded targets [batch, num_classes]
    - Class index targets [batch]
    """

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            outputs: Model logits [batch, num_classes]
            targets: Target labels
                    - [batch, num_classes] one-hot encoded
                    - [batch] class indices

        Returns:
            Scalar cross-entropy loss
        """
        # Convert one-hot to class indices if needed
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)

        return F.cross_entropy(outputs, targets, reduction='mean')

    def name(self) -> str:
        return "CrossEntropy"


class MAELoss(LossStrategy):
    """Mean Absolute Error (L1) loss for regression tasks.

    Formula: L = (1/N) Σ|y_pred - y_true|

    More robust to outliers than MSE.
    """

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute MAE loss.

        Args:
            outputs: Model outputs [batch, output_dim]
            targets: Target values [batch, output_dim]

        Returns:
            Scalar MAE loss
        """
        return F.l1_loss(outputs, targets, reduction='mean')

    def name(self) -> str:
        return "MAE"


class HuberLoss(LossStrategy):
    """Huber loss for robust regression.

    Combines MSE (for small errors) and MAE (for large errors).
    Less sensitive to outliers than pure MSE.

    Args:
        delta: Threshold for switching between MSE and MAE behavior
    """

    def __init__(self, delta: float = 1.0):
        """Initialize Huber loss.

        Args:
            delta: Threshold parameter (default: 1.0)
        """
        self.delta = delta

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Huber loss.

        Args:
            outputs: Model outputs [batch, output_dim]
            targets: Target values [batch, output_dim]

        Returns:
            Scalar Huber loss
        """
        return F.huber_loss(outputs, targets, reduction='mean', delta=self.delta)

    def name(self) -> str:
        return f"Huber(δ={self.delta})"

    def __repr__(self) -> str:
        return f"HuberLoss(delta={self.delta})"


class NoLoss(LossStrategy):
    """Null loss strategy that always returns zero.

    Useful for unsupervised learning where no targets are available.
    """

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Return zero loss.

        Args:
            outputs: Model outputs (ignored)
            targets: Targets (ignored)

        Returns:
            Zero tensor
        """
        return torch.tensor(0.0, device=outputs.device)

    def name(self) -> str:
        return "None"


class LossFactory:
    """Factory for creating loss strategies from string names.

    Makes it easy to specify losses via config files or command-line args.

    Example:
        >>> loss = LossFactory.create("mse")
        >>> loss = LossFactory.create("cross_entropy")
        >>> loss = LossFactory.create("huber", delta=2.0)
    """

    _REGISTRY = {
        "mse": MSELoss,
        "mae": MAELoss,
        "l1": MAELoss,  # Alias
        "cross_entropy": CrossEntropyLoss,
        "ce": CrossEntropyLoss,  # Alias
        "huber": HuberLoss,
        "none": NoLoss,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> LossStrategy:
        """Create a loss strategy by name.

        Args:
            name: Loss function name (case-insensitive)
                 Options: "mse", "mae", "cross_entropy", "huber", "none"
            **kwargs: Additional arguments for loss constructor (e.g., delta for Huber)

        Returns:
            LossStrategy instance

        Raises:
            ValueError: If loss name is not recognized
        """
        name_lower = name.lower()

        if name_lower not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown loss function '{name}'. "
                f"Available options: {available}"
            )

        loss_class = cls._REGISTRY[name_lower]
        return loss_class(**kwargs)

    @classmethod
    def register(cls, name: str, loss_class: type[LossStrategy]) -> None:
        """Register a custom loss strategy.

        Allows users to add their own loss functions to the factory.

        Args:
            name: Name to register under
            loss_class: LossStrategy subclass

        Example:
            >>> class MyCustomLoss(LossStrategy):
            ...     def compute(self, outputs, targets):
            ...         return ...
            >>> LossFactory.register("custom", MyCustomLoss)
        """
        if not issubclass(loss_class, LossStrategy):
            raise TypeError(
                f"loss_class must be a subclass of LossStrategy, "
                f"got {loss_class}"
            )

        cls._REGISTRY[name.lower()] = loss_class
        logger.info(f"Registered custom loss: {name}")

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available loss functions.

        Returns:
            List of registered loss names
        """
        return sorted(cls._REGISTRY.keys())
