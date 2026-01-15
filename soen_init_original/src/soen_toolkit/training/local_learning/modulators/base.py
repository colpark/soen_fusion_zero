"""Base class for modulator sources."""

from abc import ABC, abstractmethod

import torch


class AbstractModulator(ABC):
    """Base class for modulator signals in 3-factor learning rules.

    Modulators provide a global signal (reward, error, attention) that gates
    local plasticity. Examples include:
    - Error-based: negative loss as reward
    - Reward-based: external reinforcement signal
    - Attention-based: saliency or importance weighting
    """

    @abstractmethod
    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute modulator signal from outputs and targets.

        Args:
            outputs: Model outputs [batch, output_dim]
            targets: Target values [batch, output_dim] or [batch]
            **kwargs: Additional context (e.g., rewards, attention masks)

        Returns:
            modulator: Scalar or [batch] tensor
                      - Scalar: same modulation for all samples
                      - [batch]: per-sample modulation
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
