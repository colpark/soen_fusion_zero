"""Error-based modulators for 3-factor learning.

These modulators derive a reward signal from prediction error (loss).
Typically, lower error = higher reward.
"""

import torch
import torch.nn.functional as F

from soen_toolkit.training.local_learning.modulators.base import AbstractModulator


class MSEErrorModulator(AbstractModulator):
    """Use negative MSE as reward signal.

    Computes per-sample MSE and returns it as a negative reward:
        r = -scale * MSE(output, target)

    Lower error = higher reward, encouraging learning.
    """

    def __init__(self, scale: float = 1.0, per_sample: bool = True):
        """Initialize MSE error modulator.

        Args:
            scale: Scaling factor for the modulator
            per_sample: If True, return per-sample modulators [batch]
                       If False, return scalar (mean over batch)
        """
        self.scale = scale
        self.per_sample = per_sample

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute MSE-based modulator.

        Args:
            outputs: Model outputs [batch, output_dim]
            targets: Target values [batch, output_dim]

        Returns:
            modulator: Negative MSE
                      - [batch] if per_sample=True
                      - scalar if per_sample=False
        """
        # Detach to prevent backprop
        out = outputs.detach()
        tgt = targets.detach()

        # Per-sample MSE: mean over output_dim
        mse = ((out - tgt) ** 2).mean(dim=1)  # [batch]

        # Negative error = reward
        modulator = -self.scale * mse

        if self.per_sample:
            return modulator  # [batch]
        else:
            return modulator.mean()  # scalar

    def __repr__(self) -> str:
        return f"MSEErrorModulator(scale={self.scale}, per_sample={self.per_sample})"


class CrossEntropyErrorModulator(AbstractModulator):
    """Use negative cross-entropy as reward signal.

    For classification tasks. Computes per-sample cross-entropy and returns
    negative value as reward.

    Note: Expects outputs to be logits (not softmax probabilities).
    """

    def __init__(self, scale: float = 1.0, per_sample: bool = True):
        """Initialize cross-entropy error modulator.

        Args:
            scale: Scaling factor for the modulator
            per_sample: If True, return per-sample modulators [batch]
                       If False, return scalar (mean over batch)
        """
        self.scale = scale
        self.per_sample = per_sample

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute cross-entropy based modulator.

        Args:
            outputs: Model logits [batch, num_classes]
            targets: Target class indices [batch] or one-hot [batch, num_classes]

        Returns:
            modulator: Negative cross-entropy
                      - [batch] if per_sample=True
                      - scalar if per_sample=False
        """
        # Detach to prevent backprop
        out = outputs.detach()
        tgt = targets.detach()

        # Handle one-hot vs class indices
        if tgt.dim() == 2:
            # One-hot: convert to class indices
            tgt = tgt.argmax(dim=1)

        # Per-sample cross-entropy
        ce = F.cross_entropy(out, tgt, reduction='none')  # [batch]

        # Negative error = reward
        modulator = -self.scale * ce

        if self.per_sample:
            return modulator  # [batch]
        else:
            return modulator.mean()  # scalar

    def __repr__(self) -> str:
        return f"CrossEntropyErrorModulator(scale={self.scale}, per_sample={self.per_sample})"


class AccuracyBasedModulator(AbstractModulator):
    """Use classification accuracy as binary reward.

    Returns +reward for correct predictions, -reward for incorrect.
    Useful for reinforcement learning style training on classification.
    """

    def __init__(self, correct_reward: float = 1.0, incorrect_reward: float = -1.0):
        """Initialize accuracy-based modulator.

        Args:
            correct_reward: Reward for correct predictions
            incorrect_reward: Reward for incorrect predictions
        """
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute accuracy-based modulator.

        Args:
            outputs: Model logits [batch, num_classes]
            targets: Target class indices [batch] or one-hot [batch, num_classes]

        Returns:
            modulator: [batch] tensor with correct_reward or incorrect_reward
        """
        # Detach to prevent backprop
        out = outputs.detach()
        tgt = targets.detach()

        # Handle one-hot vs class indices
        if tgt.dim() == 2:
            tgt = tgt.argmax(dim=1)

        # Get predictions
        predictions = out.argmax(dim=1)

        # Compute correctness
        correct = (predictions == tgt).float()  # [batch], 1.0 or 0.0

        # Map to rewards
        modulator = correct * self.correct_reward + (1 - correct) * self.incorrect_reward

        return modulator  # [batch]

    def __repr__(self) -> str:
        return (
            f"AccuracyBasedModulator("
            f"correct={self.correct_reward}, "
            f"incorrect={self.incorrect_reward})"
        )
