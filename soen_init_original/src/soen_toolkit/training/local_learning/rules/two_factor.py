"""Two-factor local learning rules (Hebbian-like).

These rules update weights based on pre-synaptic and post-synaptic activity only,
without requiring a global error signal.
"""

import torch

from soen_toolkit.training.local_learning.rules.base import AbstractLocalRule


class HebbianRule(AbstractLocalRule):
    """Simple Hebbian learning rule.

    Updates weights based on correlation between pre and post activities:
        ΔW = η · <pre · post^T>

    where <·> denotes batch average.

    This is the simplest local learning rule and forms the basis for many
    biological learning theories.
    """

    def compute_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Hebbian weight update.

        Args:
            pre_activity: Pre-synaptic activity [batch, pre_dim]
            post_activity: Post-synaptic activity [batch, post_dim]
            weights: Current weights [pre_dim, post_dim] (not used in basic Hebbian)
            modulator: Ignored for 2-factor rule

        Returns:
            weight_update: [pre_dim, post_dim]
        """
        # Detach to prevent backprop
        pre = pre_activity.detach()
        post = post_activity.detach()

        # Compute outer product averaged over batch
        # PyTorch weights are [post_dim, pre_dim], so we compute post x pre^T
        # einsum('bj,bi->ji') computes sum_b post[b,j] * pre[b,i]
        batch_size = pre.size(0)
        update = torch.einsum('bj,bi->ji', post, pre) / batch_size

        return self.lr * update


class OjaRule(AbstractLocalRule):
    """Oja's rule: normalized Hebbian learning.

    Updates weights with normalization to prevent unbounded growth:
        ΔW = η · (<pre · post^T> - <post^2> · W)

    The normalization term keeps weights bounded and can lead to
    principal component extraction.

    Reference:
        Oja, E. (1982). Simplified neuron model as a principal component analyzer.
        Journal of Mathematical Biology, 15(3), 267-273.
    """

    def compute_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Oja's rule weight update.

        Args:
            pre_activity: Pre-synaptic activity [batch, pre_dim]
            post_activity: Post-synaptic activity [batch, post_dim]
            weights: Current weights [pre_dim, post_dim]
            modulator: Ignored for 2-factor rule

        Returns:
            weight_update: [pre_dim, post_dim]
        """
        # Detach to prevent backprop
        pre = pre_activity.detach()
        post = post_activity.detach()

        batch_size = pre.size(0)

        # Hebbian term: <post · pre^T> (PyTorch convention)
        hebbian = torch.einsum('bj,bi->ji', post, pre) / batch_size

        # Normalization term: <post^2> · W
        # post^2: [batch, post_dim] -> mean over batch: [post_dim]
        post_squared = (post ** 2).mean(dim=0)  # [post_dim]

        # Broadcast and multiply with weights
        # weights: [post_dim, pre_dim], post_squared: [post_dim]
        normalization = weights * post_squared.unsqueeze(1)  # [post_dim, pre_dim]

        update = hebbian - normalization

        return self.lr * update


class BCMRule(AbstractLocalRule):
    """Bienenstock-Cooper-Munro (BCM) learning rule.

    A sliding threshold learning rule where the threshold depends on
    the average post-synaptic activity:
        ΔW = η · pre · post · (post - θ)
        θ = <post^2>

    This rule can lead to selectivity and has been used to model
    cortical plasticity.

    Reference:
        Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982).
        Theory for the development of neuron selectivity: orientation
        specificity and binocular interaction in visual cortex.
    """

    def __init__(self, lr: float = 0.01, threshold_momentum: float = 0.9):
        """Initialize BCM rule.

        Args:
            lr: Learning rate
            threshold_momentum: Momentum for sliding threshold (0-1)
                               Higher values = slower threshold adaptation
        """
        super().__init__(lr)
        self.threshold_momentum = threshold_momentum
        self.threshold = None  # Will be initialized on first call

    def compute_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute BCM weight update.

        Args:
            pre_activity: Pre-synaptic activity [batch, pre_dim]
            post_activity: Post-synaptic activity [batch, post_dim]
            weights: Current weights [pre_dim, post_dim] (not used directly)
            modulator: Ignored for 2-factor rule

        Returns:
            weight_update: [pre_dim, post_dim]
        """
        # Detach to prevent backprop
        pre = pre_activity.detach()
        post = post_activity.detach()

        batch_size = pre.size(0)
        post_dim = post.size(1)

        # Initialize threshold if first call
        if self.threshold is None:
            self.threshold = torch.zeros(post_dim, device=post.device)

        # Update sliding threshold: θ = momentum * θ + (1-momentum) * <post^2>
        current_threshold = (post ** 2).mean(dim=0)  # [post_dim]
        self.threshold = (
            self.threshold_momentum * self.threshold +
            (1 - self.threshold_momentum) * current_threshold
        )

        # BCM learning: ΔW = η · <pre · post · (post - θ)>
        # post - threshold: [batch, post_dim]
        post_minus_threshold = post - self.threshold.unsqueeze(0)

        # Modulated post: post * (post - θ)
        modulated_post = post * post_minus_threshold  # [batch, post_dim]

        # Outer product: <modulated_post · pre^T> (PyTorch convention)
        update = torch.einsum('bj,bi->ji', modulated_post, pre) / batch_size

        return self.lr * update

    def reset_threshold(self):
        """Reset the sliding threshold (useful between episodes/tasks)."""
        self.threshold = None
