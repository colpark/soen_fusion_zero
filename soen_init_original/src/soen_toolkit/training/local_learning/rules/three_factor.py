"""Three-factor local learning rules.

These rules incorporate a third factor (modulator/reward signal) in addition to
pre- and post-synaptic activity. The modulator typically comes from a global
signal like error or reward.
"""

import torch

from soen_toolkit.training.local_learning.rules.base import AbstractLocalRule


class RewardModulatedHebbianRule(AbstractLocalRule):
    """Reward-modulated Hebbian learning (3-factor rule).

    Updates weights based on Hebbian correlation modulated by a reward/error signal:
        ΔW = η · r · <pre · post^T>

    where r is the modulator (reward, error, attention, etc.)

    This rule is biologically plausible and used in models of dopaminergic learning,
    reinforcement learning, and attention-gated plasticity.
    """

    def __init__(self, lr: float = 0.01, baseline_subtract: bool = False):
        """Initialize reward-modulated Hebbian rule.

        Args:
            lr: Learning rate
            baseline_subtract: If True, subtract mean reward as baseline
                              (reduces variance, useful for RL)
        """
        super().__init__(lr)
        self.baseline_subtract = baseline_subtract

    def compute_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute reward-modulated Hebbian update.

        Args:
            pre_activity: Pre-synaptic activity [batch, pre_dim]
            post_activity: Post-synaptic activity [batch, post_dim]
            weights: Current weights [pre_dim, post_dim] (not used)
            modulator: Modulating signal (reward/error)
                      - None: falls back to unmodulated Hebbian
                      - Scalar: same modulation for all samples
                      - [batch]: per-sample modulation

        Returns:
            weight_update: [pre_dim, post_dim]
        """
        # Detach to prevent backprop
        pre = pre_activity.detach()
        post = post_activity.detach()

        batch_size = pre.size(0)

        # Base Hebbian term
        if modulator is None:
            # No modulator: fall back to plain Hebbian (PyTorch convention)
            update = torch.einsum('bj,bi->ji', post, pre) / batch_size
            return self.lr * update

        # Detach modulator
        mod = modulator.detach()

        # Baseline subtraction (variance reduction)
        if self.baseline_subtract and mod.dim() > 0:
            mod = mod - mod.mean()

        # Apply modulation
        if mod.dim() == 0:
            # Scalar modulator: r · <post · pre^T> (PyTorch convention)
            update = mod * torch.einsum('bj,bi->ji', post, pre) / batch_size
        elif mod.dim() == 1:
            # Per-sample modulator: <r · post · pre^T>
            # einsum('b,bj,bi->ji') computes sum_b r[b] * post[b,j] * pre[b,i]
            update = torch.einsum('b,bj,bi->ji', mod, post, pre) / batch_size
        else:
            raise ValueError(
                f"Modulator must be scalar or 1D, got shape {mod.shape}"
            )

        return self.lr * update


class RewardModulatedOjaRule(AbstractLocalRule):
    """Reward-modulated Oja's rule (3-factor with normalization).

    Combines Oja's normalization with reward modulation:
        ΔW = η · r · (<pre · post^T> - <post^2> · W)

    This prevents unbounded weight growth while allowing reward-guided learning.
    """

    def __init__(self, lr: float = 0.01, baseline_subtract: bool = False):
        """Initialize reward-modulated Oja rule.

        Args:
            lr: Learning rate
            baseline_subtract: If True, subtract mean reward as baseline
        """
        super().__init__(lr)
        self.baseline_subtract = baseline_subtract

    def compute_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute reward-modulated Oja update.

        Args:
            pre_activity: Pre-synaptic activity [batch, pre_dim]
            post_activity: Post-synaptic activity [batch, post_dim]
            weights: Current weights [pre_dim, post_dim]
            modulator: Modulating signal (reward/error)

        Returns:
            weight_update: [pre_dim, post_dim]
        """
        # Detach to prevent backprop
        pre = pre_activity.detach()
        post = post_activity.detach()

        batch_size = pre.size(0)

        # Oja terms (PyTorch convention)
        hebbian = torch.einsum('bj,bi->ji', post, pre) / batch_size
        post_squared = (post ** 2).mean(dim=0)
        normalization = weights * post_squared.unsqueeze(1)
        oja_update = hebbian - normalization

        # Apply modulation
        if modulator is None:
            return self.lr * oja_update

        mod = modulator.detach()

        if self.baseline_subtract and mod.dim() > 0:
            mod = mod - mod.mean()

        if mod.dim() == 0:
            # Scalar modulator
            return self.lr * mod * oja_update
        elif mod.dim() == 1:
            # Per-sample modulator: recompute with modulation
            hebbian_mod = torch.einsum('b,bj,bi->ji', mod, post, pre) / batch_size
            # Normalization term also modulated
            post_sq_mod = ((post ** 2) * mod.unsqueeze(1)).mean(dim=0)
            norm_mod = weights * post_sq_mod.unsqueeze(1)
            return self.lr * (hebbian_mod - norm_mod)
        else:
            raise ValueError(
                f"Modulator must be scalar or 1D, got shape {mod.shape}"
            )
