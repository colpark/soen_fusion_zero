"""Adaptive learning rate wrappers for local learning rules.

These wrappers automatically adjust learning rates based on weight norms,
update magnitudes, or other dynamics to prevent instability.
"""

import torch

from soen_toolkit.training.local_learning.rules.base import AbstractLocalRule


class AdaptiveLRWrapper(AbstractLocalRule):
    """Wraps any local learning rule with adaptive learning rate.

    Automatically scales the learning rate based on:
    1. Weight norm (prevents explosion as weights grow)
    2. Update norm (prevents too-large updates)
    3. Activity statistics (adjusts for layer dynamics)

    This is particularly useful for unstable rules like pure Hebbian.

    Example:
        base_rule = HebbianRule(lr=0.1)
        adaptive_rule = AdaptiveLRWrapper(
            base_rule,
            method='weight_norm',
            target_weight_norm=1.0,
            clip_update_norm=0.1
        )
    """

    def __init__(
        self,
        base_rule: AbstractLocalRule,
        method: str = 'weight_norm',
        target_weight_norm: float = 1.0,
        clip_update_norm: float | None = None,
        min_lr_scale: float = 1e-6,
        max_lr_scale: float = 10.0,
    ):
        """Initialize adaptive learning rate wrapper.

        Args:
            base_rule: The underlying learning rule to wrap
            method: Adaptation method:
                - 'weight_norm': Scale by 1 / (1 + ||W||²)
                - 'update_norm': Clip updates by norm
                - 'activity': Scale by 1 / (1 + <activity²>)
            target_weight_norm: Target weight norm for 'weight_norm' method
            clip_update_norm: If set, clip update norm to this value
            min_lr_scale: Minimum learning rate scale factor (prevents zero)
            max_lr_scale: Maximum learning rate scale factor (prevents explosion)
        """
        super().__init__(lr=base_rule.lr)
        self.base_rule = base_rule
        self.method = method
        self.target_weight_norm = target_weight_norm
        self.clip_update_norm = clip_update_norm
        self.min_lr_scale = min_lr_scale
        self.max_lr_scale = max_lr_scale

    def compute_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute adaptive weight update.

        Args:
            pre_activity: Pre-synaptic activity [batch, pre_dim]
            post_activity: Post-synaptic activity [batch, post_dim]
            weights: Current weights [post_dim, pre_dim]
            modulator: Optional modulating signal

        Returns:
            weight_update: Adaptively scaled [post_dim, pre_dim]
        """
        # Get base update from wrapped rule
        update = self.base_rule.compute_update(
            pre_activity, post_activity, weights, modulator
        )

        # Compute adaptive scaling
        if self.method == 'weight_norm':
            # Scale by weight norm: lr_eff = lr / (1 + ||W||²/target²)
            weight_norm = torch.norm(weights).item()
            norm_ratio = (weight_norm / self.target_weight_norm) ** 2
            scale = 1.0 / (1.0 + norm_ratio)

        elif self.method == 'update_norm':
            # Scale by update norm: lr_eff = lr * min(1, target/||ΔW||)
            update_norm = torch.norm(update).item()
            if update_norm > 0:
                scale = min(1.0, self.target_weight_norm / update_norm)
            else:
                scale = 1.0

        elif self.method == 'activity':
            # Scale by activity statistics: lr_eff = lr / (1 + <activity²>)
            pre_var = (pre_activity ** 2).mean().item()
            post_var = (post_activity ** 2).mean().item()
            activity_scale = (pre_var * post_var) ** 0.5
            scale = 1.0 / (1.0 + activity_scale)

        else:
            raise ValueError(f"Unknown adaptation method: {self.method}")

        # Clamp scale to safe range
        scale = max(self.min_lr_scale, min(self.max_lr_scale, scale))

        # Apply scaling
        update = scale * update

        # Optional: clip update norm
        if self.clip_update_norm is not None:
            update_norm = torch.norm(update).item()
            if update_norm > self.clip_update_norm:
                update = update * (self.clip_update_norm / update_norm)

        return update

    def __repr__(self) -> str:
        return (
            f"AdaptiveLRWrapper({self.base_rule}, "
            f"method='{self.method}', "
            f"target_norm={self.target_weight_norm})"
        )


class LRScheduleWrapper(AbstractLocalRule):
    """Wraps a learning rule with a learning rate schedule.

    Supports common schedules like step decay, exponential decay, and cosine annealing.

    Example:
        base_rule = HebbianRule(lr=0.1)
        scheduled_rule = LRScheduleWrapper(
            base_rule,
            schedule='exponential',
            decay_rate=0.95,
            decay_every=10
        )

        # In training loop:
        for epoch in range(100):
            update = scheduled_rule.compute_update(...)
            scheduled_rule.step()  # Update schedule
    """

    def __init__(
        self,
        base_rule: AbstractLocalRule,
        schedule: str = 'step',
        decay_rate: float = 0.1,
        decay_every: int = 10,
        min_lr: float = 1e-6,
    ):
        """Initialize learning rate schedule wrapper.

        Args:
            base_rule: The underlying learning rule to wrap
            schedule: Schedule type:
                - 'step': Multiply by decay_rate every decay_every steps
                - 'exponential': Multiply by decay_rate^(1/decay_every) each step
                - 'cosine': Cosine annealing (needs max_steps set via set_max_steps)
            decay_rate: How much to decay (0 < decay_rate < 1)
            decay_every: For step/exponential schedules
            min_lr: Minimum learning rate floor
        """
        super().__init__(lr=base_rule.lr)
        self.base_rule = base_rule
        self.schedule = schedule
        self.decay_rate = decay_rate
        self.decay_every = decay_every
        self.min_lr = min_lr
        self.initial_lr = base_rule.lr
        self.step_count = 0
        self.max_steps = None  # For cosine annealing

    def set_max_steps(self, max_steps: int):
        """Set maximum steps for cosine annealing schedule."""
        self.max_steps = max_steps

    def get_current_lr(self) -> float:
        """Compute current learning rate based on schedule."""
        if self.schedule == 'step':
            # Step decay: lr = initial_lr * decay_rate^(step // decay_every)
            decay_factor = self.decay_rate ** (self.step_count // self.decay_every)
            current_lr = self.initial_lr * decay_factor

        elif self.schedule == 'exponential':
            # Exponential decay: lr = initial_lr * decay_rate^(step / decay_every)
            decay_factor = self.decay_rate ** (self.step_count / self.decay_every)
            current_lr = self.initial_lr * decay_factor

        elif self.schedule == 'cosine':
            # Cosine annealing
            if self.max_steps is None:
                raise ValueError("Must call set_max_steps() for cosine schedule")
            progress = min(self.step_count / self.max_steps, 1.0)
            current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + torch.cos(torch.tensor(progress * 3.14159)).item()
            )

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule}")

        # Apply floor
        return max(current_lr, self.min_lr)

    def compute_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute update with scheduled learning rate.

        Args:
            pre_activity: Pre-synaptic activity [batch, pre_dim]
            post_activity: Post-synaptic activity [batch, post_dim]
            weights: Current weights [post_dim, pre_dim]
            modulator: Optional modulating signal

        Returns:
            weight_update: [post_dim, pre_dim]
        """
        # Temporarily set base rule's lr to scheduled value
        original_lr = self.base_rule.lr
        self.base_rule.lr = self.get_current_lr()

        # Compute update with scheduled lr
        update = self.base_rule.compute_update(
            pre_activity, post_activity, weights, modulator
        )

        # Restore original lr
        self.base_rule.lr = original_lr

        return update

    def step(self):
        """Advance the schedule by one step (call once per epoch/batch)."""
        self.step_count += 1

    def reset(self):
        """Reset the schedule to initial state."""
        self.step_count = 0

    def __repr__(self) -> str:
        current_lr = self.get_current_lr()
        return (
            f"LRScheduleWrapper({self.base_rule}, "
            f"schedule='{self.schedule}', "
            f"current_lr={current_lr:.6f})"
        )
