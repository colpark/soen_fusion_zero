"""Feature hook infrastructure for JAX forward passes.

This module provides hook points for injecting functionality like noise,
power tracking, and quantization into the forward pass without modifying
the core forward logic.

The design mirrors the PyTorch FeatureHook pattern in the core layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp


@dataclass
class StepContext:
    """Immutable context passed to hooks during forward pass.

    Contains information about the current step that hooks may use
    for conditional behavior or logging.
    """

    batch_size: int
    time_step: int
    layer_idx: int
    layer_id: int
    dtype: jnp.dtype
    is_training: bool = False

    # Optional: solver type for solver-specific behavior
    solver: str = "layerwise"


@dataclass
class StepPayload:
    """Mutable payload for hook transformations.

    Hooks can modify these values to affect the forward computation.
    Values set to None are not modified.
    """

    phi: jax.Array | None = None  # Input flux [B, D]
    state: jax.Array | None = None  # Current layer state [B, D]
    params: dict[str, jax.Array] | None = None  # Layer parameters
    extras: dict[str, Any] = field(default_factory=dict)  # For g values, power, etc.


class FeatureHook(Protocol):
    """Protocol for feature hooks.

    Hooks are called before and after each layer step during forward passes.
    They can modify the payload to inject noise, track power, apply
    quantization, etc.

    Example implementations:
        - NoiseHook: Adds Gaussian noise to phi/state
        - PowerTracker: Records g values and power consumption
        - QATHook: Applies fake quantization to weights
    """

    def on_before_step(self, ctx: StepContext, payload: StepPayload) -> StepPayload:
        """Called before layer dynamics are applied.

        Use this to:
        - Add noise to input phi
        - Quantize weights in params
        - Initialize power tracking in extras

        Args:
            ctx: Immutable step context
            payload: Mutable payload to transform

        Returns:
            Transformed payload (may be same object)
        """
        ...

    def on_after_step(self, ctx: StepContext, payload: StepPayload) -> StepPayload:
        """Called after layer dynamics are applied.

        Use this to:
        - Add noise to output state
        - Record power consumption in extras
        - Post-process state values

        Args:
            ctx: Immutable step context
            payload: Mutable payload to transform

        Returns:
            Transformed payload (may be same object)
        """
        ...


class IdentityHook:
    """No-op hook that passes through without modification.

    Useful as a base class or for testing.
    """

    def on_before_step(self, ctx: StepContext, payload: StepPayload) -> StepPayload:
        return payload

    def on_after_step(self, ctx: StepContext, payload: StepPayload) -> StepPayload:
        return payload


def apply_hooks(
    hooks: list[FeatureHook],
    ctx: StepContext,
    payload: StepPayload,
    phase: str,
) -> StepPayload:
    """Apply a list of hooks in sequence.

    Args:
        hooks: List of hooks to apply
        ctx: Step context
        payload: Current payload
        phase: Either "before" or "after"

    Returns:
        Transformed payload after all hooks
    """
    if not hooks:
        return payload

    for hook in hooks:
        if phase == "before":
            payload = hook.on_before_step(ctx, payload)
        else:
            payload = hook.on_after_step(ctx, payload)
    return payload


# ---------------------------------------------------------------------------
# Example future hook implementations (not fully implemented yet)
# ---------------------------------------------------------------------------


class NoiseHook:
    """JAX noise injection hook.

    Adds Gaussian noise and/or perturbation to phi and state during forward pass.
    Mirrors PyTorch NoiseFeature behavior but with JAX-compatible RNG handling.

    Usage:
        from .noise_jax import build_noise_settings, precompute_perturbations

        settings = build_noise_settings({"phi": 0.01, "s": 0.005})
        offsets = precompute_perturbations(key, settings, batch_size, layer_dims)
        hook = NoiseHook(settings, rng_key, offsets)

    Note: Perturbation offsets must be precomputed before the forward pass
    and passed to this hook. This ensures the same offset is used across
    all timesteps within a forward pass.
    """

    def __init__(
        self,
        settings: Any,  # NoiseSettings from noise_jax
        rng_key: jax.Array,
        perturbation_offsets: Any = None,  # PerturbationOffsets from noise_jax
    ):
        self.settings = settings
        self.rng_key = rng_key
        self.perturbation_offsets = perturbation_offsets

    def on_before_step(self, ctx: StepContext, payload: StepPayload) -> StepPayload:
        """Apply noise to phi before layer dynamics."""
        if self.settings is None or self.settings.is_trivial():
            return payload

        # Import here to avoid circular deps
        from .noise_jax import apply_noise_step

        # Split key for this step
        self.rng_key, phi_key = jax.random.split(self.rng_key)

        # Get perturbation offset for phi
        phi_offset = None
        if self.perturbation_offsets is not None:
            phi_offset = self.perturbation_offsets.phi

        # Apply noise to phi
        if payload.phi is not None and self.settings.phi is not None:
            payload.phi = apply_noise_step(
                phi_key, payload.phi, self.settings.phi, phi_offset
            )

        # Apply noise to g if present in extras
        if "g" in payload.extras and self.settings.g is not None:
            self.rng_key, g_key = jax.random.split(self.rng_key)
            g_offset = None
            if self.perturbation_offsets and self.perturbation_offsets.layer_params:
                layer_offsets = self.perturbation_offsets.layer_params.get(
                    ctx.layer_idx, {}
                )
                g_offset = layer_offsets.get("g")
            payload.extras["g"] = apply_noise_step(
                g_key, payload.extras["g"], self.settings.g, g_offset
            )

        return payload

    def on_after_step(self, ctx: StepContext, payload: StepPayload) -> StepPayload:
        """Apply noise to state after layer dynamics."""
        if self.settings is None or self.settings.is_trivial():
            return payload

        from .noise_jax import apply_noise_step

        # Apply noise to state
        if payload.state is not None and self.settings.s is not None:
            self.rng_key, s_key = jax.random.split(self.rng_key)

            # Get perturbation offset for this layer's state
            s_offset = None
            if self.perturbation_offsets and self.perturbation_offsets.layer_params:
                layer_offsets = self.perturbation_offsets.layer_params.get(
                    ctx.layer_idx, {}
                )
                s_offset = layer_offsets.get("s")

            payload.state = apply_noise_step(
                s_key, payload.state, self.settings.s, s_offset
            )

        return payload


class PowerTracker:
    """Hook for tracking power consumption (placeholder for future implementation).

    Will record g values and compute power metrics during forward pass.
    Results are stored in payload.extras["power"].
    """

    def __init__(self):
        self.accumulated_power = 0.0

    def on_before_step(self, ctx: StepContext, payload: StepPayload) -> StepPayload:
        # Initialize power tracking for this step
        payload.extras["power_step"] = 0.0
        return payload

    def on_after_step(self, ctx: StepContext, payload: StepPayload) -> StepPayload:
        # Placeholder - actual implementation would compute power from g values
        # power = compute_power_from_g(payload.extras.get("g_values"))
        # payload.extras["power_step"] = power
        # self.accumulated_power += power
        return payload


__all__ = [
    "StepContext",
    "StepPayload",
    "FeatureHook",
    "IdentityHook",
    "apply_hooks",
    "NoiseHook",
    "PowerTracker",
]

