from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import jax
import jax.numpy as jnp

"""Learning-rate schedule utilities for the JAX backend.

This module mirrors the concepts used by the PyTorch Lightning callbacks
(`soen_toolkit.training.callbacks.schedulers`) but provides JAX/Optax-
friendly schedules that can be passed directly to optimizers.

Design goals:
- Keep parameter names similar to the Torch counterparts where practical
- Return pure schedule functions: f(step: jnp.ndarray) -> jnp.ndarray
- Provide a small registry and a convenience builder from the YAML callbacks

Notes:
- Complex adaptive schedulers that depend on validation metrics are not
  implemented here. When requested, we fall back to a constant schedule and
  log a warning via Python logging.
"""

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

SCHEDULER_REGISTRY_JAX: dict[str, Callable[..., Callable[[jnp.ndarray], jnp.ndarray]]] = {}


def register_scheduler(name: str) -> Callable[[Callable[..., Callable[[jnp.ndarray], jnp.ndarray]]], Callable[..., Callable[[jnp.ndarray], jnp.ndarray]]]:
    def _wrap(fn: Callable[..., Callable[[jnp.ndarray], jnp.ndarray]]) -> Callable[..., Callable[[jnp.ndarray], jnp.ndarray]]:
        SCHEDULER_REGISTRY_JAX[name] = fn
        return fn

    return _wrap


# -----------------------------------------------------------------------------
# Schedules
# -----------------------------------------------------------------------------


@register_scheduler("constant")
def make_constant_schedule(*, lr: float, **_: Any) -> Callable[[jnp.ndarray], jnp.ndarray]:
    lr_val = float(lr)

    def schedule(step: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001 - conforms to schedule signature
        return jnp.asarray(lr_val, dtype=jnp.float32)

    return schedule


@register_scheduler("linear")
def make_linear_schedule(
    *,
    max_lr: float = 1e-3,
    min_lr: float = 1e-6,
    total_steps: int,
    log_space: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    max_lr = float(max_lr)
    min_lr = float(min_lr)
    total = max(1, int(total_steps))

    def schedule(step: jnp.ndarray) -> jnp.ndarray:
        s = jnp.clip(step.astype(jnp.float32), 0.0, total)
        progress = s / float(total)
        if log_space:
            log_max = jnp.log(jnp.asarray(max_lr))
            log_min = jnp.log(jnp.asarray(min_lr))
            log_lr = log_max - progress * (log_max - log_min)
            lr = jnp.exp(log_lr)
        else:
            lr = max_lr - progress * (max_lr - min_lr)
        return jnp.asarray(jnp.maximum(lr, min_lr), dtype=jnp.float32)

    return schedule


@register_scheduler("cosine")
def make_cosine_schedule(
    *,
    max_lr: float = 1e-3,
    min_lr: float = 1e-6,
    warmup_epochs: int = 0,
    total_steps: int,
    steps_per_epoch: int,
    cycle_epochs: int | None = None,
    enable_restarts: bool | None = None,
    restart_decay: float | None = None,
    period_decay: float | None = None,
    amplitude_decay: float | None = None,
    soft_restart: bool | None = None,
    **_: Any,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Cosine annealing with (optional) linear warmup and optional restarts.

    Semantics:
    - Warmup: linear from min_lr -> max_lr for warmup_epochs
    - Post-warmup:
      - If cycle_epochs is None or enable_restarts is False: one cosine cycle across
        the remaining steps (half-cycle 0..π if hard restart semantics are desired, we
        use half-cycle by default for monotonic decay-like behavior; soft_restart toggles
        full 0..2π waveform which oscillates).
      - If cycle_epochs is provided and enable_restarts is True: repeat cycles of length
        L_k = L0 * (period_decay ** k) where L0 = cycle_epochs * steps_per_epoch. At each
        restart k, the effective max_lr decays by restart_decay^k and the oscillation
        amplitude decays by amplitude_decay^k. soft_restart selects full-cycle (2π) vs
        hard half-cycle (π).
    """
    max_lr = float(max_lr)
    min_lr = float(min_lr)
    total = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_epochs) * max(1, int(steps_per_epoch)))
    decay_steps = max(1, total - warmup_steps)

    # Normalize decay factors
    restart_decay = float(1.0 if restart_decay is None else restart_decay)
    period_decay = float(1.0 if period_decay is None else period_decay)
    amplitude_decay = float(1.0 if amplitude_decay is None else amplitude_decay)
    soft_restart = bool(soft_restart) if soft_restart is not None else False
    # Match PyTorch default: enable_restarts defaults to True
    enable_restarts = bool(enable_restarts) if enable_restarts is not None else True

    # Match PyTorch default: cycle_epochs defaults to 50 when enable_restarts=True
    # If cycle_epochs is not provided but restarts are enabled, use default of 50
    if cycle_epochs is None and enable_restarts:
        cycle_epochs = 50

    L0 = None
    if cycle_epochs is not None:
        assert isinstance(cycle_epochs, (int, float)), f"cycle_epochs must be numeric, got {type(cycle_epochs)}"
        ce = int(cycle_epochs)
        assert ce > 0, f"cycle_epochs must be positive, got {ce}"
        assert isinstance(steps_per_epoch, (int, float)), f"steps_per_epoch must be numeric, got {type(steps_per_epoch)}"
        assert int(steps_per_epoch) > 0, f"steps_per_epoch must be positive, got {steps_per_epoch}"
        L0 = max(1, ce * int(steps_per_epoch))

    if enable_restarts and L0 is None:
        logger.warning("[JAX Cosine] enable_restarts=True but L0 is None (cycle_epochs invalid). Restarts will be disabled.")

    def schedule(step: jnp.ndarray) -> jnp.ndarray:
        s = step.astype(jnp.float32)
        # Warmup phase (linear warmup for Torch parity)
        warm_prog = jnp.clip(s / jnp.maximum(1.0, jnp.asarray(warmup_steps, dtype=jnp.float32)), 0.0, 1.0)
        warm_lr = jnp.asarray(min_lr, dtype=jnp.float32) + (jnp.asarray(max_lr, dtype=jnp.float32) - jnp.asarray(min_lr, dtype=jnp.float32)) * warm_prog
        # Cosine phase
        after = jnp.maximum(0.0, s - jnp.asarray(warmup_steps, dtype=jnp.float32))
        # Use numeric L0 even when cycle_epochs is not provided to keep tracing valid
        l0_val = jnp.asarray(L0 if L0 is not None else decay_steps, dtype=jnp.float32)

        def single_cycle_lr(pos: jnp.ndarray, length_steps: jnp.ndarray, cur_max_lr: jnp.ndarray, amplitude: jnp.ndarray, next_amplitude: jnp.ndarray) -> jnp.ndarray:
            ls = jnp.asarray(length_steps, dtype=jnp.float32)
            prog = jnp.clip(pos / jnp.maximum(1.0, ls), 0.0, 1.0)
            if soft_restart:
                # full cycle 0..2π
                cosine_factor = (1.0 + jnp.cos(2.0 * jnp.pi * prog)) * 0.5

                # Amplitude blending for continuity with decay (matches PyTorch behavior)
                # Only blend if amplitude_decay < 1.0 (checked via next_amplitude != amplitude)
                # This ensures smooth transitions when amplitude is decaying
                should_blend = jnp.abs(next_amplitude - amplitude) > 1e-9
                effective_amplitude = jnp.where(
                    should_blend,
                    amplitude * (1.0 - prog) + next_amplitude * prog,
                    amplitude
                )
            else:
                # half cycle 0..π
                cosine_factor = (1.0 + jnp.cos(jnp.pi * prog)) * 0.5
                effective_amplitude = amplitude
            return jnp.asarray(jnp.asarray(min_lr, dtype=jnp.float32) + effective_amplitude * cosine_factor, dtype=jnp.float32)

        # If restarts disabled or no cycle_epochs provided: single cycle across remaining steps
        def single_phase() -> jnp.ndarray:
            amp = jnp.asarray(max_lr, dtype=jnp.float32) - jnp.asarray(min_lr, dtype=jnp.float32)
            # No next amplitude for single phase (use same as current)
            return single_cycle_lr(after, jnp.asarray(decay_steps, dtype=jnp.float32), jnp.asarray(max_lr, dtype=jnp.float32), amp, amp)

        # Restart phase with (optional) decays
        def restart_phase() -> jnp.ndarray:
            # Iterate cycles until remaining < current cycle length
            def cond_fun(carry):
                rem, k = carry
                len_k = jnp.maximum(1.0, l0_val * jnp.power(jnp.asarray(period_decay, dtype=jnp.float32), k))
                return rem >= len_k

            def body_fun(carry):
                rem, k = carry
                len_k = jnp.maximum(1.0, l0_val * jnp.power(jnp.asarray(period_decay, dtype=jnp.float32), k))
                return (rem - len_k, k + jnp.asarray(1.0, dtype=jnp.float32))

            init = (after, jnp.asarray(0.0, dtype=jnp.float32))
            rem_final, k_final = jax.lax.while_loop(cond_fun, body_fun, init)
            # Current cycle params at index k_final
            len_k = jnp.maximum(1.0, l0_val * jnp.power(jnp.asarray(period_decay, dtype=jnp.float32), k_final))
            cur_max = jnp.asarray(max_lr, dtype=jnp.float32) * jnp.power(jnp.asarray(restart_decay, dtype=jnp.float32), k_final)
            amplitude = (cur_max - jnp.asarray(min_lr, dtype=jnp.float32)) * jnp.power(jnp.asarray(amplitude_decay, dtype=jnp.float32), k_final)

            # Calculate next amplitude for blending (matches PyTorch behavior)
            # Only blend when soft_restart=True, amplitude_decay < 1.0, and enable_restarts=True
            # Compute next amplitude always, but it will equal current if not blending
            next_cycle_num = k_final + jnp.asarray(1.0, dtype=jnp.float32)
            next_max_lr_decayed = jnp.asarray(max_lr, dtype=jnp.float32) * jnp.power(jnp.asarray(restart_decay, dtype=jnp.float32), next_cycle_num)
            next_amplitude_val = (next_max_lr_decayed - jnp.asarray(min_lr, dtype=jnp.float32)) * jnp.power(jnp.asarray(amplitude_decay, dtype=jnp.float32), next_cycle_num)
            next_amplitude_val = jnp.maximum(0.0, next_amplitude_val)

            # If not blending (soft_restart=False or amplitude_decay >= 1.0 or not enable_restarts), use same as current
            # Note: enable_restarts is implicitly True here since restart_phase is only called when restarts are enabled
            if not soft_restart or amplitude_decay >= 1.0:
                next_amplitude_val = amplitude

            return single_cycle_lr(rem_final, len_k, cur_max, amplitude, next_amplitude_val)

        # Choose between restart and single-phase using jax.lax.cond
        def _do_restart(_):
            return restart_phase()

        def _do_single(_):
            return single_phase()

        # Compute condition: restarts enabled AND cycle_epochs provided (L0 computed)
        restarts_on = jnp.asarray(1 if enable_restarts else 0, dtype=jnp.int32)
        has_cycles = jnp.asarray(1 if (L0 is not None) else 0, dtype=jnp.int32)
        cond_flag = (restarts_on & has_cycles).astype(jnp.bool_)
        cos_lr = jax.lax.cond(cond_flag, _do_restart, _do_single, operand=None)

        lr = jnp.where(s < jnp.asarray(warmup_steps, dtype=jnp.float32), warm_lr, cos_lr)
        return jnp.asarray(lr, dtype=jnp.float32)

    return schedule


@register_scheduler("rex")
def make_rex_schedule(
    *,
    max_lr: float,
    min_lr: float = 0.0,
    warmup_epochs: int = 0,
    warmup_start_lr: float = 1e-6,
    total_steps: int,
    steps_per_epoch: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    # Accept and ignore any extra kwargs for compatibility
    def _accept_extras(**__: Any) -> None:
        return None

    """REX (Rational EXponential-like) schedule approximation.

    Mirrors the Torch callback's formula:
    - Optional warmup from warmup_start_lr -> max_lr
    - Then decay from max_lr to min_lr with
        lr = min_lr + (max_lr - min_lr) * (term / (0.5 + 0.5*term))
      where term = 1 - t, t in [0,1] over remaining steps.
    """
    max_lr_f = jnp.asarray(max_lr, dtype=jnp.float32)
    min_lr_f = jnp.asarray(min_lr, dtype=jnp.float32)
    warm_start_f = jnp.asarray(warmup_start_lr, dtype=jnp.float32)
    total = int(max(1, int(total_steps)))
    warm_steps = int(max(0, int(warmup_epochs) * max(1, int(steps_per_epoch))))
    remain = int(max(1, total - warm_steps))

    def schedule(step: jnp.ndarray) -> jnp.ndarray:
        s = step.astype(jnp.float32)
        # Warmup (cosine)
        warm_progress = jnp.clip(s / jnp.maximum(1.0, jnp.asarray(warm_steps, dtype=jnp.float32)), 0.0, 1.0)
        warm_cos = 0.5 * (1.0 - jnp.cos(jnp.pi * warm_progress))
        warm_lr = warm_start_f + (max_lr_f - warm_start_f) * warm_cos
        # Decay
        t_norm = jnp.clip((s - jnp.asarray(warm_steps, dtype=jnp.float32)) / jnp.asarray(remain, dtype=jnp.float32), 0.0, 1.0)
        term = 1.0 - t_norm
        rex_factor = term / (0.5 + 0.5 * term)
        dec_lr = min_lr_f + (max_lr_f - min_lr_f) * rex_factor
        lr = jnp.where(s <= jnp.asarray(warm_steps, dtype=jnp.float32), warm_lr, dec_lr)
        return jnp.asarray(lr, dtype=jnp.float32)

    return schedule


# Greedy/Adaptive are metric-driven and require validation feedback. We provide
# placeholders to avoid configuration errors and fall back to constant LR.


@register_scheduler("greedy")
def make_greedy_placeholder(*, lr: float, **_: Any) -> Callable[[jnp.ndarray], jnp.ndarray]:
    logger.warning("[JAX] Greedy scheduler not supported in JAX path; using constant LR.")
    return make_constant_schedule(lr=lr)


@register_scheduler("adaptive")
def make_adaptive_placeholder(*, lr: float, **_: Any) -> Callable[[jnp.ndarray], jnp.ndarray]:
    logger.warning("[JAX] Adaptive scheduler not supported in JAX path; using constant LR.")
    return make_constant_schedule(lr=lr)


# -----------------------------------------------------------------------------
# Convenience builder from ExperimentConfig.callbacks.lr_scheduler
# -----------------------------------------------------------------------------


def make_schedule_from_callbacks_config(config: Any, *, steps_per_epoch: int, max_epochs: int) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
    """Inspect the experiment config and return a JAX schedule function or None.

    Expects a block like:
        callbacks:
          lr_scheduler:
            type: cosine|linear|rex|constant|...
            ...params...
    """
    assert steps_per_epoch > 0, f"steps_per_epoch must be positive, got {steps_per_epoch}"
    assert max_epochs > 0, f"max_epochs must be positive, got {max_epochs}"

    callbacks = getattr(config, "callbacks", {}) or {}
    assert isinstance(callbacks, dict), f"callbacks must be a dict, got {type(callbacks)}"

    sched_cfg = callbacks.get("lr_scheduler", {})
    if not sched_cfg:
        return None

    assert isinstance(sched_cfg, dict), f"lr_scheduler config must be a dict, got {type(sched_cfg)}"

    sched_type = sched_cfg.get("type", "constant")
    assert isinstance(sched_type, str), f"scheduler type must be a string, got {type(sched_type)}"
    sched_type = sched_type.lower()

    # Get optimizer LR for defaults (used by multiple schedulers)
    training = getattr(config, "training", None)
    assert training is not None, "config.training is required"
    opt_config = getattr(training, "optimizer", None)
    assert opt_config is not None, "config.training.optimizer is required"
    base_lr = float(getattr(opt_config, "lr", 1e-3))

    maker = SCHEDULER_REGISTRY_JAX.get(sched_type)
    if maker is None:
        logger.warning("[JAX] Unknown scheduler type '%s'; using constant.", sched_type)
        # Fallback to constant from optimizer LR
        return make_constant_schedule(lr=base_lr)

    # Common parameters
    params: dict[str, Any] = dict(sched_cfg)
    params.pop("type", None)
    # Inject derived values
    total_steps = int(max(1, steps_per_epoch) * max(1, max_epochs))
    params.setdefault("total_steps", total_steps)
    params.setdefault("steps_per_epoch", int(max(1, steps_per_epoch)))

    # Provide defaults based on optimizer LR if not specified
    if sched_type == "constant":
        params.setdefault("lr", base_lr)
    elif sched_type in {"linear", "cosine", "rex", "greedy", "adaptive"}:
        params.setdefault("max_lr", base_lr)
        params.setdefault("min_lr", 1e-6)

        # For cosine scheduler, ensure defaults match PyTorch
        if sched_type == "cosine":
            # Match PyTorch defaults: enable_restarts=True, cycle_epochs=50
            if "enable_restarts" not in params:
                params["enable_restarts"] = True
            if params.get("enable_restarts", False) and "cycle_epochs" not in params:
                params["cycle_epochs"] = 50

    return maker(**params)


__all__ = [
    "SCHEDULER_REGISTRY_JAX",
    "register_scheduler",
    "make_schedule_from_callbacks_config",
]
