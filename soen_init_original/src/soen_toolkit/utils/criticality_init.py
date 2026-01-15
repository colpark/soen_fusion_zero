"""Criticality-based weight initialization using backpropagation.

This module tunes network weights towards critical dynamics by optimizing the
branching ratio (BR) towards a target value (default 1.0).

The branching ratio indicates how activity propagates through the network:
    - BR = 1.0: Critical point ("edge of chaos") - optimal for computation
    - BR < 1.0: Subcritical - activity decays over time
    - BR > 1.0: Supercritical - activity explodes

Algorithm Overview:
    1. Run forward pass WITH gradient tracking (essential for proper optimization)
    2. Compute branching ratio from state trajectories
    3. Compute loss = (BR - target)^2
    4. Backpropagate gradients through the full temporal dynamics
    5. Update weights using Adam with gradient clipping
    6. Track best weights and restore if divergence occurs

Key Design Decisions:
    - Gradients MUST flow through the forward pass to properly optimize
    - Learning rate scheduling prevents oscillation near the target
    - Divergence detection reverts bad updates
    - Best weights are tracked and restored at the end
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from soen_toolkit.core.soen_model_core import SOENModelCore

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CriticalityInitConfig:
    """Configuration for criticality-based weight initialization.

    Attributes:
        target: Target branching ratio. 1.0 = critical point. (kept for
            backward compatibility; prefer br_target)
        enable_br: Whether to include the branching-ratio objective.
        br_target: Target branching ratio (edge-of-chaos ~1.0).
        br_weight: Loss weight for branching ratio objective.
        enable_lyap: Whether to include the Lyapunov-like objective.
        lyap_target: Target local Lyapunov exponent (≈0 for edge-of-chaos).
        lyap_weight: Loss weight for Lyapunov objective.
        lyap_eps: Perturbation magnitude for twin-trajectory Lyapunov estimate.
        lyap_time_horizon: Optional cap on the number of time steps used for
            Lyapunov averaging (None uses full history).
        range_penalty_weight: Optional weight penalty for activity leaving the
            safe range.
        range_clip: Threshold for activity-range penalty (abs mean activity).
        log_lyap: Whether to log Lyapunov estimates per iteration.
        max_nan_tol: Abort if the number of NaN/inf detections exceeds this.
        num_iterations: Maximum optimization iterations.
        learning_rate: Initial learning rate for Adam optimizer.
        batch_size: Batch size for forward passes.
        num_batches: Number of batches to average gradients over per iteration.
        tolerance: Stop early if |BR - target| < tolerance.
        patience: Stop if no improvement for this many iterations.
        weight_decay: L2 regularization coefficient.
        grad_clip: Gradient clipping magnitude. None = no clipping.
    lr_decay_factor: Factor to reduce LR when stuck. 0.5 = halve LR.
    lr_decay_patience: Reduce LR after this many iterations without improvement.
    lr_warmup_iters: Do not decay LR before this many iterations.
        min_improvement: Minimum error drop to count as an improvement.
        min_lr: Minimum learning rate (stop decaying below this).
        disable_param_constraints: Skip constraint/mask enforcement during criticality
            init to avoid in-place mutations on parameters needed for autograd.
        detect_anomaly: Enable torch autograd anomaly detection for debugging.
        verbose: Print progress messages.
        log_fn: Custom logging function (receives string messages).
        stop_check: Callback returning True to cancel optimization.
        log_every: Log progress every N iterations.
    """

    target: float = 1.0  # legacy alias for br_target
    enable_br: bool = True
    br_target: float = 1.0
    br_weight: float = 1.0

    enable_lyap: bool = False
    lyap_target: float = 0.0
    lyap_weight: float = 0.0
    lyap_eps: float = 1e-4
    lyap_time_horizon: int | None = None
    lyap_min_delta0: float = 1e-9
    lyap_max_retries: int = 5
    lyap_noise_growth: float = 10.0

    range_penalty_weight: float = 0.0
    range_clip: float | None = None

    log_lyap: bool = False
    max_nan_tol: int = 0

    num_iterations: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    num_batches: int = 5
    tolerance: float = 0.01
    patience: int = 10
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0

    # Learning rate scheduling
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 5
    min_lr: float = 1e-6
    lr_warmup_iters: int = 10
    min_improvement: float = 1e-4
    disable_param_constraints: bool = True
    detect_anomaly: bool = True

    # Logging
    verbose: bool = True
    log_fn: Callable[[str], None] | None = None
    stop_check: Callable[[], bool] | None = None
    log_every: int = 10


# =============================================================================
# Results
# =============================================================================


@dataclass
class CriticalityInitResult:
    """Results from criticality initialization.

    Attributes:
        iteration_stats: Per-iteration statistics (BR, Lyap, loss, error, lr).
        final_branching_ratio: BR after optimization (from best weights).
        final_lyapunov: Lyapunov estimate after optimization (from best weights).
        converged: Whether |BR - target| < tolerance was achieved (or both if
            both objectives are enabled).
        best_branching_ratio: Best BR achieved during optimization.
        best_lyapunov: Best Lyapunov estimate achieved during optimization.
        best_iteration: Iteration index where best metrics were found.
        stopped_early: Whether optimization stopped before max iterations.
        stop_reason: Human-readable reason for stopping.
    """

    iteration_stats: list[dict[str, Any]] = field(default_factory=list)
    final_branching_ratio: float = 0.0
    final_lyapunov: float | None = None
    converged: bool = False
    best_branching_ratio: float = 0.0
    best_lyapunov: float | None = None
    best_iteration: int = 0
    stopped_early: bool = False
    stop_reason: str = ""


# =============================================================================
# Helpers
# =============================================================================


def _default_log_fn(msg: str) -> None:
    """Default logging function - prints to stdout."""
    import sys
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _collect_learnable_connection_params(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    """Collect all learnable connection weight parameters from the model.

    Returns:
        Tuple of (list of parameters, list of parameter names)
    """
    learnable_params = []
    param_names = []

    for name, param in model.named_parameters():
        # Include connection weights (external and internal)
        is_connection = "connections" in name or "J_" in name or "internal" in name
        if is_connection and param.requires_grad:
            learnable_params.append(param)
            param_names.append(name)

    return learnable_params, param_names


def _save_weights(model: torch.nn.Module, param_names: list[str]) -> dict[str, torch.Tensor]:
    """Save a copy of the current connection weights.

    Args:
        model: The model to save weights from
        param_names: Names of parameters to save

    Returns:
        Dictionary mapping parameter names to cloned tensors
    """
    saved = {}
    for name, param in model.named_parameters():
        if name in param_names:
            saved[name] = param.data.clone()
    return saved


def _restore_weights(model: torch.nn.Module, saved_weights: dict[str, torch.Tensor]) -> None:
    """Restore connection weights from a saved copy.

    Args:
        model: The model to restore weights to
        saved_weights: Dictionary from _save_weights()
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in saved_weights:
                param.copy_(saved_weights[name])


def _extract_state_histories(output, model) -> list[torch.Tensor] | None:
    """Extract state histories from model output.

    SOENModelCore returns (final_output, state_histories) where state_histories
    is a list of tensors [B, T+1, D] for each layer.

    Args:
        output: Raw output from model forward pass
        model: The model (used as fallback to get cached states)

    Returns:
        List of state tensors or None if not available
    """
    # Case 1: Tuple output with state histories
    if isinstance(output, tuple) and len(output) >= 2:
        if isinstance(output[1], list):
            return output[1]

    # Case 2: Fallback to cached states on model
    cached = getattr(model, "latest_all_states", None)
    if cached is not None:
        return cached

    return None


# =============================================================================
# Lyapunov Estimation Helpers
# =============================================================================


def _concat_states(
    s_histories: list[torch.Tensor] | None,
    time_horizon: int | None = None,
) -> torch.Tensor | None:
    """Concatenate layer states along feature dim with optional time cap."""
    if not s_histories:
        return None

    slices: list[torch.Tensor] = []
    for s in s_histories:
        if s.ndim != 3:
            continue
        steps = s.size(1)
        if time_horizon is not None:
            steps = min(steps, time_horizon)
        slices.append(s[:, :steps, :])

    if not slices:
        return None

    return torch.cat(slices, dim=2)


def _compute_local_lyap_twin_trajectory(
    model: torch.nn.Module,
    batch: torch.Tensor,
    base_states: list[torch.Tensor],
    *,
    eps: float = 1e-4,
    time_horizon: int | None = None,
    min_delta0: float = 1e-9,
    max_retries: int = 5,
    noise_growth: float = 10.0,
) -> torch.Tensor:
    """Twin-trajectory Lyapunov estimate using a small input perturbation.

    This stays differentiable so gradients flow back to the weights.

    Uses TOTAL log growth (start to end) rather than per-step growth ratios.
    This gives much clearer gradients because consecutive-step ratios tend
    to have correlated gradients that cancel out when averaged.

    Args:
        model: The model under optimization.
        batch: Input batch [B, T, D].
        base_states: State histories from the unperturbed pass.
        eps: Perturbation magnitude (scaled by input mean abs).
        time_horizon: Optional cap on number of steps used for averaging.
        min_delta0: Minimum delta threshold to avoid division issues.
        max_retries: Number of retry attempts with larger perturbation.
        noise_growth: Multiplier for perturbation on each retry.

    Returns:
        Scalar tensor (mean log growth per step, normalized by trajectory length).

    Raises:
        RuntimeError if states cannot be retrieved or perturbation collapses.
    """
    def _run_with_noise(noise_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(batch) * noise_scale
        perturbed_batch = batch + noise
        output_p = model(perturbed_batch)
        states = _extract_state_histories(output_p, model)
        if states is None:
            raise RuntimeError("Lyapunov estimation failed: no perturbed state histories")
        return perturbed_batch, states

    # Scale perturbation relative to signal magnitude to avoid vanishing delta
    base_scale = batch.detach().abs().mean() + 1e-12

    # Try with increasing perturbation if delta0 collapses
    noise_scale = max(eps * base_scale, min_delta0)
    delta0_val = 0.0
    pert_states = None

    for _ in range(max_retries):
        _, pert_states = _run_with_noise(noise_scale)

        base_concat = _concat_states(base_states, time_horizon)
        pert_concat = _concat_states(pert_states, time_horizon)
        if base_concat is None or pert_concat is None:
            raise RuntimeError("Lyapunov estimation failed: missing concatenated states")

        steps = min(base_concat.size(1), pert_concat.size(1))
        if steps < 2:
            raise RuntimeError("Lyapunov estimation failed: insufficient time steps")

        base_c = base_concat[:, :steps, :]
        pert_c = pert_concat[:, :steps, :]
        delta = torch.norm(pert_c - base_c, p=2, dim=2)  # [B, T]
        delta0_val = delta[:, 0].mean().item()

        if delta0_val >= min_delta0:
            break

        # Increase perturbation and retry
        noise_scale *= noise_growth

    base_concat = _concat_states(base_states, time_horizon)
    pert_concat = _concat_states(pert_states, time_horizon)
    if base_concat is None or pert_concat is None:
        raise RuntimeError("Lyapunov estimation failed: missing concatenated states")

    # Align time dimension
    steps = min(base_concat.size(1), pert_concat.size(1))
    if steps < 2:
        raise RuntimeError("Lyapunov estimation failed: insufficient time steps")

    base_concat = base_concat[:, :steps, :]
    pert_concat = pert_concat[:, :steps, :]

    # Delta trajectory: distance between perturbed and base at each timestep
    delta = torch.norm(pert_concat - base_concat, p=2, dim=2)  # [B, T]

    # Use TOTAL growth (start to end) instead of per-step ratios.
    # This gives much clearer gradients because per-step ratios have
    # correlated gradients that cancel out when summed.
    delta_start = delta[:, 0].clamp_min(min_delta0)
    delta_end = delta[:, -1].clamp_min(min_delta0)

    # Total log growth normalized by number of steps to get per-step Lyapunov
    total_log_growth = torch.log(delta_end / delta_start)
    lyap_per_step = total_log_growth / max(steps - 1, 1)

    return lyap_per_step.mean()


def _activity_range_penalty(
    s_histories: list[torch.Tensor],
    clip: float | None,
) -> torch.Tensor:
    """Penalize activity leaving a safe range (mean absolute activity)."""
    device = s_histories[0].device if s_histories else torch.device("cpu")
    if clip is None or clip <= 0:
        return torch.tensor(0.0, device=device)

    concat = _concat_states(s_histories)
    if concat is None:
        return torch.tensor(0.0, device=device)

    mean_abs = concat.abs().mean()
    return torch.relu(mean_abs - clip)


# =============================================================================
# Main Algorithm
# =============================================================================


def run_criticality_init(
    model: SOENModelCore,
    data_batches: list[torch.Tensor],
    config: CriticalityInitConfig | None = None,
) -> CriticalityInitResult:
    """Run criticality-based weight initialization.

    Uses gradient descent to tune connection weights towards a target branching
    ratio (and optionally a Lyapunov-like stability target). The branching ratio
    measures how activity propagates through layers.

    IMPORTANT: This algorithm requires gradients to flow through the forward pass.
    The optimization works by computing a differentiable criticality loss and
    backpropagating through the entire temporal dynamics to update weights.

    Args:
        model: The SOEN model to initialize (modified in-place)
        data_batches: List of input tensors [batch, seq_len, input_dim]
        config: Configuration options (uses defaults if None)

    Returns:
        CriticalityInitResult with optimization statistics and success indicators

    Example:
        >>> config = CriticalityInitConfig(target=1.0, num_iterations=100)
        >>> result = run_criticality_init(model, data_batches, config)
        >>> print(f"Final BR: {result.final_branching_ratio:.4f}")
    """
    from soen_toolkit.utils.metrics import compute_branching_ratio_differentiable

    config = config or CriticalityInitConfig()
    if not config.enable_br and not config.enable_lyap:
        raise ValueError("At least one objective must be enabled (BR or Lyapunov)")

    # Backward compatibility for legacy 'target'
    br_target = config.br_target if config.br_target is not None else config.target
    lyap_target = config.lyap_target

    # Optionally disable param constraints during init to avoid in-place mutations
    original_enforce = None
    try:
        if config.disable_param_constraints and hasattr(model, "enforce_param_constraints"):
            original_enforce = model.enforce_param_constraints

            def _no_enforce():
                return None

            model.enforce_param_constraints = _no_enforce
    except Exception:
        # Fail fast if we cannot safely disable constraints
        raise

    result = CriticalityInitResult()

    # --- Setup Logging ---
    log_fn = config.log_fn if config.log_fn else _default_log_fn

    def log(msg: str) -> None:
        if config.verbose:
            log_fn(msg)

    # --- Collect Learnable Parameters ---
    learnable_params, param_names = _collect_learnable_connection_params(model)

    if not learnable_params:
        log("No learnable connection weights found - nothing to optimize")
        result.stop_reason = "No learnable parameters"
        return result

    if len(learnable_params) <= 20:
        log(f"Optimizing the following {len(learnable_params)} parameters:")
        for name in param_names:
            log(f"  - {name}")
    else:
        log(f"Optimizing {len(learnable_params)} parameters (names hidden, count > 20)")

    # --- Device Setup ---
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    # --- Log Configuration ---
    log("=" * 60)
    log("CRITICALITY INITIALIZATION")
    log("=" * 60)
    log(f"  BR enabled: {config.enable_br} (target={br_target}, weight={config.br_weight})")
    log(f"  Lyap enabled: {config.enable_lyap} (target={lyap_target}, weight={config.lyap_weight}, eps={config.lyap_eps})")
    if config.range_penalty_weight > 0 and config.range_clip:
        log(f"  Activity range penalty: weight={config.range_penalty_weight}, clip={config.range_clip}")
    log(f"  Max iterations: {config.num_iterations}")
    log(f"  Initial learning rate: {config.learning_rate}")
    log(f"  Tolerance: {config.tolerance}")
    log(f"  Patience: {config.patience} iterations")
    log(f"  Learnable parameters: {len(learnable_params)}")
    log(f"  Batches per iteration: {min(config.num_batches, len(data_batches))}")
    log(f"  Gradient clipping: {config.grad_clip or 'disabled'}")
    log("")

    # --- Setup Optimizer with Scheduling ---
    optimizer = torch.optim.Adam(
        learnable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Track current learning rate
    current_lr = config.learning_rate

    # --- Tracking Variables ---
    best_error = float("inf")
    best_br = float("nan") if not config.enable_br else 0.0
    best_lyap: float | None = float("nan") if not config.enable_lyap else None
    best_iter = 0
    best_weights: dict[str, torch.Tensor] = {}

    no_improvement_count = 0
    lr_no_improvement_count = 0
    consecutive_divergence = 0
    prev_error = float("inf")
    nan_count = 0

    # Store original training mode
    was_training = model.training

    # ==========================================================================
    # MAIN OPTIMIZATION LOOP
    # ==========================================================================

    for iteration in range(config.num_iterations):
        # --- Check for User Cancellation ---
        if config.stop_check is not None and config.stop_check():
            log("\nOptimization cancelled by user")
            result.stopped_early = True
            result.stop_reason = "User cancelled"
            break

        # --- Zero Gradients ---
        optimizer.zero_grad()

        # --- Accumulate Gradients Over Batches ---
        total_br = 0.0
        total_lyap = 0.0
        total_loss_val = 0.0
        batches_used = 0
        br_batches = 0
        lyap_batches = 0

        for batch_idx, batch in enumerate(data_batches):
            if batch_idx >= config.num_batches:
                break

            batch = batch.to(device)

            # ------------------------------------------------------------------
            # FORWARD PASS WITH GRADIENT TRACKING
            # ------------------------------------------------------------------
            # The model MUST be run with gradients enabled. This allows the
            # optimizer to learn how weight changes affect the branching ratio
            # through the full temporal dynamics of the network.
            # ------------------------------------------------------------------

            model.train()  # Consistent behavior (though we're not training in the usual sense)
            output = model(batch)

            # Extract state histories
            s_histories = _extract_state_histories(output, model)

            if s_histories is None:
                log(f"  Warning: No state histories from batch {batch_idx}")
                continue

            # Start with a grad-enabled zero to safely accumulate terms
            loss = torch.zeros((), device=batch.device, requires_grad=True)

            # ------------------------------------------------------------------
            # BRANCHING RATIO (DIFFERENTIABLE)
            # ------------------------------------------------------------------
            br = None
            if config.enable_br:
                br = compute_branching_ratio_differentiable(model, s_histories)
                loss = loss + config.br_weight * (br - br_target) ** 2
                total_br += br.detach().item()
                br_batches += 1

            # ------------------------------------------------------------------
            # LYAPUNOV (DIFFERENTIABLE TWIN TRAJECTORY)
            # ------------------------------------------------------------------
            lyap_val = None
            if config.enable_lyap and config.lyap_weight > 0:
                try:
                    lyap_val = _compute_local_lyap_twin_trajectory(
                        model,
                        batch,
                        s_histories,
                        eps=config.lyap_eps,
                        time_horizon=config.lyap_time_horizon,
                        min_delta0=config.lyap_min_delta0,
                        max_retries=config.lyap_max_retries,
                        noise_growth=config.lyap_noise_growth,
                    )
                except Exception as exc:
                    log(f"  Lyapunov estimation failed at batch {batch_idx}: {exc}")
                    result.stopped_early = True
                    result.stop_reason = "Lyapunov estimation failed"
                    break

                loss = loss + config.lyap_weight * (lyap_val - lyap_target) ** 2
                total_lyap += lyap_val.detach().item()
                lyap_batches += 1

            # ------------------------------------------------------------------
            # OPTIONAL RANGE PENALTY
            # ------------------------------------------------------------------
            if config.range_penalty_weight > 0 and config.range_clip is not None:
                penalty = _activity_range_penalty(s_histories, config.range_clip)
                loss = loss + config.range_penalty_weight * penalty

            if not loss.requires_grad:
                log(f"  Skipping batch {batch_idx}: no objective produced gradients")
                continue

            # Debug: verify gradient attachment
            if config.verbose and iteration == 0 and batch_idx == 0:
                log(f"  Debug [Iter 0 Batch 0]: Loss requires_grad={loss.requires_grad}")
                if lyap_val is not None:
                    log(f"  Debug [Iter 0 Batch 0]: Lyap requires_grad={lyap_val.requires_grad}")
                if br is not None:
                    log(f"  Debug [Iter 0 Batch 0]: BR requires_grad={br.requires_grad}")

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                log(f"  NaN/Inf loss encountered at batch {batch_idx} (count={nan_count})")
                if config.max_nan_tol is not None and nan_count > config.max_nan_tol:
                    result.stopped_early = True
                    result.stop_reason = "NaN/Inf encountered"
                    break
                continue

            # Backward pass - accumulates gradients
            scaled_loss = loss / min(config.num_batches, len(data_batches))
            if config.detect_anomaly:
                with torch.autograd.set_detect_anomaly(True):
                    scaled_loss.backward()
            else:
                scaled_loss.backward()

            # Debug: check gradients after first backward
            if config.verbose and iteration == 0 and batch_idx == 0:
                active_grads = sum(1 for p in learnable_params if p.grad is not None)
                max_grad = max((p.grad.abs().max().item() for p in learnable_params if p.grad is not None), default=0.0)
                log(f"  Debug [Iter 0 Batch 0]: {active_grads}/{len(learnable_params)} params have grad. Max grad magnitude: {max_grad:.2e}")

            # Track statistics (detached to avoid memory leaks)
            total_loss_val += loss.detach().item()
            batches_used += 1

        if result.stopped_early and result.stop_reason:
            break

        # --- Handle Case Where No Batches Processed ---
        if batches_used == 0:
            log("Error: No batches could be processed")
            result.stopped_early = True
            result.stop_reason = "No batches processed"
            break

        # --- Compute Averages ---
        avg_br = total_br / br_batches if br_batches > 0 else None
        avg_lyap = total_lyap / lyap_batches if lyap_batches > 0 else None
        avg_loss = total_loss_val / batches_used

        errors: list[float] = []
        if config.enable_br:
            if avg_br is None:
                result.stopped_early = True
                result.stop_reason = "Branching ratio missing"
                log("Error: Branching ratio missing for enabled objective")
                break
            errors.append(abs(avg_br - br_target))

        if config.enable_lyap:
            if avg_lyap is None:
                result.stopped_early = True
                result.stop_reason = "Lyapunov missing"
                log("Error: Lyapunov estimate missing for enabled objective")
                break
            errors.append(abs(avg_lyap - lyap_target))

        current_error = max(errors) if errors else float("inf")

        # --- Gradient Clipping ---
        if config.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(learnable_params, config.grad_clip)
        else:
            grad_norm = sum(p.grad.norm().item() ** 2 for p in learnable_params if p.grad is not None) ** 0.5

        if iteration == 0 and (isinstance(grad_norm, torch.Tensor) and grad_norm.item() < 1e-9 or float(grad_norm) < 1e-9):
            log(f"  Warning: Gradient norm is extremely small ({grad_norm:.2e}). Optimization may be stalled.")

        # --- Divergence Detection ---
        # If error increased significantly, we may be oscillating or diverging
        if current_error > prev_error * 1.5 and iteration > 0:
            consecutive_divergence += 1
            if consecutive_divergence >= 3:
                log(f"  Divergence detected at iteration {iteration + 1} - reducing LR")
                current_lr = max(current_lr * config.lr_decay_factor, config.min_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
                consecutive_divergence = 0
        else:
            consecutive_divergence = 0

        prev_error = current_error

        # --- Update Weights ---
        optimizer.step()

        # --- Track Best Result ---
        improvement = best_error - current_error
        if improvement > config.min_improvement:
            best_error = current_error
            if config.enable_br and avg_br is not None:
                best_br = avg_br
            if config.enable_lyap and avg_lyap is not None:
                best_lyap = avg_lyap
            best_iter = iteration
            best_weights = _save_weights(model, param_names)
            no_improvement_count = 0
            lr_no_improvement_count = 0
        else:
            no_improvement_count += 1
            lr_no_improvement_count += 1

        # --- Learning Rate Decay ---
        if iteration + 1 >= config.lr_warmup_iters and lr_no_improvement_count >= config.lr_decay_patience:
            new_lr = max(current_lr * config.lr_decay_factor, config.min_lr)
            if new_lr < current_lr:
                log(f"  Reducing learning rate: {current_lr:.2e} -> {new_lr:.2e}")
                current_lr = new_lr
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
            lr_no_improvement_count = 0

        # --- Record Iteration Statistics ---
        iter_stats = {
            "iteration": iteration,
            "branching_ratio": avg_br,
            "lyapunov": avg_lyap,
            "loss": avg_loss,
            "error": current_error,
            "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "learning_rate": current_lr,
        }
        result.iteration_stats.append(iter_stats)

        # --- Log Progress ---
        if iteration % config.log_every == 0 or iteration == config.num_iterations - 1:
            direction = "↑" if (avg_br is not None and avg_br > br_target) else "↓" if (avg_br is not None and avg_br < br_target) else "="
            br_display = avg_br if avg_br is not None else float("nan")
            lyap_msg = ""
            if config.enable_lyap:
                lyap_display = avg_lyap if avg_lyap is not None else float("nan")
                lyap_msg = f", Lyap={lyap_display:.4f} (target={lyap_target})"
            log(
                f"Iteration {iteration + 1:4d}/{config.num_iterations}: "
                f"BR={br_display:.4f} {direction} (target={br_target}){lyap_msg}, "
                f"error={current_error:.4f}, loss={avg_loss:.6f}, "
                f"lr={current_lr:.2e}"
            )

        # --- Early Stopping: Converged ---
        br_ok = (not config.enable_br) or (avg_br is not None and abs(avg_br - br_target) < config.tolerance)
        lyap_ok = (not config.enable_lyap) or (avg_lyap is not None and abs(avg_lyap - lyap_target) < config.tolerance)
        if br_ok and lyap_ok:
            log(f"\nConverged at iteration {iteration + 1}!")
            if avg_br is not None:
                log(f"  BR = {avg_br:.4f} is within tolerance of target {br_target}")
            if config.enable_lyap and avg_lyap is not None:
                log(f"  Lyap = {avg_lyap:.4f} is within tolerance of target {lyap_target}")
            result.converged = True
            result.stopped_early = True
            result.stop_reason = "Converged to target(s)"
            break

        # --- Early Stopping: No Improvement ---
        if no_improvement_count >= config.patience:
            log(f"\nNo improvement for {config.patience} iterations - stopping early")
            result.stopped_early = True
            result.stop_reason = f"No improvement for {config.patience} iterations"
            break

        # --- Early Stopping: Learning Rate Too Small ---
        if current_lr <= config.min_lr and lr_no_improvement_count >= config.lr_decay_patience:
            log("\nLearning rate at minimum and no improvement - stopping")
            result.stopped_early = True
            result.stop_reason = "Learning rate bottomed out"
            break

    # ==========================================================================
    # FINALIZATION
    # ==========================================================================

    # Restore best weights
    if best_weights:
        log(f"\nRestoring best weights from iteration {best_iter + 1}")
        _restore_weights(model, best_weights)

    # Restore training mode
    model.train(was_training)

    # Restore param constraints hook if we disabled it
    if config.disable_param_constraints and original_enforce is not None:
        try:
            model.enforce_param_constraints = original_enforce
        except Exception:
            pass

    # Populate result
    result.final_branching_ratio = best_br
    result.final_lyapunov = best_lyap
    result.best_branching_ratio = best_br
    result.best_lyapunov = best_lyap
    result.best_iteration = best_iter

    # --- Final Summary ---
    log("")
    log("=" * 60)
    if result.converged:
        log("OPTIMIZATION CONVERGED")
    else:
        log("OPTIMIZATION COMPLETED")
    log("=" * 60)
    if config.enable_br and result.final_branching_ratio == result.final_branching_ratio:
        log(f"  Final BR: {result.final_branching_ratio:.4f} (target={br_target})")
    if config.enable_lyap and result.final_lyapunov is not None and result.final_lyapunov == result.final_lyapunov:
        log(f"  Final Lyap: {result.final_lyapunov:.4f} (target={lyap_target})")
    log(f"  Final max error: {best_error:.4f}")
    log(f"  Best iteration: {best_iter + 1}")
    log(f"  Converged: {result.converged}")
    if result.stop_reason:
        log(f"  Stop reason: {result.stop_reason}")
    log("")

    return result


# =============================================================================
# HDF5 Data Loading Wrapper
# =============================================================================


def criticality_init_from_hdf5(
    model: SOENModelCore,
    hdf5_path: str | Path,
    split: str = "train",
    config: CriticalityInitConfig | None = None,
    seq_len: int | None = None,
    feature_min: float | None = None,
    feature_max: float | None = None,
) -> CriticalityInitResult:
    """Convenience function to run criticality init from HDF5 data.

    Loads data from an HDF5 file and runs the criticality initialization.
    This is the main entry point when using the GUI.

    Args:
        model: The SOEN model to initialize
        hdf5_path: Path to HDF5 dataset
        split: Dataset split to use ('train', 'val', 'test')
        config: Configuration options
        seq_len: Truncate/pad sequences to this length
        feature_min: Scale features to have this minimum value
        feature_max: Scale features to have this maximum value

    Returns:
        CriticalityInitResult with optimization statistics
    """
    from soen_toolkit.utils.flux_matching_init import load_hdf5_batches

    config = config or CriticalityInitConfig()

    # Get device from model
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    # Load data batches
    batches = load_hdf5_batches(
        hdf5_path,
        split=split,
        batch_size=config.batch_size,
        num_batches=config.num_batches,
        device=device,
        seq_len=seq_len,
        feature_min=feature_min,
        feature_max=feature_max,
    )

    return run_criticality_init(model, batches, config)


# =============================================================================
# High-Level API
# =============================================================================


def initialize_model_for_criticality(
    model: SOENModelCore,
    data: torch.Tensor | list[torch.Tensor] | str | Path,
    *,
    target_branching_ratio: float = 1.0,
    target_lyapunov: float = 0.0,
    enable_br: bool = True,
    enable_lyap: bool = False,
    br_weight: float = 1.0,
    lyap_weight: float = 0.0,
    lyap_eps: float = 1e-4,
    lyap_time_horizon: int | None = None,
    range_penalty_weight: float = 0.0,
    range_clip: float | None = None,
    iterations: int = 100,
    learning_rate: float = 0.01,
    tolerance: float = 0.01,
    patience: int = 10,
    batch_size: int = 32,
    num_batches: int = 5,
    verbose: bool = True,
    log_lyap: bool = False,
    max_nan_tol: int = 0,
    log_fn: Callable[[str], None] | None = None,
) -> CriticalityInitResult:
    """Initialize a SOEN model's weights towards critical dynamics.

    This is the simplest API for criticality initialization. It handles
    various data formats and provides sensible defaults.

    Example usage:
        >>> # From tensor data
        >>> result = initialize_model_for_criticality(model, data_tensor)

        >>> # From HDF5 file
        >>> result = initialize_model_for_criticality(model, "data.h5")

        >>> # With custom target (slightly subcritical)
        >>> result = initialize_model_for_criticality(
        ...     model, data,
        ...     target_branching_ratio=0.95,
        ...     iterations=200,
        ... )

        >>> # Lyapunov-only tuning (BR disabled)
        >>> result = initialize_model_for_criticality(
        ...     model, data,
        ...     enable_br=False,
        ...     enable_lyap=True,
        ...     lyap_weight=1.0,
        ...     target_lyapunov=0.0,
        ... )

    Args:
        model: The SOEN model to initialize (modified in-place)
        data: Input data as Tensor, list of Tensors, or path to HDF5 file
        target_branching_ratio: Target BR value (1.0 = critical)
        target_lyapunov: Target Lyapunov estimate (≈0 for edge-of-chaos)
        enable_br: Whether to include the branching ratio objective
        enable_lyap: Whether to include the Lyapunov objective
        br_weight: Loss weight for BR objective
        lyap_weight: Loss weight for Lyapunov objective
        lyap_eps: Perturbation magnitude for Lyapunov estimate
        lyap_time_horizon: Optional cap on steps used for Lyapunov averaging
        lyap_min_delta0: Minimum acceptable initial perturbation norm
        lyap_max_retries: Maximum perturbation scaling retries if delta collapses
        lyap_noise_growth: Multiplier applied to perturbation per retry
        range_penalty_weight: Weight for optional activity-range penalty
        range_clip: Threshold for activity-range penalty (abs mean activity)
        iterations: Maximum optimization iterations
        learning_rate: Initial learning rate
        tolerance: Stop if |BR - target| < tolerance
        patience: Stop if no improvement for N iterations
        batch_size: Batch size for forward passes
        num_batches: Number of batches per iteration
        log_lyap: Whether to log Lyapunov values
        max_nan_tol: Abort if NaN/inf detections exceed this value
        verbose: Print progress
        log_fn: Custom logging function

    Returns:
        CriticalityInitResult with optimization statistics
    """
    config = CriticalityInitConfig(
        target=target_branching_ratio,
        enable_br=enable_br,
        br_target=target_branching_ratio,
        br_weight=br_weight,
        enable_lyap=enable_lyap,
        lyap_target=target_lyapunov,
        lyap_weight=lyap_weight,
        lyap_eps=lyap_eps,
        lyap_time_horizon=lyap_time_horizon,
        range_penalty_weight=range_penalty_weight,
        range_clip=range_clip,
        log_lyap=log_lyap,
        max_nan_tol=max_nan_tol,
        num_iterations=iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
        patience=patience,
        batch_size=batch_size,
        num_batches=num_batches,
        verbose=verbose,
        log_fn=log_fn,
    )

    # Handle different data types
    if isinstance(data, (str, Path)):
        return criticality_init_from_hdf5(model, data, split="train", config=config)

    if isinstance(data, torch.Tensor):
        # Single tensor - split into batches
        if data.ndim == 2:
            data = data.unsqueeze(0)  # Add batch dimension

        n_samples = data.shape[0]
        batches = []
        for i in range(0, n_samples, batch_size):
            batches.append(data[i:i + batch_size])

    elif isinstance(data, list):
        batches = data

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    return run_criticality_init(model, batches, config)
