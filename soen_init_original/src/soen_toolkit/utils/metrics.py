"""
Criticality metrics for SOEN models.

This module provides tools to quantify the criticality of a SOEN model using
both dynamical metrics (avalanche analysis) and information-theoretic metrics
(branching ratio, susceptibility).

Key Concepts:
    - Branching Ratio (σ): Ratio of output flux to input flux. σ=1.0 is critical.
    - Susceptibility (χ): Variance of network activity. Diverges at critical point.
    - Local Expansion Rate: Lyapunov-like measure of trajectory divergence.
    - Avalanche Exponents: Power-law scaling of activity bursts.

Main Functions:
    - quantify_criticality: Compute all metrics from a forward pass
    - compute_branching_ratio_differentiable: Differentiable BR for optimization
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from soen_toolkit.core.protocols import ModelProtocol

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CriticalityMetrics:
    """Container for criticality metrics results.

    Attributes:
        branching_ratio: Flux-based branching ratio (tensor, differentiable)
        susceptibility: Variance of activity over time (tensor, differentiable)
        local_expansion_rate: Perturbation-based Lyapunov estimate (tensor or None)
        avalanche_exponent_size: Power-law exponent for avalanche sizes (float)
        avalanche_exponent_duration: Power-law exponent for avalanche durations (float)
        avalanche_r_squared_size: R² of size power-law fit
        avalanche_r_squared_duration: R² of duration power-law fit
        avalanche_count: Number of avalanches detected
        is_critical: Heuristic check based on exponents
    """

    # Differentiable metrics (Tensor)
    branching_ratio: torch.Tensor
    susceptibility: torch.Tensor

    # Lyapunov-based metrics
    local_expansion_rate: torch.Tensor | None = None

    # Non-differentiable diagnostic metrics (float/None)
    avalanche_exponent_size: float | None = None
    avalanche_exponent_duration: float | None = None
    avalanche_r_squared_size: float | None = None
    avalanche_r_squared_duration: float | None = None

    # Metadata
    avalanche_count: int = 0
    is_critical: bool | None = None


# =============================================================================
# Numerical Stability Helpers
# =============================================================================


def _soft_abs(x: torch.Tensor, smoothness: float = 0.01) -> torch.Tensor:
    """Smooth approximation to absolute value.

    Uses sqrt(x² + ε²) which is differentiable everywhere and approximates
    |x| for |x| >> ε.

    This avoids the gradient discontinuity at x=0 that causes training instability.

    Args:
        x: Input tensor
        smoothness: Smoothing parameter (smaller = closer to true abs)

    Returns:
        Smooth approximation to |x|
    """
    return torch.sqrt(x ** 2 + smoothness ** 2)


def _stable_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable division that handles near-zero denominators.

    Uses: num / max(|denom|, eps) * sign(denom)

    This prevents division by zero while maintaining gradient flow.

    Args:
        numerator: The dividend
        denominator: The divisor
        eps: Minimum denominator magnitude

    Returns:
        Stable quotient
    """
    denom_abs = torch.abs(denominator)
    denom_safe = torch.where(denom_abs < eps, torch.full_like(denom_abs, eps), denom_abs)
    return numerator / denom_safe


# =============================================================================
# Connectivity Analysis
# =============================================================================


def _build_per_node_connectivity_masks(
    soen_core: ModelProtocol,
    s_histories: list[torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build per-node masks indicating which neurons have incoming/outgoing connections.

    This is used to exclude disconnected neurons from the branching ratio calculation,
    which would otherwise skew the results with extreme values.

    Args:
        soen_core: The SOEN model core
        s_histories: State histories per layer [B, T+1, D]

    Returns:
        Tuple of (has_incoming, has_outgoing) dictionaries mapping layer_id to bool tensors

    Note:
        phi_offset counts as an incoming connection source for neurons with non-zero offset.
    """
    layer_id_list = [cfg.layer_id for cfg in soen_core.layers_config]
    id_to_idx = {lid: i for i, lid in enumerate(layer_id_list)}
    device = s_histories[0].device if s_histories else torch.device("cpu")

    # Initialize masks as all False
    has_incoming: dict[int, torch.Tensor] = {}
    has_outgoing: dict[int, torch.Tensor] = {}

    for lid in layer_id_list:
        D = s_histories[id_to_idx[lid]].shape[-1]
        has_incoming[lid] = torch.zeros(D, dtype=torch.bool, device=device)
        has_outgoing[lid] = torch.zeros(D, dtype=torch.bool, device=device)

    # Parse connections to mark connected neurons
    for name, J in soen_core.connections.items():
        if not name.startswith("J_"):
            continue
        try:
            parts = name.split("_")
            from_lid = int(parts[1])
            to_lid = int(parts[3])
        except Exception:
            continue

        # J shape is [D_to, D_from]
        # A source neuron has outgoing if any row of J for that column is non-zero
        # A target neuron has incoming if any column of J for that row is non-zero
        if from_lid in has_outgoing:
            has_outgoing[from_lid] |= (J.abs().sum(dim=0) > 1e-12)
        if to_lid in has_incoming:
            has_incoming[to_lid] |= (J.abs().sum(dim=1) > 1e-12)

    # phi_offset counts as an incoming connection source
    for lid in layer_id_list:
        idx = id_to_idx[lid]
        layer = soen_core.layers[idx]
        phi_off = getattr(layer, "phi_offset", None)
        if phi_off is not None:
            phi_off_t = torch.as_tensor(phi_off, device=device, dtype=torch.float32)
            if phi_off_t.dim() == 0:
                # Scalar phi_offset applies to all neurons
                if phi_off_t.abs().item() > 1e-12:
                    has_incoming[lid] = torch.ones_like(has_incoming[lid])
            else:
                # Per-neuron phi_offset
                has_incoming[lid] |= (phi_off_t.abs() > 1e-12)

    return has_incoming, has_outgoing


# =============================================================================
# Branching Ratio Computation
# =============================================================================


def compute_branching_ratios_tensor(
    soen_core: ModelProtocol,
    s_histories: list[torch.Tensor],
    layer_ids: list[int] | None = None,
    eps_scale: float = 1e-6,
) -> torch.Tensor | None:
    """Compute raw branching ratio tensor from state histories.

    The branching ratio is defined as:
        BR = sum(output_flux) / sum(input_flux)

    Where:
        - output_flux = |state| * sum(|outgoing_weights|)
        - input_flux = |sum(incoming_state * incoming_weights) + phi_offset|

    A value of 1.0 indicates a critical state where activity is preserved.

    Key Features:
        - Only includes neurons with BOTH incoming and outgoing connections
        - Uses soft_abs for numerical stability and smooth gradients
        - Scales epsilon adaptively based on signal magnitude

    Args:
        soen_core: The SOENModelCore instance
        s_histories: List of state history tensors [B, T+1, D] per layer
        layer_ids: Optional list of layer IDs to include (default: recurrent core)
        eps_scale: Scale factor for adaptive epsilon

    Returns:
        Tensor of shape [L_sel, B, T] with per-layer, per-batch, per-timestep ratios,
        or None if no valid layers found.
    """
    # Map layer_id to index in histories
    layer_id_list = [cfg.layer_id for cfg in soen_core.layers_config]
    id_to_idx = {lid: i for i, lid in enumerate(layer_id_list)}

    # Build per-node connectivity masks
    has_incoming, has_outgoing = _build_per_node_connectivity_masks(soen_core, s_histories)

    # Build layer-level inbound/outbound maps to identify recurrent core
    has_inbound_layer = dict.fromkeys(layer_id_list, False)
    has_outbound_layer = dict.fromkeys(layer_id_list, False)
    for name in soen_core.connections:
        if not name.startswith("J_"):
            continue
        try:
            parts = name.split("_")
            from_lid = int(parts[1])
            to_lid = int(parts[3])
        except Exception:
            continue
        has_outbound_layer[from_lid] = True
        has_inbound_layer[to_lid] = True

    # Default selection: only layers with both inbound and outbound (recurrent core)
    if layer_ids is None:
        layer_ids = [
            lid for lid in layer_id_list
            if has_inbound_layer.get(lid, False) and has_outbound_layer.get(lid, False)
        ]

    if not layer_ids:
        return None

    conn = soen_core.connections
    per_element_ratios: list[torch.Tensor] = []

    for lid in layer_ids:
        L_idx = id_to_idx.get(lid)
        if L_idx is None:
            continue

        # s_histories is typically [B, T+1, D]
        s_L = s_histories[L_idx]
        if s_L.ndim == 3 and s_L.size(1) > 1:
            s_L = s_L[:, 1:, :]  # Skip t=0 (initial state)

        B, T, D_L = s_L.shape

        # Get connectivity mask for this layer - neurons must have BOTH incoming AND outgoing
        incoming_mask = has_incoming.get(lid)
        outgoing_mask = has_outgoing.get(lid)
        if incoming_mask is None or outgoing_mask is None:
            continue

        connected_mask = incoming_mask & outgoing_mask  # [D_L]
        num_connected = connected_mask.sum().item()

        if num_connected == 0:
            # No connected neurons in this layer - skip it
            continue

        # ======================================================================
        # CALCULATE INPUT FLUX (Phi_in)
        # ======================================================================
        # Input flux is the total weighted input arriving at each neuron
        # ======================================================================

        phi_in = torch.zeros(B, T, D_L, device=s_L.device, dtype=s_L.dtype)

        # 1. External Input Flux from other layers
        for j in layer_id_list:
            key = f"J_{j}_to_{lid}"
            if key not in conn:
                continue
            s_j = s_histories[id_to_idx[j]]
            if s_j.ndim == 3 and s_j.size(1) > 1:
                s_j = s_j[:, 1:, :]

            # Clone to avoid in-place version bumps breaking autograd
            J_jL = conn[key].clone()
            # phi_in += s_j @ J_jL.T  (batch matmul)
            phi_in = phi_in + torch.matmul(s_j, J_jL.t())

        # 2. Phi Offset / Internal Parameters
        try:
            layer_module = cast(Any, soen_core.layers[id_to_idx[lid]])

            # Add phi_offset (bias flux)
            phi_off = getattr(layer_module, "phi_offset", 0.0)
            if isinstance(phi_off, torch.Tensor):
                phi_off_t = phi_off.to(device=s_L.device, dtype=s_L.dtype)
                if phi_off_t.ndim == 0:
                    phi_in = phi_in + phi_off_t.view(1, 1, 1)
                else:
                    phi_in = phi_in + phi_off_t.view(1, 1, -1)
            else:
                phi_in = phi_in + torch.tensor(float(phi_off), device=s_L.device, dtype=s_L.dtype).view(1, 1, 1)

            # Add internal connections (self-loops within layer)
            if hasattr(layer_module, "internal_J") and layer_module.internal_J is not None:
                internal_J = layer_module.internal_J.clone()
                phi_in = phi_in + torch.matmul(s_L, internal_J.t())

        except Exception:
            pass

        # ======================================================================
        # CALCULATE OUTPUT FLUX (Phi_out)
        # ======================================================================
        # Output flux is the neuron's state magnitude times total outgoing weight
        # ======================================================================

        outbound_mag = torch.zeros(D_L, device=s_L.device, dtype=s_L.dtype)
        for k in layer_id_list:
            key = f"J_{lid}_to_{k}"
            if key not in conn:
                continue
            # Clone to avoid in-place version bumps breaking autograd
            J_Lk = conn[key].clone()
            # Sum of absolute outgoing weights per source neuron
            outbound_mag = outbound_mag + J_Lk.abs().sum(dim=0)

        # ======================================================================
        # APPLY CONNECTIVITY MASK AND COMPUTE RATIO
        # ======================================================================

        connected_mask_expanded = connected_mask.view(1, 1, D_L).float()

        # Use soft_abs for smooth gradients
        phi_in_abs = _soft_abs(phi_in, smoothness=1e-4) * connected_mask_expanded

        # Outgoing magnitude with small epsilon for connected neurons
        outbound_mag_masked = outbound_mag * connected_mask.float()

        # Adaptive epsilon based on signal scale
        signal_scale = max(
            phi_in_abs.detach().mean().item(),
            s_L.detach().abs().mean().item(),
            1e-6
        )
        eps = eps_scale * signal_scale

        # Add epsilon only to connected neurons
        outbound_mag_safe = outbound_mag_masked + eps * connected_mask.float()

        # Output flux: |state| * |outgoing_weights|
        phi_out = _soft_abs(s_L, smoothness=1e-4) * outbound_mag_safe.view(1, 1, D_L)

        # Sum over neuron dimension
        phi_in_sum = phi_in_abs.sum(dim=2)  # [B, T]
        phi_out_sum = phi_out.sum(dim=2)  # [B, T]

        # Stable division
        ratio_BT = _stable_divide(phi_out_sum, phi_in_sum + eps * num_connected)
        per_element_ratios.append(ratio_BT)

    if not per_element_ratios:
        return None

    return torch.stack(per_element_ratios, dim=0)  # [L_sel, B, T]


def compute_branching_ratio_differentiable(
    model: nn.Module,
    s_histories: list[torch.Tensor],
    layer_ids: list[int] | None = None,
    eps_scale: float = 1e-6,
) -> torch.Tensor:
    """Compute differentiable branching ratio for optimization.

    This is the main function used by the criticality initialization algorithm.
    It computes a scalar branching ratio that can be backpropagated through.

    The branching ratio measures how activity propagates through the network:
        - BR = 1.0: Critical - activity is preserved
        - BR < 1.0: Subcritical - activity decays
        - BR > 1.0: Supercritical - activity explodes

    Args:
        model: The SOEN model (or its core)
        s_histories: List of state history tensors [B, T+1, D] per layer
        layer_ids: Optional list of layer IDs (default: auto-detect recurrent core)
        eps_scale: Scale factor for numerical stability epsilon

    Returns:
        Scalar tensor containing mean branching ratio (differentiable)
    """
    # Access SOEN core if model is wrapped
    soen_core = getattr(model, "model", None)
    if soen_core is None:
        # Fallback for when model IS the core
        if hasattr(model, "layers_config"):
            soen_core = model
        else:
            logger.warning("Could not find SOEN core - returning zero BR")
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

    # Narrow type for mypy
    soen_core_protocol = cast("ModelProtocol", soen_core)

    ratios = compute_branching_ratios_tensor(soen_core_protocol, s_histories, layer_ids, eps_scale)

    if ratios is None:
        logger.warning("No valid layers for BR computation - returning zero")
        return torch.tensor(0.0, device=next(soen_core.parameters()).device, requires_grad=True)

    # Return mean over all layers, batches, and timesteps
    return ratios.mean()


# Legacy alias for backward compatibility
_compute_branching_ratio = compute_branching_ratio_differentiable


# =============================================================================
# Susceptibility
# =============================================================================


def _compute_susceptibility(s_histories: list[torch.Tensor]) -> torch.Tensor:
    """Compute susceptibility (χ) as the variance of network activity.

    Susceptibility measures the fluctuations in total network activity.
    It diverges at the critical point, making it useful as a diagnostic.

    Formula:
        χ = Var[A(t)] where A(t) = mean_neurons(state_t)

    Args:
        s_histories: List of state tensors [B, T+1, D] per layer

    Returns:
        Scalar tensor with susceptibility value
    """
    if not s_histories:
        return torch.tensor(0.0)

    # Concatenate all layer states to form global state [B, T, Total_Neurons]
    # Skip t=0 (initial state)
    states_list = []
    for s in s_histories:
        if s.ndim == 3 and s.size(1) > 1:
            states_list.append(s[:, 1:, :])
        elif s.ndim == 3:
            states_list.append(s)

    if not states_list:
        return torch.tensor(0.0, device=s_histories[0].device)

    global_state = torch.cat(states_list, dim=2)  # [B, T, N_total]

    # Mean activity over neurons at each time step: A(t) = mean_i(s_i(t))
    activity_t = global_state.mean(dim=2)  # [B, T]

    # Susceptibility = Variance of A(t) over time, averaged over batch
    susceptibility = torch.var(activity_t, dim=1).mean()

    return susceptibility


# =============================================================================
# Local Expansion Rate (Lyapunov-like)
# =============================================================================


def _compute_local_expansion_rate(
    model: nn.Module,
    inputs: torch.Tensor,
    clean_states: list[torch.Tensor] | None = None,
    perturbation_scale: float = 1e-5,
    eps: float = 1e-8,
) -> torch.Tensor | None:
    """Compute the local expansion rate using perturbation analysis.

    This metric approximates the largest local Lyapunov exponent by measuring
    how small perturbations to the input grow or shrink over time.

    Formula:
        σ = mean_t[ ||h'_t - h_t|| / ||h'_{t-1} - h_{t-1}|| ]

    Where h is the network state and h' is the state from perturbed input.

    Args:
        model: The SOEN model
        inputs: Input tensor [B, T, D]
        clean_states: Pre-computed clean states (optional, avoids re-running)
        perturbation_scale: Scale of input perturbation relative to signal
        eps: Small constant for numerical stability

    Returns:
        Scalar tensor with expansion rate, or None if computation fails
    """
    # 1. Get Clean States (h_t)
    if clean_states is None:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            output = model(inputs)
            # Extract states
            if isinstance(output, tuple):
                if len(output) == 2 and isinstance(output[1], list):
                    clean_states = output[1]
                elif len(output) == 3 and isinstance(output[2], list):
                    clean_states = output[2]

            if clean_states is None:
                clean_states = getattr(model, "latest_all_states", None)
        model.train(was_training)

    if clean_states is None or not clean_states:
        return None

    # 2. Run Perturbed Pass (h'_t)
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            # Create perturbation scaled to input magnitude
            input_scale = inputs.abs().mean() + eps
            perturbation = torch.randn_like(inputs) * perturbation_scale * input_scale
            inputs_perturbed = inputs + perturbation

            # Forward perturbed
            output_p = model(inputs_perturbed)

            # Extract perturbed states
            perturbed_states = None
            if isinstance(output_p, tuple):
                if len(output_p) == 2 and isinstance(output_p[1], list):
                    perturbed_states = output_p[1]
                elif len(output_p) == 3 and isinstance(output_p[2], list):
                    perturbed_states = output_p[2]

            if perturbed_states is None:
                perturbed_states = getattr(model, "latest_all_states", None)

            if perturbed_states is None or not perturbed_states:
                return None

            # 3. Compute Expansion Ratios
            clean_list = []
            pert_list = []

            for s_c, s_p in zip(clean_states, perturbed_states, strict=False):
                if s_c.ndim == 3 and s_c.size(1) > 1:
                    clean_list.append(s_c)
                    pert_list.append(s_p)

            if not clean_list:
                return None

            # Concatenate all layers: [B, T+1, N_total]
            h_clean = torch.cat(clean_list, dim=2)
            h_pert = torch.cat(pert_list, dim=2)

            # Distance at each time step: d_t = ||h'_t - h_t||
            d_t = torch.norm(h_pert - h_clean, p=2, dim=2)  # [B, T+1]

            # Expansion ratios: d_t / d_{t-1}
            d_prev = d_t[:, :-1] + eps
            d_curr = d_t[:, 1:]
            ratios = d_curr / d_prev

            # Average over time and batch
            sigma = ratios.mean()

            return sigma

    except Exception as e:
        logger.warning(f"Local expansion rate calculation failed: {e}")
        return None
    finally:
        model.train(was_training)


# =============================================================================
# Avalanche Analysis (Non-differentiable)
# =============================================================================


def _fit_power_law(data: np.ndarray) -> tuple[float, float]:
    """Fit a power law P(x) ~ x^-alpha to data.

    Uses log-log linear regression for simplicity and robustness.

    Args:
        data: Array of values to fit

    Returns:
        Tuple of (alpha, r_squared)
    """
    if len(data) < 5:
        return 0.0, 0.0

    try:
        min_val = max(1.0, np.min(data))
        max_val = np.max(data)

        if min_val >= max_val:
            return 0.0, 0.0

        # Logarithmic binning
        bins = np.logspace(np.log10(min_val), np.log10(max_val), num=20)
        hist, bin_edges = np.histogram(data, bins=bins, density=True)

        # Geometric center of bins
        bin_centers = (bin_edges[:-1] * bin_edges[1:]) ** 0.5

        # Filter non-zero bins
        valid = hist > 0
        x = bin_centers[valid]
        y = hist[valid]

        if len(x) < 3:
            return 0.0, 0.0

        # Linear fit in log-log space
        log_x = np.log10(x)
        log_y = np.log10(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = -coeffs[0]

        # R-squared
        p = np.poly1d(coeffs)
        yhat = p(log_x)
        ybar = np.mean(log_y)
        ssreg = np.sum((yhat - ybar) ** 2)
        sstot = np.sum((log_y - ybar) ** 2)
        r_squared = ssreg / sstot if sstot > 1e-9 else 0.0

        return float(alpha), float(r_squared)

    except Exception as e:
        logger.warning(f"Power law fit failed: {e}")
        return 0.0, 0.0


def _analyze_avalanches(
    s_histories: list[torch.Tensor],
    threshold: float = 0.1
) -> tuple[float, float, float, float, int]:
    """Analyze avalanche scaling properties.

    Avalanches are periods of sustained network activity above a threshold.
    At criticality, avalanche sizes and durations follow power laws with
    specific exponents (α ≈ 1.5 for size, β ≈ 2.0 for duration).

    Args:
        s_histories: List of state tensors [B, T+1, D] per layer
        threshold: Activity threshold for detecting events

    Returns:
        Tuple of (alpha, r_sq_size, beta, r_sq_dur, avalanche_count)
    """
    # Convert to numpy for analysis (non-differentiable)
    states_list = []
    for s in s_histories:
        if s.ndim == 3 and s.size(1) > 1:
            states_list.append(s[:, 1:, :])

    if not states_list:
        return 0.0, 0.0, 0.0, 0.0, 0

    global_state = torch.cat(states_list, dim=2).detach().cpu().numpy()

    # Binarize events (neuron active if |state| > threshold)
    events = (np.abs(global_state) > threshold).astype(int)

    # Sum events across neurons -> Activity A(t) [B, T]
    activity = np.sum(events, axis=2)

    avalanche_sizes = []
    avalanche_durations = []

    # Process each batch item
    for b in range(activity.shape[0]):
        act_seq = activity[b]

        # Find periods where activity > 0
        is_active = act_seq > 0

        # Identify avalanche starts and ends
        padded = np.concatenate(([0], is_active, [0]))
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends, strict=False):
            duration = e - s
            size = np.sum(act_seq[s:e])
            avalanche_sizes.append(size)
            avalanche_durations.append(duration)

    if not avalanche_sizes:
        return 0.0, 0.0, 0.0, 0.0, 0

    # Fit power laws
    alpha, r_sq_size = _fit_power_law(np.array(avalanche_sizes))
    beta, r_sq_dur = _fit_power_law(np.array(avalanche_durations))

    return alpha, r_sq_size, beta, r_sq_dur, len(avalanche_sizes)


# =============================================================================
# Main Entry Point
# =============================================================================


def quantify_criticality(
    model: nn.Module,
    inputs: torch.Tensor | None = None,
    states: list[torch.Tensor] | None = None,
    threshold: float = 0.05,
    as_loss: bool = False,
    target_branching_ratio: float = 1.0,
) -> Union[CriticalityMetrics, torch.Tensor]:
    """Quantify the criticality of a SOEN model.

    This is the main entry point for criticality analysis. It computes:
    - Branching ratio (differentiable)
    - Susceptibility (differentiable)
    - Local expansion rate (perturbation-based)
    - Avalanche statistics (non-differentiable, diagnostic only)

    Example:
        >>> # Compute all metrics
        >>> metrics = quantify_criticality(model, inputs=data)
        >>> print(f"BR: {metrics.branching_ratio.item():.4f}")
        >>> print(f"Is critical: {metrics.is_critical}")

        >>> # Use as loss for optimization
        >>> loss = quantify_criticality(model, inputs=data, as_loss=True)
        >>> loss.backward()

    Args:
        model: SOEN model (LightningModule or SOENModelCore)
        inputs: Input tensor [B, T, D]. Required if states not provided.
        states: Pre-computed state histories (list of tensors per layer).
        threshold: Threshold for avalanche detection.
        as_loss: If True, returns scalar loss = (BR - target)^2.
        target_branching_ratio: Target BR for loss computation.

    Returns:
        CriticalityMetrics object, or scalar loss tensor if as_loss=True.

    Raises:
        ValueError: If neither inputs nor states provided.
    """
    # 1. Get State Histories
    all_states = states

    # If inputs provided (and no states), run forward pass
    if all_states is None and inputs is not None:
        was_training = model.training
        if not as_loss:
            model.eval()

        output = model(inputs)

        # Capture states from return value
        if isinstance(output, tuple):
            if len(output) == 2 and isinstance(output[1], list):
                all_states = output[1]
            elif len(output) == 3 and isinstance(output[2], list):
                all_states = output[2]

        if not as_loss:
            model.train(was_training)

    # Fallback to cached states
    if all_states is None:
        all_states = getattr(model, "latest_all_states", None)

    if all_states is None:
        msg = "No state history found. Provide 'inputs' or ensure model caches 'latest_all_states'."
        if as_loss:
            logger.warning(msg)
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        raise ValueError(msg)

    # 2. Compute Differentiable Metrics
    branching_ratio = compute_branching_ratio_differentiable(model, all_states)
    susceptibility = _compute_susceptibility(all_states)

    # Compute Local Expansion Rate (if inputs available)
    local_expansion_rate = None
    if inputs is not None:
        local_expansion_rate = _compute_local_expansion_rate(
            model,
            inputs,
            clean_states=all_states
        )

    # 3. Return Loss if requested
    if as_loss:
        return (branching_ratio - target_branching_ratio) ** 2

    # 4. Compute Avalanche Statistics (Non-differentiable diagnostic)
    alpha, r_sq_size, beta, r_sq_dur, count = _analyze_avalanches(all_states, threshold=threshold)

    # 5. Heuristic Criticality Check
    # Critical if: α ≈ 1.5, β ≈ 2.0, good power-law fit
    is_critical = (1.2 <= alpha <= 1.8) and (1.7 <= beta <= 2.3) and (r_sq_size > 0.8)

    return CriticalityMetrics(
        branching_ratio=branching_ratio,
        susceptibility=susceptibility,
        local_expansion_rate=local_expansion_rate,
        avalanche_exponent_size=alpha,
        avalanche_exponent_duration=beta,
        avalanche_r_squared_size=r_sq_size,
        avalanche_r_squared_duration=r_sq_dur,
        avalanche_count=count,
        is_critical=is_critical
    )
