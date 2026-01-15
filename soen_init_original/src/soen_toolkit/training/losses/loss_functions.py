from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from soen_toolkit.core.layers.physical import SingleDendriteLayer

# Ensure LOSS_REGISTRY and register_loss are defined in src/soen_toolkit/training/losses/__init__.py
# from . import register_loss # This line might be needed if running this file directly for testing


# @register_loss("reg_J_loss") # Decorator will be in __init__.py
def reg_J_loss(model: nn.Module, threshold: float = 0.24, scale: float = 1.0, factor: float = 0.01) -> torch.Tensor:
    """Penalizes connection weights (J_ and internal_J) exponentially
    if their absolute value exceeds a threshold.
    Penalty = factor * (exp(scale * (abs(weight) - threshold)) - 1).

    Args:
        model (nn.Module): The SOENLightningModule instance.
        threshold (float): The value above which weights are penalized.
        scale (float): Controls the steepness of the exponential penalty.
        factor (float): Overall scaling factor for this loss component.

    Returns:
        torch.Tensor: Scalar loss value.

    """
    total_penalty = torch.tensor(0.0, device=next(model.parameters()).device)

    # Access the SOENModelCore (LightningModule.model now points to core)
    soen_core_model = getattr(model, "model", None)
    if soen_core_model is None:
        raise RuntimeError("reg_J_loss requires a LightningModule with `.model` (SOENModelCore) attached.")

    # Penalize all learnable connections once (unified naming includes internal as J_i_to_i)
    if hasattr(soen_core_model, "connections"):
        for param in soen_core_model.connections.values():
            if not param.requires_grad:
                continue
            # name may be J_i_to_j or legacy internal_i (if still present)
            abs_param = torch.abs(param)
            above_threshold_mask = abs_param > threshold
            penalized_values = abs_param[above_threshold_mask]
            if penalized_values.numel() > 0:
                penalty = torch.exp(scale * (penalized_values - threshold)) - 1.0
                total_penalty += torch.sum(penalty)

    return factor * total_penalty


# Note: Connection noise configuration can be extended by updating GUI components
# to include NoiseConfig settings and ensuring layer solvers apply noise correctly.


def top_gap_loss(
    model: nn.Module,
    targets: torch.Tensor,
    separation_threshold: float = 0.1,
    max_penalty_offset: float = 1.0,
    factor: float = 1.0,
) -> torch.Tensor:
    """Margin regulariser that reads the max-pooled output activations from the classifier wrapper,
    backs out any scalar post-scaling so it operates on the raw values,
    gives every wrong prediction the maximum penalty, and every correct
    prediction a linearly tapering penalty if its margin is below separation_threshold.
    """
    # ───────────────────────────────────────────────────────────────────
    # 1. Locate the processed activations
    # ───────────────────────────────────────────────────────────────────
    if not hasattr(model, "latest_processed_state"):
        # Could not find the expected attributes
        raise RuntimeError("top_gap_loss requires `model.latest_processed_state` to be cached by the Lightning wrapper.")

    proc = model.latest_processed_state  # [B, D]   (scaled)
    if proc is None or proc.ndim != 2:
        raise RuntimeError("top_gap_loss requires `model.latest_processed_state` to be a [B, D] tensor.")

    # ───────────────────────────────────────────────────────────────────
    # 2. Undo the user-defined scale factor so we work with raw values
    # ───────────────────────────────────────────────────────────────────
    scale = 1.0
    if hasattr(model, "time_pooling_params"):
        scale = float(model.time_pooling_params.get("scale", 1.0))  # type: ignore[union-attr, operator]
    if scale != 0.0:
        proc = proc / scale  # type: ignore[assignment,operator]

    B, D = proc.shape  # type: ignore[misc]
    if B == 0 or D < 2:
        return proc.new_tensor(0.0)  # type: ignore[operator]

    # ───────────────────────────────────────────────────────────────────
    # 3. Work out predictions and margins
    # ───────────────────────────────────────────────────────────────────
    top2_vals, top2_idx = torch.topk(proc, k=2, dim=1)  # type: ignore[arg-type]
    top1_vals, top1_idx = top2_vals[:, 0], top2_idx[:, 0]
    runner_up_vals = top2_vals[:, 1]

    correct_mask = top1_idx.eq(targets)  # [B]

    # -- penalties initialised to max for everyone (wrong preds keep it) --
    penalties = proc.new_full((B,), max_penalty_offset)  # type: ignore[operator]

    if torch.any(correct_mask):
        gap = top1_vals[correct_mask] - runner_up_vals[correct_mask]  # [N_correct]
        shortfall = torch.clamp(separation_threshold - gap, min=0.0)  # [N_correct]
        penalties[correct_mask] = max_penalty_offset * (shortfall / separation_threshold)

    return factor * penalties.mean()


def get_off_the_ground_loss(
    processed_output_states: torch.Tensor,
    threshold: float = 0.01,
    min_penalty_offset: float = 0.5,
    max_penalty_offset: float = 10.0,
    factor: float = 1.0,  # Overall scaling factor for this loss component
) -> torch.Tensor:
    """Penalizes processed output state values linearly if their absolute
    magnitude is below a threshold.
    The penalty ranges from max_penalty_offset (at state zero) down to
    min_penalty_offset (at abs(state) == threshold).
    No penalty if abs(state) >= threshold.

    Args:
        processed_output_states (torch.Tensor): State values after output processing.
                                               Shape: [batch_size, feature_dim]
        threshold (float): The magnitude below which states are penalized.
                           Must be > 0 for the linear scaling to make sense.
        min_penalty_offset (float): The loss added when abs(state) is at the threshold.
        max_penalty_offset (float): The loss added when state is zero.
        factor (float): Overall scaling factor for this loss component.

    Returns:
        torch.Tensor: Scalar loss value.

    """
    if processed_output_states is None:
        raise ValueError("get_off_the_ground_loss received None for processed_output_states.")

    if threshold <= 0:
        raise ValueError("get_off_the_ground_loss: threshold must be > 0.")

    if min_penalty_offset > max_penalty_offset:
        raise ValueError("get_off_the_ground_loss: min_penalty_offset must not exceed max_penalty_offset.")

    abs_states = torch.abs(processed_output_states)

    # Create a mask for states within the penalty window (0 <= abs_states < threshold)
    # Note: We use '< threshold' to ensure that at exactly abs_states == threshold, the penalty is min_penalty_offset
    # If we used '<= threshold', the linear interpolation would give min_penalty_offset at the threshold,
    # but states >= threshold should get 0 penalty from the outer logic.
    # The logic below correctly applies the linear ramp and then zeros out anything >= threshold.

    penalty_values = torch.zeros_like(abs_states)

    # Calculate the slope of the linear penalty
    # Slope = (change in penalty) / (change in abs_state)
    # Change in penalty = max_penalty_offset - min_penalty_offset
    # Change in abs_state = threshold - 0 = threshold
    slope = (max_penalty_offset - min_penalty_offset) / threshold

    # For states where 0 <= abs_s < threshold:
    # penalty(abs_s) = max_penalty_offset - slope * abs_s
    # This is equivalent to: min_penalty_offset + slope * (threshold - abs_s)

    # Apply penalty only to values within the threshold
    # Create a mask for elements where abs_states < threshold
    mask = abs_states < threshold

    # Calculate penalty for elements within the mask
    # Penalty decreases linearly from max_penalty_offset at 0 to min_penalty_offset at threshold
    penalty_for_masked_elements = max_penalty_offset - slope * abs_states[mask]

    # Ensure this penalty is not less than min_penalty_offset for the masked elements
    # (though by construction, at abs_states[mask] closest to threshold, it will be near min_penalty_offset)
    # And also not less than 0 if min_penalty_offset is negative (which it shouldn't be for a loss).
    # Clamping might be useful if min_penalty_offset could be negative, but for loss it should be positive.
    # The formula already handles the linear ramp down to min_penalty_offset at the edge of the mask.

    penalty_values[mask] = penalty_for_masked_elements

    # Final check: ensure penalty is non-negative.
    # If min_penalty_offset is, for example, 0, then states very close to the threshold
    # would get a penalty near 0.
    # The behavior when min_penalty_offset = 0 and abs_state is exactly threshold needs care.
    # The current mask `abs_states < threshold` means at `abs_states == threshold`, penalty is 0.
    # If you want `min_penalty_offset` to apply *at* the threshold, the calculation needs adjustment
    # or the interpretation of "within the window."

    # Let's refine the above for clarity and to match the verbal description:
    # Penalty is max_penalty when abs_state is 0
    # Penalty is min_penalty when abs_state is threshold
    # Penalty is 0 when abs_state > threshold

    # Re-calculate more directly:
    # For 0 <= x < threshold: y = m*x + c
    # Point 1: (0, max_penalty_offset) => c = max_penalty_offset
    # Point 2: (threshold, min_penalty_offset) => min_penalty_offset = m*threshold + max_penalty_offset
    #                                        => m = (min_penalty_offset - max_penalty_offset) / threshold
    # So, penalty_value = ((min_penalty_offset - max_penalty_offset) / threshold) * abs_s + max_penalty_offset

    penalty_values_refined = torch.zeros_like(abs_states)

    # Mask for states strictly inside the window (0 <= abs_s < threshold)
    # And also for states exactly at the threshold if min_penalty_offset > 0
    # If min_penalty_offset is 0, then at abs_s == threshold, penalty should be 0.

    inside_mask = abs_states < threshold
    at_threshold_mask = abs_states == threshold

    # Calculate for states < threshold
    if torch.any(inside_mask):
        m = (min_penalty_offset - max_penalty_offset) / threshold
        c = max_penalty_offset
        penalty_values_refined[inside_mask] = m * abs_states[inside_mask] + c

    # If min_penalty_offset > 0, apply it for states exactly at threshold
    if min_penalty_offset > 0:
        penalty_values_refined[at_threshold_mask] = min_penalty_offset

    # Ensure penalty is non-negative, especially if min_penalty < max_penalty was violated
    penalty_values_refined = torch.relu(penalty_values_refined)

    return factor * torch.mean(penalty_values_refined)


def exp_high_state_penalty(
    processed_output_states: torch.Tensor,
    threshold: float = 1.0,
    scale: float = 1.0,
    factor: float = 1.0,
) -> torch.Tensor:
    """Exponentially penalise state magnitudes above ``threshold``.

    Any absolute value greater than ``threshold`` incurs a penalty of
    ``exp(scale * (|s| - threshold)) - 1``.  The mean penalty is multiplied
    by ``factor``.
    """
    if processed_output_states is None:
        raise ValueError("exp_high_state_penalty received None for processed_output_states.")

    abs_s = torch.abs(processed_output_states)
    excess = torch.clamp(abs_s - threshold, min=0.0)
    penalties = torch.exp(scale * excess) - 1.0
    return factor * penalties.mean()


def gap_loss(
    model: nn.Module,
    targets: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """A simple margin-based loss that penalizes incorrect class logits that are
    too close to the correct class's logit.

    This version is corrected to avoid in-place operations that cause
    RuntimeErrors during the backward pass.

    Args:
        model (nn.Module): The SOENLightningModule instance.
        targets (torch.Tensor): The ground truth class indices.
        margin (float): The desired minimum gap.

    Returns:
        torch.Tensor: A scalar loss value.

    """
    logits = model.latest_processed_state

    if logits is None or logits.ndim != 2:
        raise RuntimeError("gap_loss requires `model.latest_processed_state` to be a [B, C] tensor.")

    batch_size, _num_classes = logits.shape  # type: ignore[misc]
    if batch_size == 0:
        raise ValueError("gap_loss received empty batch (batch_size == 0).")

    target_logits = logits[torch.arange(batch_size), targets].unsqueeze(1)  # type: ignore[index]

    # This calculates penalties for all classes.
    # The penalty for the correct class will always be `margin`,
    # while the penalties for incorrect classes will be correct.
    penalties = torch.relu(logits - target_logits + margin)  # type: ignore[operator]

    # Sum all penalties for each sample. This sum includes the unwanted `margin`
    # that was added for the correct class.
    total_penalty_per_sample = penalties.sum(dim=1)

    # **THE FIX**: Instead of modifying the tensor, we simply subtract the known,
    # incorrect penalty that was applied to the target class.
    correct_penalty = total_penalty_per_sample - margin

    return correct_penalty.mean()


"""
Total loss
 L = λ_gap·L_gap + λ_noise·L_noise + λ_entropy·L_entropy
   + λ_consistency·L_consis + λ_diversity·L_div
How to tune/interpret hyper-parameters
margin (m) – start with 0.2–0.5
larger encourages bigger gaps but can slow learning.
sigma_noise (σ) – 0.03–0.1 keeps noise small relative to logits.
λ_gap – must dominate (typically 1.0)
drives core classification.
λ_noise – 0.2–0.5 if you value robustness.
λ_entropy – 0.05–0.2 moderates confidence
larger discourages over-confidence.
λ_consistency – 0.1–0.3 controls intra-batch structure
set to 0 if batch classes are very unbalanced.
λ_diversity – 0.05–0.2 pushes class means apart
useful when many classes cluster too closely.
"""


class rich_margin_loss(nn.Module):
    def __init__(
        self,
        margin: float = 0.5,
        sigma_noise: float = 0.05,
        λ_gap: float = 1.0,
        λ_noise: float = 0.3,
        λ_entropy: float = 0.1,
        λ_consistency: float = 0.2,
        λ_diversity: float = 0.1,
    ) -> None:
        super().__init__()
        self.m = margin
        self.sigma_noise = sigma_noise
        self.λ_gap = λ_gap
        self.λ_noise = λ_noise
        self.λ_entropy = λ_entropy
        self.λ_consistency = λ_consistency
        self.λ_diversity = λ_diversity

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape
        device = logits.device

        # 1) Your existing gap loss
        tgt_logits = logits[torch.arange(B, device=device), targets].unsqueeze(1)
        margins = self.m + logits - tgt_logits
        hinge = F.relu(margins)
        focal_w = 1 - torch.sigmoid(tgt_logits - logits)
        gap_loss = (hinge.pow(2) * focal_w).sum(dim=1).mean()

        # 2) Your existing noise consistency
        noise = torch.randn_like(logits) * self.sigma_noise * logits.abs()
        p = F.softmax(logits, dim=1)
        q = F.softmax(logits + noise, dim=1)
        noise_loss = F.kl_div(q.log(), p, reduction="batchmean")

        # 3) NEW: Entropy regularization (prevent overconfidence)
        entropy_loss = -(p * p.log()).sum(dim=1).mean()

        # 4) NEW: Consistency across mini-batch (similar samples should be similar)
        features_norm = F.normalize(logits, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())
        same_class = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()
        consistency_loss = ((similarity - same_class) ** 2).mean()

        # 5) NEW: Diversity loss (encourage different classes to be different)
        class_means_list: list[torch.Tensor] = []
        for c in range(C):
            mask = targets == c
            if mask.sum() > 0:
                class_means_list.append(logits[mask].mean(0))

        if len(class_means_list) > 1:
            class_means = torch.stack(class_means_list)
            class_similarity = F.cosine_similarity(
                class_means.unsqueeze(1),
                class_means.unsqueeze(0),
                dim=2,
            )
            # Penalize high similarity between different classes
            diversity_loss = class_similarity.triu(diagonal=1).mean()
        else:
            diversity_loss = torch.tensor(0.0, device=device)

        return (
            self.λ_gap * gap_loss
            + self.λ_noise * noise_loss
            + self.λ_entropy * (-entropy_loss)  # Negative because we want some entropy
            + self.λ_consistency * consistency_loss
            + self.λ_diversity * diversity_loss
        )


def autoregressive_loss(
    model: nn.Module,
    targets: torch.Tensor,
    sequence_outputs: torch.Tensor,
    time_steps_per_token: int = 1,
    start_timestep: int = 0,
    vocab_size: int | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Autoregressive sequence-to-sequence loss for character/token prediction.

    This loss function computes cross-entropy at each timestep between the model's
    predictions and the input sequence shifted by 1 (next token prediction).

    Args:
        model: The SOENLightningModule instance (not directly used, for compatibility)
        targets: Original input sequence [batch, seq_len] - contains character indices
        sequence_outputs: Model outputs at all timesteps [batch, seq_len+1, output_dim]
        time_steps_per_token: How many simulation timesteps per token (default: 1)
        start_timestep: Which timestep to start computing loss from (default: 0)
        vocab_size: Size of vocabulary (inferred from output_dim if None)
        ignore_index: Index to ignore in loss computation (default: -100)

    Returns:
        torch.Tensor: Mean loss across valid timesteps

    """
    _batch_size, seq_len = targets.shape
    _, output_seq_len, output_dim = sequence_outputs.shape

    if vocab_size is None:
        vocab_size = output_dim
    elif output_dim != vocab_size:
        msg = f"Output dimension {output_dim} doesn't match vocab_size {vocab_size}"
        raise ValueError(msg)

    # Create shifted targets: predict next token at each timestep
    # Input:  [0, 1, 2, 3, 4]
    # Target: [1, 2, 3, 4, 5] where 5 could be <END> token or padded

    # For autoregressive training:
    # - At timestep t, we predict targets[t+1]
    # - We need seq_len predictions to cover all target positions
    # - sequence_outputs has shape [batch, seq_len+1, vocab_size] (includes initial state)

    total_loss: torch.Tensor | float = 0.0
    valid_timesteps = 0

    # Process in chunks of time_steps_per_token
    for token_idx in range(seq_len - 1):  # -1 because we predict the next token
        # Determine which simulation timestep corresponds to this token
        sim_timestep = start_timestep + token_idx * time_steps_per_token

        # Ensure we don't exceed available outputs
        if sim_timestep + 1 >= output_seq_len:
            break

        # Get predictions at this timestep (skip initial state at t=0)
        logits = sequence_outputs[:, sim_timestep + 1, :]  # +1 to skip initial state

        # Get target (next token)
        target_tokens = targets[:, token_idx + 1]

        # Skip if target contains ignore_index
        valid_mask = target_tokens != ignore_index
        if valid_mask.sum() == 0:
            continue

        # Compute cross-entropy loss for valid targets
        loss = F.cross_entropy(
            logits[valid_mask],
            target_tokens[valid_mask],
            reduction="mean",
        )

        total_loss += loss
        valid_timesteps += 1

    if valid_timesteps == 0:
        raise ValueError("autoregressive_loss: no valid timesteps found (all targets ignored or no outputs available).")

    if isinstance(total_loss, float):
        return torch.tensor(total_loss / valid_timesteps, device=targets.device, requires_grad=True)
    return total_loss / valid_timesteps


# Simplified autoregressive loss for direct use
def autoregressive_cross_entropy(outputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Compute cross-entropy loss for autoregressive training.

    Args:
        outputs: Model predictions [batch, seq_len, vocab_size]
        targets: Target sequence [batch, seq_len] - already properly shifted

    Returns:
        Cross-entropy loss

    """
    # Check if targets are scalar (wrong usage)
    if targets.dim() == 1 or (targets.dim() == 2 and targets.shape[1] == 1):
        raise ValueError("autoregressive_cross_entropy received scalar targets; expected [B, seq_len] token indices.")

    # Handle shape mismatch between outputs and targets
    batch_size, seq_len_out, vocab_size = outputs.shape
    batch_size_tgt, seq_len_tgt = targets.shape

    if batch_size != batch_size_tgt:
        raise ValueError(f"Batch size mismatch: outputs batch={batch_size}, targets batch={batch_size_tgt}")

    # Handle sequence length mismatch (SOEN outputs seq_len+1, we need seq_len)
    if seq_len_out != seq_len_tgt:
        import logging

        logger = logging.getLogger(__name__)
        if seq_len_out == seq_len_tgt + 1:
            # Common case: SOEN outputs seq_len+1 (includes initial state at timestep 0)
            # Skip the first timestep (initial state) and use timesteps 1 through seq_len
            logger.info(f"Skipping initial timestep (0) from outputs, using timesteps 1-{seq_len_tgt} for autoregressive loss")
            outputs = outputs[:, 1:, :]  # Skip first timestep, take timesteps 1 through seq_len
        else:
            msg = f"Sequence length mismatch for autoregressive_cross_entropy: outputs seq_len={seq_len_out}, targets seq_len={seq_len_tgt}"
            raise ValueError(msg)

    # Compute cross-entropy loss
    # Flatten for cross_entropy: [batch*seq_len, vocab_size] and [batch*seq_len]
    outputs_flat = outputs.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    return F.cross_entropy(outputs_flat, targets_flat)


def mse(outputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Mean squared error suitable for regression tasks.

    Handles common SOEN shapes:
    - outputs [B, T+1, D] vs targets [B, T, D] -> drop first output timestep
    - outputs [B, D] vs targets [B, T, D] -> reduce targets by mean over time
    - outputs [B, T, D] vs targets [B, D] -> reduce outputs by mean over time
    """
    # Align time dimensions when SOEN includes initial state
    if outputs.dim() == 3 and targets.dim() == 3:
        if outputs.size(1) == targets.size(1) + 1:
            outputs = outputs[:, 1:, :]
    elif outputs.dim() == 2 and targets.dim() == 3:
        # Pool targets if model produced pooled outputs
        targets = targets.mean(dim=1)
    elif outputs.dim() == 3 and targets.dim() == 2:
        # Pool outputs to match static targets
        outputs = outputs[:, 1:, :].mean(dim=1) if outputs.size(1) > 1 else outputs.mean(dim=1)

    # Scalar regression convenience: allow targets [B] for outputs [B, 1]
    if outputs.dim() == 2 and targets.dim() == 1 and outputs.size(1) == 1:
        targets = targets[:, None]

    if outputs.shape != targets.shape:
        raise ValueError(f"mse shape mismatch after alignment: outputs.shape={tuple(outputs.shape)} targets.shape={tuple(targets.shape)}")
    return F.mse_loss(outputs, targets)


def mse_gradient_cutoff(outputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """MSE loss that cuts off gradients from all but the final timestep.

    This prevents gradients from propagating through the entire sequence,
    only allowing gradient flow through the final timestep computation.

    Args:
        outputs: Model outputs - either pooled [batch, features] or
                 full sequence [batch, seq_len+1, features]
        targets: Target values [batch, features]

    """
    if outputs.dim() == 3:
        # Full sequence case
        # Create tensor with gradient cutoff
        outputs_cutoff = torch.zeros_like(outputs)

        # Detach all timesteps except the final one
        outputs_cutoff[:, :-1, :] = outputs[:, :-1, :].detach()  # No gradients for early timesteps
        outputs_cutoff[:, -1, :] = outputs[:, -1, :]  # Keep gradients for final timestep

        # Use final timestep for loss computation
        final_output = outputs_cutoff[:, -1, :]
        return F.mse_loss(final_output, targets)
    # Already pooled - use as-is
    return F.mse_loss(outputs, targets)


def final_timestep_zero_mse(model: nn.Module) -> torch.Tensor:
    """Mean squared error between the final timestep states of the last layer and zero.

    This loss bypasses any time pooling by reading the raw state history cached on
    the LightningModule and only penalizes the last timestep (t = T).

    Args:
        model: The SOEN LightningModule (provides `latest_final_state`).

    Returns:
        Scalar tensor: MSE(final_state[T], 0).

    """
    # Access the last layer's full time history stored by the wrapper in forward()
    final_state = getattr(model, "latest_final_state", None)

    # If unavailable or malformed, this is a bug: the wrapper should cache histories.
    if final_state is None or not isinstance(final_state, torch.Tensor) or final_state.ndim != 3:
        raise RuntimeError(
            "final_timestep_zero_mse requires `model.latest_final_state` as a [B, T+1, D] tensor. "
            "Ensure the Lightning wrapper caches outputs during forward()."
        )

    # Take only the final timestep (shape: [batch, features])
    final_timestep_state = final_state[:, -1, :]

    # Pull final states toward zero
    zero_target = torch.zeros_like(final_timestep_state)
    return F.mse_loss(final_timestep_state, zero_target)


def ensure_post_sample_decay(
    model: nn.Module,
    *,
    extra_zero_step: bool = False,
    num_zero_steps: int = 50,
) -> torch.Tensor:
    """Penalise g at the last timestep to be zero by recomputing g in a differentiable way.

    - Aggregates across all SingleDendrite layers present, averaging their MAE-to-zero
      at the final timestep.
    - Rebuilds φ using upstream connections, φ offset, and internal_J, all at t_last.
    - Uses the layer's effective bias rule via `_get_effective_bias_current`.
    - If ``extra_zero_step=True``, also evaluates a hypothetical extra step with zero
      external upstream input (still including φ offset and internal_J) and averages
      the two g values before computing MAE to zero.
    """
    core = getattr(model, "model", None)
    if core is None or not hasattr(core, "layers"):
        raise RuntimeError("ensure_post_sample_decay requires a LightningModule with a `.model` (SOENModelCore) attached.")

    final_state = getattr(model, "latest_final_state", None)  # [B, T+1, D_last]
    all_states = getattr(model, "latest_all_states", None)  # list of [B, T+1, D_i]
    if not isinstance(final_state, torch.Tensor) or not isinstance(all_states, (list, tuple)):
        raise RuntimeError(
            "ensure_post_sample_decay requires `latest_final_state` and `latest_all_states` cached on the LightningModule."
        )

    try:
        B, T1, _ = final_state.shape
        if T1 <= 1:
            return final_state.new_tensor(0.0, requires_grad=True)
        t_last = T1 - 1

        # Build a map from layer_id to index
        layer_ids = [cfg.layer_id for cfg in core.layers_config]
        id_to_idx = {lid: i for i, lid in enumerate(layer_ids)}

        total_sum = final_state.new_tensor(0.0)
        for idx, layer in enumerate(core.layers):
            if not isinstance(layer, SingleDendriteLayer):
                continue

            s_hist = all_states[idx]
            if not isinstance(s_hist, torch.Tensor) or s_hist.ndim != 3 or s_hist.size(1) <= t_last:
                continue

            s_curr = s_hist[:, t_last, :]  # [B, D_L]
            D_L = s_curr.size(-1)

            # Gather constants/params
            phi_off = getattr(layer, "phi_offset", 0.0)
            J_int = getattr(layer, "internal_J", None)
            if J_int is not None:
                try:
                    J_int_eff = layer.apply_qat_ste_to_weight(J_int)
                except Exception:
                    J_int_eff = J_int

            dt = getattr(layer, "dt", 1.0)
            gp = getattr(layer, "gamma_plus", 0.0)
            gm = getattr(layer, "gamma_minus", 0.0)
            dt_t = s_curr.new_tensor(dt)
            gp_t = s_curr.new_tensor(gp) if not torch.is_tensor(gp) else gp.to(s_curr)
            gm_t = s_curr.new_tensor(gm) if not torch.is_tensor(gm) else gm.to(s_curr)
            gp_t = torch.clamp(gp_t, min=0.0)
            gm_t = torch.clamp(gm_t, min=0.0)

            steps = int(num_zero_steps) if isinstance(num_zero_steps, int) else 50
            steps = max(0, steps)

            # If no extra zeros requested, compute g at real last timestep (with full external phi)
            if steps == 0:
                # Build phi at t_last using upstream connections + offset + internal
                phi = torch.zeros(B, D_L, device=s_curr.device, dtype=s_curr.dtype)
                target_lid = layer_ids[idx]
                for name, J in core.connections.items():
                    if not name.startswith("J_"):
                        continue
                    try:
                        parts = name.split("_")
                        from_lid = int(parts[1])
                        to_lid = int(parts[3])
                    except Exception:
                        continue
                    if to_lid != target_lid:
                        continue
                    src_idx = id_to_idx.get(from_lid)
                    if src_idx is None or src_idx >= len(all_states):
                        continue
                    s_from = all_states[src_idx]
                    if not isinstance(s_from, torch.Tensor) or s_from.ndim != 3 or s_from.size(1) <= t_last:
                        continue
                    phi = phi + torch.matmul(s_from[:, t_last, :], J.t())
                # offset
                if isinstance(phi_off, torch.Tensor):
                    phi = phi + (phi_off.view(1, 1).expand(B, D_L) if phi_off.ndim == 0 else phi_off.view(1, -1))
                else:
                    phi = phi + s_curr.new_tensor(float(phi_off)).view(1, 1).expand(B, D_L)
                # internal
                if J_int is not None:
                    phi = phi + torch.matmul(s_curr, J_int_eff.t())
                eff_bias = layer._get_effective_bias_current(s_curr, None)  # type: ignore[operator]
                g_final = layer.source_function.g(phi, squid_current=eff_bias)  # type: ignore[union-attr,operator]
                total_sum = total_sum + g_final.sum()
                continue

            # Simulate steps of zero external input; include offset and internal each step
            g_step = None
            for _ in range(steps):
                # Build zero-external phi
                phi_zero = torch.zeros(B, D_L, device=s_curr.device, dtype=s_curr.dtype)
                if isinstance(phi_off, torch.Tensor):
                    phi_zero = phi_zero + (phi_off.view(1, 1).expand(B, D_L) if phi_off.ndim == 0 else phi_off.view(1, -1))
                else:
                    phi_zero = phi_zero + s_curr.new_tensor(float(phi_off)).view(1, 1).expand(B, D_L)
                if J_int is not None:
                    phi_zero = phi_zero + torch.matmul(s_curr, J_int_eff.t())

                eff_bias_zero = layer._get_effective_bias_current(s_curr, None)  # type: ignore[operator]
                g_step = layer.source_function.g(phi_zero, squid_current=eff_bias_zero)  # type: ignore[union-attr,operator]

                # forward-euler update
                ds = gp_t * g_step - gm_t * s_curr
                s_curr = s_curr + dt_t * ds

            # use g at last appended step
            if g_step is not None:
                total_sum = total_sum + g_step.sum()

        return total_sum

    except Exception as e:
        raise RuntimeError(f"ensure_post_sample_decay failed: {e}") from e


def gravity_quantization_loss(
    model: nn.Module,
    codebook: list[float] | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    num_levels: int | None = None,
    levels: int | None = None,
    bits: int | None = None,
    factor: float = 1.0,
    connections: list[str] | None = None,
    mode: str = "mae",
) -> torch.Tensor:
    """Compute gravity quantization loss that pulls weights towards quantization levels.

    This loss encourages model weights to cluster around discrete quantization levels,
    reducing the performance impact of post-training quantization. Uses the same
    quantization level generation algorithm as the robustness tool for consistency.

    All learnable connections (both inter-layer like J_0_to_1 and intra-layer like
    internal_1) are accessed directly from model.connections.

    Args:
        model (nn.Module): The model containing weights to quantize.
        codebook (list, optional): Explicit list of quantization levels. If provided,
            other codebook generation parameters are ignored.
        min_val (float, optional): Minimum value for automatic codebook generation.
        max_val (float, optional): Maximum value for automatic codebook generation.
        num_levels (int, optional): Number of levels for automatic codebook generation.
        levels (int, optional): Alias for num_levels.
        bits (int, optional): If provided, number of quantization levels is 2**bits.
        factor (float): Scaling factor for the loss.
        connections (list[str] | None): Optional subset of connection names to include.
            If provided, only learnable parameters for these connections will contribute.
        mode (str): Error mode between weights and nearest codebook value. One of
            {"mae", "mse", "l1", "l2"}. Defaults to "mae".

    Returns:
        torch.Tensor: The computed gravity quantization loss.

    Examples:
        # Using explicit codebook
        loss = gravity_quantization_loss(model, codebook=[-1, -0.5, 0, 0.5, 1])

        # Using automatic codebook generation (matches robustness tool exactly)
        loss = gravity_quantization_loss(model, min_val=-0.24, max_val=0.24, num_levels=17)

        # For 3-bit quantization: num_levels = 2^3 = 8
        # For 2-bit quantization: num_levels = 2^2 = 4

    """
    # Access the SOEN model core
    soen_core: nn.Module | None = None
    if hasattr(model, "connections") and hasattr(model, "layers"):
        # Direct SOENModelCore
        soen_core = model
    elif hasattr(model, "soen_model"):
        # Wrapper exposing core
        soen_core = model.soen_model  # type: ignore[assignment]
    elif hasattr(model, "soen_core"):
        # Some other wrapper with soen_core
        soen_core = model.soen_core  # type: ignore[assignment]
    elif hasattr(model, "model") and hasattr(model.model, "connections") and hasattr(model.model, "layers"):
        # LightningModule where the core is stored directly under `.model`
        soen_core = model.model  # type: ignore[assignment]
    elif hasattr(model, "model") and hasattr(model.model, "soen_model"):
        # Legacy: Lightning wrapper that stored core under model.soen_model
        soen_core = model.model.soen_model  # type: ignore[assignment]
    elif hasattr(model, "model") and hasattr(model.model, "soen_core"):
        # Lightning wrapper -> direct SOEN core
        soen_core = model.model.soen_core  # type: ignore[assignment]
    else:
        # Debug information for troubleshooting
        model_attrs = [attr for attr in dir(model) if not attr.startswith("_")]
        model_type = type(model).__name__
        error_msg = (
            f"Could not find SOEN model core in {model_type}. "
            f"Available attributes: {model_attrs[:10]}{'...' if len(model_attrs) > 10 else ''}. "
            f"Expected model to have 'connections' and 'layers' attributes (SOENModelCore), "
            f"or 'soen_model'/'soen_core' attribute, or be wrapped in a structure that contains it."
        )
        raise ValueError(error_msg)

    # Generate codebook if not provided
    if codebook is None:
        # Determine effective number of levels from aliases
        effective_num_levels = None
        if num_levels is not None:
            effective_num_levels = int(num_levels)
        elif levels is not None:
            effective_num_levels = int(levels)
        elif bits is not None:
            # Guard bits -> levels conversion
            b = int(bits)
            if b < 0:
                msg = "bits must be non-negative"
                raise ValueError(msg)
            # Bits count EXCLUDES zero; always include a single zero level
            effective_num_levels = (2**b) + 1

        if min_val is None or max_val is None or effective_num_levels is None:
            codebook = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]  # Default
        else:
            # Use the robustness tool's algorithm (the only method now)
            codebook = _generate_uniform_codebook(min_val, max_val, effective_num_levels)

    # Normalize mode and validate
    mode_normalized = (mode or "mae").lower()
    if mode_normalized == "l1":
        mode_normalized = "mae"
    if mode_normalized == "l2":
        mode_normalized = "mse"
    if mode_normalized not in {"mae", "mse"}:
        msg = f"gravity_quantization_loss: invalid mode '{mode}'. Expected one of 'mae', 'mse', 'l1', 'l2'."
        raise ValueError(
            msg,
        )

    if soen_core is None:
        raise RuntimeError("gravity_quantization_loss requires `model.model` (SOENModelCore) attached to the LightningModule.")

    # Convert codebook to tensor
    codebook_tensor = torch.tensor(codebook, dtype=torch.float32, device=next(soen_core.parameters()).device)

    total_error_sum = torch.tensor(0.0, device=codebook_tensor.device)
    total_elements = 0

    # Determine targeting behavior
    target_connection_names = set(connections) if connections is not None else None

    # Process connections directly from model.connections
    # Always include ONLY learnable (requires_grad) params so that the resulting loss
    # participates in autograd. If a target list is provided but none are learnable,
    # we will raise a helpful error below.
    if hasattr(soen_core, "connections"):
        non_learnable_requested: list[str] = []
        for connection_name, param in soen_core.connections.items():  # type: ignore[union-attr,operator]
            # Skip connections not in the target list if one is provided
            if target_connection_names is not None and connection_name not in target_connection_names:
                continue
            # Operate only on learnable params
            if not param.requires_grad:
                if target_connection_names is not None:
                    non_learnable_requested.append(connection_name)
                continue

            # Flatten the parameter tensor
            weights_flat = param.view(-1)

            # For each weight, find the nearest codebook value
            # Expand dimensions for broadcasting: weights [N, 1], codebook [1, K]
            weights_expanded = weights_flat.unsqueeze(1)  # [N, 1]
            codebook_expanded = codebook_tensor.unsqueeze(0)  # [1, K]

            # Compute absolute differences: [N, K]
            abs_diffs = torch.abs(weights_expanded - codebook_expanded)

            # Find minimum distance to codebook for each weight: [N]
            min_distances, _ = torch.min(abs_diffs, dim=1)

            # Accumulate error depending on mode
            if mode_normalized == "mae":
                total_error_sum += torch.sum(min_distances)
            else:  # mse
                total_error_sum += torch.sum(min_distances.pow(2))
            total_elements += weights_flat.numel()

    # Compute mean and apply factor. If no elements were included, raise a helpful error
    if total_elements > 0:
        mean_error = total_error_sum / total_elements
        return factor * mean_error
    # Build a clearer diagnostic to avoid silent zeros that break backward when this
    # is the only active loss.
    available_keys = list(getattr(soen_core, "connections", {}).keys())
    req = sorted(target_connection_names) if target_connection_names is not None else None
    if req is not None:
        if len(req) > 0:
            msg = (
                "gravity_quantization_loss: requested connections contributed no learnable elements. "
                f"Requested={req}, Non-learnable-requested={non_learnable_requested}, Available={available_keys}. "
                "Ensure the specified connections are marked learnable or remove the filter."
            )
            raise ValueError(
                msg,
            )
        msg = "gravity_quantization_loss: empty 'connections' list was provided."
        raise ValueError(
            msg,
        )
    msg = "gravity_quantization_loss: no learnable connections found to quantify. Either mark connections learnable in the config or pass an explicit 'connections' list."
    raise ValueError(
        msg,
    )


def local_expansion_loss(
    model: nn.Module,
    target_value: float = 1.0,
    perturbation_scale: float = 1e-5,
    factor: float = 1.0,
    per_layer: bool = False,
    eps: float = 1e-8,
    use_log_loss: bool = True,
    clamp_min: float = 1e-2,
    clamp_max: float = 1e2,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute local expansion loss for criticality control.

    Measures how perturbations grow/shrink through the network over time by running
    two forward passes: one clean, one with a small fixed perturbation added to the input.
    The loss encourages the expansion ratio σ_t to match target_value (typically 1.0 for criticality).

    This is a discrete-time approximation of the local Lyapunov exponent:
        σ_t = ||h'_t - h_t|| / ||h'_{t-1} - h_{t-1}||

    Where h_t is the clean state and h'_t is the perturbed state.

    **Gradient Flow**: Clean states are detached to serve as a fixed reference. Gradients
    only flow through the perturbed trajectory. This ensures the loss provides a clear
    signal: "adjust parameters so perturbed trajectories maintain the target expansion ratio
    relative to a fixed reference."

    **Important**: This loss function requires that inputs are cached on the model during
    forward pass. The Lightning wrapper should store `model.latest_inputs` during forward().

    Args:
        model: The SOENLightningModule instance
        target_value: Target expansion ratio (1.0 = edge of chaos)
        perturbation_scale: Size of input perturbation relative to input magnitude
        factor: Overall scaling factor for this loss
        per_layer: If True, compute separate ratios per layer; if False, aggregate across layers
        eps: Small constant for numerical stability
        use_log_loss: If True, use log-domain loss (more stable, scale-invariant)
        clamp_min: Minimum value to clamp ratios (prevents log of zero)
        clamp_max: Maximum value to clamp ratios (prevents extreme outliers)
        **kwargs: Additional parameters (ignored, for compatibility with training loop)

    Returns:
        torch.Tensor: Scalar loss value

    Example:
        In training config:
        ```yaml
        training:
          loss:
            losses:
              - name: cross_entropy
                weight: 1.0
              - name: local_expansion_loss
                weight: 0.1
                params:
                  target_value: 1.0
                  perturbation_scale: 1.0e-5
                  use_log_loss: true
        ```

    Note on loss formulations:
        - use_log_loss=True (default, recommended):
            loss = mean(log(clamp(σ, min, max))^2)
            Scale-invariant, symmetric, smooth gradients

        - use_log_loss=False (legacy):
            loss = mean((σ - target)^2)
            Raw MSE, less stable for extreme ratios

    """
    # Get cached inputs from the most recent forward pass
    inputs = getattr(model, "latest_inputs", None)
    if inputs is None:
        raise RuntimeError(
            "local_expansion_loss requires `model.latest_inputs` cached from the most recent forward pass. "
            "Ensure the Lightning wrapper caches inputs during forward()."
        )

    # Access the SOEN core
    soen_core = getattr(model, "model", None)
    if soen_core is None or not hasattr(soen_core, "layers"):
        raise RuntimeError("local_expansion_loss requires `model.model` to be a SOENModelCore with `.layers`.")

    # Get cached clean states from most recent forward and DETACH them
    # Clean states should be a fixed reference, not part of the optimization
    all_states_clean = getattr(model, "latest_all_states", None)
    if all_states_clean is None or len(all_states_clean) == 0:
        raise RuntimeError(
            "local_expansion_loss requires `model.latest_all_states` cached from the most recent forward pass."
        )

    # Detach clean states so gradients only flow through perturbed trajectory
    all_states_clean_detached = [s.detach() for s in all_states_clean]

    # Create fixed perturbation: same offset across all timesteps, scaled by input magnitude
    with torch.no_grad():
        input_scale = inputs.abs().mean() + eps
        perturbation = torch.randn_like(inputs) * perturbation_scale * input_scale

    # Run perturbed forward pass in eval mode to disable stochastic layers
    # This ensures we measure the deterministic dynamics, not dropout noise
    was_training = model.training
    model.eval()

    try:
        # Run perturbed forward with gradients enabled
        # Gradients will only flow through the perturbed trajectory since clean is detached
        inputs_perturbed = inputs + perturbation

        # Forward through the model
        _ = model.forward(inputs_perturbed)
        all_states_perturbed = getattr(model, "latest_all_states", None)

        if all_states_perturbed is None or len(all_states_perturbed) == 0:
            raise RuntimeError(
                "local_expansion_loss: perturbed forward did not populate `latest_all_states` (unexpected)."
            )

        # Compute expansion ratios
        # For each layer, compute ||h'_t - h_t|| / ||h'_{t-1} - h_{t-1}||
        ratios = []

        for _layer_idx, (states_clean, states_pert) in enumerate(zip(all_states_clean_detached, all_states_perturbed, strict=True)):
            # states shape: [batch, time+1, features]
            # Skip t=0 (initial state) for ratio computation
            if states_clean.shape[1] <= 2:  # Need at least 2 timesteps for ratio
                continue

            # Compute distances at each timestep (starting from t=1)
            # states_clean is detached, so gradients only flow through states_pert
            diffs = states_pert[:, 1:, :] - states_clean[:, 1:, :]  # [B, T, D]
            distances = torch.norm(diffs, dim=2)  # [B, T]

            # Compute ratios: dist[t] / dist[t-1]
            distances_prev = distances[:, :-1] + eps  # [B, T-1]
            distances_curr = distances[:, 1:]  # [B, T-1]

            layer_ratios = distances_curr / distances_prev  # [B, T-1]

            if per_layer:
                ratios.append(layer_ratios)
            else:
                ratios.append(layer_ratios.flatten())

    finally:
        # Restore training mode
        model.train(was_training)

        # Restore clean states (original, not detached)
        model.latest_all_states = all_states_clean
        # Restore clean inputs
        model.latest_inputs = inputs

    if not ratios:
        raise RuntimeError("local_expansion_loss: no ratios computed (insufficient timesteps or missing state histories).")

    # Aggregate ratios and compute loss
    if use_log_loss:
        # Log-domain loss: more stable and scale-invariant
        # Treats σ=0.5 and σ=2.0 symmetrically (both are log(2) away from 1.0)
        if per_layer:
            layer_losses = []
            for layer_ratio in ratios:
                # Clamp to prevent log(0) and extreme outliers
                clamped = torch.clamp(layer_ratio, min=float(clamp_min), max=float(clamp_max))
                # Log of ratio should be near log(target_value)
                log_ratios = torch.log(clamped)
                log_target = torch.log(torch.tensor(target_value, device=clamped.device, dtype=clamped.dtype))
                layer_losses.append(((log_ratios - log_target) ** 2).mean())
            total_loss = torch.stack(layer_losses).mean()
        else:
            all_ratios = torch.cat(ratios)
            clamped = torch.clamp(all_ratios, min=float(clamp_min), max=float(clamp_max))
            log_ratios = torch.log(clamped)
            log_target = torch.log(torch.tensor(target_value, device=clamped.device, dtype=clamped.dtype))
            total_loss = ((log_ratios - log_target) ** 2).mean()
    # Raw MSE loss (legacy, less stable)
    elif per_layer:
        layer_losses = []
        for layer_ratio in ratios:
            deviation = layer_ratio - target_value
            layer_losses.append((deviation ** 2).mean())
        total_loss = torch.stack(layer_losses).mean()
    else:
        all_ratios = torch.cat(ratios)
        deviation = all_ratios - target_value
        total_loss = (deviation ** 2).mean()

    # Scale by factor (convert to tensor to ensure proper dtype/device)
    return total_loss * total_loss.new_tensor(factor)


def average_state_magnitude_loss(
    model: nn.Module,
    target_magnitude: float = 0.5,
    loss_type: str = "mse",
    factor: float = 1.0,
    layer_ids: list[int] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Encourages average state magnitude across layers to match a target.

    **PyTorch Only**: This loss is NOT supported in JAX due to infrastructure limitations.

    This loss drives network "aliveness" by controlling the average magnitude of
    states. Useful for preventing dead networks (too small) or saturated networks
    (too large).

    Computes mean(|states|) across all specified layers, batch, time, and neurons,
    then penalizes deviation from target_magnitude.

    Args:
        model: The SOENLightningModule instance
        target_magnitude: Target average |state| value (default: 0.5)
        loss_type: "mse" (default) or "mae" for loss computation
        factor: Overall scaling factor for this loss
        layer_ids: Optional list of layer IDs to include (default: all layers)
        **kwargs: Additional parameters (ignored, for compatibility)

    Returns:
        torch.Tensor: Scalar loss value

    Example:
        ```yaml
        training:
          loss:
            losses:
              - name: cross_entropy
                weight: 1.0
              - name: average_state_magnitude_loss
                weight: 0.01
                params:
                  target_magnitude: 0.5
                  loss_type: mse
        ```

    """
    # Access cached state histories
    all_states = getattr(model, "latest_all_states", None)
    if all_states is None or len(all_states) == 0:
        raise RuntimeError("average_state_magnitude_loss requires `model.latest_all_states` cached from the most recent forward pass.")

    # Get SOEN core for layer IDs
    soen_core = getattr(model, "model", None)
    if soen_core is None:
        raise RuntimeError("average_state_magnitude_loss requires `model.model` (SOENModelCore) attached to the LightningModule.")

    # Build layer ID map
    layer_id_list = [cfg.layer_id for cfg in soen_core.layers_config]

    # Filter layers if specific IDs requested
    if layer_ids is not None:
        layer_id_set = set(layer_ids)
        states_to_use = [
            states for idx, states in enumerate(all_states)
            if idx < len(layer_id_list) and layer_id_list[idx] in layer_id_set
        ]
    else:
        states_to_use = all_states

    if not states_to_use:
        raise ValueError("average_state_magnitude_loss: no states selected (layer_ids filter excluded all layers?).")

    # Compute average magnitude across all layers, batch, time, neurons
    magnitudes = []
    for states in states_to_use:
        # states: [B, T+1, D]
        # Skip initial state at t=0, use t=1 onwards
        if states.shape[1] > 1:
            states_active = states[:, 1:, :]  # [B, T, D]
        else:
            states_active = states  # Use what we have

        # Compute magnitude
        mag = torch.abs(states_active)
        magnitudes.append(mag.flatten())

    # Concatenate all magnitudes
    all_magnitudes = torch.cat(magnitudes)

    # Compute mean magnitude
    mean_magnitude = all_magnitudes.mean()

    # Compute loss
    deviation = mean_magnitude - target_magnitude

    if loss_type.lower() in ["mae", "l1"]:
        loss = torch.abs(deviation)
    else:  # mse, l2, or default
        loss = deviation ** 2

    return factor * loss


def _generate_uniform_codebook(min_val: float, max_val: float, num_levels: int) -> list[float]:
    """Generate a quantization codebook using the same algorithm as the robustness tool.

    This ensures that the training loss uses identical quantization levels to those used
    in robustness studies, providing consistency between training and evaluation.
    Zero is always included as one of the quantization levels.

    Args:
        min_val (float): Minimum value in the codebook.
        max_val (float): Maximum value in the codebook.
        num_levels (int): Total number of quantization levels.

    Returns:
        list: Sorted list of quantization levels (identical to robustness tool).

    """
    import torch

    device = "cpu"
    dtype = torch.float32

    # Always produce exactly num_levels unique values with at most one zero.
    # If zero lies in [min_val, max_val], include it ONCE and distribute the remaining (num_levels-1)
    # levels evenly on both sides. If zero is outside the range, still include 0 and distribute the
    # remaining levels across [min_val, max_val].

    if num_levels <= 0:
        return []

    include_zero = True
    remaining = max(0, num_levels - (1 if include_zero else 0))

    if remaining == 0:
        quantization_levels = torch.tensor([0.0], device=device, dtype=dtype)
    else:
        if min_val <= 0.0 <= max_val:
            # Split remaining across negatives and positives
            neg_count = remaining // 2
            pos_count = remaining - neg_count
            neg = torch.linspace(min_val, 0.0, neg_count + 1, device=device, dtype=dtype)[:-1] if neg_count > 0 else torch.tensor([], device=device, dtype=dtype)
            pos = torch.linspace(0.0, max_val, pos_count + 1, device=device, dtype=dtype)[1:] if pos_count > 0 else torch.tensor([], device=device, dtype=dtype)
            quantization_levels = torch.cat([neg, torch.tensor([0.0], device=device, dtype=dtype), pos])
        else:
            # Zero outside the range: include it and distribute remaining across the range
            spread = torch.linspace(min_val, max_val, remaining, device=device, dtype=dtype)
            quantization_levels = torch.cat([spread, torch.tensor([0.0], device=device, dtype=dtype)])

        # Ensure uniqueness and sorted order (guard against accidental duplicates)
        quantization_levels = torch.unique(torch.sort(quantization_levels)[0])
        # If uniqueness reduced the count, pad by adjusting extremes slightly to maintain count
        while quantization_levels.numel() < num_levels:
            # Add a tiny perturbation near the extremes to keep count stable but practically same
            eps = 1e-12
            quantization_levels = torch.cat(
                [
                    torch.tensor([min_val - eps], device=device, dtype=dtype),
                    quantization_levels,
                    torch.tensor([max_val + eps], device=device, dtype=dtype),
                ]
            )
            quantization_levels = torch.unique(torch.sort(quantization_levels)[0])

    return quantization_levels.cpu().numpy().tolist()


def branching_loss(
    model: nn.Module,
    target_sigma: float = 1.0,
    factor: float = 1.0,
    use_log_domain: bool = True,
    clamp_min: float = 1e-1,
    clamp_max: float = 1e1,
    eps_scale: float = 1e-6,
) -> torch.Tensor:
    """Stable branching loss. By default uses log-domain squared error:
        loss = mean( log(clamp(sigma, clamp_min, clamp_max))^2 ) * factor
    If use_log_domain=False, falls back to classic (sigma - target_sigma)^2 averaged.

    Notes:
    - Only layers with both inbound and outbound connections are considered.
    - Neurons with no incoming or outgoing connections are excluded.
    - phi_offset counts as an input source for neurons.
    - No labels needed - operates on internal state histories.
    """
    # Import from metrics to avoid DRY violation
    from soen_toolkit.utils.metrics import compute_branching_ratios_tensor

    # Resolve SOEN core and histories
    soen_core = getattr(model, "model", None)
    all_states = getattr(model, "latest_all_states", None)
    if soen_core is None or all_states is None or len(all_states) == 0:
        raise RuntimeError(
            "branching_loss requires `model.model` (SOENModelCore) and `model.latest_all_states` from the most recent forward pass."
        )

    # Get raw ratios tensor from metrics module
    ratios = compute_branching_ratios_tensor(soen_core, all_states, eps_scale=eps_scale)

    if ratios is None:
        raise RuntimeError("branching_loss: compute_branching_ratios_tensor returned None (unexpected).")

    # Apply loss transformation
    if use_log_domain:
        clamped = torch.clamp(ratios, min=float(clamp_min), max=float(clamp_max))
        log_vals = torch.log(clamped)
        base_loss = (log_vals**2).mean()
    else:
        # classic (target_sigma - sigma)^2 averaged over all elements
        base_loss = ((ratios - target_sigma) ** 2).mean()

    factor_val = float(factor)
    return base_loss * base_loss.new_tensor(factor_val)
