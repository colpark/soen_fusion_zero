from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import optax

LOSS_REGISTRY: dict[str, Callable[..., jnp.ndarray]] = {}


def register_loss(name: str) -> Callable[[Callable[..., jnp.ndarray]], Callable[..., jnp.ndarray]]:
    def decorator(fn: Callable[..., jnp.ndarray]) -> Callable[..., jnp.ndarray]:
        LOSS_REGISTRY[name] = fn
        return fn

    return decorator


@register_loss("cross_entropy")
def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray, *, ignore_index: int | None = -100) -> jnp.ndarray:
    """Cross entropy over classes for classification.

    Uses optax for numerical stability to match PyTorch's F.cross_entropy behavior.

    Args:
        logits: [B, C] unnormalized scores
        targets: [B] int32/64 class ids
        ignore_index: Optional sentinel value to skip (matches PyTorch default ``-100``).
    Returns:
        scalar loss

    """
    logits = jnp.asarray(logits)
    targets = jnp.asarray(targets, dtype=jnp.int32)

    if ignore_index is not None:
        num_classes = logits.shape[-1]
        valid_mask = jnp.logical_and(targets != ignore_index, targets >= 0)
        valid_mask = jnp.logical_and(valid_mask, targets < num_classes)
        # Replace ignored indices with zero to keep gather in range
        safe_targets = jnp.where(valid_mask, targets, 0)
        loss_ce = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=safe_targets,
        )
        loss_ce = loss_ce * valid_mask.astype(loss_ce.dtype)
        valid_count = jnp.sum(valid_mask.astype(loss_ce.dtype))
        return jnp.where(valid_count > 0, jnp.sum(loss_ce) / valid_count, jnp.asarray(0.0, dtype=loss_ce.dtype))

    loss_ce = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=targets,
    )
    return jnp.mean(loss_ce)


@register_loss("mse")
def mse_loss(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    diff = preds - targets
    return jnp.mean(jnp.square(diff))


@register_loss("final_timestep_zero_mse")
def final_timestep_zero_mse(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    MSE loss between the final timestep output and zero tensor.
    Args:
        logits: [B, T, D] or [B, D]
        targets: unused (for signature compatibility)
    Returns:
        scalar loss
    """
    if logits.ndim == 2:
        final = logits
    else:
        final = logits[:, -1, :]
    zeros = jnp.zeros_like(final)
    return jnp.mean(jnp.square(final - zeros))


@register_loss("mse_aligned")
def mse_aligned(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared error over aligned dimensions (fail-fast on incompatible shapes).
    Args:
        logits: [B, ...]
        targets: [B, ...] or [B, 1] or [B]
    Returns:
        scalar loss
    """
    x = jnp.asarray(logits)
    y = jnp.asarray(targets)

    # Common SOEN alignment rules (mirrors metrics.mse and Torch-side behavior):
    # - If x [B, T+1, D] and y [B, T, D] -> drop first output timestep
    # - If x [B, D] and y [B, T, D] -> pool y over time
    # - If x [B, T, D] and y [B, D] -> pool x over time (drop t=0 if present)
    if x.ndim == 3 and y.ndim == 3:
        if x.shape[1] == y.shape[1] + 1:
            x = x[:, 1:, :]
    elif x.ndim == 2 and y.ndim == 3:
        y = jnp.mean(y, axis=1)
    elif x.ndim == 3 and y.ndim == 2:
        if x.shape[1] > 1:
            x = jnp.mean(x[:, 1:, :], axis=1)
        else:
            x = jnp.mean(x, axis=1)

    # Scalar regression convenience: allow y [B] for x [B, 1]
    if x.ndim == 2 and y.ndim == 1 and x.shape[1] == 1:
        y = y[:, None]

    if x.shape != y.shape:
        raise ValueError(f"mse_aligned shape mismatch: logits.shape={tuple(x.shape)} targets.shape={tuple(y.shape)}")

    diff = x - y
    return jnp.mean(jnp.square(diff))


@register_loss("l1")
def l1_loss(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    L1 loss (mean absolute error) between preds and targets.
    Args:
        preds: [B, ...]
        targets: [B, ...]
    Returns:
        scalar loss
    """
    return jnp.mean(jnp.abs(preds - targets))


@register_loss("nll")
def nll_loss(log_probs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Negative log likelihood loss for classification, given log-probabilities.
    Args:
        log_probs: [B, C] log-probabilities
        targets: [B] class indices
    Returns:
        scalar loss
    """
    n = log_probs.shape[0]
    idx = jnp.arange(n)
    return -jnp.mean(log_probs[idx, targets])


@register_loss("gap_loss")
def gap_loss(logits: jnp.ndarray, targets: jnp.ndarray, margin: float = 1.0) -> jnp.ndarray:
    """
    Gap loss for classification: penalize if the correct class's score is not greater than other classes by a margin.
    Args:
        logits: [B, C] unnormalized model scores
        targets: [B] integer class indices
        margin: margin value (default 1.0)
    Returns:
        scalar loss
    """
    B, C = logits.shape
    idx = jnp.arange(B)

    # Gather scores for target classes: shape [B, 1] to match PyTorch broadcasting
    target_logits = logits[idx, targets].reshape(-1, 1)  # [B, 1]

    # Calculate penalties for all classes: shape [B, C]
    # This calculates penalties for all classes, where the penalty for the correct class
    # will always be `margin`, while penalties for incorrect classes will be correct.
    penalties = jnp.maximum(0.0, logits - target_logits + margin)

    # Sum all penalties for each sample. This sum includes the unwanted `margin`
    # that was added for the correct class.
    total_penalty_per_sample = jnp.sum(penalties, axis=1)

    # Subtract the known incorrect penalty that was applied to the target class
    correct_penalty = total_penalty_per_sample - margin

    return jnp.mean(correct_penalty)


@register_loss("local_expansion_loss")
def local_expansion_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    """Local expansion loss is NOT supported in JAX.

    This loss requires running dual forward passes (clean + perturbed) and accessing
    internal state histories, which the current JAX training infrastructure does not support.

    The JAX trainer only passes (logits, targets, **config_params) to loss functions,
    with no mechanism to inject runtime-computed arrays like state histories.

    Use PyTorch backend for criticality control via local_expansion_loss.

    Raises:
        NotImplementedError: Always, to prevent silent failures

    """
    raise NotImplementedError(
        "local_expansion_loss is NOT supported in JAX backend. "
        "This loss requires dual forward passes and state history access, "
        "which the JAX training infrastructure does not provide. "
        "Please use PyTorch backend (backend: torch) for criticality control. "
        "Remove 'local_expansion_loss' from your JAX training config."
    )


@register_loss("average_state_magnitude_loss")
def average_state_magnitude_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    """Average state magnitude loss is NOT supported in JAX.

    This loss requires access to all layer state histories to compute the average
    magnitude across the entire network. The JAX training infrastructure only provides
    final output logits to loss functions.

    The JAX trainer would only measure output layer magnitudes, which is not comparable
    to the PyTorch version that measures all layers. This creates inconsistent behavior.

    Use PyTorch backend for network aliveness control via average_state_magnitude_loss.

    Raises:
        NotImplementedError: Always, to prevent silent failures

    """
    raise NotImplementedError(
        "average_state_magnitude_loss is NOT supported in JAX backend. "
        "This loss requires access to all layer state histories, "
        "which the JAX training infrastructure does not provide. "
        "The JAX version would only measure output logits (not comparable to PyTorch). "
        "Please use PyTorch backend (backend: torch) for aliveness control. "
        "Remove 'average_state_magnitude_loss' from your JAX training config."
    )


@register_loss("branching_loss")
def branching_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    histories: tuple[jnp.ndarray, ...] | None = None,
    topology=None,
    params: dict | None = None,
    connection_params: jnp.ndarray | None = None,
    internal_connections: dict[int, jnp.ndarray] | None = None,
    layer_ids: list[int] | None = None,
    target_sigma: float = 1.0,
    factor: float = 1.0,
    use_log_domain: bool = True,
    clamp_min: float = 1e-1,
    clamp_max: float = 1e1,
    eps_scale: float = 1e-6,
) -> jnp.ndarray:
    """Differentiable branching loss for JAX.

    This mirrors the Torch definition in `soen_toolkit.utils.metrics.compute_branching_ratios_tensor`:
      BR = sum(output_flux) / sum(input_flux)
    where:
      - output_flux = |s| * sum(|outgoing_weights|)
      - input_flux  = |Σ_in(s_in @ J_in^T) + (optional internal self-loop flux)|

    Notes / differences vs Torch:
    - Dynamic connection modes (WICC/NOCC) are treated as their effective weight matrices
      `connection_params` (static J), which is the best we can do without edge-state traces.
    """
    _ = logits
    _ = targets

    if histories is None or connection_params is None or topology is None:
        raise ValueError(
            "branching_loss requires histories, topology, and connection_params. "
            "This indicates the trainer did not pass the required runtime trace inputs."
        )

    # Optional: include per-layer phi_offset (matches Torch compute_branching_ratios_tensor semantics).
    # We extract phi_offset from params["layer_params"] using topology.layer_kinds.
    layer_params = None
    if params is not None:
        layer_params = params.get("layer_params")

    # ---------------------- helpers ----------------------
    def _soft_abs(x: jnp.ndarray, smoothness: float = 1e-4) -> jnp.ndarray:
        return jnp.sqrt(jnp.square(x) + jnp.asarray(smoothness, dtype=x.dtype))

    # Resolve layer selection
    layer_ids_sorted = getattr(topology, "layer_ids", ())
    layer_dims = getattr(topology, "layer_dims", ())
    layer_kinds = getattr(topology, "layer_kinds", ())
    edge_starts = getattr(topology, "edge_starts", ())
    edge_from_layers = getattr(topology, "edge_from_layers", ())
    edge_dst_dims = getattr(topology, "edge_dst_dims", ())
    edge_src_dims = getattr(topology, "edge_src_dims", ())
    edge_masks = getattr(topology, "edge_masks", None)

    num_layers = len(layer_ids_sorted)
    if num_layers == 0:
        return jnp.asarray(0.0, dtype=connection_params.dtype)

    # Determine which layers are part of the “recurrent core” if not specified:
    # include layers that have both inbound and outbound edges.
    has_inbound = [False] * num_layers
    has_outbound = [False] * num_layers
    # inbound (CSR)
    for li in range(num_layers):
        start = int(edge_starts[li])
        end = int(edge_starts[li + 1])
        has_inbound[li] = end > start
    # outbound
    for e, src_li in enumerate(edge_from_layers):
        has_outbound[int(src_li)] = True

    if layer_ids is None:
        sel_layer_indices = [i for i in range(num_layers) if has_inbound[i] and has_outbound[i]]
    else:
        lid_to_idx = {int(lid): i for i, lid in enumerate(layer_ids_sorted)}
        sel_layer_indices = [lid_to_idx[int(lid)] for lid in layer_ids if int(lid) in lid_to_idx]

    if not sel_layer_indices:
        # No valid layers to score; return a neutral zero term.
        return jnp.asarray(0.0, dtype=connection_params.dtype)

    # Precompute outbound edge lists per layer (python static)
    outbound_edges_by_layer: list[list[int]] = [[] for _ in range(num_layers)]
    for e, src_li in enumerate(edge_from_layers):
        outbound_edges_by_layer[int(src_li)].append(int(e))

    # ---------------------- compute BR ----------------------
    # Import kind codes lazily to avoid module import at file import time.
    from soen_toolkit.utils.port_to_jax.pure_forward import KIND_NONLINEAR, KIND_SINGLEDENDRITE

    ratios_all: list[jnp.ndarray] = []
    for li in sel_layer_indices:
        D = int(layer_dims[li])

        s_L = histories[li]
        # histories are [B, T+1, D]; skip t=0
        if s_L.ndim == 3 and s_L.shape[1] > 1:
            s_L = s_L[:, 1:, :]
        if s_L.ndim != 3:
            continue
        B, T, _D = s_L.shape

        # Build node-level incoming/outgoing masks (based on current J values)
        incoming_mask = jnp.zeros((D,), dtype=jnp.bool_)
        start = int(edge_starts[li])
        end = int(edge_starts[li + 1])
        for e in range(start, end):
            dst_dim = int(edge_dst_dims[e])
            src_dim = int(edge_src_dims[e])
            J = connection_params[e, :dst_dim, :src_dim]
            if edge_masks is not None:
                J = J * edge_masks[e, :dst_dim, :src_dim]
            incoming_mask = incoming_mask.at[:dst_dim].set(
                jnp.logical_or(incoming_mask[:dst_dim], jnp.any(J != 0, axis=1))
            )

        outgoing_mask = jnp.zeros((D,), dtype=jnp.bool_)
        for e in outbound_edges_by_layer[li]:
            dst_dim = int(edge_dst_dims[e])
            src_dim = int(edge_src_dims[e])
            J = connection_params[e, :dst_dim, :src_dim]
            if edge_masks is not None:
                J = J * edge_masks[e, :dst_dim, :src_dim]
            outgoing_mask = outgoing_mask.at[:src_dim].set(
                jnp.logical_or(outgoing_mask[:src_dim], jnp.any(J != 0, axis=0))
            )

        connected_mask = jnp.logical_and(incoming_mask, outgoing_mask)
        num_connected = jnp.sum(connected_mask.astype(connection_params.dtype))
        has_any_connected = num_connected > 0

        # Input flux: sum over inbound edges (and optional internal connectivity)
        phi_in = jnp.zeros((B, T, D), dtype=connection_params.dtype)
        for e in range(start, end):
            src_li = int(edge_from_layers[e])
            dst_dim = int(edge_dst_dims[e])
            src_dim = int(edge_src_dims[e])
            s_src = histories[src_li]
            if s_src.ndim == 3 and s_src.shape[1] > 1:
                s_src = s_src[:, 1:, :]
            J = connection_params[e, :dst_dim, :src_dim]
            if edge_masks is not None:
                J = J * edge_masks[e, :dst_dim, :src_dim]
            phi_add = jnp.matmul(s_src[:, :, :src_dim], J.T)  # [B,T,dst_dim]
            phi_in = phi_in.at[:, :, :dst_dim].add(phi_add)

        # phi_offset counts as an input source (Torch behavior)
        if layer_params is not None and li < len(layer_params) and layer_kinds:
            k = int(layer_kinds[li])
            p = layer_params[li]
            phi_off = None
            if k == KIND_SINGLEDENDRITE:
                # [phi_offset, bias_current, gamma_plus, gamma_minus] -> [4, D]
                if isinstance(p, jnp.ndarray) and p.ndim == 2 and p.shape[0] >= 1:
                    phi_off = p[0, :D]
            elif k == KIND_NONLINEAR:
                # [phi_offset, bias_current] -> [2, D]
                if isinstance(p, jnp.ndarray) and p.ndim == 2 and p.shape[0] >= 1:
                    phi_off = p[0, :D]
            if phi_off is not None:
                if phi_off.ndim == 1:
                    phi_in = phi_in + phi_off[None, None, :]
                elif phi_off.ndim == 2:
                    # Allow [B, D] if ever provided
                    phi_in = phi_in + phi_off[:, None, :D]

        # Internal self-loops (if provided)
        if internal_connections is not None:
            lid = int(layer_ids_sorted[li])
            J_int = internal_connections.get(lid)
            if J_int is not None:
                phi_in = phi_in + jnp.matmul(s_L, J_int.T)

        # Outbound magnitude per source neuron: sum |J| over destinations for each outbound edge
        outbound_mag = jnp.zeros((D,), dtype=connection_params.dtype)
        for e in outbound_edges_by_layer[li]:
            dst_dim = int(edge_dst_dims[e])
            src_dim = int(edge_src_dims[e])
            J = connection_params[e, :dst_dim, :src_dim]
            if edge_masks is not None:
                J = J * edge_masks[e, :dst_dim, :src_dim]
            outbound_mag = outbound_mag.at[:src_dim].add(jnp.sum(jnp.abs(J), axis=0))

        # Mask down to connected neurons
        cm_f = connected_mask.astype(connection_params.dtype)
        phi_in_abs = _soft_abs(phi_in) * cm_f[None, None, :]
        outbound_mag_masked = outbound_mag * cm_f

        # Adaptive epsilon (avoid zero division)
        signal_scale = jnp.maximum(jnp.mean(phi_in_abs), jnp.mean(jnp.abs(s_L)))
        signal_scale = jnp.maximum(signal_scale, jnp.asarray(1e-6, dtype=signal_scale.dtype))
        eps = jnp.asarray(eps_scale, dtype=signal_scale.dtype) * signal_scale
        outbound_mag_safe = outbound_mag_masked + eps * cm_f

        phi_out = _soft_abs(s_L) * outbound_mag_safe[None, None, :]

        phi_in_sum = jnp.sum(phi_in_abs, axis=2)  # [B,T]
        phi_out_sum = jnp.sum(phi_out, axis=2)  # [B,T]

        num_connected_safe = jnp.where(has_any_connected, num_connected, jnp.asarray(1.0, dtype=num_connected.dtype))
        denom = phi_in_sum + eps * num_connected_safe
        ratio_raw = phi_out_sum / denom
        ratio_bt = jnp.where(has_any_connected, ratio_raw, jnp.asarray(target_sigma, dtype=ratio_raw.dtype))
        ratios_all.append(ratio_bt)

    if not ratios_all:
        return jnp.asarray(0.0, dtype=connection_params.dtype)

    ratios = jnp.stack(ratios_all, axis=0)  # [L_sel, B, T]
    if use_log_domain:
        clamped = jnp.clip(ratios, a_min=jnp.asarray(clamp_min, dtype=ratios.dtype), a_max=jnp.asarray(clamp_max, dtype=ratios.dtype))
        log_vals = jnp.log(clamped)
        log_target = jnp.log(jnp.asarray(target_sigma, dtype=log_vals.dtype))
        base = jnp.mean(jnp.square(log_vals - log_target))
    else:
        base = jnp.mean(jnp.square(ratios - jnp.asarray(target_sigma, dtype=ratios.dtype)))

    return jnp.asarray(factor, dtype=base.dtype) * base


@register_loss("multilayer_distillation_mse")
def multilayer_distillation_mse(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    histories: tuple[jnp.ndarray, ...] | None = None,
    topology=None,
    layer_ids: list[int] | None = None,
    include_input_layer: bool = False,
    include_t0: bool = True,
    factor: float = 1.0,
) -> jnp.ndarray:
    """Distill full-network trajectories by concatenating selected layer states.

    Expected target format:
      - `targets` is a single tensor shaped `[B, T+1, D_total]` (or `[B, T, D_total]`),
        where `D_total` is the sum of the selected layers' dims, concatenated in
        **sorted layer_id order**.

    This avoids ragged per-layer tensors while still enabling multi-layer distillation.
    """
    _ = logits
    if histories is None or topology is None:
        raise ValueError("multilayer_distillation_mse requires histories and topology.")

    layer_ids_sorted = getattr(topology, "layer_ids", ())
    num_layers = len(layer_ids_sorted)
    if num_layers == 0:
        return jnp.asarray(0.0, dtype=targets.dtype)

    if layer_ids is None:
        sel = list(range(num_layers))
        if not include_input_layer and sel:
            sel = sel[1:]
    else:
        lid_to_idx = {int(lid): i for i, lid in enumerate(layer_ids_sorted)}
        sel = [lid_to_idx[int(lid)] for lid in layer_ids if int(lid) in lid_to_idx]

    if not sel:
        return jnp.asarray(0.0, dtype=targets.dtype)

    # Build student concatenated trajectory
    parts = []
    for li in sel:
        h = histories[li]
        if not include_t0 and h.ndim == 3 and h.shape[1] > 1:
            h = h[:, 1:, :]
        parts.append(h)
    student = jnp.concatenate(parts, axis=-1)

    # Align time dimension with targets if needed (common: teacher drops/keeps t=0 differently)
    t = jnp.asarray(targets)
    if student.ndim == 3 and t.ndim == 3:
        if student.shape[1] == t.shape[1] + 1:
            student = student[:, 1:, :]
        elif t.shape[1] == student.shape[1] + 1:
            t = t[:, 1:, :]
        # If still mismatched, truncate to min length
        if student.shape[1] != t.shape[1]:
            min_T = min(student.shape[1], t.shape[1])
            student = student[:, :min_T, :]
            t = t[:, :min_T, :]

    # Align feature dimension
    if student.shape[-1] != t.shape[-1]:
        raise ValueError(
            f"multilayer_distillation_mse feature dim mismatch: student={student.shape[-1]} target={t.shape[-1]}. "
            "Ensure targets are concatenated using the same layer selection and order."
        )

    base = jnp.mean(jnp.square(student - t))
    return jnp.asarray(factor, dtype=base.dtype) * base

@register_loss("autoregressive_cross_entropy")
def autoregressive_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Autoregressive cross entropy loss.

    Args:
        logits: [B, T, V] unnormalized scores (T includes t=0)
        targets: [B, T-1] or [B, T] token indices.
                 Note: The trainer passes targets as-is.
                 For AR, we expect targets to be aligned such that:
                 logits[:, t] predicts targets[:, t]

                 However, usually logits includes t=0 (initial state).
                 So logits[:, 1:] should predict targets.

    Returns:
        scalar loss
    """
    # Ensure logits are [B, T, V]
    if logits.ndim != 3:
        raise ValueError(f"autoregressive_cross_entropy expects 3D logits [B, T, V], got {logits.shape}")

    # Align logits and targets
    # Logits usually have length T+1 (including t=0)
    # Targets usually have length T

    # If logits has one more step than targets, assume t=0 is extra
    if logits.shape[1] == targets.shape[1] + 1:
        logits = logits[:, 1:, :]

    # If targets has one more step (e.g. if targets includes t=0 which is invalid), slice it?
    # But usually targets are just token indices.

    # Flatten batch and time dimensions
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)

    # Use optax for stable cross entropy
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)

    return jnp.mean(loss)
