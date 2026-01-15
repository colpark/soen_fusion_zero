from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from soen_toolkit.core.protocols import ModelProtocol

LossFn = Callable[[nn.Module, Any], torch.Tensor]


def _resolve_connection_keys(model: ModelProtocol | nn.Module, connection_keys: Sequence[str] | None) -> list[str]:
    m_any = cast(Any, model)
    masks = getattr(m_any, "connection_masks", {})
    if connection_keys is None:
        keys = list(masks.keys())
    else:
        keys = list(connection_keys)
    if not keys:
        msg = "No connection keys provided and model.connection_masks is empty."
        raise ValueError(msg)
    missing = [k for k in keys if k not in getattr(m_any, "connections", {})]
    if missing:
        msg = f"Missing connections for keys: {missing}"
        raise KeyError(msg)
    return keys


def _hvp(loss: torch.Tensor, params: Sequence[torch.Tensor], vectors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=False)
    grad_dot = sum(torch.sum(g * v) for g, v in zip(grads, vectors, strict=True))
    if isinstance(grad_dot, float) and grad_dot == 0.0:
        return [torch.zeros_like(p) for p in params]
    hvps = torch.autograd.grad(cast(torch.Tensor, grad_dot), params, retain_graph=False, allow_unused=False)
    return [hvp.detach() for hvp in hvps]


def estimate_fisher_diag(
    model: ModelProtocol | nn.Module,
    loss_fn: LossFn,
    data_loader: Iterable[Any],
    *,
    connection_keys: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Estimate diagonal Fisher information (squared gradients) for connection weights.

    This only requires first derivatives and works for models where second derivatives
    are not implemented (e.g., models using grid_sampler).
    Fisher diagonal approximates Hessian as F_ii = E[g_i^2].
    """
    m_any = cast(Any, model)
    keys = _resolve_connection_keys(model, connection_keys)
    params = [m_any.connections[k] for k in keys]

    diag_accum = {k: torch.zeros_like(p) for k, p in zip(keys, params, strict=True)}

    previous_mode = m_any.training
    m_any.eval()
    prev_requires_grad = {k: p.requires_grad for k, p in zip(keys, params, strict=True)}
    for p in params:
        p.requires_grad_(True)

    num_batches = 0
    try:
        for batch in data_loader:
            num_batches += 1
            loss = loss_fn(cast(nn.Module, model), batch)
            if loss.dim() != 0:
                msg = f"loss_fn must return a scalar. Got shape {tuple(loss.shape)}"
                raise ValueError(msg)

            grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=False)
            for key, g in zip(keys, grads, strict=True):
                diag_accum[key] = diag_accum[key] + g.detach().pow(2)

        if num_batches == 0:
            raise ValueError("Data loader produced zero batches during Fisher estimation.")

        for key in keys:
            diag_accum[key] = diag_accum[key] / num_batches
    finally:
        m_any.train(previous_mode)
        for key, p in zip(keys, params, strict=True):
            if not prev_requires_grad[key]:
                p.requires_grad_(False)

    return diag_accum


def estimate_hessian_diag_hutchinson(
    model: ModelProtocol | nn.Module,
    loss_fn: LossFn,
    data_loader: Iterable[Any],
    *,
    connection_keys: Sequence[str] | None = None,
    num_samples: int = 8,
) -> dict[str, torch.Tensor]:
    """Estimate diagonal Hessian entries for connection weights via Hutchinson."""
    m_any = cast(Any, model)
    keys = _resolve_connection_keys(model, connection_keys)
    params = [m_any.connections[k] for k in keys]

    if num_samples <= 0:
        msg = "num_samples must be positive for Hutchinson estimation."
        raise ValueError(msg)

    diag_accum = {k: torch.zeros_like(p) for k, p in zip(keys, params, strict=True)}

    previous_mode = m_any.training
    m_any.eval()
    # Temporarily enable gradients on target params (restore afterward)
    prev_requires_grad = {k: p.requires_grad for k, p in zip(keys, params, strict=True)}
    for p in params:
        p.requires_grad_(True)
    expected_batches: int | None = None
    try:
        for sample_idx in range(num_samples):
            vectors = [
                (torch.randint_like(p, low=0, high=2) * 2 - 1).to(device=p.device, dtype=p.dtype)
                for p in params
            ]
            sample_batches = 0
            sample_hv_sum = {k: torch.zeros_like(p) for k, p in zip(keys, params, strict=True)}

            for batch in data_loader:
                sample_batches += 1
                loss = loss_fn(cast(nn.Module, model), batch)
                if loss.dim() != 0:
                    msg = f"loss_fn must return a scalar. Got shape {tuple(loss.shape)}"
                    raise ValueError(msg)
                hvp_vals = _hvp(loss, params, vectors)
                for key, hv in zip(keys, hvp_vals, strict=True):
                    sample_hv_sum[key] = sample_hv_sum[key] + hv

            if sample_batches == 0:
                raise ValueError("Data loader produced zero batches during curvature estimation.")

            if expected_batches is None:
                expected_batches = sample_batches
            elif expected_batches != sample_batches:
                msg = f"Data loader batch count changed between samples: {expected_batches} vs {sample_batches}"
                raise ValueError(msg)

            for key, vec in zip(keys, vectors, strict=True):
                diag_accum[key] = diag_accum[key] + vec * sample_hv_sum[key]

        if expected_batches is None:
            raise ValueError("Data loader produced zero batches during curvature estimation.")

        normalizer = float(num_samples * expected_batches)
        for key in keys:
            diag_accum[key] = diag_accum[key] / normalizer
    finally:
        m_any.train(previous_mode)
        for key, p in zip(keys, params, strict=True):
            if not prev_requires_grad[key]:
                p.requires_grad_(False)

    return diag_accum


def compute_obd_saliencies(
    model: ModelProtocol | nn.Module,
    hessian_diag: dict[str, torch.Tensor],
    *,
    connection_keys: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute element-wise OBD saliency 0.5 * H_ii * w_i^2."""
    m_any = cast(Any, model)
    keys = _resolve_connection_keys(model, connection_keys)
    saliencies: dict[str, torch.Tensor] = {}
    masks = getattr(m_any, "connection_masks", {})

    for key in keys:
        weight = m_any.connections[key].detach()
        diag = hessian_diag.get(key)
        if diag is None:
            msg = f"Hessian diagonal missing for connection '{key}'."
            raise KeyError(msg)
        if diag.shape != weight.shape:
            msg = f"Hessian diag shape {diag.shape} does not match weight shape {weight.shape} for '{key}'."
            raise ValueError(msg)
        mask = masks.get(key)
        active = torch.ones_like(weight, dtype=weight.dtype, device=weight.device) if mask is None else mask.to(
            device=weight.device, dtype=weight.dtype
        )
        saliencies[key] = 0.5 * diag.to(weight.device, dtype=weight.dtype) * weight.pow(2) * active

    return saliencies


def _reduce_saliency(
    saliency: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    block_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce saliency to blocks; returns (scores, active_mask)."""
    if mode == "element":
        return saliency, mask > 0
    if mode == "row":
        return saliency.sum(dim=1), (mask.sum(dim=1) > 0)
    if mode == "column":
        return saliency.sum(dim=0), (mask.sum(dim=0) > 0)
    if mode == "block":
        if block_shape is None:
            raise ValueError("block_shape is required when mode='block'.")
        rows, cols = block_shape
        if rows <= 0 or cols <= 0:
            raise ValueError("block_shape values must be positive.")
        if saliency.shape[0] % rows != 0 or saliency.shape[1] % cols != 0:
            msg = f"Saliency shape {tuple(saliency.shape)} is not divisible by block_shape {block_shape}."
            raise ValueError(msg)
        row_blocks = saliency.shape[0] // rows
        col_blocks = saliency.shape[1] // cols
        sal_view = saliency.view(row_blocks, rows, col_blocks, cols)
        mask_view = mask.view(row_blocks, rows, col_blocks, cols)
        scores = sal_view.sum(dim=(1, 3))
        active = mask_view.sum(dim=(1, 3)) > 0
        return scores, active
    raise ValueError(f"Unknown reduction mode '{mode}'. Use 'element', 'row', 'column', or 'block'.")


def prune_connections_by_saliencies(
    model: ModelProtocol | nn.Module,
    saliencies: dict[str, torch.Tensor],
    *,
    connection_keys: Sequence[str] | None = None,
    target_sparsity: float,
    mode: str = "element",
    block_shape: tuple[int, int] | None = None,
    per_connection: bool = False,
    combine_magnitude_alpha: float | None = None,
    variance_floor: float = 0.0,
) -> dict[str, Any]:
    """Prune connections with lowest saliency. Updates weights and masks in-place."""
    if not 0.0 <= target_sparsity < 1.0:
        msg = f"target_sparsity must be in [0,1). Got {target_sparsity}."
        raise ValueError(msg)
    m_any = cast(Any, model)
    keys = connection_keys if connection_keys is not None else list(saliencies.keys())
    keys = _resolve_connection_keys(model, keys)
    masks = getattr(m_any, "connection_masks", {})
    if not masks:
        raise ValueError("Model has no connection_masks to prune.")

    def _apply_prune(
        key: str,
        threshold_value: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor | None]:
        mask = masks.get(key)
        if mask is None:
            raise KeyError(f"Mask for connection '{key}' is missing.")
        sal = saliencies[key].to(device=mask.device)
        if combine_magnitude_alpha is not None:
            if not 0.0 <= combine_magnitude_alpha <= 1.0:
                raise ValueError("combine_magnitude_alpha must be in [0,1].")
            weight = m_any.connections[key].detach().to(device=mask.device)
            mag = weight.abs()
            active_mask_for_var = mask > 0
            if variance_floor > 0.0:
                active_sal = sal[active_mask_for_var]
                if active_sal.numel() > 0 and torch.var(active_sal) < variance_floor:
                    sal = mag
                else:
                    sal = combine_magnitude_alpha * sal + (1.0 - combine_magnitude_alpha) * mag
            else:
                sal = combine_magnitude_alpha * sal + (1.0 - combine_magnitude_alpha) * mag
        scores, active_mask = _reduce_saliency(sal, mask, mode, block_shape)
        active_scores = scores[active_mask]
        if active_scores.numel() == 0:
            raise ValueError(f"No active connections found for '{key}'.")

        if threshold_value is None:
            num_to_prune_local = int(active_scores.numel() * target_sparsity)
            if num_to_prune_local == 0:
                return 0, None
            threshold_local = torch.topk(active_scores, k=num_to_prune_local, largest=False).values.max()
        else:
            threshold_local = threshold_value.to(device=mask.device)

        if mode == "element":
            flat_active_idx = torch.nonzero(active_mask.reshape(-1), as_tuple=False).flatten()
            active_scores_flat = active_scores.reshape(-1)
            if threshold_value is None:
                # Select exactly num_to_prune_local smallest scores to avoid over-pruning on ties
                num_to_prune_local = int(active_scores_flat.numel() * target_sparsity)
                if num_to_prune_local == 0:
                    return 0, None
                order = torch.argsort(active_scores_flat)
                selected_idx = flat_active_idx[order[:num_to_prune_local]]
                threshold_local = active_scores_flat[order[:num_to_prune_local]].max()
            else:
                # Global mode: keep threshold-based pruning for compatibility
                selected_idx = flat_active_idx[(sal.reshape(-1)[flat_active_idx] <= threshold_local)]
            prune_mask_flat = torch.zeros_like(active_mask.reshape(-1), dtype=torch.bool)
            prune_mask_flat[selected_idx] = True
            prune_mask = prune_mask_flat.view_as(mask)
        elif mode == "row":
            if threshold_value is None:
                num_to_prune_local = int(active_scores.numel() * target_sparsity)
                if num_to_prune_local == 0:
                    return 0, None
                order = torch.argsort(active_scores)
                selected_rows = order[:num_to_prune_local]
                threshold_local = active_scores[selected_rows].max()
                row_prune = torch.zeros_like(active_mask)
                row_prune[selected_rows] = True
                row_prune = row_prune & active_mask
            else:
                row_prune = (scores <= threshold_local) & active_mask
            prune_mask = row_prune.unsqueeze(1).expand_as(mask)
        elif mode == "column":
            if threshold_value is None:
                num_to_prune_local = int(active_scores.numel() * target_sparsity)
                if num_to_prune_local == 0:
                    return 0, None
                order = torch.argsort(active_scores)
                selected_cols = order[:num_to_prune_local]
                threshold_local = active_scores[selected_cols].max()
                col_prune = torch.zeros_like(active_mask)
                col_prune[selected_cols] = True
                col_prune = col_prune & active_mask
            else:
                col_prune = (scores <= threshold_local) & active_mask
            prune_mask = col_prune.unsqueeze(0).expand_as(mask)
        else:  # block
            if block_shape is None:
                raise ValueError("block_shape is required when mode='block'.")
            rows, cols = block_shape  # already validated
            row_blocks = mask.shape[0] // rows
            col_blocks = mask.shape[1] // cols
            block_scores = (scores <= threshold_local) & active_mask
            block_mask = block_scores.unsqueeze(1).unsqueeze(3).expand(row_blocks, rows, col_blocks, cols)
            prune_mask = block_mask.reshape_as(mask)

        pruned_count = int(prune_mask.sum().item())
        updated_mask = mask * (~prune_mask).to(mask.dtype)
        masks[key] = updated_mask
        weight = m_any.connections[key]
        weight.data = weight.data * updated_mask.to(device=weight.device, dtype=weight.dtype)
        return pruned_count, threshold_local

    thresholds: dict[str, torch.Tensor | None] = {}
    pruned_total = 0

    if per_connection:
        with torch.no_grad():
            for key in keys:
                pruned_count, threshold_local = _apply_prune(key, None)
                thresholds[key] = threshold_local
                pruned_total += pruned_count
    else:
        flat_scores = []
        for key in keys:
            mask = masks.get(key)
            if mask is None:
                raise KeyError(f"Mask for connection '{key}' is missing.")
            sal = saliencies[key].to(device=mask.device)
            if combine_magnitude_alpha is not None:
                weight = m_any.connections[key].detach().to(device=mask.device)
                mag = weight.abs()
                active_mask_for_var = mask > 0
                if variance_floor > 0.0:
                    active_sal = sal[active_mask_for_var]
                    if active_sal.numel() > 0 and torch.var(active_sal) < variance_floor:
                        sal = mag
                    else:
                        sal = combine_magnitude_alpha * sal + (1.0 - combine_magnitude_alpha) * mag
                else:
                    sal = combine_magnitude_alpha * sal + (1.0 - combine_magnitude_alpha) * mag
            scores, active_mask = _reduce_saliency(sal, mask, mode, block_shape)
            active_scores = scores[active_mask]
            if active_scores.numel() == 0:
                raise ValueError(f"No active connections found for '{key}'.")
            flat_scores.append(active_scores.reshape(-1))

        all_scores = torch.cat(flat_scores)
        active_total = all_scores.numel()
        if active_total == 0:
            raise ValueError("No active connections available for pruning.")
        num_to_prune = int(active_total * target_sparsity)
        if num_to_prune == 0:
            return {"threshold": None, "pruned": 0}

        global_threshold = torch.topk(all_scores, k=num_to_prune, largest=False).values.max()

        with torch.no_grad():
            for key in keys:
                pruned_count, _ = _apply_prune(key, global_threshold)
                thresholds[key] = global_threshold
                pruned_total += pruned_count

    if hasattr(m_any, "apply_masks"):
        m_any.apply_masks()

    result: dict[str, Any] = {"threshold": thresholds if per_connection else global_threshold, "pruned": pruned_total}
    return result


def post_training_prune(
    model: ModelProtocol | nn.Module,
    loss_fn: LossFn,
    data_loader: Iterable[Any],
    *,
    connection_keys: Sequence[str] | None = None,
    target_sparsity: float = 0.2,
    num_samples: int = 8,
    mode: str = "element",
    block_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """End-to-end post-training pruning for connection masks."""
    diag = estimate_hessian_diag_hutchinson(
        model,
        loss_fn,
        data_loader,
        connection_keys=connection_keys,
        num_samples=num_samples,
    )
    saliency = compute_obd_saliencies(model, diag, connection_keys=connection_keys)
    prune_result = prune_connections_by_saliencies(
        model,
        saliency,
        connection_keys=connection_keys,
        target_sparsity=target_sparsity,
        mode=mode,
        block_shape=block_shape,
    )
    final_result: dict[str, Any] = {
        "hessian_diag": diag,
        "saliency": saliency,
        "prune_result": prune_result,
    }
    return final_result
