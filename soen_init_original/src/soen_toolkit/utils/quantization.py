"""Shared quantization helpers to keep training-time losses, runtime metrics and
robustness tooling perfectly in sync.

Semantics:
- When specifying ``bits``, the total number of levels is ``2**bits + 1``.
  This reserves one explicit zero level in addition to the non-zero levels.
- Codebooks always contain exactly ``num_levels`` unique, sorted values and
  include a single zero level regardless of whether zero lies within the range
  [min_val, max_val].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable


def calculate_num_levels(*, bits: int | None = None, levels: int | None = None) -> int:
    """Resolve total codebook size from ``bits`` or explicit ``levels``.

    - If ``bits`` is provided, return ``2**bits + 1`` (non-zero levels plus a single 0).
    - If ``levels`` is provided, return it as-is.
    - If both or neither are provided, raise a ValueError.
    """
    if bits is not None and levels is not None:
        msg = "Specify either 'bits' or 'levels', not both"
        raise ValueError(msg)

    if bits is not None:
        b = int(bits)
        if b < 0:
            msg = "bits must be non-negative"
            raise ValueError(msg)
        return (2**b) + 1

    if levels is not None:
        num_levels = int(levels)
        if num_levels <= 0:
            msg = "levels must be a positive integer"
            raise ValueError(msg)
        return num_levels

    msg = "Must specify either 'bits' or 'levels'"
    raise ValueError(msg)


def generate_uniform_codebook(
    min_val: float,
    max_val: float,
    num_levels: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a uniform codebook with a single zero level.

    This function is the single source of truth for codebook creation across:
    - gravity quantization loss (training)
    - quantized accuracy metric (callback)
    - robustness tool (weight quantization)
    """
    if num_levels <= 0:
        return torch.tensor([], device=device, dtype=dtype)

    # Always include a single zero; place remaining levels evenly on both sides
    include_zero = True
    remaining = max(0, num_levels - (1 if include_zero else 0))

    if remaining == 0:
        levels_t = torch.tensor([0.0], device=device, dtype=dtype)
    else:
        if min_val <= 0.0 <= max_val:
            neg_count = remaining // 2
            pos_count = remaining - neg_count

            neg = torch.linspace(min_val, 0.0, neg_count + 1, device=device, dtype=dtype)[:-1] if neg_count > 0 else torch.tensor([], device=device, dtype=dtype)
            pos = torch.linspace(0.0, max_val, pos_count + 1, device=device, dtype=dtype)[1:] if pos_count > 0 else torch.tensor([], device=device, dtype=dtype)
            levels_t = torch.cat([neg, torch.tensor([0.0], device=device, dtype=dtype), pos])
        else:
            # Zero is outside range: still include it and distribute the rest across the range
            spread = torch.linspace(min_val, max_val, remaining, device=device, dtype=dtype)
            levels_t = torch.cat([spread, torch.tensor([0.0], device=device, dtype=dtype)])

        # Ensure uniqueness and target count; sort for consistency
        levels_t = torch.unique(torch.sort(levels_t)[0])

        # If uniqueness reduced the count (e.g. min/max equal to zero), pad near extremes
        # We may add two elements per iteration; stop once count >= num_levels then trim.
        eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-12
        while levels_t.numel() < num_levels:
            levels_t = torch.cat(
                [
                    torch.tensor([min_val - eps], device=device, dtype=dtype),
                    levels_t,
                    torch.tensor([max_val + eps], device=device, dtype=dtype),
                ]
            )
            levels_t = torch.unique(torch.sort(levels_t)[0])

        if levels_t.numel() > num_levels:
            # Trim symmetrically from the ends to hit the exact target size
            excess = levels_t.numel() - num_levels
            if excess > 0:
                trim_left = excess // 2
                trim_right = excess - trim_left
                levels_t = levels_t[trim_left : levels_t.numel() - trim_right]

    return levels_t


def build_codebook_from_params(
    *,
    min_val: float,
    max_val: float,
    bits: int | None = None,
    levels: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Helper to compute a codebook from min/max and either bits or levels."""
    num_levels = calculate_num_levels(bits=bits, levels=levels)
    return generate_uniform_codebook(min_val, max_val, num_levels, device=device, dtype=dtype)


def quantize_connections_in_place(
    soen_model: object,
    codebook: torch.Tensor,
    *,
    connections: Iterable[str] | None = None,
    include_non_learnable: bool = False,
) -> None:
    """Snap connection weights to nearest codebook values in-place.

    - soen_model is expected to provide a ``connections`` mapping of name -> tensor/Parameter
    - If ``connections`` is provided, only those keys are quantized
    - If ``include_non_learnable`` is False, only parameters with ``requires_grad=True`` are quantized
    """
    target_names = set(connections) if connections is not None else None

    for name, param in getattr(soen_model, "connections", {}).items():
        if not isinstance(param, torch.Tensor):
            continue
        if target_names is not None and name not in target_names:
            continue
        if not include_non_learnable and hasattr(param, "requires_grad") and not param.requires_grad:
            continue

        with torch.no_grad():
            flat = param.view(-1)
            diffs = (flat.unsqueeze(1) - codebook.to(param.device).unsqueeze(0)).abs()
            idx = diffs.argmin(dim=1)
            snapped = codebook.to(param.device)[idx].view_as(param)
            param.copy_(snapped)


def list_quantizable_connection_names(
    soen_model: object,
    *,
    connections: Iterable[str] | None = None,
    include_non_learnable: bool = False,
) -> list[str]:
    """Return the list of connection names that would be quantized.

    Applies the same filtering rules as ``quantize_connections_in_place`` without
    mutating any tensors.
    """
    target_names = set(connections) if connections is not None else None
    result: list[str] = []
    for name, param in getattr(soen_model, "connections", {}).items():
        if not isinstance(param, torch.Tensor):
            continue
        if target_names is not None and name not in target_names:
            continue
        if not include_non_learnable and hasattr(param, "requires_grad") and not param.requires_grad:
            continue
        result.append(name)
    return result


def snapshot_connection_tensors(soen_model: object, names: Iterable[str]) -> dict[str, torch.Tensor]:
    """Clone and return a snapshot of the specified connection tensors."""
    snapshots: dict[str, torch.Tensor] = {}
    for name in names:
        param = getattr(soen_model, "connections", {}).get(name, None)
        if isinstance(param, torch.Tensor):
            snapshots[name] = param.detach().clone()
    return snapshots


def restore_connection_tensors(soen_model: object, snapshots: dict[str, torch.Tensor]) -> None:
    """Restore connection tensors from a snapshot created by ``snapshot_connection_tensors``."""
    with torch.no_grad():
        for name, tensor in snapshots.items():
            if name in getattr(soen_model, "connections", {}):
                getattr(soen_model, "connections", {})[name].data.copy_(tensor)


def snapped_copy(param: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Return a snapped copy of ``param`` to nearest values in ``codebook`` (no mutation)."""
    flat = param.view(-1)
    cb = codebook.to(param.device)
    diffs = (flat.unsqueeze(1) - cb.unsqueeze(0)).abs()
    idx = diffs.argmin(dim=1)
    return cb[idx].view_as(param)


def snapped_copy_stochastic(
    param: torch.Tensor,
    codebook: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a stochastically-rounded snapped copy to values in ``codebook``.

    For each element x, identify neighbouring codebook values v_lo <= x <= v_hi and
    choose v_hi with probability p = (x - v_lo) / (v_hi - v_lo), else choose v_lo.
    Elements outside the codebook range, or where v_lo == v_hi, snap deterministically.
    """
    flat = param.view(-1)
    cb = codebook.to(param.device)
    if cb.numel() == 0:
        return param.clone()

    # Find bracketing indices in the sorted codebook
    j = torch.bucketize(flat, cb, right=False)
    lo = (j - 1).clamp(min=0, max=cb.numel() - 1)
    hi = j.clamp(min=0, max=cb.numel() - 1)

    v_lo = cb[lo]
    v_hi = cb[hi]

    same = (lo == hi) | (v_hi == v_lo)
    denom = v_hi - v_lo
    p_hi = torch.zeros_like(flat)
    nonzero = ~same
    p_hi[nonzero] = ((flat[nonzero] - v_lo[nonzero]) / denom[nonzero]).clamp(0.0, 1.0)

    # Use explicit torch.rand with shape/device/dtype for generator compatibility across PyTorch versions
    if generator is None:
        rnd = torch.rand_like(flat)
    else:
        rnd = torch.rand(flat.shape, device=flat.device, dtype=flat.dtype, generator=generator)

    choose_hi = (rnd < p_hi) & nonzero
    idx = torch.where(choose_hi, hi, lo)
    return cb[idx].view_as(param)


def compute_quantization_deltas(
    soen_model: object,
    codebook: torch.Tensor,
    names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Compute simple delta stats between current tensors and their snapped versions.

    Returns per-connection metrics: mean_abs, max_abs.
    """
    stats: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name in names:
            param = getattr(soen_model, "connections", {}).get(name, None)
            if not isinstance(param, torch.Tensor):
                continue
            snapped = snapped_copy(param, codebook)
            diff = (snapped - param).abs()
            stats[name] = {
                "mean_abs": float(diff.mean().item()) if diff.numel() > 0 else 0.0,
                "max_abs": float(diff.max().item()) if diff.numel() > 0 else 0.0,
            }
    return stats


def ste_snap(
    param: torch.Tensor,
    codebook: torch.Tensor,
    *,
    stochastic: bool = False,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Straight-through estimator for quantization to a given codebook.

    Forward: returns a snapped tensor (deterministic nearest neighbour by default,
    or stochastic rounding between neighbouring codebook values when ``stochastic=True``).
    Backward: passes gradients as identity w.r.t. the original ``param``.

    Implemented via the common trick: x_q = x + (snap(x) - x).detach().
    """
    if not isinstance(param, torch.Tensor):
        return param
    if codebook is None:
        return param
    if stochastic:
        snapped = snapped_copy_stochastic(param, codebook, generator=rng)
    else:
        snapped = snapped_copy(param, codebook)
    return param + (snapped - param).detach()
