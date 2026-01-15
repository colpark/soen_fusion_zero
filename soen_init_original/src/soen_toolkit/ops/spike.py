"""Hard spike with surrogate gradients (Torch + JAX)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import torch

from .surrogates import SURROGATES, SurrogateSpec


def _sum_to_torch(grad: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reduce `grad` to match `target`'s shape under broadcasting."""
    if grad.shape == target.shape:
        return grad
    # Sum over leading axes until ranks match
    while grad.dim() > target.dim():
        grad = grad.sum(dim=0)
    # Sum over broadcasted axes (where target has size 1)
    for i, (g, t) in enumerate(zip(grad.shape, target.shape, strict=True)):
        if t == 1 and g != 1:
            grad = grad.sum(dim=i, keepdim=True)
    return grad


def _sum_to_jax(grad: jax.Array, target: jax.Array) -> jax.Array:
    """Reduce `grad` to match `target`'s shape under broadcasting."""
    if grad.shape == target.shape:
        return grad
    # Sum over leading axes until ranks match
    while grad.ndim > target.ndim:
        grad = jnp.sum(grad, axis=0)
    # Sum over broadcasted axes (where target has size 1)
    axes = tuple(i for i, (g, t) in enumerate(zip(grad.shape, target.shape, strict=True)) if t == 1 and g != 1)
    if axes:
        grad = jnp.sum(grad, axis=axes, keepdims=True)
    return grad


class _SpikeTorchFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, threshold: torch.Tensor, surrogate: SurrogateSpec) -> torch.Tensor:
        # Save for backward; surrogate is a small frozen dataclass
        ctx.save_for_backward(x, threshold)
        ctx.surrogate = surrogate
        return (x > threshold).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        surrogate: SurrogateSpec = ctx.surrogate
        deriv = SURROGATES.torch_derivative(surrogate.kind)(x - threshold, surrogate)
        grad_x = grad_out * deriv
        grad_threshold_full = -grad_out * deriv
        grad_threshold = _sum_to_torch(grad_threshold_full, threshold)
        return grad_x, grad_threshold, None


def spike_torch(x: torch.Tensor, *, threshold: torch.Tensor | float = 0.0, surrogate: SurrogateSpec | None = None) -> torch.Tensor:
    """Hard spike with surrogate gradient for training in PyTorch."""
    s = surrogate or SurrogateSpec()
    thr = threshold if isinstance(threshold, torch.Tensor) else torch.tensor(float(threshold), dtype=x.dtype, device=x.device)
    return _SpikeTorchFn.apply(x, thr, s)


def spike_jax(x: jax.Array, *, threshold: jax.Array | float = 0.0, surrogate: SurrogateSpec | None = None) -> jax.Array:
    """Hard spike with surrogate gradient for training in JAX."""
    s = surrogate or SurrogateSpec()
    deriv_fn = SURROGATES.jax_derivative(s.kind)

    thr = threshold if isinstance(threshold, jax.Array) else jnp.asarray(float(threshold), dtype=x.dtype)

    @jax.custom_vjp
    def _spike(x_: jax.Array, thr_: jax.Array) -> jax.Array:
        return (x_ > thr_).astype(x_.dtype)

    def fwd(x_: jax.Array, thr_: jax.Array):
        y = (x_ > thr_).astype(x_.dtype)
        return y, (x_, thr_)

    def bwd(res, g):
        x_, thr_ = res
        d = deriv_fn(x_ - thr_, s)
        grad_x = g * d
        grad_thr_full = -g * d
        grad_thr = _sum_to_jax(grad_thr_full, thr_)
        return (grad_x, grad_thr)

    _spike.defvjp(fwd, bwd)
    return _spike(x, thr)


__all__ = ["spike_jax", "spike_torch"]

