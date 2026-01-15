"""Surrogate gradient registry (Torch + JAX).

We keep surrogate definitions centralized so new surrogates can be added without
touching layer code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import torch

TorchDeriv = Callable[[torch.Tensor, "SurrogateSpec"], torch.Tensor]
JaxDeriv = Callable[[jax.Array, "SurrogateSpec"], jax.Array]


@dataclass(frozen=True, slots=True)
class SurrogateSpec:
    """Configuration for surrogate gradients.

    Notes:
    - `kind` selects the surrogate derivative shape (e.g. triangle).
    - Parameters are interpreted by the registered derivative functions.
    """

    kind: str = "triangle"
    width: float = 1.0
    scale: float = 1.0
    clip: float | None = None


class SurrogateRegistry:
    """Registry mapping surrogate kind -> derivative implementations (Torch + JAX)."""

    def __init__(self) -> None:
        self._torch: dict[str, TorchDeriv] = {}
        self._jax: dict[str, JaxDeriv] = {}

    def register(self, kind: str, *, torch_derivative: TorchDeriv, jax_derivative: JaxDeriv) -> None:
        k = str(kind).strip().lower()
        if not k:
            raise ValueError("Surrogate kind must be a non-empty string.")
        if k in self._torch or k in self._jax:
            raise ValueError(f"Surrogate kind '{k}' is already registered.")
        self._torch[k] = torch_derivative
        self._jax[k] = jax_derivative

    def torch_derivative(self, kind: str) -> TorchDeriv:
        k = str(kind).strip().lower()
        try:
            return self._torch[k]
        except KeyError as e:
            raise KeyError(f"Unknown surrogate kind '{kind}'. Available: {sorted(self._torch.keys())}") from e

    def jax_derivative(self, kind: str) -> JaxDeriv:
        k = str(kind).strip().lower()
        try:
            return self._jax[k]
        except KeyError as e:
            raise KeyError(f"Unknown surrogate kind '{kind}'. Available: {sorted(self._jax.keys())}") from e

    def available(self) -> list[str]:
        # Keep a single source of truth (they're always registered together)
        return sorted(self._torch.keys())


def _clip_torch(x: torch.Tensor, clip: float | None) -> torch.Tensor:
    if clip is None:
        return x
    return torch.clamp(x, min=-float(clip), max=float(clip))


def _clip_jax(x: jax.Array, clip: float | None) -> jax.Array:
    if clip is None:
        return x
    c = float(clip)
    return jnp.clip(x, a_min=-c, a_max=c)


def triangle_derivative_torch(x: torch.Tensor, spec: SurrogateSpec) -> torch.Tensor:
    """Triangle surrogate derivative centered at 0.

    d/dx is non-zero only for |x| <= width.
    """
    w = float(spec.width)
    if w <= 0:
        raise ValueError(f"triangle surrogate requires width > 0, got {w}")
    a = (1.0 - torch.abs(x) / w).clamp_min(0.0)
    out = (float(spec.scale) / w) * a
    return _clip_torch(out, spec.clip)


def triangle_derivative_jax(x: jax.Array, spec: SurrogateSpec) -> jax.Array:
    w = float(spec.width)
    if w <= 0:
        raise ValueError(f"triangle surrogate requires width > 0, got {w}")
    a = jnp.maximum(0.0, 1.0 - jnp.abs(x) / w)
    out = (float(spec.scale) / w) * a
    return _clip_jax(out, spec.clip)


SURROGATES = SurrogateRegistry()
SURROGATES.register("triangle", torch_derivative=triangle_derivative_torch, jax_derivative=triangle_derivative_jax)

__all__ = ["SURROGATES", "SurrogateRegistry", "SurrogateSpec"]

