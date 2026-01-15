from __future__ import annotations

import jax
import jax.numpy as jnp
import torch

from soen_toolkit.ops.spike import spike_jax, spike_torch
from soen_toolkit.ops.surrogates import SurrogateSpec


def test_spike_torch_triangle_surrogate_grad() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float32, requires_grad=True)
    thr = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    y = spike_torch(x, threshold=thr, surrogate=SurrogateSpec(kind="triangle", width=1.0, scale=1.0))
    y.sum().backward()

    # Triangle derivative: (1 - |x|/w)/w for |x|<=w else 0; with w=1 => 1-|x|
    expected = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0], dtype=torch.float32)
    assert x.grad is not None
    assert torch.allclose(x.grad, expected, atol=1e-6)
    assert thr.grad is not None
    assert torch.allclose(thr.grad, -expected.sum(), atol=1e-6)


def test_spike_jax_triangle_surrogate_grad() -> None:
    x = jnp.asarray([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=jnp.float32)

    def f(x_):
        y = spike_jax(x_, threshold=0.0, surrogate=SurrogateSpec(kind="triangle", width=1.0, scale=1.0))
        return jnp.sum(y)

    g = jax.grad(f)(x)
    expected = jnp.asarray([0.0, 0.5, 1.0, 0.5, 0.0], dtype=jnp.float32)
    assert jnp.allclose(g, expected, atol=1e-6)

