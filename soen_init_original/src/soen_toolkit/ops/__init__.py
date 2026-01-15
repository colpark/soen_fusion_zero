"""Backend-agnostic primitives for non-smooth operations.

This package holds small, reusable building blocks that can be used from both the
Torch and JAX backends.
"""

from .spike import spike_jax, spike_torch
from .surrogates import SURROGATES, SurrogateSpec

__all__ = ["SURROGATES", "SurrogateSpec", "spike_jax", "spike_torch"]

