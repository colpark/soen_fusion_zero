from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp


@dataclass
class HeavisideStateDepJAX:
    """JAX implementation of the state‑dependent smooth Heaviside fit used in SOEN.

    Mirrors soen_toolkit.core.source_functions.heaviside.HeavisideFitStateDep.g
    but in pure JAX. Only supports the (phi, squid_current) interface and does not
    provide coefficient support or lookup tables.
    """

    A: float = 0.37091212
    B: float = 0.31903101
    C: float = 1.06435066
    K: float = 1.92138556
    M: float = 2.50322787
    N: float = 2.62706077
    epsilon: float = 1e-6
    uses_squid_current: bool = True

    def g(self, phi: jax.Array, *, squid_current: jax.Array | None = None) -> jax.Array:
        """Evaluate the state‑dependent Heaviside nonlinearity.

        Args:
            phi: Flux input (same shape as state). Shape [..., D]
            squid_current: Total current through the SQUID part of the dendrite circuit. If None, defaults to 1.7 broadcast.

        Returns:
            g(phi, squid_current) with the same shape as phi.

        """
        if squid_current is None:
            squid_current = jnp.ones_like(phi) * 1.7

        bias_diff = jnp.clip(squid_current - self.C, a_min=self.epsilon)
        # Add epsilon to cos_term before fractional power to avoid NaN gradients at cos_term=0
        cos_term = jnp.abs(jnp.cos(jnp.pi * phi)) + self.epsilon
        disc = self.A * (bias_diff**self.K) - self.B * (cos_term**self.M)
        activation = jax.nn.sigmoid(100.0 * disc)
        return activation * (jnp.clip(disc, a_min=self.epsilon) ** (1.0 / self.N))


__all__ = ["HeavisideStateDepJAX"]


@dataclass
class RateArrayJAX:
    """JAX implementation of the RateArray source function using bilinear interpolation.

    Mirrors soen_toolkit.core.source_functions.rate_array.RateArraySource.g
    with a pure-JAX interpolation kernel. If arrays are not provided, attempts
    to load the default base table via soen_toolkit.utils.paths.BASE_RATE_ARRAY_PATH.
    """

    ib_list: jax.Array | None = None
    phi_array: jax.Array | None = None
    g_table: jax.Array | None = None  # shape [H_phi, W_ib]
    data_path: str | None = None
    uses_squid_current: bool = True

    def __post_init__(self):
        if self.g_table is not None and self.ib_list is not None and self.phi_array is not None:
            return
        # Move file I/O and pickle out of any JIT/tracing by using host callbacks
        import numpy as np

        def _host_load(path_str: str):
            from soen_toolkit.utils.paths import BASE_RATE_ARRAY_PATH

            p = path_str or BASE_RATE_ARRAY_PATH
            data_bytes = Path(p).read_bytes()
            data = pickle.loads(data_bytes)
            return (
                np.asarray(data["ib_list"], dtype=np.float32),
                np.asarray(data["phi_array"], dtype=np.float32),
                np.asarray(data["g_array"], dtype=np.float32),
            )

        # Ensure this runs outside of tracing (called at model.prepare())
        ib_np, phi_np, g_np = _host_load(self.data_path or "")
        self.ib_list = jnp.asarray(ib_np)
        self.phi_array = jnp.asarray(phi_np)
        self.g_table = jnp.asarray(g_np)

    def g(self, phi: jax.Array, *, squid_current: jax.Array | None = None) -> jax.Array:
        if squid_current is None:
            squid_current = jnp.ones_like(phi) * 1.7
        # Ensure tables exist
        assert self.g_table is not None
        assert self.ib_list is not None
        assert self.phi_array is not None
        g_table = self.g_table

        # Fold phi to [0, 0.5] (periodic, even symmetry)
        phi_mod = jnp.mod(phi, 1.0)
        phi_eff = jnp.minimum(phi_mod, 1.0 - phi_mod)

        # Normalize to grid_sample-style coords with align_corners=True
        # For phi in [0, 0.5] → norm_phi in [-1, 1]
        norm_phi = 4.0 * phi_eff - 1.0

        # For ib in [ib_min, ib_max] → norm_ib in [-1, 1]
        ib_min = self.ib_list[0]
        ib_max = self.ib_list[-1]
        norm_ib = 2.0 * ((squid_current - ib_min) / jnp.maximum(ib_max - ib_min, 1e-12)) - 1.0

        # grid_sample takes grid[..., 0] as x (width, W) and grid[..., 1] as y (height, H)
        # Torch builds grid=(norm_phi, norm_ib). To match exactly, map:
        #   x <- phi, across W dimension; y <- ib, across H dimension
        H = g_table.shape[0]  # rows correspond to ib samples
        W = g_table.shape[1]  # cols correspond to phi samples
        x_idx = (norm_phi + 1.0) * 0.5 * (W - 1)
        y_idx = (norm_ib + 1.0) * 0.5 * (H - 1)

        # Out-of-range mask to mimic padding_mode='zeros' in torch.grid_sample
        in_x = jnp.logical_and(norm_phi >= -1.0, norm_phi <= 1.0)
        in_y = jnp.logical_and(norm_ib >= -1.0, norm_ib <= 1.0)
        in_range = jnp.logical_and(in_x, in_y)

        # Clamp and compute neighbors (used only for in-range samples)
        x_idx = jnp.clip(x_idx, 0.0, W - 1.0)
        y_idx = jnp.clip(y_idx, 0.0, H - 1.0)
        j0 = jnp.floor(x_idx).astype(jnp.int32)
        i0 = jnp.floor(y_idx).astype(jnp.int32)
        j1 = jnp.minimum(j0 + 1, W - 1)
        i1 = jnp.minimum(i0 + 1, H - 1)
        tx = x_idx - j0.astype(x_idx.dtype)
        ty = y_idx - i0.astype(y_idx.dtype)

        # Gather corners
        def gather(ii, jj):
            return g_table[ii, jj]

        g00 = gather(i0, j0)
        g10 = gather(i0, j1)
        g01 = gather(i1, j0)
        g11 = gather(i1, j1)

        # Bilinear blend (first along x/width, then along y/height)
        top = g00 * (1.0 - tx) + g10 * tx
        bottom = g01 * (1.0 - tx) + g11 * tx
        val = top * (1.0 - ty) + bottom * ty

        # Zero-out out-of-range samples to mirror torch.grid_sample padding
        return val * in_range.astype(val.dtype)


@dataclass
class TanhJAX:
    def g(self, phi: jax.Array, *, squid_current: jax.Array | None = None) -> jax.Array:
        return jnp.tanh(phi)


__all__ = ["HeavisideStateDepJAX", "RateArrayJAX", "TanhJAX"]


@dataclass
class SimpleGELUJAX:
    """JAX implementation of SimpleGELU to match core SimpleGELUSourceFunction:
    g(phi) = 0.5 * phi * tanh(0.8 * phi) + 0.5 * phi.
    """

    def g(self, phi: jax.Array, *, squid_current: jax.Array | None = None) -> jax.Array:
        return 0.5 * phi * jnp.tanh(0.8 * phi) + 0.5 * phi


@dataclass
class TeLUJAX:
    """JAX implementation of TeLU to match core TeLUSourceFunction:
    g(phi) = phi * tanh(exp(phi)).
    """

    def g(self, phi: jax.Array, *, squid_current: jax.Array | None = None) -> jax.Array:
        return phi * jnp.tanh(jnp.exp(phi))


@dataclass
class ReLUJAX:
    """JAX implementation of ReLU nonlinearity."""

    def g(self, phi: jax.Array, *, squid_current: jax.Array | None = None) -> jax.Array:
        return jax.nn.relu(phi)


__all__ = ["HeavisideStateDepJAX", "RateArrayJAX", "ReLUJAX", "SimpleGELUJAX", "TanhJAX", "TeLUJAX"]


@dataclass
class TanhGauss1p7IBFitJAX:
    """JAX version of TanhGauss1p7IBFitSourceFunction.

    Assumes squid_current in [bias_center - bias_span, bias_center].
    Maps squid_current -> s via s = bias_center - squid_current, then applies tanh(Gaussian) in phi.
    """

    sigma: float = 0.12632733583450317
    s_power: float = 2.5484097003936768
    amp: float = 0.7845368385314941
    bias_center: float = 1.7
    bias_span: float = 1.0
    uses_squid_current: bool = True

    def g(self, phi: jax.Array, *, squid_current: jax.Array | None = None) -> jax.Array:
        if squid_current is None:
            squid_current = jnp.ones_like(phi) * self.bias_center

        # Softly clamp squid_current to valid range [bias_center - bias_span, bias_center]
        # This avoids NaNs and JAX tracer errors from hard exceptions
        bias_low = self.bias_center - self.bias_span
        bias_high = self.bias_center
        bias_clamped = jnp.clip(squid_current, bias_low, bias_high)

        s_val = jnp.clip(self.bias_center - bias_clamped, 0.0, 1.0)
        # Add epsilon to base to avoid NaN gradients for fractional power at 0
        s_term = jnp.clip(1.0 - s_val, a_min=1e-6) ** self.s_power

        phi_mod = jnp.mod(phi, 1.0)
        gauss = jnp.exp(-((phi_mod - 0.5) ** 2) / (2.0 * self.sigma * self.sigma))
        phi_term = jnp.tanh(gauss)
        return self.amp * s_term * phi_term


__all__ = [
    "HeavisideStateDepJAX",
    "RateArrayJAX",
    "ReLUJAX",
    "SimpleGELUJAX",
    "TanhJAX",
    "TeLUJAX",
    "TanhGauss1p7IBFitJAX",
]
