from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from soen_toolkit.ops.spike import spike_jax
from soen_toolkit.ops.surrogates import SurrogateSpec

from .integrators_jax import ForwardEulerJAX
from .source_functions_jax import HeavisideStateDepJAX, SimpleGELUJAX, TanhJAX, TeLUJAX

if TYPE_CHECKING:
    from collections.abc import Mapping


# Smoothing parameter for soft_abs
_SOFT_ABS_EPS = 0.01


def soft_abs_jax(x: jax.Array, eps: float = _SOFT_ABS_EPS) -> jax.Array:
    """Smooth approximation of abs(x) using sqrt(x² + ε).

    Unlike jnp.abs(), this function has a continuous derivative everywhere,
    which is important for numerical stability in Forward Euler integration.

    Args:
        x: Input array
        eps: Smoothing parameter (default: 0.01)

    Returns:
        Smooth approximation of abs(x)
    """
    return jnp.sqrt(x * x + eps)


@dataclass
class MultiplierParamsJAX:
    phi_y: jax.Array  # [B,D] or broadcastable to [B,D]
    bias_current: jax.Array  # [B,D] or broadcastable
    gamma_plus: jax.Array  # [B,D] or scalar/broadcastable
    gamma_minus: jax.Array  # [B,D] or scalar/broadcastable
    internal_J: jax.Array | None = None  # [D_to, D_from] or [B, D_to, D_from]


def _apply_internal_mask(internal_J: jax.Array | None, internal_mask: jax.Array | None) -> jax.Array | None:
    """Apply structural mask to internal connection matrix if present.

    Args:
        internal_J: Weight matrix [D_to, D_from] or None
        internal_mask: Structural mask [D_to, D_from] or None

    Returns:
        Masked weight matrix or None if internal_J is None

    Raises:
        ValueError: If mask shape doesn't match J shape
    """
    if internal_J is None:
        return None
    if internal_mask is None:
        return internal_J
    if internal_J.shape != internal_mask.shape:
        msg = f"Internal mask shape mismatch: J {internal_J.shape} != mask {internal_mask.shape}"
        raise ValueError(msg)
    return internal_J * internal_mask


def _build_edge_index_jax(internal_J: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Build edge indices from weight matrix for JAX (similar to PyTorch build_edge_index)."""
    # Assume dense connectivity for JAX (mask typically embedded in J via masking)
    to_dim, from_dim = internal_J.shape[-2:]
    dst_idx = jnp.arange(to_dim, dtype=jnp.int32).reshape(-1, 1).repeat(from_dim, axis=1).reshape(-1)
    src_idx = jnp.arange(from_dim, dtype=jnp.int32).reshape(1, -1).repeat(to_dim, axis=0).reshape(-1)
    return src_idx, dst_idx


def _apply_connectivity_jax(
    state: jax.Array,
    phi: jax.Array,
    internal_J: jax.Array | None,
    mode: str = "fixed",
    dynamic_params: dict | None = None,
    edge_state: jax.Array | tuple | None = None,
    dt: float = 1.0,
    source_impl=None,
    internal_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | tuple | None]:
    """Apply internal connectivity with mode-based dispatching.

    Args:
        internal_mask: Structural mask [D_to, D_from] to apply to internal_J if present

    Returns:
        (phi_out, new_edge_state)
    """
    if internal_J is None:
        return phi, None

    mode_norm = str(mode or "fixed").strip().lower()
    if mode_norm in {"dynamic", "dynamic_v1", "v1", "wicc"}:
        mode_norm = "wicc"
    elif mode_norm in {"dynamic_v2", "v2", "nocc"}:
        mode_norm = "nocc"
    elif mode_norm in {"fixed"}:
        mode_norm = "fixed"
    else:
        raise ValueError(
            f"Unknown internal connectivity mode '{mode}'. Supported: fixed, WICC (dynamic/v1), NOCC (dynamic_v2/v2)."
        )

    # Apply mask if present
    if internal_J is None:
        return phi, None
    internal_J = _apply_internal_mask(internal_J, internal_mask)
    if internal_J is None:
        return phi, None

    if mode_norm == "fixed":
        # Static matrix multiplication
        if internal_J.ndim == 3 and internal_J.shape[0] == state.shape[0]:
            phi_out = phi + jnp.einsum("bd,bdo->bo", state, internal_J.transpose(0, 2, 1))
        else:
            phi_out = phi + (state @ internal_J.T)
        return phi_out, None

    if mode_norm == "wicc":
        if source_impl is None:
            raise ValueError(
                "Internal connectivity mode WICC requires a source implementation (source_impl). "
                "Passing None would silently fall back to a generic nonlinearity, which is not allowed."
            )
        # WICC: With Collection Coil (v1 multiplier dynamics)
        params = dynamic_params or {}
        gamma_plus = params.get("gamma_plus", 0.001)
        gamma_minus = params.get("gamma_minus", 0.001)
        bias_current = params.get("bias_current", 2.0)
        j_in = params.get("j_in", 0.38)
        j_out = params.get("j_out", 0.38)
        half_flux_offset = params.get("half_flux_offset", False)

        # Build edge indices
        if internal_J is None:
            return phi, None
        src_idx, dst_idx = _build_edge_index_jax(internal_J)
        E = src_idx.shape[0]
        B = state.shape[0]

        # Initialize edge state if needed
        if edge_state is None or isinstance(edge_state, tuple) or edge_state.shape != (B, E):
            edge_state = jnp.zeros((B, E), dtype=state.dtype)

        # Gather per-edge inputs
        x_e = state[:, src_idx] * j_in  # [B, E]
        phi_y_e = internal_J[dst_idx, src_idx]  # [E]
        phi_y_e = phi_y_e.reshape(1, -1)  # [1, E]

        # Apply half-flux offset if enabled (mirrors PyTorch)
        if half_flux_offset:
            phi_y_e = phi_y_e + 0.5

        # Multiplier dynamics per edge
        phi_a = x_e + phi_y_e
        phi_b = x_e - phi_y_e

        # Mirror PyTorch: check if source uses squid current
        if getattr(source_impl, "uses_squid_current", False):
            squid_a = bias_current - edge_state
            squid_b = bias_current + edge_state
            g_a = source_impl.g(phi_a, squid_current=squid_a)
            g_b = source_impl.g(phi_b, squid_current=squid_b)
        else:
            g_a = source_impl.g(phi_a, squid_current=bias_current)
            g_b = source_impl.g(phi_b, squid_current=bias_current)

        ds_dt = gamma_plus * (g_a - g_b) - gamma_minus * edge_state  # [B, E]
        edge_state_next = edge_state + dt * ds_dt

        # Scatter to destination
        phi_out = jnp.zeros_like(state)
        phi_out = phi_out.at[:, dst_idx].add(edge_state_next * j_out)
        return phi + phi_out, edge_state_next

    if mode_norm == "nocc":
        if source_impl is None:
            raise ValueError(
                "Internal connectivity mode NOCC requires a source implementation (source_impl). "
                "Passing None would silently fall back to a generic nonlinearity, which is not allowed."
            )
        # NOCC: No Collection Coil (v2 multiplier dynamics with dual states)
        params = dynamic_params or {}
        alpha = params.get("alpha", 1.64053)
        beta = params.get("beta", 303.85)
        beta_out = params.get("beta_out", 91.156)
        bias_current = params.get("bias_current", 2.1)
        j_in = params.get("j_in", 0.38)
        j_out = params.get("j_out", 0.38)
        half_flux_offset = params.get("half_flux_offset", False)

        # Build edge indices
        if internal_J is None:
            return phi, None
        src_idx, dst_idx = _build_edge_index_jax(internal_J)
        E = src_idx.shape[0]
        B, D = state.shape

        # Initialize edge states if needed (tuple of s1_e, s2_e, m_d)
        if edge_state is None or not isinstance(edge_state, tuple) or edge_state[0].shape != (B, E):
            edge_state = (
                jnp.zeros((B, E), dtype=state.dtype),
                jnp.zeros((B, E), dtype=state.dtype),
                jnp.zeros((B, D), dtype=state.dtype),
            )

        s1_e, s2_e, m_d = edge_state

        # Gather per-edge inputs
        x_e = state[:, src_idx] * j_in  # [B, E]
        phi_y_e = internal_J[dst_idx, src_idx].reshape(1, -1)  # [1, E]

        # Apply half-flux offset if enabled (mirrors PyTorch)
        if half_flux_offset:
            phi_y_e = phi_y_e + 0.5

        # Compute phi_a and phi_b
        phi_a = x_e + phi_y_e
        phi_b = x_e - phi_y_e

        # Compute source functions (mirror PyTorch logic)
        if getattr(source_impl, "uses_squid_current", False):
            squid_current_1 = bias_current - s1_e
            squid_current_2 = -bias_current + s2_e
            g1 = source_impl.g(phi_a, squid_current=soft_abs_jax(squid_current_1))
            g2 = source_impl.g(phi_b, squid_current=soft_abs_jax(squid_current_2))
        else:
            g1 = source_impl.g(phi_a, squid_current=bias_current)
            g2 = source_impl.g(phi_b, squid_current=-bias_current)

        # Aggregate g per destination node
        g_sum = g1 - g2  # [B, E]
        g_agg = jnp.zeros((B, D), dtype=state.dtype)
        g_agg = g_agg.at[:, dst_idx].add(g_sum)  # [B, D]

        # Compute fan-in per destination
        fan_in = jnp.zeros(D, dtype=jnp.int32)
        fan_in = fan_in.at[dst_idx].add(jnp.ones_like(dst_idx))
        fan_in = fan_in.astype(state.dtype)  # [D]

        # Compute dot_m
        beta_eff_d = beta + 2 * fan_in * beta_out  # [D]
        dot_m = (g_agg - alpha * m_d) / beta_eff_d  # [B, D]

        # Gather dot_m for each edge
        dot_m_e = dot_m[:, dst_idx]  # [B, E]

        # Compute dot_s1 and dot_s2
        dot_s1 = (g1 - beta_out * dot_m_e - alpha * s1_e) / beta
        dot_s2 = (g2 - beta_out * dot_m_e - alpha * s2_e) / beta

        # Update states
        s1_next = s1_e + dt * dot_s1
        s2_next = s2_e + dt * dot_s2
        m_next = m_d + dt * dot_m

        # Output is m state scaled by j_out
        return phi + m_next * j_out, (s1_next, s2_next, m_next)
    raise RuntimeError("Unreachable: internal connectivity mode dispatch failed.")


@dataclass
class MultiplierLayerJAX:
    dim: int
    dt: float
    source: HeavisideStateDepJAX
    internal_mode: str = "fixed"
    internal_dynamic_params: dict | None = None
    internal_mask: jax.Array | None = None  # Structural mask for internal connections

    def dynamics(self, state: jax.Array, phi: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        phi_y = params["phi_y"]
        gamma_plus = params["gamma_plus"]
        gamma_minus = params["gamma_minus"]
        bias_current = params["bias_current"]

        phi_a = phi + phi_y
        phi_b = phi - phi_y

        # squid currents for state‑dependent heaviside
        squid_current_a = bias_current - state
        squid_current_b = bias_current + state
        g_a = self.source.g(phi_a, squid_current=squid_current_a)
        g_b = self.source.g(phi_b, squid_current=squid_current_b)
        return gamma_plus * (g_a - g_b) - gamma_minus * state

    def forward(self, phi: jax.Array, params: MultiplierParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        batch, steps, dim = phi.shape
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)

        internal_J = params.internal_J

        mode_norm = str(self.internal_mode or "fixed").strip().lower()
        if mode_norm in {"dynamic", "dynamic_v1", "v1", "wicc"}:
            mode_norm = "wicc"
        elif mode_norm in {"dynamic_v2", "v2", "nocc"}:
            mode_norm = "nocc"
        elif mode_norm in {"fixed"}:
            mode_norm = "fixed"
        else:
            raise ValueError(
                f"Unknown internal_mode '{self.internal_mode}'. Supported: fixed, WICC (dynamic/v1), NOCC (dynamic_v2/v2)."
            )

        # Use scan to integrate with stateful internal connections
        if mode_norm != "fixed" and internal_J is not None:
            # Dynamic internal connections: initialize edge state based on mode
            if mode_norm == "nocc":
                # V2 uses tuple of (s1_e, s2_e, m_d)
                if internal_J is None:
                    return phi, None
                src_idx, dst_idx = _build_edge_index_jax(internal_J)
                E = src_idx.shape[0]
                edge_state: Any = (
                    jnp.zeros((batch, E), dtype=phi.dtype),
                    jnp.zeros((batch, E), dtype=phi.dtype),
                    jnp.zeros((batch, dim), dtype=phi.dtype),
                )
            else:
                # V1 uses single tensor [B, E]
                if internal_J is None:
                    return phi, None
                src_idx, dst_idx = _build_edge_index_jax(internal_J)
                E = src_idx.shape[0]
                edge_state = jnp.zeros((batch, E), dtype=phi.dtype)

            def scan_step(carry, phi_t):
                state, edge_st = carry
                phi_t_conn, edge_st_next = _apply_connectivity_jax(
                    state,
                    phi_t,
                    internal_J,
                    mode_norm,
                    self.internal_dynamic_params,
                    edge_st,
                    self.dt,
                    self.source,
                    self.internal_mask,
                )
                # Compute dynamics
                ds_dt = self.dynamics(
                    state,
                    phi_t_conn,
                    {
                        "phi_y": params.phi_y,
                        "gamma_plus": params.gamma_plus,
                        "gamma_minus": params.gamma_minus,
                        "bias_current": params.bias_current,
                    },
                )
                state_next = state + self.dt * ds_dt
                return (state_next, edge_st_next), state_next

            (s_final, _), s_hist = jax.lax.scan(scan_step, (s0, edge_state), phi.swapaxes(0, 1))
            return jnp.concatenate([s0[:, None, :], s_hist.swapaxes(0, 1)], axis=1)

        # Fixed mode: use original logic
        def phi_transform(state: jax.Array, phi_t: jax.Array) -> jax.Array:
            phi_out, _ = _apply_connectivity_jax(state, phi_t, internal_J, "fixed", None, None, self.dt, None, self.internal_mask)
            return phi_out

        def dyn_wrapped(state: jax.Array, phi_t: jax.Array, pmap) -> jax.Array:
            phi_t = phi_transform(state, phi_t)
            phi_off = pmap.get("phi_offset", None)
            if phi_off is not None:
                phi_t = phi_t + phi_off
            return self.dynamics(state, phi_t, pmap)

        solver = ForwardEulerJAX(dynamics=dyn_wrapped, dt=self.dt)
        pmap = {
            "phi_y": params.phi_y,
            "bias_current": params.bias_current,
            "gamma_plus": params.gamma_plus,
            "gamma_minus": params.gamma_minus,
        }
        return solver.integrate(s0, phi, pmap)


@dataclass
class SingleDendriteParamsJAX:
    phi_offset: jax.Array  # [B,D] or broadcastable to [B,D]
    bias_current: jax.Array  # [B,D] or broadcastable
    gamma_plus: jax.Array  # [B,D] or scalar/broadcastable
    gamma_minus: jax.Array  # [B,D] or scalar/broadcastable
    internal_J: jax.Array | None = None  # [D_to, D_from] or [B, D_to, D_from]


@dataclass
class SingleDendriteLayerJAX:
    dim: int
    dt: float
    source: HeavisideStateDepJAX
    internal_mode: str = "fixed"
    internal_dynamic_params: dict | None = None
    internal_mask: jax.Array | None = None  # Structural mask for internal connections

    def dynamics(self, state: jax.Array, phi: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        _ = params["phi_offset"]  # Unused but kept for parameter consistency
        gamma_plus = params["gamma_plus"]
        gamma_minus = params["gamma_minus"]
        bias_current = params["bias_current"]

        # phi has already been adjusted by the wrapper (mirrors Torch solver)
        phi_eff = phi
        squid_current = bias_current - state
        g_val = self.source.g(phi_eff, squid_current=squid_current)
        return gamma_plus * g_val - gamma_minus * state

    def step(self, s_prev: jax.Array, phi_t: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Discrete step function for ParaRNN solver.

        Computes s_{t+1} = s_t + dt * ds/dt using Forward Euler discretization.
        This is equivalent to: s_{t+1} = alpha * s_t + beta * g(phi, squid_current)
        where alpha = (1 - dt * gamma_minus) and beta = dt * gamma_plus.

        Args:
            s_prev: Previous state [B, D]
            phi_t: Input flux at current timestep [B, D]
            params: Parameter dict with gamma_plus, gamma_minus, bias_current, phi_offset

        Returns:
            Next state [B, D]
        """
        gamma_plus = params["gamma_plus"]
        gamma_minus = params["gamma_minus"]
        bias_current = params["bias_current"]
        phi_offset = params.get("phi_offset")

        # Apply phi offset if present
        phi_eff = phi_t
        if phi_offset is not None:
            phi_eff = phi_t + phi_offset

        # Compute source function
        squid_current = bias_current - s_prev
        g_val = self.source.g(phi_eff, squid_current=squid_current)

        # Discrete step: s_next = s_prev * (1 - dt*gamma_minus) + dt*gamma_plus*g
        alpha = 1.0 - self.dt * gamma_minus
        beta = self.dt * gamma_plus
        return alpha * s_prev + beta * g_val

    def forward(self, phi: jax.Array, params: SingleDendriteParamsJAX, s0: jax.Array | None = None, solver: str = "fe") -> jax.Array:
        """Forward pass through SingleDendrite layer.

        Args:
            phi: Input flux [B, T, D]
            params: Layer parameters
            s0: Initial state [B, D] (optional, defaults to zeros)
            solver: Integration method - "fe" (Forward Euler) or "pararnn" (parallel Newton)

        Returns:
            State history [B, T+1, D]
        """
        batch, steps, dim = phi.shape
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)

        internal_J = params.internal_J

        mode_norm = str(self.internal_mode or "fixed").strip().lower()
        if mode_norm in {"dynamic", "dynamic_v1", "v1", "wicc"}:
            mode_norm = "wicc"
        elif mode_norm in {"dynamic_v2", "v2", "nocc"}:
            mode_norm = "nocc"
        elif mode_norm in {"fixed"}:
            mode_norm = "fixed"
        else:
            raise ValueError(
                f"Unknown internal_mode '{self.internal_mode}'. Supported: fixed, WICC (dynamic/v1), NOCC (dynamic_v2/v2)."
            )

        # Use scan to integrate with stateful internal connections
        if mode_norm != "fixed" and internal_J is not None:
            # Dynamic internal connections: initialize edge state based on mode
            if mode_norm == "nocc":
                # V2 uses tuple of (s1_e, s2_e, m_d)
                if internal_J is None:
                    return phi, None
                src_idx, dst_idx = _build_edge_index_jax(internal_J)
                E = src_idx.shape[0]
                edge_state: Any = (
                    jnp.zeros((batch, E), dtype=phi.dtype),
                    jnp.zeros((batch, E), dtype=phi.dtype),
                    jnp.zeros((batch, dim), dtype=phi.dtype),
                )
            else:
                # V1 uses single tensor [B, E]
                if internal_J is None:
                    return phi, None
                src_idx, dst_idx = _build_edge_index_jax(internal_J)
                E = src_idx.shape[0]
                edge_state = jnp.zeros((batch, E), dtype=phi.dtype)

            def scan_step(carry, phi_t):
                state, edge_st = carry
                phi_t_conn, edge_st_next = _apply_connectivity_jax(
                    state,
                    phi_t,
                    internal_J,
                    mode_norm,
                    self.internal_dynamic_params,
                    edge_st,
                    self.dt,
                    self.source,
                    self.internal_mask,
                )
                # Add phi_offset
                phi_off = params.phi_offset
                if phi_off is not None:
                    phi_t_conn = phi_t_conn + phi_off
                # Compute dynamics
                ds_dt = self.dynamics(
                    state,
                    phi_t_conn,
                    {
                        "phi_offset": params.phi_offset,
                        "gamma_plus": params.gamma_plus,
                        "gamma_minus": params.gamma_minus,
                        "bias_current": params.bias_current,
                    },
                )
                state_next = state + self.dt * ds_dt
                return (state_next, edge_st_next), state_next

            (s_final, _), s_hist = jax.lax.scan(scan_step, (s0, edge_state), phi.swapaxes(0, 1))
            return jnp.concatenate([s0[:, None, :], s_hist.swapaxes(0, 1)], axis=1)

        # Fixed mode: build parameter map
        pmap = {
            "phi_offset": params.phi_offset,
            "bias_current": params.bias_current,
            "gamma_plus": params.gamma_plus,
            "gamma_minus": params.gamma_minus,
        }

        # ParaRNN solver path - O(log T) parallel Newton method
        #
        # ParaRNN works with recurrent weights, but requires DIAGONAL Jacobian
        # structure for computational efficiency. The Jacobian for SingleDendrite is:
        #   J = ∂s_next/∂s_prev = α·I + β·diag(g') @ internal_J.T
        #
        # This is diagonal IFF internal_J is diagonal (element-wise recurrent weights).
        # Dense internal_J creates O(d³) matrix operations → use FE solver instead.
        #
        # See ParaRNN paper (Danieli et al. 2025), Equation 3.3 for details.
        if solver.lower() == "pararnn":
            if internal_J is not None:
                # Verify connectivity is diagonal (element-wise recurrent weights)
                if internal_J.ndim == 2:
                    off_diag = internal_J - jnp.diag(jnp.diag(internal_J))
                    if jnp.abs(off_diag).max() > 1e-8:
                        msg = (
                            "ParaRNN requires diagonal (element-wise) recurrent weights. "
                            "Dense weights create O(d³) Jacobian operations - use FE solver, "
                            "or restructure to use diagonal recurrence with inter-layer mixing."
                        )
                        raise RuntimeError(msg)

            from .integrators_jax import ParaRNNIntegratorJAX

            # Build step function that applies diagonal connectivity
            def step_with_connectivity(s_prev: jax.Array, phi_t: jax.Array, step_params: Mapping[str, jax.Array]) -> jax.Array:
                if internal_J is not None:
                    phi_t, _ = _apply_connectivity_jax(s_prev, phi_t, internal_J, "fixed", None, None, self.dt, None, self.internal_mask)
                return self.step(s_prev, phi_t, step_params)

            integrator = ParaRNNIntegratorJAX(step_fn=step_with_connectivity)
            return integrator.integrate(s0, phi, pmap)

        # Forward Euler path (default)
        def phi_transform(state: jax.Array, phi_t: jax.Array) -> jax.Array:
            phi_out, _ = _apply_connectivity_jax(state, phi_t, internal_J, "fixed", None, None, self.dt, None, self.internal_mask)
            return phi_out

        def dyn_wrapped(state: jax.Array, phi_t: jax.Array, dyn_pmap) -> jax.Array:
            phi_t = phi_transform(state, phi_t)
            phi_off = dyn_pmap.get("phi_offset", None)
            if phi_off is not None:
                phi_t = phi_t + phi_off
            return self.dynamics(state, phi_t, dyn_pmap)

        fe_solver = ForwardEulerJAX(dynamics=dyn_wrapped, dt=self.dt)
        return fe_solver.integrate(s0, phi, pmap)


@dataclass
class MultiplierNOCCState:
    """Container for the three state components of multiplier v2.

    Attributes:
        s1: Left branch SQUID states [B, D]
        s2: Right branch SQUID states [B, D]
        m: Aggregated output states [B, D]
    """

    s1: jax.Array
    s2: jax.Array
    m: jax.Array


@dataclass
class MultiplierNOCCParamsJAX:
    phi_y: jax.Array  # [B,D] or broadcastable to [B,D]
    alpha: jax.Array  # [B,D] or scalar/broadcastable
    beta: jax.Array  # [B,D] or scalar/broadcastable
    beta_out: jax.Array  # [B,D] or scalar/broadcastable
    bias_current: jax.Array  # [B,D] or scalar/broadcastable
    internal_J: jax.Array | None = None  # [D_to, D_from] or [B, D_to, D_from]


@dataclass
class MultiplierNOCCLayerJAX:
    """JAX implementation of MultiplierNOCC with dual SQUID states and aggregated output.

    This version uses a new flux collection mechanism with:
    - Two SQUID states per edge: s1 (left branch) and s2 (right branch)
    - One aggregated output state per node: m (post fan-in)
    - Coupled dynamics: solve for dot_m first, then use it to solve SQUID states
    """

    dim: int
    dt: float
    source: HeavisideStateDepJAX
    internal_mode: str = "fixed"
    internal_dynamic_params: dict | None = None
    internal_mask: jax.Array | None = None  # Structural mask for internal connections

    def dynamics(self, state: MultiplierNOCCState, phi: jax.Array, params: Mapping[str, jax.Array]) -> MultiplierNOCCState:
        """Compute time derivatives for all three state components.

        Args:
            state: MultiplierNOCCState with s1, s2, m tensors
            phi: Input flux [B, D] where D is number of nodes
            params: Dictionary containing phi_y, alpha, beta, beta_out, bias_current

        Returns:
            MultiplierNOCCState with dot_s1, dot_s2, dot_m
        """
        phi_y = params["phi_y"]
        alpha = params["alpha"]
        beta = params["beta"]
        beta_out = params["beta_out"]
        bias_current = params["bias_current"]

        # Extract state components
        s1 = state.s1  # [B, D] - left branch SQUID states
        s2 = state.s2  # [B, D] - right branch SQUID states
        m = state.m  # [B, D] - aggregated output states

        # Compute phi inputs (phi_a and phi_b for the two branches)
        phi_a = phi + phi_y  # [B, D]
        phi_b = phi - phi_y  # [B, D]

        # Compute source functions for both branches
        # Using state-dependent source function (heaviside)
        # Mirror Torch: RateArray supports only positive currents -> use soft_abs() for lookup
        # soft_abs provides a smooth derivative for numerical stability in Forward Euler
        squid_current_1 = bias_current - s1
        squid_current_2 = -bias_current + s2
        g1 = self.source.g(phi_a, squid_current=soft_abs_jax(squid_current_1))
        g2 = self.source.g(phi_b, squid_current=soft_abs_jax(squid_current_2))

        # Aggregate per node (no reduction across nodes)
        # Torch MultiplierNOCCLayer uses per-node aggregation; do not sum over D here.
        g_sum = g1 - g2

        # Get fan-in (default to 1 if not provided)
        fan_in = params.get("fan_in")
        if fan_in is None:
            fan_in = jnp.ones_like(m)  # [B, D] broadcastable

        # Step 1: Compute dot_m using aggregated source terms
        # (beta + 2*N*beta_out) * dot_m_i = sum_j(g1_ij - g2_ij) - alpha * m_i
        # Broadcast parameters to [B, D]
        beta_bc = jnp.broadcast_to(beta, m.shape)
        beta_out_bc = jnp.broadcast_to(beta_out, m.shape)
        # Effective inductance per node
        beta_eff = beta_bc + 2 * fan_in * beta_out_bc

        # Solve for dot_m
        dot_m = (g_sum - alpha * m) / beta_eff  # [B, D]

        # Step 2: Compute dot_s1 and dot_s2 using dot_m
        # beta * dot_s1_ij = g1_ij - beta_out * dot_m_i - alpha * s1_ij
        # beta * dot_s2_ij = g2_ij - beta_out * dot_m_i - alpha * s2_ij
        alpha_bc = jnp.broadcast_to(alpha, m.shape)
        dot_s1 = (g1 - beta_out_bc * dot_m - alpha_bc * s1) / beta_bc  # [B, D]
        dot_s2 = (g2 - beta_out_bc * dot_m - alpha_bc * s2) / beta_bc  # [B, D]

        return MultiplierNOCCState(s1=dot_s1, s2=dot_s2, m=dot_m)

    def _forward_impl(
        self,
        phi: jax.Array,
        params: MultiplierNOCCParamsJAX,
        s0: jax.Array | None = None,
        s1_init: jax.Array | None = None,
        s2_init: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass through the MultiplierNOCC layer.

        Args:
            phi: Input flux over time [B, T, D]
            params: MultiplierNOCCParamsJAX with all parameters
            s0: Initial m state [B, D] (optional, defaults to zeros)
            s1_init: Initial s1 state [B, D] (optional, for stepwise solvers)
            s2_init: Initial s2 state [B, D] (optional, for stepwise solvers)

        Returns:
            State history [B, T+1, D] containing m state trajectory
        """
        batch, steps, dim = phi.shape

        # Initialize all three states
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)
        s1 = s1_init if s1_init is not None else jnp.zeros((batch, dim), dtype=phi.dtype)
        s2 = s2_init if s2_init is not None else jnp.zeros((batch, dim), dtype=phi.dtype)
        m = s0

        # Internal connectivity (optional) applied per step
        internal_J = params.internal_J

        # Compute fan-in for each node if internal connectivity present
        if internal_J is not None:
            fan_in = (jnp.abs(internal_J) > 0).sum(axis=1).astype(phi.dtype)  # [D]
        else:
            fan_in = jnp.ones(dim, dtype=phi.dtype)

        # Map params to dict
        pmap = {
            "phi_y": params.phi_y,
            "alpha": params.alpha,
            "beta": params.beta,
            "beta_out": params.beta_out,
            "bias_current": params.bias_current,
            "fan_in": fan_in,
        }

        mode_norm = str(self.internal_mode or "fixed").strip().lower()
        if mode_norm in {"dynamic", "dynamic_v1", "v1", "wicc"}:
            mode_norm = "wicc"
        elif mode_norm in {"dynamic_v2", "v2", "nocc"}:
            mode_norm = "nocc"
        elif mode_norm in {"fixed"}:
            mode_norm = "fixed"
        else:
            raise ValueError(
                f"Unknown internal_mode '{self.internal_mode}'. Supported: fixed, WICC (dynamic/v1), NOCC (dynamic_v2/v2)."
            )

        # Use scan to integrate with stateful internal connections
        if mode_norm != "fixed" and internal_J is not None:
            # Dynamic internal connections: initialize edge state based on mode
            if mode_norm == "nocc":
                # V2 uses tuple of (s1_e, s2_e, m_d)
                if internal_J is None:
                    return phi, None
                src_idx, dst_idx = _build_edge_index_jax(internal_J)
                E = src_idx.shape[0]
                edge_state: Any = (
                    jnp.zeros((batch, E), dtype=phi.dtype),
                    jnp.zeros((batch, E), dtype=phi.dtype),
                    jnp.zeros((batch, dim), dtype=phi.dtype),
                )
            else:
                # V1 uses single tensor [B, E]
                if internal_J is None:
                    return phi, None
                src_idx, dst_idx = _build_edge_index_jax(internal_J)
                E = src_idx.shape[0]
                edge_state = jnp.zeros((batch, E), dtype=phi.dtype)

            def scan_step(carry, phi_t):
                s1, s2, m, edge_st = carry
                phi_t_conn, edge_st_next = _apply_connectivity_jax(
                    m,
                    phi_t,
                    internal_J,
                    mode_norm,
                    self.internal_dynamic_params,
                    edge_st,
                    self.dt,
                    self.source,
                    self.internal_mask,
                )
                # Create state wrapper
                state = MultiplierNOCCState(s1=s1, s2=s2, m=m)
                # Compute derivatives
                d_state = self.dynamics(state, phi_t_conn, pmap)
                # Update states
                s1_next = s1 + self.dt * d_state.s1
                s2_next = s2 + self.dt * d_state.s2
                m_next = m + self.dt * d_state.m
                return (s1_next, s2_next, m_next, edge_st_next), m_next

            (s1_final, s2_final, m_final, _), m_seq = jax.lax.scan(scan_step, (s1, s2, m, edge_state), phi.swapaxes(0, 1))
        else:
            # Fixed mode
            def scan_step(carry, phi_t):
                s1, s2, m = carry
                # Apply internal connectivity (fixed mode)
                if internal_J is not None:
                    phi_t, _ = _apply_connectivity_jax(m, phi_t, internal_J, "fixed", None, None, self.dt, None, self.internal_mask)
                # Create state wrapper
                state = MultiplierNOCCState(s1=s1, s2=s2, m=m)
                # Compute derivatives
                d_state = self.dynamics(state, phi_t, pmap)
                # Update states
                s1_next = s1 + self.dt * d_state.s1
                s2_next = s2 + self.dt * d_state.s2
                m_next = m + self.dt * d_state.m
                return (s1_next, s2_next, m_next), m_next

            (s1_final, s2_final, m_final), m_seq = jax.lax.scan(scan_step, (s1, s2, m), phi.swapaxes(0, 1))

        # Build history: prepend initial m state
        m_hist = jnp.concatenate([m[:, None, :], m_seq.swapaxes(0, 1)], axis=1)
        return m_hist, s1_final, s2_final

    def forward(
        self,
        phi: jax.Array,
        params: MultiplierNOCCParamsJAX,
        s0: jax.Array | None = None,
        s1_init: jax.Array | None = None,
        s2_init: jax.Array | None = None,
    ) -> jax.Array:
        m_hist, _s1_final, _s2_final = self._forward_impl(phi, params, s0=s0, s1_init=s1_init, s2_init=s2_init)
        return m_hist

    def forward_with_aux(
        self,
        phi: jax.Array,
        params: MultiplierNOCCParamsJAX,
        s0: jax.Array | None = None,
        s1_init: jax.Array | None = None,
        s2_init: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Pure forward that also returns final (s1, s2) auxiliary states.

        This replaces the legacy pattern of mutating `self` during forward.
        """
        return self._forward_impl(phi, params, s0=s0, s1_init=s1_init, s2_init=s2_init)

    def step(self, phi_t: jax.Array, params: MultiplierNOCCParamsJAX, s0: jax.Array, s1_state: jax.Array, s2_state: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Single step forward for stepwise solvers, returning updated s1, s2, m states.

        Args:
            phi_t: Single timestep flux [B, D]
            params: MultiplierNOCCParamsJAX with all parameters
            s0: Current m state [B, D]
            s1_state: Current s1 state [B, D]
            s2_state: Current s2 state [B, D]

        Returns:
            Tuple of (s1_next, s2_next, m_next)
        """
        internal_J = params.internal_J

        # Compute fan-in for each node (mirrors PyTorch)
        if internal_J is not None:
            fan_in = (jnp.abs(internal_J) > 0).sum(axis=1).astype(phi_t.dtype)  # [D]
        else:
            fan_in = jnp.ones(self.dim, dtype=phi_t.dtype)

        pmap = {
            "phi_y": params.phi_y,
            "alpha": params.alpha,
            "beta": params.beta,
            "beta_out": params.beta_out,
            "bias_current": params.bias_current,
            "fan_in": fan_in,
        }

        # Apply internal connectivity (fixed mode for step method - dynamic handled in forward)
        if internal_J is not None:
            phi_t, _ = _apply_connectivity_jax(s0, phi_t, internal_J, "fixed", None, None, self.dt, None, self.internal_mask)

        # Create state wrapper
        state = MultiplierNOCCState(s1=s1_state, s2=s2_state, m=s0)

        # Compute derivatives
        d_state = self.dynamics(state, phi_t, pmap)

        # Update states with Forward Euler
        s1_next = s1_state + self.dt * d_state.s1
        s2_next = s2_state + self.dt * d_state.s2
        m_next = s0 + self.dt * d_state.m

        return s1_next, s2_next, m_next


__all__ = [
    "MultiplierLayerJAX",
    "MultiplierParamsJAX",
    "MultiplierNOCCLayerJAX",
    "MultiplierNOCCParamsJAX",
    "MultiplierNOCCState",
    "SingleDendriteLayerJAX",
    "SingleDendriteParamsJAX",
]


# -----------------------------------------------------------------------------
# MinGRU JAX implementation (virtual recurrent layer)
# -----------------------------------------------------------------------------


@dataclass
class MinGRUParamsJAX:
    W_hidden: jax.Array  # [D, D]
    W_gate: jax.Array  # [D, D]
    internal_J: jax.Array | None = None  # [D_to, D_from] or [B, D_to, D_from]
    force_sequential: bool = False


@dataclass
class MinGRULayerJAX:
    dim: int
    dt: float  # unused (kept for API uniformity)
    internal_mask: jax.Array | None = None  # Structural mask for internal connections

    def _apply_connectivity(self, state: jax.Array, phi_t: jax.Array, internal_J: jax.Array | None) -> jax.Array:
        phi_out, _ = _apply_connectivity_jax(state, phi_t, internal_J, "fixed", None, None, self.dt, None, self.internal_mask)
        return phi_out

    def _step(self, state: jax.Array, phi_t: jax.Array, params: MinGRUParamsJAX) -> jax.Array:
        # Linear projections without bias: y = phi_t @ W^T
        gate_lin = phi_t @ params.W_gate.T
        z = jax.nn.sigmoid(gate_lin)
        hidden_lin = phi_t @ params.W_hidden.T
        g_val = jnp.tanh(hidden_lin)
        return (1.0 - z) * state + z * g_val

    def forward(self, phi: jax.Array, params: MinGRUParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        batch, _steps, dim = phi.shape
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)

        internal_J = params.internal_J
        use_parallel = (not params.force_sequential) and (internal_J is None)

        if not use_parallel:

            def scan_step(state, phi_t):
                # apply internal connectivity per step
                phi_eff = self._apply_connectivity(state, phi_t, internal_J)
                next_state = self._step(state, phi_eff, params)
                return next_state, next_state

            _, s_seq = jax.lax.scan(scan_step, s0, phi.swapaxes(0, 1))
            return jnp.concatenate([s0[:, None, :], s_seq.swapaxes(0, 1)], axis=1)

        # Parallel associative scan path (requires no internal connectivity)
        # Compute gate and proposal for all timesteps
        gate_lin = phi @ params.W_gate.T  # [B,T,D]
        z = jax.nn.sigmoid(gate_lin)
        hidden_lin = phi @ params.W_hidden.T
        g_val = jnp.tanh(hidden_lin)

        a = 1.0 - z
        b = z * g_val

        # Prepend identity element at t=0: a0=1, b0=0
        a0 = jnp.ones((batch, 1, dim), dtype=phi.dtype)
        b0 = jnp.zeros((batch, 1, dim), dtype=phi.dtype)
        a_seq = jnp.concatenate([a0, a], axis=1)  # [B,T+1,D]
        b_seq = jnp.concatenate([b0, b], axis=1)  # [B,T+1,D]

        # Move time to leading axis
        a_t = a_seq.swapaxes(0, 1)  # [T+1,B,D]
        b_t = b_seq.swapaxes(0, 1)

        def combine(x, y):
            a0, b0 = x
            a1, b1 = y
            return a0 * a1, a1 * b0 + b1

        A_t, B_t = jax.lax.associative_scan(combine, (a_t, b_t))
        # Compute states: s_t = A_t * s0 + B_t
        s_t = A_t * s0[None, :, :] + B_t  # [T+1,B,D]
        return s_t.swapaxes(0, 1)


__all__ += [
    "MinGRULayerJAX",
    "MinGRUParamsJAX",
]


# -----------------------------------------------------------------------------
# GRU JAX implementation (virtual recurrent layer)
# -----------------------------------------------------------------------------


@dataclass
class GRUParamsJAX:
    weight_ih: jax.Array  # [3, D, D]
    weight_hh: jax.Array  # [3, D, D]
    bias_ih: jax.Array  # [3, D]
    bias_hh: jax.Array  # [3, D]
    internal_J: jax.Array | None = None  # [D,D] or [B,D,D]
    force_sequential: bool = True


@dataclass
class GRULayerJAX:
    dim: int
    dt: float  # unused (kept for API uniformity)
    internal_mask: jax.Array | None = None  # Structural mask for internal connections

    def _apply_connectivity(self, state: jax.Array, phi_t: jax.Array, internal_J: jax.Array | None) -> jax.Array:
        phi_out, _ = _apply_connectivity_jax(state, phi_t, internal_J, "fixed", None, None, self.dt, None, self.internal_mask)
        return phi_out

    def _step(self, state: jax.Array, phi_t: jax.Array, params: GRUParamsJAX) -> jax.Array:
        # Unpack params
        W_ih = params.weight_ih  # [3,D,D]
        W_hh = params.weight_hh  # [3,D,D]
        b_ih = params.bias_ih  # [3,D]
        b_hh = params.bias_hh  # [3,D]

        # Compute gates
        x_lin = jnp.stack(
            [
                phi_t @ W_ih[0].T + b_ih[0],  # r
                phi_t @ W_ih[1].T + b_ih[1],  # z
                phi_t @ W_ih[2].T + b_ih[2],  # n (candidate)
            ],
            axis=0,
        )  # [3,B,D]
        h_lin = jnp.stack(
            [
                state @ W_hh[0].T + b_hh[0],
                state @ W_hh[1].T + b_hh[1],
                state @ W_hh[2].T + b_hh[2],
            ],
            axis=0,
        )

        # Gate order mirrors PyTorch GRU: (r, z, n)
        r = jax.nn.sigmoid(x_lin[0] + h_lin[0])
        z = jax.nn.sigmoid(x_lin[1] + h_lin[1])
        n = jnp.tanh(x_lin[2] + r * h_lin[2])

        return (1.0 - z) * n + z * state

    def forward(self, phi: jax.Array, params: GRUParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        batch, _steps, dim = phi.shape
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)

        internal_J = params.internal_J

        def scan_step(state, phi_t):
            phi_eff = self._apply_connectivity(state, phi_t, internal_J)
            next_state = self._step(state, phi_eff, params)
            return next_state, next_state

        # Always sequential for GRU (non-associative)
        _, s_seq = jax.lax.scan(scan_step, s0, phi.swapaxes(0, 1))
        return jnp.concatenate([s0[:, None, :], s_seq.swapaxes(0, 1)], axis=1)


__all__ += [
    "GRULayerJAX",
    "GRUParamsJAX",
]


# -----------------------------------------------------------------------------
# LSTM JAX implementation (virtual recurrent layer)
# -----------------------------------------------------------------------------


@dataclass
class LSTMParamsJAX:
    weight_ih: jax.Array  # [4*D, D] - Input-hidden weights for 4 gates (i, f, g, o)
    weight_hh: jax.Array  # [4*D, D] - Hidden-hidden weights for 4 gates
    bias_ih: jax.Array  # [4*D] - Input-hidden biases
    bias_hh: jax.Array  # [4*D] - Hidden-hidden biases
    internal_J: jax.Array | None = None  # [D,D] or [B,D,D]
    force_sequential: bool = True


@dataclass
class LSTMLayerJAX:
    dim: int
    dt: float  # unused (kept for API uniformity)
    internal_mask: jax.Array | None = None  # Structural mask for internal connections

    def _apply_connectivity(self, state: jax.Array, phi_t: jax.Array, internal_J: jax.Array | None) -> jax.Array:
        phi_out, _ = _apply_connectivity_jax(state, phi_t, internal_J, "fixed", None, None, self.dt, None, self.internal_mask)
        return phi_out

    def _step(self, h: jax.Array, c: jax.Array, phi_t: jax.Array, params: LSTMParamsJAX) -> tuple[jax.Array, jax.Array]:
        """Single LSTM step.

        Args:
            h: Hidden state [B, D]
            c: Cell state [B, D]
            phi_t: Input [B, D]
            params: LSTM parameters

        Returns:
            (h_next, c_next): Updated hidden and cell states
        """
        # Unpack params - weights are stored as [4*D, D], reshape to [4, D, D]
        W_ih = params.weight_ih.reshape(4, self.dim, self.dim)  # [4, D, D]
        W_hh = params.weight_hh.reshape(4, self.dim, self.dim)  # [4, D, D]
        b_ih = params.bias_ih.reshape(4, self.dim)  # [4, D]
        b_hh = params.bias_hh.reshape(4, self.dim)  # [4, D]

        # Compute all gates at once
        # PyTorch LSTM gate order: input, forget, cell, output (i, f, g, o)
        gates_x = jnp.stack([
            phi_t @ W_ih[0].T + b_ih[0],  # input gate (i)
            phi_t @ W_ih[1].T + b_ih[1],  # forget gate (f)
            phi_t @ W_ih[2].T + b_ih[2],  # cell gate (g)
            phi_t @ W_ih[3].T + b_ih[3],  # output gate (o)
        ], axis=0)  # [4, B, D]

        gates_h = jnp.stack([
            h @ W_hh[0].T + b_hh[0],
            h @ W_hh[1].T + b_hh[1],
            h @ W_hh[2].T + b_hh[2],
            h @ W_hh[3].T + b_hh[3],
        ], axis=0)  # [4, B, D]

        # Apply activations
        i = jax.nn.sigmoid(gates_x[0] + gates_h[0])  # input gate
        f = jax.nn.sigmoid(gates_x[1] + gates_h[1])  # forget gate
        g = jnp.tanh(gates_x[2] + gates_h[2])        # cell gate
        o = jax.nn.sigmoid(gates_x[3] + gates_h[3])  # output gate

        # Update cell state
        c_next = f * c + i * g

        # Update hidden state
        h_next = o * jnp.tanh(c_next)

        return h_next, c_next

    def forward(self, phi: jax.Array, params: LSTMParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        """Forward pass through LSTM.

        Args:
            phi: Input sequence [B, T, D]
            params: LSTM parameters
            s0: Initial hidden state [B, D] (optional, defaults to zeros)

        Returns:
            State history [B, T+1, D] containing hidden state trajectory
        """
        batch, _steps, dim = phi.shape
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)

        # Initialize cell state to zeros (matches PyTorch LSTMLayer line 308)
        c0 = jnp.zeros((batch, dim), dtype=phi.dtype)

        internal_J = params.internal_J

        def scan_step(carry, phi_t):
            h, c = carry
            # Apply internal connectivity to hidden state before computing gates
            phi_eff = self._apply_connectivity(h, phi_t, internal_J)
            h_next, c_next = self._step(h, c, phi_eff, params)
            return (h_next, c_next), h_next

        # Always sequential for LSTM (non-associative)
        (h_final, c_final), h_seq = jax.lax.scan(scan_step, (s0, c0), phi.swapaxes(0, 1))

        # Return hidden state history (not cell state), prepend initial state
        return jnp.concatenate([s0[:, None, :], h_seq.swapaxes(0, 1)], axis=1)


__all__ += [
    "LSTMLayerJAX",
    "LSTMParamsJAX",
]


# -----------------------------------------------------------------------------
# Linear and NonLinear JAX implementations (virtual basic layers)
# -----------------------------------------------------------------------------


@dataclass
class LinearLayerJAX:
    dim: int
    dt: float  # unused (for API consistency)

    def forward(self, phi: jax.Array, s0: jax.Array | None = None) -> jax.Array:
        batch, _steps, dim = phi.shape
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)
        # Pass-through: history[0]=s0, history[1:]=phi
        return jnp.concatenate([s0[:, None, :], phi], axis=1)


@dataclass
class NonLinearParamsJAX:
    phi_offset: jax.Array | None = None  # [B,D]
    bias_current: jax.Array | None = None  # [B,D] (used by some sources)


@dataclass
class NonLinearLayerJAX:
    dim: int
    dt: float  # unused
    source_key: str = "Tanh"

    def _select_source(self):
        key = self.source_key.lower()
        if key == "tanh":
            return TanhJAX()
        if key in ("simplegelu", "simple_gelu", "gelu_simple"):
            return SimpleGELUJAX()
        if key in ("telu",):
            return TeLUJAX()
        msg = f"Unsupported NonLinear source_key '{self.source_key}'. Supported: 'Tanh', 'SimpleGELU', 'Telu'"
        raise ValueError(
            msg,
        )

    def forward(self, phi: jax.Array, params: NonLinearParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        batch, _steps, dim = phi.shape
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)

        # Apply offset if provided
        if params.phi_offset is not None:
            # Broadcast [B,D] -> [B,1,D] then to [B,T,D]
            phi_eff = phi + params.phi_offset[:, None, :]
        else:
            phi_eff = phi

        source = self._select_source()
        # Apply elementwise source per step; pass bias_current if provided
        if params.bias_current is not None:
            # Broadcast [B,D] to match phi_eff [B,T,D]
            bias_current_broadcasted = params.bias_current[:, None, :]
            g_val = jax.vmap(lambda x, sc: source.g(x, squid_current=sc))(phi_eff, bias_current_broadcasted)
        else:
            g_val = jax.vmap(lambda x: source.g(x))(phi_eff)

        # Build history
        return jnp.concatenate([s0[:, None, :], g_val], axis=1)


__all__ += [
    "LinearLayerJAX",
    "NonLinearLayerJAX",
    "NonLinearParamsJAX",
]


# -----------------------------------------------------------------------------
# ScalingLayer JAX implementation (virtual scaling layer)
# -----------------------------------------------------------------------------


@dataclass
class ScalingParamsJAX:
    scale_factor: jax.Array  # [B,D] or broadcastable to [B,D]


@dataclass
class ScalingLayerJAX:
    dim: int
    dt: float  # unused (for API consistency)

    def forward(self, phi: jax.Array, params: ScalingParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        batch, _steps, dim = phi.shape
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)

        # Apply per-feature scaling: scaled = phi * scale_factor
        # Broadcast scale_factor from [B,D] to [B,T,D]
        scale = params.scale_factor[:, None, :]  # [B,1,D]
        scaled = phi * scale  # [B,T,D]

        # Build history: history[0]=s0, history[1:]=scaled
        return jnp.concatenate([s0[:, None, :], scaled], axis=1)


__all__ += [
    "ScalingLayerJAX",
    "ScalingParamsJAX",
]


# -----------------------------------------------------------------------------
# SoftmaxLayer JAX implementation (virtual softmax normalization layer)
# -----------------------------------------------------------------------------


@dataclass
class SoftmaxParamsJAX:
    beta: jax.Array  # [B,D] or broadcastable to [B,D] - temperature parameter


@dataclass
class SoftmaxLayerJAX:
    dim: int
    dt: float  # unused (for API consistency)

    def forward(self, phi: jax.Array, params: SoftmaxParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        batch, _steps, dim = phi.shape
        if s0 is None:
            s0 = jnp.zeros((batch, dim), dtype=phi.dtype)

        # Apply temperature-scaled softmax: softmax(beta * phi)
        # Broadcast beta from [B,D] to [B,T,D]
        beta = params.beta[:, None, :]  # [B,1,D]
        scaled_phi = phi * beta  # [B,T,D]
        softmax_out = jax.nn.softmax(scaled_phi, axis=-1)  # [B,T,D]

        # Build history: history[0]=s0, history[1:]=softmax_out
        return jnp.concatenate([s0[:, None, :], softmax_out], axis=1)


__all__ += [
    "SoftmaxLayerJAX",
    "SoftmaxParamsJAX",
]


# -----------------------------------------------------------------------------
# Soma (spiking) + Synapse (smoothing) JAX implementations (Equinox-based)
# -----------------------------------------------------------------------------


@dataclass
class SomaParamsJAX:
    phi_offset: jax.Array
    bias_current: jax.Array
    gamma_plus: jax.Array
    gamma_minus: jax.Array
    threshold: jax.Array
    internal_J: jax.Array | None = None


class SomaLayerJAX(eqx.Module):
    """Spiking Soma layer (skeleton, Equinox).

    - Continuous state evolution reuses SingleDendriteLayerJAX
    - Spike head uses hard threshold + surrogate gradients
    """

    dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    source: object = eqx.field(static=True)
    surrogate_kind: str = eqx.field(static=True)
    surrogate_params: dict[str, float] = eqx.field(static=True)
    internal_mode: str = eqx.field(static=True)
    internal_dynamic_params: dict | None = eqx.field(static=True)
    internal_mask: jax.Array | None = eqx.field(static=True)

    def forward(self, phi: jax.Array, params: SomaParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        raise NotImplementedError(
            "SomaLayerJAX is a placeholder skeleton and is not implemented yet. "
            "It exists only to reserve the API surface while the soma coupling/threshold circuit is defined."
        )
        base = SingleDendriteLayerJAX(
            dim=self.dim,
            dt=self.dt,
            source=self.source,
            internal_mode=self.internal_mode,
            internal_dynamic_params=self.internal_dynamic_params,
            internal_mask=self.internal_mask,
        )
        cont_hist = base.forward(
            phi,
            SingleDendriteParamsJAX(
                phi_offset=params.phi_offset,
                bias_current=params.bias_current,
                gamma_plus=params.gamma_plus,
                gamma_minus=params.gamma_minus,
                internal_J=params.internal_J,
            ),
            s0=s0,
            solver="fe",
        )
        spec = SurrogateSpec(kind=self.surrogate_kind, **(self.surrogate_params or {}))
        return spike_jax(cont_hist, threshold=params.threshold, surrogate=spec)


@dataclass
class SynapseParamsJAX:
    alpha: jax.Array


class SynapseLayerJAX(eqx.Module):
    """Synapse smoothing for spike trains (skeleton, Equinox)."""

    dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def forward(self, spikes: jax.Array, params: SynapseParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        raise NotImplementedError(
            "SynapseLayerJAX is a placeholder skeleton and is not implemented yet. "
            "It exists only to reserve the API surface while synapse dynamics/state handling are finalized."
        )
        if spikes.ndim != 3:
            raise ValueError(f"Synapse expects [B,T,D], got shape {spikes.shape}")
        B, T, D = spikes.shape
        if D != self.dim:
            raise ValueError(f"Synapse expected dim={self.dim}, got {D}")

        alpha = params.alpha
        if s0 is None:
            y0 = jnp.zeros((B, D), dtype=spikes.dtype)
        else:
            y0 = s0

        def step(y_prev, z_t):
            y_next = alpha * y_prev + (1.0 - alpha) * z_t
            return y_next, y_next

        _, ys = jax.lax.scan(step, y0, spikes.swapaxes(0, 1))
        return jnp.concatenate([y0[:, None, :], ys.swapaxes(0, 1)], axis=1)


__all__ += [
    "SomaLayerJAX",
    "SomaParamsJAX",
    "SynapseLayerJAX",
    "SynapseParamsJAX",
]

@dataclass
class LeakyGRUParamsJAX:
    W_in: jax.Array  # [H, I]
    W_hn: jax.Array  # [H, H]
    bias_z: jax.Array  # [H]
    bias_n: jax.Array  # [H]
    bias_r: jax.Array  # [H] - fixed
    internal_J: jax.Array | None = None
    force_sequential: bool = False


@dataclass
class LeakyGRULayerJAX:
    """JAX implementation of LeakyGRU.

    Equations:
        r = sigmoid(b_r)                 (fixed ~ 1)
        z = sigmoid(b_z)                 (learnable)
        n_t = tanh(W_in x_t + b_n + r * (W_hn h_{t-1}))
        h_t = z * h_{t-1} + (1 - z) * n_t
    """

    dim: int
    dt: float
    internal_mask: jax.Array | None = None

    def step(self, h_prev: jax.Array, x_t: jax.Array, params: LeakyGRUParamsJAX) -> jax.Array:
        W_in = params.W_in
        W_hn = params.W_hn
        b_z = params.bias_z
        b_n = params.bias_n
        b_r = params.bias_r

        r = jax.nn.sigmoid(b_r)
        z = jax.nn.sigmoid(b_z)

        # W_in @ x_t.T + b_n
        # x_t is [B, I] (implicitly, via caller)
        # We need to handle potential broadcasting if implemented as pure vector func or batch func.
        # This implementation assumes inputs are batched [B, ...].
        # But JAX scan over time usually passes [B, I] slices.

        # W_in: [H, I], x_t: [B, I] -> [B, H]
        in_proj = x_t @ W_in.T + b_n

        # W_hn: [H, H], h_prev: [B, H] -> [B, H]
        hid_proj = h_prev @ W_hn.T

        n_preact = in_proj + r * hid_proj
        n = jnp.tanh(n_preact)

        h_new = z * h_prev + (1.0 - z) * n
        return h_new

    def forward(self, x: jax.Array, params: LeakyGRUParamsJAX, s0: jax.Array | None = None) -> jax.Array:
        """Forward pass.

        Args:
            x: Input sequence [B, T, I]
            params: LeakyGRUParamsJAX
            s0: Initial state [B, H]

        Returns:
            State history [B, T+1, H]
        """
        batch, steps, _ = x.shape
        if s0 is None:
            s0 = jnp.zeros((batch, self.dim), dtype=x.dtype)

        internal_J = params.internal_J

        def scan_step(carry, x_t):
            h_prev, edge_st = carry

            # Handle internal connectivity input (x_t is external input + internal fb)
            # For LeakyGRU, strict internal connectivity is "fixed" mode (additive flux).
            # But the layer IS the recurrence, so internal_J here usually refers to
            # *additional* recurrent connections outside the GRU kernel (like lateral inhibition).
            # The GRU kernel's own recurrence is W_hn.

            phi_total = x_t
            edge_st_next = edge_st
            if internal_J is not None:
                phi_fb, edge_st_next = _apply_connectivity_jax(
                    h_prev, x_t, internal_J, "fixed", None, edge_st, self.dt, None, self.internal_mask
                )
                phi_total = phi_fb

            h_new = self.step(h_prev, phi_total, params)
            return (h_new, edge_st_next), h_new

        # Edge state for internal J (if any)
        edge_state_init = None  # Fixed mode doesn't use edge state, but kept for API consistency
        if internal_J is not None:
            src_idx, _ = _build_edge_index_jax(internal_J)
            E = src_idx.shape[0]
            edge_state_init = jnp.zeros((batch, E), dtype=x.dtype)

        (h_final, _), h_seq = jax.lax.scan(scan_step, (s0, edge_state_init), x.swapaxes(0, 1))

        # Prepend s0
        return jnp.concatenate([s0[:, None, :], h_seq.swapaxes(0, 1)], axis=1)


# Register LeakyGRUParamsJAX as a PyTree node so it can be passed to JAX (e.g. jit)
jax.tree_util.register_pytree_node(
    LeakyGRUParamsJAX,
    lambda p: ((p.W_in, p.W_hn, p.bias_z, p.bias_n, p.bias_r, p.internal_J), (p.force_sequential,)),
    lambda aux, children: LeakyGRUParamsJAX(*children[:5], children[5], aux[0]),
)
