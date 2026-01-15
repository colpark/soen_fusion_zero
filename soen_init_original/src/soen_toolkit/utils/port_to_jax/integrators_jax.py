"""JAX integrators for SOEN layer dynamics.

This module provides JAX implementations of numerical integrators for solving
the ODE/recurrence dynamics of SOEN layers.

Available Integrators
---------------------
ForwardEulerJAX
    Sequential O(T) integrator using lax.scan. Most compatible, works with
    any dynamics and connectivity structure.

ParaRNNIntegratorJAX
    Parallel O(log T) integrator using Newton's method with associative scan.
    Based on the ParaRNN paper (Danieli et al. 2025). Requires diagonal Jacobian
    structure - see class docstring for details on recurrent weight constraints.

See Also
--------
ParaRNN paper: https://arxiv.org/abs/2505.14825
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
from jax import lax, vmap
import jax.numpy as jnp

DynamicsFn = Callable[[jax.Array, jax.Array, Mapping[str, jax.Array]], jax.Array]
# Step function: (s_prev, phi_t, params) -> s_next
StepFn = Callable[[jax.Array, jax.Array, Mapping[str, jax.Array]], jax.Array]


@dataclass
class ForwardEulerJAX:
    """Forward Euler sequence integrator implemented with JAX and lax.scan.

    Integrates over timesteps of phi: given initial state s0 and per‑step phi[t],
    computes s_{t+1} = s_t + dt * f(s_t, phi[t], params).

    Tracks and returns the full state history with shape [B, T+1, D].
    """

    dynamics: DynamicsFn
    dt: float

    def integrate(self, s0: jax.Array, phi: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Args:
            s0: Initial state [B, D]
            phi: Input over time [B, T, D]
            params: Mapping of parameter tensors (broadcastable to [B,D]).

        Returns:
            State history [B, T+1, D]

        """
        _batch, _steps, _dim = phi.shape

        def step_fn(state, phi_t):
            ds_dt = self.dynamics(state, phi_t, params)
            next_state = state + self.dt * ds_dt
            return next_state, next_state

        # Scan over time: carry is state [B,D], outputs [T,B,D]
        _s_last, s_seq = jax.lax.scan(step_fn, s0, phi.swapaxes(0, 1))
        # Build history: prepend s0
        return jnp.concatenate([s0[:, None, :], s_seq.swapaxes(0, 1)], axis=1)


def _associative_scan_log_op(e1: tuple[jax.Array, jax.Array], e2: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Associative operator for parallel prefix sum in log-space.

    For the linear recurrence s_t = a_t * s_{t-1} + b_t, we track cumulative
    products of a (in log space as L) and cumulative sums of b (weighted by
    cumulative products).

    Args:
        e1: (L1, b1) from left segment
        e2: (L2, b2) from right segment

    Returns:
        Combined (L_new, b_new) representing merged segment
    """
    L1, b1 = e1
    L2, b2 = e2
    L_new = L2 + L1
    b_new = b2 + b1 * jnp.exp(L2)
    return L_new, b_new


@dataclass
class ParaRNNIntegratorJAX:
    """Parallel RNN solver using Newton's method with associative scan.

    Implements the ParaRNN algorithm from Danieli et al. (2025) for solving
    nonlinear RNN recurrences in parallel. The sequential RNN application is
    cast as a system of equations solved via Newton iterations, where each
    iteration's linear system is solved in O(log T) using associative scan.

    Algorithm Overview
    ------------------
    1. Initialize: h⁰_t = f(0, x_t) for all timesteps t
    2. Newton iteration k:
       a. Compute Jacobian A_t = ∂f/∂h at current guess h^k_{t-1}
       b. Compute residual b_t = f(h^k_{t-1}, x_t) - A_t · h^k_{t-1}
       c. Solve h^{k+1}_t = A_t · h^{k+1}_{t-1} + b_t via parallel prefix scan
    3. Converge when ||h^{k+1} - h^k|| < tolerance

    Jacobian Structure Requirement
    ------------------------------
    The algorithm works mathematically for any Jacobian, but computational
    efficiency requires DIAGONAL Jacobian structure:

    - Dense Jacobian: O(d³) per matrix multiply → prohibitive for large d
    - Diagonal Jacobian: O(d) element-wise ops → very fast, used here

    For SingleDendrite layers:
    - No recurrent weights: Jacobian is trivially diagonal ✓
    - Diagonal (element-wise) recurrent weights: Jacobian stays diagonal ✓
    - Dense recurrent weights: Jacobian becomes dense ✗

    The diagonal constraint means each neuron's update depends only on its own
    previous state during linearization. Feature mixing across neurons should
    be done via inter-layer connections (like Mamba/Transformer architectures).

    Attributes:
        step_fn: Discrete step function (s_prev, phi_t, params) -> s_next
        max_iter: Maximum Newton iterations (3-5 for smooth nonlinearities,
            more for discontinuous functions like Heaviside)
        tol: Convergence tolerance for early stopping

    See Also
    --------
    ParaRNN paper: https://arxiv.org/abs/2505.14825
    Equation 3.3: A_* = diag(a_*) for diagonal Jacobian structure
    """

    step_fn: StepFn
    max_iter: int = 15
    tol: float = 1e-6

    def integrate(self, s0: jax.Array, phi: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        """Solve the RNN recurrence in parallel using Newton's method.

        Args:
            s0: Initial state [B, D]
            phi: Input over time [B, T, D]
            params: Parameter dict (values broadcastable to [B, D])

        Returns:
            State history [B, T+1, D]
        """
        batch_size, seq_len, dim = phi.shape

        # Compute step value and diagonal Jacobian for a single timestep
        def step_and_jacobian(s_prev: jax.Array, phi_t: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Compute step output and diagonal Jacobian for [B, D] inputs."""
            # Compute step value
            s_next = self.step_fn(s_prev, phi_t, params)  # [B, D]

            # Compute diagonal Jacobian: d(s_next[b,d]) / d(s_prev[b,d])
            # Use vmap to compute gradient for each (batch, dim) element
            def grad_for_element(b_idx, d_idx):
                """Compute gradient for single element."""
                def scalar_fn(s_val):
                    s_full = s_prev.at[b_idx, d_idx].set(s_val)
                    return self.step_fn(s_full, phi_t, params)[b_idx, d_idx]
                return jax.grad(scalar_fn)(s_prev[b_idx, d_idx])

            # Create index grids and vectorize
            b_indices = jnp.arange(batch_size)[:, None].repeat(dim, axis=1)  # [B, D]
            d_indices = jnp.arange(dim)[None, :].repeat(batch_size, axis=0)  # [B, D]

            # Flatten and compute gradients
            a_flat = vmap(grad_for_element)(b_indices.ravel(), d_indices.ravel())
            a_val = a_flat.reshape(batch_size, dim)  # [B, D]

            return s_next, a_val

        # Apply step_and_jacobian across time dimension
        def step_and_jacobian_over_time(s_prev_seq: jax.Array, phi_seq: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Apply step_and_jacobian for each timestep. Inputs/outputs are [B, T, D]."""
            # Transpose to [T, B, D] for easier vmapping over time
            s_prev_T = s_prev_seq.swapaxes(0, 1)  # [T, B, D]
            phi_T = phi_seq.swapaxes(0, 1)  # [T, B, D]

            # vmap over time
            f_vals, a_vals = vmap(step_and_jacobian)(s_prev_T, phi_T)  # [T, B, D] each

            # Transpose back to [B, T, D]
            return f_vals.swapaxes(0, 1), a_vals.swapaxes(0, 1)

        # Initial guess: apply step with zero previous state
        zeros_state = jnp.zeros((batch_size, dim), dtype=phi.dtype)
        phi_T = phi.swapaxes(0, 1)  # [T, B, D]
        s_guess = vmap(lambda phi_t: self.step_fn(zeros_state, phi_t, params))(phi_T)  # [T, B, D]
        s_guess = s_guess.swapaxes(0, 1)  # [B, T, D]

        def newton_body(val: tuple[int, jax.Array, jax.Array]) -> tuple[int, jax.Array, jax.Array]:
            """Single Newton iteration."""
            i, s_curr, _ = val

            # s_prev_guess[t] = s_curr[t-1] for t > 0, s0 for t = 0
            s_prev_guess = jnp.concatenate([s0[:, None, :], s_curr[:, :-1, :]], axis=1)  # [B, T, D]

            # Compute step values and Jacobians
            f_val, a_val = step_and_jacobian_over_time(s_prev_guess, phi)  # [B, T, D] each

            # Residual: b = f(s_prev, phi) - a * s_prev
            b_val = f_val - a_val * s_prev_guess  # [B, T, D]

            # Solve linear recurrence using associative scan in log-space
            log_a = jnp.log(jnp.clip(jnp.abs(a_val), 1e-9, None))  # [B, T, D]

            def scan_over_time(log_a_seq: jax.Array, b_seq: jax.Array) -> tuple[jax.Array, jax.Array]:
                """Scan over time for [T] arrays."""
                return lax.associative_scan(_associative_scan_log_op, (log_a_seq, b_seq))

            # Reshape to [B*D, T] for vectorized scan
            log_a_flat = log_a.transpose(0, 2, 1).reshape(-1, seq_len)  # [B*D, T]
            b_flat = b_val.transpose(0, 2, 1).reshape(-1, seq_len)  # [B*D, T]

            cum_L, cum_B = vmap(scan_over_time)(log_a_flat, b_flat)  # [B*D, T] each

            # Reconstruct and reshape
            cum_L = cum_L.reshape(batch_size, dim, seq_len).transpose(0, 2, 1)  # [B, T, D]
            cum_B = cum_B.reshape(batch_size, dim, seq_len).transpose(0, 2, 1)  # [B, T, D]

            s_new = jnp.exp(cum_L) * s0[:, None, :] + cum_B  # [B, T, D]
            diff = jnp.max(jnp.abs(s_new - s_curr))

            return i + 1, s_new, diff

        def newton_cond(val: tuple[int, jax.Array, jax.Array]) -> jax.Array:
            """Continue while not converged and under max iterations."""
            i, _, diff = val
            return (i < self.max_iter) & (diff > self.tol)

        # Run Newton iterations
        init_val = (0, s_guess, jnp.array(1.0))
        _, s_final, _ = lax.while_loop(newton_cond, newton_body, init_val)

        # Build history: prepend s0
        return jnp.concatenate([s0[:, None, :], s_final], axis=1)  # [B, T+1, D]


__all__ = ["ForwardEulerJAX", "ParaRNNIntegratorJAX"]
