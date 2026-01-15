"""ParaRNN solver for parallel training of nonlinear RNNs.

Implements Newton's method with parallel prefix scan for solving nonlinear
RNN recurrences in O(log T) depth, based on:

    Danieli et al. (2025) "ParaRNN: Unlocking Parallel Training of
    Nonlinear RNNs for Large Language Models"

Jacobian Structure Requirements
-------------------------------
The ParaRNN framework mathematically works for ANY differentiable recurrence
function f, regardless of Jacobian structure. However, for computational
efficiency, the Jacobian J_f = ∂f(h_{t-1}, x_t)/∂h_{t-1} must have special
structure:

- **Dense Jacobian**: O(d³) per matrix multiply - too slow for large d
- **Diagonal Jacobian**: O(d) per multiply - very fast, used here
- **Block-diagonal**: O(k³ · d/k) - feasible with custom CUDA kernels

For SOEN SingleDendrite layers:
- Without recurrent weights: Jacobian is trivially diagonal
- With diagonal (element-wise) recurrent weights: Jacobian stays diagonal
- With dense recurrent weights: Jacobian becomes dense → not supported

The diagonal constraint means each neuron's state update depends only on its
own previous state (not other neurons') during the Newton linearization step.
Feature mixing across neurons is handled by the feed-forward connections
between layers, similar to Mamba or Transformer architectures.

See Also
--------
- ParaRNN paper: https://arxiv.org/abs/2505.14825
- Equation 3.3: A_* = diag(a_*) for diagonal Jacobian structure
- Appendix D.4: Discussion of block-diagonal extensions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
import weakref

import torch

from soen_toolkit.core.layers.common.features import (
    CompositeFeature,
    FeatureHook,
)

from .base import SolverBase, SupportsState
from .parallel_scan import sign_log_scan

if TYPE_CHECKING:
    from collections.abc import Mapping


class StepProvider(Protocol):
    """Interface for layers to provide discrete step function for ParaRNN.

    The step function computes s_{t+1} = f(s_t, phi_t) directly rather than
    through ODE integration. For Forward Euler discretization:
        s_{t+1} = s_t + dt * ds/dt
               = s_t * (1 - dt*gamma_minus) + dt*gamma_plus*g(phi, squid_current)
    """

    def step(
        self,
        s_prev: torch.Tensor,
        phi_t: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute next state from previous state and input.

        Args:
            s_prev: Previous state [B, D]
            phi_t: Input flux at current timestep [B, D]
            params: Parameter dict with layer-specific values
            dt: Timestep size

        Returns:
            Next state [B, D]
        """
        ...


@dataclass(slots=True)
class _ParaRNNContext:
    dt: torch.Tensor


class ParaRNNSolver(SolverBase):
    """Parallel RNN solver using Newton's method with prefix scan.

    This solver casts the sequential RNN application as a system of nonlinear
    equations and solves it via Newton iterations. Each iteration's linearized
    system has block bi-diagonal structure and is solved in O(log T) using
    parallel prefix scan (associative scan).

    The algorithm:
    1. Initialize guess h⁰ = f(0, x) for all timesteps
    2. For each Newton iteration k:
       a. Compute Jacobian A_t = ∂f/∂h_{t-1} at current guess
       b. Compute residual b_t = f(h^k_{t-1}, x_t) - A_t · h^k_{t-1}
       c. Solve linear recurrence h^{k+1}_t = A_t · h^{k+1}_{t-1} + b_t
          using parallel prefix scan in O(log T)
    3. Return when ||h^{k+1} - h^k|| < tolerance

    For layers with diagonal Jacobians, step 2c reduces to element-wise
    operations, making the entire algorithm highly parallelizable on GPU.

    Attributes:
        max_iter: Maximum Newton iterations (typically 3-5 suffice for smooth
            nonlinearities like tanh; may need more for discontinuous functions)
        tol: Convergence tolerance for early stopping

    Notes:
        - Recurrent weights are supported if they are DIAGONAL (element-wise)
        - Dense recurrent weights create non-diagonal Jacobians → use FE solver
        - Compatible with all source functions (Tanh, Heaviside, RateArray, etc.)
    """

    def __init__(
        self,
        *,
        step_provider: StepProvider,
        feature: FeatureHook | None = None,
        layer=None,
        max_iter: int = 15,
        tol: float = 1e-6,
    ) -> None:
        super().__init__()
        self._step_provider = step_provider

        # Register dynamics module if step_provider has one (for state_dict compatibility)
        # This ensures source function buffers (like g_table) are properly saved/loaded
        if hasattr(step_provider, "_dynamics"):
            from torch import nn
            dynamics = step_provider._dynamics
            if isinstance(dynamics, nn.Module):
                self._dynamics = dynamics

        if isinstance(feature, CompositeFeature):
            self._feature = feature
        elif feature is None:
            self._feature = CompositeFeature()
        else:
            self._feature = CompositeFeature([feature])
        self._layer_ref = weakref.ref(layer) if layer is not None else None
        self._max_iter = max_iter
        self._tol = tol

    def integrate(
        self,
        *,
        state: SupportsState,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the RNN recurrence in parallel using Newton's method.

        Args:
            state: Initial state wrapper with .values [B, D] or [D]
            phi: Input sequence [B, T, D]
            params: Parameter dict (values broadcastable to [B, D])
            dt: Timestep size

        Returns:
            State history [B, T+1, D]
        """
        if phi.dim() != 3:
            msg = f"Expected phi with shape [batch, steps, dim], received {tuple(phi.shape)}"
            raise ValueError(msg)

        batch, steps, dim = phi.shape
        dt = dt.to(device=phi.device, dtype=phi.dtype)

        state_tensor = state.values.to(device=phi.device, dtype=phi.dtype)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.view(1, -1).expand(batch, -1)
        elif state_tensor.shape != (batch, dim):
            msg = f"Initial state must have shape [batch, dim] or [dim]; got {tuple(state_tensor.shape)}"
            raise ValueError(msg)

        # Expand params to [B, D] shape
        expanded_params = {
            name: _expand_param(tensor, batch, dim, phi.device, phi.dtype)
            for name, tensor in params.items()
        }

        # Setup feature hooks
        context = _ParaRNNContext(dt=dt)
        layer_ref = self._layer_ref() if self._layer_ref is not None else None
        if hasattr(layer_ref, "_clear_histories"):
            layer_ref._clear_histories() # type: ignore[union-attr]
        if layer_ref is not None and hasattr(self._feature, "attach_layer"):
            self._feature.attach_layer(layer_ref)
        self._feature.on_integration_start(
            context=context,
            state=state_tensor,
            phi=phi,
            params=expanded_params,
        )

        # Helper to build params for a single batch element with T as pseudo-batch
        def _build_params_for_batch(b: int) -> dict[str, torch.Tensor]:
            params_b = {}
            for k, v in expanded_params.items():
                if v.dim() == 2 and v.shape == (dim, dim):
                    # Weight matrix like internal_J - keep as is
                    params_b[k] = v
                elif v.dim() == 2 and v.shape[0] == batch:
                    # [B, D] -> [T, D] for this batch element
                    params_b[k] = v[b : b + 1].expand(steps, -1)
                else:
                    # Scalar or other - pass through
                    params_b[k] = v
            return params_b

        # Initial guess: apply step with zero previous state
        # Process batch-by-batch to handle pseudo-batching over time
        zeros_state = torch.zeros((steps, dim), device=phi.device, dtype=phi.dtype)
        s_guess_list = []
        for b in range(batch):
            params_b = _build_params_for_batch(b)
            s_guess_b = self._step_provider.step(zeros_state, phi[b], params_b, dt)  # [T, D]
            s_guess_list.append(s_guess_b)
        s_guess = torch.stack(s_guess_list, dim=0)  # [B, T, D]

        # Newton iterations
        s_curr = s_guess
        for iteration in range(self._max_iter):
            # s_prev_guess[t] = s_curr[t-1] for t > 0, s0 for t = 0
            s_prev_guess = torch.cat([state_tensor[:, None, :], s_curr[:, :-1, :]], dim=1)  # [B, T, D]

            # Compute f(s_prev, phi) and diagonal Jacobian for all [B, T] elements
            # Loop over batch, treating T as pseudo-batch for each batch element
            f_vals = []
            a_vals = []
            with torch.enable_grad():
                for b in range(batch):
                    s_prev_b = s_prev_guess[b]  # [T, D]
                    phi_b = phi[b]  # [T, D]
                    params_b = _build_params_for_batch(b)

                    # Compute step values for all timesteps at once
                    f_b = self._step_provider.step(s_prev_b, phi_b, params_b, dt)  # [T, D]
                    f_vals.append(f_b.detach())

                    # Compute diagonal Jacobians via autograd
                    s_prev_b_grad = s_prev_b.detach().clone().requires_grad_(True)
                    f_b_grad = self._step_provider.step(s_prev_b_grad, phi_b.detach(), params_b, dt)
                    grad_out = torch.ones_like(f_b_grad)
                    (a_b,) = torch.autograd.grad(f_b_grad, s_prev_b_grad, grad_out)
                    a_vals.append(a_b.detach())  # [T, D]

            f_val = torch.stack(f_vals, dim=0)  # [B, T, D]
            a_val = torch.stack(a_vals, dim=0)  # [B, T, D]

            # Residual: b = f(s_prev, phi) - a * s_prev
            b_val = f_val - a_val * s_prev_guess  # [B, T, D]

            # Solve linear recurrence using Heinsen's sign-log scan for numerical stability
            # sign_log_scan returns [B, T+1, D] including x0, we need [B, T, D]
            s_new = sign_log_scan(a_val, b_val, x0=state_tensor)[:, 1:, :]  # [B, T, D]

            # Check convergence
            diff = (s_new - s_curr).abs().max().item()
            s_curr = s_new

            if diff < self._tol:
                break

        # Build history: prepend s0
        history = torch.cat([state_tensor[:, None, :], s_curr], dim=1)  # [B, T+1, D]

        # Call feature hooks for compatibility (simplified - not per-step)
        self._feature.on_integration_end(
            context=context,
            history=history,
            params=expanded_params,
        )

        return history


def _expand_param(
    tensor: torch.Tensor,
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand parameter tensor to [B, D] shape."""
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.dim() == 0:
        return tensor.view(1, 1).expand(batch, dim)
    if tensor.dim() == 1:
        if tensor.shape[0] not in {1, dim}:
            msg = f"Cannot broadcast parameter of shape {tuple(tensor.shape)} to dimension {dim}"
            raise ValueError(msg)
        return tensor.view(1, -1).expand(batch, -1)
    if tensor.dim() == 2:
        if tensor.shape == (dim, dim):
            return tensor
        if tensor.shape == (batch, dim):
            return tensor
        msg = f"Cannot use parameter with shape {tuple(tensor.shape)} for batch {batch} and dim {dim}"
        raise ValueError(msg)
    msg = f"Unsupported parameter tensor rank: {tensor.dim()}"
    raise ValueError(msg)


__all__ = ["ParaRNNSolver", "StepProvider"]

