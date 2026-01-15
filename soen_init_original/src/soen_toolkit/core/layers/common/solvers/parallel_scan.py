"""Generic parallel scan solver infrastructure.

The goal is to share all logic that is truly solver-specific (prefix scan,
feature-hook orchestration, parameter broadcasting) while allowing each layer
to provide its own coefficient generator.  Keep this module completely free
of layer-specific math
concrete layers supply a ``CoefficientProvider`` that
knows how to map their ODE into ``a`` and ``b`` sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
import weakref

import torch
import torch.nn.functional as F

from soen_toolkit.core.layers.common.features import (
    CompositeFeature,
    FeatureHook,
    StepPayload,
)

from ._phi_utils import compute_phi_with_offset
from .base import SolverBase, SupportsState

if TYPE_CHECKING:
    from collections.abc import Mapping


class CoefficientProvider(Protocol):
    """Interface a layer must supply to use :class:`ParallelScanSolver`.

    ``coefficients`` should return the sequences ``a_t`` and ``b_t`` for the
    recurrence ``x_t = a_t * x_{t-1} + b_t``.  ``observable`` returns whatever
    quantity the layer wants to expose via feature hooks (typically ``g``).
    """

    def coefficients(
        self,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def observable(
        self,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor: ...


@dataclass(slots=True)
class _ParallelScanContext:
    dt: torch.Tensor


class ParallelScanSolver(SolverBase):
    """Layer-agnostic parallel scan solver.

    This class orchestrates the generic parts of a prefix-scan integration.
    Layer-specific behaviour is delegated to ``CoefficientProvider`` which
    produces the coefficients/observables needed for the scan.
    """

    def __init__(
        self,
        *,
        coeff_provider: CoefficientProvider,
        feature: FeatureHook | None = None,
        layer=None,
    ) -> None:
        super().__init__()
        self._coeff_provider = coeff_provider
        if isinstance(feature, CompositeFeature):
            self._feature = feature
        elif feature is None:
            self._feature = CompositeFeature()
        else:
            self._feature = CompositeFeature([feature])
        self._layer_ref = weakref.ref(layer) if layer is not None else None

    def integrate(
        self,
        *,
        state: SupportsState,
        phi: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> torch.Tensor:
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
            raise ValueError(
                msg,
            )

        # Bring all parameters into a common ``[batch, steps, dim]`` view once.
        expanded_params = {name: _broadcast_param(tensor, batch, steps, dim, phi.device, phi.dtype) for name, tensor in params.items()}

        context = _ParallelScanContext(dt=dt)
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

        payload = StepPayload(state=state_tensor, phi=phi, params=expanded_params)
        payload = self._feature.on_before_step(
            context=context,
            step_index=0,
            payload=payload,
        )

        phi_tensor = compute_phi_with_offset(payload.phi, payload.params)
        params_tensor = payload.params

        a_tensor, b_tensor = self._coeff_provider.coefficients(
            phi_tensor,
            params_tensor,
            dt,
        )

        history = sign_log_scan(a_tensor, b_tensor, x0=state_tensor)

        observables = self._coeff_provider.observable(phi_tensor, params_tensor)
        prev_states = history[:, :-1, :]
        next_states = history[:, 1:, :]
        gamma_plus = params_tensor.get("gamma_plus", torch.ones_like(prev_states))
        gamma_minus = params_tensor.get("gamma_minus", torch.zeros_like(prev_states))
        ds_dt = gamma_plus * observables - gamma_minus * prev_states

        bulk_handled = self._feature.on_scan_batch(
            context=context,
            history=history,
            phi=phi_tensor,
            params=params_tensor,
            observables=observables,
            ds_dt=ds_dt,
        )

        if bulk_handled:
            if hasattr(layer_ref, "_set_phi_history_sequence"):
                layer_ref._set_phi_history_sequence(phi_tensor) # type: ignore[union-attr]
            if hasattr(layer_ref, "_set_g_history_sequence"):
                layer_ref._set_g_history_sequence(observables) # type: ignore[union-attr]
            if hasattr(layer_ref, "_set_state_history_sequence"):
                layer_ref._set_state_history_sequence(next_states) # type: ignore[union-attr]
        else:
            for t in range(steps):
                step_params = {key: tensor[:, t, :] for key, tensor in params_tensor.items()}
                payload = StepPayload(
                    state=next_states[:, t, :],
                    phi=phi_tensor[:, t, :],
                    params=step_params,
                    ds_dt=ds_dt[:, t, :],
                )
                payload.extras["g"] = observables[:, t, :]
                payload.extras["prev_state"] = prev_states[:, t, :]
                if hasattr(layer_ref, "_add_phi_to_history"):
                    layer_ref._add_phi_to_history(phi_tensor[:, t, :]) # type: ignore[union-attr]
                if hasattr(layer_ref, "_add_g_to_history"):
                    layer_ref._add_g_to_history(observables[:, t, :]) # type: ignore[union-attr]
                if hasattr(layer_ref, "_add_state_to_history"):
                    layer_ref._add_state_to_history(next_states[:, t, :]) # type: ignore[union-attr]
                self._feature.on_after_step(
                    context=context,
                    step_index=t,
                    payload=payload,
                )
                next_states[:, t, :] = payload.state

        self._feature.on_integration_end(
            context=context,
            history=history,
            params=params_tensor,
        )
        history[:, 1:, :] = next_states
        return history


def sign_log_scan(
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    x0: float | torch.Tensor = 0.0,
    *,
    atol: float = 1e-12,
) -> torch.Tensor:
    batch, steps, dim = a_t.shape
    device = a_t.device
    dtype = a_t.dtype
    BD = batch * dim
    eps = torch.finfo(dtype).eps

    def _flatten(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).reshape(BD, steps)

    def _unflatten(y: torch.Tensor) -> torch.Tensor:
        return y.reshape(batch, dim, -1).permute(0, 2, 1)

    a = _flatten(a_t)
    b = _flatten(b_t)

    if isinstance(x0, torch.Tensor):
        x0 = x0.to(device=device, dtype=dtype)
        if x0.dim() == 0:
            x0 = x0.view(1)
        if x0.dim() > 1:
            x0 = x0.reshape(-1)
        if x0.numel() == 1:
            x0 = x0.expand(BD)
        elif x0.numel() != BD:
            msg = f"Initial state tensor must have {BD} elements; received {x0.numel()}"
            raise ValueError(
                msg,
            )
    else:
        x0 = torch.tensor(float(x0), device=device, dtype=dtype).expand(BD)

    ones = torch.ones_like(a)
    unit_a_rows = (a - ones).abs().amax(dim=1) <= atol
    x_hist_unit = None
    if unit_a_rows.any():
        vals = torch.empty(BD, b.shape[1] + 1, device=device, dtype=dtype)
        vals[:, 0] = x0
        vals[:, 1:] = b
        x_hist_unit = torch.cumsum(vals, dim=1)

    pos_rows = (a.min(dim=1).values >= 0) & (b.min(dim=1).values >= 0) & (x0 >= 0)
    pos_rows = pos_rows & (~unit_a_rows)
    x_hist_pos = None
    if pos_rows.any():
        a_pos = a[pos_rows]
        b_pos = b[pos_rows]
        x0_pos = x0[pos_rows]
        log_a = torch.log(a_pos.clamp_min(eps))
        a_star_no_pad = torch.cumsum(log_a, dim=1)
        a_star = F.pad(a_star_no_pad, (1, 0), value=0.0)
        log_b = torch.log(b_pos.clamp_min(eps))
        log_x0 = torch.where(x0_pos > 0, torch.log(x0_pos), torch.full_like(x0_pos, float("-inf")))
        log_values = torch.cat([log_x0.view(-1, 1), log_b], dim=1)
        log_sum = torch.logcumsumexp(log_values - a_star, dim=1)
        log_x = a_star + log_sum
        x_hist_pos = torch.exp(log_x)

    gen_rows = ~(unit_a_rows | pos_rows)
    a_gen = a[gen_rows]
    b_gen = b[gen_rows]
    x0_gen = x0[gen_rows]

    par_a = (a_gen < 0).to(torch.int64)
    logabs_a = torch.log(a_gen.abs().clamp_min(eps))
    a_star_log_no_pad = torch.cumsum(logabs_a, dim=1)
    a_star_log = F.pad(a_star_log_no_pad, (1, 0), value=0.0)
    par_a_star_no_pad = torch.cumsum(par_a, dim=1) & 1
    par_a_star = F.pad(par_a_star_no_pad, (1, 0), value=0)

    par_b = (b_gen < 0).to(torch.int64)
    logabs_b = torch.log(b_gen.abs().clamp_min(eps))

    N = gen_rows.sum().item()
    log_values = torch.empty(N, steps + 1, device=device, dtype=dtype) # type: ignore[arg-type]
    log_values[:, 0] = torch.where(x0_gen > 0, torch.log(x0_gen), torch.full_like(x0_gen, float("-inf")))
    log_values[:, 1:] = logabs_b

    par_values = torch.zeros(N, steps + 1, device=device, dtype=torch.int64) # type: ignore[arg-type]
    par_values[:, 1:] = par_b ^ par_a_star[:, 1:]

    adj_logs = log_values - a_star_log

    neg_inf = torch.full((), float("-inf"), device=device, dtype=dtype)
    pos_logs = torch.where(par_values == 0, adj_logs, neg_inf)
    neg_logs = torch.where(par_values == 1, adj_logs, neg_inf)

    Lp = torch.logcumsumexp(pos_logs, dim=1)
    Ln = torch.logcumsumexp(neg_logs, dim=1)

    A = torch.maximum(Lp, Ln)
    B = torch.minimum(Lp, Ln)
    mask_A_inf = torch.isneginf(A)
    z = torch.exp(B - A)
    logabs_sum = torch.where(mask_A_inf, A, A + torch.log1p(-z))
    par_sum = (Ln > Lp).to(torch.int64)

    logabs_x = a_star_log + logabs_sum
    par_x = (par_a_star ^ par_sum).to(torch.int64)

    x_mag = torch.exp(torch.clamp_max(logabs_x, 80))
    x_mag = torch.where(torch.isneginf(logabs_x), torch.zeros_like(x_mag), x_mag)
    sign = torch.where(par_x == 1, -torch.ones_like(x_mag), torch.ones_like(x_mag))
    x_hist_gen = sign * x_mag

    x_hist = torch.empty(BD, b.shape[1] + 1, device=device, dtype=dtype)
    if unit_a_rows.any():
        x_hist[unit_a_rows] = x_hist_unit[unit_a_rows] # type: ignore[index]
    if pos_rows.any():
        x_hist[pos_rows] = x_hist_pos # type: ignore[assignment]
    x_hist[gen_rows] = x_hist_gen
    return _unflatten(x_hist)


def _broadcast_param(
    tensor: torch.Tensor,
    batch: int,
    steps: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.dim() == 0:
        return tensor.view(1, 1, 1).expand(batch, steps, dim)
    if tensor.dim() == 1:
        if tensor.shape[0] == dim:
            base = tensor.view(1, 1, -1)
            return base.expand(batch, steps, -1)
        if tensor.shape[0] == batch:
            base = tensor.view(batch, 1, 1)
            return base.expand(-1, steps, dim)
        if tensor.shape[0] == 1:
            return tensor.view(1, 1, 1).expand(batch, steps, dim)
        msg = f"Cannot broadcast parameter of shape {tuple(tensor.shape)}"
        raise ValueError(msg)
    if tensor.dim() == 2:
        if tensor.shape == (dim, dim):
            base = tensor.unsqueeze(0)
            return base.expand(batch, -1, -1)
        if tensor.shape == (batch, dim):
            return tensor.unsqueeze(1).expand(-1, steps, -1)
        if tensor.shape == (steps, dim):
            return tensor.unsqueeze(0).expand(batch, -1, -1)
    if tensor.dim() == 3:
        out = tensor
        if out.shape[0] == 1 and out.shape[1] == 1:
            out = out.expand(batch, steps, dim)
        elif out.shape[0] == 1:
            out = out.expand(batch, -1, -1)
        elif out.shape[1] == 1:
            out = out.expand(-1, steps, -1)
        return out
    msg = f"Unsupported parameter tensor rank: {tensor.dim()}"
    raise ValueError(msg)


__all__ = ["CoefficientProvider", "ParallelScanSolver", "sign_log_scan"]
