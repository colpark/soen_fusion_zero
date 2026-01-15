"""Forward Euler solver skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import weakref

import torch

from soen_toolkit.core.layers.common.features import (
    CompositeFeature,
    FeatureHook,
    StepPayload,
)

from ._phi_utils import compute_phi_with_offset
from .base import SolverBase, SupportsState

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(slots=True)
class _ForwardEulerContext:
    dt: torch.Tensor


class ForwardEulerSolver(SolverBase):
    """Forward Euler integration for sequence inputs."""

    def __init__(
        self,
        *,
        dynamics,
        feature: FeatureHook | None = None,
        phi_transform=None,
        layer=None,
    ) -> None:
        super().__init__()
        self._dynamics = dynamics
        if isinstance(feature, CompositeFeature):
            self._feature = feature
        elif feature is None:
            self._feature = CompositeFeature()
        else:
            self._feature = CompositeFeature([feature])
        self._phi_transform = phi_transform
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

        history = torch.empty(batch, steps + 1, dim, device=phi.device, dtype=phi.dtype)
        history[:, 0, :] = state_tensor

        expanded_params = {name: _expand_param(tensor, batch, dim, phi.device, phi.dtype) for name, tensor in params.items()}

        context = _ForwardEulerContext(dt=dt)
        layer_ref = self._layer_ref() if self._layer_ref is not None else None
        if layer_ref is not None and hasattr(layer_ref, "_clear_histories"):
            layer_ref._clear_histories()
        if layer_ref is not None and hasattr(self._feature, "attach_layer"):
            self._feature.attach_layer(layer_ref)
        self._feature.on_integration_start(
            context=context,
            state=state_tensor,
            phi=phi,
            params=expanded_params,
        )

        current_state = state_tensor
        for t in range(steps):
            payload = StepPayload(state=current_state, phi=phi[:, t, :], params=expanded_params)
            payload.extras["prev_state"] = current_state.detach()
            # Apply feature hooks before any phi adjustments (noise, etc.).
            payload = self._feature.on_before_step(
                context=context,
                step_index=t,
                payload=payload,
            )
            if payload.params is not expanded_params:
                expanded_params = dict(payload.params)

            # Apply internal connectivity (if provided) followed by phi offset.
            if self._phi_transform is not None:
                payload.phi = self._phi_transform(
                    state=payload.state,
                    phi=payload.phi,
                    params=payload.params,
                )
            payload.phi = compute_phi_with_offset(payload.phi, payload.params)

            ds_dt = self._dynamics(payload.state, payload.phi, payload.params)
            payload.ds_dt = ds_dt
            current_state = payload.state + dt * ds_dt
            payload.state = current_state
            if layer_ref is not None and hasattr(layer_ref, "_add_phi_to_history"):
                layer_ref._add_phi_to_history(payload.phi)

            if ds_dt is not None:
                g_val = payload.extras.get("g")
                prev_state = payload.extras.get("prev_state")
                gp = payload.params.get("gamma_plus") if isinstance(payload.params, dict) else None
                if g_val is None and gp is not None and prev_state is not None:
                    gm = payload.params.get("gamma_minus") if isinstance(payload.params, dict) else None
                    eps = 1e-12
                    numerator = ds_dt
                    if gm is not None:
                        numerator = numerator + gm * prev_state
                    g_val = numerator / gp.clamp_min(eps)
                    payload.extras["g"] = g_val

            self._feature.on_after_step(
                context=context,
                step_index=t,
                payload=payload,
            )
            current_state = payload.state
            if layer_ref is not None and hasattr(layer_ref, "_add_g_to_history"):
                if payload.extras.get("g") is not None:
                    layer_ref._add_g_to_history(payload.extras["g"])
                elif ds_dt is not None:
                    layer_ref._add_g_to_history(ds_dt)
            if layer_ref is not None and hasattr(layer_ref, "_add_state_to_history"):
                layer_ref._add_state_to_history(current_state)
            history[:, t + 1, :] = current_state

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
        raise ValueError(
            msg,
        )
    msg = f"Unsupported parameter tensor rank: {tensor.dim()}"
    raise ValueError(msg)


__all__ = ["ForwardEulerSolver"]
