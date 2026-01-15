"""Internal connectivity helpers shared by physical layers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .connectivity_builders import build_connectivity


class ConnectivityModule(nn.Module):
    """Encapsulate an ``internal_J`` style connectivity matrix."""

    # Type hint for registered parameter (mypy doesn't track register_parameter)
    weight: torch.Tensor

    def __init__(
        self,
        *,
        dim: int,
        init: torch.Tensor | None = None,
        learnable: bool = True,
        constraints: Mapping[str, float] | None = None,
        default_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        weight = init if init is not None else torch.zeros(self.dim, self.dim)
        weight = weight.to(dtype=torch.float32)
        param = nn.Parameter(weight, requires_grad=bool(learnable))
        self.register_parameter("weight", param)
        if not learnable:
            param.requires_grad_(False)
        self.learnable = learnable
        self.constraints = dict(constraints or {})
        self.default_bounds = default_bounds

    def materialised(self) -> torch.Tensor:
        """Return the connectivity weight for use in forward computations.

        Returns a clone to prevent in-place constraint modifications from
        invalidating the autograd graph (which creates views via transpose).
        """
        return self.weight.clone()

    def apply_constraints(self) -> None:
        min_val = self.constraints.get("min") if self.constraints else None
        max_val = self.constraints.get("max") if self.constraints else None

        if min_val is None and max_val is None and self.default_bounds is None:
            return

        if min_val is None and self.default_bounds is not None:
            min_val = self.default_bounds[0]
        if max_val is None and self.default_bounds is not None:
            max_val = self.default_bounds[1]

        if min_val is None and max_val is None:
            return

        with torch.no_grad():
            if min_val is not None:
                self.weight.clamp_(min=min_val)
            if max_val is not None:
                self.weight.clamp_(max=max_val)


def apply_connectivity(
    *,
    state: torch.Tensor,
    phi: torch.Tensor,
    params: Mapping[str, torch.Tensor],
    key: str = "internal_J",
) -> torch.Tensor:
    weight = params.get(key)
    if weight is None:
        return phi
    weight = weight.to(device=phi.device, dtype=phi.dtype)
    if weight.dim() == 3 and weight.shape[0] == state.shape[0]:
        # Per-batch matrices
        return phi + torch.matmul(state.unsqueeze(-2), weight.transpose(-1, -2)).squeeze(-2)
    return phi + state @ weight.transpose(-1, -2)


def resolve_connectivity_matrix(
    *,
    dim: int,
    connectivity: torch.Tensor | None,
    spec: Mapping[str, object] | None,
) -> torch.Tensor | None:
    if connectivity is not None:
        return connectivity.to(dtype=torch.float32)
    if spec is None:
        return None
    kind = spec.get("type")
    if kind is None:
        msg = "connectivity spec must include 'type'"
        raise ValueError(msg)
    params = spec.get("params")
    if params is not None and not isinstance(params, Mapping):
        msg = "connectivity spec 'params' must be a mapping if provided"
        raise ValueError(msg)
    return build_connectivity(str(kind), from_nodes=dim, to_nodes=dim, params=params)


__all__ = ["ConnectivityModule", "apply_connectivity", "resolve_connectivity_matrix"]
