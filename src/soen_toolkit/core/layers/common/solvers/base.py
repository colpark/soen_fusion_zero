"""Solver abstractions for layers."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class SupportsState(Protocol):
    """Protocol for state objects accepted by solver implementations."""

    values: torch.Tensor


class SolverBase(nn.Module):
    """Minimal solver interface expected by new layers."""

    def integrate(
        self,
        *,
        state: SupportsState,
        phi: torch.Tensor,
        params: dict[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


__all__ = ["SolverBase", "SupportsState"]
