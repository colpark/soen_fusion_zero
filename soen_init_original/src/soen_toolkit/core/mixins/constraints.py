# FILEPATH: src/soen_toolkit/core/mixins/constraints.py

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

import torch

from soen_toolkit.core.layers.common.metadata import apply_param_constraints

if TYPE_CHECKING:
    from torch import nn

    from soen_toolkit.core.configs import ConnectionConfig, LayerConfig


class ConstraintsMixin:
    """Mixin providing constraint enforcement methods."""

    if TYPE_CHECKING:
        from torch import nn
        from soen_toolkit.core.configs import ConnectionConfig, LayerConfig

        # Attributes expected from the composed class
        connection_masks: dict[str, torch.Tensor]
        connections: nn.ParameterDict
        connection_constraints: dict[str, dict[str, float]]
        layers_config: list[LayerConfig]
        layers: nn.ModuleList
    def apply_masks(self) -> None:
        """Apply connectivity masks to enforce connection patterns."""
        with torch.no_grad():
            for key, mask in self.connection_masks.items():
                if key in self.connections:
                    self.connections[key].data *= mask.to(self.connections[key].device)

    def enforce_param_constraints(self) -> None:
        """Enforce constraints on connectivity weights and call layer constraint methods."""
        with torch.no_grad():
            for cfg, layer in zip(self.layers_config, self.layers, strict=False):
                l_any = cast(Any, layer)
                with contextlib.suppress(Exception):
                    l_any._apply_constraints()
                try:
                    layer_type = cfg.layer_type
                    apply_param_constraints(l_any, layer_type)
                except Exception:
                    pass

        with torch.no_grad():
            min_mats = getattr(self, "connection_constraint_min_matrices", {})
            max_mats = getattr(self, "connection_constraint_max_matrices", {})
            for key, param in self.connections.items():
                if key in min_mats and key in max_mats:
                    min_mat = min_mats[key].to(param.device)
                    max_mat = max_mats[key].to(param.device)
                    param.data = torch.clamp(param.data, min=min_mat, max=max_mat)
                else:
                    constraint = self.connection_constraints.get(
                        key,
                        {"min": -float("inf"), "max": float("inf")},
                    )
                    if constraint:
                        param.data.clamp_(
                            min=constraint.get("min", -float("inf")),
                            max=constraint.get("max", float("inf")),
                        )

        self.apply_masks()
