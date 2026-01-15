# FILEPATH: src/soen_toolkit/core/mixins/quantization.py

from __future__ import annotations

from collections.abc import Iterator
import copy
from typing import TYPE_CHECKING, Any, cast

import torch

from soen_toolkit.utils.quantization import build_codebook_from_params, ste_snap

if TYPE_CHECKING:
    from torch import nn

    from soen_toolkit.core.configs import LayerConfig


class QuantizationMixin:
    """Mixin providing quantization methods."""

    if TYPE_CHECKING:
        # Attributes expected from the composed class
        connections: nn.ParameterDict
        layers_config: list[LayerConfig]
        layers: nn.ModuleList

        # QAT attributes added by this mixin
        _qat_ste_active: bool
        _qat_codebook: torch.Tensor | None
        _qat_target_connection_names: set[str] | None
        _qat_stochastic_rounding: bool

        def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...
    @staticmethod
    def _generate_uniform_codebook(min_val: float, max_val: float, num_levels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if num_levels <= 0:
            return torch.tensor([], device=device, dtype=dtype)

        include_zero = True
        remaining = max(0, num_levels - (1 if include_zero else 0))

        if remaining == 0:
            levels = torch.tensor([0.0], device=device, dtype=dtype)
        else:
            if min_val <= 0.0 <= max_val:
                neg_count = remaining // 2
                pos_count = remaining - neg_count
                neg = torch.linspace(min_val, 0.0, neg_count + 1, device=device, dtype=dtype)[:-1] if neg_count > 0 else torch.tensor([], device=device, dtype=dtype)
                pos = torch.linspace(0.0, max_val, pos_count + 1, device=device, dtype=dtype)[1:] if pos_count > 0 else torch.tensor([], device=device, dtype=dtype)
                levels = torch.cat([neg, torch.tensor([0.0], device=device, dtype=dtype), pos])
            else:
                spread = torch.linspace(min_val, max_val, remaining, device=device, dtype=dtype)
                levels = torch.cat([spread, torch.tensor([0.0], device=device, dtype=dtype)])

            levels = torch.unique(torch.sort(levels)[0])
            while levels.numel() < num_levels:
                eps = 1e-12
                levels = torch.cat(
                    [
                        torch.tensor([min_val - eps], device=device, dtype=dtype),
                        levels,
                        torch.tensor([max_val + eps], device=device, dtype=dtype),
                    ]
                )
                levels = torch.unique(torch.sort(levels)[0])

        return levels

    def quantize(
        self,
        *,
        bits: int | None = None,
        min: float | None = None,
        max: float | None = None,
        format: str = "uniform",
        levels: int | None = None,
        connections: list[str] | None = None,
        include_non_learnable: bool = False,
        in_place: bool = False,
    ):
        if format.lower() != "uniform":
            msg = f"Unsupported quantisation format '{format}'. Only 'uniform' is supported."
            raise ValueError(msg)

        if (bits is None and levels is None) or (bits is not None and levels is not None):
            msg = "Specify either 'bits' or 'levels' (exclusively)."
            raise ValueError(msg)

        if min is None or max is None:
            msg = "Both 'min' and 'max' must be provided for quantisation."
            raise ValueError(msg)

        if levels is not None:
            num_levels = int(levels)
        else:
            if bits is None:
                msg = "'bits' or 'levels' must be provided"
                raise ValueError(msg)
            b = int(bits)
            if b < 0:
                msg = "'bits' must be non-negative"
                raise ValueError(msg)
            num_levels = (2**b) + 1

        target_model = self if in_place else copy.deepcopy(self)

        first_param = None
        for p in target_model.connections.values():
            if isinstance(p, torch.Tensor):
                first_param = p
                break

        device = first_param.device if first_param is not None else torch.device("cpu")
        dtype = first_param.dtype if first_param is not None else torch.float32

        codebook = self._generate_uniform_codebook(float(min), float(max), int(num_levels), device, dtype)

        target_names = set(connections) if connections is not None else None

        with torch.no_grad():
            for name, param in target_model.connections.items():
                if not isinstance(param, torch.Tensor):
                    continue
                if target_names is not None and name not in target_names:
                    continue
                if (not include_non_learnable) and hasattr(param, "requires_grad") and (not param.requires_grad):
                    continue

                flat = param.view(-1)
                diffs = (flat.unsqueeze(1) - codebook.unsqueeze(0)).abs()
                idx = diffs.argmin(dim=1)
                snapped = codebook[idx].view_as(param)
                param.copy_(snapped)

        return target_model

    def quantise(
        self,
        *,
        bits: int | None = None,
        min: float | None = None,
        max: float | None = None,
        format: str = "uniform",
        levels: int | None = None,
        connections: list[str] | None = None,
        include_non_learnable: bool = False,
        in_place: bool = False,
    ):
        return self.quantize(
            bits=bits,
            min=min,
            max=max,
            format=format,
            levels=levels,
            connections=connections,
            include_non_learnable=include_non_learnable,
            in_place=in_place,
        )

    def enable_qat_ste(
        self,
        *,
        min_val: float,
        max_val: float,
        bits: int | None = None,
        levels: int | None = None,
        connections: list[str] | None = None,
        stochastic_rounding: bool = False,
    ) -> None:
        try:
            first_param = next(self.parameters())
            device = first_param.device
            dtype = first_param.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32

        codebook = build_codebook_from_params(
            min_val=float(min_val),
            max_val=float(max_val),
            bits=bits,
            levels=levels,
            device=device,
            dtype=dtype,
        )

        self._qat_codebook = codebook
        self._qat_ste_active = True
        if connections is not None:
            norm: set[str] = set()
            for name in connections:
                if isinstance(name, str) and name.startswith("internal_"):
                    try:
                        lid = int(name.split("_")[1])
                        norm.add(f"J_{lid}_to_{lid}")
                    except Exception:
                        norm.add(name)
                else:
                    norm.add(name)
            self._qat_target_connection_names = norm
        else:
            self._qat_target_connection_names = None
        self._qat_stochastic_rounding = bool(stochastic_rounding)

        for cfg, layer in zip(self.layers_config, self.layers, strict=False):
            l_any = cast(Any, layer)
            internal_key = f"J_{cfg.layer_id}_to_{cfg.layer_id}"
            applies = self._qat_target_connection_names is None or internal_key in self._qat_target_connection_names
            l_any._qat_ste_active = True
            l_any._qat_codebook = self._qat_codebook
            l_any._qat_internal_active = applies
            l_any._qat_stochastic_rounding = self._qat_stochastic_rounding

    def disable_qat_ste(self) -> None:
        self._qat_ste_active = False
        self._qat_codebook = None
        self._qat_target_connection_names = None
        self._qat_stochastic_rounding = False
        for layer in self.layers:
            l_any = cast(Any, layer)
            if hasattr(l_any, "_qat_ste_active"):
                l_any._qat_ste_active = False
            if hasattr(l_any, "_qat_internal_active"):
                l_any._qat_internal_active = False
            if hasattr(l_any, "_qat_stochastic_rounding"):
                l_any._qat_stochastic_rounding = False

    def _apply_qat_ste_if_enabled(self, name: str, param: torch.Tensor) -> torch.Tensor:
        if not self._qat_ste_active or self._qat_codebook is None:
            return param
        if self._qat_target_connection_names is not None and name not in self._qat_target_connection_names:
            return param
        return ste_snap(param, self._qat_codebook, stochastic=self._qat_stochastic_rounding)

    # Sign preconditioning removed entirely
