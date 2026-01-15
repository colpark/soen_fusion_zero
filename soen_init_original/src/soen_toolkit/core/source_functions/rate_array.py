# FILEPATH: src/soen_toolkit/core/source_functions/rate_array.py

from __future__ import annotations

from pathlib import Path
import pickle
from typing import NoReturn

import torch
from torch.nn import Module
from torch.nn.functional import grid_sample

from soen_toolkit.utils import paths

from .base import SourceFunctionBase, SourceFunctionInfo

"""
squid_current is the total current in the integration loop (SQUID CURRENT)
"""


class RateArraySource(SourceFunctionBase, Module):
    info = SourceFunctionInfo(
        key="RateArray",
        title="Rate Array",
        description="Interpolated SOEN rate array lookup.",
        category="SOEN",
        uses_squid_current=True,
        supports_coefficients=False,
    )

    def __init__(
        self,
        ib_list: torch.Tensor | None = None,
        phi_array: torch.Tensor | None = None,
        g_array: torch.Tensor | None = None,
        data_path: Path | None = None,
    ) -> None:
        super().__init__()
        Module.__init__(self)
        ib_list, phi_array, g_array = self._resolve_arrays(ib_list, phi_array, g_array, data_path)
        self.register_buffer("g_table", g_array.unsqueeze(0).unsqueeze(0))
        self.ib_list = ib_list
        self.phi_list = phi_array
        self.ib_min = ib_list[0].item()
        self.ib_max = ib_list[-1].item()

    def _resolve_arrays(
        self,
        ib_list: torch.Tensor | None,
        phi_array: torch.Tensor | None,
        g_array: torch.Tensor | None,
        data_path: Path | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ib_list is not None and phi_array is not None and g_array is not None:
            return ib_list, phi_array, g_array
        data_bytes = self._load_bytes(data_path)
        data = pickle.loads(data_bytes)
        return (
            ib_list or torch.tensor(data["ib_list"], dtype=torch.float32),
            phi_array or torch.tensor(data["phi_array"], dtype=torch.float32),
            g_array or torch.tensor(data["g_array"], dtype=torch.float32),
        )

    def _load_bytes(self, data_path: Path | None) -> bytes:
        candidates = []
        if data_path:
            candidates.append(Path(data_path))
        candidates.append(Path(__file__).with_name("base_rate_array.soen"))
        candidates.append(Path(paths.BASE_RATE_ARRAY_PATH))
        for path in candidates:
            if path.is_file():
                return path.read_bytes()
        msg = "base_rate_array.soen could not be located"
        raise FileNotFoundError(msg)

    def g(self, phi: torch.Tensor, *, squid_current: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        # Note: squid_current is the total current through the SQUID part of the dendrite circuit
        if squid_current is None:
            squid_current = torch.full_like(phi, 1.7)
        return self._interpolate(phi, squid_current)

    def forward(self, phi: torch.Tensor, squid_current: torch.Tensor | None = None, **kwargs):
        return self.g(phi, squid_current=squid_current, **kwargs)

    def get_coefficients(self, *args, **kwargs) -> NoReturn:  # pragma: no cover - not supported
        raise NotImplementedError

    def _interpolate(self, phi: torch.Tensor, squid_current: torch.Tensor) -> torch.Tensor:
        device, dtype = self.g_table.device, self.g_table.dtype
        phi, squid_current = torch.broadcast_tensors(
            phi.to(device=device, dtype=dtype),
            squid_current.to(device=device, dtype=dtype),
        )
        phi_mod = torch.remainder(phi, 1.0)
        phi_eff = torch.minimum(phi_mod, 1.0 - phi_mod)
        norm_phi = 4.0 * phi_eff - 1.0
        norm_ib = 2.0 * ((squid_current - self.ib_min) / (self.ib_max - self.ib_min)) - 1.0

        # Use manual interpolation on MPS (grid_sample backward not supported)
        if device.type == 'mps':
            return self._interpolate_manual(norm_phi, norm_ib, phi.shape)

        # Use grid_sample on CUDA/CPU (faster)
        grid = torch.stack((norm_phi, norm_ib), dim=-1)
        grid_flat = grid.reshape(-1, 1, 1, 2)
        out = grid_sample(
            self.g_table.expand(grid_flat.size(0), -1, -1, -1),
            grid_flat,
            mode="bilinear",
            align_corners=True,
        )
        return out.view(phi.shape)

    def _interpolate_manual(self, norm_phi: torch.Tensor, norm_ib: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        """Manual bilinear interpolation for MPS compatibility."""
        # g_table is [1, 1, H, W] where H=ib axis, W=phi axis
        H, W = self.g_table.shape[2], self.g_table.shape[3]

        # Flatten input tensors for easier indexing
        norm_phi_flat = norm_phi.reshape(-1)
        norm_ib_flat = norm_ib.reshape(-1)

        # Convert normalized coords [-1, 1] to pixel coords [0, W-1] and [0, H-1]
        # grid_sample with align_corners=True maps -1 to 0 and 1 to size-1
        x = (norm_phi_flat + 1.0) * 0.5 * (W - 1) # phi is columns (W)
        y = (norm_ib_flat + 1.0) * 0.5 * (H - 1)  # ib is rows (H)

        # Clamp to valid range
        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        # Get integer coordinates
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = torch.minimum(x0 + 1, torch.tensor(W - 1, device=x.device, dtype=torch.long))
        y1 = torch.minimum(y0 + 1, torch.tensor(H - 1, device=y.device, dtype=torch.long))

        # Get fractional parts
        fx = x - x0.float()
        fy = y - y0.float()

        # Gather values from 4 corners (indexing into [1, 1, H, W] table)
        g00 = self.g_table[0, 0, y0, x0]
        g01 = self.g_table[0, 0, y0, x1]
        g10 = self.g_table[0, 0, y1, x0]
        g11 = self.g_table[0, 0, y1, x1]

        # Bilinear interpolation
        out = (g00 * (1 - fx) * (1 - fy) +
               g01 * fx * (1 - fy) +
               g10 * (1 - fx) * fy +
               g11 * fx * fy)

        return out.view(output_shape)
