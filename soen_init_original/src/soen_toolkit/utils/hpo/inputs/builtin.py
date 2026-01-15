from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from . import InputProvider


def _maybe_scale(x: torch.Tensor, mn: float | None, mx: float | None) -> torch.Tensor:
    if mn is None or mx is None:
        return x
    mn = float(mn)
    mx = float(mx)
    if mx == mn:
        return torch.full_like(x, mn)
    cur_min = torch.nanmin(x)
    cur_max = torch.nanmax(x)
    if (cur_max - cur_min) <= 0:
        return torch.full_like(x, mn)
    return mn + (mx - mn) * (x - cur_min) / (cur_max - cur_min)  # small eps not necessary for float32 range


@dataclass
class _Common:
    config: dict
    base_spec: dict
    device: torch.device

    def _dims(self) -> int:
        # Extract input dim from base_spec (layer 0 params.dim), fallback to 1
        try:
            for layer in self.base_spec.get("layers") or []:
                if int(layer.get("layer_id", -1)) == 0:
                    d = layer.get("params", {}).get("dim")
                    if isinstance(d, int) and d > 0:
                        return d
            for layer in self.base_spec.get("layers") or []:
                d = layer.get("params", {}).get("dim")
                if isinstance(d, int) and d > 0:
                    return d
        except Exception:
            pass
        return 1


class WhiteNoiseInput(InputProvider, _Common):
    def get_batch(self, *, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        delta_n = float(self.config.get("delta_n", 1.0))
        x = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=torch.float32) * math.sqrt(delta_n)
        return _maybe_scale(x, self.config.get("input_scale_min"), self.config.get("input_scale_max"))


class ColoredNoiseInput(InputProvider, _Common):
    def get_batch(self, *, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        beta = float(self.config.get("colored_beta", 2.0))
        x = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=torch.float32)
        X = torch.fft.rfft(x, dim=1)
        # Calculate proper default dt_s: assume dt_units=37, convert to seconds
        default_dt_s = 37 * 1.28e-12  # ~1e-10 seconds, dependent on omega_c specified in physics/constants.py
        dt_s = float(self.config.get("dt_s", default_dt_s))
        f = torch.fft.rfftfreq(seq_len, d=dt_s).to(device=self.device)
        # Avoid a dominant DC; scale relative to first non-zero bin
        if f.numel() > 1:
            f0 = f[1]
            f = torch.where(f == 0, f0, f)
            gain = (f / f0) ** (-beta * 0.5)
        else:
            gain = torch.ones_like(f)
        Xc = X * gain.view(1, -1, 1)
        y = torch.fft.irfft(Xc, n=seq_len, dim=1)
        # Zero-mean and unit-std per sample to avoid flat lines
        y = y - y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-12)
        y = y / std
        return _maybe_scale(y, self.config.get("input_scale_min"), self.config.get("input_scale_max"))


class GPRBFInput(InputProvider, _Common):
    def get_batch(self, *, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        sigma = float(self.config.get("input_sigma", 1.0))
        ell_ns = float(self.config.get("input_ell_ns", 0.1))
        # Calculate proper default dt_s: assume dt_units=37, convert to seconds
        default_dt_s = 37 * 1.28e-12  # ~1e-10 seconds, dependent on omega_c specified in physics/constants.py
        dt_s = float(self.config.get("dt_s", default_dt_s))
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32) * dt_s
        tau = t[:, None] - t[None, :]
        ell = torch.tensor(ell_ns * 1e-9, device=self.device, dtype=torch.float32)
        K = (sigma**2) * torch.exp(-0.5 * (tau / ell) ** 2)
        K = K + 1e-6 * torch.eye(seq_len, device=self.device, dtype=torch.float32)
        L = torch.linalg.cholesky(K)
        z = torch.randn(batch_size, dim, seq_len, device=self.device, dtype=torch.float32)
        y = torch.matmul(z, L.T)
        y = y.transpose(1, 2)  # [B,T,D]
        return _maybe_scale(y, self.config.get("input_scale_min"), self.config.get("input_scale_max"))


class LogSlopeNoiseInput(InputProvider, _Common):
    def get_batch(self, *, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        """Generate band-limited colored noise with specified dB/decade slope."""
        slope_db_per_dec = float(self.config.get("log_slope_db_per_dec", -20.0))
        fmin_frac = float(self.config.get("log_slope_fmin_frac", 0.01))
        fmax_frac = float(self.config.get("log_slope_fmax_frac", 0.5))

        # Calculate proper default dt_s
        default_dt_s = 37 * 1.28e-12  # ~1e-10 seconds
        dt_s = float(self.config.get("dt_s", default_dt_s))

        # Generate white noise
        x = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=torch.float32)
        X = torch.fft.rfft(x, dim=1)

        # Frequency array
        f = torch.fft.rfftfreq(seq_len, d=dt_s).to(device=self.device)
        f[0] = 1e-12  # Avoid division by zero

        # Calculate frequency limits
        nyquist_freq = 1.0 / (2.0 * dt_s)
        fmin = fmin_frac * nyquist_freq
        fmax = fmax_frac * nyquist_freq

        # Create mask for frequency band
        mask = (f >= fmin) & (f <= fmax)

        # Calculate gain: convert dB/decade slope to power law exponent
        # slope_db_per_dec = 10 * log10(f2/f1) * beta where beta is power law exponent
        # For -20 dB/decade: beta = -2, so gain = f^(-1)
        beta = slope_db_per_dec / 10.0  # Convert dB to power law exponent

        # Apply gain only within the specified frequency band
        gain = torch.ones_like(f)
        gain[mask] = (f[mask] / fmin) ** (beta / 2.0)

        # Apply gain and convert back
        Xc = X * gain.view(1, -1, 1)
        y = torch.fft.irfft(Xc, n=seq_len, dim=1)

        return _maybe_scale(y, self.config.get("input_scale_min"), self.config.get("input_scale_max"))


class HDF5DatasetInput(InputProvider, _Common):
    def get_batch(self, *, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        from soen_toolkit.training.data.dataloaders import GenericHDF5Dataset

        h5_path = str(self.config.get("input_h5_path", ""))
        split = str(self.config.get("input_h5_split")) if self.config.get("input_h5_split") else None
        # Try auto-detect split if not provided
        if not split:
            try:
                import h5py

                with h5py.File(h5_path, "r", swmr=True, libver="latest", locking=False) as f:
                    for name in ("train", "val", "test"):
                        if name in f and "data" in f[name]:
                            split = name
                            break
            except Exception:
                pass
        ds = GenericHDF5Dataset(
            hdf5_path=h5_path,
            split=split,
            data_key=str(self.config.get("input_h5_data_key", "data")),
            cache_in_memory=False,
            target_seq_len=int(seq_len),
            scale_min=(float(self.config.get("input_scale_min")) if self.config.get("input_scale_min") is not None else None),
            scale_max=(float(self.config.get("input_scale_max")) if self.config.get("input_scale_max") is not None else None),
        )
        if len(ds) < 1:
            msg = "HDF5 dataset is empty or not found"
            raise RuntimeError(msg)
        # Sample without replacement if possible
        import numpy as _np

        idxs = _np.random.choice(len(ds), size=batch_size, replace=(len(ds) < batch_size))
        batch = [ds[i][0].numpy() if hasattr(ds[i][0], "numpy") else np.asarray(ds[i][0]) for i in idxs]
        arr = _np.stack(batch)  # [B,T,D]
        # Adjust feature dim to dim expected by model
        b, t, d = arr.shape
        if d != dim:
            if d > dim:
                arr = arr[:, :, :dim]
            else:
                pad = _np.zeros((b, t, dim - d), dtype=arr.dtype)
                arr = _np.concatenate([arr, pad], axis=2)
        return torch.from_numpy(arr).to(self.device, dtype=torch.float32)
