"""Input generation strategies for state trajectory simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
import torch

from .errors import InputGenerationError
from .settings import EncodingSettings, ScalingBounds
from .timebase import Timebase


class InputSource(Protocol):
    """Protocol for input generators."""

    def make(self, seq_len: int, dim: int, tb: Timebase, rng: np.random.Generator) -> torch.Tensor:
        """Generate input tensor of shape [seq_len, dim].

        Args:
            seq_len: Desired sequence length
            dim: Desired dimensionality (number of features/neurons)
            tb: Timebase for time/frequency conversions
            rng: Random number generator for reproducibility

        Returns:
            Input tensor of shape [seq_len, dim]
        """
        ...


@dataclass
class ConstantInput:
    """Constant value across all timesteps and dimensions."""

    value: float = 1.0

    def make(self, seq_len: int, dim: int, tb: Timebase, rng: np.random.Generator) -> torch.Tensor:
        return torch.full((seq_len, dim), self.value, dtype=torch.float32)


@dataclass
class GaussianNoise:
    """Zero-mean Gaussian noise (white noise)."""

    std: float = 0.1

    def make(self, seq_len: int, dim: int, tb: Timebase, rng: np.random.Generator) -> torch.Tensor:
        # Use numpy RNG for determinism, then convert to torch
        arr = rng.normal(0.0, self.std, size=(seq_len, dim)).astype(np.float32)
        return torch.from_numpy(arr)


@dataclass
class ColoredNoise:
    """Colored noise with 1/f^β power spectrum.

    Generated via FFT shaping: start from white noise, apply frequency-dependent
    gain in Fourier domain, then convert back to time domain.
    """

    beta: float = 2.0  # Spectral exponent (0=white, ~1=pink, ~2=brown/red)

    def make(self, seq_len: int, dim: int, tb: Timebase, rng: np.random.Generator) -> torch.Tensor:
        if seq_len < 2:
            # For extremely short sequences, fall back to white noise
            arr = rng.normal(0.0, 1.0, size=(seq_len, dim)).astype(np.float32)
            return torch.from_numpy(arr)

        # Start from white noise
        x_np = rng.normal(0.0, 1.0, size=(seq_len, dim)).astype(np.float32)
        x = torch.from_numpy(x_np)

        # FFT along time axis (dim=0)
        X = torch.fft.rfft(x, dim=0)

        # Frequency bins for rfft
        f = torch.fft.rfftfreq(seq_len, d=float(tb.step_seconds))
        f = f.view(-1, 1)  # [F, 1] for broadcasting

        # Gain ~ f^(-beta/2), avoid f=0
        eps = 1e-12
        gain = (f.clamp_min(eps)) ** (-self.beta * 0.5)

        # Zero DC component
        gain[0:1, :] = 0.0

        # Normalize gain energy
        denom = torch.sqrt(torch.mean(gain.squeeze(1) ** 2))
        if torch.isfinite(denom) and denom.item() > 0:
            gain = gain / denom

        # Apply gain
        Xc = X * gain

        # IFFT back to time domain
        xc = torch.fft.irfft(Xc, n=seq_len, dim=0)

        # Remove residual mean per column
        xc = xc - xc.mean(dim=0, keepdim=True)

        # Replace NaN/Inf if any
        xc = torch.nan_to_num(xc, nan=0.0, posinf=0.0, neginf=0.0)

        # If all zeros (degenerate case), fall back to white noise
        if float(xc.abs().sum().item()) == 0.0:
            return x

        return xc.to(dtype=torch.float32)


@dataclass
class Sinusoid:
    """Sinusoidal input: amp * sin(2π f t + phase) + offset"""

    freq_mhz: float = 10.0  # Frequency in MHz
    amp: float = 1.0
    phase_deg: float = 0.0
    offset: float = 0.0

    def make(self, seq_len: int, dim: int, tb: Timebase, rng: np.random.Generator) -> torch.Tensor:
        # Build time axis in seconds
        t = torch.arange(seq_len, dtype=torch.float32) * tb.step_seconds

        # Convert frequency to Hz and phase to radians
        f_hz = self.freq_mhz * 1e6
        phase_rad = math.radians(self.phase_deg)

        # Generate sine wave
        s = self.offset + self.amp * torch.sin(2 * math.pi * f_hz * t + phase_rad)

        # Tile to [seq_len, dim]
        return s.view(-1, 1).repeat(1, dim)


@dataclass
class SquareWave:
    """Square wave input: amp * sign(sin(2π f t + phase)) + offset"""

    freq_mhz: float = 10.0  # Frequency in MHz
    amp: float = 1.0
    phase_deg: float = 0.0
    offset: float = 0.0

    def make(self, seq_len: int, dim: int, tb: Timebase, rng: np.random.Generator) -> torch.Tensor:
        # Build time axis in seconds
        t = torch.arange(seq_len, dtype=torch.float32) * tb.step_seconds

        # Convert frequency to Hz and phase to radians
        f_hz = self.freq_mhz * 1e6
        phase_rad = math.radians(self.phase_deg)

        # Generate square wave via sign of sine
        base = torch.sin(2 * math.pi * f_hz * t + phase_rad)
        sq = torch.sign(base)
        s = self.offset + self.amp * sq

        # Tile to [seq_len, dim]
        return s.view(-1, 1).repeat(1, dim)


@dataclass
class DatasetInput:
    """Input from dataset sample with optional encoding and scaling."""

    dataset: object  # GenericHDF5Dataset
    data_index: int
    encoding: EncodingSettings
    scaling: ScalingBounds
    target_length: int

    def make(self, seq_len: int, dim: int, tb: Timebase, rng: np.random.Generator) -> torch.Tensor:
        """Load and process dataset sample."""
        # Get sample from dataset (returns tensors)
        try:
            spect_tensor, _ = self.dataset[self.data_index]
        except Exception as e:
            msg = f"Failed to load dataset sample {self.data_index}: {e}"
            raise InputGenerationError(msg) from e

        # Convert to numpy for processing
        spect = spect_tensor.numpy() if hasattr(spect_tensor, "numpy") else np.array(spect_tensor)

        # Downsample to target length
        mat = self._downsample(spect, self.target_length)

        # Apply one-hot encoding if requested
        if self.encoding.mode == "one_hot":
            mat = self._apply_one_hot(mat, self.encoding.vocab_size)

        # Apply scaling if bounds specified
        if self.scaling.min_val is not None and self.scaling.max_val is not None:
            mat = self._apply_scaling(mat, self.scaling.min_val, self.scaling.max_val)

        result = torch.tensor(mat, dtype=torch.float32)

        # Validate dimensions
        if result.shape[1] != dim:
            msg = f"Dataset sample has dimension {result.shape[1]} but model expects {dim}. Consider adjusting encoding settings or model input dimension."
            raise InputGenerationError(msg)

        return result

    def _downsample(self, spect: np.ndarray, target_len: int) -> np.ndarray:
        """Downsample spectrogram to target length via linear interpolation."""
        T = spect.shape[0]
        if spect.ndim == 1:
            spect = spect.reshape(-1, 1)
        n_features = spect.shape[1]

        if target_len == T:
            return spect

        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_len)
        down = np.zeros((target_len, n_features), dtype=np.float32)

        for i in range(n_features):
            down[:, i] = np.interp(x_new, x_old, spect[:, i])

        return down

    def _apply_one_hot(self, mat: np.ndarray, vocab_size: int) -> np.ndarray:
        """Apply one-hot encoding to integer data."""
        # Convert to integers
        if mat.ndim == 2 and mat.shape[1] == 1:
            mat = mat.squeeze(1)

        if mat.ndim != 1:
            # Already multi-dimensional, assume pre-encoded
            return mat

        mat_int = mat.astype(np.int64)
        seq_len = len(mat_int)

        # Clamp to valid range
        invalid_mask = (mat_int < 0) | (mat_int >= vocab_size)
        if np.any(invalid_mask):
            mat_int = np.clip(mat_int, 0, vocab_size - 1)

        # Create one-hot encoding
        one_hot = np.zeros((seq_len, vocab_size), dtype=np.float32)
        one_hot[np.arange(seq_len), mat_int] = 1.0

        return one_hot

    def _apply_scaling(self, mat: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Scale features to [min_val, max_val]."""
        if max_val <= min_val:
            return mat

        fmin = mat.min()
        fmax = mat.max()

        if fmax == fmin:
            return mat

        # Normalize to [0, 1] then scale to [min_val, max_val]
        normalized = (mat - fmin) / (fmax - fmin)
        return normalized * (max_val - min_val) + min_val
