"""Tests for input generation strategies."""

import numpy as np
import torch

from soen_toolkit.model_creation_gui.features.state_trajectory.inputs import (
    ColoredNoise,
    ConstantInput,
    GaussianNoise,
    Sinusoid,
    SquareWave,
)
from soen_toolkit.model_creation_gui.features.state_trajectory.timebase import Timebase


class TestConstantInput:
    """Test ConstantInput generator."""

    def test_shape(self):
        """Test output shape."""
        gen = ConstantInput(value=1.5)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.shape == (100, 10)

    def test_value(self):
        """Test all values are constant."""
        gen = ConstantInput(value=2.5)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert torch.allclose(result, torch.full((100, 10), 2.5))

    def test_dtype(self):
        """Test output dtype is float32."""
        gen = ConstantInput(value=1.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.dtype == torch.float32


class TestGaussianNoise:
    """Test GaussianNoise generator."""

    def test_shape(self):
        """Test output shape."""
        gen = GaussianNoise(std=0.1)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.shape == (100, 10)

    def test_mean_near_zero(self):
        """Test mean is close to zero."""
        gen = GaussianNoise(std=1.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(10000, 10, tb, rng)

        mean = result.mean().item()
        assert abs(mean) < 0.05  # Should be close to 0 for large sample

    def test_std_matches(self):
        """Test standard deviation matches specification."""
        target_std = 0.5
        gen = GaussianNoise(std=target_std)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(10000, 10, tb, rng)

        actual_std = result.std().item()
        assert abs(actual_std - target_std) < 0.05

    def test_determinism(self):
        """Test same seed produces same output."""
        gen = GaussianNoise(std=0.1)
        tb = Timebase.from_dt(37.0)

        rng1 = np.random.default_rng(42)
        result1 = gen.make(100, 10, tb, rng1)

        rng2 = np.random.default_rng(42)
        result2 = gen.make(100, 10, tb, rng2)

        assert torch.allclose(result1, result2)

    def test_dtype(self):
        """Test output dtype is float32."""
        gen = GaussianNoise(std=0.1)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.dtype == torch.float32


class TestColoredNoise:
    """Test ColoredNoise generator."""

    def test_shape(self):
        """Test output shape."""
        gen = ColoredNoise(beta=2.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.shape == (100, 10)

    def test_beta_zero_is_white(self):
        """Test beta=0 produces white-like noise."""
        gen = ColoredNoise(beta=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(10000, 5, tb, rng)

        # White noise should have flat spectrum (autocorrelation near zero at lag > 0)
        # Just check mean is near zero and variance is reasonable
        mean = result.mean().item()
        assert abs(mean) < 0.1

    def test_determinism(self):
        """Test same seed produces same output."""
        gen = ColoredNoise(beta=1.5)
        tb = Timebase.from_dt(37.0)

        rng1 = np.random.default_rng(42)
        result1 = gen.make(100, 10, tb, rng1)

        rng2 = np.random.default_rng(42)
        result2 = gen.make(100, 10, tb, rng2)

        assert torch.allclose(result1, result2, rtol=1e-5, atol=1e-7)

    def test_short_sequence_fallback(self):
        """Test very short sequences fall back to white noise."""
        gen = ColoredNoise(beta=2.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(1, 5, tb, rng)

        assert result.shape == (1, 5)
        assert torch.isfinite(result).all()

    def test_no_nan_or_inf(self):
        """Test output has no NaN or Inf values."""
        gen = ColoredNoise(beta=2.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert torch.isfinite(result).all()

    def test_dtype(self):
        """Test output dtype is float32."""
        gen = ColoredNoise(beta=1.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.dtype == torch.float32


class TestSinusoid:
    """Test Sinusoid generator."""

    def test_shape(self):
        """Test output shape."""
        gen = Sinusoid(freq_mhz=10.0, amp=1.0, phase_deg=0.0, offset=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.shape == (100, 10)

    def test_all_dims_same(self):
        """Test all dimensions have same waveform."""
        gen = Sinusoid(freq_mhz=10.0, amp=1.0, phase_deg=0.0, offset=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        # All columns should be identical
        for i in range(1, 10):
            assert torch.allclose(result[:, i], result[:, 0])

    def test_amplitude(self):
        """Test amplitude is correct."""
        amp = 2.5
        gen = Sinusoid(freq_mhz=10.0, amp=amp, phase_deg=0.0, offset=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(1000, 1, tb, rng)

        # Max and min should be approximately +/- amp
        max_val = result.max().item()
        min_val = result.min().item()
        assert abs(max_val - amp) < 0.1
        assert abs(min_val + amp) < 0.1

    def test_offset(self):
        """Test offset shifts waveform."""
        offset = 5.0
        gen = Sinusoid(freq_mhz=10.0, amp=1.0, phase_deg=0.0, offset=offset)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(1000, 1, tb, rng)

        # Mean should be approximately the offset
        mean = result.mean().item()
        assert abs(mean - offset) < 0.1

    def test_phase_shift(self):
        """Test phase shift affects waveform."""
        gen1 = Sinusoid(freq_mhz=10.0, amp=1.0, phase_deg=0.0, offset=0.0)
        gen2 = Sinusoid(freq_mhz=10.0, amp=1.0, phase_deg=90.0, offset=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result1 = gen1.make(100, 1, tb, rng)
        result2 = gen2.make(100, 1, tb, rng)

        # Different phases should give different values
        assert not torch.allclose(result1, result2)

    def test_dtype(self):
        """Test output dtype is float32."""
        gen = Sinusoid(freq_mhz=10.0, amp=1.0, phase_deg=0.0, offset=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.dtype == torch.float32


class TestSquareWave:
    """Test SquareWave generator."""

    def test_shape(self):
        """Test output shape."""
        gen = SquareWave(freq_mhz=10.0, amp=1.0, phase_deg=0.0, offset=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.shape == (100, 10)

    def test_all_dims_same(self):
        """Test all dimensions have same waveform."""
        gen = SquareWave(freq_mhz=10.0, amp=1.0, phase_deg=0.0, offset=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        # All columns should be identical
        for i in range(1, 10):
            assert torch.allclose(result[:, i], result[:, 0])

    def test_only_two_levels(self):
        """Test square wave has only two distinct levels (approximately)."""
        amp = 1.0
        gen = SquareWave(freq_mhz=5.0, amp=amp, phase_deg=0.0, offset=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(1000, 1, tb, rng)

        # Unique values should be close to +amp and -amp
        unique_vals = torch.unique(result)
        assert len(unique_vals) <= 3  # May have 0 at transitions

        max_val = result.max().item()
        min_val = result.min().item()
        assert abs(max_val - amp) < 0.01
        assert abs(min_val + amp) < 0.01

    def test_offset(self):
        """Test offset shifts levels."""
        offset = 3.0
        amp = 1.0
        gen = SquareWave(freq_mhz=5.0, amp=amp, phase_deg=0.0, offset=offset)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(1000, 1, tb, rng)

        # Mean should be approximately the offset
        mean = result.mean().item()
        assert abs(mean - offset) < 0.2

    def test_dtype(self):
        """Test output dtype is float32."""
        gen = SquareWave(freq_mhz=10.0, amp=1.0, phase_deg=0.0, offset=0.0)
        tb = Timebase.from_dt(37.0)
        rng = np.random.default_rng(42)

        result = gen.make(100, 10, tb, rng)

        assert result.dtype == torch.float32
