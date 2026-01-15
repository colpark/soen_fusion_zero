"""Tests for padding module."""

import torch

from soen_toolkit.model_creation_gui.features.state_trajectory.padding import (
    apply_append,
    apply_prepend,
)
from soen_toolkit.model_creation_gui.features.state_trajectory.settings import TimeMode, ZeroPaddingSpec
from soen_toolkit.model_creation_gui.features.state_trajectory.timebase import Timebase


class TestPadding:
    """Test padding operations."""

    def test_prepend_disabled(self):
        """Test that disabled prepend returns input unchanged."""
        x = torch.randn(10, 5)
        spec = ZeroPaddingSpec(enabled=False)
        tb = Timebase.from_dt(37.0)

        result = apply_prepend(x, spec, TimeMode.DT, tb)

        assert result.shape == x.shape
        assert torch.allclose(result, x)

    def test_prepend_count_steps_mode(self):
        """Test prepend in count_steps mode."""
        x = torch.randn(10, 5)
        spec = ZeroPaddingSpec(enabled=True, count_steps=3, mode="zeros")
        tb = Timebase.from_dt(37.0)

        result = apply_prepend(x, spec, TimeMode.DT, tb)

        assert result.shape == (13, 5)
        # First 3 steps should be zeros
        assert torch.allclose(result[:3, :], torch.zeros(3, 5))
        # Remaining should match original
        assert torch.allclose(result[3:, :], x)

    def test_prepend_time_ns_mode(self):
        """Test prepend in time_ns mode."""
        x = torch.randn(10, 5)
        tb = Timebase.from_dt(37.0)

        # Calculate time for 5 steps
        time_ns = 5 * tb.step_ns
        spec = ZeroPaddingSpec(enabled=True, time_ns=time_ns, mode="zeros")

        result = apply_prepend(x, spec, TimeMode.TOTAL_NS, tb)

        assert result.shape == (15, 5)
        assert torch.allclose(result[:5, :], torch.zeros(5, 5))
        assert torch.allclose(result[5:, :], x)

    def test_prepend_1d_input(self):
        """Test prepend handles 1D input by expanding to 2D."""
        x = torch.randn(10)
        spec = ZeroPaddingSpec(enabled=True, count_steps=2, mode="zeros")
        tb = Timebase.from_dt(37.0)

        result = apply_prepend(x, spec, TimeMode.DT, tb)

        assert result.ndim == 2
        assert result.shape == (12, 1)

    def test_append_disabled(self):
        """Test that disabled append returns input unchanged."""
        x = torch.randn(10, 5)
        spec = ZeroPaddingSpec(enabled=False)
        tb = Timebase.from_dt(37.0)

        result = apply_append(x, spec, TimeMode.DT, tb)

        assert result.shape == x.shape
        assert torch.allclose(result, x)

    def test_append_count_steps_zeros(self):
        """Test append with zeros mode."""
        x = torch.randn(10, 5)
        spec = ZeroPaddingSpec(enabled=True, count_steps=3, mode="zeros")
        tb = Timebase.from_dt(37.0)

        result = apply_append(x, spec, TimeMode.DT, tb)

        assert result.shape == (13, 5)
        # First 10 steps should match original
        assert torch.allclose(result[:10, :], x)
        # Last 3 steps should be zeros
        assert torch.allclose(result[10:, :], torch.zeros(3, 5))

    def test_append_count_steps_hold_last(self):
        """Test append with hold_last mode."""
        x = torch.randn(10, 5)
        spec = ZeroPaddingSpec(enabled=True, count_steps=3, mode="hold_last")
        tb = Timebase.from_dt(37.0)

        result = apply_append(x, spec, TimeMode.DT, tb)

        assert result.shape == (13, 5)
        # First 10 steps should match original
        assert torch.allclose(result[:10, :], x)
        # Last 3 steps should repeat last value
        for i in range(10, 13):
            assert torch.allclose(result[i, :], x[-1, :])

    def test_append_time_ns_mode(self):
        """Test append in time_ns mode."""
        x = torch.randn(10, 5)
        tb = Timebase.from_dt(37.0)

        # Calculate time for 4 steps
        time_ns = 4 * tb.step_ns
        spec = ZeroPaddingSpec(enabled=True, time_ns=time_ns, mode="zeros")

        result = apply_append(x, spec, TimeMode.TOTAL_NS, tb)

        assert result.shape == (14, 5)
        assert torch.allclose(result[:10, :], x)
        assert torch.allclose(result[10:, :], torch.zeros(4, 5))

    def test_append_1d_input(self):
        """Test append handles 1D input by expanding to 2D."""
        x = torch.randn(10)
        spec = ZeroPaddingSpec(enabled=True, count_steps=2, mode="zeros")
        tb = Timebase.from_dt(37.0)

        result = apply_append(x, spec, TimeMode.DT, tb)

        assert result.ndim == 2
        assert result.shape == (12, 1)

    def test_zero_padding_count(self):
        """Test that zero padding count returns unchanged input."""
        x = torch.randn(10, 5)
        spec = ZeroPaddingSpec(enabled=True, count_steps=0, mode="zeros")
        tb = Timebase.from_dt(37.0)

        result = apply_prepend(x, spec, TimeMode.DT, tb)
        assert result.shape == x.shape

        result = apply_append(x, spec, TimeMode.DT, tb)
        assert result.shape == x.shape

    def test_both_prepend_and_append(self):
        """Test applying both prepend and append."""
        x = torch.randn(10, 5)
        prepend_spec = ZeroPaddingSpec(enabled=True, count_steps=2, mode="zeros")
        append_spec = ZeroPaddingSpec(enabled=True, count_steps=3, mode="hold_last")
        tb = Timebase.from_dt(37.0)

        result = apply_prepend(x, prepend_spec, TimeMode.DT, tb)
        result = apply_append(result, append_spec, TimeMode.DT, tb)

        assert result.shape == (15, 5)
        # First 2: zeros
        assert torch.allclose(result[:2, :], torch.zeros(2, 5))
        # Middle 10: original
        assert torch.allclose(result[2:12, :], x)
        # Last 3: held
        for i in range(12, 15):
            assert torch.allclose(result[i, :], x[-1, :])

    def test_preserves_dtype(self):
        """Test that padding preserves dtype."""
        x = torch.randn(10, 5, dtype=torch.float64)
        spec = ZeroPaddingSpec(enabled=True, count_steps=2, mode="zeros")
        tb = Timebase.from_dt(37.0)

        result = apply_prepend(x, spec, TimeMode.DT, tb)
        assert result.dtype == torch.float64

    def test_preserves_device(self):
        """Test that padding preserves device."""
        x = torch.randn(10, 5)
        device = x.device
        spec = ZeroPaddingSpec(enabled=True, count_steps=2, mode="zeros")
        tb = Timebase.from_dt(37.0)

        result = apply_prepend(x, spec, TimeMode.DT, tb)
        assert result.device == device
