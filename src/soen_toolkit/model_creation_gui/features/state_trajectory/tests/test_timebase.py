"""Tests for timebase module."""

import pytest

from soen_toolkit.model_creation_gui.features.state_trajectory.timebase import Timebase
from soen_toolkit.physics.constants import get_omega_c


class TestTimebase:
    """Test Timebase class."""

    def test_from_dt_basic(self):
        """Test construction from dt."""
        tb = Timebase.from_dt(37.0)
        assert tb.dt == 37.0
        assert tb.omega_c == get_omega_c()

    def test_from_dt_with_custom_omega_c(self):
        """Test construction from dt with custom omega_c."""
        custom_omega = 1e9
        tb = Timebase.from_dt(50.0, omega_c=custom_omega)
        assert tb.dt == 50.0
        assert tb.omega_c == custom_omega

    def test_step_seconds(self):
        """Test step_seconds property."""
        dt = 37.0
        omega_c = get_omega_c()
        tb = Timebase.from_dt(dt, omega_c)
        expected = dt / omega_c
        assert abs(tb.step_seconds - expected) < 1e-15

    def test_step_ns(self):
        """Test step_ns property."""
        dt = 37.0
        omega_c = get_omega_c()
        tb = Timebase.from_dt(dt, omega_c)
        expected = (dt / omega_c) * 1e9
        assert abs(tb.step_ns - expected) < 1e-6

    def test_sampling_rate_hz(self):
        """Test sampling_rate_hz property."""
        dt = 37.0
        omega_c = get_omega_c()
        tb = Timebase.from_dt(dt, omega_c)
        expected_step_s = dt / omega_c
        expected_fs = 1.0 / expected_step_s
        assert abs(tb.sampling_rate_hz - expected_fs) < 1e-6

    def test_nyquist_hz(self):
        """Test nyquist_hz property."""
        tb = Timebase.from_dt(37.0)
        expected = tb.sampling_rate_hz / 2.0
        assert abs(tb.nyquist_hz - expected) < 1e-6

    def test_total_ns(self):
        """Test total_ns method."""
        tb = Timebase.from_dt(37.0)
        steps = 100
        expected = tb.step_ns * steps
        assert abs(tb.total_ns(steps) - expected) < 1e-6

    def test_total_seconds(self):
        """Test total_seconds method."""
        tb = Timebase.from_dt(37.0)
        steps = 100
        expected = tb.step_seconds * steps
        assert abs(tb.total_seconds(steps) - expected) < 1e-15

    def test_from_total_ns_basic(self):
        """Test construction from total time."""
        total_ns = 1000.0
        steps = 100
        omega_c = get_omega_c()

        tb = Timebase.from_total_ns(total_ns, steps, omega_c)

        # Verify roundtrip: total_ns(steps) should equal input
        recovered = tb.total_ns(steps)
        assert abs(recovered - total_ns) < 1e-6

    def test_from_total_ns_roundtrip(self):
        """Test roundtrip: dt → total_ns → dt."""
        original_dt = 37.0
        steps = 100
        omega_c = get_omega_c()

        # Create from dt
        tb1 = Timebase.from_dt(original_dt, omega_c)
        total = tb1.total_ns(steps)

        # Recreate from total
        tb2 = Timebase.from_total_ns(total, steps, omega_c)

        # Should recover original dt
        assert abs(tb2.dt - original_dt) < 1e-10

    def test_from_total_ns_zero_steps_raises(self):
        """Test that zero steps raises ValueError."""
        with pytest.raises(ValueError, match="steps must be positive"):
            Timebase.from_total_ns(1000.0, 0)

    def test_from_total_ns_negative_steps_raises(self):
        """Test that negative steps raises ValueError."""
        with pytest.raises(ValueError, match="steps must be positive"):
            Timebase.from_total_ns(1000.0, -10)

    def test_from_total_ns_negative_time_raises(self):
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError, match="total_ns must be non-negative"):
            Timebase.from_total_ns(-1000.0, 100)

    def test_immutability(self):
        """Test that Timebase is immutable (frozen dataclass)."""
        tb = Timebase.from_dt(37.0)
        with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError or AttributeError
            tb.dt = 50.0

    def test_edge_case_very_small_dt(self):
        """Test edge case with very small dt."""
        tb = Timebase.from_dt(0.1)
        assert tb.step_ns > 0
        assert tb.sampling_rate_hz > 0

    def test_edge_case_very_large_dt(self):
        """Test edge case with very large dt."""
        tb = Timebase.from_dt(10000.0)
        assert tb.step_ns > 0
        assert tb.sampling_rate_hz > 0
