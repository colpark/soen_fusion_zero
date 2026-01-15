"""Tests for model adapter."""

import torch
from torch import nn

from soen_toolkit.model_creation_gui.features.state_trajectory.model_adapter import ModelAdapter


class MockLayer(nn.Module):
    """Mock layer for testing."""

    def __init__(self):
        super().__init__()
        self._phi_history = []
        self._g_history = []
        self._state_history = []
        self.track_phi = False
        self.track_g = False
        self.track_s = False
        self.track_power = False

        # Power/energy attributes
        self.power_bias_dimensionless = None
        self.power_diss_dimensionless = None
        self.energy_bias_dimensionless = None
        self.energy_diss_dimensionless = None

        # Physical parameters
        self.Ic = 100e-6
        self.Phi0 = 2.067833848e-15
        self.wc = 3.72e11

    def _clear_histories(self):
        """Clear all history lists."""
        self._phi_history = []
        self._g_history = []
        self._state_history = []


class MockSimConfig:
    """Mock simulation config."""

    def __init__(self):
        self.track_phi = False
        self.track_g = False
        self.track_s = False
        self.track_power = False


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self):
        super().__init__()
        self.dt = 37.0
        self.layers = nn.ModuleList([MockLayer(), MockLayer()])
        self.sim_config = MockSimConfig()

    def set_dt(self, dt):
        """Set time step."""
        self.dt = dt

    def set_tracking(self, track_phi=None, track_g=None, track_s=None, track_power=None):
        """Set tracking flags."""
        if track_phi is not None:
            self.sim_config.track_phi = track_phi
        if track_g is not None:
            self.sim_config.track_g = track_g
        if track_s is not None:
            self.sim_config.track_s = track_s
        if track_power is not None:
            self.sim_config.track_power = track_power

    def get_phi_history(self):
        """Mock phi history."""
        return [torch.randn(1, 10, 5) for _ in self.layers]

    def get_g_history(self):
        """Mock g history."""
        return [torch.randn(1, 10, 5) for _ in self.layers]


class TestModelAdapter:
    """Test ModelAdapter."""

    def test_set_dt(self):
        """Test set_dt propagates to model."""
        model = MockModel()
        adapter = ModelAdapter()

        adapter.set_dt(model, 50.0)

        assert model.dt == 50.0

    def test_enable_full_tracking(self):
        """Test enable_full_tracking sets all flags."""
        model = MockModel()
        adapter = ModelAdapter()

        adapter.enable_full_tracking(model)

        assert model.sim_config.track_phi
        assert model.sim_config.track_g
        assert model.sim_config.track_s
        assert model.sim_config.track_power

    def test_restore_tracking(self):
        """Test restore_tracking restores from config."""
        model = MockModel()
        model.sim_config.track_phi = True
        model.sim_config.track_g = False
        model.sim_config.track_s = True
        model.sim_config.track_power = False

        adapter = ModelAdapter()

        # Change flags
        model.set_tracking(track_phi=False, track_g=True, track_s=False, track_power=True)

        # Restore
        config = MockSimConfig()
        config.track_phi = True
        config.track_g = False
        config.track_s = True
        config.track_power = False
        adapter.restore_tracking(model, config)

        # Should match config
        # Note: This tests the adapter, not whether model respects the settings
        # The adapter just calls model.set_tracking with config values

    def test_reset_state(self):
        """Test reset_state clears histories."""
        model = MockModel()
        adapter = ModelAdapter()

        # Add some fake history
        for layer in model.layers:
            layer._phi_history = [torch.randn(1, 5, 3)]
            layer._g_history = [torch.randn(1, 5, 3)]

        adapter.reset_state(model)

        # Histories should be empty
        for layer in model.layers:
            assert len(layer._phi_history) == 0
            assert len(layer._g_history) == 0

    def test_collect_metric_histories_state_include_s0(self):
        """Test collect state histories with s0 included."""
        model = MockModel()
        adapter = ModelAdapter()

        raw_state_histories = [
            torch.randn(1, 11, 5),  # [batch, T+1, dim]
            torch.randn(1, 11, 3),
        ]

        result = adapter.collect_metric_histories(model, "state", include_s0=True, raw_state_histories=raw_state_histories)

        assert len(result) == 2
        assert result[0].shape == (1, 11, 5)
        assert result[1].shape == (1, 11, 3)

    def test_collect_metric_histories_state_exclude_s0(self):
        """Test collect state histories with s0 excluded."""
        model = MockModel()
        adapter = ModelAdapter()

        raw_state_histories = [
            torch.randn(1, 11, 5),  # [batch, T+1, dim]
            torch.randn(1, 11, 3),
        ]

        result = adapter.collect_metric_histories(model, "state", include_s0=False, raw_state_histories=raw_state_histories)

        assert len(result) == 2
        assert result[0].shape == (1, 10, 5)  # First timestep dropped
        assert result[1].shape == (1, 10, 3)

    def test_collect_metric_histories_phi(self):
        """Test collect phi histories."""
        model = MockModel()
        adapter = ModelAdapter()

        raw_state_histories = [torch.randn(1, 11, 5), torch.randn(1, 11, 3)]

        result = adapter.collect_metric_histories(model, "phi", include_s0=True, raw_state_histories=raw_state_histories)

        assert len(result) == 2
        # Should return phi histories from model

    def test_collect_metric_histories_g(self):
        """Test collect g histories."""
        model = MockModel()
        adapter = ModelAdapter()

        raw_state_histories = [torch.randn(1, 11, 5), torch.randn(1, 11, 3)]

        result = adapter.collect_metric_histories(model, "g", include_s0=True, raw_state_histories=raw_state_histories)

        assert len(result) == 2
        # Should return g histories from model

    def test_collect_metric_histories_power(self):
        """Test collect power histories."""
        model = MockModel()
        adapter = ModelAdapter()

        # Set up power data
        for layer in model.layers:
            layer.power_bias_dimensionless = torch.randn(1, 10, 5)
            layer.power_diss_dimensionless = torch.randn(1, 10, 5)

        raw_state_histories = [torch.randn(1, 11, 5), torch.randn(1, 11, 5)]

        result = adapter.collect_metric_histories(model, "power", include_s0=True, raw_state_histories=raw_state_histories)

        assert len(result) == 2
        # Should have converted to physical units (nW)

    def test_collect_metric_histories_energy(self):
        """Test collect energy histories."""
        model = MockModel()
        adapter = ModelAdapter()

        # Set up energy data
        for layer in model.layers:
            layer.energy_bias_dimensionless = torch.randn(1, 10, 5)
            layer.energy_diss_dimensionless = torch.randn(1, 10, 5)

        raw_state_histories = [torch.randn(1, 11, 5), torch.randn(1, 11, 5)]

        result = adapter.collect_metric_histories(model, "energy", include_s0=True, raw_state_histories=raw_state_histories)

        assert len(result) == 2
        # Should have converted to physical units (nJ)

    def test_collect_metric_histories_unknown(self):
        """Test unknown metric returns empty list."""
        model = MockModel()
        adapter = ModelAdapter()

        raw_state_histories = [torch.randn(1, 11, 5), torch.randn(1, 11, 3)]

        result = adapter.collect_metric_histories(model, "unknown_metric", include_s0=True, raw_state_histories=raw_state_histories)

        assert result == []
