"""Tests for custom weights loading from .npy and .npz files."""

import numpy as np
import pytest
import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.core.layers.common.connectivity_metadata import WEIGHT_INITIALIZERS
from soen_toolkit.core.layers.common.weight_initializers import init_custom_weights
from soen_toolkit.utils.weights_utils import (
    load_weights_from_file,
    save_weights_to_npy,
    save_weights_to_npz,
    validate_weight_shape,
)


class TestCustomWeightsLoader:
    """Test the init_custom_weights function directly."""

    def test_load_from_npy_file(self, tmp_path) -> None:
        """Test loading weights from .npy file."""
        # Create sample weights
        weights = np.random.randn(5, 10).astype(np.float32)
        weights_file = tmp_path / "weights.npy"
        np.save(str(weights_file), weights)

        # Create mask
        mask = torch.ones(5, 10)

        # Load via init_custom_weights
        result = init_custom_weights(10, 5, mask, weights_file=str(weights_file))

        # Verify
        assert result.shape == (5, 10)
        assert torch.allclose(result, torch.from_numpy(weights))

    def test_load_from_npz_file(self, tmp_path) -> None:
        """Test loading weights from .npz file with 'weights' key."""
        # Create sample weights
        weights = np.random.randn(3, 8).astype(np.float32)
        npz_file = tmp_path / "weights.npz"
        np.savez(str(npz_file), weights=weights)

        # Create mask
        mask = torch.ones(3, 8)

        # Load via init_custom_weights
        result = init_custom_weights(8, 3, mask, weights_file=str(npz_file))

        # Verify
        assert result.shape == (3, 8)
        assert torch.allclose(result, torch.from_numpy(weights))

    def test_mask_application(self, tmp_path) -> None:
        """Test that mask is properly applied to weights."""
        # Create weights (all ones for easy verification)
        weights = np.ones((4, 6), dtype=np.float32)
        weights_file = tmp_path / "weights.npy"
        np.save(str(weights_file), weights)

        # Create sparse mask (only 50% of connections)
        mask = torch.bernoulli(torch.full((4, 6), 0.5))

        # Load and apply mask
        result = init_custom_weights(6, 4, mask, weights_file=str(weights_file))

        # Verify mask was applied
        assert torch.allclose(result, mask)

    def test_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        mask = torch.ones(5, 10)
        with pytest.raises(ValueError, match="Weights file not found"):
            init_custom_weights(10, 5, mask, weights_file="/nonexistent/weights.npy")

    def test_missing_weights_file_param(self) -> None:
        """Test error when weights_file parameter is missing."""
        mask = torch.ones(5, 10)
        with pytest.raises(ValueError, match="requires 'weights_file'"):
            init_custom_weights(10, 5, mask)

    def test_invalid_npz_key(self, tmp_path) -> None:
        """Test error when 'weights' key doesn't exist in .npz."""
        weights = np.ones((5, 10), dtype=np.float32)
        npz_file = tmp_path / "wrong_key.npz"
        np.savez(str(npz_file), wrong_key=weights)

        mask = torch.ones(5, 10)
        with pytest.raises(ValueError, match="Key 'weights' not found"):
            init_custom_weights(10, 5, mask, weights_file=str(npz_file))

    def test_shape_mismatch(self, tmp_path) -> None:
        """Test error when weights shape doesn't match."""
        # Create 3x4 weights but expect 5x10
        weights = np.ones((3, 4), dtype=np.float32)
        weights_file = tmp_path / "wrong_shape.npy"
        np.save(str(weights_file), weights)

        mask = torch.ones(5, 10)
        with pytest.raises(ValueError, match="Weights shape mismatch"):
            init_custom_weights(10, 5, mask, weights_file=str(weights_file))

    def test_non_2d_array(self, tmp_path) -> None:
        """Test error when weights are not 2D."""
        # Create 1D array
        weights = np.ones(50, dtype=np.float32)
        weights_file = tmp_path / "1d_weights.npy"
        np.save(str(weights_file), weights)

        mask = torch.ones(5, 10)
        with pytest.raises(ValueError, match="Weights array must be 2D"):
            init_custom_weights(10, 5, mask, weights_file=str(weights_file))

    def test_unsupported_file_format(self, tmp_path) -> None:
        """Test error with unsupported file format."""
        weights_file = tmp_path / "weights.txt"
        weights_file.write_text("invalid")

        mask = torch.ones(5, 10)
        with pytest.raises(ValueError, match="Unsupported file format"):
            init_custom_weights(10, 5, mask, weights_file=str(weights_file))


class TestCustomWeightsIntegration:
    """Test custom weights in full model context."""

    def test_model_with_custom_weights(self, tmp_path) -> None:
        """Test building a model with custom weights."""
        # Create custom weights
        weights = np.random.randn(5, 10).astype(np.float32)
        weights_file = tmp_path / "custom.npy"
        np.save(str(weights_file), weights)

        # Create model config
        sim_config = SimulationConfig(dt=0.01)
        layers_config = [
            LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 10}),
            LayerConfig(layer_id=1, layer_type="SingleDendrite", params={"dim": 5}),
        ]
        connections_config = [
            ConnectionConfig(
                from_layer=0,
                to_layer=1,
                connection_type="dense",
                params={
                    "init": "custom",
                    "weights_file": str(weights_file),
                },
            ),
        ]

        # Build model
        model = SOENModelCore(
            sim_config=sim_config,
            layers_config=layers_config,
            connections_config=connections_config,
        )

        # Verify weights were loaded
        assert "J_0_to_1" in model.connections
        loaded_weights = model.connections["J_0_to_1"].data
        assert loaded_weights.shape == (5, 10)
        assert torch.allclose(loaded_weights, torch.from_numpy(weights), atol=1e-6)

    def test_model_with_custom_weights_npz(self, tmp_path) -> None:
        """Test model with weights in .npz format."""
        weights = np.random.randn(3, 8).astype(np.float32)
        weights_file = tmp_path / "custom.npz"
        np.savez(str(weights_file), weights=weights)

        sim_config = SimulationConfig(dt=0.01)
        layers_config = [
            LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 8}),
            LayerConfig(layer_id=1, layer_type="SingleDendrite", params={"dim": 3}),
        ]
        connections_config = [
            ConnectionConfig(
                from_layer=0,
                to_layer=1,
                connection_type="dense",
                params={
                    "init": "custom",
                    "weights_file": str(weights_file),
                },
            ),
        ]

        model = SOENModelCore(
            sim_config=sim_config,
            layers_config=layers_config,
            connections_config=connections_config,
        )

        # Verify
        assert "J_0_to_1" in model.connections
        assert torch.allclose(
            model.connections["J_0_to_1"].data,
            torch.from_numpy(weights),
            atol=1e-6,
        )


class TestWeightsUtilities:
    """Test utility functions for weights."""

    def test_save_weights_to_npy(self, tmp_path) -> None:
        """Test saving weights to .npy file."""
        weights = torch.randn(5, 10)
        filepath = tmp_path / "weights.npy"

        save_weights_to_npy(weights, filepath)

        # Verify
        assert filepath.exists()
        loaded = np.load(str(filepath))
        assert torch.allclose(torch.from_numpy(loaded), weights)

    def test_save_weights_to_npz(self, tmp_path) -> None:
        """Test saving weights to .npz file."""
        weights = torch.randn(5, 10)
        filepath = tmp_path / "weights.npz"

        save_weights_to_npz(weights, filepath)

        # Verify
        assert filepath.exists()
        data = np.load(str(filepath))
        assert "weights" in data
        assert torch.allclose(torch.from_numpy(data["weights"]), weights)

    def test_validate_weight_shape_correct(self) -> None:
        """Test validation with correct shape."""
        weights = np.ones((5, 10))
        assert validate_weight_shape(weights, from_nodes=10, to_nodes=5)

    def test_validate_weight_shape_wrong(self) -> None:
        """Test validation with wrong shape."""
        weights = np.ones((5, 10))
        assert not validate_weight_shape(weights, from_nodes=8, to_nodes=5)

    def test_load_weights_from_file_npy(self, tmp_path) -> None:
        """Test loading weights from .npy file."""
        weights = np.random.randn(5, 10).astype(np.float32)
        filepath = tmp_path / "weights.npy"
        np.save(str(filepath), weights)

        loaded = load_weights_from_file(filepath)
        assert np.allclose(loaded, weights)

    def test_load_weights_from_file_npz(self, tmp_path) -> None:
        """Test loading weights from .npz file."""
        weights = np.random.randn(5, 10).astype(np.float32)
        filepath = tmp_path / "weights.npz"
        np.savez(str(filepath), weights=weights)

        loaded = load_weights_from_file(filepath)
        assert np.allclose(loaded, weights)

    def test_load_weights_with_validation(self, tmp_path) -> None:
        """Test loading weights with shape validation."""
        weights = np.random.randn(5, 10).astype(np.float32)
        filepath = tmp_path / "weights.npy"
        np.save(str(filepath), weights)

        # Correct shape
        loaded = load_weights_from_file(filepath, from_nodes=10, to_nodes=5)
        assert np.allclose(loaded, weights)

        # Wrong shape should raise
        with pytest.raises(ValueError, match="doesn't match expected shape"):
            load_weights_from_file(filepath, from_nodes=8, to_nodes=5)

    def test_load_weights_file_not_found(self, tmp_path) -> None:
        """Test error when file not found."""
        filepath = tmp_path / "nonexistent.npy"
        with pytest.raises(FileNotFoundError):
            load_weights_from_file(filepath)


class TestWeightsInRegistry:
    """Test that custom weights initializer is properly registered."""

    def test_custom_in_registry(self) -> None:
        """Test that 'custom' is in WEIGHT_INITIALIZERS."""
        assert "custom" in WEIGHT_INITIALIZERS
        assert WEIGHT_INITIALIZERS["custom"] == init_custom_weights

    def test_all_initializers_callable(self) -> None:
        """Test all initializers are callable."""
        for name, initializer in WEIGHT_INITIALIZERS.items():
            assert callable(initializer), f"{name} is not callable"
