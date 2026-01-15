"""Tests for custom connectivity mask loading from .npz files."""

import numpy as np
import pytest
import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.core.layers.common.connectivity_builders import build_custom


class TestCustomConnectivityBuilder:
    """Test the build_custom function directly."""

    def test_basic_binary_mask(self, tmp_path) -> None:
        """Test loading a basic binary mask."""
        # Create a simple binary mask
        mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        npz_file = tmp_path / "test_mask.npz"
        np.savez(npz_file, mask=mask)

        # Build connectivity
        result = build_custom(3, 2, {"mask_file": str(npz_file)})

        # Verify
        assert result.shape == (2, 3)
        assert torch.allclose(result, torch.from_numpy(mask))

    def test_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(ValueError, match="Mask file not found"):
            build_custom(3, 2, {"mask_file": "/nonexistent/file.npz"})

    def test_missing_mask_file_param(self) -> None:
        """Test error when mask_file parameter is missing."""
        with pytest.raises(ValueError, match="requires 'mask_file'"):
            build_custom(3, 2, {})

        with pytest.raises(ValueError, match="requires 'mask_file'"):
            build_custom(3, 2, None)

    def test_invalid_npz_key(self, tmp_path) -> None:
        """Test error when 'mask' key doesn't exist."""
        mask = np.ones((2, 3), dtype=np.float32)
        npz_file = tmp_path / "test.npz"
        np.savez(npz_file, wrong_key=mask)  # Save with wrong key

        with pytest.raises(ValueError, match="Key 'mask' not found"):
            build_custom(3, 2, {"mask_file": str(npz_file)})

    def test_shape_mismatch(self, tmp_path) -> None:
        """Test error when mask shape doesn't match dimensions."""
        # Create 3x4 mask but request 5x10 connection
        mask = np.ones((3, 4), dtype=np.float32)
        npz_file = tmp_path / "wrong_shape.npz"
        np.savez(npz_file, mask=mask)

        with pytest.raises(ValueError, match="Mask shape mismatch"):
            build_custom(10, 5, {"mask_file": str(npz_file)})

    def test_non_2d_array(self, tmp_path) -> None:
        """Test error when mask is not 2D."""
        # Create 1D array
        mask = np.ones(10, dtype=np.float32)
        npz_file = tmp_path / "1d_mask.npz"
        np.savez(npz_file, mask=mask)

        with pytest.raises(ValueError, match="Mask array must be 2D"):
            build_custom(10, 1, {"mask_file": str(npz_file)})

    def test_binary_validation(self, tmp_path) -> None:
        """Test that mask must contain only 0/1 values."""
        # Create mask with non-binary values
        mask = np.array([[0.5, 1.0], [0.0, 0.7]], dtype=np.float32)
        npz_file = tmp_path / "non_binary.npz"
        np.savez(npz_file, mask=mask)

        with pytest.raises(ValueError, match="Mask must contain only 0 or 1 values"):
            build_custom(2, 2, {"mask_file": str(npz_file)})


class TestCustomConnectivityIntegration:
    """Test custom connectivity in full model context."""

    def test_model_with_custom_connectivity(self, tmp_path) -> None:
        """Test building a complete model with custom connectivity."""
        # Create custom mask
        mask = np.array(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        npz_file = tmp_path / "custom_conn.npz"
        np.savez(npz_file, mask=mask)

        # Build model
        sim = SimulationConfig(dt=37, network_evaluation_method="layerwise")
        layers = [
            LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 5}),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 3}),
        ]
        connections = [
            ConnectionConfig(
                from_layer=0,
                to_layer=1,
                connection_type="custom",
                params={"mask_file": str(npz_file)},
            ),
        ]

        model = SOENModelCore(sim, layers, connections)

        # Verify connection exists
        assert "J_0_to_1" in model.connections

        # Verify mask is applied
        model.connections["J_0_to_1"]
        mask_applied = model.connection_masks["J_0_to_1"]
        assert mask_applied.shape == (3, 5)

        # Verify masked positions are zero
        expected_mask = torch.from_numpy(mask)
        assert torch.allclose(mask_applied, expected_mask)

    def test_multiple_custom_connections(self, tmp_path) -> None:
        """Test model with multiple custom connectivity patterns."""
        # Create two different masks in separate files
        mask1 = np.eye(5, dtype=np.float32)  # Diagonal
        mask2 = np.ones((5, 5), dtype=np.float32)  # Dense

        npz_file1 = tmp_path / "diagonal_mask.npz"
        npz_file2 = tmp_path / "dense_mask.npz"
        np.savez(npz_file1, mask=mask1)
        np.savez(npz_file2, mask=mask2)

        sim = SimulationConfig(dt=37, network_evaluation_method="stepwise_gauss_seidel")
        layers = [
            LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 5}),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 5}),
            LayerConfig(layer_id=2, layer_type="Linear", params={"dim": 5}),
        ]
        connections = [
            ConnectionConfig(
                from_layer=0,
                to_layer=1,
                connection_type="custom",
                params={"mask_file": str(npz_file1)},
            ),
            ConnectionConfig(
                from_layer=1,
                to_layer=2,
                connection_type="custom",
                params={"mask_file": str(npz_file2)},
            ),
        ]

        model = SOENModelCore(sim, layers, connections)

        # Verify both connections exist
        assert "J_0_to_1" in model.connections
        assert "J_1_to_2" in model.connections

        # Verify masks are different
        mask_01 = model.connection_masks["J_0_to_1"]
        mask_12 = model.connection_masks["J_1_to_2"]
        assert not torch.allclose(mask_01, mask_12)

    def test_model_save_and_load(self, tmp_path) -> None:
        """Test that custom masks are preserved through save/load cycle."""
        # Create mask
        mask = np.array([[1, 0], [1, 1]], dtype=np.float32)
        npz_file = tmp_path / "mask.npz"
        np.savez(npz_file, mask=mask)

        # Build model
        sim = SimulationConfig(dt=37)
        layers = [
            LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 2}),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 2}),
        ]
        connections = [
            ConnectionConfig(
                from_layer=0,
                to_layer=1,
                connection_type="custom",
                params={"mask_file": str(npz_file)},
            ),
        ]

        model = SOENModelCore(sim, layers, connections)

        # Save model
        save_path = tmp_path / "model.soen"
        model.save(str(save_path))

        # Load model (mask file still exists)
        loaded_model = SOENModelCore.load(str(save_path))

        # Verify mask is preserved
        original_mask = model.connection_masks["J_0_to_1"]
        loaded_mask = loaded_model.connection_masks["J_0_to_1"]
        assert torch.allclose(original_mask, loaded_mask)

        # Verify connection weights match
        original_weights = model.connections["J_0_to_1"]
        loaded_weights = loaded_model.connections["J_0_to_1"]
        assert torch.allclose(original_weights, loaded_weights)

    def test_forward_pass_with_custom_mask(self, tmp_path) -> None:
        """Test that forward pass works correctly with custom connectivity."""
        # Create sparse mask
        mask = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        npz_file = tmp_path / "sparse.npz"
        np.savez(npz_file, mask=mask)

        sim = SimulationConfig(dt=37)
        layers = [
            LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 5}),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 3}),
        ]
        connections = [
            ConnectionConfig(
                from_layer=0,
                to_layer=1,
                connection_type="custom",
                params={"mask_file": str(npz_file)},
            ),
        ]

        model = SOENModelCore(sim, layers, connections)

        # Run forward pass
        x = torch.randn(2, 10, 5)  # (batch, time, features)
        output, _ = model(x)

        # Verify output shape
        assert output.shape == (2, 11, 3)  # +1 for initial state


class TestPyTorchAPICustomConnectivity:
    """Test custom connectivity through PyTorch-style API."""

    def test_pytorch_api_custom_structure(self, tmp_path) -> None:
        """Test using structure.custom() in PyTorch API."""
        # Create mask
        mask = np.ones((3, 5), dtype=np.float32)
        npz_file = tmp_path / "mask.npz"
        np.savez(npz_file, mask=mask)

        from soen_toolkit.nn import Graph, init, layers, structure

        g = Graph(dt=37, network_evaluation_method="layerwise")
        g.add_layer(0, layers.Linear(dim=5))
        g.add_layer(1, layers.Linear(dim=3))
        g.connect(0, 1, structure=structure.custom(str(npz_file)), init=init.xavier_uniform())

        # Compile and check
        g.compile()
        assert "J_0_to_1" in g._compiled_core.connections


class TestYAMLConfigCustomConnectivity:
    """Test custom connectivity through YAML config."""

    def test_yaml_build_with_custom_connectivity(self, tmp_path) -> None:
        """Test building model from YAML with custom connectivity."""
        # Create mask
        mask = np.eye(5, dtype=np.float32)
        npz_file = tmp_path / "mask.npz"
        np.savez(npz_file, mask=mask)

        # Create YAML-like config dict
        config = {
            "simulation": {"dt": 37, "network_evaluation_method": "layerwise"},
            "layers": [
                {"layer_id": 0, "layer_type": "Linear", "params": {"dim": 5}},
                {"layer_id": 1, "layer_type": "Linear", "params": {"dim": 5}},
            ],
            "connections": [
                {
                    "from_layer": 0,
                    "to_layer": 1,
                    "connection_type": "custom",
                    "params": {
                        "mask_file": str(npz_file),
                    },
                },
            ],
        }

        model = SOENModelCore.build(config)

        # Verify
        assert "J_0_to_1" in model.connections
        mask_tensor = model.connection_masks["J_0_to_1"]
        assert torch.allclose(mask_tensor, torch.eye(5))


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_very_sparse_mask(self, tmp_path) -> None:
        """Test with extremely sparse connectivity."""
        # Only 1 connection in a 100x100 matrix
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[50, 25] = 1.0
        npz_file = tmp_path / "very_sparse.npz"
        np.savez(npz_file, mask=mask)

        result = build_custom(100, 100, {"mask_file": str(npz_file)})
        assert result.sum() == 1.0
        assert result[50, 25] == 1.0

    def test_all_zeros_mask(self, tmp_path) -> None:
        """Test with mask that has no connections."""
        mask = np.zeros((5, 10), dtype=np.float32)
        npz_file = tmp_path / "zeros.npz"
        np.savez(npz_file, mask=mask)

        result = build_custom(10, 5, {"mask_file": str(npz_file)})
        assert result.sum() == 0.0

    def test_internal_connection_custom_mask(self, tmp_path) -> None:
        """Test custom mask for internal (same layer) connection."""
        # Create mask without diagonal (no self-connections)
        mask = np.ones((5, 5), dtype=np.float32) - np.eye(5, dtype=np.float32)
        npz_file = tmp_path / "internal.npz"
        np.savez(npz_file, mask=mask)

        sim = SimulationConfig(dt=37, network_evaluation_method="stepwise_gauss_seidel")
        layers = [
            LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 5}),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 5}),
        ]
        connections = [
            ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense"),
            ConnectionConfig(
                from_layer=1,
                to_layer=1,  # Internal connection
                connection_type="custom",
                params={"mask_file": str(npz_file)},
            ),
        ]

        model = SOENModelCore(sim, layers, connections)
        assert "J_1_to_1" in model.connections


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
