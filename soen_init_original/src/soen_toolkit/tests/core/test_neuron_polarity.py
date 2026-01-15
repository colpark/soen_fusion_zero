"""Unit tests for neuron polarity constraints feature."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from soen_toolkit.core.configs import ConnectionConfig, LayerConfig, SimulationConfig
from soen_toolkit.core.layers.common.connectivity_metadata import load_neuron_polarity
from soen_toolkit.core.soen_model_core import SOENModelCore
from soen_toolkit.utils.polarity_utils import (
    generate_alternating_polarity,
    generate_excitatory_polarity,
    generate_inhibitory_polarity,
    generate_random_polarity,
    save_polarity,
)


class TestPolarityInit:
    """Test convenience polarity_init parameter."""

    def test_alternating_polarity_init(self) -> None:
        """Test alternating polarity without file."""
        sim = SimulationConfig(dt=37)
        layers = [
            LayerConfig(layer_id=0, layer_type="Input", params={"dim": 10, "polarity_init": "alternating"}),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 5}),
        ]
        conns = [
            ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}),
        ]

        model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

        # Check constraint matrices exist
        key = "J_0_to_1"
        assert key in model.connection_constraint_min_matrices

        # Verify alternating pattern (even=excitatory, odd=inhibitory)
        min_mat = model.connection_constraint_min_matrices[key]
        max_mat = model.connection_constraint_max_matrices[key]

        for i in range(10):
            if i % 2 == 0:  # Excitatory
                assert torch.all(min_mat[:, i] == 0.0)
            else:  # Inhibitory
                assert torch.all(max_mat[:, i] == 0.0)

    def test_50_50_polarity_init(self) -> None:
        """Test 50_50 alias for alternating."""
        sim = SimulationConfig(dt=37)
        layers = [
            LayerConfig(layer_id=0, layer_type="Input", params={"dim": 6, "polarity_init": "50_50"}),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 3}),
        ]
        conns = [
            ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}),
        ]

        model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

        # Should work exactly like "alternating"
        assert "J_0_to_1" in model.connection_constraint_min_matrices

    def test_excitatory_polarity_init(self) -> None:
        """Test explicit excitatory polarity init."""
        sim = SimulationConfig(dt=37)
        layers = [
            LayerConfig(layer_id=0, layer_type="Input", params={"dim": 10, "polarity_init": "excitatory"}),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 5}),
        ]
        conns = [
            ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}),
        ]

        model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

        key = "J_0_to_1"
        min_mat = model.connection_constraint_min_matrices[key]
        # All columns should have min constraint 0.0
        assert torch.all(min_mat == 0.0)

    def test_inhibitory_polarity_init(self) -> None:
        """Test explicit inhibitory polarity init."""
        sim = SimulationConfig(dt=37)
        layers = [
            LayerConfig(layer_id=0, layer_type="Input", params={"dim": 10, "polarity_init": "inhibitory"}),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 5}),
        ]
        conns = [
            ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}),
        ]

        model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

        key = "J_0_to_1"
        max_mat = model.connection_constraint_max_matrices[key]
        # All columns should have max constraint 0.0
        assert torch.all(max_mat == 0.0)

    def test_random_polarity_init(self) -> None:
        """Test random polarity with custom ratio."""
        sim = SimulationConfig(dt=37)
        layers = [
            LayerConfig(
                layer_id=0,
                layer_type="Input",
                params={
                    "dim": 100,
                    "polarity_init": {"excitatory_ratio": 0.8, "seed": 42}
                }
            ),
            LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 50}),
        ]
        conns = [
            ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}),
        ]

        model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

        # Check that constraints exist
        assert "J_0_to_1" in model.connection_constraint_min_matrices


class TestPolarityUtils:
    """Test polarity generation utilities."""

    def test_generate_alternating_polarity(self) -> None:
        """Test alternating polarity generation."""
        polarity = generate_alternating_polarity(10)

        assert len(polarity) == 10
        assert polarity.dtype == np.int8

        # Check alternating pattern
        assert np.array_equal(polarity[::2], np.ones(5, dtype=np.int8))  # Even indices: excitatory
        assert np.array_equal(polarity[1::2], -np.ones(5, dtype=np.int8))  # Odd indices: inhibitory

    def test_generate_random_polarity(self) -> None:
        """Test random polarity generation with seed."""
        polarity1 = generate_random_polarity(100, excitatory_ratio=0.8, seed=42)
        polarity2 = generate_random_polarity(100, excitatory_ratio=0.8, seed=42)

        # Same seed should give same result
        assert np.array_equal(polarity1, polarity2)

        # Check ratio (allow ±1 due to rounding)
        n_excitatory = (polarity1 == 1).sum()
        assert 79 <= n_excitatory <= 81

    def test_generate_random_polarity_all_excitatory(self) -> None:
        """Test random polarity with 100% excitatory."""
        polarity = generate_random_polarity(50, excitatory_ratio=1.0)
        assert np.all(polarity == 1)

    def test_generate_random_polarity_all_inhibitory(self) -> None:
        """Test random polarity with 0% excitatory."""
        polarity = generate_random_polarity(50, excitatory_ratio=0.0)
        assert np.all(polarity == -1)

    def test_generate_excitatory_polarity(self) -> None:
        """Test pure excitatory generator."""
        polarity = generate_excitatory_polarity(50)
        assert len(polarity) == 50
        assert np.all(polarity == 1)

    def test_generate_inhibitory_polarity(self) -> None:
        """Test pure inhibitory generator."""
        polarity = generate_inhibitory_polarity(50)
        assert len(polarity) == 50
        assert np.all(polarity == -1)

    def test_generate_random_polarity_invalid_ratio(self) -> None:
        """Test that invalid ratios raise errors."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            generate_random_polarity(10, excitatory_ratio=1.5)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            generate_random_polarity(10, excitatory_ratio=-0.5)

    def test_save_and_load_polarity(self) -> None:
        """Test saving and loading polarity arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "test_polarity.npy"

            # Create and save
            original = generate_alternating_polarity(20)
            save_polarity(original, str(polarity_file))

            # Load and verify
            loaded = load_neuron_polarity(str(polarity_file))
            assert torch.all(loaded == torch.from_numpy(original))


class TestPolarityLoading:
    """Test polarity file loading and validation."""

    def test_load_valid_polarity(self) -> None:
        """Test loading a valid polarity file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "polarity.npy"

            # Create valid polarity
            polarity = np.array([1, -1, 0, 1, -1], dtype=np.int8)
            np.save(polarity_file, polarity)

            # Load
            loaded = load_neuron_polarity(str(polarity_file))
            assert torch.all(loaded == torch.tensor([1, -1, 0, 1, -1], dtype=torch.int8))

    def test_load_invalid_polarity_values(self) -> None:
        """Test that invalid polarity values raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "invalid_polarity.npy"

            # Create invalid polarity (value 2 is not allowed)
            invalid_polarity = np.array([1, -1, 2, 1], dtype=np.int8)
            np.save(polarity_file, invalid_polarity)

            # Should raise ValueError
            with pytest.raises(ValueError, match="Polarity values must be -1, 0, or 1"):
                load_neuron_polarity(str(polarity_file))


class TestConstraintMatrixGeneration:
    """Test constraint matrix generation from polarity."""

    def test_excitatory_constraints(self) -> None:
        """Test that excitatory neurons have min=0 constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "excitatory.npy"

            # All excitatory
            polarity = np.ones(5, dtype=np.int8)
            save_polarity(polarity, str(polarity_file))

            # Create simple model - polarity specified in LAYER config
            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 5, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 3}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "uniform", "min": -1, "max": 1},
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

            # Check constraint matrices exist
            key = "J_0_to_1"
            assert key in model.connection_constraint_min_matrices
            assert key in model.connection_constraint_max_matrices

            # All outgoing connections should have min=0
            min_mat = model.connection_constraint_min_matrices[key]
            assert torch.all(min_mat == 0.0)

    def test_inhibitory_constraints(self) -> None:
        """Test that inhibitory neurons have max=0 constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "inhibitory.npy"

            # All inhibitory
            polarity = -np.ones(5, dtype=np.int8)
            save_polarity(polarity, str(polarity_file))

            # Create simple model - polarity in layer config
            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 5, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 3}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={},
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

            # All outgoing connections should have max=0
            key = "J_0_to_1"
            max_mat = model.connection_constraint_max_matrices[key]
            assert torch.all(max_mat == 0.0)

    def test_mixed_polarity_constraints(self) -> None:
        """Test mixed excitatory/inhibitory neurons."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "mixed.npy"

            # Mixed: [excitatory, inhibitory, normal, excitatory, inhibitory]
            polarity = np.array([1, -1, 0, 1, -1], dtype=np.int8)
            save_polarity(polarity, str(polarity_file))

            # Create model - polarity in layer config
            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 5, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 4}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={},
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

            key = "J_0_to_1"
            min_mat = model.connection_constraint_min_matrices[key]
            max_mat = model.connection_constraint_max_matrices[key]

            # Check constraints per source neuron (columns)
            # Source 0 (excitatory): min=0
            assert torch.all(min_mat[:, 0] == 0.0)
            # Source 1 (inhibitory): max=0
            assert torch.all(max_mat[:, 1] == 0.0)
            # Source 2 (normal): no constraints
            assert torch.all(min_mat[:, 2] == -float('inf'))
            assert torch.all(max_mat[:, 2] == float('inf'))

    def test_polarity_with_scalar_constraints(self) -> None:
        """Test that polarity constraints combine with scalar constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "excitatory.npy"

            # All excitatory
            polarity = np.ones(3, dtype=np.int8)
            save_polarity(polarity, str(polarity_file))

            # Create model with polarity in layer + scalar constraint in connection
            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 3, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 2}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={
                        "constraints": {"min": -0.5, "max": 0.5},
                    },
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

            key = "J_0_to_1"
            min_mat = model.connection_constraint_min_matrices[key]
            max_mat = model.connection_constraint_max_matrices[key]

            # Should have min=0 (from polarity) and max=0.5 (from scalar constraint)
            assert torch.all(min_mat == 0.0)
            assert torch.all(max_mat == 0.5)


class TestConstraintEnforcement:
    """Test that constraints are enforced on weights."""

    def test_excitatory_enforcement_after_init(self) -> None:
        """Test excitatory constraints are enforced after initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "excitatory.npy"
            polarity = np.ones(5, dtype=np.int8)
            save_polarity(polarity, str(polarity_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 5, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 3}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "uniform", "min": -1, "max": 1},
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

            # Apply constraints
            model.enforce_param_constraints()

            # All weights should be non-negative
            weights = model.connections["J_0_to_1"]
            assert torch.all(weights >= 0.0)

    def test_inhibitory_enforcement_after_gradient_update(self) -> None:
        """Test inhibitory constraints are enforced after gradient updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "inhibitory.npy"
            polarity = -np.ones(4, dtype=np.int8)
            save_polarity(polarity, str(polarity_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 4, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 3}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "uniform", "min": -1, "max": 1},
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

            # Simulate gradient update by adding positive values
            key = "J_0_to_1"
            with torch.no_grad():
                model.connections[key].add_(0.5)  # Try to make weights positive

            # Apply constraints
            model.enforce_param_constraints()

            # All weights should still be non-positive
            weights = model.connections[key]
            assert torch.all(weights <= 0.0)


class TestPolarityErrors:
    """Test error handling for polarity constraints."""

    def test_dynamic_weights_incompatibility(self) -> None:
        """Test that polarity with dynamic weights raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "polarity.npy"
            polarity = generate_alternating_polarity(5)
            save_polarity(polarity, str(polarity_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="SingleDendrite", params={"dim": 5, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="SingleDendrite", params={"dim": 3}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={
                        "connection_params": {"mode": "WICC"},
                    },
                ),
            ]

            # Should raise error about incompatibility
            with pytest.raises(ValueError, match="incompatible with dynamic weights"):
                SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

    def test_wrong_polarity_size(self) -> None:
        """Test that mismatched polarity size raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "polarity.npy"
            # Create polarity for 3 neurons
            polarity = generate_alternating_polarity(3)
            save_polarity(polarity, str(polarity_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 5, "polarity_file": str(polarity_file)}),  # 5 neurons, not 3!
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 2}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={},
                ),
            ]

            # Should raise error about size mismatch
            with pytest.raises(ValueError, match="Polarity length"):
                SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


class TestRecurrentConnections:
    """Test polarity with recurrent (internal) connections."""

    def test_recurrent_with_alternating_polarity(self) -> None:
        """Test that recurrent connections work with mixed polarity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "alternating.npy"
            polarity = generate_alternating_polarity(6)
            save_polarity(polarity, str(polarity_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 6}),
                LayerConfig(layer_id=1, layer_type="SingleDendrite", params={"dim": 6, "polarity_file": str(polarity_file)}),
            ]
            conns = [
                ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}),
                # Recurrent connection - polarity from layer config
                ConnectionConfig(
                    from_layer=1,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "uniform", "min": -0.5, "max": 0.5},
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
            model.enforce_param_constraints()

            # Check that constraints are applied
            key = "J_1_to_1"
            weights = model.connections[key]

            # Check each source neuron's outgoing weights
            # Even indices (0, 2, 4) are excitatory: weights should be >= 0
            for i in range(0, 6, 2):
                assert torch.all(weights[:, i] >= 0.0), f"Excitatory neuron {i} has negative outgoing weights"

            # Odd indices (1, 3, 5) are inhibitory: weights should be <= 0
            for i in range(1, 6, 2):
                assert torch.all(weights[:, i] <= 0.0), f"Inhibitory neuron {i} has positive outgoing weights"
