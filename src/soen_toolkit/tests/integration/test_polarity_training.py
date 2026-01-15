"""Integration tests for neuron polarity with training and YAML loading."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch

from soen_toolkit.core.configs import ConnectionConfig, LayerConfig, SimulationConfig
from soen_toolkit.core.model_yaml import build_model_from_yaml
from soen_toolkit.core.soen_model_core import SOENModelCore
from soen_toolkit.utils.polarity_utils import (
    POLARITY_ENFORCEMENT_CLIP,
    POLARITY_ENFORCEMENT_SIGN_FLIP,
    generate_alternating_polarity,
    save_polarity,
)


class TestPolarityTraining:
    """Test that polarity constraints are maintained during training."""

    def test_gradient_flow_with_polarity(self) -> None:
        """Test that gradients flow correctly with polarity constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "polarity.npy"
            polarity = generate_alternating_polarity(10)
            save_polarity(polarity, str(polarity_file))

            # Create simple 2-layer model - polarity in layer config
            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 10, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 5}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "uniform", "min": -0.5, "max": 0.5},
                    learnable=True,
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
            model.enforce_param_constraints()

            # Create dummy input and target
            batch_size = 4
            seq_len = 10
            x = torch.randn(batch_size, seq_len, 10)
            target = torch.randn(batch_size, seq_len, 5)

            # Forward pass
            output_tuple = model(x)
            output = output_tuple[0] if isinstance(output_tuple, tuple) else output_tuple

            # Compute loss
            loss = torch.nn.functional.mse_loss(output[:, -1, :], target[:, -1, :])

            # Backward pass
            loss.backward()

            # Check that gradients exist
            key = "J_0_to_1"
            assert model.connections[key].grad is not None
            assert not torch.all(model.connections[key].grad == 0.0)

    def test_constraints_maintained_after_optimizer_step(self) -> None:
        """Test that polarity constraints are maintained after optimizer updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "polarity.npy"
            # All excitatory
            polarity = np.ones(8, dtype=np.int8)
            save_polarity(polarity, str(polarity_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 8, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 4}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "uniform", "min": -0.3, "max": 0.3},
                    learnable=True,
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            # Training loop
            for _ in range(5):
                optimizer.zero_grad()

                x = torch.randn(2, 5, 8)
                target = torch.randn(2, 5, 4)

                output_tuple = model(x)
                output = output_tuple[0] if isinstance(output_tuple, tuple) else output_tuple
                loss = torch.nn.functional.mse_loss(output[:, -1, :], target[:, -1, :])
                loss.backward()

                optimizer.step()

                # Apply constraints after optimizer step
                model.enforce_param_constraints()

                # Verify all weights are non-negative (excitatory)
                weights = model.connections["J_0_to_1"]
                assert torch.all(weights >= 0.0), "Excitatory constraint violated after step"

    def test_sign_violations_prevented(self) -> None:
        """Test that sign violations are prevented throughout training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "mixed.npy"
            # Alternating excitatory/inhibitory
            polarity = generate_alternating_polarity(6)
            save_polarity(polarity, str(polarity_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 6, "polarity_file": str(polarity_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 3}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "normal", "std": 0.1},
                    learnable=True,
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Train for several steps
            num_steps = 20
            for step in range(num_steps):
                optimizer.zero_grad()

                x = torch.randn(4, 10, 6)
                target = torch.randn(4, 11, 3)  # Model outputs seq_len+1 (includes initial state)

                output_tuple = model(x)
                output = output_tuple[0] if isinstance(output_tuple, tuple) else output_tuple
                loss = torch.nn.functional.mse_loss(output, target)
                loss.backward()

                optimizer.step()
                model.enforce_param_constraints()

                # Verify constraints for each neuron
                weights = model.connections["J_0_to_1"]
                for i in range(6):
                    if polarity[i] == 1:  # Excitatory
                        assert torch.all(weights[:, i] >= 0.0), f"Step {step}: Excitatory neuron {i} has negative weights"
                    elif polarity[i] == -1:  # Inhibitory
                        assert torch.all(weights[:, i] <= 0.0), f"Step {step}: Inhibitory neuron {i} has positive weights"


class TestPolarityRecurrent:
    """Test polarity with recurrent connections."""

    def test_recurrent_sign_flip_default(self) -> None:
        """Test that recurrent connections with mixed polarity use sign_flip by default.

        With sign_flip (the default), magnitudes are preserved but signs are adjusted:
        - Excitatory neurons: all outgoing weights become abs(weight) >= 0
        - Inhibitory neurons: all outgoing weights become -abs(weight) <= 0
        No weights should be zero (unless initialized to zero).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "alternating.npy"
            polarity = generate_alternating_polarity(10)
            save_polarity(polarity, str(polarity_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 10}),
                LayerConfig(
                    layer_id=1,
                    layer_type="SingleDendrite",
                    params={
                        "dim": 10,
                        "polarity_file": str(polarity_file),
                        # sign_flip is default, but be explicit for clarity
                        "polarity_enforcement_method": POLARITY_ENFORCEMENT_SIGN_FLIP,
                    },
                ),
            ]
            conns = [
                ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}),
                # Recurrent - polarity from layer config
                ConnectionConfig(
                    from_layer=1,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "uniform", "min": -1, "max": 1},
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
            model.enforce_param_constraints()

            # Check recurrent weights
            weights = model.connections["J_1_to_1"]

            # With sign_flip, no weights should be zeroed (magnitudes preserved)
            # Check that sign constraints are satisfied
            for src in range(10):
                if polarity[src] == 1:  # Excitatory
                    assert torch.all(weights[:, src] >= 0.0), f"Excitatory neuron {src} has negative weights"
                elif polarity[src] == -1:  # Inhibitory
                    assert torch.all(weights[:, src] <= 0.0), f"Inhibitory neuron {src} has positive weights"

            # With sign_flip and uniform [-1, 1] init, weights should NOT be zeroed
            # (they keep their magnitude). Count zeros to verify.
            num_zeros = torch.sum(torch.abs(weights) < 1e-6).item()
            total_weights = weights.numel()
            zero_percentage = num_zeros / total_weights
            # Very few zeros expected (only if init happened to be exactly 0)
            assert zero_percentage < 0.05, f"sign_flip should preserve magnitudes, but got {zero_percentage:.2%} zeros"

    def test_recurrent_clip_to_zero(self) -> None:
        """Test that clip_to_zero method zeros ~50% of weights with mixed polarity.

        With clip_to_zero, violating weights are clipped to zero:
        - Excitatory neurons: negative weights become 0
        - Inhibitory neurons: positive weights become 0
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            polarity_file = Path(tmpdir) / "alternating.npy"
            polarity = generate_alternating_polarity(10)
            save_polarity(polarity, str(polarity_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 10}),
                LayerConfig(
                    layer_id=1,
                    layer_type="SingleDendrite",
                    params={
                        "dim": 10,
                        "polarity_file": str(polarity_file),
                        "polarity_enforcement_method": POLARITY_ENFORCEMENT_CLIP,
                    },
                ),
            ]
            conns = [
                ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}),
                # Recurrent - polarity from layer config
                ConnectionConfig(
                    from_layer=1,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "uniform", "min": -1, "max": 1},
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
            model.enforce_param_constraints()

            # Check recurrent weights
            weights = model.connections["J_1_to_1"]

            # Count zeros (or near-zeros)
            num_zeros = torch.sum(torch.abs(weights) < 1e-6).item()
            total_weights = weights.numel()

            # With alternating polarity and uniform init from [-1, 1],
            # ~50% of weights will be clipped to zero
            zero_percentage = num_zeros / total_weights

            # Allow some tolerance since init is random
            assert zero_percentage > 0.2, f"Expected significant weight zeroing with clip_to_zero, got {zero_percentage:.2%}"


class TestYAMLLoading:
    """Test loading models with polarity from YAML configuration."""

    def test_yaml_with_polarity(self) -> None:
        """Test loading a model with polarity specification from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create polarity file
            polarity_file = Path(tmpdir) / "layer0_polarity.npy"
            polarity = generate_alternating_polarity(8)
            save_polarity(polarity, str(polarity_file))

            # Create YAML config
            yaml_content = f"""
simulation:
  dt: 37
  input_type: state

layers:
  - layer_id: 0
    layer_type: Input
    params:
      dim: 8
      polarity_file: "{polarity_file}"

  - layer_id: 1
    layer_type: Linear
    params:
      dim: 4

connections:
  - from_layer: 0
    to_layer: 1
    connection_type: dense
    params:
      init: uniform
      min: -0.5
      max: 0.5
    learnable: true
"""

            yaml_file = Path(tmpdir) / "model_config.yaml"
            with open(yaml_file, "w") as f:
                f.write(yaml_content)

            # Load model from YAML
            model = build_model_from_yaml(str(yaml_file))

            # Verify polarity constraints were applied
            assert "J_0_to_1" in model.connection_constraint_min_matrices
            assert "J_0_to_1" in model.connection_constraint_max_matrices

            # Apply constraints and verify
            model.enforce_param_constraints()
            weights = model.connections["J_0_to_1"]

            # Check alternating pattern
            for i in range(8):
                if i % 2 == 0:  # Excitatory (even indices)
                    assert torch.all(weights[:, i] >= 0.0)
                else:  # Inhibitory (odd indices)
                    assert torch.all(weights[:, i] <= 0.0)

    def test_yaml_without_polarity_backward_compatible(self) -> None:
        """Test that models without polarity still work (backward compatibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = """
simulation:
  dt: 37

layers:
  - layer_id: 0
    layer_type: Input
    params:
      dim: 5

  - layer_id: 1
    layer_type: Linear
    params:
      dim: 3

connections:
  - from_layer: 0
    to_layer: 1
    connection_type: dense
    params:
      init: normal
      std: 0.1
"""

            yaml_file = Path(tmpdir) / "simple_config.yaml"
            with open(yaml_file, "w") as f:
                f.write(yaml_content)

            # Should load without errors
            model = build_model_from_yaml(str(yaml_file))

            # Should not have constraint matrices
            assert "J_0_to_1" not in getattr(model, "connection_constraint_min_matrices", {})
            assert "J_0_to_1" not in getattr(model, "connection_constraint_max_matrices", {})

            # Should work normally
            x = torch.randn(2, 10, 5)
            output_tuple = model(x)
            output = output_tuple[0] if isinstance(output_tuple, tuple) else output_tuple
            # Model outputs seq_len+1 (includes initial state)
            assert output.shape == (2, 11, 3)


class TestMultiLayerPolarity:
    """Test polarity with multiple layers and connections."""

    def test_multiple_connections_with_different_polarity(self) -> None:
        """Test model with multiple connections, each with different polarity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create different polarity files
            polarity0_file = Path(tmpdir) / "layer0_polarity.npy"
            polarity0 = np.ones(6, dtype=np.int8)  # All excitatory
            save_polarity(polarity0, str(polarity0_file))

            polarity1_file = Path(tmpdir) / "layer1_polarity.npy"
            polarity1 = -np.ones(4, dtype=np.int8)  # All inhibitory
            save_polarity(polarity1, str(polarity1_file))

            sim = SimulationConfig(dt=37)
            layers = [
                LayerConfig(layer_id=0, layer_type="Input", params={"dim": 6, "polarity_file": str(polarity0_file)}),
                LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 4, "polarity_file": str(polarity1_file)}),
                LayerConfig(layer_id=2, layer_type="Linear", params={"dim": 3}),
            ]
            conns = [
                ConnectionConfig(
                    from_layer=0,
                    to_layer=1,
                    connection_type="dense",
                    params={"init": "uniform", "min": -1, "max": 1},
                ),
                ConnectionConfig(
                    from_layer=1,
                    to_layer=2,
                    connection_type="dense",
                    params={"init": "uniform", "min": -1, "max": 1},
                ),
            ]

            model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
            model.enforce_param_constraints()

            # Verify first connection (all excitatory)
            weights_0_1 = model.connections["J_0_to_1"]
            assert torch.all(weights_0_1 >= 0.0)

            # Verify second connection (all inhibitory)
            weights_1_2 = model.connections["J_1_to_2"]
            assert torch.all(weights_1_2 <= 0.0)

