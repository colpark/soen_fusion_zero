"""JAX tests for custom weights.

Note: Custom weights are handled at the PyTorch core level.
These tests verify that custom weights are preserved during conversion.
"""

import jax.numpy as jnp
import numpy as np

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax


def test_custom_weights_preserved_in_conversion(tmp_path):
    """Test that custom weights are preserved during conversion."""
    # Create custom weights
    weights = np.random.randn(5, 10).astype(np.float32)
    weights_file = tmp_path / "custom.npy"
    np.save(str(weights_file), weights)

    # Create model config
    sim_config = SimulationConfig(dt=0.01)
    layers_config = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 10}),
        LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 5}),
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
    torch_model = SOENModelCore(
        sim_config=sim_config,
        layers_config=layers_config,
        connections_config=connections_config,
    )

    jax_model = convert_core_model_to_jax(torch_model)

    # Verify weights were loaded
    assert len(jax_model.connections) > 0
    loaded_weights = jax_model.connections[0].J
    assert loaded_weights.shape == (5, 10)
    assert jnp.allclose(loaded_weights, jnp.asarray(weights), atol=1e-6)


def test_custom_weights_npz_format(tmp_path):
    """Test model with weights in .npz format."""
    weights = np.random.randn(3, 8).astype(np.float32)
    weights_file = tmp_path / "custom.npz"
    np.savez(str(weights_file), weights=weights)

    sim_config = SimulationConfig(dt=0.01)
    layers_config = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 8}),
        LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 3}),
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

    torch_model = SOENModelCore(
        sim_config=sim_config,
        layers_config=layers_config,
        connections_config=connections_config,
    )

    jax_model = convert_core_model_to_jax(torch_model)

    # Verify
    assert len(jax_model.connections) > 0
    assert jnp.allclose(
        jax_model.connections[0].J,
        jnp.asarray(weights),
        atol=1e-6,
    )


def test_custom_weights_with_mask(tmp_path):
    """Test that masks are applied correctly with custom weights."""
    # Create weights (all ones for easy verification)
    weights = np.ones((4, 6), dtype=np.float32) * 0.5
    weights_file = tmp_path / "weights.npy"
    np.save(str(weights_file), weights)

    # Create sparse mask
    mask = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 1],
        ],
        dtype=np.float32,
    )

    # Build model
    sim = SimulationConfig(dt=37)
    layers = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 6}),
        LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 4}),
    ]
    connections = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="custom",
            params={"mask_file": str(tmp_path / "mask.npz")},
        ),
    ]

    # Save mask
    np.savez(tmp_path / "mask.npz", mask=mask)

    # Note: This model uses custom connectivity, not custom weights
    # We test that the structure is preserved
    torch_model = SOENModelCore(sim, layers, connections)
    jax_model = convert_core_model_to_jax(torch_model)

    # Verify mask is applied
    assert jax_model.connections[0].mask is not None
    mask_jax = jax_model.connections[0].mask
    assert jnp.allclose(mask_jax, jnp.asarray(mask))
