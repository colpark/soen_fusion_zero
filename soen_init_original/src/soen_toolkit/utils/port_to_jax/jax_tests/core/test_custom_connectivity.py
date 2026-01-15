"""JAX tests for custom connectivity.

Note: Custom connectivity is handled at the PyTorch core level.
These tests verify that custom connectivity is preserved during conversion.
"""

import jax
import jax.numpy as jnp
import numpy as np

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax


def test_custom_mask_preserved_in_conversion(tmp_path):
    """Test that custom connectivity masks are preserved during conversion."""
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

    # Build model with custom connectivity
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

    torch_model = SOENModelCore(sim, layers, connections)
    jax_model = convert_core_model_to_jax(torch_model)

    # Verify connection exists
    assert len(jax_model.connections) > 0
    conn = jax_model.connections[0]
    assert conn.mask is not None
    assert conn.mask.shape == (3, 5)

    # Verify mask matches
    expected_mask = jnp.asarray(mask)
    assert jnp.allclose(conn.mask, expected_mask)


def test_multiple_custom_connections(tmp_path):
    """Test model with multiple custom connectivity patterns."""
    # Create two different masks
    mask1 = np.eye(5, dtype=np.float32)
    mask2 = np.ones((5, 5), dtype=np.float32)

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

    torch_model = SOENModelCore(sim, layers, connections)
    jax_model = convert_core_model_to_jax(torch_model)

    # Verify both connections exist
    assert len(jax_model.connections) == 2

    # Verify masks are different
    mask_01 = jax_model.connections[0].mask
    mask_12 = jax_model.connections[1].mask
    assert mask_01 is not None
    assert mask_12 is not None
    assert not jnp.allclose(mask_01, mask_12)


def test_forward_pass_with_custom_mask(tmp_path):
    """Test that forward pass works correctly with custom connectivity."""
    # Create sparse mask (diagonal)
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

    torch_model = SOENModelCore(sim, layers, connections)
    jax_model = convert_core_model_to_jax(torch_model)

    # Run forward pass
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 5))
    output, _ = jax_model(x)

    # Verify output shape
    assert output.shape == (2, 11, 3)
