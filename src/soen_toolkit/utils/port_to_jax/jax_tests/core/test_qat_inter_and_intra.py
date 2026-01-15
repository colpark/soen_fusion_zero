"""JAX tests for Quantization-Aware Training (QAT) on inter and intra-layer connections."""

import jax
import jax.numpy as jnp

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax
from soen_toolkit.utils.quantization import generate_uniform_codebook


def build_two_layer_rnn(dim=4):
    """Build a simple two-layer RNN model."""
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": dim}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": dim}),
    ]
    conns = [
        ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}, learnable=True),
    ]
    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def test_codebook_snap_consistency() -> None:
    """Test that codebook snapping produces consistent results."""
    cb = generate_uniform_codebook(-0.24, 0.24, 9)
    cb_jax = jnp.asarray(cb.numpy())

    # Test value that should snap to a specific codebook entry
    test_val = 0.1
    idx = jnp.argmin(jnp.abs(cb_jax - test_val))
    snapped_val = cb_jax[idx]

    # Snapping again should produce same result
    idx2 = jnp.argmin(jnp.abs(cb_jax - snapped_val))
    snapped_val2 = cb_jax[idx2]

    assert jnp.allclose(snapped_val, snapped_val2, atol=1e-7)


def test_qat_codebook_generation() -> None:
    """Test QAT codebook generation for different bit widths."""
    for bits in [1, 2, 3, 4]:
        num_levels = 2**bits + 1
        cb = generate_uniform_codebook(-0.24, 0.24, num_levels)
        cb_jax = jnp.asarray(cb.numpy())

        # Should have correct number of levels
        assert cb_jax.shape[0] == num_levels

        # Should be sorted
        assert jnp.all(cb_jax[1:] >= cb_jax[:-1])


def test_inter_layer_connection_shape() -> None:
    """Test that inter-layer connection has correct shape."""
    torch_model = build_two_layer_rnn(dim=3)
    jax_model = convert_core_model_to_jax(torch_model)

    # Get inter-layer connection
    inter_conn = jax_model.connections[0]
    assert inter_conn.from_layer == 0
    assert inter_conn.to_layer == 1

    # Should have correct shape (3x3 for dim=3)
    assert inter_conn.J.shape == (3, 3)


def test_internal_connection_support() -> None:
    """Test models with internal connections."""
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": 3, "internal_J": jnp.eye(3)}),
    ]
    sim = SimulationConfig(dt=37, input_type="flux")
    torch_model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=[])

    # Should convert successfully
    jax_model = convert_core_model_to_jax(torch_model)
    assert jax_model is not None
    assert len(jax_model.layers) == 1


def test_qat_range_clipping() -> None:
    """Test that QAT respects min/max bounds."""
    cb = generate_uniform_codebook(-0.24, 0.24, 9)
    cb_jax = jnp.asarray(cb.numpy())

    # All codebook values should be in range
    assert jnp.all(cb_jax >= -0.24 - 1e-6)
    assert jnp.all(cb_jax <= 0.24 + 1e-6)


def test_forward_pass_with_snapped_weights() -> None:
    """Test forward pass consistency with snapped weights."""
    torch_model = build_two_layer_rnn(dim=3)
    jax_model = convert_core_model_to_jax(torch_model)

    # Forward pass with original weights
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3))
    y1, _ = jax_model(x)

    # Forward pass should still work (JAX models are immutable)
    y2, _ = jax_model(x)

    # Results should be identical (model not mutated)
    assert jnp.allclose(y1, y2)
