"""JAX tests for noise strategies.

Note: Noise strategies are primarily implemented in PyTorch core.
These tests verify that noise configurations are properly handled in conversion.
"""

import jax
import jax.numpy as jnp

from ...test_helpers import build_small_model_jax


def test_model_builds_without_noise() -> None:
    """Test that models can be built and converted without noise."""
    torch_model, jax_model = build_small_model_jax(dims=(3, 3), connectivity_type="dense", init="constant", init_value=0.1)

    # Model should be valid
    assert jax_model is not None
    assert len(jax_model.layers) > 0
    assert len(jax_model.connections) > 0


def test_deterministic_forward_without_noise() -> None:
    """Test that forward passes are deterministic when no noise is applied."""
    torch_model, jax_model = build_small_model_jax(dims=(4, 4), connectivity_type="dense", init="constant", init_value=0.05)

    x = jax.random.normal(jax.random.PRNGKey(999), (2, 5, 4))

    # Two forward passes should produce identical results
    y1, _ = jax_model(x)
    y2, _ = jax_model(x)

    assert jnp.allclose(y1, y2)


def test_input_noise_handling() -> None:
    """Test that models handle noisy inputs correctly."""
    torch_model, jax_model = build_small_model_jax(dims=(3, 3), connectivity_type="dense", init="constant", init_value=0.1)

    # Create noisy input
    base = jnp.zeros((2, 5, 3))
    noise = jax.random.normal(jax.random.PRNGKey(42), (2, 5, 3)) * 0.1
    x_noisy = base + noise

    # Should handle without error
    y, _ = jax_model(x_noisy)
    assert y.shape == (2, 6, 3)


def test_forward_shape_consistency() -> None:
    """Test that forward pass maintains shape consistency under various conditions."""
    torch_model, jax_model = build_small_model_jax(dims=(5, 7), connectivity_type="dense", init="constant", init_value=0.1)

    # Different input seeds
    for seed in range(5):
        x = jax.random.normal(jax.random.PRNGKey(seed), (3, 10, 5))
        y, _ = jax_model(x)
        # Output should have correct shape
        assert y.shape == (3, 11, 7)
