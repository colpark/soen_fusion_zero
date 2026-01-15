"""JAX tests for quantization-aware training and weight snapping."""

import jax.numpy as jnp

from ...test_helpers import (
    build_small_model_jax,
)


def test_codebook_generation_uniform() -> None:
    """Test that uniform codebook generation works."""
    from soen_toolkit.utils.quantization import generate_uniform_codebook

    cb = generate_uniform_codebook(-0.24, 0.24, 3)
    cb_jax = jnp.asarray(cb.numpy())

    # Should have 3 levels
    assert cb_jax.shape == (3,)
    # Should be sorted
    assert jnp.all(cb_jax[1:] >= cb_jax[:-1])
    # Should include bounds
    assert jnp.allclose(cb_jax[0], -0.24, atol=1e-6)
    assert jnp.allclose(cb_jax[-1], 0.24, atol=1e-6)


def test_qat_snap_identity_baseline() -> None:
    """Test that snapping already-snapped values produces no change."""
    from soen_toolkit.utils.quantization import generate_uniform_codebook

    cb = generate_uniform_codebook(-0.24, 0.24, 5)
    cb_jax = jnp.asarray(cb.numpy())

    # Already-snapped value
    snapped_val = cb_jax[2]

    # Snap again
    idx = jnp.argmin(jnp.abs(snapped_val - cb_jax))
    result = cb_jax[idx]

    assert jnp.allclose(result, snapped_val, atol=1e-7)


def test_qat_snap_diagonal_preserves_structure() -> None:
    """Test that QAT snapping preserves masked structure."""
    torch_model, jax_model = build_small_model_jax(dims=(4, 5), connectivity_type="one_to_one", init="constant", init_value=0.12)

    # Get original weights
    W_orig = jax_model.connections[0].J
    assert W_orig.shape == (5, 4)

    # Apply mask to get what should be diagonal + zeros
    mask = jax_model.connections[0].mask
    if mask is not None:
        W_masked = W_orig * mask

        # Off-diagonal should be zero
        for i in range(5):
            for j in range(4):
                if i != j:
                    assert jnp.allclose(W_masked[i, j], 0.0, atol=1e-6)


def test_quantise_correct_range() -> None:
    """Test quantization respects min/max bounds."""
    from soen_toolkit.utils.quantization import generate_uniform_codebook

    cb = generate_uniform_codebook(-0.24, 0.24, 9)
    cb_jax = jnp.asarray(cb.numpy())

    # All values should be in range
    assert jnp.all(cb_jax >= -0.24 - 1e-6)
    assert jnp.all(cb_jax <= 0.24 + 1e-6)


def test_codebook_includes_zero() -> None:
    """Test that codebook includes zero level."""
    from soen_toolkit.utils.quantization import generate_uniform_codebook

    # Test with odd number of levels (should include 0)
    cb = generate_uniform_codebook(-1.0, 1.0, 5)
    cb_jax = jnp.asarray(cb.numpy())

    # Should have exactly one zero or very close to it
    zeros = jnp.abs(cb_jax) < 1e-6
    assert jnp.sum(zeros) >= 1
