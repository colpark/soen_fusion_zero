import jax
import jax.numpy as jnp

from ...test_helpers import (
    build_small_model_jax,
    make_random_series_jax,
)


def test_forward_determinism_no_noise_fixed_seed() -> None:
    torch_model, jax_model = build_small_model_jax(dims=(4, 4), connectivity_type="dense", init="constant", init_value=0.05)
    x = make_random_series_jax(batch=2, seq_len=5, dim=4, seed=999)

    y1, _ = jax_model(x)
    # Reset nothing; just run again with same inputs (no noise in model)
    y2, _ = jax_model(x)
    assert jnp.allclose(y1, y2)


def test_forward_determinism_across_instances_same_seed() -> None:
    _, jax_model1 = build_small_model_jax(dims=(3, 3), connectivity_type="dense", init="constant", init_value=0.1)
    _, jax_model2 = build_small_model_jax(dims=(3, 3), connectivity_type="dense", init="constant", init_value=0.1)

    x = make_random_series_jax(batch=1, seq_len=3, dim=3, seed=7)
    y1, _ = jax_model1(x)
    y2, _ = jax_model2(x)
    assert jnp.allclose(y1, y2)


def test_torch_jax_output_equivalence() -> None:
    """Test that converted JAX model produces same output as PyTorch model."""
    torch_model, jax_model = build_small_model_jax(dims=(3, 3), connectivity_type="dense", init="constant", init_value=0.1)

    # Use same input
    x_np = jax.random.normal(jax.random.PRNGKey(42), (2, 4, 3))

    # PyTorch forward
    import torch

    x_torch = torch.from_numpy(jnp.asarray(x_np))
    with torch.no_grad():
        y_torch, _ = torch_model(x_torch)

    # JAX forward
    y_jax, _ = jax_model(x_np)

    # Compare outputs
    y_torch_np = jnp.asarray(y_torch.numpy())
    assert jnp.allclose(y_jax, y_torch_np, atol=1e-5)
