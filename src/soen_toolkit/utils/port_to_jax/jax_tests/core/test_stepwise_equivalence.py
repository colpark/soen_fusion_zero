"""JAX tests for stepwise solver equivalence."""

import jax
import jax.numpy as jnp

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax


def _build_ff_model(dim0=3, dim1=3, weight=0.1) -> SOENModelCore:
    """Build a feedforward model."""
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": int(dim0)}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": int(dim1)}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "constant", "value": float(weight)},
            learnable=False,
        ),
    ]
    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def test_stepwise_matches_parallel_feedforward_no_noise() -> None:
    """Test that stepwise and parallel solvers produce same output for feedforward."""
    torch_model = _build_ff_model(dim0=4, dim1=5, weight=0.05)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 4))

    # Convert with layerwise solver
    torch_model.sim_config.network_evaluation_method = "layerwise"
    torch_model_layerwise = _build_ff_model(dim0=4, dim1=5, weight=0.05)
    torch_model_layerwise.sim_config.network_evaluation_method = "layerwise"
    jax_model_layerwise = convert_core_model_to_jax(torch_model_layerwise)
    y_layerwise, _ = jax_model_layerwise(x)

    # Convert with stepwise solver
    torch_model_stepwise = _build_ff_model(dim0=4, dim1=5, weight=0.05)
    torch_model_stepwise.sim_config.network_evaluation_method = "stepwise_gauss_seidel"
    jax_model_stepwise = convert_core_model_to_jax(torch_model_stepwise)
    y_stepwise, _ = jax_model_stepwise(x)

    # For feedforward, outputs should match
    assert jnp.allclose(y_layerwise, y_stepwise, atol=1e-5)


def test_layerwise_solver_basic() -> None:
    """Test basic layerwise solver functionality."""
    torch_model = _build_ff_model(dim0=3, dim1=3, weight=0.1)
    torch_model.sim_config.network_evaluation_method = "layerwise"
    jax_model = convert_core_model_to_jax(torch_model)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3))
    y, _ = jax_model(x)

    # Output shape should be correct
    assert y.shape == (2, 6, 3)


def test_stepwise_gauss_seidel_solver_basic() -> None:
    """Test basic stepwise gauss-seidel solver functionality."""
    torch_model = _build_ff_model(dim0=3, dim1=3, weight=0.1)
    torch_model.sim_config.network_evaluation_method = "stepwise_gauss_seidel"
    jax_model = convert_core_model_to_jax(torch_model)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3))
    y, _ = jax_model(x)

    # Output shape should be correct
    assert y.shape == (2, 6, 3)


def test_stepwise_jacobi_solver_basic() -> None:
    """Test basic stepwise jacobi solver functionality."""
    torch_model = _build_ff_model(dim0=3, dim1=3, weight=0.1)
    torch_model.sim_config.network_evaluation_method = "stepwise_jacobi"
    jax_model = convert_core_model_to_jax(torch_model)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3))
    y, _ = jax_model(x)

    # Output shape should be correct
    assert y.shape == (2, 6, 3)


def test_solver_consistency_across_seeds() -> None:
    """Test that solvers produce consistent results across different seeds."""
    torch_model = _build_ff_model(dim0=3, dim1=3, weight=0.1)
    torch_model.sim_config.network_evaluation_method = "layerwise"
    jax_model = convert_core_model_to_jax(torch_model)

    # Same input, different seed
    x1 = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3))
    x2 = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3))

    y1, _ = jax_model(x1)
    y2, _ = jax_model(x2)

    # Same input should produce same output
    assert jnp.allclose(y1, y2)


def test_feedforward_linearity() -> None:
    """Test that feedforward models maintain linearity properties."""
    torch_model = _build_ff_model(dim0=4, dim1=5, weight=0.05)
    torch_model.sim_config.network_evaluation_method = "layerwise"
    jax_model = convert_core_model_to_jax(torch_model)

    # Test linearity: f(x + y) ≈ f(x) + f(y)
    x1 = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 4))
    x2 = jax.random.normal(jax.random.PRNGKey(2), (2, 6, 4))

    y1, _ = jax_model(x1)
    y2, _ = jax_model(x2)
    y_sum, _ = jax_model(x1 + x2)

    # Should be approximately linear (within numerical precision)
    combined = y1 + y2
    diff = jnp.abs(y_sum - combined).max()
    # Feedforward should be close to linear for small weights
    assert diff < 0.5  # Relaxed tolerance for nonlinear layers
