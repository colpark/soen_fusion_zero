import jax
import jax.numpy as jnp

from soen_toolkit.core import LayerConfig, SimulationConfig, SOENModelCore
from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax


def _build_single_dendrite_model(dim=4, *, dt=37, learnable=False, source_func="Tanh"):
    layers = [
        LayerConfig(
            layer_id=0,
            layer_type="SingleDendrite",
            params={
                "dim": int(dim),
                "solver": "FE",
                "source_func": source_func,
                "gamma_plus": 0.001,
                "gamma_minus": 0.001,
                "phi_offset": 0.23,
                "bias_current": 1.7,
            },
        ),
    ]
    sim = SimulationConfig(dt=float(dt), dt_learnable=bool(learnable), input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=[])


def test_dt_preserved_in_conversion() -> None:
    """Test that dt value is preserved when converting to JAX."""
    dt_value = 37.0
    torch_model = _build_single_dendrite_model(dim=3, dt=dt_value, learnable=False, source_func="Tanh")
    jax_model = convert_core_model_to_jax(torch_model)

    # DT should be preserved
    assert abs(jax_model.dt - dt_value) < 1e-6


def test_dt_learnable_flag_preserved() -> None:
    """Test that dt_learnable flag is preserved in conversion."""
    # Note: JAX models don't currently support learnable dt
    # This test verifies that the value is correctly set even if not used
    torch_model = _build_single_dendrite_model(dim=3, dt=37, learnable=True, source_func="Tanh")
    jax_model = convert_core_model_to_jax(torch_model)

    # Model should convert successfully
    assert jax_model is not None
    assert abs(jax_model.dt - 37.0) < 1e-6


def test_different_dt_values() -> None:
    """Test conversion with different dt values."""
    for dt in [10.0, 37.0, 100.0, 200.0]:
        torch_model = _build_single_dendrite_model(dim=2, dt=dt, learnable=False, source_func="Tanh")
        jax_model = convert_core_model_to_jax(torch_model)

        assert abs(jax_model.dt - dt) < 1e-6


def test_dt_affects_forward_output() -> None:
    """Test that different dt values produce different outputs."""
    # Build models with different dt
    torch_model1 = _build_single_dendrite_model(dim=2, dt=37, learnable=False, source_func="Tanh")
    torch_model2 = _build_single_dendrite_model(dim=2, dt=155.8, learnable=False, source_func="Tanh")

    jax_model1 = convert_core_model_to_jax(torch_model1)
    jax_model2 = convert_core_model_to_jax(torch_model2)

    # Same input
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 2)) * 3.0

    # Forward pass
    y1, _ = jax_model1(x)
    y2, _ = jax_model2(x)

    # Outputs should differ due to different dt
    max_diff = jnp.abs(y2 - y1).max()
    assert max_diff > 1e-6, f"Different dt values should produce different outputs (max diff: {max_diff})"
