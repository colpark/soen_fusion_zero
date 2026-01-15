import jax.numpy as jnp

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax
from soen_toolkit.utils.port_to_jax.jax_model import JAXModel, LayerSpec
from soen_toolkit.utils.port_to_jax.jax_tests.test_helpers import build_small_model_jax, make_random_series_jax
from soen_toolkit.utils.port_to_jax.pure_forward import build_topology, convert_params_to_arrays


def test_connectivity_mask_application_preserves_zeros() -> None:
    """Test that masked zeros are preserved after forward pass."""
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": 4}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": 5}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="one_to_one",
            params={"init": "constant", "value": 1.0},
        ),
    ]
    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)

    torch_model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
    jax_model = convert_core_model_to_jax(torch_model)

    # Weight should be masked to diagonal
    w = jax_model.connections[0].J
    mask = jax_model.connections[0].mask
    assert w.shape == (5, 4)

    # If mask exists, off-diagonal should be zero
    if mask is not None:
        w_masked = w * mask
        # Check off-diagonal positions
        for i in range(5):
            for j in range(4):
                if i != j or mask[i, j] == 0:
                    assert jnp.allclose(w_masked[i, j], 0.0, atol=1e-6)

    # After forward pass, zeros should still be preserved
    x = make_random_series_jax(batch=2, seq_len=3, dim=4, seed=1)
    _ = jax_model(x)

    # Weights should not have changed (JAX model is immutable)
    w2 = jax_model.connections[0].J
    assert jnp.allclose(w, w2)


def test_connection_constraints_clamp_after_forward() -> None:
    """Test that constraints are enforced on connection weights."""
    # Set tight constraints
    constraints = {"min": -0.1, "max": 0.1}
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": 3}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": 3}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "normal", "value": 0.5, "constraints": constraints},
        ),
    ]
    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)

    torch_model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
    jax_model = convert_core_model_to_jax(torch_model)

    # Run forward pass
    x = make_random_series_jax(batch=1, seq_len=2, dim=3, seed=2)
    _ = jax_model(x)

    # Check that weights are within constraints
    w = jax_model.connections[0].J
    # Note: JAX models don't mutate weights during forward, so we check the original constraints
    # In practice, constraints are enforced on the PyTorch model before conversion
    assert jnp.all(w <= 0.1 + 1e-5)
    assert jnp.all(w >= -0.1 - 1e-5)


def test_jax_model_connection_constraints_propagate_and_clamp() -> None:
    """Ensure connection constraints propagate to the JAX model and clamp weights."""
    constraints = {"min": -0.05, "max": 0.05}
    _torch_model, jax_model = build_small_model_jax(init="constant", init_value=0.5, constraints=constraints)

    key = "J_0_to_1"
    assert jax_model.connection_constraints is not None
    assert key in jax_model.connection_constraints
    assert jax_model.connection_constraints[key]["max"] == constraints["max"]

    conn = jax_model.connections[0]
    clamped = jax_model._clamp_connections(conn)
    assert jnp.all(clamped <= constraints["max"] + 1e-6)
    assert jnp.all(clamped >= constraints["min"] - 1e-6)


def test_convert_params_to_arrays_respects_constraints() -> None:
    """Verify convert_params_to_arrays applies connection constraints to padded arrays."""
    constraints = {"min": -0.05, "max": 0.05}
    _torch_model, jax_model = build_small_model_jax(init="constant", init_value=0.5, constraints=constraints)
    topology = build_topology(jax_model)

    key = "J_0_to_1"
    assert topology.connection_constraints is not None
    assert key in topology.connection_constraints

    param_tree = {
        "layers": {spec.layer_id: spec.params for spec in jax_model.layers},
        "connections": {(conn.from_layer, conn.to_layer): jnp.full(conn.J.shape, 0.5) for conn in jax_model.connections},
        "internal_connections": {},
    }
    _, conn_params = convert_params_to_arrays(param_tree, topology, connection_constraints=topology.connection_constraints)
    assert conn_params.shape[0] == len(jax_model.connections)
    assert jnp.all(conn_params <= constraints["max"] + 1e-6)
    assert jnp.all(conn_params >= constraints["min"] - 1e-6)


def test_layer_creation_dims_match() -> None:
    """Test that layer dimensions are correctly set."""
    d0, d1 = 5, 7
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": d0}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": d1}),
    ]
    conns = [
        ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={"init": "constant", "value": 0.1}),
    ]
    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)

    torch_model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
    jax_model = convert_core_model_to_jax(torch_model)

    # Check dimensions
    assert jax_model.layers[0].dim == d0
    assert jax_model.layers[1].dim == d1


def test_input_shape_matching() -> None:
    """Test that input shape matches first layer dimension."""
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": 4}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": 5}),
    ]
    conns = [
        ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={"init": "constant", "value": 0.1}),
    ]
    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)

    torch_model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
    jax_model = convert_core_model_to_jax(torch_model)

    # Test correct input shape
    x = make_random_series_jax(batch=2, seq_len=5, dim=4, seed=3)
    y, _ = jax_model(x)

    # Output should have shape [batch, seq_len+1, dim_last_layer]
    assert y.shape == (2, 6, 5)


def test_layer_param_constraints_multiplier_bias_current() -> None:
    """Multiplier layer parameters are clamped when applying optimizer arrays."""
    dim = 3
    layer_spec = LayerSpec(
        layer_id=0,
        kind="Multiplier",
        dim=dim,
        params={
            "phi_y": jnp.zeros(dim, dtype=jnp.float32),
            "bias_current": jnp.array([-1.0, 2.0, -0.5], dtype=jnp.float32),
            "gamma_plus": jnp.ones(dim, dtype=jnp.float32) * 0.1,
            "gamma_minus": jnp.ones(dim, dtype=jnp.float32) * 0.1,
        },
        internal_J=None,
    )
    jax_model = JAXModel(dt=1.0, layers=[layer_spec], connections=[])
    arr = jnp.stack(
        [
            layer_spec.params["phi_y"],
            layer_spec.params["bias_current"],
            layer_spec.params["gamma_plus"],
            layer_spec.params["gamma_minus"],
        ],
        axis=0,
    )
    clamped_arrays = jax_model.clamp_and_apply_layer_param_arrays((arr,))
    assert jnp.all(clamped_arrays[0][1] >= 0.0)
    params = jax_model._layer_params(layer_spec, batch=1)
    assert jnp.all(params.bias_current >= 0.0)


def test_layer_param_constraints_single_dendrite_bias_current() -> None:
    """SingleDendrite layer arrays obey constraints."""
    dim = 2
    layer_spec = LayerSpec(
        layer_id=0,
        kind="SingleDendrite",
        dim=dim,
        params={
            "phi_offset": jnp.zeros(dim, dtype=jnp.float32),
            "bias_current": jnp.array([-0.75, -0.25], dtype=jnp.float32),
            "gamma_plus": jnp.ones(dim, dtype=jnp.float32) * 0.1,
            "gamma_minus": jnp.ones(dim, dtype=jnp.float32) * 0.1,
        },
        internal_J=None,
    )
    jax_model = JAXModel(dt=1.0, layers=[layer_spec], connections=[])
    arr = jnp.stack(
        [
            layer_spec.params["phi_offset"],
            layer_spec.params["bias_current"],
            layer_spec.params["gamma_plus"],
            layer_spec.params["gamma_minus"],
        ],
        axis=0,
    )
    clamped_arrays = jax_model.clamp_and_apply_layer_param_arrays((arr,))
    assert jnp.all(clamped_arrays[0][1] >= 0.0)
    params = jax_model._layer_params(layer_spec, batch=1)
    assert jnp.all(params.bias_current >= 0.0)
