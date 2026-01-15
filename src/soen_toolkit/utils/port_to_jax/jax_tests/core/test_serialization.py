import os
import tempfile

from soen_toolkit.core import SOENModelCore

from ...test_helpers import build_small_model_jax, numpy_allclose


def test_save_and_load_round_trip_preserves_connections_and_configs() -> None:
    """Test that converted JAX model reflects saved/loaded PyTorch model."""
    torch_model = None
    jax_model = None

    # Build model
    layers_torch, jax_m = build_small_model_jax(dims=(3, 4), connectivity_type="dense", init="constant", init_value=0.33, with_internal_first=True)
    torch_model = layers_torch
    jax_model = jax_m

    # Save to temp file
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model_test.soen")
        torch_model.save(path)
        assert os.path.isfile(path)

        # Load back
        m2 = SOENModelCore.load(path, show_logs=False)

        # Convert both to JAX
        from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax

        jax_model2 = convert_core_model_to_jax(m2)

        # Check that connection shapes match
        assert len(jax_model.connections) == len(jax_model2.connections)

        # Compare connection weights (allowing for small numerical differences)
        for c1, c2 in zip(jax_model.connections, jax_model2.connections, strict=False):
            assert c1.J.shape == c2.J.shape
            assert numpy_allclose(c1.J, c2.J, atol=1e-5)

            if c1.mask is not None and c2.mask is not None:
                assert numpy_allclose(c1.mask, c2.mask, atol=1e-5)


def test_layer_structure_preserved() -> None:
    """Test that layer structure is preserved in conversion."""
    torch_model, jax_model = build_small_model_jax(dims=(3, 4), connectivity_type="dense", init="constant", init_value=0.5)

    # Check layer count
    assert len(jax_model.layers) == len(torch_model.layers_config)

    # Check dimensions match
    for layer_spec in jax_model.layers:
        # Find corresponding PyTorch layer
        torch_layer = None
        for cfg in torch_model.layers_config:
            if cfg.layer_id == layer_spec.layer_id:
                torch_layer = cfg
                break

        assert torch_layer is not None, f"Layer {layer_spec.layer_id} not found in PyTorch model"
        assert layer_spec.dim == torch_layer.params.get("dim")
