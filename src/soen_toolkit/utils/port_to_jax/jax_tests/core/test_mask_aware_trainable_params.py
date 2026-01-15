import jax.numpy as jnp

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax


def _build_model_dense_3x4_learning() -> SOENModelCore:
    """Build a 3->4 dense inter-layer, learnable; no internal J."""
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": 3}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": 4}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "constant", "value": 0.1},
            learnable=True,
        ),
    ]
    sim = SimulationConfig(dt=1.0, input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def _build_model_one_to_one_5x3_learning() -> SOENModelCore:
    """Build a 5->3 one_to_one inter-layer, learnable; 3 ones on diagonal."""
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": 5}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": 3}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="one_to_one",
            params={"init": "constant", "value": 0.1},
            learnable=True,
        ),
    ]
    sim = SimulationConfig(dt=1.0, input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def _build_model_with_internal(dim: int = 4, learn_internal: bool = True) -> SOENModelCore:
    """Build a model with internal connection."""
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": dim}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": dim}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "constant", "value": 0.1},
            learnable=True,
        ),
        ConnectionConfig(
            from_layer=0,
            to_layer=0,
            connection_type="one_to_one",
            params={"init": "constant", "value": 0.2},
            learnable=learn_internal,
        ),
    ]
    sim = SimulationConfig(dt=1.0, input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def test_mask_aware_trainable_for_dense_matches_numel() -> None:
    """Test that dense connection has correct number of parameters."""
    torch_model = _build_model_dense_3x4_learning()
    jax_model = convert_core_model_to_jax(torch_model)

    # Dense 4x3 has 12 parameters
    J = jax_model.connections[0].J
    assert J.shape == (4, 3)
    assert jnp.size(J) == 12


def test_mask_aware_trainable_for_one_to_one_uses_mask_ones() -> None:
    """Test that one_to_one connection preserves mask structure."""
    torch_model = _build_model_one_to_one_5x3_learning()
    jax_model = convert_core_model_to_jax(torch_model)

    # One-to-one from 5 to 3 gives min(5,3)=3 diagonal connections
    J = jax_model.connections[0].J
    mask = jax_model.connections[0].mask

    # Full connection is 3x5 but mask should show only 3 active
    assert J.shape == (3, 5)
    if mask is not None:
        # Count non-zero mask values
        active_count = jnp.sum(mask > 0.5)
        assert active_count == 3


def test_mask_aware_internal_self_connection_counts_mask_ones() -> None:
    """Test that internal self-connection has correct structure."""
    torch_model = _build_model_with_internal(dim=4, learn_internal=True)
    jax_model = convert_core_model_to_jax(torch_model)

    # Should have two connections
    assert len(jax_model.connections) == 2

    # Find internal connection (from_layer == to_layer)
    internal_conn = None
    for conn in jax_model.connections:
        if conn.from_layer == conn.to_layer:
            internal_conn = conn
            break

    assert internal_conn is not None
    # Internal one_to_one on 4x4 contributes 4 diagonal connections
    J = internal_conn.J
    mask = internal_conn.mask
    assert J.shape == (4, 4)

    if mask is not None:
        active_count = jnp.sum(mask > 0.5)
        assert active_count == 4


def test_internal_mask_enforcement_during_training() -> None:
    """Test that internal connection masks are enforced during training."""
    import torch

    # Build model with internal connection and mask
    dim = 4
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": dim}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=0,
            connection_type="dense",
            params={"init": "constant", "value": 0.1},
            learnable=True,
        ),
    ]
    sim = SimulationConfig(dt=1.0, input_type="flux", track_phi=False, track_s=False, track_g=False)
    torch_model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

    # Create a mask that zeros out some entries
    internal_key = "J_0_to_0"
    if internal_key in torch_model.connection_masks:
        mask = torch_model.connection_masks[internal_key]
    else:
        mask = torch.ones(dim, dim)
        torch_model.connection_masks[internal_key] = mask

    # Zero out some entries (e.g., upper triangle)
    mask.data = torch.tril(torch.ones(dim, dim))
    torch_model.connection_masks[internal_key] = mask

    # Convert to JAX
    jax_model = convert_core_model_to_jax(torch_model)

    # Verify mask was extracted
    layer_spec = jax_model.layers[0]
    assert layer_spec.internal_J is not None
    assert layer_spec.internal_mask is not None
    assert layer_spec.internal_mask.shape == layer_spec.internal_J.shape

    # Count masked (zero) entries
    masked_count = jnp.sum(layer_spec.internal_mask == 0)
    assert masked_count > 0, "Test requires some masked entries"

    # Initialize trainer with dummy data
    from soen_toolkit.utils.port_to_jax.jax_training.trainer import DataConfigJAX, ExperimentConfigJAX, JaxTrainer, TrainingConfigJAX

    # Create dummy data loaders
    batch_size = 2
    seq_len = 5
    dummy_x = jnp.zeros((batch_size, seq_len, dim))
    dummy_y = jnp.zeros((batch_size, dim))

    class DummyLoader:
        def __iter__(self):
            yield dummy_x, dummy_y

    train_loader = DummyLoader()
    val_loader = DummyLoader()

    # Create temporary model file for trainer
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".soen", delete=False) as f:
        torch_model.save(f.name)
        model_path = f.name

    try:
        cfg = ExperimentConfigJAX(
            model_path=model_path,
            data=DataConfigJAX(train_loader=train_loader, val_loader=val_loader),
            training=TrainingConfigJAX(lr=0.01, max_epochs=1, seed=42),
        )

        trainer = JaxTrainer(cfg)

        # Get initial internal connection values
        initial_internal_J = trainer.params["internal_connections"].get(0)
        assert initial_internal_J is not None

        # Verify initial masked entries are zero
        mask = trainer._internal_masks.get(0)
        assert mask is not None
        masked_positions = mask == 0
        initial_masked_values = initial_internal_J[masked_positions]
        assert jnp.allclose(initial_masked_values, 0.0), "Masked entries should start at zero"

        # Run one training step
        params, opt_state, loss, metrics, grads = trainer.train_step(trainer.params, trainer.opt_state, dummy_x, dummy_y)

        # Get updated internal connection values
        updated_internal_J = params["internal_connections"].get(0)
        assert updated_internal_J is not None

        # Verify masked entries remain zero after training step
        updated_masked_values = updated_internal_J[masked_positions]
        assert jnp.allclose(updated_masked_values, 0.0, atol=1e-6), f"Masked entries should remain zero after training step. Found values: {updated_masked_values}"

        # Verify unmasked entries may have changed
        unmasked_positions = mask != 0
        initial_unmasked = initial_internal_J[unmasked_positions]
        updated_unmasked = updated_internal_J[unmasked_positions]
        # At least some unmasked entries should have changed (due to gradient updates)
        changes = jnp.abs(updated_unmasked - initial_unmasked)
        assert jnp.any(changes > 1e-6), "Some unmasked entries should have changed"

    finally:
        import os

        if os.path.exists(model_path):
            os.unlink(model_path)
