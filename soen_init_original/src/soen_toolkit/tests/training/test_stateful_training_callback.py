"""Tests for stateful training callback."""

import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.training.callbacks.stateful_training import StatefulTrainingCallback


class DummyTrainer:
    """Minimal trainer for testing callbacks."""

    def __init__(self) -> None:
        self.current_epoch = 0
        self.global_step = 0


class DummyPLModule:
    """Minimal PyTorch Lightning module for testing."""

    def __init__(self, model: SOENModelCore) -> None:
        self.model = model
        self.latest_final_state = None
        self.latest_all_states = None
        self._pending_initial_states = None
        self._pending_s1_states = None
        self._pending_s2_states = None


def build_test_model(dim: int = 4, num_layers: int = 2, use_multiplier_v2: bool = False):
    """Build a simple test model."""
    if use_multiplier_v2:
        layers = [
            LayerConfig(layer_id=0, layer_type="RNN", params={"dim": dim}),
            LayerConfig(layer_id=1, layer_type="MultiplierNOCC", params={"dim": dim}),
        ]
    else:
        layers = [
            LayerConfig(layer_id=i, layer_type="RNN", params={"dim": dim})
            for i in range(num_layers)
        ]

    conns = [
        ConnectionConfig(from_layer=i, to_layer=i+1, connection_type="dense", params={}, learnable=True)
        for i in range(num_layers - 1)
    ]

    sim = SimulationConfig(dt=0.1, input_type="flux")
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def run_forward_pass(pl_module: DummyPLModule, batch_size: int = 2, seq_len: int = 5):
    """Simulate a forward pass and store results in pl_module."""
    dim = pl_module.model.layer_nodes[0]
    x = torch.randn(batch_size, seq_len, dim)

    # Check if pending initial states are set
    initial_states = getattr(pl_module, '_pending_initial_states', None)
    s1_states = getattr(pl_module, '_pending_s1_states', None)
    s2_states = getattr(pl_module, '_pending_s2_states', None)

    # Call model forward
    if initial_states is not None or s1_states is not None or s2_states is not None:
        final_state, all_states = pl_module.model(
            x,
            initial_states=initial_states,
            s1_inits=s1_states,
            s2_inits=s2_states
        )
        # Clear after use (as the real forward does)
        pl_module._pending_initial_states = None
        pl_module._pending_s1_states = None
        pl_module._pending_s2_states = None
    else:
        final_state, all_states = pl_module.model(x)

    # Store results
    pl_module.latest_final_state = final_state
    pl_module.latest_all_states = all_states

    return final_state, all_states


def test_stateful_training_basic_carryover():
    """Test basic state carryover in training mode."""
    model = build_test_model(dim=4, num_layers=2)
    pl_module = DummyPLModule(model)
    trainer = DummyTrainer()

    callback = StatefulTrainingCallback(
        enable_for_training=True,
        enable_for_validation=False,
        sample_selection="first",
        verbose=False
    )

    # Start epoch
    callback.on_train_epoch_start(trainer, pl_module)

    # First batch - no initial states should be set
    batch = (torch.randn(2, 5, 4), torch.randn(2, 1))
    callback.on_train_batch_start(trainer, pl_module, batch, batch_idx=0)
    assert pl_module._pending_initial_states is None

    # Run forward pass
    run_forward_pass(pl_module, batch_size=2, seq_len=5)

    # Extract and store states
    callback.on_train_batch_end(trainer, pl_module, None, batch, batch_idx=0)

    # Verify states were stored
    assert callback._train_states is not None
    assert len(callback._train_states) == 2  # 2 layers
    assert 0 in callback._train_states
    assert 1 in callback._train_states

    # Second batch - states should be injected
    callback.on_train_batch_start(trainer, pl_module, batch, batch_idx=1)
    assert pl_module._pending_initial_states is not None
    assert len(pl_module._pending_initial_states) == 2

    # Verify state shapes are correct (should be [dim])
    for layer_id, state in pl_module._pending_initial_states.items():
        assert state.shape == (4,)


def test_stateful_training_epoch_reset():
    """Test that states reset at epoch boundaries."""
    model = build_test_model(dim=4, num_layers=2)
    pl_module = DummyPLModule(model)
    trainer = DummyTrainer()

    callback = StatefulTrainingCallback(
        enable_for_training=True,
        sample_selection="first"
    )

    # Epoch 0
    callback.on_train_epoch_start(trainer, pl_module)
    batch = (torch.randn(2, 5, 4), torch.randn(2, 1))

    # First batch
    callback.on_train_batch_start(trainer, pl_module, batch, batch_idx=0)
    run_forward_pass(pl_module, batch_size=2, seq_len=5)
    callback.on_train_batch_end(trainer, pl_module, None, batch, batch_idx=0)

    # Verify states are stored
    assert callback._train_states is not None

    # New epoch - states should be cleared
    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, pl_module)
    assert callback._train_states is None
    assert callback._train_s1_states is None
    assert callback._train_s2_states is None


def test_stateful_training_multiplier_v2_states():
    """Test s1/s2 state carryover for MultiplierV2 layers."""
    model = build_test_model(dim=4, num_layers=2, use_multiplier_v2=True)
    pl_module = DummyPLModule(model)
    trainer = DummyTrainer()

    callback = StatefulTrainingCallback(
        enable_for_training=True,
        sample_selection="first"
    )

    # Start epoch
    callback.on_train_epoch_start(trainer, pl_module)
    batch = (torch.randn(2, 5, 4), torch.randn(2, 1))

    # First batch
    callback.on_train_batch_start(trainer, pl_module, batch, batch_idx=0)
    run_forward_pass(pl_module, batch_size=2, seq_len=5)

    # Extract states
    callback.on_train_batch_end(trainer, pl_module, None, batch, batch_idx=0)

    # MultiplierV2 layer should have s1/s2 states stored
    # Note: The callback extracts these from layer._s1_final and layer._s2_final
    # which are set during the MultiplierV2 forward pass
    # Layer 1 is MultiplierV2
    multiplier_layer = pl_module.model.layers[1]
    if hasattr(multiplier_layer, '_s1_final') and multiplier_layer._s1_final is not None:
        assert callback._train_s1_states is not None
        assert callback._train_s2_states is not None
        assert 1 in callback._train_s1_states  # layer_id 1
        assert 1 in callback._train_s2_states


def test_stateful_training_validation_mode():
    """Test state carryover in validation mode."""
    model = build_test_model(dim=4, num_layers=2)
    pl_module = DummyPLModule(model)
    trainer = DummyTrainer()

    callback = StatefulTrainingCallback(
        enable_for_training=False,
        enable_for_validation=True,
        sample_selection="first"
    )

    # Start validation epoch
    callback.on_validation_epoch_start(trainer, pl_module)
    batch = (torch.randn(2, 5, 4), torch.randn(2, 1))

    # First validation batch
    callback.on_validation_batch_start(trainer, pl_module, batch, batch_idx=0)
    assert pl_module._pending_initial_states is None  # First batch has no states

    run_forward_pass(pl_module, batch_size=2, seq_len=5)
    callback.on_validation_batch_end(trainer, pl_module, None, batch, batch_idx=0)

    # Verify validation states were stored
    assert callback._val_states is not None
    assert callback._train_states is None  # Training states should remain None

    # Second validation batch
    callback.on_validation_batch_start(trainer, pl_module, batch, batch_idx=1)
    assert pl_module._pending_initial_states is not None


def test_stateful_training_sample_selection_modes():
    """Test different sample selection modes."""
    batch_size = 5

    # Test "first"
    callback_first = StatefulTrainingCallback(sample_selection="first")
    idx = callback_first._pick_sample_index(batch_size)
    assert idx == 0

    # Test "last"
    callback_last = StatefulTrainingCallback(sample_selection="last")
    idx = callback_last._pick_sample_index(batch_size)
    assert idx == batch_size - 1

    # Test "random"
    callback_random = StatefulTrainingCallback(sample_selection="random")
    idx = callback_random._pick_sample_index(batch_size)
    assert 0 <= idx < batch_size


def test_stateful_training_state_extraction():
    """Test that extracted states have correct shapes and are detached."""
    model = build_test_model(dim=4, num_layers=2)
    pl_module = DummyPLModule(model)
    trainer = DummyTrainer()

    callback = StatefulTrainingCallback(
        enable_for_training=True,
        sample_selection="first"
    )

    callback.on_train_epoch_start(trainer, pl_module)
    batch = (torch.randn(2, 5, 4), torch.randn(2, 1))

    callback.on_train_batch_start(trainer, pl_module, batch, batch_idx=0)
    run_forward_pass(pl_module, batch_size=2, seq_len=5)
    callback.on_train_batch_end(trainer, pl_module, None, batch, batch_idx=0)

    # Check extracted states
    assert callback._train_states is not None
    for layer_id, state in callback._train_states.items():
        # State should be 1D tensor [dim]
        assert state.ndim == 1
        assert state.shape[0] == 4  # dim=4
        # State should not require grad (detached)
        assert not state.requires_grad


def test_stateful_training_invalid_sample_selection():
    """Test that invalid sample_selection raises an error."""
    try:
        StatefulTrainingCallback(sample_selection="invalid")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "sample_selection must be" in str(e)


def test_stateful_training_disabled_by_default():
    """Test that the callback does nothing when both modes are disabled."""
    model = build_test_model(dim=4, num_layers=2)
    pl_module = DummyPLModule(model)
    trainer = DummyTrainer()

    callback = StatefulTrainingCallback(
        enable_for_training=False,
        enable_for_validation=False
    )

    batch = (torch.randn(2, 5, 4), torch.randn(2, 1))

    # Training batch
    callback.on_train_batch_start(trainer, pl_module, batch, batch_idx=0)
    assert pl_module._pending_initial_states is None

    run_forward_pass(pl_module, batch_size=2, seq_len=5)
    callback.on_train_batch_end(trainer, pl_module, None, batch, batch_idx=0)

    # States should not be stored
    assert callback._train_states is None

    # Validation batch
    callback.on_validation_batch_start(trainer, pl_module, batch, batch_idx=0)
    assert pl_module._pending_initial_states is None

    run_forward_pass(pl_module, batch_size=2, seq_len=5)
    callback.on_validation_batch_end(trainer, pl_module, None, batch, batch_idx=0)

    assert callback._val_states is None

