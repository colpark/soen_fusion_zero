"""Test JAX gradient clipping functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.tests.utils.synthetic_datasets import (
    make_temp_hdf5_pulse_classification,
)
from soen_toolkit.training.configs.config_classes import LossItemConfig
from soen_toolkit.training.configs.experiment_config import load_config
from soen_toolkit.training.trainers.experiment import ExperimentRunner


def _build_test_core(input_dim: int, hidden_dim: int, output_dim: int) -> SOENModelCore:
    """Build a simple test model."""
    sim = SimulationConfig(network_evaluation_method="layerwise", input_type="state", dt=1.0)
    layers = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": input_dim}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": hidden_dim}),
        LayerConfig(layer_id=2, layer_type="Linear", params={"dim": output_dim}),
    ]
    connections = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"structure": {"type": "dense"}},
            learnable=True,
        ),
        ConnectionConfig(
            from_layer=1,
            to_layer=2,
            connection_type="dense",
            params={"structure": {"type": "dense"}},
            learnable=True,
        ),
    ]
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=connections)


@pytest.mark.parametrize(
    "clip_val,algorithm",
    [
        (None, "norm"),  # No clipping
        (1.0, "norm"),  # Global norm clipping
        (0.5, "value"),  # Element-wise clipping
    ],
)
def test_jax_gradient_clipping(tmp_path, clip_val, algorithm) -> None:
    """Test JAX training with different gradient clipping configurations."""
    pytest.importorskip("jax")

    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"
    config = load_config(base_config, validate=False)

    with make_temp_hdf5_pulse_classification(
        num_samples=24,
        seq_len=12,
        feat_dim=1,
        num_classes=3,
        with_splits=True,
    ) as (h5_path, num_classes):
        # Configure for JAX backend
        config.model.backend = "jax"
        core = _build_test_core(input_dim=1, hidden_dim=4, output_dim=num_classes)
        model_path = tmp_path / "base_model.soen"
        core.save(str(model_path))
        config.model.base_model_path = model_path
        config.model.load_exact_model_state = True

        # Minimal training config
        config.training.max_epochs = 1
        config.training.num_repeats = 1
        config.training.batch_size = 8
        config.training.mapping = "seq2static"
        config.training.loss.losses = [LossItemConfig(name="cross_entropy", weight=1.0, params={})]

        # Set gradient clipping
        config.training.gradient_clip_val = clip_val
        config.training.gradient_clip_algorithm = algorithm

        config.data.data_path = h5_path
        config.data.num_classes = num_classes
        config.data.target_seq_len = 12

        config.logging.project_dir = tmp_path / "runs"
        config.logging.upload_logs_and_checkpoints = False
        config.logging.mlflow_active = False
        config.logging.log_freq = 1

        config.callbacks = {}

        # Run training
        runner = ExperimentRunner(config, script_dir=tmp_path, project_root_dir=tmp_path)
        best_ckpt = runner.run_single_repeat(repeat=0, seed=config.seed)

        # Verify checkpoint was created
        assert isinstance(best_ckpt, str)
        assert Path(best_ckpt).exists()
