"""Comprehensive training pipeline tests for all task paradigms.
Tests the full training pipeline with YAML configs and synthetic datasets.
"""

import os
from pathlib import Path
import tempfile

import pytorch_lightning as pl
import torch

from soen_toolkit.tests.training.test_models import (
    build_autoencoder_model,
    build_classification_model,
    build_pulse_classification_model,
    build_regression_model,
    build_sequence_model,
)
from soen_toolkit.tests.utils.hdf5_test_helpers import (
    make_temp_hdf5_classification,
    make_temp_hdf5_regression,
)
from soen_toolkit.tests.utils.synthetic_datasets import (
    make_temp_hdf5_pulse_classification,
    make_temp_hdf5_seq2seq_classification,
    make_temp_hdf5_time_series_forecasting,
    make_temp_hdf5_unsupervised_seq2static,
)
from soen_toolkit.training.configs.experiment_config import load_config
from soen_toolkit.training.data.data_module import SOENDataModule
from soen_toolkit.training.models.lightning_wrapper import SOENLightningModule


class _TestingLightningWrapper(SOENLightningModule):
    """Custom wrapper for testing that bypasses file loading."""

    def __init__(self, config, core_model) -> None:
        pl.LightningModule.__init__(self)
        self.config = config
        self.model = core_model

        # Set up time pooling
        tp_cfg = self.config.model.time_pooling
        if isinstance(tp_cfg, dict):
            self.time_pooling_method_name = tp_cfg.get("name", "final")
            self.time_pooling_params = tp_cfg.get("params", {})
        else:
            self.time_pooling_method_name = tp_cfg
            self.time_pooling_params = {}
        if "scale" not in self.time_pooling_params:
            self.time_pooling_params["scale"] = 1.0

        # Set up other required attributes
        self.range_start = None
        self.range_end = None
        self.autoregressive = False
        self.latest_processed_state = None
        self.latest_final_state = None
        self.latest_all_states = None
        self._training_step_outputs = []  # Store training outputs for epoch-end metrics

        # Initialize loss functions
        self._initialize_loss_functions()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Override forward to handle seq2seq mapping correctly."""
        final_state, all_states = self.model(x)
        self.latest_final_state = final_state
        self.latest_all_states = all_states

        # Check mapping to determine output format
        mapping = str(getattr(self.config.training, "mapping", "seq2static")).lower()
        if mapping == "seq2seq":
            # For seq2seq, return full sequence (drop initial state)
            output = final_state[:, 1:, :]
            self.latest_processed_state = output
        elif self.autoregressive:
            # In autoregressive mode we expose the full time sequence from the core
            output = final_state
            self.latest_processed_state = output
        else:
            # For seq2static, use time pooling
            processed_state = self.process_output(final_state)
            self.latest_processed_state = processed_state
            output = processed_state

        return output, final_state, all_states


def create_test_config_override(base_config_path: str, overrides: dict) -> str:
    """Create a temporary config file with overrides applied to base config."""
    import yaml

    # Load base config
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Apply overrides
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    deep_update(config, overrides)

    # Save to temporary file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.close()

    return tmp.name


def test_supervised_seq2static_classification_training() -> None:
    """Test supervised sequence-to-static classification with full training pipeline."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_pulse_classification(
        num_samples=60,
        seq_len=16,
        feat_dim=1,
        num_classes=3,
        with_splits=True,
    ) as (h5_path, num_classes):
        # Create config with overrides
        overrides = {
            "training": {
                "paradigm": "supervised",
                "mapping": "seq2static",
                "losses": [{"name": "cross_entropy", "weight": 1.0, "params": {}}],
            },
            "data": {
                "data_path": h5_path,
                "num_classes": num_classes,
                "target_seq_len": 16,
            },
            "model": {
                "time_pooling": {"name": "max", "params": {"scale": 1.0}},
            },
        }

        config_path = create_test_config_override(str(base_config), overrides)

        try:
            # Load config and create components
            config = load_config(config_path, validate=False)  # Skip validation in tests
            model = build_pulse_classification_model(input_dim=1, hidden_dim=5, num_classes=num_classes)

            # Test data loading
            dm = SOENDataModule(config)
            dm.prepare_data()
            dm.setup()
            train_loader = dm.train_dataloader()

            # Test model wrapper
            wrapper = _TestingLightningWrapper(config, model)

            # Test single training step
            x, y = next(iter(train_loader))
            assert x.shape[-1] == 1  # feat_dim
            assert y.dtype == torch.long  # classification labels

            # Forward pass
            out = wrapper.forward(x)[0]
            assert out.shape[0] == x.shape[0]  # batch size
            assert out.shape[1] == num_classes  # output classes

            # Training step
            result = wrapper.training_step((x, y), batch_idx=0)
            loss = result["loss"]
            assert torch.isfinite(loss)

            # Test backpropagation
            loss.backward()
            grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)

        finally:
            os.unlink(config_path)


def test_supervised_seq2static_regression_training() -> None:
    """Test supervised sequence-to-static regression with full training pipeline."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_regression(
        num_samples=50,
        seq_len=20,
        feat_dim=6,
        target_dim=3,
        target_sequence=False,
        with_splits=True,
    ) as (h5_path, target_dim):
        overrides = {
            "training": {
                "paradigm": "supervised",
                "mapping": "seq2static",
                "losses": [{"name": "mse", "weight": 1.0, "params": {}}],
            },
            "data": {
                "data_path": h5_path,
                "target_seq_len": 20,
            },
            "model": {
                "time_pooling": {"name": "mean", "params": {"scale": 1.0}},
            },
        }

        config_path = create_test_config_override(str(base_config), overrides)

        try:
            config = load_config(config_path, validate=False)  # Skip validation in tests
            model = build_regression_model(input_dim=6, hidden_dim=8, output_dim=target_dim)

            dm = SOENDataModule(config)
            dm.prepare_data()
            dm.setup()

            wrapper = _TestingLightningWrapper(config, model)

            x, y = next(iter(dm.train_dataloader()))
            assert y.dtype.is_floating_point
            assert y.ndim == 2  # [N, target_dim]

            result = wrapper.training_step((x, y), batch_idx=0)
            loss = result["loss"]
            assert torch.isfinite(loss)

            loss.backward()
            grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)

        finally:
            os.unlink(config_path)


def test_supervised_seq2seq_classification_training() -> None:
    """Test supervised sequence-to-sequence classification."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_seq2seq_classification(
        num_samples=40,
        seq_len=12,
        feat_dim=4,
        num_classes=3,
        with_splits=True,
    ) as (h5_path, num_classes):
        overrides = {
            "training": {
                "paradigm": "supervised",
                "mapping": "seq2seq",
                "losses": [{"name": "cross_entropy", "weight": 1.0, "params": {}}],
            },
            "data": {
                "data_path": h5_path,
                "num_classes": num_classes,
                "target_seq_len": 12,
            },
            "model": {
                "time_pooling": {"name": "final", "params": {"scale": 1.0}},
            },
        }

        config_path = create_test_config_override(str(base_config), overrides)

        try:
            config = load_config(config_path, validate=False)  # Skip validation in tests
            model = build_sequence_model(input_dim=4, hidden_dim=8, output_dim=num_classes)

            dm = SOENDataModule(config)
            dm.prepare_data()
            dm.setup()

            wrapper = _TestingLightningWrapper(config, model)

            x, y = next(iter(dm.train_dataloader()))
            assert y.ndim == 2  # [N, T] for seq2seq classification
            assert y.dtype == torch.long

            result = wrapper.training_step((x, y), batch_idx=0)
            loss = result["loss"]
            assert torch.isfinite(loss)

            loss.backward()
            grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)

        finally:
            os.unlink(config_path)


def test_supervised_seq2seq_regression_training() -> None:
    """Test supervised sequence-to-sequence regression (time series forecasting)."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_time_series_forecasting(
        num_samples=60,
        seq_len=16,
        feat_dim=3,
        forecast_len=8,
        with_splits=True,
    ) as (h5_path, feat_dim):
        overrides = {
            "training": {
                "paradigm": "supervised",
                "mapping": "seq2seq",
                "losses": [{"name": "mse", "weight": 1.0, "params": {}}],
            },
            "data": {
                "data_path": h5_path,
                "target_seq_len": 8,  # Match the forecast length
            },
            "model": {
                "time_pooling": {"name": "final", "params": {"scale": 1.0}},
            },
        }

        config_path = create_test_config_override(str(base_config), overrides)

        try:
            config = load_config(config_path, validate=False)  # Skip validation in tests
            model = build_sequence_model(input_dim=feat_dim, hidden_dim=10, output_dim=feat_dim)

            dm = SOENDataModule(config)
            dm.prepare_data()
            dm.setup()

            wrapper = _TestingLightningWrapper(config, model)

            x, y = next(iter(dm.train_dataloader()))
            assert y.ndim == 3  # [N, T, D] for seq2seq regression
            assert y.dtype.is_floating_point

            result = wrapper.training_step((x, y), batch_idx=0)
            loss = result["loss"]
            assert torch.isfinite(loss)

            loss.backward()
            grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)

        finally:
            os.unlink(config_path)


def test_unsupervised_seq2seq_reconstruction_training() -> None:
    """Test self-supervised sequence-to-sequence reconstruction (uses paradigm='unsupervised' in config)."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_regression(
        num_samples=50,
        seq_len=14,
        feat_dim=5,
        target_dim=5,
        target_sequence=True,
        with_splits=False,
    ) as (h5_path, feat_dim):
        overrides = {
            "training": {
                "paradigm": "self_supervised",
                "mapping": "seq2seq",
                "losses": [{"name": "mse", "weight": 1.0, "params": {}}],
            },
            "data": {
                "data_path": h5_path,
                "target_seq_len": 14,
            },
            "model": {
                "time_pooling": {"name": "final", "params": {"scale": 1.0}},
            },
        }

        config_path = create_test_config_override(str(base_config), overrides)

        try:
            config = load_config(config_path, validate=False)  # Skip validation in tests
            model = build_autoencoder_model(input_dim=feat_dim, hidden_dim=3)

            dm = SOENDataModule(config)
            dm.prepare_data()
            dm.setup()

            wrapper = _TestingLightningWrapper(config, model)

            x, y = next(iter(dm.train_dataloader()))
            # In unsupervised learning, targets should be ignored and inputs used instead

            result = wrapper.training_step((x, y), batch_idx=0)
            loss = result["loss"]
            assert torch.isfinite(loss)

            loss.backward()
            grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)

        finally:
            os.unlink(config_path)


def test_unsupervised_seq2static_training() -> None:
    """Test self-supervised sequence-to-static learning (uses paradigm='unsupervised' in config)."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_unsupervised_seq2static(
        num_samples=45,
        seq_len=18,
        feat_dim=6,
        target_dim=4,
        with_splits=True,
    ) as (h5_path, target_dim):
        overrides = {
            "training": {
                "paradigm": "self_supervised",
                "mapping": "seq2static",
                "losses": [{"name": "mse", "weight": 1.0, "params": {}}],
            },
            "data": {
                "data_path": h5_path,
                "target_seq_len": 18,
            },
            "model": {
                "time_pooling": {"name": "mean", "params": {"scale": 1.0}},
            },
        }

        config_path = create_test_config_override(str(base_config), overrides)

        try:
            config = load_config(config_path, validate=False)  # Skip validation in tests
            model = build_regression_model(input_dim=6, hidden_dim=8, output_dim=target_dim)

            dm = SOENDataModule(config)
            dm.prepare_data()
            dm.setup()

            wrapper = _TestingLightningWrapper(config, model)

            x, y = next(iter(dm.train_dataloader()))
            assert y.dtype.is_floating_point
            assert y.ndim == 2  # [N, target_dim]

            result = wrapper.training_step((x, y), batch_idx=0)
            loss = result["loss"]
            assert torch.isfinite(loss)

            loss.backward()
            grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)

        finally:
            os.unlink(config_path)


def test_multiple_loss_functions_training() -> None:
    """Test training with multiple weighted loss functions."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_classification(
        num_samples=40,
        seq_len=12,
        feat_dim=5,
        num_classes=4,
        with_splits=True,
    ) as (h5_path, num_classes):
        overrides = {
            "training": {
                "paradigm": "supervised",
                "mapping": "seq2static",
                "losses": [
                    {"name": "cross_entropy", "weight": 1.0, "params": {}},
                    {"name": "gap_loss", "weight": 0.3, "params": {"margin": 0.2}},
                ],
            },
            "data": {
                "data_path": h5_path,
                "num_classes": num_classes,
                "target_seq_len": 12,
            },
            "model": {
                "time_pooling": {"name": "final", "params": {"scale": 1.0}},
            },
        }

        config_path = create_test_config_override(str(base_config), overrides)

        try:
            config = load_config(config_path, validate=False)  # Skip validation in tests
            model = build_classification_model(input_dim=5, hidden_dim=8, num_classes=num_classes)

            dm = SOENDataModule(config)
            dm.prepare_data()
            dm.setup()

            wrapper = _TestingLightningWrapper(config, model)

            x, y = next(iter(dm.train_dataloader()))

            result = wrapper.training_step((x, y), batch_idx=0)
            loss = result["loss"]
            assert torch.isfinite(loss)

            # Multiple losses should be working (we can't easily test the internal structure)

            loss.backward()
            grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)

        finally:
            os.unlink(config_path)


def test_different_pooling_methods() -> None:
    """Test different time pooling methods work correctly."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    pooling_methods = [
        {"name": "final", "params": {"scale": 1.0}},
        {"name": "mean", "params": {"scale": 1.0}},
        {"name": "max", "params": {"scale": 1.0}},
        {"name": "mean_last_n", "params": {"n": 3, "scale": 1.0}},
    ]

    with make_temp_hdf5_classification(
        num_samples=30,
        seq_len=10,
        feat_dim=4,
        num_classes=3,
        with_splits=False,
    ) as (h5_path, num_classes):
        for pooling in pooling_methods:
            overrides = {
                "training": {
                    "paradigm": "supervised",
                    "mapping": "seq2static",
                    "losses": [{"name": "cross_entropy", "weight": 1.0, "params": {}}],
                },
                "data": {
                    "data_path": h5_path,
                    "num_classes": num_classes,
                    "target_seq_len": 10,
                },
                "model": {
                    "time_pooling": pooling,
                },
            }

            config_path = create_test_config_override(str(base_config), overrides)

            try:
                config = load_config(config_path, validate=False)  # Skip validation in tests
                model = build_classification_model(input_dim=4, hidden_dim=6, num_classes=num_classes)

                dm = SOENDataModule(config)
                dm.prepare_data()
                dm.setup()

                wrapper = _TestingLightningWrapper(config, model)

                x, y = next(iter(dm.train_dataloader()))
                result = wrapper.training_step((x, y), batch_idx=0)
                loss = result["loss"]
                assert torch.isfinite(loss), f"Loss not finite for pooling {pooling['name']}"

            finally:
                os.unlink(config_path)


if __name__ == "__main__":
    # Run a quick test
    test_supervised_seq2static_classification_training()
