"""Test backward compatibility for paradigm terminology.
Ensures both 'unsupervised' and 'self_supervised' work correctly.
"""

from pathlib import Path

import torch

from soen_toolkit.tests.training.test_comprehensive_training_pipeline import (
    _TestingLightningWrapper,
    create_test_config_override,
)
from soen_toolkit.tests.training.test_models import build_autoencoder_model
from soen_toolkit.tests.utils.hdf5_test_helpers import make_temp_hdf5_regression
from soen_toolkit.training.configs.experiment_config import load_config


def test_backward_compatibility_unsupervised_paradigm() -> None:
    """Test that 'unsupervised' paradigm still works for backward compatibility."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_regression(
        num_samples=30,
        seq_len=10,
        feat_dim=4,
        target_dim=4,
        target_sequence=True,
        with_splits=False,
    ) as (h5_path, feat_dim):
        # Test with old 'unsupervised' paradigm
        overrides = {
            "training": {
                "paradigm": "unsupervised",  # Old terminology
                "mapping": "seq2seq",
                "losses": [{"name": "mse", "weight": 1.0, "params": {}}],
            },
            "data": {
                "data_path": h5_path,
                "target_seq_len": 10,
            },
            "model": {
                "time_pooling": {"name": "final", "params": {"scale": 1.0}},
            },
        }

        config_path = create_test_config_override(str(base_config), overrides)

        try:
            config = load_config(config_path)
            model = build_autoencoder_model(input_dim=feat_dim, hidden_dim=3)
            wrapper = _TestingLightningWrapper(config, model)

            # Create dummy data
            x = torch.randn(4, 10, feat_dim)
            y = torch.randn(4, 10, feat_dim)  # Will be ignored in unsupervised mode

            # Test that training step works
            result = wrapper.training_step((x, y), batch_idx=0)
            loss = result["loss"]
            assert torch.isfinite(loss)

            # Test backpropagation
            loss.backward()
            grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)

        finally:
            import os

            os.unlink(config_path)


def test_new_self_supervised_paradigm() -> None:
    """Test that 'self_supervised' paradigm works correctly."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_regression(
        num_samples=30,
        seq_len=10,
        feat_dim=4,
        target_dim=4,
        target_sequence=True,
        with_splits=False,
    ) as (h5_path, feat_dim):
        # Test with new 'self_supervised' paradigm
        overrides = {
            "training": {
                "paradigm": "self_supervised",  # New correct terminology
                "mapping": "seq2seq",
                "losses": [{"name": "mse", "weight": 1.0, "params": {}}],
            },
            "data": {
                "data_path": h5_path,
                "target_seq_len": 10,
            },
            "model": {
                "time_pooling": {"name": "final", "params": {"scale": 1.0}},
            },
        }

        config_path = create_test_config_override(str(base_config), overrides)

        try:
            config = load_config(config_path)
            model = build_autoencoder_model(input_dim=feat_dim, hidden_dim=3)
            wrapper = _TestingLightningWrapper(config, model)

            # Create dummy data
            x = torch.randn(4, 10, feat_dim)
            y = torch.randn(4, 10, feat_dim)  # Will be ignored in self_supervised mode

            # Test that training step works
            result = wrapper.training_step((x, y), batch_idx=0)
            loss = result["loss"]
            assert torch.isfinite(loss)

            # Test backpropagation
            loss.backward()
            grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)

        finally:
            import os

            os.unlink(config_path)


def test_both_paradigms_equivalent() -> None:
    """Test that 'unsupervised' and 'self_supervised' produce equivalent results."""
    base_config = Path(__file__).parent / "configs" / "base_test_config.yaml"

    with make_temp_hdf5_regression(
        num_samples=20,
        seq_len=8,
        feat_dim=3,
        target_dim=3,
        target_sequence=True,
        with_splits=False,
    ) as (h5_path, feat_dim):
        # Test data
        torch.manual_seed(42)
        x = torch.randn(2, 8, feat_dim)
        y = torch.randn(2, 8, feat_dim)

        results = {}

        for paradigm in ["unsupervised", "self_supervised"]:
            overrides = {
                "training": {
                    "paradigm": paradigm,
                    "mapping": "seq2seq",
                    "losses": [{"name": "mse", "weight": 1.0, "params": {}}],
                },
                "data": {
                    "data_path": h5_path,
                    "target_seq_len": 8,
                },
            }

            config_path = create_test_config_override(str(base_config), overrides)

            try:
                config = load_config(config_path)

                # Use same model architecture and weights
                torch.manual_seed(123)
                model = build_autoencoder_model(input_dim=feat_dim, hidden_dim=2)
                wrapper = _TestingLightningWrapper(config, model)

                # Test that both produce same loss (should use inputs as targets)
                result = wrapper.training_step((x, y), batch_idx=0)
                results[paradigm] = result["loss"].item()

            finally:
                import os

                os.unlink(config_path)

        # Both paradigms should produce the same loss value
        assert abs(results["unsupervised"] - results["self_supervised"]) < 1e-6, f"Paradigms produced different losses: {results}"


if __name__ == "__main__":
    test_backward_compatibility_unsupervised_paradigm()
    test_new_self_supervised_paradigm()
    test_both_paradigms_equivalent()
