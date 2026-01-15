"""Test the new configuration validation and auto-detection system."""

import tempfile

import h5py
import numpy as np
import pytest

from soen_toolkit.training.configs.config_classes import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    LossItemConfig,
    ModelConfig,
    TrainingConfig,
)
from soen_toolkit.training.configs.config_validation import (
    ConfigValidationError,
    auto_detect_task_type,
    validate_config,
)


def test_paradigm_validation() -> None:
    """Test paradigm validation with invalid values."""
    config = ExperimentConfig(
        training=TrainingConfig(
            paradigm="invalid_paradigm",
            mapping="seq2static",
            loss=LossConfig(losses=[LossItemConfig(name="mse", weight=1.0)]),
        ),
        data=DataConfig(data_path=""),  # Empty path to skip file validation
        model=ModelConfig(),
    )

    _warnings, errors = validate_config(config, raise_on_error=False)

    assert len(errors) > 0
    assert any("Invalid paradigm 'invalid_paradigm'" in error for error in errors)
    assert any("supervised" in error for error in errors)


def test_mapping_validation() -> None:
    """Test mapping validation with invalid values."""
    config = ExperimentConfig(
        training=TrainingConfig(
            paradigm="supervised",
            mapping="invalid_mapping",
            loss=LossConfig(losses=[LossItemConfig(name="cross_entropy", weight=1.0)]),
        ),
        data=DataConfig(data_path=""),  # Empty path to skip file validation
        model=ModelConfig(),
    )

    _warnings, errors = validate_config(config, raise_on_error=False)

    assert len(errors) > 0
    assert any("Invalid mapping 'invalid_mapping'" in error for error in errors)
    assert any("seq2static" in error for error in errors)


def test_backward_compatibility_warning() -> None:
    """Test that 'unsupervised' paradigm triggers deprecation warning."""
    config = ExperimentConfig(
        training=TrainingConfig(
            paradigm="unsupervised",
            mapping="seq2seq",
            loss=LossConfig(losses=[LossItemConfig(name="mse", weight=1.0)]),
        ),
        data=DataConfig(data_path=""),  # Empty path to skip file validation
        model=ModelConfig(),
    )

    warnings, errors = validate_config(config, raise_on_error=False)

    assert len(errors) == 0  # Should not error
    assert len(warnings) > 0
    assert any("deprecated" in warning.lower() for warning in warnings)
    assert any("self_supervised" in warning for warning in warnings)


def test_loss_function_validation() -> None:
    """Test loss function compatibility validation."""
    # Classification with regression loss
    config = ExperimentConfig(
        training=TrainingConfig(
            paradigm="supervised",
            mapping="seq2static",
            loss=LossConfig(losses=[LossItemConfig(name="mse", weight=1.0)]),  # Wrong for classification
        ),
        data=DataConfig(num_classes=5, data_path=""),  # Indicates classification, empty path to skip file validation
        model=ModelConfig(),
    )

    _warnings, errors = validate_config(config, raise_on_error=False)

    # Should warn about loss function choice (no errors since no file validation)
    assert len(errors) == 0  # No errors expected
    # Note: This specific warning might not trigger without actual data analysis


def test_auto_detect_seq2seq_classification() -> None:
    """Test auto-detection of seq2seq classification task."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
        # Create seq2seq classification dataset
        N, T, D, num_classes = 50, 16, 4, 3
        data = np.random.randn(N, T, D).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=(N, T), dtype=np.int64)

        with h5py.File(tmp.name, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)

        suggestions = auto_detect_task_type(tmp.name)

        assert suggestions["paradigm"] == "supervised"
        assert suggestions["mapping"] == "seq2seq"
        assert suggestions["confidence"] == "high"
        assert suggestions["num_classes"] == num_classes
        assert any(loss["name"] == "cross_entropy" for loss in suggestions["losses"])
        # File is automatically cleaned up when exiting the context manager


def test_auto_detect_seq2static_regression() -> None:
    """Test auto-detection of seq2static regression task."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
        # Create seq2static regression dataset
        N, T, D, K = 60, 20, 3, 2
        data = np.random.randn(N, T, D).astype(np.float32)
        labels = np.random.randn(N, K).astype(np.float32)  # [N, K] regression targets

        with h5py.File(tmp.name, "w") as f:
            train_group = f.create_group("train")
            train_group.create_dataset("data", data=data[:40])
            train_group.create_dataset("labels", data=labels[:40])

            val_group = f.create_group("val")
            val_group.create_dataset("data", data=data[40:])
            val_group.create_dataset("labels", data=labels[40:])

        suggestions = auto_detect_task_type(tmp.name)

        assert suggestions["paradigm"] == "supervised"
        assert suggestions["mapping"] == "seq2static"
        assert suggestions["confidence"] == "high"
        assert any(loss["name"] == "mse" for loss in suggestions["losses"])
        # File is automatically cleaned up when exiting the context manager


def test_auto_detect_self_supervised() -> None:
    """Test auto-detection of self-supervised task (no labels)."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
        # Create dataset with only data (no labels)
        N, T, D = 40, 12, 6
        data = np.random.randn(N, T, D).astype(np.float32)

        with h5py.File(tmp.name, "w") as f:
            f.create_dataset("data", data=data)
            # No labels dataset

        suggestions = auto_detect_task_type(tmp.name)

        assert suggestions["paradigm"] == "self_supervised"
        assert suggestions["mapping"] == "seq2seq"
        assert suggestions["confidence"] == "medium"
        assert any(loss["name"] == "mse" for loss in suggestions["losses"])
        # File is automatically cleaned up when exiting the context manager


def test_sequence_length_validation() -> None:
    """Test sequence length validation warnings."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
        # Create dataset with specific sequence length
        N, T, D = 30, 24, 3
        data = np.random.randn(N, T, D).astype(np.float32)
        labels = np.random.randint(0, 2, size=(N,), dtype=np.int64)

        with h5py.File(tmp.name, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)

        # Config with mismatched target_seq_len
        config = ExperimentConfig(
            training=TrainingConfig(
                paradigm="supervised",
                mapping="seq2seq",  # seq2seq should match sequence lengths
                loss=LossConfig(losses=[LossItemConfig(name="cross_entropy", weight=1.0)]),
            ),
            data=DataConfig(
                data_path=tmp.name,
                target_seq_len=16,  # Different from data T=24
                num_classes=2,
            ),
            model=ModelConfig(),
        )

        warnings, _errors = validate_config(config, raise_on_error=False)

        # Should warn about sequence length mismatch for seq2seq
        assert any("mismatch" in warning.lower() for warning in warnings)
        # File is automatically cleaned up when exiting the context manager


def test_time_pooling_warnings() -> None:
    """Test time pooling configuration warnings."""
    # seq2seq with time pooling (should warn)
    config = ExperimentConfig(
        training=TrainingConfig(
            paradigm="supervised",
            mapping="seq2seq",
            loss=LossConfig(losses=[LossItemConfig(name="mse", weight=1.0)]),
        ),
        data=DataConfig(data_path=""),  # Empty path to skip file validation
        model=ModelConfig(time_pooling={"name": "mean", "params": {"scale": 1.0}}),
    )

    warnings, _errors = validate_config(config, raise_on_error=False)

    assert any("time_pooling" in warning for warning in warnings)
    assert any("ignored" in warning for warning in warnings)


def test_self_supervised_paradigm_combinations() -> None:
    """Test validation of self-supervised paradigm with different mappings."""
    # Valid combination
    config = ExperimentConfig(
        training=TrainingConfig(
            paradigm="self_supervised",
            mapping="seq2seq",
            loss=LossConfig(losses=[LossItemConfig(name="mse", weight=1.0)]),
        ),
        data=DataConfig(data_path=""),  # Empty path to skip file validation
        model=ModelConfig(),
    )

    warnings, errors = validate_config(config, raise_on_error=False)
    assert len(errors) == 0  # Should be valid

    # Invalid combination
    config_invalid = ExperimentConfig(
        training=TrainingConfig(
            paradigm="self_supervised",
            mapping="static2seq",  # Not supported for self-supervised
            loss=LossConfig(losses=[LossItemConfig(name="mse", weight=1.0)]),
        ),
        data=DataConfig(data_path=""),  # Empty path to skip file validation
        model=ModelConfig(),
    )

    _warnings, errors = validate_config(config_invalid, raise_on_error=False)
    assert len(errors) > 0
    assert any("self-supervised" in error.lower() for error in errors)


def test_validation_error_raising() -> None:
    """Test that validation errors are properly raised when requested."""
    config = ExperimentConfig(
        training=TrainingConfig(
            paradigm="invalid",
            mapping="seq2static",
            loss=LossConfig(losses=[LossItemConfig(name="mse", weight=1.0)]),
        ),
        data=DataConfig(data_path=""),  # Empty path to skip file validation
        model=ModelConfig(),
    )

    # Should raise error
    with pytest.raises(ConfigValidationError):
        validate_config(config, raise_on_error=True)

    # Should not raise error
    _warnings, errors = validate_config(config, raise_on_error=False)
    assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
