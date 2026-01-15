# FILEPATH: src/soen_toolkit/tests/training/test_distillation.py

"""Tests for the distillation training module."""

from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest
import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.training.configs import DistillationConfig, TrainingConfig
from soen_toolkit.training.distillation import generate_teacher_trajectories


def _build_teacher_model(input_dim: int = 3, output_dim: int = 5) -> SOENModelCore:
    """Build a simple teacher model for testing."""
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": input_dim}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": output_dim}),
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
    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def _build_student_model(input_dim: int = 3, output_dim: int = 5) -> SOENModelCore:
    """Build a simple student model (same architecture as teacher for testing)."""
    return _build_teacher_model(input_dim, output_dim)


def _create_source_dataset(path: Path, num_samples: int = 20, seq_len: int = 10, input_dim: int = 3) -> None:
    """Create a simple HDF5 dataset for testing."""
    rng = np.random.default_rng(42)

    # Split sizes
    n_train = int(0.7 * num_samples)
    n_val = int(0.15 * num_samples)
    n_test = num_samples - n_train - n_val

    with h5py.File(path, "w") as f:
        for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
            grp = f.create_group(split)
            data = rng.standard_normal((n, seq_len, input_dim)).astype(np.float32)
            labels = rng.integers(0, 5, size=(n,)).astype(np.int64)
            grp.create_dataset("data", data=data)
            grp.create_dataset("labels", data=labels)


class TestDistillationConfig:
    """Test DistillationConfig dataclass."""

    def test_valid_config(self) -> None:
        cfg = DistillationConfig(
            teacher_model_path="/path/to/teacher.pth",
            subset_fraction=0.5,
            output_layer_only=True,
        )
        assert cfg.subset_fraction == 0.5
        assert cfg.output_layer_only is True

    def test_default_values(self) -> None:
        cfg = DistillationConfig(teacher_model_path="/path/to/teacher.pth")
        assert cfg.subset_fraction == 1.0
        assert cfg.output_layer_only is True
        assert cfg.batch_size == 32

    def test_invalid_subset_fraction_zero(self) -> None:
        with pytest.raises(ValueError, match="subset_fraction"):
            DistillationConfig(teacher_model_path="/tmp/x.pth", subset_fraction=0.0)

    def test_invalid_subset_fraction_negative(self) -> None:
        with pytest.raises(ValueError, match="subset_fraction"):
            DistillationConfig(teacher_model_path="/tmp/x.pth", subset_fraction=-0.5)

    def test_invalid_subset_fraction_too_large(self) -> None:
        with pytest.raises(ValueError, match="subset_fraction"):
            DistillationConfig(teacher_model_path="/tmp/x.pth", subset_fraction=1.5)

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            DistillationConfig(teacher_model_path="/tmp/x.pth", batch_size=0)


class TestTrainingConfigDistillation:
    """Test distillation integration in TrainingConfig."""

    def test_distillation_paradigm_requires_config(self) -> None:
        with pytest.raises(ValueError, match="distillation"):
            TrainingConfig(paradigm="distillation")

    def test_distillation_config_sets_paradigm(self) -> None:
        cfg = TrainingConfig(
            paradigm="supervised",  # Will be overridden
            distillation=DistillationConfig(teacher_model_path="/tmp/x.pth"),
        )
        assert cfg.paradigm == "distillation"
        assert cfg.mapping == "seq2seq"

    def test_distillation_config_from_dict(self) -> None:
        cfg = TrainingConfig(
            distillation={"teacher_model_path": "/tmp/x.pth", "subset_fraction": 0.5},
        )
        assert isinstance(cfg.distillation, DistillationConfig)
        assert cfg.distillation.subset_fraction == 0.5


class TestTeacherDataGenerator:
    """Test teacher trajectory generation."""

    def test_generate_teacher_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create teacher model
            teacher = _build_teacher_model(input_dim=3, output_dim=5)
            teacher_path = tmpdir / "teacher.pth"
            teacher.save(str(teacher_path))

            # Create source dataset
            source_path = tmpdir / "source_data.hdf5"
            _create_source_dataset(source_path, num_samples=20, seq_len=10, input_dim=3)

            # Generate distillation data
            output_path = tmpdir / "distillation_data.hdf5"
            generate_teacher_trajectories(
                teacher_model_path=teacher_path,
                source_data_path=source_path,
                output_path=output_path,
                subset_fraction=1.0,
                batch_size=4,
            )

            # Verify output structure
            assert output_path.exists()

            with h5py.File(output_path, "r") as f:
                assert "train" in f
                assert "val" in f
                assert "test" in f

                # Check train split
                train_data = f["train/data"][:]
                train_labels = f["train/labels"][:]

                # Data should have original shape
                assert train_data.ndim == 3
                # Labels should be teacher states [N, T+1, output_dim]
                # T+1 because teacher output includes t=0 state
                assert train_labels.ndim == 3
                assert train_labels.shape[0] == train_data.shape[0]  # Same number of samples
                assert train_labels.shape[1] == train_data.shape[1] + 1  # T+1 timesteps
                assert train_labels.shape[2] == 5  # Output dimension

    def test_generate_teacher_trajectories_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            teacher = _build_teacher_model(input_dim=3, output_dim=5)
            teacher_path = tmpdir / "teacher.pth"
            teacher.save(str(teacher_path))

            source_path = tmpdir / "source_data.hdf5"
            _create_source_dataset(source_path, num_samples=100, seq_len=10, input_dim=3)

            output_path = tmpdir / "distillation_data.hdf5"
            generate_teacher_trajectories(
                teacher_model_path=teacher_path,
                source_data_path=source_path,
                output_path=output_path,
                subset_fraction=0.5,
                batch_size=8,
            )

            with h5py.File(source_path, "r") as src_f, h5py.File(output_path, "r") as out_f:
                # Train split should have roughly half the samples
                original_train = src_f["train/data"].shape[0]
                distill_train = out_f["train/data"].shape[0]
                # Allow some tolerance due to rounding
                assert 0.4 * original_train <= distill_train <= 0.6 * original_train

    def test_teacher_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_path = tmpdir / "source_data.hdf5"
            _create_source_dataset(source_path, num_samples=10, seq_len=5, input_dim=3)

            with pytest.raises(FileNotFoundError, match="Teacher model"):
                generate_teacher_trajectories(
                    teacher_model_path=tmpdir / "nonexistent.pth",
                    source_data_path=source_path,
                    output_path=tmpdir / "output.hdf5",
                )

    def test_source_dataset_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            teacher = _build_teacher_model()
            teacher_path = tmpdir / "teacher.pth"
            teacher.save(str(teacher_path))

            with pytest.raises(FileNotFoundError, match="Source dataset"):
                generate_teacher_trajectories(
                    teacher_model_path=teacher_path,
                    source_data_path=tmpdir / "nonexistent.hdf5",
                    output_path=tmpdir / "output.hdf5",
                )


class TestNoneTimePooling:
    """Test the 'none' time pooling option."""

    def test_none_pooling_returns_full_sequence(self) -> None:
        from soen_toolkit.training.configs import DataConfig, ExperimentConfig, ModelConfig

        # Create a minimal config with 'none' time pooling
        cfg = ExperimentConfig(
            data=DataConfig(num_classes=5, synthetic=True),
            model=ModelConfig(
                time_pooling={"name": "none", "params": {}},
                architecture={
                    "simulation": {"dt": 37, "input_type": "flux"},
                    "layers": [
                        {"layer_id": 0, "layer_type": "RNN", "params": {"dim": 3}},
                        {"layer_id": 1, "layer_type": "RNN", "params": {"dim": 5}},
                    ],
                    "connections": [
                        {"from_layer": 0, "to_layer": 1, "connection_type": "dense"},
                    ],
                },
            ),
        )

        from soen_toolkit.training.models import SOENLightningModule

        model = SOENLightningModule(cfg)

        # Create dummy input
        batch_size, seq_len, input_dim = 2, 8, 3
        x = torch.randn(batch_size, seq_len, input_dim)

        # Forward pass
        output, final_state, all_states = model(x)

        # With 'none' pooling, output should be 3D: [batch, seq_len, dim]
        assert output.ndim == 3
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len  # Full sequence
        assert output.shape[2] == 5  # Output dimension
