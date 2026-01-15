#!/usr/bin/env python3
"""Demo script showing the new configuration validation and auto-detection capabilities."""

from pathlib import Path
import tempfile

import h5py
import numpy as np

from soen_toolkit.training.configs.config_validation import auto_detect_task_type
from soen_toolkit.training.configs.experiment_config import load_config


def create_demo_datasets() -> dict[str, str]:
    """Create demo datasets for different task types."""
    datasets: dict[str, str] = {}

    # 1. seq2seq classification dataset
    with tempfile.NamedTemporaryFile(suffix="_seq2seq_cls.h5", delete=False) as tmp:
        N, T, D, num_classes = 100, 32, 4, 5
        data = np.random.randn(N, T, D).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=(N, T), dtype=np.int64)

        with h5py.File(tmp.name, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)

        datasets["seq2seq_classification"] = tmp.name

    # 2. seq2static regression dataset
    with tempfile.NamedTemporaryFile(suffix="_seq2static_reg.h5", delete=False) as tmp:
        N, T, D, K = 80, 24, 3, 2
        data = np.random.randn(N, T, D).astype(np.float32)
        labels = np.random.randn(N, K).astype(np.float32)

        with h5py.File(tmp.name, "w") as f:
            train_group = f.create_group("train")
            train_group.create_dataset("data", data=data[:60])
            train_group.create_dataset("labels", data=labels[:60])

            val_group = f.create_group("val")
            val_group.create_dataset("data", data=data[60:])
            val_group.create_dataset("labels", data=labels[60:])

        datasets["seq2static_regression"] = tmp.name

    # 3. Self-supervised dataset (no labels)
    with tempfile.NamedTemporaryFile(suffix="_self_supervised.h5", delete=False) as tmp:
        N, T, D = 60, 20, 6
        data = np.random.randn(N, T, D).astype(np.float32)

        with h5py.File(tmp.name, "w") as f:
            f.create_dataset("data", data=data)
            # No labels - self-supervised

        datasets["self_supervised"] = tmp.name

    return datasets


def demo_auto_detection() -> None:
    """Demonstrate auto-detection capabilities."""
    datasets = create_demo_datasets()

    for dataset_path in datasets.values():
        suggestions = auto_detect_task_type(dataset_path)

        if suggestions["paradigm"]:
            pass
        if suggestions["mapping"]:
            pass
        if suggestions["losses"]:
            [loss["name"] for loss in suggestions["losses"]]
        if suggestions["num_classes"]:
            pass

    # Cleanup
    for path in datasets.values():
        Path(path).unlink(missing_ok=True)


def demo_validation_errors() -> None:
    """Demonstrate validation error detection."""
    # Create a config with intentional errors
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        bad_config = """
description: "Demo config with errors"
seed: 42

training:
  paradigm: invalid_paradigm  # ERROR: Invalid paradigm
  mapping: seq2seq
  batch_size: 8
  max_epochs: 2

  optimizer:
    name: "adamw"
    lr: 0.001

  losses:
    - name: cross_entropy
      weight: 1.0
      params: {}
    - name: mse  # WARNING: Mixing classification and regression losses
      weight: 0.5
      params: {}

data:
  data_path: "/nonexistent/file.h5"  # ERROR: File not found
  cache_data: true
  target_seq_len: 16
  num_classes: 0  # ERROR: Invalid for classification

model:
  base_model_path: null
  time_pooling:   # WARNING: time_pooling ignored for seq2seq
    name: "mean"
    params: {scale: 1.0}

logging:
  project_dir: "temp"
  project_name: "Demo"

callbacks:
  lr_scheduler:
    type: "constant"
        """
        tmp.write(bad_config)
        tmp.flush()
        config_path = tmp.name

    try:
        # This will show detailed validation errors
        load_config(config_path, validate=True)

    except Exception:
        pass

    finally:
        Path(config_path).unlink(missing_ok=True)


def demo_enhanced_error_messages() -> None:
    """Demonstrate enhanced error messages during training."""
    pass


if __name__ == "__main__":
    # Run demos
    demo_auto_detection()
    demo_validation_errors()
    demo_enhanced_error_messages()
