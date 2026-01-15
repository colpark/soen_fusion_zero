"""Synthetic dataset generators for training tests across all paradigms.
Extends the existing hdf5_test_helpers.py with more task-specific generators.
"""

from contextlib import contextmanager, suppress
import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np


@contextmanager
def make_temp_hdf5_seq2seq_classification(
    *,
    num_samples: int = 60,
    seq_len: int = 16,
    feat_dim: int = 6,
    num_classes: int = 3,
    with_splits: bool = True,
) -> tuple[str, int]:
    """Create a temporary HDF5 sequence-to-sequence classification dataset.

    Layout: labels shape [N, T] with class indices for each timestep.
    """
    tmp = NamedTemporaryFile(delete=False, suffix=".h5")
    tmp.close()
    path = tmp.name

    rng = np.random.default_rng(456)
    data = rng.normal(size=(num_samples, seq_len, feat_dim)).astype(np.float32)
    labels = rng.integers(low=0, high=num_classes, size=(num_samples, seq_len), dtype=np.int64)

    with h5py.File(path, "w") as f:
        if with_splits:
            n_train = int(0.7 * num_samples)
            n_val = int(0.15 * num_samples)
            num_samples - n_train - n_val
            splits = {
                "train": (0, n_train),
                "val": (n_train, n_train + n_val),
                "test": (n_train + n_val, num_samples),
            }
            for name, (s, e) in splits.items():
                g = f.create_group(name)
                g.create_dataset("data", data=data[s:e])
                g.create_dataset("labels", data=labels[s:e])
        else:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)

    try:
        yield path, num_classes
    finally:
        with suppress(OSError):
            os.remove(path)


@contextmanager
def make_temp_hdf5_unsupervised_seq2static(
    *,
    num_samples: int = 60,
    seq_len: int = 16,
    feat_dim: int = 6,
    target_dim: int = 4,
    with_splits: bool = True,
) -> tuple[str, int]:
    """Create a temporary HDF5 dataset for unsupervised seq2static tasks.

    The idea is to have input sequences and learn to predict a summary statistic
    of the sequence (like mean, max, etc.) in an unsupervised manner.
    Labels will be derived from the input data itself.
    """
    tmp = NamedTemporaryFile(delete=False, suffix=".h5")
    tmp.close()
    path = tmp.name

    rng = np.random.default_rng(789)
    data = rng.normal(size=(num_samples, seq_len, feat_dim)).astype(np.float32)

    # Create "unsupervised" targets by computing statistics of the input
    # Take mean over time and select first target_dim features
    labels = np.mean(data, axis=1)[:, :target_dim].astype(np.float32)

    with h5py.File(path, "w") as f:
        if with_splits:
            n_train = int(0.7 * num_samples)
            n_val = int(0.15 * num_samples)
            num_samples - n_train - n_val
            splits = {
                "train": (0, n_train),
                "val": (n_train, n_train + n_val),
                "test": (n_train + n_val, num_samples),
            }
            for name, (s, e) in splits.items():
                g = f.create_group(name)
                g.create_dataset("data", data=data[s:e])
                g.create_dataset("labels", data=labels[s:e])
        else:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)

    try:
        yield path, target_dim
    finally:
        with suppress(OSError):
            os.remove(path)


@contextmanager
def make_temp_hdf5_pulse_classification(
    *,
    num_samples: int = 100,
    seq_len: int = 32,
    feat_dim: int = 1,
    num_classes: int = 3,
    with_splits: bool = True,
) -> tuple[str, int]:
    """Create a synthetic pulse classification dataset similar to the tutorial example.

    Classes:
    0: No pulse
    1: Single pulse
    2: Two pulses
    """
    tmp = NamedTemporaryFile(delete=False, suffix=".h5")
    tmp.close()
    path = tmp.name

    rng = np.random.default_rng(101112)
    data = np.zeros((num_samples, seq_len, feat_dim), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        # Add some noise
        data[i] = rng.normal(0, 0.1, size=(seq_len, feat_dim))

        class_choice = rng.integers(0, num_classes)
        labels[i] = class_choice

        if class_choice == 1:  # Single pulse
            pulse_pos = rng.integers(5, seq_len - 5)
            pulse_width = rng.integers(2, 6)
            pulse_amp = rng.uniform(0.8, 1.2)
            data[i, pulse_pos : pulse_pos + pulse_width, 0] += pulse_amp

        elif class_choice == 2:  # Two pulses
            pulse1_pos = rng.integers(3, seq_len // 2)
            pulse2_pos = rng.integers(seq_len // 2 + 2, seq_len - 3)
            pulse_width = rng.integers(2, 4)
            pulse1_amp = rng.uniform(0.6, 1.0)
            pulse2_amp = rng.uniform(0.6, 1.0)
            data[i, pulse1_pos : pulse1_pos + pulse_width, 0] += pulse1_amp
            data[i, pulse2_pos : pulse2_pos + pulse_width, 0] += pulse2_amp

    with h5py.File(path, "w") as f:
        if with_splits:
            n_train = int(0.7 * num_samples)
            n_val = int(0.15 * num_samples)
            num_samples - n_train - n_val
            splits = {
                "train": (0, n_train),
                "val": (n_train, n_train + n_val),
                "test": (n_train + n_val, num_samples),
            }
            for name, (s, e) in splits.items():
                g = f.create_group(name)
                g.create_dataset("data", data=data[s:e])
                g.create_dataset("labels", data=labels[s:e])
        else:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)

    try:
        yield path, num_classes
    finally:
        with suppress(OSError):
            os.remove(path)


@contextmanager
def make_temp_hdf5_time_series_forecasting(
    *,
    num_samples: int = 80,
    seq_len: int = 24,
    feat_dim: int = 3,
    forecast_len: int = 8,
    with_splits: bool = True,
) -> tuple[str, int]:
    """Create a synthetic time series forecasting dataset.

    Input: [N, seq_len, feat_dim]
    Labels: [N, forecast_len, feat_dim] (next forecast_len timesteps)
    """
    tmp = NamedTemporaryFile(delete=False, suffix=".h5")
    tmp.close()
    path = tmp.name

    rng = np.random.default_rng(131415)

    # Generate synthetic time series with trends and seasonality
    total_len = seq_len + forecast_len
    data_full = np.zeros((num_samples, total_len, feat_dim), dtype=np.float32)

    for i in range(num_samples):
        for d in range(feat_dim):
            # Create synthetic signal with trend + seasonality + noise
            t = np.arange(total_len)
            trend = rng.uniform(-0.01, 0.01) * t
            seasonal = rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * t / rng.uniform(8, 16))
            noise = rng.normal(0, 0.2, total_len)
            data_full[i, :, d] = trend + seasonal + noise

    # Split into input and target sequences
    data = data_full[:, :seq_len, :].astype(np.float32)
    labels = data_full[:, seq_len:, :].astype(np.float32)

    with h5py.File(path, "w") as f:
        if with_splits:
            n_train = int(0.7 * num_samples)
            n_val = int(0.15 * num_samples)
            num_samples - n_train - n_val
            splits = {
                "train": (0, n_train),
                "val": (n_train, n_train + n_val),
                "test": (n_train + n_val, num_samples),
            }
            for name, (s, e) in splits.items():
                g = f.create_group(name)
                g.create_dataset("data", data=data[s:e])
                g.create_dataset("labels", data=labels[s:e])
        else:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)

    try:
        yield path, feat_dim
    finally:
        with suppress(OSError):
            os.remove(path)
