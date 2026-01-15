from contextlib import contextmanager, suppress
import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np


@contextmanager
def make_temp_hdf5_classification(
    *,
    num_samples: int = 60,
    seq_len: int = 20,
    feat_dim: int = 8,
    num_classes: int = 4,
    with_splits: bool = True,
) -> tuple[str, int]:
    """Create a temporary HDF5 classification dataset for tests.

    Layout:
    - If with_splits=True: groups train/val/test each with datasets 'data' [N,T,D] and 'labels' [N]
    - Else: datasets at root 'data' [N,T,D] and 'labels' [N]

    Yields:
    - path to temp file
    - number of classes

    """
    tmp = NamedTemporaryFile(delete=False, suffix=".h5")
    tmp.close()
    path = tmp.name

    rng = np.random.default_rng(123)
    data = rng.normal(size=(num_samples, seq_len, feat_dim)).astype(np.float32)
    labels = rng.integers(low=0, high=num_classes, size=(num_samples,), dtype=np.int64)

    with h5py.File(path, "w") as f:
        if with_splits:
            # 70/15/15 split
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
def make_temp_hdf5_regression(
    *,
    num_samples: int = 64,
    seq_len: int = 20,
    feat_dim: int = 6,
    target_dim: int = 3,
    target_sequence: bool = False,
    with_splits: bool = True,
) -> tuple[str, int]:
    """Create a temporary HDF5 regression dataset with float targets.

    - If target_sequence=False: labels shape [N, target_dim]
    - If target_sequence=True: labels shape [N, seq_len, target_dim]

    Returns path and target_dim.
    """
    tmp = NamedTemporaryFile(delete=False, suffix=".h5")
    tmp.close()
    path = tmp.name

    rng = np.random.default_rng(321)
    data = rng.normal(size=(num_samples, seq_len, feat_dim)).astype(np.float32)
    if target_sequence:
        labels = rng.normal(size=(num_samples, seq_len, target_dim)).astype(np.float32)
    else:
        labels = rng.normal(size=(num_samples, target_dim)).astype(np.float32)

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
