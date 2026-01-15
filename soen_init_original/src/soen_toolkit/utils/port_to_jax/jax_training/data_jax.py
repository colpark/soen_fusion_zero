# FILEPATH: src/soen_toolkit/utils/port_to_jax/jax_training/data_jax.py


from __future__ import annotations

from pathlib import Path
import queue
import threading
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterator


def _resample_np(seq: NDArray[np.floating], target_len: int) -> NDArray[np.floating]:
    """Resample a sequence to a target length using linear interpolation.

    Args:
        seq: Input sequence of shape (T,) or (T, D)
        target_len: Target sequence length

    Returns:
        Resampled sequence of shape (target_len,) or (target_len, D)
    """
    T = seq.shape[0]
    if target_len == T:
        return seq
    x_old = np.linspace(0.0, 1.0, T, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    if seq.ndim == 1:
        return np.interp(x_new, x_old, seq).astype(seq.dtype)
    seq.shape[1:]
    out = np.vstack([np.interp(x_new, x_old, seq[:, i]) for i in range(seq.shape[1])]).T
    return out.astype(seq.dtype)


def load_hdf5_splits(
    path: str | Path,
    *,
    target_seq_len: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    cache: bool = True,
    labels_as_int: bool = True,
) -> tuple[
    tuple[NDArray[np.float32], NDArray[np.int64] | NDArray[np.float32]],
    tuple[NDArray[np.float32], NDArray[np.int64] | NDArray[np.float32]],
]:
    """Load train/val groups from HDF5 into NumPy arrays.

    Applies optional linear resampling along time and global min/max scaling.

    Args:
        path: Path to HDF5 file.
        target_seq_len: If set, resample sequences to this length.
        scale_min: Minimum value for input scaling.
        scale_max: Maximum value for input scaling.
        cache: Unused, kept for compatibility.
        labels_as_int: If True, convert labels to int64 (for classification).
                      If False, keep labels as float32 (for regression/distillation).

    Returns:
        ((X_train, y_train), (X_val, y_val))
    """
    from soen_toolkit.training.data.dataloaders import open_hdf5_with_consistent_locking

    with open_hdf5_with_consistent_locking(path) as f:
        if not all(g in f for g in ("train", "val")):
            msg = "HDF5 must contain 'train' and 'val' groups for pure JAX pipeline"
            raise ValueError(msg)

        def _load_group(gname: str):
            g = f[gname]
            X = g["data"][:]
            y = g["labels"][:]
            return X, y

        Xtr, ytr = _load_group("train")
        Xva, yva = _load_group("val")

    # Resample if requested
    def _maybe_resample(X: np.ndarray) -> np.ndarray:
        if target_seq_len is None:
            return X.astype(np.float32, copy=False)
        out = np.stack([_resample_np(X[i], target_seq_len) for i in range(X.shape[0])], axis=0)
        return out.astype(np.float32, copy=False)

    Xtr = _maybe_resample(Xtr)
    Xva = _maybe_resample(Xva)

    # Scale with global min/max if provided
    if scale_min is not None and scale_max is not None:
        gmin = float(min(np.min(Xtr), np.min(Xva)))
        gmax = float(max(np.max(Xtr), np.max(Xva)))
        if gmax == gmin:
            gmax = gmin + 1e-8
        rng = scale_max - scale_min
        Xtr = scale_min + rng * (Xtr - gmin) / (gmax - gmin)
        Xva = scale_min + rng * (Xva - gmin) / (gmax - gmin)

    # Convert labels based on paradigm
    if labels_as_int:
        # Classification: convert to int64
        ytr = ytr.astype(np.int64, copy=False)
        yva = yva.astype(np.int64, copy=False)
    else:
        # Regression/distillation: keep as float32
        # Validate that labels are actually float-like (catch misconfigurations early)
        if ytr.ndim > 1 and ytr.shape[1] > 1:
            # Multi-dimensional labels (e.g., [N, T, D]) suggest regression/distillation
            if np.issubdtype(ytr.dtype, np.integer):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Labels have integer dtype {ytr.dtype} but labels_as_int=False. "
                    f"This may indicate incorrect paradigm setting. Converting to float32 anyway."
                )
        ytr = ytr.astype(np.float32, copy=False)
        yva = yva.astype(np.float32, copy=False)
    return (Xtr, ytr), (Xva, yva)


class JAXBatchIterator:
    """Batched iterator for JAX training that supports optional prefetching.

    Args:
        X: Input data array of shape (N, ...)
        y: Label array of shape (N, ...)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data at each epoch
        prefetch: Number of batches to prefetch in background (0 = no prefetch)
    """

    def __init__(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int64] | NDArray[np.float32],
        *,
        batch_size: int,
        shuffle: bool = True,
        prefetch: int = 2,
    ) -> None:
        self.X: NDArray[np.float32] = X
        self.y: NDArray[np.int64] | NDArray[np.float32] = y
        self.N: int = X.shape[0]
        self.batch_size: int = int(batch_size)
        self.shuffle: bool = bool(shuffle)
        self.prefetch: int = max(0, int(prefetch))

    def __len__(self) -> int:
        return int(np.ceil(self.N / self.batch_size))

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
        idx = np.arange(self.N)
        if self.shuffle:
            np.random.shuffle(idx)

        def producer(q: queue.Queue[tuple[jnp.ndarray, jnp.ndarray] | None]) -> None:
            for start in range(0, self.N, self.batch_size):
                sl = idx[start : start + self.batch_size]
                xb = jnp.asarray(self.X[sl])
                yb = jnp.asarray(self.y[sl])
                q.put((xb, yb))
            q.put(None)

        if self.prefetch > 0:
            q: queue.Queue[tuple[jnp.ndarray, jnp.ndarray] | None] = queue.Queue(maxsize=self.prefetch)
            t = threading.Thread(target=producer, args=(q,), daemon=True)
            t.start()
            while True:
                item = q.get()
                if item is None:
                    break
                yield item
        else:
            for start in range(0, self.N, self.batch_size):
                sl = idx[start : start + self.batch_size]
                yield jnp.asarray(self.X[sl]), jnp.asarray(self.y[sl])
