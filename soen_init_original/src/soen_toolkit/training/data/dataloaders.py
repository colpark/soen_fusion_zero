# FILEPATH: src/soen_toolkit/training/data/dataloaders.py

"""Contains Dataset classes for loading Google Speech Commands HDF5 data.
Includes standard and optimized cached versions.
"""

import logging
import os
import time
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Setup logging
logger = logging.getLogger(__name__)


def open_hdf5_with_consistent_locking(path: str, mode: str = "r") -> h5py.File:
    """Open HDF5 file with consistent locking flags to prevent file locking conflicts.

    Args:
        path: Path to the HDF5 file
        mode: File opening mode (default "r")

    Returns:
        Opened h5py.File handle
    """
    try:
        return h5py.File(path, mode, swmr=True, libver="latest", locking=False)
    except TypeError:
        # Fallback for older h5py versions without locking parameter
        try:
            return h5py.File(path, mode, swmr=True, libver="latest")
        except Exception:
            return h5py.File(path, mode)


class GenericHDF5Dataset(Dataset[Any]):
    """Generic dataset loader for HDF5 files with optional splits."""

    def __init__(
        self,
        hdf5_path: str,
        split: str | None = None,
        data_key: str = "data",
        label_key: str = "labels",
        cache_in_memory: bool = False,
        target_seq_len: int | None = None,
        scale_min: float | None = None,
        scale_max: float | None = None,
        # One-hot encoding parameters
        input_encoding: str = "raw",
        vocab_size: int | None = None,
        one_hot_dtype: str = "float32",
    ) -> None:
        """Initialize a generic HDF5 dataset.

        Args:
            hdf5_path: Path to the HDF5 file.
            split: Optional group name within the HDF5 file (e.g. ``train``).
            data_key: Dataset name for input features.
            label_key: Dataset name for labels.
            cache_in_memory: If True, load the entire dataset into memory.
            target_seq_len: If set, sequences are linearly resampled to this length.
            scale_min: Minimum value for scaling. If None (along with scale_max), no scaling is performed.
            scale_max: Maximum value for scaling. If None (along with scale_min), no scaling is performed.
            input_encoding: ``raw`` or ``one_hot``.
            vocab_size: Vocabulary size when using one-hot encoding.
            one_hot_dtype: Data type for one-hot vectors.

        """
        super().__init__()
        self.hdf5_path = hdf5_path
        self.split = split
        self.data_key = data_key
        self.label_key = label_key
        self.cache_in_memory = cache_in_memory
        # ensure target_seq_len is an int for safety
        self.target_seq_len = int(target_seq_len) if target_seq_len is not None else None

        # Scaling settings
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.do_scale = self.scale_min is not None and self.scale_max is not None

        # Global min/max for consistent scaling (will be computed if needed)
        self.global_min = None
        self.global_max = None

        # One-hot encoding settings
        self.input_encoding = input_encoding
        self.vocab_size = vocab_size
        self.one_hot_dtype = one_hot_dtype

        # Validate one-hot settings
        if self.input_encoding == "one_hot":
            if self.vocab_size is None:
                msg = "vocab_size must be specified when input_encoding='one_hot'"
                raise ValueError(msg)
            if self.vocab_size <= 0:
                msg = "vocab_size must be positive"
                raise ValueError(msg)

        self._h5 = None
        self._group_name = None
        if not os.path.exists(self.hdf5_path):
            msg = f"HDF5 not found: {self.hdf5_path}"
            raise FileNotFoundError(msg)

        with open_hdf5_with_consistent_locking(self.hdf5_path) as f:
            grp = f
            if self.split and self.split in f:
                grp = f[self.split]
                self._group_name = self.split
            if self.data_key not in grp or self.label_key not in grp:
                msg = f"Datasets '{self.data_key}' and '{self.label_key}' not found in group '{grp.name}'."
                raise ValueError(
                    msg,
                )
            self.n_samples = grp[self.label_key].shape[0]
            self.data_shape = grp[self.data_key].shape
            self.label_shape = grp[self.label_key].shape

            # Sequence length metadata for optional downsampling
            self.original_seq_len = self.data_shape[1] if len(self.data_shape) > 1 else self.data_shape[0]
            self.do_downsample = self.target_seq_len is not None and self.target_seq_len != self.original_seq_len
            self.final_seq_len = self.target_seq_len if self.do_downsample else self.original_seq_len

            if cache_in_memory:
                self.data_cache = grp[self.data_key][:]

                # Apply transformations to cached data
                if self.do_downsample or self.do_scale:
                    processed_data = []
                    for sample in self.data_cache:
                        sample_array = np.array(sample)

                        # Downsample if requested
                        if self.do_downsample:
                            sample_array = self._resample_sequence(sample_array, self.final_seq_len)  # type: ignore[arg-type]

                        # Scale if requested
                        if self.do_scale:
                            sample_array = self._scale_data(sample_array)

                        processed_data.append(sample_array)

                    self.data_cache = np.stack(processed_data)

                # Preserve label dtype from file (ints for classification, floats for regression)
                self.labels_cache = grp[self.label_key][:]
            else:
                self.data_cache = None
                self.labels_cache = None
        logger.info(
            f"GenericHDF5Dataset loaded: {self.n_samples} samples from {self.hdf5_path}"
            + (f" group '{self._group_name}'" if self._group_name else "")
            + (f" with {self.input_encoding} encoding" if self.input_encoding != "raw" else ""),
        )
        logger.info(
            f"  original_len={self.original_seq_len}, target_len={self.target_seq_len}, final_len={self.final_seq_len}, downsample={self.do_downsample}, scale={self.do_scale}",
        )
        if self.do_scale:
            logger.info(f"  scaling range: [{self.scale_min}, {self.scale_max}]")

    def _get_group(self) -> Any:
        if self._h5 is None:
            # Open a per-worker HDF5 handle with settings that improve parallel reads.
            self._h5 = open_hdf5_with_consistent_locking(self.hdf5_path)
        grp = self._h5
        if self._group_name:
            grp = grp[self._group_name]  # type: ignore[index]
        return grp

    def __len__(self) -> int:
        return int(self.n_samples)

    def _apply_one_hot_encoding(self, data: NDArray[Any]) -> NDArray[Any]:
        """Convert integer sequence data to one-hot encoding."""
        if self.input_encoding != "one_hot":
            return data

        # Ensure data is integer type for one-hot encoding
        data = data.astype(np.int64)

        # Handle different input shapes
        if data.ndim == 1:
            # Shape: (seq_len,) -> (seq_len, vocab_size)
            seq_len = data.shape[0]
            one_hot = np.zeros((seq_len, self.vocab_size), dtype=self.one_hot_dtype)  # type: ignore[arg-type]

            # Clamp values to valid range
            valid_indices = (data >= 0) & (data < self.vocab_size)
            if not np.all(valid_indices):
                logger.warning(f"Found indices outside vocab range [0, {self.vocab_size - 1}]: {data[~valid_indices]}")  # type: ignore[operator]
                data = np.clip(data, 0, self.vocab_size - 1)  # type: ignore[operator]

            one_hot[np.arange(seq_len), data] = 1.0
            return one_hot

        if data.ndim == 2:
            # Shape: (seq_len, 1) -> (seq_len, vocab_size)
            if data.shape[1] == 1:
                return self._apply_one_hot_encoding(data.squeeze(1))
            # Already has multiple features - assume raw encoding is intended
            logger.warning(f"Data shape {data.shape} with multiple features - skipping one-hot encoding")
            return data
        logger.warning(f"Unexpected data shape {data.shape} for one-hot encoding - skipping")
        return data

    def _resample_sequence(self, data: NDArray[Any], target_len: int) -> NDArray[Any]:
        """Resample a sequence along the first dimension using linear interpolation."""
        seq_len = data.shape[0]
        if seq_len == target_len:
            return data

        if data.ndim == 1:
            x_old = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
            return np.interp(x_new, x_old, data).astype(data.dtype)
        feature_dims = data.shape[1:]
        tensor = torch.tensor(data.reshape(seq_len, -1).T, dtype=torch.float32).unsqueeze(0)
        ds = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
        ds_np = ds.squeeze(0).T.reshape(target_len, *feature_dims).numpy()
        return ds_np.astype(data.dtype)

    # backwards compatible alias
    _downsample_sequence = _resample_sequence

    def _scale_data(self, data: NDArray[Any]) -> NDArray[Any]:
        """Scale data to [scale_min, scale_max] range."""
        if not self.do_scale:
            return data

        # Use global min/max if available, otherwise compute per-sample
        if self.global_min is not None and self.global_max is not None:
            data_min, data_max = self.global_min, self.global_max
        # Compute global min/max if not already done
        elif self.global_min is None or self.global_max is None:
            self._compute_global_min_max()
            data_min, data_max = self.global_min, self.global_max
        else:
            # Fallback to per-sample min/max (less ideal)
            data_min, data_max = data.min(), data.max()

        # Avoid division by zero
        if data_max == data_min:
            return np.full_like(data, self.scale_min)

        # Scale to [scale_min, scale_max]
        scale_range = self.scale_max - self.scale_min  # type: ignore[operator]
        scaled = self.scale_min + scale_range * (data - data_min) / (data_max - data_min + 1e-8)  # type: ignore[operator]
        return scaled.astype(data.dtype)

    def _compute_global_min_max(self) -> None:
        """Compute global min/max across the entire dataset for consistent scaling."""
        if not self.do_scale:
            return

        logger.info("Computing global min/max for data scaling...")

        # Sample a subset of data to estimate global min/max (for efficiency)
        sample_size = min(1000, self.n_samples)
        indices = np.random.choice(self.n_samples, sample_size, replace=False)

        global_min = float("inf")
        global_max = float("-inf")

        with open_hdf5_with_consistent_locking(self.hdf5_path) as f:
            grp = f
            if self._group_name:
                grp = f[self._group_name]
            for idx in indices:
                data = np.array(grp[self.data_key][idx])
                global_min = min(global_min, float(data.min()))
                global_max = max(global_max, float(data.max()))

        self.global_min = global_min  # type: ignore[assignment]
        self.global_max = global_max  # type: ignore[assignment]

        logger.info(f"Global min/max computed: [{global_min:.6f}, {global_max:.6f}]")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_memory:
            # Cached data is already downsampled and scaled during __init__
            data = self.data_cache[idx]
            label = self.labels_cache[idx]
            data_array = np.array(data)
            label_array = np.array(label)
        else:
            grp = self._get_group()
            data = grp[self.data_key][idx]
            label = grp[self.label_key][idx]
            # Preserve dtype from file; do not coerce to integer for regression labels

            # Convert to numpy arrays first
            data_array = np.array(data)
            label_array = np.array(label)

            # Downsample if requested
            if self.do_downsample:
                assert self.final_seq_len is not None  # do_downsample implies final_seq_len is set
                data_array = self._resample_sequence(data_array, self.final_seq_len)
                # If labels carry a time dimension (seq2seq/regression), downsample them too
                try:
                    if label_array.ndim >= 2:
                        label_array = self._resample_sequence(label_array, self.final_seq_len)
                except Exception:
                    # Leave labels as-is on failure; loss code will log shape mismatches
                    pass

            # Apply scaling if requested
            if self.do_scale:
                data_array = self._scale_data(data_array)

        # Apply one-hot encoding to input data if specified
        if self.input_encoding == "one_hot":
            data_array = self._apply_one_hot_encoding(data_array)

        # Ensure model-friendly dtype for inputs
        if data_array.dtype != np.float32:
            try:
                data_array = data_array.astype(np.float32, copy=False)
            except Exception:
                data_array = np.asarray(data_array, dtype=np.float32)

        # Cast labels according to their dtype: integers -> long, floats -> float32
        if np.issubdtype(label_array.dtype, np.integer):
            label_tensor = torch.from_numpy(label_array.astype(np.int64))
        else:
            label_tensor = torch.from_numpy(label_array.astype(np.float32))

        return torch.from_numpy(data_array), label_tensor

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


class MelDataset(Dataset[Any]):
    """A unified Dataset class for HDF5 files containing audio spectrograms.

    Supports:
    - Lazy loading (default) or in-memory caching.
    - Optional downsampling to a target sequence length using interpolation.
    - Optional feature scaling to a specified range [scale_min, scale_max].

    Data is assumed to be stored in the HDF5 file under 'data' (N, T, n_mels)
    and 'labels' (N,).
    """

    def __init__(
        self, hdf5_path: str, target_seq_len: int | None = None, scale_min: float | None = None, scale_max: float | None = None, cache_in_memory: bool = False, downsample_mode: str = "pytorch"
    ) -> None:
        """Args:
        hdf5_path: Path to the HDF5 file.
        target_seq_len: Target sequence length for downsampling. If None,
                        no downsampling is performed.
        scale_min: Minimum value for scaling. If None (along with scale_max),
                   no scaling is performed.
        scale_max: Maximum value for scaling. If None (along with scale_min),
                   no scaling is performed.
        cache_in_memory: If True, load the entire dataset (potentially
                         downsampled and scaled) into memory.
        downsample_mode: Interpolation method ('pytorch' or 'numpy').
                         'pytorch' uses F.interpolate (faster), 'numpy' uses
                         np.interp (potentially more memory-efficient for lazy loading).

        """
        super().__init__()

        self.hdf5_path = hdf5_path
        self.target_seq_len = target_seq_len
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.cache_in_memory = cache_in_memory
        self.downsample_mode = downsample_mode
        self._h5: h5py.File | None = None  # For lazy loading

        # Configuration checks
        self.do_scale = self.scale_min is not None and self.scale_max is not None
        if self.do_scale and self.scale_min >= self.scale_max:  # type: ignore[operator]
            msg = "scale_min must be less than scale_max"
            raise ValueError(msg)
        if self.downsample_mode not in ["pytorch", "numpy"]:
            msg = "downsample_mode must be 'pytorch' or 'numpy'"
            raise ValueError(msg)

        if not os.path.exists(self.hdf5_path):
            msg = f"HDF5 not found: {self.hdf5_path}"
            raise FileNotFoundError(msg)

        with open_hdf5_with_consistent_locking(self.hdf5_path) as f:
            if "data" not in f or "labels" not in f:
                msg = "HDF5 file must contain 'data' and 'labels' datasets."
                raise ValueError(msg)
            self.n_samples = f["labels"].shape[0]
            # Original shape (T, n_mels)
            self.original_shape = f["data"].shape[1:]
            self.original_seq_len = self.original_shape[0]
            self.n_mels = self.original_shape[1]

        # Determine if downsampling is needed
        self.do_downsample = self.target_seq_len is not None and self.target_seq_len != self.original_seq_len
        self.final_seq_len = self.target_seq_len if self.do_downsample else self.original_seq_len

        # Cached data attributes
        self.cached_data: torch.Tensor | None = None
        self.cached_labels: torch.Tensor | None = None

        # Lazy loading attributes
        self.global_min: float | None = None
        self.global_max: float | None = None

        # --- Initialization logic ---
        if self.cache_in_memory:
            self._cache_all()
            mode = "Cached"
        else:
            # For lazy loading with scaling, pre-compute global min/max from a subset
            if self.do_scale:
                logger.info("Calculating global min/max for scaling from subset...")
                self._compute_global_min_max_from_subset()
            mode = "Lazy loading"

        logger.info(
            f"UnifiedHDF5Dataset ({mode}) ready: {self.n_samples} samples.",
        )
        logger.info(f"  Original shape: ({self.original_seq_len}, {self.n_mels})")
        if self.do_downsample:
            logger.info(f"  Downsampling: Yes -> Target seq len: {self.final_seq_len}")
        else:
            logger.info("  Downsampling: No")
        if self.do_scale:
            range_info = f"[{self.scale_min:.2f}, {self.scale_max:.2f}]"
            if not self.cache_in_memory and self.global_min is not None:
                range_info += f" (using global min/max: [{self.global_min:.2f}, {self.global_max:.2f}])"
            logger.info(f"  Scaling: Yes -> Target range: {range_info}")
        else:
            logger.info("  Scaling: No")

    def _compute_global_min_max_from_subset(self, subset_size: int = 256) -> None:
        """Estimate global min/max from a random subset for lazy scaling."""
        import random

        subset_size = min(subset_size, self.n_samples)
        if subset_size == 0:
            self.global_min, self.global_max = 0.0, 1.0  # Default if no samples
            return

        sample_indices = random.sample(range(self.n_samples), subset_size)
        mins, maxs = [], []

        with open_hdf5_with_consistent_locking(self.hdf5_path) as f:
            d = f["data"]
            for idx in sample_indices:
                spec = d[idx]  # Shape (T, n_mels)
                # IMPORTANT: Calculate min/max *after* potential downsampling
                # This is tricky for lazy loading without loading all data.
                # We estimate based on original data here. If downsampling
                # significantly alters range, caching might be better.
                if self.do_downsample:
                    # Downsample just this sample to estimate range
                    spec = self._downsample_spectrogram(spec, self.final_seq_len)  # type: ignore[arg-type]

                mins.append(spec.min())
                maxs.append(spec.max())

        self.global_min = float(np.min(mins)) if mins else 0.0
        self.global_max = float(np.max(maxs)) if maxs else 1.0
        # Avoid division by zero if all values in subset are identical
        if self.global_max == self.global_min:
            self.global_max += 1e-8
        logger.info(f"Estimated global range from subset: min={self.global_min:.4f}, max={self.global_max:.4f}")

    def _downsample_spectrogram(self, spect: NDArray[Any], target_len: int) -> NDArray[Any]:
        """Downsample using the selected mode."""
        T, n_mels = spect.shape
        if target_len == T:
            return spect

        if self.downsample_mode == "pytorch":
            # Shape for interpolate: (N=1, C=n_mels, L=T)
            tensor = torch.tensor(spect.T, dtype=torch.float32).unsqueeze(0)
            ds_tensor = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
            # Back to shape (target_len, n_mels)
            return ds_tensor.squeeze(0).T.numpy()
        # numpy mode
        x_old = np.linspace(0.0, 1.0, T, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        # Interpolate each mel bin
        return np.vstack([np.interp(x_new, x_old, spect[:, i]) for i in range(n_mels)]).T.astype(np.float32)

    def _scale_spectrogram(self, spect: NDArray[Any]) -> NDArray[Any]:
        """Scale spectrogram to [scale_min, scale_max]."""
        if not self.do_scale:
            return spect

        # Use pre-computed global min/max if available (lazy mode)
        if self.global_min is not None and self.global_max is not None:
            data_min, data_max = self.global_min, self.global_max
        else:
            # Otherwise, compute per-sample min/max (less ideal but necessary)
            data_min, data_max = spect.min(), spect.max()

        # Avoid division by zero
        if data_max == data_min:
            return np.full_like(spect, self.scale_min)  # Or maybe average?

        scale_range = self.scale_max - self.scale_min  # type: ignore[operator]
        return self.scale_min + scale_range * (spect - data_min) / (data_max - data_min + 1e-8)  # type: ignore[operator]
        # Optional: Clip to ensure bounds, though theoretically unnecessary if min/max are accurate
        # scaled = np.clip(scaled, self.scale_min, self.scale_max)

    def _cache_all(self) -> None:
        """Load, process (downsample/scale), and cache the entire dataset."""
        logger.info("🔁 Caching full dataset in memory...")
        start = time.time()

        with open_hdf5_with_consistent_locking(self.hdf5_path) as f:
            raw_data = f["data"][:]  # Shape: (N, T_orig, n_mels)
            raw_labels = f["labels"][:]  # Shape: (N,)

        processed_data = []
        logger.info(f"Processing {self.n_samples} samples...")
        for i in range(self.n_samples):
            spect = raw_data[i]  # Shape (T_orig, n_mels)

            # 1. Downsample if needed
            if self.do_downsample:
                spect = self._downsample_spectrogram(spect, self.final_seq_len)  # type: ignore[arg-type]
                # Now spect shape is (final_seq_len, n_mels)

            processed_data.append(spect)
            if (i + 1) % 1000 == 0:  # Log progress
                logger.info(f"  Processed {i + 1}/{self.n_samples}...")

        # Stack processed data: Shape (N, final_seq_len, n_mels)
        stacked_data = np.stack(processed_data)

        # 2. Scale if needed (using global min/max of *processed* data)
        if self.do_scale:
            logger.info("Calculating global min/max for scaling...")
            data_min, data_max = np.min(stacked_data), np.max(stacked_data)
            # Avoid division by zero
            if data_max == data_min:
                data_max += 1e-8
            logger.info(f"Processed data range: min={data_min:.4f}, max={data_max:.4f}")
            scale_range = self.scale_max - self.scale_min  # type: ignore[operator]
            scaled_data = self.scale_min + scale_range * (stacked_data - data_min) / (data_max - data_min)
            # Optional clip
            # scaled_data = np.clip(scaled_data, self.scale_min, self.scale_max)
            final_data = scaled_data
            final_min, final_max = self.scale_min, self.scale_max  # Target range
        else:
            final_data = stacked_data
            # Calculate range even if not scaling, for info
            final_min, final_max = np.min(final_data), np.max(final_data)

        # Convert to tensors
        self.cached_data = torch.tensor(final_data, dtype=torch.float32)
        self.cached_labels = torch.tensor(raw_labels, dtype=torch.long)

        duration = time.time() - start
        mem_gb = self.cached_data.nbytes / (1024**3)
        logger.info(f"Cached {self.n_samples} samples in {duration:.2f}s ({mem_gb:.2f} GB).")
        logger.info(f"   Final data shape: {self.cached_data.shape}")
        logger.info(f"   Final data range: min={final_min:.4f}, max={final_max:.4f}")

    # --- Pickling safety for DataLoader workers ---
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Don't pickle the HDF5 file handle if using lazy loading
        if not self.cache_in_memory:
            state["_h5"] = None
        return state

    def _get_file(self) -> h5py.File:
        """Return an open h5py.File handle (one per worker in lazy mode)."""
        if self._h5 is None:
            self._h5 = open_hdf5_with_consistent_locking(self.hdf5_path)
        return self._h5

    # --- Dataset Methods ---
    def __len__(self) -> int:
        return int(self.n_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_memory:
            if self.cached_data is None or self.cached_labels is None:
                msg = "Dataset is set to cache but data is not loaded."
                raise RuntimeError(msg)
            return self.cached_data[idx], self.cached_labels[idx]
        # Lazy loading
        f = self._get_file()
        spect = f["data"][idx]  # (T_orig, n_mels)
        label = f["labels"][idx]

        # 1. Downsample if needed
        if self.do_downsample:
            assert self.final_seq_len is not None  # do_downsample implies final_seq_len is set
            spect = self._downsample_spectrogram(spect, self.final_seq_len)
            # Shape is now (final_seq_len, n_mels)

        # 2. Scale if needed
        spect = self._scale_spectrogram(spect)

        # Convert to tensor
        spect_tensor = torch.tensor(spect, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return spect_tensor, label_tensor


class GenericCSVDataset(Dataset[Any]):
    """CSV dataset with input_* and target[*] columns; tiles static inputs to sequences if requested."""

    def __init__(self, csv_path: str, target_seq_len: int | None = None) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.input_cols = [c for c in self.df.columns if c.startswith("input_")]
        # Accept either target or target_* columns
        if "target" in self.df.columns:
            self.target_scalar = True
            self.target_cols = ["target"]
        else:
            self.target_scalar = False
            self.target_cols = [c for c in self.df.columns if c.startswith("target_")]
        if not self.input_cols or not self.target_cols:
            msg = f"CSV {csv_path} must contain input_* and target or target_* columns"
            raise ValueError(msg)
        self.N = len(self.df)
        self.D = len(self.input_cols)
        self.K = 1 if self.target_scalar else len(self.target_cols)
        self.target_seq_len = int(target_seq_len) if target_seq_len is not None else None

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        x = row[self.input_cols].to_numpy(dtype=np.float32)  # [D]
        y = row[self.target_cols].to_numpy()
        # Determine y dtype: int -> long (classification), float -> float32 (regression)
        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64)
        else:
            y = y.astype(np.float32)
        # Tile static inputs to sequence if requested: [D] -> [T, D]
        if self.target_seq_len is not None:
            x = np.repeat(x[None, :], self.target_seq_len, axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def _maybe_build_csv_loaders(
    csv_paths: dict[str, str],
    batch_size: int,
    num_workers: int,
    target_seq_len: int | None,
    persistent_workers: bool | None,
    prefetch_factor: int | None,
    multiprocessing_context: str | None,
    accelerator: str | None,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    loaders = {}
    persistent = num_workers > 0 if persistent_workers is None else bool(persistent_workers)
    for split in ["train", "val", "test"]:
        if split not in csv_paths:
            msg = f"csv_data_paths missing required split '{split}'"
            raise ValueError(msg)
        ds = GenericCSVDataset(csv_paths[split], target_seq_len=target_seq_len)
        dl_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": (split == "train"),
            "num_workers": num_workers,
            "pin_memory": False if accelerator == "cpu" else torch.cuda.is_available(),
        }
        if num_workers > 0:
            if persistent is not None:
                dl_kwargs["persistent_workers"] = bool(persistent)
            if prefetch_factor is not None:
                dl_kwargs["prefetch_factor"] = int(prefetch_factor)
            if multiprocessing_context:
                dl_kwargs["multiprocessing_context"] = multiprocessing_context
        loaders[split] = DataLoader(ds, **dl_kwargs)
    return loaders["train"], loaders["val"], loaders["test"]


def create_data_loaders(
    data_path: str | None,
    batch_size: int = 64,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 0,
    cache_data: bool = True,
    scale_min: float | None = -1,
    scale_max: float | None = 1,
    target_seq_len: int | None = None,
    # One-hot encoding parameters
    input_encoding: str = "raw",
    vocab_size: int | None = None,
    one_hot_dtype: str = "float32",
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    multiprocessing_context: str | None = None,
    accelerator: str | None = "cpu",
    # NEW: optional explicit CSV splits
    csv_data_paths: dict[str, str] | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Create training, validation and test loaders from an HDF5 or CSV dataset.

    If an HDF5 file contains train/val/test groups, they will be used directly.
    If explicit CSV split paths are provided, those will be used.
    Otherwise the data will be split from a single HDF5 dataset using MelDataset.
    """
    # CSV split handling (explicit)
    if csv_data_paths is not None:
        logger.info("Using explicit CSV split paths for train/val/test")
        return _maybe_build_csv_loaders(csv_data_paths, batch_size, num_workers, target_seq_len, persistent_workers, prefetch_factor, multiprocessing_context, accelerator)

    # Require data_path if csv_data_paths is not provided
    if data_path is None:
        msg = "data_path must be provided if csv_data_paths is not specified"
        raise ValueError(msg)

    # CSV directory auto-detect: if data_path is a directory with train/val/test.csv
    if os.path.isdir(data_path):
        possible = {s: os.path.join(data_path, f"{s}.csv") for s in ("train", "val", "test")}
        if all(os.path.exists(p) for p in possible.values()):
            logger.info("Found CSV files in directory; building CSV loaders")
            return _maybe_build_csv_loaders(possible, batch_size, num_workers, target_seq_len, persistent_workers, prefetch_factor, multiprocessing_context, accelerator)

    # HDF5 handling (existing behavior)
    with open_hdf5_with_consistent_locking(data_path) as f:
        group_names = {k for k in f if isinstance(f[k], h5py.Group)}

    if {"train", "val", "test"}.issubset(group_names):
        logger.info("HDF5 file contains predefined splits")
        persistent = num_workers > 0 if persistent_workers is None else bool(persistent_workers)
        loaders = {}
        for split in ["train", "val", "test"]:
            ds = GenericHDF5Dataset(
                data_path,
                split=split,
                cache_in_memory=cache_data,
                target_seq_len=target_seq_len,
                scale_min=scale_min,
                scale_max=scale_max,
                input_encoding=input_encoding,
                vocab_size=vocab_size,
                one_hot_dtype=one_hot_dtype,
            )
            allowed_accelerators = {"cpu", "gpu", "cuda", "mps", "tpu", "auto"}
            assert accelerator in allowed_accelerators, f"accelerator '{accelerator}' is an unrecognized string."
            dl_kwargs: dict[str, Any] = {
                "batch_size": batch_size,
                "shuffle": (split == "train"),
                "num_workers": num_workers,
                "pin_memory": False if accelerator == "cpu" else torch.cuda.is_available(),
            }
            if num_workers > 0:
                if persistent is not None:
                    dl_kwargs["persistent_workers"] = bool(persistent)
                if prefetch_factor is not None:
                    dl_kwargs["prefetch_factor"] = int(prefetch_factor)
                else:
                    dl_kwargs["prefetch_factor"] = 1
                if multiprocessing_context:
                    dl_kwargs["multiprocessing_context"] = multiprocessing_context
            if persistent and dl_kwargs.get("prefetch_factor") is not None:
                try:
                    dl_kwargs["prefetch_factor"] = max(1, int(dl_kwargs["prefetch_factor"]))
                except Exception:
                    dl_kwargs["prefetch_factor"] = 1
            loaders[split] = DataLoader(ds, **dl_kwargs)
        return loaders["train"], loaders["val"], loaders["test"]

    logger.info(
        f"Creating MelDataset: cache={cache_data}, scale=[{scale_min},{scale_max}], target_len={target_seq_len}",
    )
    dataset = MelDataset(
        hdf5_path=data_path,
        target_seq_len=target_seq_len,
        scale_min=scale_min,
        scale_max=scale_max,
        cache_in_memory=cache_data,
    )

    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size

    if train_size + val_size + test_size != total_size:
        msg = f"Internal error: Split sizes do not sum correctly. total={total_size}, train={train_size}, val={val_size}, test={test_size}"
        raise ValueError(
            msg,
        )

    logger.info(
        f"Splitting dataset: {total_size} total -> {train_size} train, {val_size} val, {test_size} test",
    )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    persistent = num_workers > 0
    common_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False if accelerator == "cpu" else torch.cuda.is_available(),
    }
    if persistent:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_kwargs,
    )

    return train_loader, val_loader, test_loader
