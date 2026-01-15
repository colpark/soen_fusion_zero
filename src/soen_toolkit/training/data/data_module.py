# FILEPATH: src/soen_toolkit/training/data/data_module.py

"""Data module for SOEN model training.

This module provides a PyTorch Lightning data module for loading and processing data.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# Import SOEN-specific data loading utilities
from soen_toolkit.training.data.dataloaders import create_data_loaders

logger = logging.getLogger(__name__)


class SOENDataModule(pl.LightningDataModule):
    # add docstring
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.data_path = config.data.data_path
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.accel = config.training.accelerator

        # Synthetic/no-dataset mode
        self.synthetic = bool(getattr(config.data, "synthetic", False))
        self.synthetic_kwargs = getattr(config.data, "synthetic_kwargs", {}) or {}

        self.train_loader: DataLoader[Any] | None = None
        self.val_loader: DataLoader[Any] | None = None
        self.test_loader: DataLoader[Any] | None = None

        # Cache last-used target seq length to detect changes
        self._current_target_seq_len = self.config.data.target_seq_len

        logger.info(f"Initialized SOENDataModule with data_path={self.data_path}, batch_size={self.batch_size}")

    def prepare_data(self) -> None:
        """Verify data exists (called once by Lightning)."""
        if self.synthetic:
            logger.info("Synthetic data mode enabled; skipping dataset file checks")
            return
        # If explicit CSV paths are provided, check them
        csv_paths = getattr(self.config.data, "csv_data_paths", None)
        if csv_paths:
            missing = [k for k, v in csv_paths.items() if not Path(v).exists()]
            if missing:
                msg = f"CSV paths missing for splits: {missing}"
                raise FileNotFoundError(msg)
            logger.info("CSV split files exist: train/val/test")
            return
        # Otherwise expect HDF5 file
        if not self.data_path.exists() or self.data_path.is_dir():
            msg = f"HDF5 file not found: {self.data_path}"
            raise FileNotFoundError(msg)
        logger.info(f"HDF5 file exists: {self.data_path}")

    def setup(self, stage: str | None = None) -> None:
        """Set up data loaders."""
        if self.synthetic:
            # Build simple synthetic noise dataset aligned to model input dims
            logger.info("Setting up synthetic dataloaders (noise-driven)")
            seq_len = int(self.synthetic_kwargs.get("seq_len", self.config.data.target_seq_len or 100))
            input_dim = int(self.synthetic_kwargs.get("input_dim", self._infer_first_layer_dim_safe()))
            num_classes = int(getattr(self.config.data, "num_classes", 1) or 1)
            task = str(self.synthetic_kwargs.get("task", "unsupervised"))  # 'unsupervised' | 'classification' | 'regression'
            dataset_size = int(self.synthetic_kwargs.get("dataset_size", 1024))
            val_size = max(1, int(self.config.data.val_split * dataset_size))
            test_size = max(1, int(self.config.data.test_split * dataset_size))
            train_size = max(1, dataset_size - val_size - test_size)

            class _Synthetic(torch.utils.data.Dataset):
                def __init__(self, n, seq_len, input_dim, task, num_classes) -> None:
                    self.n = n
                    self.seq_len = seq_len
                    self.input_dim = input_dim
                    self.task = task
                    self.num_classes = num_classes

                def __len__(self) -> int:
                    return self.n

                def __getitem__(self, idx):
                    x = torch.randn(self.seq_len, self.input_dim)
                    if self.task == "classification":
                        y = torch.randint(0, self.num_classes, (1,)).squeeze(0)
                    elif self.task == "regression":
                        y = torch.randn(self.input_dim)
                    else:
                        # unsupervised/self-supervised priming: provide a dummy scalar target
                        y = torch.tensor(0, dtype=torch.long)
                    return x, y

            ds_train = _Synthetic(train_size, seq_len, input_dim, task, num_classes)
            ds_val = _Synthetic(val_size, seq_len, input_dim, task, num_classes)
            ds_test = _Synthetic(test_size, seq_len, input_dim, task, num_classes)

            common = {"batch_size": self.batch_size, "num_workers": self.num_workers, "pin_memory": torch.cuda.is_available()}
            self.train_loader = DataLoader(ds_train, shuffle=True, **common)
            self.val_loader = DataLoader(ds_val, shuffle=False, **common)
            self.test_loader = DataLoader(ds_test, shuffle=False, **common)
            return
        try:
            # Use regular HDF5 loader
            from soen_toolkit.training.data.dataloaders import create_data_loaders

            self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                data_path=str(self.data_path),
                batch_size=self.batch_size,
                val_split=self.config.data.val_split,
                test_split=self.config.data.test_split,
                num_workers=self.num_workers,
                cache_data=self.config.data.cache_data,
                scale_min=self.config.data.min_scale,
                scale_max=self.config.data.max_scale,
                target_seq_len=self.config.data.target_seq_len,
                # One-hot encoding parameters
                input_encoding=self.config.data.input_encoding,
                vocab_size=self.config.data.vocab_size,
                one_hot_dtype=self.config.data.one_hot_dtype,
                persistent_workers=self.config.training.persistent_workers,
                prefetch_factor=self.config.training.prefetch_factor,
                multiprocessing_context=self.config.training.multiprocessing_context,
                accelerator=str(self.config.training.accelerator),
                # NEW: CSV split paths
                csv_data_paths={k: str(v) for k, v in (self.config.data.csv_data_paths or {}).items()} or None,
            )

            logger.info(f"Data loaders created successfully ({'CSV' if getattr(self.config.data, 'csv_data_paths', None) else 'HDF5'})")
        except Exception as e:
            logger.exception(f"Failed to create data loaders: {e}")
            raise

    def infer_num_classes(self) -> int:
        """Infer the number of classes from the HDF5 labels for classification tasks.

        For seq2seq classification (2D integer labels), counts unique values excluding padding.
        For standard classification (1D integer labels), returns the unique class count.
        For regression/non-integer labels, returns 0.
        """
        if self.synthetic:
            try:
                return int(getattr(self.config.data, "num_classes", 0) or 0)
            except Exception:
                return 0
        label_values: set[Any] = set()
        path_str = str(self.data_path)
        from soen_toolkit.training.data.dataloaders import open_hdf5_with_consistent_locking

        with open_hdf5_with_consistent_locking(path_str) as f:
            # If split groups exist, accumulate uniques across them; else use root
            groups = [g for g in ("train", "val", "test") if g in f]
            if groups:
                for g in groups:
                    if "labels" in f[g]:
                        arr = f[g]["labels"][:]
                        # Handle seq2seq classification: 2D integer labels [B, T]
                        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
                            # Flatten and exclude padding values (-100, negative values, or values >= some threshold)
                            arr_flat = arr.flatten()
                            # Filter out padding values (typically -100 or negative values)
                            valid_mask = arr_flat >= 0
                            if np.any(valid_mask):
                                uniques = np.unique(arr_flat[valid_mask].astype(np.int64))
                                label_values.update(int(x) for x in uniques.tolist())
                            else:
                                logger.warning("infer_num_classes: All labels in group '%s' appear to be padding. Skipping.", g)
                        # Handle standard classification: 1D integer labels
                        elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
                            try:
                                uniques = np.unique(arr.astype(np.int64))
                                label_values.update(int(x) for x in uniques.tolist())
                            except Exception:
                                uniques = np.unique(arr)
                                label_values.update(int(x) for x in uniques.tolist())
                        # Regression/non-integer: return 0
                        else:
                            logger.info("infer_num_classes: Detected non-classification labels in group '%s' (ndim=%d, dtype=%s); returning 0.", g, arr.ndim, arr.dtype)
                            return 0
            else:
                if "labels" not in f:
                    msg = "Could not find 'labels' dataset in HDF5 file to infer num_classes"
                    raise ValueError(msg)
                arr = f["labels"][:]
                # Handle seq2seq classification: 2D integer labels [B, T]
                if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
                    arr_flat = arr.flatten()
                    valid_mask = arr_flat >= 0
                    if np.any(valid_mask):
                        uniques = np.unique(arr_flat[valid_mask].astype(np.int64))
                        label_values.update(int(x) for x in uniques.tolist())
                    else:
                        logger.warning("infer_num_classes: All labels appear to be padding. Cannot infer num_classes.")
                # Handle standard classification: 1D integer labels
                elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
                    try:
                        uniques = np.unique(arr.astype(np.int64))
                        label_values.update(int(x) for x in uniques.tolist())
                    except Exception:
                        uniques = np.unique(arr)
                        label_values.update(int(x) for x in uniques.tolist())
                # Regression/non-integer: return 0
                else:
                    logger.info("infer_num_classes: Detected non-classification labels at root (ndim=%d, dtype=%s); returning 0.", arr.ndim, arr.dtype)
                    return 0

        inferred = len(label_values)
        # If zero (unexpected), fallback to config value
        if inferred == 0:
            logger.warning("infer_num_classes: Could not infer num_classes from dataset. Falling back to config value.")
            inferred = int(getattr(self.config.data, "num_classes", 0) or 0)
        else:
            logger.info("infer_num_classes: Inferred %d classes from dataset labels (unique values: %s)", inferred, sorted(label_values))
        return inferred

    def train_dataloader(self):
        """Get training data loader.

        Returns:
            DataLoader: Training data loader

        """
        return self.train_loader

    def val_dataloader(self):
        """Get validation data loader.

        Returns:
            DataLoader: Validation data loader

        """
        return self.val_loader

    def test_dataloader(self):
        """Get test data loader.

        Returns:
            DataLoader: Test data loader

        """
        return self.test_loader

    # ------------------------------------------------------------------
    # Dynamic sequence-length update helper (used by SeqLenScheduler)
    # ------------------------------------------------------------------

    def update_target_seq_len(self, new_len: int) -> None:
        """Re-create loaders with a new `target_seq_len`.

        Called by callbacks at runtime to change spectrogram down-sampling.
        """
        if new_len == self._current_target_seq_len:
            return  # nothing to do

        logger.info("[DataModule] Rebuilding dataloaders for target_seq_len=%d", new_len)
        self._current_target_seq_len = new_len
        self.config.data.target_seq_len = new_len

        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_path=str(self.data_path),
            batch_size=self.config.training.batch_size,
            val_split=self.config.data.val_split,
            test_split=self.config.data.test_split,
            num_workers=self.config.training.num_workers,
            cache_data=self.config.data.cache_data,
            scale_min=self.config.data.min_scale,
            scale_max=self.config.data.max_scale,
            target_seq_len=new_len,
            # One-hot encoding parameters (propagate exactly as in setup)
            input_encoding=self.config.data.input_encoding,
            vocab_size=self.config.data.vocab_size,
            one_hot_dtype=self.config.data.one_hot_dtype,
            persistent_workers=self.config.training.persistent_workers,
            prefetch_factor=self.config.training.prefetch_factor,
            multiprocessing_context=self.config.training.multiprocessing_context,
            # FIX: Pass CSV split paths when rebuilding loaders
            csv_data_paths={k: str(v) for k, v in (self.config.data.csv_data_paths or {}).items()} or None,
        )
        logger.info("[DataModule] Dataloaders rebuilt.")

    def _infer_first_layer_dim_safe(self) -> int:
        """Attempt to infer model input dimension from base model, fallback to 1."""
        try:
            from soen_toolkit.training.models import SOENLightningModule

            if getattr(self.config.model, "base_model_path", None):
                tmp = SOENLightningModule(self.config)
                first_layer_dim = tmp.model.layers_config[0].params.get("dim", 1)
                return int(first_layer_dim)
        except Exception:
            pass
        return 1
