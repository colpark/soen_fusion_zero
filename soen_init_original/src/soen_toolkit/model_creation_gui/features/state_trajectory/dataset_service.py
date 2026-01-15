"""Dataset service for loading and managing HDF5 datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import DatasetServiceError
from .settings import TaskType


class DatasetService:
    """Service for loading and managing datasets.

    Provides:
    - Lazy loading of HDF5 datasets
    - Group detection (train/val/test splits)
    - Sample indexing by class
    - RNG for deterministic input generation
    """

    def __init__(self):
        self._dataset = None
        self._dataset_path: Path | None = None
        self._current_group: str | None = None
        # When datasets are very large, avoid materializing labels in memory
        self._labels: list | None = None
        self._num_samples: int = 0
        self._classes_cache: list[int] | None = None
        self._rng = np.random.default_rng(1337)  # Default seed

    def is_loaded(self) -> bool:
        """Check if dataset is currently loaded."""
        return self._dataset is not None

    def get_dataset(self):
        """Get the loaded dataset object."""
        return self._dataset

    def get_current_path(self) -> Path | None:
        """Get path of currently loaded dataset."""
        return self._dataset_path

    def get_current_group(self) -> str | None:
        """Get currently selected group."""
        return self._current_group

    def rng(self) -> np.random.Generator:
        """Get RNG for deterministic generation."""
        return self._rng

    def set_seed(self, seed: int) -> None:
        """Set RNG seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def detect_groups(self, path: Path) -> tuple[list[str], bool, bool]:
        """Detect available groups in HDF5 file.

        Args:
            path: Path to HDF5 file

        Returns:
            Tuple of (group_names, has_root_data, is_seq2seq)

        Raises:
            DatasetServiceError: If file cannot be read
        """
        try:
            import h5py

            from soen_toolkit.training.data.dataloaders import open_hdf5_with_consistent_locking
        except ImportError as e:
            raise DatasetServiceError("h5py not available") from e

        if not path.exists():
            raise DatasetServiceError(f"Dataset file not found: {path}")

        try:
            with open_hdf5_with_consistent_locking(str(path)) as f:
                # Find groups with data and labels
                group_names = {k for k in f if isinstance(f[k], h5py.Group)}
                valid_groups = []

                for group_name in group_names:
                    try:
                        grp = f[group_name]
                        if "data" in grp and "labels" in grp:
                            valid_groups.append(group_name)
                    except Exception:
                        continue

                # Check if root level has data
                has_root_data = "data" in f and "labels" in f

                # Auto-detect task type from labels
                is_seq2seq = False
                try:
                    tgt = None
                    if valid_groups:
                        preferred = "train" if "train" in valid_groups else valid_groups[0]
                        if "labels" in f[preferred]:
                            tgt = f[preferred]["labels"]
                    elif has_root_data:
                        tgt = f["labels"]

                    if tgt is not None:
                        is_int = np.issubdtype(tgt.dtype, np.integer)
                        is_1d = tgt.ndim == 1
                        is_seq2seq = not (is_int and is_1d)
                except Exception:
                    pass

                return sorted(valid_groups), has_root_data, is_seq2seq

        except Exception as e:
            raise DatasetServiceError(f"Failed to read HDF5 file: {e}") from e

    def load(self, path: Path, group: str | None = None, seed: int | None = None) -> None:
        """Load dataset from HDF5 file.

        Args:
            path: Path to HDF5 file
            group: Optional group name (None for root level)
            seed: Optional seed for RNG

        Raises:
            DatasetServiceError: If loading fails
        """
        try:
            from soen_toolkit.training.data.dataloaders import GenericHDF5Dataset
        except ImportError as e:
            raise DatasetServiceError("GenericHDF5Dataset not available") from e

        if not path.exists():
            raise DatasetServiceError(f"Dataset file not found: {path}")

        try:
            self._dataset = GenericHDF5Dataset(
                hdf5_path=str(path),
                split=group,
                cache_in_memory=False,
                input_encoding="raw",  # Default to raw, encoding applied by InputSource
                vocab_size=None,
                one_hot_dtype="float32",
            )
            self._dataset_path = path
            self._current_group = group

            # Load lightweight label metadata for indexing
            self._load_labels()

            # Set seed if provided
            if seed is not None:
                self.set_seed(seed)

        except Exception as e:
            raise DatasetServiceError(f"Failed to load dataset: {e}") from e

    def _load_labels(self) -> None:
        """Load lightweight label metadata for indexing.

        Avoids materializing full label arrays for large datasets. For small datasets,
        stores labels in-memory; for large ones, stores only counts and a small
        class sample cache for UI population.
        """
        self._labels = None
        self._classes_cache = None
        self._num_samples = 0
        try:
            from soen_toolkit.training.data.dataloaders import open_hdf5_with_consistent_locking

            with open_hdf5_with_consistent_locking(str(self._dataset_path)) as f:
                labels_ds = f[self._current_group]["labels"] if self._current_group else f["labels"]
                n = int(labels_ds.shape[0])
                self._num_samples = n

                # Small dataset: keep labels in memory for fast class filtering
                if n <= 10000:
                    self._labels = labels_ds[:].tolist()
                    try:
                        self._classes_cache = sorted({int(lbl) for lbl in self._labels if isinstance(lbl, (int, np.integer))})
                    except Exception:
                        self._classes_cache = None
                else:
                    # Large dataset: compute a quick unique set from a small prefix sample
                    try:
                        sample_size = min(100000, n)
                        sample = np.array(labels_ds[:sample_size])
                        # Keep only integer-like classes
                        uniques = np.unique(sample[np.isfinite(sample)]) if sample.dtype.kind in ("i", "u", "f") else []
                        ints = [int(x) for x in uniques if float(x).is_integer()]
                        self._classes_cache = sorted(set(ints)) if ints else None
                    except Exception:
                        self._classes_cache = None

        except Exception:
            # Fallback: use dataset object length
            try:
                self._num_samples = len(self._dataset)
            except Exception:
                self._num_samples = 0

    def get_indices_for_class(self, class_id: int | None, task_type: TaskType) -> list[int]:
        """Get indices of samples for a given class.

        Args:
            class_id: Class identifier (for classification)
            task_type: Task type (classification or seq2seq)

        Returns:
            List of dataset indices
        """
        n = self._num_samples if self._num_samples > 0 else (len(self._labels) if self._labels is not None else len(self._dataset) if self._dataset is not None else 0)
        if task_type == TaskType.SEQ2SEQ:
            # For seq2seq/regression, all indices are valid; return a lazy range
            return range(n)
        # Classification
        if class_id is None:
            return range(n)
        if self._labels is None:
            # Large dataset: avoid expensive full scan; allow any index
            return range(n)
        try:
            indices = [i for i, lbl in enumerate(self._labels) if int(lbl) == int(class_id)]
            return indices if indices else range(n)
        except (ValueError, TypeError):
            return range(n)

    def get_available_classes(self) -> list[int]:
        """Get list of unique class IDs in dataset.

        Returns:
            Sorted list of unique class identifiers
        """
        # Prefer cached classes (fast path)
        if self._classes_cache:
            return self._classes_cache
        # If labels are in-memory, compute uniques
        if self._labels is not None:
            try:
                unique = sorted({int(lbl) for lbl in self._labels if isinstance(lbl, (int, np.integer))})
                return unique if unique else [0]
            except (ValueError, TypeError):
                return [0]
        # Unknown: fall back to a conservative default
        return [0]

    def get_sample_count(self) -> int:
        """Get total number of samples in dataset."""
        if self._num_samples > 0:
            return self._num_samples
        if self._labels is not None:
            return len(self._labels)
        try:
            return len(self._dataset) if self._dataset is not None else 0
        except Exception:
            return 0
