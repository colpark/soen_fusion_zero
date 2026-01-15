import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetStats:
    """Statistics for a dataset or tensor."""

    def __init__(
        self,
        shape: tuple,
        dtype: str,
        min_val: float,
        max_val: float,
        mean: float,
        std: float,
        num_samples: int,
        num_features: int,
        missing_values: int,
        unique_values: int,
        class_distribution: dict[Any, int] | None = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.mean = mean
        self.std = std
        self.num_samples = num_samples
        self.num_features = num_features
        self.missing_values = missing_values
        self.unique_values = unique_values
        self.class_distribution = class_distribution


class DatasetProcessor:
    """Robust analyzer for ML datasets that handles various input types and structures.
    This version can detect one-hot encoded labels and convert them into class indices.
    """

    def __init__(
        self,
        dataset: Dataset | torch.Tensor | np.ndarray | list,
        batch_size: int = 32,
        num_workers: int = 0,
        cache_dir: str | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        # stats will be computed on demand
        self.stats: DatasetStats | None = None

    def _convert_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert labels to integer class indices if they are one-hot encoded."""
        if labels.dim() > 1 and labels.size(-1) > 1:
            # Check if one-hot encoded: the sum along last dim should be 1 for each sample
            if torch.all(torch.sum(labels, dim=-1) == 1):
                labels = torch.argmax(labels, dim=-1)
        return labels

    def _compute_dataset_stats(self) -> DatasetStats:
        """Compute comprehensive dataset statistics."""
        # Get a sample batch to determine the structure
        sample_batch = next(iter(self.dataloader))
        if isinstance(sample_batch, (tuple, list)):
            has_labels = True
            features_sample = sample_batch[0]
            sample_batch[1]
        else:
            has_labels = False
            features_sample = sample_batch

        total_samples = 0
        total_sum = 0.0
        total_sq_sum = 0.0
        min_val = float("inf")
        max_val = float("-inf")
        all_labels = []

        # Process each batch
        for batch in tqdm(self.dataloader, desc="Computing statistics"):
            if has_labels:
                features, labels = batch
                labels = self._convert_labels(labels)
                all_labels.append(labels)
            else:
                features = batch

            features_float = features.float()
            batch_flat = features_float.view(-1)

            total_sum += batch_flat.sum().item()
            total_sq_sum += (batch_flat**2).sum().item()
            min_val = min(min_val, batch_flat.min().item())
            max_val = max(max_val, batch_flat.max().item())
            total_samples += batch_flat.numel()

        mean = total_sum / total_samples
        variance = (total_sq_sum / total_samples) - (mean**2)
        std = np.sqrt(variance + 1e-8)

        # Compute class distribution if applicable
        if has_labels:
            labels_cat = torch.cat(all_labels, dim=0)
            # Ensure labels are one-dimensional integer tensors
            labels_cat = self._convert_labels(labels_cat)
            unique_labels = torch.unique(labels_cat)
            class_dist = {int(label): int((labels_cat == label).sum()) for label in unique_labels}
        else:
            class_dist = None

        # Count missing values
        num_missing = 0
        for batch in self.dataloader:
            features = batch[0] if has_labels else batch
            num_missing += torch.isnan(features).sum().item()

        # Count unique values (sample limited)
        sample_size = min(10000, total_samples)
        unique_values = set()
        processed_elements = 0

        for batch in self.dataloader:
            features = batch[0] if has_labels else batch
            flat_features = features.view(-1)
            remaining = sample_size - processed_elements
            if remaining <= 0:
                break
            elements_to_process = min(remaining, flat_features.numel())
            unique_values.update(flat_features[:elements_to_process].tolist())
            processed_elements += elements_to_process

        return DatasetStats(
            shape=tuple(features_sample.shape),
            dtype=str(features_sample.dtype),
            min_val=min_val,
            max_val=max_val,
            mean=mean,
            std=std,
            num_samples=len(self.dataset),
            num_features=int(np.prod(features_sample.shape[1:])),
            missing_values=num_missing,
            unique_values=len(unique_values),
            class_distribution=class_dist,
        )

    def compute_stats(self) -> None:
        """Computes and stores dataset statistics."""
        self.stats = self._compute_dataset_stats()

    def print_summary(self) -> None:
        """Print comprehensive dataset summary."""
        if self.stats is None:
            msg = "Statistics have not been computed yet. Call 'compute_stats()' first."
            raise RuntimeError(msg)

        if self.stats.class_distribution:
            total = sum(self.stats.class_distribution.values())
            for _label, count in sorted(self.stats.class_distribution.items()):
                (count / total) * 100

    def time_stretch(self, stretch_factor: float = 1.0, method: str = "interpolate", time_dim: int = 1) -> Dataset:
        """Stretch the time dimension of the data."""
        if stretch_factor <= 0:
            msg = "stretch_factor must be positive"
            raise ValueError(msg)

        if stretch_factor == 1.0:
            return self.dataset

        all_stretched = []
        all_labels = []

        for batch in tqdm(self.dataloader, desc=f"Time stretching (factor={stretch_factor:.2f})"):
            if isinstance(batch, (tuple, list)):
                features, labels = batch
                all_labels.append(labels)
            else:
                features = batch

            orig_shape = list(features.shape)
            new_time_len = int(orig_shape[time_dim] * stretch_factor)

            if method == "interpolate":
                if features.dim() > 3:
                    features = features.view(-1, orig_shape[time_dim], features.shape[-1])
                stretched = torch.nn.functional.interpolate(
                    features.transpose(1, 2),
                    size=new_time_len,
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
            elif method == "repeat":
                indices = torch.linspace(0, orig_shape[time_dim] - 1, new_time_len).long()
                stretched = features.index_select(time_dim, indices)
            elif method == "nearest":
                stretched = torch.nn.functional.interpolate(
                    features.transpose(1, 2),
                    size=new_time_len,
                    mode="nearest",
                ).transpose(1, 2)
            else:
                msg = f"Unknown stretching method: {method}"
                raise ValueError(msg)

            if features.dim() > 3:
                stretched = stretched.view(orig_shape[0], -1, new_time_len, features.shape[-1])

            all_stretched.append(stretched)

        stretched_features = torch.cat(all_stretched, dim=0)

        if all_labels:
            labels = torch.cat(all_labels, dim=0)
            return torch.utils.data.TensorDataset(stretched_features, labels)
        return torch.utils.data.TensorDataset(stretched_features)

    def plot_time_stretch_comparison(self, sample_index: int = 0, stretch_factor: float = 2.0, method: str = "interpolate", figsize: tuple = (12, 4)) -> plt.Figure:
        """Create a side-by-side plot comparing original and time-stretched features."""
        sample = self.dataset[sample_index]
        if isinstance(sample, (tuple, list)):
            original_features, label = sample
        else:
            original_features, label = sample, None

        stretched_dataset = self.time_stretch(stretch_factor, method)
        stretched_sample = stretched_dataset[sample_index]
        if isinstance(stretched_sample, (tuple, list)):
            stretched_features, _ = stretched_sample
        else:
            stretched_features = stretched_sample

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        vmin = min(original_features.min(), stretched_features.min())
        vmax = max(original_features.max(), stretched_features.max())

        im1 = ax1.imshow(original_features.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Original Features\nShape: {tuple(original_features.shape)}")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Mel Frequency Channels")

        ax2.imshow(stretched_features.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax2.set_title(f"Stretched Features ({method})\nShape: {tuple(stretched_features.shape)}")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Mel Frequency Channels")

        fig.colorbar(im1, ax=[ax1, ax2], label="Feature Value")
        label_str = f" | Label: {label}" if label is not None else ""
        plt.suptitle(f"Time Stretch Comparison (factor={stretch_factor}){label_str}")
        plt.tight_layout()
        return fig

    def normalize_data(self, method: str = "standard") -> Dataset:
        """Normalize the dataset using specified method.

        Methods:
        - 'standard': zero mean, unit variance
        - 'minmax': scale to [0, 1]
        - 'robust': using median and IQR
        Returns a TensorDataset with normalized features.

        """
        all_labels = []

        total_sum = 0.0
        total_sq_sum = 0.0
        min_val = float("inf")
        max_val = float("-inf")
        total_samples = 0

        for batch in tqdm(self.dataloader, desc=f"Computing {method} normalization parameters"):
            if isinstance(batch, (tuple, list)):
                features, labels = batch
                all_labels.append(labels)
            else:
                features = batch

            features_float = features.float()
            batch_flat = features_float.view(-1)
            total_sum += batch_flat.sum().item()
            total_sq_sum += (batch_flat**2).sum().item()
            min_val = min(min_val, batch_flat.min().item())
            max_val = max(max_val, batch_flat.max().item())
            total_samples += batch_flat.numel()

        mean = total_sum / total_samples
        std = np.sqrt((total_sq_sum / total_samples) - (mean**2) + 1e-8)

        normalized_data = []
        for batch in tqdm(self.dataloader, desc="Normalizing data"):
            if isinstance(batch, (tuple, list)):
                features, _ = batch
            else:
                features = batch

            if method == "standard":
                features_norm = (features - mean) / std
            elif method == "minmax":
                features_norm = (features - min_val) / (max_val - min_val + 1e-8)
            elif method == "robust":
                q75 = np.percentile(features.numpy(), 75)
                q25 = np.percentile(features.numpy(), 25)
                iqr = q75 - q25 + 1e-8
                features_norm = (features - ((q75 + q25) / 2)) / iqr
            else:
                msg = f"Unknown normalization method: {method}"
                raise ValueError(msg)
            normalized_data.append(features_norm)

        normalized_features = torch.cat(normalized_data, dim=0)
        if all_labels:
            labels = torch.cat(all_labels, dim=0)
            return torch.utils.data.TensorDataset(normalized_features, labels)
        return torch.utils.data.TensorDataset(normalized_features)

    def normalize_to_range(self, target_min: float = 0.0, target_max: float = 0.5) -> Dataset:
        """Normalize the dataset to a specific range [target_min, target_max].
        Internally performs minmax normalization to [0,1] and then scales to the target range.
        Returns a TensorDataset with the scaled features.
        """
        # First, normalize to [0,1] using minmax
        norm_dataset = self.normalize_data(method="minmax")
        if not hasattr(norm_dataset, "tensors"):
            msg = "Expected a TensorDataset from normalize_data()"
            raise RuntimeError(msg)
        norm_features = norm_dataset.tensors[0]
        # Scale to [target_min, target_max]
        scaled_features = norm_features * (target_max - target_min) + target_min
        if len(norm_dataset.tensors) > 1:
            labels = norm_dataset.tensors[1]
            return torch.utils.data.TensorDataset(scaled_features, labels)
        return torch.utils.data.TensorDataset(scaled_features)
