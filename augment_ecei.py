"""
Data augmentation for ECEi TCN training, with options to focus on the disruptive segment.

- Gaussian noise: optional stronger noise on the disruptive (label=1) region to reduce overfitting to exact amplitudes.
- Amplitude scaling: random global scale (e.g. 0.9–1.1).
- Boundary jitter: randomly shift the clear/disruptive boundary by ± a few samples so the model does not overfit to the exact Twarn boundary.

Use via AugmentWrapper(dataset, config, train=True) so augmentation is applied only at training time.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def apply_augment(
    X: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply augmentations to one sample (in-place style; copies are made where needed).

    X: (C, T), target: (T,), weight: (T).
    config: dict with optional keys
      - noise_std: float, sigma for Gaussian noise on X (default 0 → no noise)
      - noise_std_disrupt: float, extra sigma added only where target==1 (default 0)
      - scale_range: (low, high) for uniform scale factor (default None → no scale)
      - boundary_jitter: int, max samples to shift clear/disrupt boundary (default 0)
    """
    X = np.asarray(X, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)

    noise_std = config.get("noise_std", 0.0)
    noise_std_disrupt = config.get("noise_std_disrupt", 0.0)
    scale_range = config.get("scale_range")
    boundary_jitter = config.get("boundary_jitter", 0)

    # 1. Gaussian noise (optionally stronger on disruptive part)
    if noise_std > 0 or noise_std_disrupt > 0:
        noise = rng.normal(0, noise_std, size=X.shape).astype(np.float32)
        if noise_std_disrupt > 0 and target.sum() > 0:
            disrupt_mask = target > 0.5  # (T,)
            extra = rng.normal(0, noise_std_disrupt, size=X.shape).astype(np.float32)
            noise += extra * disrupt_mask[np.newaxis, :]
        X = X + noise

    # 2. Amplitude scaling
    if scale_range is not None:
        low, high = scale_range
        scale = rng.uniform(low, high)
        X = X * scale

    # 3. Boundary jitter: shift the clear/disrupt boundary by ± boundary_jitter samples
    if boundary_jitter > 0 and target.sum() > 0:
        T = target.shape[0]
        first_disrupt = int(np.argmax(target > 0.5))
        neg_w = float(weight[0])
        pos_w = float(weight[first_disrupt]) if first_disrupt < T else neg_w
        shift = rng.integers(-boundary_jitter, boundary_jitter + 1)
        new_first = np.clip(first_disrupt + shift, 0, T)
        target = np.zeros(T, dtype=np.float32)
        target[new_first:] = 1.0
        weight = np.full(T, neg_w, dtype=np.float32)
        if new_first < T:
            weight[new_first:] = pos_w

    return X, target, weight


class AugmentWrapper(Dataset):
    """
    Wraps a dataset and applies augmentation in __getitem__ when train=True and config is set.

    Preserves seq_has_disrupt, get_split_indices, pos_weight, neg_weight by delegating to the inner dataset.
    """

    def __init__(
        self,
        inner: Dataset,
        config: Optional[dict] = None,
        train: bool = True,
        seed: int = 42,
    ):
        self._inner = inner
        self._config = config or {}
        self._train = train
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int):
        item = self._inner[index]
        if not self._train or not self._config:
            return item
        X, target, weight = item
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        if isinstance(weight, torch.Tensor):
            weight = weight.cpu().numpy()
        rng = np.random.default_rng(self._seed + self._epoch * 100000 + index)
        X, target, weight = apply_augment(X, target, weight, self._config, rng)
        return (
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(weight.astype(np.float32)),
        )

    @property
    def seq_has_disrupt(self):
        return self._inner.seq_has_disrupt

    def get_split_indices(self, split: str):
        return self._inner.get_split_indices(split)

    @property
    def pos_weight(self) -> float:
        return self._inner.pos_weight

    @property
    def neg_weight(self) -> float:
        return self._inner.neg_weight

    def _compute_class_weights(self, indices: Optional[Any] = None):
        if hasattr(self._inner, "_compute_class_weights"):
            self._inner._compute_class_weights(indices)
