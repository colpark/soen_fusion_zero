"""
ECEi data for WaveStitch-style conditioning.
Same as diffusion/data/dataset.py: (x, class_id, t_disrupt_cond).
- x: (C, T) e.g. (160, T) or (20, 8, T), flattened to 160 in train.
- class_id: 0 = clear, 1 = disruption.
- t_disrupt_cond: [0, 1] normalised time of (disruption start − 300 ms); 0 for clear.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_ragged_mmap(path: Path):
    try:
        from mmap_ninja import RaggedMmap
        return RaggedMmap(str(path))
    except ImportError as e:
        raise ImportError("Install mmap_ninja: pip install mmap-ninja") from e


class DecimatedEceiMmapDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        decimate_factor: int = 10,
        split: str = "train",
        twarn_ms: float = 300.0,
        dt_ms_per_decimated_step: float = 0.1,
    ):
        self._root = Path(root)
        self._decimate = max(1, int(decimate_factor))
        self._split = split
        self._twarn_ms = twarn_ms
        self._dt_ms = dt_ms_per_decimated_step
        if not self._root.exists():
            raise FileNotFoundError(f"Prebuilt mmap dir not found: {self._root}")

        self._X = _load_ragged_mmap(self._root / "X")
        self._target = _load_ragged_mmap(self._root / "target")
        self._labels = np.load(self._root / "labels.npy")
        fname = {"train": "train_inds.npy", "val": "val_inds.npy", "test": "test_inds.npy"}.get(split, "train_inds.npy")
        inds_path = self._root / fname
        self._inds = np.load(inds_path) if inds_path.exists() else np.arange(len(self._X))

    def __len__(self) -> int:
        return len(self._inds)

    def _first_disrupt_index_decimated(self, target_decimated: np.ndarray) -> int:
        pos = np.where(np.asarray(target_decimated).ravel() >= 0.5)[0]
        if len(pos) == 0:
            return -1
        return int(pos[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, float]:
        i = self._inds[index]
        x = np.ascontiguousarray(self._X[i]).astype(np.float32)
        target = np.asarray(self._target[i], dtype=np.float32)
        if self._decimate > 1:
            x = x[..., ::self._decimate].copy()
            target = target[::self._decimate]

        T = x.shape[-1]
        is_disrupt = bool(self._labels[i]) if i < len(self._labels) else (target.max() >= 0.5)
        class_id = 1 if is_disrupt else 0

        if is_disrupt:
            first_idx = self._first_disrupt_index_decimated(target)
            if first_idx < 0:
                t_disrupt_cond = 0.0
            else:
                t_minus_twarn_steps = (first_idx * self._dt_ms - self._twarn_ms) / self._dt_ms
                t_minus_twarn_steps = max(0.0, t_minus_twarn_steps)
                t_disrupt_cond = float(t_minus_twarn_steps / max(T, 1))
                t_disrupt_cond = max(0.0, min(1.0, t_disrupt_cond))
        else:
            t_disrupt_cond = 0.0

        return torch.from_numpy(x), class_id, t_disrupt_cond
