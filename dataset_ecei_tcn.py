"""
ECEi TCN-style Dataset for disruption prediction.
Adapted from disruptcnn (Churchill et al. 2019) for KSTAR ECEi data.

Data layout
-----------
root/
  meta.csv          – columns: shot, split, t_disruption (ms), ...
  {shot}.h5         – HDF5 with key 'LFS' → (20, 8, T) float at 1 MHz

Preprocessing pipeline (applied in __getitem__)
------------------------------------------------
1. DC offset removal   – mean of first *baseline_length* samples per channel
2. Temporal decimation – keep every *data_step*-th sample (1 MHz → 100 kHz at step=10)
3. Z-score normalisation – per-channel (20×8) mean/std computed from training shots

Label construction
------------------
Per-timestep binary:
  0 (clear)      for time > Twarn before disruption
  1 (disruptive) for time ≤ Twarn before disruption
All shots in the dsrpt directory are disruptive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict


# ══════════════════════════════════════════════════════════════════════
#  Standalone helpers – useful for visualisation outside the Dataset
# ══════════════════════════════════════════════════════════════════════

def read_raw_shot(root: str | Path, shot: int) -> np.ndarray:
    """Read raw LFS array for one shot.  Returns (20, 8, T) float32."""
    with h5py.File(Path(root) / f'{shot}.h5', 'r') as f:
        return f['LFS'][...].astype(np.float32)


def remove_offset(data: np.ndarray, baseline_length: int = 50_000):
    """Subtract per-channel DC offset estimated from the first samples.

    Parameters
    ----------
    data : (20, 8, T)
    baseline_length : number of leading samples used for baseline

    Returns
    -------
    data_corrected : (20, 8, T)  offset-removed signal
    offset         : (20, 8)     the estimated offset per channel
    """
    baseline = data[..., :baseline_length]
    offset = np.mean(baseline, axis=-1, keepdims=True)
    return data - offset, offset.squeeze(-1)


def normalize_array(data: np.ndarray,
                    mean: np.ndarray,
                    std: np.ndarray) -> np.ndarray:
    """Z-score normalise *data* (…, T) using per-channel mean/std (…)."""
    return (data - mean[..., np.newaxis]) / std[..., np.newaxis]


def decimate(data: np.ndarray, step: int) -> np.ndarray:
    """Simple decimation by slicing every *step*-th sample along last axis."""
    if step <= 1:
        return data
    return data[..., ::step]


# ══════════════════════════════════════════════════════════════════════
#  Main Dataset
# ══════════════════════════════════════════════════════════════════════

class ECEiTCNDataset(Dataset):
    """
    PyTorch Dataset that yields fixed-length subsequences with
    per-timestep binary disruption labels.

    Returns
    -------
    X      : (20, 8, T_sub)   preprocessed signal  (float32)
    target : (T_sub,)          per-timestep label    (float32, 0 or 1)
    weight : (T_sub,)          per-timestep BCE weight (float32)
    """

    FS = 1_000_000  # 1 MHz native sample rate

    def __init__(
        self,
        root:             str,
        Twarn:            int   = 300_000,    # 300 ms in samples
        baseline_length:  int   = 50_000,     # 50 ms
        data_step:        int   = 10,         # → 100 kHz
        nsub:             int   = 500_000,    # 500 ms window
        stride:           int | None = None,  # default = nsub (no overlap)
        normalize:        bool  = True,
        label_balance:    str   = 'const',    # 'const' | 'none'
        norm_stats_path:  str | None = 'norm_stats.npz',  # path to cache
        norm_train_split: str   = 'train',
        norm_max_shots:   int   = 100,
    ):
        """
        Args (new, related to normalisation caching):
            norm_stats_path:  Where to save / load per-channel mean & std.
                              Set to None to skip automatic loading/computing.
            norm_train_split: Which split to use when computing stats.
            norm_max_shots:   Max number of shots sampled for stat estimation.
        """
        self.root             = Path(root)
        self.Twarn            = Twarn
        self.baseline_length  = baseline_length
        self.data_step        = data_step
        self.nsub             = nsub
        self.stride           = stride if stride is not None else nsub
        self.normalize        = normalize
        self.label_balance    = label_balance

        # ── metadata ──────────────────────────────────────────────────
        self.meta = pd.read_csv(self.root / 'meta.csv')
        self.shots   = self.meta['shot'].values.astype(int)
        self.splits  = self.meta['split'].values.astype(str)
        # t_disruption is in ms → samples at 1 MHz
        self.t_dis   = (self.meta['t_disruption'].values * 1000).astype(int)

        # index where label flips 0 → 1
        self.disrupt_idx = self.t_dis - self.Twarn

        # usable window: [baseline_length, t_dis)
        self.start_idx = np.full(len(self.shots), self.baseline_length)
        self.stop_idx  = self.t_dis.copy()

        # normalisation – auto load / compute / save
        self.norm_mean: Optional[np.ndarray] = None   # (20, 8)
        self.norm_std:  Optional[np.ndarray] = None   # (20, 8)

        if self.normalize and norm_stats_path is not None:
            p = Path(norm_stats_path)
            if p.exists():
                self.load_norm_stats(str(p))
                print(f'[ECEiTCNDataset] Loaded norm stats from {p}')
            else:
                self.compute_norm_stats(split=norm_train_split,
                                        max_shots=norm_max_shots)
                self.save_norm_stats(str(p))
                print(f'[ECEiTCNDataset] Computed & saved norm stats to {p}')

        # subsequences & class weights
        self._build_subsequences()
        self._compute_class_weights()

    # ── subsequence tiling ────────────────────────────────────────────

    def _build_subsequences(self):
        """Tile each shot into fixed-length windows of *nsub* samples."""
        shot_idx, starts, stops, d_local = [], [], [], []

        for s in range(len(self.shots)):
            a, b = int(self.start_idx[s]), int(self.stop_idx[s])
            if b - a < self.nsub:
                continue                          # shot too short

            d = int(self.disrupt_idx[s])          # absolute disrupt index

            pos = a
            while pos + self.nsub <= b:
                shot_idx.append(s)
                starts.append(pos)
                stops.append(pos + self.nsub)

                # local disrupt offset inside this window
                if d <= pos:
                    d_local.append(0)             # fully disruptive
                elif d >= pos + self.nsub:
                    d_local.append(-1)            # fully clear
                else:
                    d_local.append(d - pos)       # transition inside

                pos += self.stride

            # snap last window to the end of the shot
            last_start = b - self.nsub
            if last_start > (pos - self.stride):
                shot_idx.append(s)
                starts.append(last_start)
                stops.append(b)
                if d <= last_start:
                    d_local.append(0)
                elif d >= b:
                    d_local.append(-1)
                else:
                    d_local.append(d - last_start)

        self.seq_shot_idx     = np.array(shot_idx,  dtype=int)
        self.seq_start        = np.array(starts,    dtype=int)
        self.seq_stop         = np.array(stops,     dtype=int)
        self.seq_disrupt_local = np.array(d_local,  dtype=int)
        self.seq_has_disrupt  = self.seq_disrupt_local >= 0

    # ── class weights ─────────────────────────────────────────────────

    def _compute_class_weights(self):
        T = self.nsub // self.data_step
        total_pos, total_neg = 0, 0
        for dl in self.seq_disrupt_local:
            if dl < 0:
                total_neg += T
            elif dl == 0:
                total_pos += T
            else:
                d = dl // self.data_step
                total_neg += d
                total_pos += T - d
        total = total_pos + total_neg
        if self.label_balance == 'const' and total_pos > 0 and total_neg > 0:
            self.pos_weight = float(0.5 * total / total_pos)
            self.neg_weight = float(0.5 * total / total_neg)
        else:
            self.pos_weight = 1.0
            self.neg_weight = 1.0

    # ── normalisation statistics ──────────────────────────────────────

    def compute_norm_stats(self, split: str = 'train',
                           max_shots: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-channel mean/std from *split* shots (online Welford)."""
        mask    = self.splits == split
        indices = np.where(mask)[0]
        if len(indices) > max_shots:
            indices = np.random.choice(indices, max_shots, replace=False)

        running_sum    = np.zeros((20, 8), dtype=np.float64)
        running_sq_sum = np.zeros((20, 8), dtype=np.float64)
        n_total = 0

        for s in tqdm(indices, desc=f'Norm stats ({split})'):
            shot = self.shots[s]
            with h5py.File(self.root / f'{shot}.h5', 'r') as f:
                raw = f['LFS'][..., self.start_idx[s]:self.stop_idx[s]]
                baseline = f['LFS'][..., :self.baseline_length]

            offset = np.mean(baseline, axis=-1, keepdims=True)
            data = (raw - offset).astype(np.float64)

            T = data.shape[-1]
            running_sum    += data.sum(axis=-1)
            running_sq_sum += (data ** 2).sum(axis=-1)
            n_total += T

        mean = (running_sum / n_total).astype(np.float32)
        var  = (running_sq_sum / n_total) - mean.astype(np.float64) ** 2
        std  = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)

        self.norm_mean = mean
        self.norm_std  = std
        return mean, std

    def save_norm_stats(self, path: str = 'norm_stats.npz'):
        np.savez(path, mean=self.norm_mean, std=self.norm_std)

    def load_norm_stats(self, path: str = 'norm_stats.npz'):
        f = np.load(path)
        self.norm_mean = f['mean']
        self.norm_std  = f['std']

    # ── split helpers ─────────────────────────────────────────────────

    def get_split_indices(self, split: str) -> np.ndarray:
        """Return subsequence indices that belong to *split*."""
        split_shots = set(np.where(self.splits == split)[0])
        return np.array([i for i, s in enumerate(self.seq_shot_idx)
                         if s in split_shots])

    def summary(self) -> Dict:
        """Print a human-readable summary of the dataset."""
        info = {}
        for sp in np.unique(self.splits):
            idx = self.get_split_indices(sp)
            n_dis = int(self.seq_has_disrupt[idx].sum())
            info[sp] = {'n_subseqs': len(idx),
                        'n_disruptive': n_dis,
                        'n_clear': len(idx) - n_dis}
            print(f"  {sp:>5s}: {len(idx):5d} subseqs  "
                  f"({n_dis} disruptive, {len(idx)-n_dis} clear)")
        T_sub = self.nsub // self.data_step
        print(f"  Subseq length after decimation: {T_sub:,} samples "
              f"({self.nsub/self.FS*1e3:.0f} ms raw, step={self.data_step})")
        print(f"  pos_weight={self.pos_weight:.3f}  neg_weight={self.neg_weight:.3f}")
        return info

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self):
        return len(self.seq_shot_idx)

    def __getitem__(self, index):
        s    = int(self.seq_shot_idx[index])
        shot = self.shots[s]

        with h5py.File(self.root / f'{shot}.h5', 'r') as f:
            X = f['LFS'][..., self.seq_start[index]:self.seq_stop[index]].astype(np.float32)
            baseline = f['LFS'][..., :self.baseline_length].astype(np.float32)

        # 1. offset removal
        offset = np.mean(baseline, axis=-1, keepdims=True)
        X -= offset

        # 2. temporal decimation
        if self.data_step > 1:
            X = X[..., ::self.data_step]

        # 3. normalisation
        if self.normalize and self.norm_mean is not None:
            X = (X - self.norm_mean[..., np.newaxis]) / self.norm_std[..., np.newaxis]

        # 4. per-timestep label & weight
        T = X.shape[-1]
        target = np.zeros(T, dtype=np.float32)
        weight = np.full(T, self.neg_weight, dtype=np.float32)

        dl = int(self.seq_disrupt_local[index])
        if dl >= 0:
            d = min(dl // self.data_step, T)
            target[d:] = 1.0
            weight[d:] = self.pos_weight

        return (torch.from_numpy(np.ascontiguousarray(X)),
                torch.from_numpy(target),
                torch.from_numpy(weight))


# ══════════════════════════════════════════════════════════════════════
#  DataLoader factory
# ══════════════════════════════════════════════════════════════════════

def create_loaders(
    dataset: ECEiTCNDataset,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for every split present in meta.csv."""
    loaders = {}
    for split in np.unique(dataset.splits):
        idx = dataset.get_split_indices(split)
        if len(idx) == 0:
            continue
        subset = Subset(dataset, idx)
        loaders[split] = DataLoader(
            subset,
            batch_size  = batch_size,
            shuffle     = (split == 'train'),
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = (split == 'train'),
        )
    return loaders
