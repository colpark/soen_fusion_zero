"""
ECEi TCN-style Dataset for disruption prediction.
Adapted from disruptcnn (Churchill et al. 2019) for KSTAR ECEi data.

Data layout
-----------
root/
  meta.csv          – columns: shot, split, t_disruption (ms), ...
  {shot}.h5         – HDF5 with key 'LFS' → (20, 8, T) float at 1 MHz

Optionally, a *decimated_root* directory can hold pre-processed files:
  decimated_root/
    meta.csv        – same as root
    {shot}.h5       – offset-removed & decimated: (20, 8, T/data_step)

When decimated_root exists the dataset reads from it directly, skipping
offset removal and decimation in __getitem__ for much faster training.

Preprocessing pipeline (applied in __getitem__ when reading raw data)
---------------------------------------------------------------------
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


def remove_offset(data: np.ndarray, baseline_length: int = 40_000):
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
        Twarn:            int   = 300_000,    # 300 ms in samples (1 MHz)
        baseline_length:  int   = 40_000,     # 40 ms  (1 MHz), matches disruptcnn
        data_step:        int   = 10,         # → 100 kHz
        nsub:             int   = 781_250,    # ~781 ms window (matches disruptcnn run.sh)
        stride:           int   = 481_090,    # (nsub/step - nrecept + 1) * step = (78125-30017+1)*10
        normalize:        bool  = True,
        label_balance:    str   = 'const',    # 'const' | 'none'
        norm_stats_path:  str | None = 'norm_stats.npz',
        norm_train_split: str   = 'train',
        norm_max_shots:   int   = 100,
        decimated_root:   str | None = None,
    ):
        """
        Args:
            root:             Directory with meta.csv and raw {shot}.h5 files.
            decimated_root:   Directory with pre-decimated h5 files (offset-
                              removed & decimated).  If it exists the dataset
                              reads from here, skipping offset removal and
                              decimation for faster I/O.
            norm_stats_path:  Where to save / load per-channel mean & std.
                              Set to None to skip automatic loading/computing.
            norm_train_split: Which split to use when computing stats.
            norm_max_shots:   Max number of shots sampled for stat estimation.
        All time parameters (Twarn, baseline_length, nsub, stride) are
        specified in **raw 1 MHz samples** regardless of whether decimated
        data is used — the class converts internally.
        """
        self.root             = Path(root)
        self.Twarn            = Twarn
        self.baseline_length  = baseline_length
        self.data_step        = data_step
        self.nsub             = nsub
        self.stride           = stride if stride is not None else nsub
        self.normalize        = normalize
        self.label_balance    = label_balance

        # ── check for pre-decimated data ──────────────────────────────
        self._use_decimated = False
        self._decimated_root: Optional[Path] = None
        if decimated_root is not None:
            p = Path(decimated_root)
            if p.exists() and (p / 'meta.csv').exists():
                self._use_decimated = True
                self._decimated_root = p
                print(f'[ECEiTCNDataset] Using pre-decimated data '
                      f'from {p}')

        # ── metadata ──────────────────────────────────────────────────
        self.meta = pd.read_csv(self.root / 'meta.csv')
        self.shots   = self.meta['shot'].values.astype(int)
        self.splits  = self.meta['split'].values.astype(str)

        # ── index math ────────────────────────────────────────────────
        # All internal indices (_data_*) are in "data-file sample space":
        #   raw mode      → 1 MHz   (indices as-is)
        #   decimated mode → 100 kHz (indices / data_step)
        # _step_in_getitem is the decimation left to do at read time.
        if self._use_decimated:
            q = self.data_step
            self.t_dis       = (self.meta['t_disruption'].values * 1000 / q).astype(int)
            self.disrupt_idx = self.t_dis - self.Twarn // q
            self.start_idx   = np.full(len(self.shots), self.baseline_length // q)
            self.stop_idx    = self.t_dis.copy()
            self._data_nsub    = self.nsub   // q
            self._data_stride  = self.stride // q
            self._step_in_getitem = 1        # already decimated
            self._data_root    = self._decimated_root
        else:
            self.t_dis       = (self.meta['t_disruption'].values * 1000).astype(int)
            self.disrupt_idx = self.t_dis - self.Twarn
            self.start_idx   = np.full(len(self.shots), self.baseline_length)
            self.stop_idx    = self.t_dis.copy()
            self._data_nsub    = self.nsub
            self._data_stride  = self.stride
            self._step_in_getitem = self.data_step
            self._data_root    = self.root

        # Output temporal length (always the same regardless of mode)
        self._T_sub = self.nsub // self.data_step

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
        """Tile each shot into fixed-length windows (in data-file space)."""
        shot_idx, starts, stops, d_local = [], [], [], []
        nsub   = self._data_nsub
        stride = self._data_stride

        for s in range(len(self.shots)):
            a, b = int(self.start_idx[s]), int(self.stop_idx[s])
            if b - a < nsub:
                continue                          # shot too short

            d = int(self.disrupt_idx[s])          # absolute disrupt index

            pos = a
            while pos + nsub <= b:
                shot_idx.append(s)
                starts.append(pos)
                stops.append(pos + nsub)

                # local disrupt offset inside this window
                if d <= pos:
                    d_local.append(0)             # fully disruptive
                elif d >= pos + nsub:
                    d_local.append(-1)            # fully clear
                else:
                    d_local.append(d - pos)       # transition inside

                pos += stride

            # snap last window to the end of the shot
            last_start = b - nsub
            if last_start > (pos - stride):
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

    def _compute_class_weights(self, indices: np.ndarray | None = None):
        """Compute pos_weight / neg_weight from per-timestep label counts.

        Parameters
        ----------
        indices : array of subsequence indices to consider.
                  If None, uses all subsequences.
                  Pass the *effective* training indices (e.g. after
                  undersampling / stratified rebalancing) to get weights
                  consistent with what the model actually sees.
        """
        T = self._T_sub
        step = self._step_in_getitem
        total_pos, total_neg = 0, 0

        dls = self.seq_disrupt_local if indices is None else self.seq_disrupt_local[indices]
        for dl in dls:
            if dl < 0:
                total_neg += T
            elif dl == 0:
                total_pos += T
            else:
                d = (dl + 1) // step   # +1 matches label boundary in __getitem__
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
        """Compute per-channel mean/std from *split* shots (online Welford).

        Works with both raw and pre-decimated data transparently.
        """
        mask    = self.splits == split
        indices = np.where(mask)[0]
        if len(indices) > max_shots:
            indices = np.random.choice(indices, max_shots, replace=False)

        running_sum    = np.zeros((20, 8), dtype=np.float64)
        running_sq_sum = np.zeros((20, 8), dtype=np.float64)
        n_total = 0

        for s in tqdm(indices, desc=f'Norm stats ({split})'):
            shot = self.shots[s]
            with h5py.File(self._data_root / f'{shot}.h5', 'r') as f:
                chunk = f['LFS'][..., self.start_idx[s]:self.stop_idx[s]]
                if not self._use_decimated:
                    baseline = f['LFS'][..., :self.baseline_length]

            if self._use_decimated:
                data = chunk.astype(np.float64)
            else:
                offset = np.mean(baseline, axis=-1, keepdims=True)
                data = (chunk - offset).astype(np.float64)

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
        src = 'decimated' if self._use_decimated else 'raw'
        print(f"  Subseq length: {self._T_sub:,} samples "
              f"({self.nsub/self.FS*1e3:.0f} ms, step={self.data_step}, "
              f"source={src})")
        print(f"  pos_weight={self.pos_weight:.3f}  neg_weight={self.neg_weight:.3f}")
        return info

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self):
        return len(self.seq_shot_idx)

    def __getitem__(self, index):
        s    = int(self.seq_shot_idx[index])
        shot = self.shots[s]

        with h5py.File(self._data_root / f'{shot}.h5', 'r') as f:
            X = f['LFS'][..., self.seq_start[index]:self.seq_stop[index]].astype(np.float32)
            if not self._use_decimated:
                baseline = f['LFS'][..., :self.baseline_length].astype(np.float32)

        # 1. offset removal  (skip if pre-decimated)
        if not self._use_decimated:
            offset = np.mean(baseline, axis=-1, keepdims=True)
            X -= offset

        # 2. temporal decimation  (skip if pre-decimated)
        if self._step_in_getitem > 1:
            X = X[..., ::self._step_in_getitem]

        # 3. normalisation
        if self.normalize and self.norm_mean is not None:
            X = (X - self.norm_mean[..., np.newaxis]) / self.norm_std[..., np.newaxis]

        # 4. per-timestep label & weight
        T = X.shape[-1]
        target = np.zeros(T, dtype=np.float32)
        weight = np.full(T, self.neg_weight, dtype=np.float32)

        dl = int(self.seq_disrupt_local[index])
        if dl >= 0:
            # +1 matches disruptcnn: (disrupt_idxi - start_idxi + 1) / data_step
            d = min((dl + 1) // self._step_in_getitem, T)
            target[d:] = 1.0
            weight[d:] = self.pos_weight

        return (torch.from_numpy(np.ascontiguousarray(X)),
                torch.from_numpy(target),
                torch.from_numpy(weight))


# ══════════════════════════════════════════════════════════════════════
#  Stratified batch sampler — balanced pos/neg in every batch
# ══════════════════════════════════════════════════════════════════════

import math
from torch.utils.data import Sampler

class StratifiedBatchSampler(Sampler):
    """Yield batches where positive and negative subsequences are balanced.

    Each batch contains ``batch_size // 2`` positive (disruptive) indices
    and ``batch_size - batch_size // 2`` negative (clear) indices.
    The minority class is oversampled (cycled) so every epoch still
    covers all samples from the majority class.

    Parameters
    ----------
    labels : array-like of bool / 0-1
        Per-subsequence label (True = contains disruption onset).
    indices : array-like of int
        Global dataset indices that belong to this split.
    batch_size : int
    drop_last : bool
        If True, drop the final incomplete batch.
    seed : int
        Base random seed; call ``set_epoch(e)`` to re-seed each epoch.
    """

    def __init__(self, labels, indices, batch_size, drop_last=True, seed=42):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        labels = np.asarray(labels)
        indices = np.asarray(indices)
        self.pos_idx = indices[labels[indices] == 1]
        self.neg_idx = indices[labels[indices] == 0]

        # how many of each per batch
        self.n_pos_per_batch = batch_size // 2
        self.n_neg_per_batch = batch_size - self.n_pos_per_batch

        # total batches so majority class is covered once
        n_from_pos = math.ceil(len(self.pos_idx) / max(self.n_pos_per_batch, 1))
        n_from_neg = math.ceil(len(self.neg_idx) / max(self.n_neg_per_batch, 1))
        self._n_batches = max(n_from_pos, n_from_neg)
        if self.drop_last:
            self._total = self._n_batches * batch_size
        else:
            # add one partial batch if there are remainders
            self._total = self._n_batches * batch_size

    def set_epoch(self, epoch: int):
        """Re-seed the RNG for a new epoch (ensures different shuffles)."""
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)

        # shuffle and cycle to fill exactly _n_batches * n_per_batch
        pos_shuf = rng.permutation(self.pos_idx)
        neg_shuf = rng.permutation(self.neg_idx)

        n_pos_needed = self._n_batches * self.n_pos_per_batch
        n_neg_needed = self._n_batches * self.n_neg_per_batch

        # tile (oversample) the minority side if needed
        if len(pos_shuf) > 0:
            pos_ext = np.tile(pos_shuf, math.ceil(n_pos_needed / len(pos_shuf)))[:n_pos_needed]
        else:
            pos_ext = np.array([], dtype=int)
        if len(neg_shuf) > 0:
            neg_ext = np.tile(neg_shuf, math.ceil(n_neg_needed / len(neg_shuf)))[:n_neg_needed]
        else:
            neg_ext = np.array([], dtype=int)

        for b in range(self._n_batches):
            batch = np.concatenate([
                pos_ext[b * self.n_pos_per_batch : (b + 1) * self.n_pos_per_batch],
                neg_ext[b * self.n_neg_per_batch : (b + 1) * self.n_neg_per_batch],
            ])
            rng.shuffle(batch)          # shuffle within the batch
            yield batch.tolist()

    def __len__(self):
        return self._n_batches


# ══════════════════════════════════════════════════════════════════════
#  DataLoader factory
# ══════════════════════════════════════════════════════════════════════

def create_loaders(
    dataset: ECEiTCNDataset,
    batch_size: int = 4,
    num_workers: int = 4,
    stratified_train: bool = True,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for every split present in meta.csv.

    Parameters
    ----------
    stratified_train : bool
        If True (default), the training split uses a
        ``StratifiedBatchSampler`` so every batch has roughly
        equal positive / negative subsequences.
    """
    loaders = {}
    for split in np.unique(dataset.splits):
        idx = dataset.get_split_indices(split)
        if len(idx) == 0:
            continue

        is_train = (split == 'train')

        if is_train and stratified_train:
            sampler = StratifiedBatchSampler(
                labels     = dataset.seq_has_disrupt.astype(int),
                indices    = idx,
                batch_size = batch_size,
                drop_last  = True,
            )

            # Recompute class weights based on the *effective* training
            # distribution after stratified balancing.  With undersample=1.0
            # in disruptcnn, negatives are subsampled to match positives;
            # here we achieve the same by computing weights from an equal
            # mix of the positive and negative subsequence pools.
            n_pos = len(sampler.pos_idx)
            n_neg = len(sampler.neg_idx)
            n_eff = min(n_pos, n_neg)   # effective count per class
            eff_indices = np.concatenate([
                sampler.pos_idx[:n_eff],
                sampler.neg_idx[:n_eff],
            ])
            old_pw, old_nw = dataset.pos_weight, dataset.neg_weight
            dataset._compute_class_weights(indices=eff_indices)
            print(f'[create_loaders] Recomputed class weights after '
                  f'stratified balancing:')
            print(f'  before: pos_weight={old_pw:.4f}, '
                  f'neg_weight={old_nw:.4f}, ratio={old_pw/old_nw:.2f}x')
            print(f'  after : pos_weight={dataset.pos_weight:.4f}, '
                  f'neg_weight={dataset.neg_weight:.4f}, '
                  f'ratio={dataset.pos_weight/dataset.neg_weight:.2f}x')

            loaders[split] = DataLoader(
                dataset,
                batch_sampler = sampler,
                num_workers   = num_workers,
                pin_memory    = True,
            )
        else:
            subset = Subset(dataset, idx)
            loaders[split] = DataLoader(
                subset,
                batch_size  = batch_size,
                shuffle     = False,
                num_workers = num_workers,
                pin_memory  = True,
                drop_last   = False,
            )
    return loaders
