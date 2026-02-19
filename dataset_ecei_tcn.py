"""
ECEi TCN-style Dataset for disruption prediction.
Adapted from disruptcnn (Churchill et al. 2019) for KSTAR ECEi data.

Data layout
-----------
root/  (disruptive shots)
  meta.csv          – columns: shot, split, t_disruption (ms), ...
  {shot}.h5         – HDF5 with key 'LFS' → (20, 8, T) float at 1 MHz

Optionally, *clear_root* adds non-disruptive shots (no disruption; whole shot = clear):
  clear_root/
    meta.csv        – columns: shot, split (t_disruption optional/NaN)
    {shot}.h5       – same LFS layout
  If meta.csv is missing, shot IDs are discovered from *.h5 and splits
  assigned by clear_split_frac (e.g. 80% train, 20% test).

Optionally, *decimated_root* / *clear_decimated_root* hold pre-processed files:
  decimated_root/
    meta.csv        – same as root
    {shot}.h5       – offset-removed & decimated: (20, 8, T/data_step)

When decimated_root exists the dataset reads from it directly, skipping
offset removal and decimation in __getitem__ for much faster training.

For PCA data (n_input_channels in {1, 4, 8, 16}), LFS shape is (C, T). PCA
projections have different variances per component (PC1 >> PC2 >> ...); per-
component normalization is recommended so the model sees comparable scales.
Norm stats are (C,) and should be stored separately per PCA variant (e.g.
norm_stats_pca1.npz, norm_stats_pca8.npz) so switching --pca-components
does not overwrite or mix stats.

Preprocessing pipeline (applied in __getitem__ when reading raw data)
---------------------------------------------------------------------
1. DC offset removal   – mean of first *baseline_length* samples per channel
2. Temporal decimation – keep every *data_step*-th sample (1 MHz → 100 kHz at step=10)
3. Z-score normalisation – per-channel (20×8) mean/std computed from training shots

Twarn (warning window)
----------------------
Twarn is the pre-disruption window in samples at 1 MHz (e.g. 300_000 = 300 ms).
It defines (t_disrupt - Twarn, t_disrupt]. By default we label that window as 1.
With ignore_twarn=True we do not train on it (weight=0); only clear (0) is
learned and the boundary is not assumed.

Label construction
------------------
Per-timestep binary (when ignore_twarn=False):
  0 (clear)      for time > Twarn before disruption (and optionally in the last
                 exclude_last_ms before disruption, if set)
  1 (disruptive) for time in (t_disrupt - Twarn, t_disrupt - exclude_last_ms]
  Optionally the last exclude_last_ms (e.g. 30 ms) can be excluded from the
  positive class (Churchill et al.: mitigation timing).
When ignore_twarn=True: the Twarn window is masked from loss (weight=0).
Shots from root are disruptive; optional clear_root adds non-disruptive shots (whole shot = clear).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List

try:
    from mmap_ninja import RaggedMmap as _RaggedMmap
except ImportError:
    _RaggedMmap = None


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
        clear_root:       str | None = None,
        clear_decimated_root: str | None = None,
        clear_split_frac: Optional[Dict[str, float]] = None,
        exclude_last_ms:  float = 0.0,        # don't label last N ms as 1 (Churchill: 30 ms for mitigation)
        ignore_twarn:     bool   = False,     # if True, do not train on Twarn window (weight=0); learn boundary
        n_input_channels: Optional[int] = None,  # 1, 4, 8, or 16 for PCA data (LFS shape (C,T)); None = 160 (20×8)
    ):
        """
        Args:
            root:             Directory with meta.csv and raw {shot}.h5 (disruptive).
            decimated_root:   Directory with pre-decimated h5 files (disruptive).
            clear_root:      Optional directory with non-disruptive shots (whole shot = clear).
            clear_decimated_root: Optional pre-decimated clear shots.
            clear_split_frac: If clear_root has no meta.csv, assign splits by fraction
                              e.g. {'train': 0.8, 'test': 0.2}. Default 80% train, 20% test.
            norm_stats_path:  Where to save / load per-channel mean & std.
            norm_train_split: Which split to use when computing stats.
            norm_max_shots:   Max number of shots sampled for stat estimation.
            n_input_channels: If 1, 4, 8, or 16, use PCA data: LFS shape (C, T); norm (C,). None = full 160 (20×8).
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
        self.exclude_last_ms  = exclude_last_ms
        self.ignore_twarn    = ignore_twarn
        self._n_input_channels = n_input_channels  # None → (20,8); 1,4,8,16 → PCA (C,T)

        # ── check for pre-decimated data ──────────────────────────────
        self._use_decimated = False
        self._decimated_root: Optional[Path] = None
        if decimated_root is not None:
            p = Path(decimated_root)
            if p.exists():
                self._use_decimated = True
                self._decimated_root = p
                print(f'[ECEiTCNDataset] Using pre-decimated data '
                      f'from {p}')

        # ── metadata: disruptive from root ─────────────────────────────
        self.meta = pd.read_csv(self.root / 'meta.csv')
        self.shots   = self.meta['shot'].values.astype(int)
        self.splits  = self.meta['split'].values.astype(str)

        q = self.data_step if self._use_decimated else 1
        self.t_dis       = (self.meta['t_disruption'].values * 1000 / q).astype(int)
        self.disrupt_idx = self.t_dis - self.Twarn // q
        exclude_samps   = int(self.exclude_last_ms * (1000 / q))  # ms → samples in data space
        self.positive_end_idx = np.maximum(
            self.disrupt_idx,
            self.t_dis - exclude_samps,
        ).astype(np.int64)  # end of "label 1" region (don't label last exclude_last_ms as 1)
        self.start_idx   = np.full(len(self.shots), self.baseline_length // q)
        self.stop_idx    = self.t_dis.copy()

        # Per-shot data root and step (for mixed disruptive + clear)
        self._shot_data_root: list = [
            self._decimated_root if self._use_decimated else self.root
        ] * len(self.shots)
        self._shot_step: list = [1 if self._use_decimated else self.data_step] * len(self.shots)

        # ── append clear (non-disruptive) shots if requested ──────────
        if clear_root is not None:
            clear_path = Path(clear_root)
            if clear_path.exists():
                self._add_clear_shots(
                    clear_path,
                    Path(clear_decimated_root) if clear_decimated_root else None,
                    clear_split_frac,
                )

        # ── index math (common) ────────────────────────────────────────
        self._data_nsub    = self.nsub // q
        self._data_stride  = self.stride // q
        self._data_root    = self._decimated_root if self._use_decimated else self.root
        self._step_in_getitem = 1 if self._use_decimated else self.data_step

        # Output temporal length (always the same regardless of mode)
        self._T_sub = self.nsub // self.data_step

        # normalisation – auto load / compute / save
        self.norm_mean: Optional[np.ndarray] = None   # (20, 8) or (C,) for PCA
        self.norm_std:  Optional[np.ndarray] = None   # (20, 8) or (C,) for PCA

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

    def _add_clear_shots(
        self,
        clear_root: Path,
        clear_decimated_root: Optional[Path],
        clear_split_frac: Optional[Dict[str, float]],
    ) -> None:
        """Append non-disruptive shots from clear_root. Whole shot = clear (label 0)."""
        q = self.data_step if self._use_decimated else 1
        use_clear_decimated = (
            clear_decimated_root is not None
            and clear_decimated_root.exists()
        )

        # Discover clear shots and splits
        clear_meta_path = clear_root / "meta.csv"
        if clear_meta_path.exists():
            clear_meta = pd.read_csv(clear_meta_path)
            if "shot" not in clear_meta.columns:
                raise ValueError(f"{clear_meta_path} must have 'shot' column")
            clear_shots = clear_meta["shot"].values.astype(int)
            clear_splits = (
                clear_meta["split"].values.astype(str)
                if "split" in clear_meta.columns
                else np.array(["train"] * len(clear_shots))
            )
        else:
            h5_files = list(clear_root.glob("*.h5"))
            clear_shots = np.array([int(f.stem) for f in h5_files if f.stem.isdigit()])
            if len(clear_shots) == 0:
                clear_shots = np.array([int(f.stem) for f in h5_files])
            if len(clear_shots) == 0:
                print(f"[ECEiTCNDataset] No clear shots found in {clear_root}")
                return
            frac = clear_split_frac or {"train": 0.8, "test": 0.2}
            n_train = int(len(clear_shots) * frac.get("train", 0.8))
            n_test = len(clear_shots) - n_train
            clear_splits = np.array(
                ["train"] * n_train + ["test"] * n_test,
                dtype=object,
            )

        start_list = []
        stop_list = []
        disrupt_list = []
        root_list: List[Path] = []
        step_list: List[int] = []
        shots_loaded = []
        splits_loaded = []

        for i, shot in enumerate(clear_shots):
            read_root = (
                clear_decimated_root
                if (use_clear_decimated and self._use_decimated)
                else clear_root
            )
            h5_path = read_root / f"{shot}.h5"
            if not h5_path.exists():
                continue
            with h5py.File(h5_path, "r") as f:
                T = f["LFS"].shape[-1]
            if use_clear_decimated and self._use_decimated:
                # Clear decimated: T already in decimated space
                start_list.append(self.baseline_length // self.data_step)
                stop_list.append(T)
                disrupt_list.append(T + 1)
                root_list.append(clear_decimated_root)
                step_list.append(1)
            elif self._use_decimated and not use_clear_decimated:
                # Main uses decimated, clear is raw: convert to decimated space
                start_list.append(self.baseline_length // self.data_step)
                stop_list.append(T // self.data_step)
                disrupt_list.append(T // self.data_step + 1)
                root_list.append(clear_root)
                step_list.append(self.data_step)
            else:
                # raw space (main is raw)
                start_list.append(self.baseline_length)
                stop_list.append(T)
                disrupt_list.append(T + 1)
                root_list.append(clear_root)
                step_list.append(self.data_step)
            shots_loaded.append(shot)
            splits_loaded.append(clear_splits[i])

        if not start_list:
            print(f"[ECEiTCNDataset] No valid clear shots loaded from {clear_root}")
            return

        n_clear = len(start_list)
        self.shots = np.concatenate([self.shots, np.array(shots_loaded)])
        self.splits = np.concatenate([self.splits, np.array(splits_loaded)])
        self.start_idx = np.concatenate(
            [self.start_idx, np.array(start_list, dtype=np.int64)]
        )
        self.stop_idx = np.concatenate(
            [self.stop_idx, np.array(stop_list, dtype=np.int64)]
        )
        self.disrupt_idx = np.concatenate(
            [self.disrupt_idx, np.array(disrupt_list, dtype=np.int64)]
        )
        self.positive_end_idx = np.concatenate(
            [self.positive_end_idx, np.array(stop_list, dtype=np.int64)]
        )
        self.t_dis = np.concatenate(
            [self.t_dis, np.array(stop_list, dtype=np.int64)]
        )
        self._shot_data_root = list(self._shot_data_root) + root_list
        self._shot_step = list(self._shot_step) + step_list

        self.meta = pd.DataFrame(
            {"shot": self.shots, "split": self.splits, "t_disruption": self.t_dis * q / 1000}
        )
        clear_source = (
            clear_decimated_root
            if (use_clear_decimated and self._use_decimated)
            else clear_root
        )
        print(
            f"[ECEiTCNDataset] Added {n_clear} non-disruptive (clear) shots from {clear_source}"
        )

    # ── subsequence tiling ────────────────────────────────────────────

    def _build_subsequences(self):
        """Tile each shot into fixed-length windows (in data-file space)."""
        shot_idx, starts, stops, d_local, e_local = [], [], [], [], []
        nsub   = self._data_nsub
        stride = self._data_stride

        for s in range(len(self.shots)):
            a, b = int(self.start_idx[s]), int(self.stop_idx[s])
            if b - a < nsub:
                continue                          # shot too short

            d = int(self.disrupt_idx[s])          # absolute disrupt index
            e_abs = int(self.positive_end_idx[s]) # end of "label 1" region

            pos = a
            while pos + nsub <= b:
                shot_idx.append(s)
                starts.append(pos)
                stops.append(pos + nsub)

                # local disrupt offset inside this window
                if d <= pos:
                    d_local.append(0)             # fully disruptive
                    e_local.append(min(pos + nsub, e_abs) - pos)
                elif d >= pos + nsub:
                    d_local.append(-1)            # fully clear
                    e_local.append(0)
                else:
                    d_local.append(d - pos)       # transition inside
                    e_local.append(min(pos + nsub, e_abs) - pos)

                pos += stride

            # snap last window to the end of the shot
            last_start = b - nsub
            if last_start > (pos - stride):
                shot_idx.append(s)
                starts.append(last_start)
                stops.append(b)
                if d <= last_start:
                    d_local.append(0)
                    e_local.append(min(b, e_abs) - last_start)
                elif d >= b:
                    d_local.append(-1)
                    e_local.append(0)
                else:
                    d_local.append(d - last_start)
                    e_local.append(min(b, e_abs) - last_start)

        self.seq_shot_idx        = np.array(shot_idx,  dtype=int)
        self.seq_start           = np.array(starts,    dtype=int)
        self.seq_stop            = np.array(stops,     dtype=int)
        self.seq_disrupt_local   = np.array(d_local,  dtype=int)
        self.seq_positive_end_local = np.array(e_local, dtype=int)
        self.seq_has_disrupt     = self.seq_disrupt_local >= 0

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
        els = self.seq_positive_end_local if indices is None else self.seq_positive_end_local[indices]
        for dl, el in zip(dls, els):
            if dl < 0:
                total_neg += T
            elif self.ignore_twarn:
                # Twarn window is masked; only pre-d timesteps count as neg
                d = (dl + 1) // step
                total_neg += d
            elif dl == 0:
                e = min((el + step - 1) // step, T)
                total_pos += e
                total_neg += T - e
            else:
                d = (dl + 1) // step   # +1 matches label boundary in __getitem__
                e = min((el + step - 1) // step, T)
                e = max(d, e)
                total_neg += d + (T - e)
                total_pos += e - d
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

        C = self._n_input_channels
        if C is None:
            running_sum    = np.zeros((20, 8), dtype=np.float64)
            running_sq_sum = np.zeros((20, 8), dtype=np.float64)
        else:
            running_sum    = np.zeros((C,), dtype=np.float64)
            running_sq_sum = np.zeros((C,), dtype=np.float64)
        n_total = 0

        for s in tqdm(indices, desc=f'Norm stats ({split})'):
            shot = self.shots[s]
            data_root = Path(self._shot_data_root[s])
            step = self._shot_step[s]
            a, b = self.start_idx[s], self.stop_idx[s]
            if step > 1:
                a, b = a * step, b * step
            try:
                with h5py.File(data_root / f'{shot}.h5', 'r') as f:
                    chunk = np.asarray(f['LFS'][..., a:b], dtype=np.float64)
                    if step > 1:
                        baseline = np.asarray(f['LFS'][..., :self.baseline_length], dtype=np.float64)
            except (OSError, IOError) as e:
                print(f'[ECEiTCNDataset] Skipping shot {shot} for norm stats: {e}')
                continue

            try:
                if step == 1:
                    data = chunk
                else:
                    offset = np.mean(baseline, axis=-1, keepdims=True)
                    data = (chunk - offset)[..., ::step].astype(np.float64)
                T = data.shape[-1]
                running_sum    += data.sum(axis=-1)
                running_sq_sum += (data ** 2).sum(axis=-1)
                n_total += T
            except (OSError, IOError) as e:
                print(f'[ECEiTCNDataset] Skipping shot {shot} for norm stats (after read): {e}')
                continue

        if n_total == 0:
            raise RuntimeError('Norm stats: no shots could be read (all failed with OSError)')
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

    def __getitem__(self, index, _retry: int = 0):
        s    = int(self.seq_shot_idx[index])
        shot = self.shots[s]
        data_root = self._shot_data_root[s]
        step = self._shot_step[s]
        start, stop = int(self.seq_start[index]), int(self.seq_stop[index])
        if step > 1:
            start_raw, stop_raw = start * step, stop * step
        else:
            start_raw, stop_raw = start, stop

        try:
            with h5py.File(Path(data_root) / f'{shot}.h5', 'r') as f:
                X = np.asarray(f['LFS'][..., start_raw:stop_raw], dtype=np.float32)
                if step > 1:
                    baseline = np.asarray(
                        f['LFS'][..., :self.baseline_length], dtype=np.float32
                    )
        except (OSError, IOError) as e:
            # Skip bad file (e.g. unsupported compression filter); try another sample
            if _retry < 20:
                alt_idx = (index + _retry + 1) % len(self)
                return self.__getitem__(alt_idx, _retry=_retry + 1)
            raise RuntimeError(
                f'HDF5 read failed for shot {shot} ({data_root / f"{shot}.h5"}): {e}. '
                'File may use an unsupported compression filter.'
            ) from e

        # 1. offset removal  (skip if pre-decimated, i.e. step==1)
        if step > 1:
            offset = np.mean(baseline, axis=-1, keepdims=True)
            X -= offset

        # 2. temporal decimation  (skip if already decimated)
        if step > 1:
            X = X[..., ::step]

        # 3. normalisation
        if self.normalize and self.norm_mean is not None:
            X = (X - self.norm_mean[..., np.newaxis]) / self.norm_std[..., np.newaxis]

        # 4. per-timestep label & weight
        T = X.shape[-1]
        target = np.zeros(T, dtype=np.float32)
        weight = np.full(T, self.neg_weight, dtype=np.float32)

        dl = int(self.seq_disrupt_local[index])
        el = int(self.seq_positive_end_local[index])  # end of "1" region in data space
        step = self._step_in_getitem
        if dl >= 0:
            # +1 matches disruptcnn: (disrupt_idxi - start_idxi + 1) / data_step
            d = min((dl + 1) // step, T)
            if self.ignore_twarn:
                # Do not train on the Twarn window: mask it from loss (weight=0).
                # Only "clear" (0) before the window is learned; boundary is not assumed.
                weight[d:] = 0.0
            else:
                e = min((el + step - 1) // step, T)  # end of positive in output steps
                e = max(d, e)  # ensure e >= d
                target[d:e] = 1.0
                weight[d:e] = self.pos_weight
                # Excluded segment [e:T] (e.g. last 30 ms): leave target=0, weight=0 so no loss
                if e < T:
                    weight[e:] = 0.0

        return (torch.from_numpy(np.ascontiguousarray(X)),
                torch.from_numpy(target),
                torch.from_numpy(weight))


# ══════════════════════════════════════════════════════════════════════
#  Prebuilt subsequence dataset — load from preprocess_subseqs.py output
# ══════════════════════════════════════════════════════════════════════

class PrebuiltSubseqDataset(Dataset):
    """
    Dataset that loads pre-saved subsequence data (from preprocess_subseqs.py).

    Supports two formats (auto-detected from the split directory):
    - mmap: RaggedMmap dirs X/, target/, weight/ — fast random access, no per-sample file open.
    - npz:  One .npz per sample with keys X, target, weight (legacy).
    """

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.subdir = self.root / split
        if not self.subdir.exists():
            raise FileNotFoundError(f"Prebuilt subseq dir not found: {self.subdir}")
        x_dir = self.subdir / "X"
        if x_dir.is_dir():
            if _RaggedMmap is None:
                raise ImportError("Prebuilt subseq data is in mmap format; install mmap_ninja: pip install mmap_ninja")
            self._use_mmap = True
        else:
            self._use_mmap = False
        if self._use_mmap:
            self._X_mmap = _RaggedMmap(str(x_dir))
            self._target_mmap = _RaggedMmap(str(self.subdir / "target"))
            self._weight_mmap = _RaggedMmap(str(self.subdir / "weight"))
            self._n = len(self._X_mmap)
            self._files = []
        else:
            self._files = sorted(
                [p for p in self.subdir.glob("*.npz") if p.stem != "labels"],
                key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
            )
            self._n = len(self._files)
        labels_path = self.subdir / "labels.npy"
        self.seq_has_disrupt = np.load(labels_path) if labels_path.exists() else np.ones(self._n, dtype=np.int64)
        self.pos_weight = 1.0
        self.neg_weight = 1.0

    @property
    def format(self) -> str:
        """'mmap' or 'npz' — which format is being used for loading."""
        return "mmap" if self._use_mmap else "npz"

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        if self._use_mmap:
            X = np.ascontiguousarray(self._X_mmap[index])
            target = self._target_mmap[index]
            weight = self._weight_mmap[index]
        else:
            data = np.load(self._files[index])
            X = np.ascontiguousarray(data["X"])
            target = data["target"]
            weight = data["weight"]
        return (
            torch.from_numpy(X),
            torch.from_numpy(target),
            torch.from_numpy(weight),
        )

    def get_split_indices(self, split: str) -> np.ndarray:
        """Return indices for this split (this dataset is already one split)."""
        return np.arange(len(self)) if split == self.split else np.array([], dtype=np.int64)


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
