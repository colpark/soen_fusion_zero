"""
Original DisruptCNN data setting: segment and label logic from the paper/shot lists.

This module implements the original recipe only:
- Read shot lists from d3d_*_ecei.final.txt (columns: Shot, # segments, tstart, tlast, dt, SNR min, t_flat_start, t_flat_last, tdisrupt).
- Flattop-only mode: segment start = t_flat_start, segment end = tend = max(tdisrupt, min(tlast, t_flat_stop)) with t_flat_stop = t_flat_start + t_flat_last; drop shots with NaN t_flat_start.
- Non-flattop: segment start = 0 (relative to tstart), tend = max(tdisrupt, tlast).
- Twarn = 300 ms: label as disruptive from sample index disrupt_idx = ceil((tdisrupt - Twarn - tstart) / dt) to end of segment.
- All indices are in "sample space" (tstart = 0 at index 0, dt per sample).

Segment/tiling/read/weights match original: (1) clear_file optional; when provided, data_all = vstack(disrupt, clear), else disrupt-only. (2) shots2seqs same formulas, no file-length skip/tail. (3) _read_data exact slice, no clamping. (4) calc_label_weights over sequence indices with safe division.

Use this dataloader for training that exactly matches the original DisruptCNN setting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.utils.data as data
import h5py
from sklearn.model_selection import train_test_split

from disruptcnn.sampler import StratifiedSampler


# Shot list column layout (0-indexed after header)
# Shot=0, segments=1, tstart=2, tlast=3, dt=4, SNR_min=5, t_flat_start=6, t_flat_last=7, tdisrupt=8
COL_SHOT = 0
COL_TSTART = 2
COL_TLAST = 3
COL_DT = 4
COL_SNR_MIN = 5
COL_T_FLAT_START = 6
COL_T_FLAT_LAST = 7
COL_TDISRUPT = 8

TWARN_MS = 300.0


def segment_info_for_comparison(
    shot_list_path: str,
    flattop_only: bool,
    snr_min_threshold: Optional[float] = None,
) -> List[dict]:
    """
    Parse a single shot list (e.g. disrupt only) and return per-shot segment info
    for comparison notebooks. No H5 or clear list required.

    Returns list of dicts with keys: shot, tstart, tlast, dt, t_flat_start, t_flat_last,
    t_flat_stop, tdisrupt, start_idx, stop_idx, disrupt_idx, segment_length_samples,
    segment_length_ms, has_flattop (not NaN).
    """
    data = np.loadtxt(shot_list_path, skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if snr_min_threshold is not None:
        snr = data[:, COL_SNR_MIN].astype(float)
        data = data[snr > snr_min_threshold]
    tstart = data[:, COL_TSTART].astype(float)
    tlast = data[:, COL_TLAST].astype(float)
    dt = data[:, COL_DT].astype(float)
    t_flat_start = data[:, COL_T_FLAT_START].astype(float)
    t_flat_last = data[:, COL_T_FLAT_LAST].astype(float)
    tdisrupt = data[:, COL_TDISRUPT].astype(float)
    t_flat_stop = t_flat_start + t_flat_last
    has_flattop = ~np.isnan(t_flat_start)
    disrupt_idx_raw = np.ceil((tdisrupt - TWARN_MS - tstart) / dt).astype(int)
    disrupt_idx_raw[tdisrupt < 0] = -1000
    out = []
    for i in range(len(data)):
        if flattop_only and np.isnan(t_flat_start[i]):
            continue
        if flattop_only:
            start_idx = int(np.ceil((t_flat_start[i] - tstart[i]) / dt[i]))
            tend = max(tdisrupt[i], min(tlast[i], t_flat_stop[i]))
        else:
            start_idx = int(np.ceil((0.0 - tstart[i]) / dt[i]))
            tend = max(tdisrupt[i], tlast[i])
        stop_idx = int(np.floor((tend - tstart[i]) / dt[i]))
        dis_idx = int(disrupt_idx_raw[i])
        n_samp = max(0, stop_idx - start_idx + 1)
        out.append({
            "shot": int(data[i, COL_SHOT]),
            "tstart": float(tstart[i]),
            "tlast": float(tlast[i]),
            "dt": float(dt[i]),
            "t_flat_start": float(t_flat_start[i]) if has_flattop[i] else np.nan,
            "t_flat_last": float(t_flat_last[i]) if has_flattop[i] else np.nan,
            "t_flat_stop": float(t_flat_stop[i]) if has_flattop[i] else np.nan,
            "tdisrupt": float(tdisrupt[i]),
            "start_idx": start_idx,
            "stop_idx": stop_idx,
            "disrupt_idx": dis_idx,
            "segment_length_samples": n_samp,
            "segment_length_ms": n_samp * dt[i],
            "has_flattop": bool(has_flattop[i]),
            "zero_idx": int(np.ceil((0.0 - tstart[i]) / dt[i])),
        })
    return out


def subsequences_original_tiling(
    segment: dict,
    nsub: int = 78125,
    nrecept: int = 30000,
    data_step: int = 1,
) -> List[dict]:
    """
    Given one shot's segment info (from segment_info_for_comparison), return the list of
    subsequences produced by dataset_original's shots2seqs tiling (overlap = nsub - nrecept + 1).

    No H5 or full dataset required. Used to compare "how a shot is turned into subsequences"
    in the original DisruptCNN (dataset_original) pipeline.

    Returns list of dicts: seq_idx, start, stop, length, has_disrupt, disrupt_local (offset in window or -1).
    """
    start_idx = segment["start_idx"]
    stop_idx = segment["stop_idx"]
    disrupt_idx = segment["disrupt_idx"]
    zero_idx = segment.get("zero_idx")
    if zero_idx is None:
        tstart = segment["tstart"]
        dt = segment["dt"]
        zero_idx = int(np.ceil((0.0 - tstart) / dt))

    N = max(0, int((stop_idx - start_idx + 1) / data_step))
    num_seq_frac = (N - nsub) / float(nsub - nrecept + 1) + 1
    num_seq = max(1, int(np.ceil(num_seq_frac)))
    Nseq = nsub + (num_seq - 1) * (nsub - nrecept + 1)
    if (start_idx > zero_idx) and ((start_idx - zero_idx + 1) > (Nseq - N) * data_step):
        start_idx = start_idx - (Nseq - N) * data_step
    else:
        num_seq = max(1, num_seq - 1)
        Nseq = nsub + (num_seq - 1) * (nsub - nrecept + 1)
        start_idx = start_idx + (N - Nseq) * data_step

    out = []
    for m in range(num_seq):
        start = start_idx + (m * nsub - m * nrecept + m) * data_step
        stop = start_idx + ((m + 1) * nsub - m * nrecept + m) * data_step
        if start <= disrupt_idx <= stop:
            has_disrupt = True
            disrupt_local = disrupt_idx - start
        else:
            has_disrupt = False
            disrupt_local = -1
        out.append({
            "seq_idx": m,
            "start": start,
            "stop": stop,
            "length": stop - start,
            "has_disrupt": has_disrupt,
            "disrupt_local": disrupt_local,
        })
    return out


def subsequences_past_tiling(
    start_idx: int,
    stop_idx: int,
    disrupt_idx: int,
    nsub: int,
    stride: int,
    stop_at_last_window_containing_disrupt: bool = True,
    t_dis_samples: Optional[int] = None,
) -> List[dict]:
    """
    Tile a segment into subsequences the "past" way (ECEiTCNDataset-style): fixed stride,
    windows [pos, min(pos+nsub, b)], optional cap so last window still contains t_disrupt.

    t_dis_samples: absolute sample index of disruption (disrupt_idx + Twarn). If None, not used for pos_limit.
    Returns list of dicts: seq_idx, start, stop, length, has_disrupt, disrupt_local.
    """
    a, b = start_idx, stop_idx
    d = disrupt_idx
    if b - a < 1:
        return []
    pos_limit = b
    if stop_at_last_window_containing_disrupt and d <= b and t_dis_samples is not None:
        pos_limit = min(b, t_dis_samples + 1)
    out = []
    pos = a
    seq_idx = 0
    while pos < pos_limit:
        stop = min(pos + nsub, b)
        if d <= pos:
            has_disrupt = True
            disrupt_local = 0
        elif d >= stop:
            has_disrupt = False
            disrupt_local = -1
        else:
            has_disrupt = True
            disrupt_local = d - pos
        out.append({
            "seq_idx": seq_idx,
            "start": pos,
            "stop": stop,
            "length": stop - pos,
            "has_disrupt": has_disrupt,
            "disrupt_local": disrupt_local,
        })
        pos += stride
        seq_idx += 1
    return out


def _parse_shot_lists(
    disrupt_file: str,
    clear_file: Optional[str],
    flattop_only: bool,
    snr_min_threshold: Optional[float],
) -> tuple:
    """
    Parse disrupt (and optionally clear) shot list files and compute per-shot segment indices.
    If clear_file is provided and exists, data_all = vstack(disrupt, clear). Else disrupt-only.

    Returns:
        shot, start_idx, stop_idx, disrupt_idx, disrupted, dt, zero_idx
    """
    if not disrupt_file or not str(disrupt_file).strip():
        raise ValueError("disrupt_file is required.")
    data_disrupt = np.loadtxt(disrupt_file, skiprows=1)
    if data_disrupt.ndim == 1:
        data_disrupt = data_disrupt[np.newaxis, :]
    if clear_file and str(clear_file).strip() and Path(clear_file).exists():
        data_clear = np.loadtxt(clear_file, skiprows=1)
        if data_clear.ndim == 1:
            data_clear = data_clear[np.newaxis, :]
        data_all = np.vstack((data_disrupt, data_clear))
    else:
        data_all = data_disrupt

    if snr_min_threshold is not None:
        snr_min = data_all[:, COL_SNR_MIN].astype(float)
        keep = snr_min > snr_min_threshold
        data_all = data_all[keep]

    tflatstarts = data_all[:, COL_T_FLAT_START].astype(float)
    if np.any(np.isnan(tflatstarts)) and flattop_only:
        data_all = data_all[~np.isnan(tflatstarts)]
        tflatstarts = data_all[:, COL_T_FLAT_START].astype(float)

    tflatstops = data_all[:, COL_T_FLAT_START].astype(float) + data_all[:, COL_T_FLAT_LAST].astype(float)
    tstarts = data_all[:, COL_TSTART].astype(float)
    tstops = data_all[:, COL_TLAST].astype(float)
    dt = data_all[:, COL_DT].astype(float)
    tdisrupt = data_all[:, COL_TDISRUPT].astype(float)

    Twarn = 300.0  # ms

    disrupt_idx = np.ceil((tdisrupt - Twarn - tstarts) / dt).astype(int)
    disrupt_idx[tdisrupt < 0] = -1000
    disrupted = disrupt_idx > 0

    zero_idx = np.ceil((0.0 - tstarts) / dt).astype(int)

    if flattop_only:
        start_idx = np.ceil((tflatstarts - tstarts) / dt).astype(int)
        tend = np.maximum(tdisrupt, np.minimum(tstops, tflatstops))
    else:
        start_idx = np.ceil((0.0 - tstarts) / dt).astype(int)
        tend = np.maximum(tdisrupt, tstops)

    stop_idx = np.floor((tend - tstarts) / dt).astype(int)
    shot = data_all[:, COL_SHOT].astype(int)

    return shot, start_idx, stop_idx, disrupt_idx, disrupted, dt, zero_idx


class EceiDatasetOriginal(data.Dataset):
    """
    ECEi dataset with the original DisruptCNN segment and label logic.

    Segment:
        - flattop_only=True: start at t_flat_start, end at tend = max(tdisrupt, min(tlast, t_flat_stop)); shots with NaN t_flat_start dropped.
        - flattop_only=False: start at 0 (relative to tstart), end at max(tdisrupt, tlast).
    Labels:
        - Binary per time step. Label 1 from disrupt_idx to end of segment; disrupt_idx = ceil((tdisrupt - 300 - tstart) / dt).
    """

    def __init__(
        self,
        root: str,
        clear_file: Optional[str],
        disrupt_file: str,
        train: bool = True,
        flattop_only: bool = True,
        Twarn: int = 300,
        test: int = 0,
        test_indices: Optional[List[int]] = None,
        label_balance: str = "const",
        normalize: bool = True,
        data_step: int = 1,
        nsub: Optional[int] = None,
        nrecept: Optional[int] = None,
        snr_min_threshold: Optional[float] = None,
        decimated_root: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
    ):
        self.root = root
        self._decimated_root = Path(decimated_root) if decimated_root and str(decimated_root).strip() else None
        self._norm_stats_path = norm_stats_path
        self.train = train
        self.Twarn = Twarn
        self.test = test
        self.label_balance = label_balance
        self.normalize = normalize
        self.data_step = data_step
        _nsub = nsub if nsub is not None else 78125
        _nrecept = nrecept if nrecept is not None else 30000

        (
            self.shot,
            self.start_idx,
            self.stop_idx,
            self.disrupt_idx,
            self.disrupted,
            _dt,
            self.zero_idx,
        ) = _parse_shot_lists(disrupt_file, clear_file, flattop_only, snr_min_threshold)

        # Keep only shots whose H5 file actually exists (not all shot list entries may be available)
        available = np.array([
            self._path_for_shot(self.shot[i], self.disrupted[i]).exists()
            for i in range(len(self.shot))
        ], dtype=bool)
        if not np.all(available):
            n_dropped = int((~available).sum())
            self.shot = self.shot[available]
            self.start_idx = self.start_idx[available]
            self.stop_idx = self.stop_idx[available]
            self.disrupt_idx = self.disrupt_idx[available]
            self.disrupted = self.disrupted[available]
            self.zero_idx = self.zero_idx[available]
            print(f"[EceiDatasetOriginal] Using only shots with available H5: {len(self.shot)} kept, {n_dropped} dropped (no file).")
        if len(self.shot) == 0:
            raise FileNotFoundError("No H5 files found for any shot in the list. Check data_root/decimated_root and shot list.")

        self.length = len(self.shot)

        # When using decimated H5, indices and window sizes are in raw (1 MHz) space; convert to decimated
        if self._decimated_root is not None:
            self.start_idx = (self.start_idx // self.data_step).astype(np.int64)
            self.stop_idx = (self.stop_idx // self.data_step).astype(np.int64)
            nondisrupt = self.disrupt_idx < 0  # -1000 for clear
            self.disrupt_idx = (self.disrupt_idx // self.data_step).astype(np.int64)
            self.disrupt_idx[nondisrupt] = -1000
            self.zero_idx = (self.zero_idx // self.data_step).astype(np.int64)
            self._step_in_getitem = 1
            self.nsub = max(1, _nsub // self.data_step)
            self.nrecept = max(1, _nrecept // self.data_step)
        else:
            self._step_in_getitem = self.data_step
            self.nsub = _nsub
            self.nrecept = _nrecept

        filename = self._filename(0)
        with h5py.File(filename, "r") as f:
            if "offsets" in f:
                self.offsets = np.zeros(f["offsets"].shape + (self.shot.size,), dtype=f["offsets"].dtype)
            else:
                # Decimated data often has offset already removed; use zeros
                LFS = f["LFS"]
                base = np.zeros((LFS.shape[0], LFS.shape[1], self.shot.size), dtype=np.float64)
                self.offsets = base

        if self._norm_stats_path and os.path.isfile(self._norm_stats_path):
            norm_path = self._norm_stats_path
        else:
            norm_path = None
            for candidate in [
                os.path.join(self.root, "normalization.npz"),
                os.path.join(self.root, "norm_stats.npz"),
                str(self._decimated_root / "normalization.npz") if self._decimated_root else None,
                str(self._decimated_root / "norm_stats.npz") if self._decimated_root else None,
            ]:
                if candidate and os.path.isfile(candidate):
                    norm_path = candidate
                    break
        if norm_path is None:
            raise FileNotFoundError(
                "Normalization stats not found. Set norm_stats_path or place normalization.npz or norm_stats.npz in root/decimated_root."
            )
        f = np.load(norm_path)
        # Accept DisruptCNN keys (mean_flat, std_flat, mean_all, std_all) or fallback to mean/std
        def _get(key: str, fallback: str):
            return f[key] if key in f.files else f[fallback]
        if flattop_only:
            self.normalize_mean = _get("mean_flat", "mean")
            self.normalize_std = _get("std_flat", "std")
        else:
            self.normalize_mean = _get("mean_all", "mean")
            self.normalize_std = _get("std_all", "std")
        f.close()

        self.shots2seqs()
        self.calc_label_weights()

        if self.test > 0:
            if self.test == 1:
                disinds = np.where(self.disruptedi)[0]
                self.test_indices = np.array([disinds[np.random.randint(disinds.size)]])
            else:
                if test_indices is None:
                    disinds = np.where(self.disruptedi)[0]
                    disinds = np.random.choice(disinds, size=min(int(self.test / 2), len(disinds)), replace=False)
                    nondisinds = np.where(self.disruptedi == 0)[0]
                    n_neg = self.test - len(disinds)
                    nondisinds = np.random.choice(nondisinds, size=min(n_neg, len(nondisinds)), replace=False)
                    self.test_indices = np.concatenate([disinds, nondisinds])
                else:
                    self.test_indices = np.array(test_indices)
            self.length = len(self.test_indices)

    def _path_for_shot(self, shot: int, is_disrupt: bool) -> Path:
        """Path to H5 for a given shot (for existence check)."""
        if self._decimated_root is not None:
            return self._decimated_root / f"{shot}.h5"
        folder = "disrupt" if is_disrupt else "clear"
        return Path(self.root) / folder / f"{shot}.h5"

    def _filename(self, shot_index: int) -> str:
        shot = self.shot[shot_index]
        return str(self._path_for_shot(shot, self.disrupted[shot_index]))

    def shots2seqs(self) -> None:
        """Build per-subsequence indices; same formulas as original loader (no file-length skip/tail)."""
        self.shot_idxi = []
        self.start_idxi = []
        self.stop_idxi = []
        self.disrupt_idxi = []
        step = self._step_in_getitem
        for s in range(len(self.shot)):
            N = int((self.stop_idx[s] - self.start_idx[s] + 1) / step)
            num_seq_frac = (N - self.nsub) / float(self.nsub - self.nrecept + 1) + 1
            num_seq = np.ceil(num_seq_frac).astype(int)
            if num_seq < 1:
                num_seq = 1
            Nseq = self.nsub + (num_seq - 1) * (self.nsub - self.nrecept + 1)
            if (self.start_idx[s] > self.zero_idx[s]) and (
                (self.start_idx[s] - self.zero_idx[s] + 1) > (Nseq - N) * step
            ):
                self.start_idx[s] -= (Nseq - N) * step
            else:
                num_seq -= 1
                Nseq = self.nsub + (num_seq - 1) * (self.nsub - self.nrecept + 1)
                self.start_idx[s] += (N - Nseq) * step
            for m in range(num_seq):
                start_i = int(
                    self.start_idx[s] + (m * self.nsub - m * self.nrecept + m) * step
                )
                stop_i = int(
                    self.start_idx[s] + ((m + 1) * self.nsub - m * self.nrecept + m) * step
                )
                self.shot_idxi.append(s)
                self.start_idxi.append(start_i)
                self.stop_idxi.append(stop_i)
                if start_i <= self.disrupt_idx[s] <= stop_i:
                    self.disrupt_idxi.append(self.disrupt_idx[s])
                else:
                    self.disrupt_idxi.append(-1000)
        self.shot_idxi = np.array(self.shot_idxi)
        self.start_idxi = np.array(self.start_idxi)
        self.stop_idxi = np.array(self.stop_idxi)
        self.disrupt_idxi = np.array(self.disrupt_idxi)
        self.disruptedi = self.disrupt_idxi > 0
        if self.test == 0:
            self.length = len(self.shot_idxi)

    def calc_label_weights(self, inds: Optional[np.ndarray] = None) -> None:
        """Weights for BCE; inds are sequence indices (default all). Safe division (max(..., 1))."""
        if inds is None:
            inds = np.arange(len(self.shot_idxi))
        if "const" in self.label_balance:
            N = np.sum(self.stop_idxi[inds] - self.start_idxi[inds])
            disinds = inds[self.disruptedi[inds]]
            Ndisrupt = np.sum(self.stop_idxi[disinds] - self.disrupt_idxi[disinds])
            Nnondisrupt = N - Ndisrupt
            self.pos_weight = 0.5 * N / max(Ndisrupt, 1)
            self.neg_weight = 0.5 * N / max(Nnondisrupt, 1)
        else:
            self.pos_weight = 1.0
            self.neg_weight = 1.0

    def train_val_test_split(
        self,
        sizes: tuple = (0.8, 0.1, 0.1),
        random_seed: int = 42,
        train_inds: Optional[np.ndarray] = None,
        val_inds: Optional[np.ndarray] = None,
        test_inds: Optional[np.ndarray] = None,
    ) -> None:
        assert len(sizes) == 3 and np.isclose(sum(sizes), 1.0)
        labels = self.disrupted
        if self.test > 0:
            self.train_inds = self.test_indices
            self.val_inds = np.array([], dtype=int)
            self.test_inds = np.array([], dtype=int)
            return
        if train_inds is not None:
            self.train_inds = train_inds
            self.val_inds = val_inds
            self.test_inds = test_inds
        else:
            train_shot_inds, valtest_shot_inds, _, _ = train_test_split(
                np.arange(len(self.shot)), labels, stratify=labels, test_size=sizes[1] + sizes[2], random_state=random_seed
            )
            val_shot_inds, test_shot_inds, _, _ = train_test_split(
                valtest_shot_inds,
                labels[valtest_shot_inds],
                stratify=labels[valtest_shot_inds],
                test_size=sizes[2] / (sizes[1] + sizes[2]),
                random_state=random_seed,
            )
            self.train_inds = np.where(np.in1d(self.shot_idxi, train_shot_inds))[0]
            self.val_inds = np.where(np.in1d(self.shot_idxi, val_shot_inds))[0]
            self.test_inds = np.where(np.in1d(self.shot_idxi, test_shot_inds))[0]
        self.calc_label_weights(inds=self.train_inds)

    def _read_data(self, index: int) -> np.ndarray:
        """Read LFS slice [start_idxi:stop_idxi] with step; no clamping (matches original)."""
        shot_index = self.shot_idxi[index]
        filename = self._filename(shot_index)
        with h5py.File(filename, "r") as f:
            if np.all(self.offsets[..., shot_index] == 0):
                self.offsets[..., shot_index] = f["offsets"][...]
            X = (
                f["LFS"][..., self.start_idxi[index] : self.stop_idxi[index]][..., :: self._step_in_getitem]
                - self.offsets[..., shot_index][..., np.newaxis]
            )
        if self.normalize:
            X = (X - self.normalize_mean[..., np.newaxis]) / self.normalize_std[..., np.newaxis]
        return X

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple:
        if self.test > 0 and hasattr(self, "test_indices"):
            idx = self.test_indices[index]
        else:
            idx = index
        X = self._read_data(idx)
        target = np.zeros((X.shape[-1],), dtype=X.dtype)
        weight = self.neg_weight * np.ones((X.shape[-1],), dtype=X.dtype)
        if self.disruptedi[idx]:
            first_disrupt = int((self.disrupt_idxi[idx] - self.start_idxi[idx] + 1) / self._step_in_getitem)
            target[first_disrupt:] = 1
            weight[first_disrupt:] = self.pos_weight
        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(target).float(),
            idx,
            torch.from_numpy(weight).float(),
        )


def data_generator_original(
    dataset: EceiDatasetOriginal,
    batch_size: int,
    distributed: bool = False,
    num_workers: int = 0,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    undersample: Optional[float] = None,
):
    if not hasattr(dataset, "train_inds"):
        dataset.train_val_test_split()
    train_dataset = data.Subset(dataset, dataset.train_inds)
    val_dataset = data.Subset(dataset, dataset.val_inds)
    test_dataset = data.Subset(dataset, dataset.test_inds)
    train_sampler = StratifiedSampler(
        train_dataset,
        num_replicas=num_replicas,
        rank=rank,
        stratify=dataset.disruptedi[dataset.train_inds],
        distributed=distributed,
        undersample=undersample,
    )
    val_sampler = StratifiedSampler(
        val_dataset,
        num_replicas=num_replicas,
        rank=rank,
        stratify=dataset.disruptedi[dataset.val_inds],
        distributed=distributed,
        undersample=undersample,
    )
    if undersample is not None:
        inds = np.array([dataset.train_inds[i] for i in train_sampler])
        dataset.calc_label_weights(inds=inds)
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=train_sampler
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=val_sampler
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════
#  Wrapper for train_tcn_ddp: fixed-length (X, target, weight) + split API
# ═══════════════════════════════════════════════════════════════════════


class OriginalStyleDatasetForDDP:
    """Wraps EceiDatasetOriginal so it matches the interface expected by train_tcn_ddp.

    - __getitem__(i) returns (X, target, weight) with X padded/cropped to fixed T = nsub.
    - seq_has_disrupt, get_split_indices('train'/'test'/'val'), pos_weight, neg_weight,
      _compute_class_weights(indices) for stratified sampling and class weights.
    """

    def __init__(self, inner: EceiDatasetOriginal):
        self._inner = inner
        if not hasattr(inner, "train_inds"):
            inner.train_val_test_split()
        self._T_fixed = int(inner.nsub)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int):
        X, target, _idx, weight = self._inner[index]
        X = X.cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        target = target.cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)
        weight = weight.cpu().numpy() if isinstance(weight, torch.Tensor) else np.asarray(weight)
        T = X.shape[-1]
        if T >= self._T_fixed:
            X = X[..., : self._T_fixed].copy()
            target = target[: self._T_fixed].copy()
            weight = weight[: self._T_fixed].copy()
        else:
            pad = self._T_fixed - T
            X = np.concatenate([X, np.zeros((*X.shape[:-1], pad), dtype=X.dtype)], axis=-1)
            target = np.concatenate([target, np.zeros(pad, dtype=target.dtype)])
            weight = np.concatenate([weight, np.zeros(pad, dtype=weight.dtype)])
        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(target).float(),
            torch.from_numpy(weight).float(),
        )

    @property
    def seq_has_disrupt(self) -> np.ndarray:
        return self._inner.disruptedi

    def get_split_indices(self, split: str) -> np.ndarray:
        if split == "train":
            return self._inner.train_inds
        if split == "test":
            return self._inner.test_inds
        if split == "val":
            return self._inner.val_inds
        raise ValueError(f"Unknown split: {split}")

    @property
    def pos_weight(self) -> float:
        return self._inner.pos_weight

    @property
    def neg_weight(self) -> float:
        return self._inner.neg_weight

    def _compute_class_weights(self, indices: Optional[np.ndarray] = None):
        self._inner.calc_label_weights(inds=indices)


# ═══════════════════════════════════════════════════════════════════════
#  Prebuilt mmap dataset — load from preprocessing_mmap.ipynb output
# ═══════════════════════════════════════════════════════════════════════

def _load_ragged_mmap(path):
    try:
        from mmap_ninja import RaggedMmap
        return RaggedMmap(str(path))
    except ImportError as e:
        raise ImportError(
            "PrebuiltOriginalSubseqDataset requires mmap_ninja. Install with: pip install mmap_ninja"
        ) from e


class PrebuiltOriginalSubseqDataset:
    """
    Dataset that loads pre-saved subsequences from preprocessing_mmap.ipynb output.

    Same interface as OriginalStyleDatasetForDDP: __getitem__(i) returns (X, target, weight),
    seq_has_disrupt, get_split_indices('train'/'test'/'val'), pos_weight, neg_weight,
    _compute_class_weights (no-op; uses saved weights). All normalization is already
    applied in the saved data.
    """

    def __init__(self, root: str | Path):
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(f"Prebuilt mmap dir not found: {self._root}")
        self._X = _load_ragged_mmap(self._root / "X")
        self._target = _load_ragged_mmap(self._root / "target")
        self._weight = _load_ragged_mmap(self._root / "weight")
        self._labels = np.load(self._root / "labels.npy")
        self._train_inds = np.load(self._root / "train_inds.npy")
        self._test_inds = np.load(self._root / "test_inds.npy")
        val_path = self._root / "val_inds.npy"
        self._val_inds = np.load(val_path) if val_path.exists() else np.array([], dtype=np.int64)
        meta_path = self._root / "meta.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            self._pos_weight = float(meta.get("pos_weight", 1.0))
            self._neg_weight = float(meta.get("neg_weight", 1.0))
        else:
            self._pos_weight = 1.0
            self._neg_weight = 1.0

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, index: int):
        X = np.ascontiguousarray(self._X[index]).astype(np.float32)
        target = np.asarray(self._target[index], dtype=np.float32)
        weight = np.asarray(self._weight[index], dtype=np.float32)
        return (
            torch.from_numpy(X),
            torch.from_numpy(target),
            torch.from_numpy(weight),
        )

    @property
    def seq_has_disrupt(self) -> np.ndarray:
        return self._labels

    def get_split_indices(self, split: str) -> np.ndarray:
        if split == "train":
            return self._train_inds
        if split == "test":
            return self._test_inds
        if split == "val":
            return self._val_inds
        raise ValueError(f"Unknown split: {split}")

    @property
    def pos_weight(self) -> float:
        return self._pos_weight

    @property
    def neg_weight(self) -> float:
        return self._neg_weight

    def _compute_class_weights(self, indices: Optional[np.ndarray] = None):
        """No-op; weights are fixed from preprocessing."""
        pass
