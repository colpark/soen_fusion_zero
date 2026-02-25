"""
Original DisruptCNN data setting: segment and label logic from the paper/shot lists.

This module implements the original recipe only:
- Read shot lists from d3d_*_ecei.final.txt (columns: Shot, # segments, tstart, tlast, dt, SNR min, t_flat_start, t_flat_last, tdisrupt).
- Flattop-only mode: segment start = t_flat_start, segment end = tend = max(tdisrupt, min(tlast, t_flat_stop)) with t_flat_stop = t_flat_start + t_flat_last; drop shots with NaN t_flat_start.
- Non-flattop: segment start = 0 (relative to tstart), tend = max(tdisrupt, tlast).
- Twarn = 300 ms: label as disruptive from sample index disrupt_idx = ceil((tdisrupt - Twarn - tstart) / dt) to end of segment.
- All indices are in "sample space" (tstart = 0 at index 0, dt per sample).

Use this dataloader for training that exactly matches the original DisruptCNN setting.
"""

from __future__ import annotations

import os
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
        })
    return out


def _parse_shot_lists(
    disrupt_file: str,
    clear_file: str,
    flattop_only: bool,
    snr_min_threshold: Optional[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse disrupt and clear shot list files and compute per-shot segment indices.

    Returns:
        shot: (N,) shot numbers
        start_idx: (N,) first sample index of segment (inclusive)
        stop_idx: (N,) last sample index of segment (inclusive)
        disrupt_idx: (N,) sample index where label becomes 1 (tdisrupt - Twarn); -1000 if not disrupted
        disrupted: (N,) bool, True if shot is disrupted
        dt: (N,) timestep in ms (for reference)
        zero_idx: (N,) sample index corresponding to t=0 (for compatibility)
    """
    data_disrupt = np.loadtxt(disrupt_file, skiprows=1)
    data_clear = np.loadtxt(clear_file, skiprows=1)
    data_all = np.vstack((data_disrupt, data_clear))

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

    # Disrupt index: first sample labeled as disruptive = ceil((tdisrupt - Twarn - tstart) / dt)
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
        clear_file: str,
        disrupt_file: str,
        train: bool = True,
        flattop_only: bool = True,
        Twarn: int = 300,
        test: int = 0,
        test_indices: Optional[list[int]] = None,
        label_balance: str = "const",
        normalize: bool = True,
        data_step: int = 1,
        nsub: Optional[int] = None,
        nrecept: Optional[int] = None,
        snr_min_threshold: Optional[float] = None,
    ):
        """
        root: Data root with 'disrupt/', 'clear/', and normalization.npz.
        clear_file, disrupt_file: Paths to shot list .txt (tab/space separated, header row).
        flattop_only: If True, use only flattop window and drop shots with NaN t_flat_start.
        Twarn: Time (ms) before tdisrupt at which to start labeling as disruptive (default 300).
        test, test_indices: Overfit test setup (subset of data).
        label_balance: 'const' for class weights, else no weighting.
        data_step: Step in sample index when reading (1 = every sample).
        nsub, nrecept: Subsequence length and receptive field for sliding windows.
        snr_min_threshold: If set (e.g. 3.0), keep only shots with SNR min > threshold.
        """
        self.root = root
        self.train = train
        self.Twarn = Twarn
        self.test = test
        self.label_balance = label_balance
        self.normalize = normalize
        self.data_step = data_step
        self.nsub = nsub if nsub is not None else 78125
        self.nrecept = nrecept if nrecept is not None else 30000

        (
            self.shot,
            self.start_idx,
            self.stop_idx,
            self.disrupt_idx,
            self.disrupted,
            _dt,
            self.zero_idx,
        ) = _parse_shot_lists(disrupt_file, clear_file, flattop_only, snr_min_threshold)

        self.length = len(self.shot)

        # Offsets placeholder (filled on first read per shot)
        filename = self._filename(0)
        with h5py.File(filename, "r") as f:
            self.offsets = np.zeros(f["offsets"].shape + (self.shot.size,), dtype=f["offsets"].dtype)

        # Normalization
        norm_path = os.path.join(self.root, "normalization.npz")
        f = np.load(norm_path)
        if flattop_only:
            self.normalize_mean = f["mean_flat"]
            self.normalize_std = f["std_flat"]
        else:
            self.normalize_mean = f["mean_all"]
            self.normalize_std = f["std_all"]
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

    def _filename(self, shot_index: int) -> str:
        shot = self.shot[shot_index]
        folder = "disrupt" if self.disrupted[shot_index] else "clear"
        return os.path.join(self.root, folder, f"{shot}.h5")

    def shots2seqs(self) -> None:
        """Build per-subsequence indices: shot_idxi, start_idxi, stop_idxi, disrupt_idxi, disruptedi."""
        self.shot_idxi = []
        self.start_idxi = []
        self.stop_idxi = []
        self.disrupt_idxi = []
        for s in range(len(self.shot)):
            N = int((self.stop_idx[s] - self.start_idx[s] + 1) / self.data_step)
            num_seq_frac = (N - self.nsub) / float(self.nsub - self.nrecept + 1) + 1
            num_seq = max(1, int(np.ceil(num_seq_frac)))
            Nseq = self.nsub + (num_seq - 1) * (self.nsub - self.nrecept + 1)
            if (self.start_idx[s] > self.zero_idx[s]) and (
                (self.start_idx[s] - self.zero_idx[s] + 1) > (Nseq - N) * self.data_step
            ):
                self.start_idx[s] -= (Nseq - N) * self.data_step
            else:
                num_seq -= 1
                if num_seq < 1:
                    num_seq = 1
                Nseq = self.nsub + (num_seq - 1) * (self.nsub - self.nrecept + 1)
                self.start_idx[s] += (N - Nseq) * self.data_step
            for m in range(num_seq):
                self.shot_idxi.append(s)
                self.start_idxi.append(
                    self.start_idx[s] + (m * self.nsub - m * self.nrecept + m) * self.data_step
                )
                self.stop_idxi.append(
                    self.start_idx[s] + ((m + 1) * self.nsub - m * self.nrecept + m) * self.data_step
                )
                if self.start_idxi[-1] <= self.disrupt_idx[s] <= self.stop_idxi[-1]:
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
        sizes: tuple[float, float, float] = (0.8, 0.1, 0.1),
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
        shot_index = self.shot_idxi[index]
        filename = self._filename(shot_index)
        with h5py.File(filename, "r") as f:
            if np.all(self.offsets[..., shot_index] == 0):
                self.offsets[..., shot_index] = f["offsets"][...]
            X = (
                f["LFS"][..., self.start_idxi[index] : self.stop_idxi[index]][..., :: self.data_step]
                - self.offsets[..., shot_index][..., np.newaxis]
            )
        if self.normalize:
            X = (X - self.normalize_mean[..., np.newaxis]) / self.normalize_std[..., np.newaxis]
        return X

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        if self.test > 0 and hasattr(self, "test_indices"):
            # Map index to stored test index
            idx = self.test_indices[index]
        else:
            idx = index
        X = self._read_data(idx)
        target = np.zeros((X.shape[-1],), dtype=X.dtype)
        weight = self.neg_weight * np.ones((X.shape[-1],), dtype=X.dtype)
        if self.disruptedi[idx]:
            first_disrupt = int((self.disrupt_idxi[idx] - self.start_idxi[idx] + 1) / self.data_step)
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
    """Build train/val/test DataLoaders for EceiDatasetOriginal (same API as loader.data_generator)."""
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
