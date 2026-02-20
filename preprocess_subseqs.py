#!/usr/bin/env python3
"""
Preprocess ECEi data to subsequence-level files (.npz or mmap_ninja).

Reads from decimated shot-level HDF5 (dsrpt_decimated, clear_decimated),
builds the same subsequence windows as ECEiTCNDataset, applies normalization,
and saves under output_dir/train/ and output_dir/test/. With --format mmap (default),
uses mmap_ninja RaggedMmap for fast random access during training; with --format npz
writes one .npz per subsequence (slower load, backward compatible).

Usage:
  python preprocess_subseqs.py --output-dir subseqs [OPTIONS]
  python preprocess_subseqs.py --output-dir subseqs --format npz  # legacy

Requires norm_stats.npz (run ECEiTCNDataset once with normalize=True to create it).
Requires mmap_ninja for --format mmap: pip install mmap_ninja
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from mmap_ninja import RaggedMmap
    _HAS_MMAP_NINJA = True
except ImportError:
    _HAS_MMAP_NINJA = False


def parse_args():
    p = argparse.ArgumentParser(description="Save subsequence-level .npz for ECEi TCN")
    p.add_argument("--root", "--disrupt-root", dest="root", type=str, default="/home/idies/workspace/Storage/yhuang2/persistent/ecei/dsrpt", help="Disruptive shots root (meta.csv + .h5)")
    p.add_argument("--decimated-root", type=str, default="/home/idies/workspace/Storage/yhuang2/persistent/ecei/dsrpt_decimated")
    p.add_argument("--clear-root", type=str, default="/home/idies/workspace/Storage/yhuang2/persistent/ecei/clear_decimated")
    p.add_argument("--clear-decimated-root", type=str, default="/home/idies/workspace/Storage/yhuang2/persistent/ecei/clear_decimated")
    p.add_argument("--output-dir", type=str, default="subseqs", help="Output directory (train/ and test/ subdirs)")
    p.add_argument("--norm-stats", type=str, default="norm_stats.npz")
    p.add_argument("--nsub", type=int, default=781_250)
    p.add_argument("--stride", type=int, default=481_090)
    p.add_argument("--twarn", type=int, default=300_000)
    p.add_argument("--baseline-length", type=int, default=40_000)
    p.add_argument("--data-step", type=int, default=10)
    p.add_argument("--exclude-last-ms", type=float, default=0.0)
    p.add_argument("--ignore-twarn", action="store_true")
    p.add_argument("--format", choices=("mmap", "npz"), default="mmap",
                   help="Output format: mmap (fast load via mmap_ninja) or npz (one file per sample)")
    p.add_argument("--mmap-batch-size", type=int, default=512,
                   help="Batch size for flushing to mmap (only for --format mmap)")
    return p.parse_args()


def build_shot_metadata(
    root: Path,
    decimated_root: Optional[Path],
    clear_root: Optional[Path],
    clear_decimated_root: Optional[Path],
    twarn: int,
    baseline_length: int,
    data_step: int,
    exclude_last_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Path], List[int], np.ndarray]:
    """Build shot-level arrays and per-shot data roots. Returns (shots, splits, start_idx, stop_idx, disrupt_idx, positive_end_idx, _shot_data_root, _shot_step, t_dis)."""
    use_decimated = decimated_root is not None and decimated_root.exists() and (decimated_root / "meta.csv").exists()
    q = data_step if use_decimated else 1

    meta = pd.read_csv(root / "meta.csv")
    shots = meta["shot"].values.astype(int)
    splits = meta["split"].values.astype(str)
    t_dis = (meta["t_disruption"].values * 1000 / q).astype(int)
    disrupt_idx = t_dis - twarn // q
    exclude_samps = int(exclude_last_ms * (1000 / q))
    positive_end_idx = np.maximum(disrupt_idx, t_dis - exclude_samps).astype(np.int64)
    start_idx = np.full(len(shots), baseline_length // q, dtype=np.int64)
    stop_idx = t_dis.copy()

    # Segment end (DisruptCNN): use meta t_last/t_segment_end if present, else file length T
    seg_end_col = next((c for c in ("t_last", "t_segment_end", "t_segment_end_ms") if c in meta.columns), None)
    if seg_end_col is not None:
        seg_ms = pd.to_numeric(meta[seg_end_col], errors="coerce").values
        segment_end_samp = (seg_ms * 1000 / q).astype(np.float64)
    else:
        segment_end_samp = None
    read_root = decimated_root if use_decimated else root
    for i, shot in enumerate(shots):
        h5_path = Path(read_root) / f"{shot}.h5"
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                T = f["LFS"].shape[-1]
            if segment_end_samp is not None and not np.isnan(segment_end_samp[i]):
                end = min(int(segment_end_samp[i]), T)
            else:
                end = T
            stop_idx[i] = end
            positive_end_idx[i] = end

    shot_data_root: List[Path] = [decimated_root if use_decimated else root] * len(shots)
    shot_step: List[int] = [1 if use_decimated else data_step] * len(shots)

    # Append clear shots
    if clear_root is not None:
        clear_path = Path(clear_root)
        if clear_path.exists():
            use_clear_dec = (
                clear_decimated_root is not None
                and Path(clear_decimated_root).exists()
            )
            if (clear_path / "meta.csv").exists():
                cm = pd.read_csv(clear_path / "meta.csv")
                clear_shots = cm["shot"].values.astype(int)
                clear_splits = cm["split"].values.astype(str) if "split" in cm.columns else np.array(["train"] * len(clear_shots))
            else:
                h5s = list(clear_path.glob("*.h5"))
                clear_shots = np.array([int(f.stem) for f in h5s if f.stem.isdigit()] or [int(f.stem) for f in h5s])
                ntr = int(len(clear_shots) * 0.8)
                clear_splits = np.array(["train"] * ntr + ["test"] * (len(clear_shots) - ntr), dtype=object)
            read_root = Path(clear_decimated_root) if (use_clear_dec and use_decimated) else clear_path
            for i, shot in enumerate(clear_shots):
                h5_path = read_root / f"{shot}.h5"
                if not h5_path.exists():
                    continue
                with h5py.File(h5_path, "r") as f:
                    T = f["LFS"].shape[-1]
                shots = np.concatenate([shots, [shot]])
                splits = np.concatenate([splits, [clear_splits[i]]])
                start_idx = np.concatenate([start_idx, [baseline_length // q if (use_clear_dec and use_decimated) else baseline_length // data_step]])
                stop_idx = np.concatenate([stop_idx, [T]])
                disrupt_idx = np.concatenate([disrupt_idx, [T + 1]])
                positive_end_idx = np.concatenate([positive_end_idx, [T]])
                t_dis = np.concatenate([t_dis, [T]])
                shot_data_root.append(Path(clear_decimated_root) if (use_clear_dec and use_decimated) else clear_path)
                shot_step.append(1 if (use_clear_dec and use_decimated) else data_step)

    return shots, splits, start_idx, stop_idx, disrupt_idx, positive_end_idx, shot_data_root, shot_step, t_dis


def build_subsequence_index(
    n_shots: int,
    start_idx: np.ndarray,
    stop_idx: np.ndarray,
    disrupt_idx: np.ndarray,
    positive_end_idx: np.ndarray,
    nsub: int,
    stride: int,
    q: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (seq_shot_idx, seq_start, seq_stop, seq_disrupt_local, seq_positive_end_local)."""
    shot_idx, starts, stops, d_local, e_local = [], [], [], [], []
    data_nsub = nsub // q
    data_stride = stride // q

    for s in range(n_shots):
        a, b = int(start_idx[s]), int(stop_idx[s])
        if b - a < data_nsub:
            continue
        d = int(disrupt_idx[s])
        e_abs = int(positive_end_idx[s])
        pos = a
        while pos + data_nsub <= b:
            shot_idx.append(s)
            starts.append(pos)
            stops.append(pos + data_nsub)
            if d <= pos:
                d_local.append(0)
                e_local.append(min(pos + data_nsub, e_abs) - pos)
            elif d >= pos + data_nsub:
                d_local.append(-1)
                e_local.append(0)
            else:
                d_local.append(d - pos)
                e_local.append(min(pos + data_nsub, e_abs) - pos)
            pos += data_stride
        last_start = b - data_nsub
        if last_start > (pos - data_stride):
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

    return (
        np.array(shot_idx, dtype=int),
        np.array(starts, dtype=int),
        np.array(stops, dtype=int),
        np.array(d_local, dtype=int),
        np.array(e_local, dtype=int),
    )


def compute_labels_and_weights(
    T: int,
    dl: int,
    el: int,
    step: int,
    ignore_twarn: bool,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (target, weight) of shape (T,) each."""
    target = np.zeros(T, dtype=np.float32)
    weight = np.full(T, neg_weight, dtype=np.float32)
    if dl >= 0:
        d = min((dl + 1) // step, T)
        if ignore_twarn:
            weight[d:] = 0.0
        else:
            e = min((el + step - 1) // step, T)
            e = max(d, e)
            target[d:e] = 1.0
            weight[d:e] = pos_weight
            if e < T:
                weight[e:] = 0.0
    return target, weight


def run_preprocess(
    output_dir: str | Path,
    root: str | Path,
    decimated_root: str | Path | None = None,
    clear_root: str | Path | None = None,
    clear_decimated_root: str | Path | None = None,
    norm_stats: str | Path = "norm_stats.npz",
    nsub: int = 781_250,
    stride: int = 481_090,
    twarn: int = 300_000,
    baseline_length: int = 40_000,
    data_step: int = 10,
    exclude_last_ms: float = 0.0,
    ignore_twarn: bool = False,
    format_mode: str = "mmap",
    mmap_batch_size: int = 512,
) -> None:
    """Run subsequence preprocessing in-process. Call from notebook or script with same options as CLI."""
    root = Path(root)
    decimated_root = Path(decimated_root) if decimated_root else None
    clear_root = Path(clear_root) if clear_root else None
    clear_decimated_root = Path(clear_decimated_root) if clear_decimated_root else None
    out_dir = Path(output_dir)
    norm_path = Path(norm_stats)
    if not norm_path.exists():
        raise FileNotFoundError(f"Norm stats not found: {norm_path}. Run dataset once with normalize=True to create it.")

    use_decimated = decimated_root is not None and decimated_root.exists()
    q = data_step if use_decimated else 1
    step_in = 1 if use_decimated else data_step
    T_sub = nsub // data_step

    print("Building shot metadata...")
    shots, splits, start_idx, stop_idx, disrupt_idx, positive_end_idx, shot_data_root, shot_step, _ = build_shot_metadata(
        root, decimated_root, clear_root, clear_decimated_root,
        twarn, baseline_length, data_step, exclude_last_ms,
    )
    n_shots = len(shots)
    print(f"  Shots: {n_shots}")

    print("Building subsequence index...")
    seq_shot_idx, seq_start, seq_stop, seq_disrupt_local, seq_positive_end_local = build_subsequence_index(
        n_shots, start_idx, stop_idx, disrupt_idx, positive_end_idx,
        nsub, stride, q,
    )
    n_subseqs = len(seq_shot_idx)
    print(f"  Subsequences: {n_subseqs}")

    norm = np.load(norm_path)
    norm_mean = norm["mean"].astype(np.float32)   # (20, 8)
    norm_std = norm["std"].astype(np.float32)

    split_indices = {}
    for sp in np.unique(splits):
        split_shots = set(np.where(splits == sp)[0])
        idx = np.array([i for i, s in enumerate(seq_shot_idx) if s in split_shots])
        split_indices[sp] = idx
    if "val" in split_indices and "test" not in split_indices:
        split_indices["test"] = split_indices["val"]
    if "test" not in split_indices:
        split_indices["test"] = np.array([], dtype=int)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train").mkdir(exist_ok=True)
    (out_dir / "test").mkdir(exist_ok=True)

    meta_out = {"nsub": nsub, "data_step": data_step, "T_sub": T_sub, "format": format_mode}
    use_mmap = format_mode == "mmap" and _HAS_MMAP_NINJA
    if format_mode == "mmap" and not _HAS_MMAP_NINJA:
        raise RuntimeError("format_mode='mmap' requires mmap_ninja; install with: pip install mmap_ninja")

    for split_name, indices in split_indices.items():
        if len(indices) == 0:
            meta_out[split_name] = 0
            continue
        meta_out[split_name] = len(indices)
        subdir = out_dir / split_name
        subdir.mkdir(exist_ok=True)
        labels_split = []
        batch_size = mmap_batch_size if use_mmap else 1
        X_batch, target_batch, weight_batch = [], [], []
        mmap_initialized = False

        def flush_mmap():
            nonlocal mmap_initialized
            if not X_batch:
                return
            if not mmap_initialized:
                RaggedMmap.from_lists(str(subdir / "X"), X_batch)
                RaggedMmap.from_lists(str(subdir / "target"), target_batch)
                RaggedMmap.from_lists(str(subdir / "weight"), weight_batch)
                mmap_initialized = True
            else:
                x_m = RaggedMmap(str(subdir / "X"))
                x_m.extend(X_batch)
                t_m = RaggedMmap(str(subdir / "target"))
                t_m.extend(target_batch)
                w_m = RaggedMmap(str(subdir / "weight"))
                w_m.extend(weight_batch)
            X_batch.clear()
            target_batch.clear()
            weight_batch.clear()

        for local_i, global_idx in enumerate(tqdm(indices, desc=f"Writing {split_name}")):
            s = int(seq_shot_idx[global_idx])
            shot = int(shots[s])
            data_root = shot_data_root[s]
            step = shot_step[s]
            start, stop = int(seq_start[global_idx]), int(seq_stop[global_idx])
            if step > 1:
                start_raw, stop_raw = start * step, stop * step
            else:
                start_raw, stop_raw = start, stop

            with h5py.File(data_root / f"{shot}.h5", "r") as f:
                X = np.asarray(f["LFS"][..., start_raw:stop_raw], dtype=np.float32)
                if step > 1:
                    bl = np.asarray(f["LFS"][..., :baseline_length], dtype=np.float32)
                    X = (X - np.mean(bl, axis=-1, keepdims=True))[..., ::step]

            X = (X - norm_mean[..., np.newaxis]) / norm_std[..., np.newaxis]
            dl = int(seq_disrupt_local[global_idx])
            el = int(seq_positive_end_local[global_idx])
            target, weight = compute_labels_and_weights(
                T_sub, dl, el, step_in, ignore_twarn,
            )
            has_disrupt = 1 if dl >= 0 else 0
            labels_split.append(has_disrupt)

            if use_mmap:
                X_batch.append(X.astype(np.float32))
                target_batch.append(target)
                weight_batch.append(weight.astype(np.float32))
                if len(X_batch) >= batch_size:
                    flush_mmap()
            else:
                np.savez_compressed(
                    subdir / f"{local_i}.npz",
                    X=X.astype(np.float32),
                    target=target,
                    weight=weight.astype(np.float32),
                )

        if use_mmap:
            flush_mmap()
        np.save(subdir / "labels.npy", np.array(labels_split, dtype=np.int64))
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"Saved to {out_dir}; meta: {meta_out}")


def main():
    """CLI entry point: parse args and run in a subprocess environment."""
    args = parse_args()
    run_preprocess(
        output_dir=args.output_dir,
        root=args.root,
        decimated_root=args.decimated_root or None,
        clear_root=args.clear_root or None,
        clear_decimated_root=args.clear_decimated_root or None,
        norm_stats=args.norm_stats,
        nsub=args.nsub,
        stride=args.stride,
        twarn=args.twarn,
        baseline_length=args.baseline_length,
        data_step=args.data_step,
        exclude_last_ms=args.exclude_last_ms,
        ignore_twarn=args.ignore_twarn,
        format_mode=args.format,
        mmap_batch_size=args.mmap_batch_size,
    )


if __name__ == "__main__":
    main()
