#!/usr/bin/env python3
"""
Benchmark memmap read throughput to compare locations (e.g. /storage vs /temporary).
Uses dummy data by default so no real dataset is required.

Usage:
  # Generate dummy memmap at two paths and compare read speed (storage vs temporary)
  python benchmark_mmap_read.py --path /storage/.../bench_mmap --path2 /temporary/.../bench_mmap --generate

  # Benchmark existing real data
  python benchmark_mmap_read.py --path /path/to/subseqs_mmap_all --no-generate

  # More samples / parallel readers
  python benchmark_mmap_read.py --path /tmp/bench --generate --samples 1000 --workers 8
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np


def generate_dummy_mmap(path: Path, n_segments: int = 500, T: int = 7000) -> None:
    """Write a small dummy per-split memmap (train only) with shape similar to ecei_mc: X (20, 8, T)."""
    from mmap_ninja import RaggedMmap

    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

    train_dir = path / "train"
    train_dir.mkdir()

    # X: (20, 8, T) per segment, like ecei_mc
    X_list = [np.random.randn(20, 8, T).astype(np.float32) for _ in range(n_segments)]
    RaggedMmap.from_lists(str(train_dir / "X"), X_list)

    # target / weight: (T,) per segment
    target_list = [np.zeros(T, dtype=np.float32) for _ in range(n_segments)]
    weight_list = [np.ones(T, dtype=np.float32) for _ in range(n_segments)]
    RaggedMmap.from_lists(str(train_dir / "target"), target_list)
    RaggedMmap.from_lists(str(train_dir / "weight"), weight_list)

    # labels.npy so layout matches
    np.save(train_dir / "labels.npy", np.zeros(n_segments, dtype=np.int64))

    total_mb = sum(a.nbytes for a in X_list) / (1024 ** 2)
    print(f"  Created dummy memmap at {path}: {n_segments} segments, X shape (20, 8, {T}) ~ {total_mb:.1f} MB")


def load_train_x(path: Path):
    from mmap_ninja import RaggedMmap
    train_x = Path(path) / "train" / "X"
    if not train_x.exists():
        raise FileNotFoundError(f"Expected {train_x} (per-split layout)")
    return RaggedMmap(str(train_x))


def benchmark_path(path: Path, n_samples: int, n_workers: int, label: str) -> dict:
    """Touch n_samples from train/X; return timing and throughput."""
    mmap = load_train_x(path)
    total = len(mmap)
    n_samples = min(n_samples, total)
    indices = np.linspace(0, total - 1, n_samples, dtype=int)

    if n_workers <= 1:
        t0 = time.perf_counter()
        bytes_read = 0
        for i in indices:
            arr = np.asarray(mmap[i])
            bytes_read += arr.nbytes
        elapsed = time.perf_counter() - t0
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def read_one(idx):
            arr = np.asarray(mmap[idx])
            return arr.nbytes

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(read_one, i) for i in indices]
            bytes_read = sum(f.result() for f in as_completed(futures))
        elapsed = time.perf_counter() - t0

    mb = bytes_read / (1024 ** 2)
    return {
        "label": label or str(path),
        "samples": n_samples,
        "time_s": elapsed,
        "samples_per_s": n_samples / elapsed if elapsed > 0 else 0,
        "MB_per_s": mb / elapsed if elapsed > 0 else 0,
        "MB_read": mb,
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark memmap read speed (storage vs temporary)")
    p.add_argument("--path", type=str, required=True, help="Path for memmap (create here if --generate)")
    p.add_argument("--path2", type=str, default=None, help="Second path to compare (e.g. /temporary/.../bench_mmap)")
    p.add_argument("--generate", action="store_true", dest="generate", help="Generate dummy memmap at path(s) first (default)")
    p.add_argument("--no-generate", action="store_false", dest="generate", help="Use existing data at path(s), do not generate")
    p.add_argument("--segments", type=int, default=500, help="Dummy segments when using --generate (default 500)")
    p.add_argument("--T", type=int, default=7000, dest="T", help="Time dimension per segment when using --generate (default 7000)")
    p.add_argument("--samples", type=int, default=500, help="Number of samples to read in benchmark (default 500)")
    p.add_argument("--workers", type=int, default=1, help="Parallel readers (default 1)")
    p.set_defaults(generate=True)
    args = p.parse_args()

    paths = [(Path(args.path), "path")]
    if args.path2:
        paths.append((Path(args.path2), "path2"))

    if args.generate:
        print("Generating dummy memmap data...")
        for path, _ in paths:
            generate_dummy_mmap(path, n_segments=args.segments, T=args.T)
        print()

    print("Memmap read benchmark (train/X)")
    print(f"  samples={args.samples}, workers={args.workers}")
    print()

    for path, label in paths:
        if not path.exists():
            print(f"  {label}: {path} — NOT FOUND")
            continue
        try:
            r = benchmark_path(path, args.samples, args.workers, label)
            print(f"  {r['label']}: {path}")
            print(f"    {r['time_s']:.2f} s  |  {r['samples_per_s']:.1f} samples/s  |  {r['MB_per_s']:.1f} MB/s  ({r['MB_read']:.1f} MB read)")
            print()
        except Exception as e:
            print(f"  {label}: {path} — ERROR: {e}")
            print()

    if len(paths) > 1:
        print("Higher MB/s suggests that path is faster (e.g. /temporary vs /storage).")


if __name__ == "__main__":
    main()
