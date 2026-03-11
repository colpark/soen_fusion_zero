#!/usr/bin/env python3
"""
Build a pre-decimated (1/10 length) memmap from subseqs_mmap_all.
This avoids 10× I/O in the dataloader: training reads small segments directly.

Usage:
  python build_decimated_1_10_mmap.py --source /path/to/subseqs_mmap_all --out /path/to/subseqs_mmap_all_decimated
  # Then train with: --prebuilt-mmap-dir /path/to/subseqs_mmap_all_decimated --decimate-factor 1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from mmap_ninja import RaggedMmap
except ImportError:
    raise ImportError("pip install mmap_ninja")

BATCH = 512
DECIMATE = 10


def main():
    p = argparse.ArgumentParser(description="Build 1/10-length memmap from subseqs_mmap_all")
    p.add_argument("--source", type=str, required=True, help="Path to subseqs_mmap_all (per-split: train/val/test)")
    p.add_argument("--out", type=str, required=True, help="Output path (e.g. subseqs_mmap_all_decimated)")
    p.add_argument("--batch", type=int, default=BATCH, help=f"Flush batch size (default {BATCH})")
    args = p.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    if not (src / "train" / "X").exists():
        raise FileNotFoundError(f"Expected per-split layout: {src / 'train/X'}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for split in ("train", "val", "test"):
        sub_src = src / split
        sub_out = out / split
        if not (sub_src / "X").exists():
            continue
        sub_out.mkdir()

        x_mmap = RaggedMmap(str(sub_src / "X"))
        t_mmap = RaggedMmap(str(sub_src / "target"))
        w_mmap = RaggedMmap(str(sub_src / "weight"))
        labels = np.load(sub_src / "labels.npy")
        n = len(labels)

        X_batch, target_batch, weight_batch = [], [], []
        started = False
        x_out = t_out = w_out = None

        for i in tqdm(range(n), desc=f"Decimate {split}"):
            X = np.asarray(x_mmap[i])[..., ::DECIMATE].astype(np.float32).copy()
            target = np.asarray(t_mmap[i], dtype=np.float32)[::DECIMATE].copy()
            weight = np.asarray(w_mmap[i], dtype=np.float32)[::DECIMATE].copy()
            X_batch.append(X)
            target_batch.append(target)
            weight_batch.append(weight)

            if len(X_batch) >= args.batch:
                if not started:
                    RaggedMmap.from_lists(str(sub_out / "X"), X_batch)
                    RaggedMmap.from_lists(str(sub_out / "target"), target_batch)
                    RaggedMmap.from_lists(str(sub_out / "weight"), weight_batch)
                    started = True
                else:
                    x_out = x_out or RaggedMmap(str(sub_out / "X"))
                    t_out = t_out or RaggedMmap(str(sub_out / "target"))
                    w_out = w_out or RaggedMmap(str(sub_out / "weight"))
                    x_out.extend(X_batch)
                    t_out.extend(target_batch)
                    w_out.extend(weight_batch)
                X_batch.clear()
                target_batch.clear()
                weight_batch.clear()

        if X_batch:
            if not started:
                RaggedMmap.from_lists(str(sub_out / "X"), X_batch)
                RaggedMmap.from_lists(str(sub_out / "target"), target_batch)
                RaggedMmap.from_lists(str(sub_out / "weight"), weight_batch)
            else:
                x_out.extend(X_batch)
                t_out.extend(target_batch)
                w_out.extend(weight_batch)

        np.save(sub_out / "labels.npy", labels)

    # Copy meta if present
    if (src / "meta.json").exists():
        with open(src / "meta.json") as f:
            meta = json.load(f)
        meta["decimated_from"] = str(src)
        meta["decimate_factor"] = DECIMATE
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    else:
        with open(out / "meta.json", "w") as f:
            json.dump({"decimate_factor": DECIMATE}, f, indent=2)

    print(f"Done. Use: --prebuilt-mmap-dir {out} --decimate-factor 1 (10× faster loading).")


if __name__ == "__main__":
    main()
