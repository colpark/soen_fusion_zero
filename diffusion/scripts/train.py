#!/usr/bin/env python3
"""
Train diffusion model on decimated ECEi (160, ~7512).
Usage:
  python diffusion/scripts/train.py --prebuilt-mmap-dir ./subseqs_original_mmap --decimate-factor 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Add repo root so we can import diffusion.data without touching disruptcnn
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))

from diffusion.data.dataset import DecimatedEceiMmapDataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Train diffusion on (160, 7512) ECEi")
    p.add_argument("--prebuilt-mmap-dir", type=str, default="./subseqs_original_mmap")
    p.add_argument("--decimate-factor", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1, help="Placeholder")
    args = p.parse_args()

    ds = DecimatedEceiMmapDataset(args.prebuilt_mmap_dir, decimate_factor=args.decimate_factor, split="train")
    x = ds[0]
    print(f"Dataset size: {len(ds)}, sample shape: {x.shape}")
    # TODO: diffusion training loop
    print("Diffusion training loop not yet implemented.")


if __name__ == "__main__":
    main()
