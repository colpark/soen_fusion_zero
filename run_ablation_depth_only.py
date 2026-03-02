#!/usr/bin/env python3
"""
Run only the depth-at-~1k ablation configs: L=2,3,4 with H=1, K=3 (~982–1006 params).
Same training setup as run_ablation_small_models.py, but only these three configs.

Usage:
  python run_ablation_depth_only.py
  python run_ablation_depth_only.py --epochs 100
  NGPUS=2 python run_ablation_depth_only.py

Checkpoints: checkpoints_tcn_ddp_original/<suffix>_L<L>_H1_K3/
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Only these (levels, nhid, kernel): depth ~1k comparison
DEPTH_ONLY_CONFIGS = [(2, 1, 982, 3), (3, 1, 994, 3), (4, 1, 1006, 3)]


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run depth-only ablation (L2/L3/L4 @ ~1k params)")
    p.add_argument("--ngpus", type=int, default=None,
                   help="Number of GPUs (default: env NGPUS or 4)")
    p.add_argument("--checkpoint-suffix", type=str, default="ablation_depth",
                   help="Checkpoint dir suffix (default: ablation_depth)")
    p.add_argument("extra", nargs="*", help="Extra args to training (e.g. --epochs 100)")
    args = p.parse_args()

    ngpus = args.ngpus or int(os.environ.get("NGPUS", "4"))
    checkpoint_suffix = args.checkpoint_suffix or os.environ.get("CHECKPOINT_SUFFIX", "ablation_depth")
    extra = args.extra

    env = os.environ.copy()
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("OMP_NUM_THREADS", "4")

    train_script = SCRIPT_DIR / "train_tcn_ddp_original.py"
    torchrun_cmd = ["torchrun", "--standalone", f"--nproc_per_node={ngpus}", str(train_script)]
    base_args = [
        "--flattop-only",
        "--use-prenorm",
        "--lr-schedule", "cosine_warmup",
        "--warmup-epochs", "5",
        "--batch-size", "16",
        "--lr", "0.0005",
        "--min-lr", "0.00001",
        "--no-checkpoint-by-time",
        "--prebuilt-mmap-dir", "subseqs_original_mmap",
    ]

    print("════════════════════════════════════════════════════════════════")
    print("  Ablation: depth only (L=2,3,4 @ H=1 K=3, ~1k params)")
    print(f"  GPUs: {ngpus}  |  Extra args: {extra}")
    print("  Configs: L2 H1 K3, L3 H1 K3, L4 H1 K3")
    print("════════════════════════════════════════════════════════════════")

    for idx, (L, H, N, K) in enumerate(DEPTH_ONLY_CONFIGS, start=1):
        ckpt_dir = f"checkpoints_tcn_ddp_original/{checkpoint_suffix}_L{L}_H{H}_K{K}"
        cmd_args = (
            base_args
            + ["--levels", str(L), "--nhid", str(H), "--kernel-size", str(K)]
            + ["--checkpoint-dir", ckpt_dir]
            + extra
        )
        cmd = torchrun_cmd + cmd_args
        print("")
        print(f"────────────────── Run {idx}/3: L{L} H{H} K{K} ({N} params) ──────────────────")
        subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)

    print("")
    print("════════════════════════════════════════════════════════════════")
    print("  Depth-only ablation complete.")
    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
