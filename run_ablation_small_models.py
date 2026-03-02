#!/usr/bin/env python3
"""
Run ablation: train TCN for each (levels, nhid, kernel) from ablation_model_sizes.
No sbatch — runs directly via subprocess(torchrun + train_tcn_ddp_original.py).

Usage:
  python run_ablation_small_models.py
  python run_ablation_small_models.py --epochs 100
  python run_ablation_small_models.py --run-index 3    # run only config index (1-based)
  NGPUS=2 python run_ablation_small_models.py         # fewer GPUs

Checkpoints: checkpoints_tcn_ddp_original/<CHECKPOINT_SUFFIX>_L<L>_H<H>[_K<K>]/
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def get_configs():
    """Return list of (levels, nhid, params, kernel) from ablation_model_sizes.py --list."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "ablation_model_sizes.py"), "--list"],
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
    )
    result.check_returncode()
    configs = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        L, H, N = int(parts[0]), int(parts[1]), int(parts[2])
        K = int(parts[3]) if len(parts) > 3 else 15
        configs.append((L, H, N, K))
    return configs


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run ablation (no sbatch)")
    p.add_argument("--run-index", type=int, default=None,
                   help="Run only this config index (1-based), e.g. for SLURM array")
    p.add_argument("--ngpus", type=int, default=None,
                   help="Number of GPUs (default: env NGPUS or 4)")
    p.add_argument("--checkpoint-suffix", type=str, default=None,
                   help="Checkpoint dir suffix (default: env CHECKPOINT_SUFFIX or 'ablation')")
    p.add_argument("extra", nargs="*", help="Extra args passed to training (e.g. --epochs 100)")
    args = p.parse_args()

    ngpus = args.ngpus or int(os.environ.get("NGPUS", "4"))
    checkpoint_suffix = args.checkpoint_suffix or os.environ.get("CHECKPOINT_SUFFIX", "ablation")
    run_index = args.run_index
    if run_index is None and os.environ.get("RUN_ABLATION_INDEX"):
        run_index = int(os.environ["RUN_ABLATION_INDEX"])
    extra = args.extra

    configs = get_configs()
    if not configs:
        print("ablation_model_sizes.py --list returned no configs", file=sys.stderr)
        sys.exit(1)

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
    print("  Ablation: small models (baseline → ~1000 params, incl. small kernel)")
    print(f"  GPUs: {ngpus}  |  Extra args: {extra}")
    print("  Configs:")
    for i, (L, H, N, K) in enumerate(configs, start=1):
        k_str = f" kernel={K}" if K != 15 else ""
        print(f"    {i}: levels={L} nhid={H}{k_str}")
    print("════════════════════════════════════════════════════════════════")

    for idx, (L, H, N, K) in enumerate(configs, start=1):
        if run_index is not None and idx != run_index:
            continue

        ckpt_dir = f"checkpoints_tcn_ddp_original/{checkpoint_suffix}_L{L}_H{H}"
        if K != 15:
            ckpt_dir += f"_K{K}"

        cmd_args = (
            base_args
            + ["--levels", str(L), "--nhid", str(H)]
            + (["--kernel-size", str(K)] if K != 15 else [])
            + ["--checkpoint-dir", ckpt_dir]
            + extra
        )
        cmd = torchrun_cmd + cmd_args

        print("")
        print(f"────────────────── Ablation run {idx}: levels={L} nhid={H} kernel={K} ──────────────────")
        subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)

    print("")
    print("════════════════════════════════════════════════════════════════")
    print("  Ablation complete.")
    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
