#!/usr/bin/env python3
"""
Ablation: systematically reduce TCN model size (levels × nhid) until ~1000 params.

Usage:
  python ablation_model_sizes.py                    # print table and ablation sequence
  python ablation_model_sizes.py --list             # print only levels nhid params for run script
  python ablation_model_sizes.py --target 500       # minimum param target (default 1000)

Baseline: levels=4, nhid=80 → ~877k params. We pick configs that roughly halve each step.

Param count is computed with a closed form for PreNorm TCN (no need to load PyTorch).
"""

from __future__ import annotations

import argparse

INPUT_CHANNELS = 160
KERNEL_SIZE = 15


def count_params(levels: int, nhid: int) -> int:
    """
    PreNorm TCN param count: channel_sizes = [nhid]*levels, input_channels=160.
    Block 0: InstanceNorm(160), Conv(160, nhid, k), InstanceNorm(nhid), Conv(nhid, nhid, k), Down(160, nhid).
    Block i>0: InstanceNorm(nhid), Conv(nhid, nhid, k), InstanceNorm(nhid), Conv(nhid, nhid, k).
    Linear(nhid, 1).
    """
    k = KERNEL_SIZE
    in_ch = INPUT_CHANNELS
    # Block 0
    n0 = 2 * in_ch + (in_ch * nhid * k + nhid) + 2 * nhid + (nhid * nhid * k + nhid)
    if in_ch != nhid:
        n0 += in_ch * nhid * 1 + nhid
    # Blocks 1..levels-1
    n_block = 2 * nhid + (nhid * nhid * k + nhid) + 2 * nhid + (nhid * nhid * k + nhid)
    # Linear
    n_lin = nhid * 1 + 1
    return n0 + (levels - 1) * n_block + n_lin


def main():
    p = argparse.ArgumentParser(description="TCN ablation: param count vs levels/nhid")
    p.add_argument("--list", action="store_true", help="Print only levels nhid params for run script")
    p.add_argument("--target", type=int, default=1000, help="Target min param count")
    p.add_argument("--baseline-levels", type=int, default=4)
    p.add_argument("--baseline-nhid", type=int, default=80)
    args = p.parse_args()

    baseline_params = count_params(args.baseline_levels, args.baseline_nhid)
    if not args.list:
        print(f"Baseline: levels={args.baseline_levels}, nhid={args.baseline_nhid} → {baseline_params:,} params\n")

    # Candidate grid
    candidates = []
    for L in range(1, args.baseline_levels + 1):
        for H in [80, 56, 40, 28, 20, 14, 10, 8, 6, 5, 4, 3, 2, 1]:
            if H < 1:
                continue
            n = count_params(L, H)
            if n >= args.target:
                candidates.append((L, H, n))
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Target sequence: baseline, baseline/2, baseline/4, ... down to ~args.target
    targets = []
    t = baseline_params
    while t > args.target:
        targets.append(t)
        t = max(args.target, t // 2)
    targets.append(args.target)
    targets = sorted(set(targets), reverse=True)

    # For each target, pick config with smallest params >= target (or closest)
    chosen = []
    used = set()
    for target in targets:
        best = None
        best_diff = float("inf")
        for L, H, n in candidates:
            if (L, H) in used:
                continue
            diff = n - target if n >= target else target - n
            if diff < best_diff:
                best_diff = diff
                best = (L, H, n)
        if best and best[2] >= args.target:
            chosen.append(best)
            used.add((best[0], best[1]))

    seen = set()
    configs = []
    for L, H, N in chosen:
        if (L, H) not in seen:
            seen.add((L, H))
            configs.append((L, H, N))
    configs.sort(key=lambda x: -x[2])

    if args.list:
        for L, H, N in configs:
            print("{}  {}  {}".format(L, H, N))
        return

    print("Ablation sequence (roughly halving params each step):")
    print("levels  nhid   params")
    print("-" * 28)
    for L, H, N in configs:
        print("{:6}  {:4}  {:>10,}".format(L, H, N))
    print("\nUse with: --levels L --nhid H (e.g. run_ablation_small_models.sh)")
  if configs and configs[-1][2] > args.target * 1.5:
    print("\nNote: smallest config is ~{} params; reaching ~{} would need smaller kernel or different arch.".format(configs[-1][2], args.target))


if __name__ == "__main__":
    main()
