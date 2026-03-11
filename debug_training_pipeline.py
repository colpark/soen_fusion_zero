#!/usr/bin/env python3
"""
Interpretable debugging script for the TCN training pipeline (rolled-back version).

Walks through: input/dataset formation, normalization, labels (with segments for
disruptive data), model and per-layer receptive fields, how prediction is made,
and how loss weighting is applied. Uses real dataset if paths are provided,
otherwise runs with synthetic data so the script always runs.

Usage:
  python debug_training_pipeline.py
  python debug_training_pipeline.py --root /path/to/dsrpt --decimated-root /path/to/dsrpt_decimated --clear-root /path/to/clear_decimated --clear-decimated-root /path/to/clear_decimated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root so we can import dataset and model
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_ecei_tcn import ECEiTCNDataset
from train_tcn_ddp import (
    build_model,
    batch_weights,
    calc_receptive_field,
)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def subsection(title: str) -> None:
    print(f"\n--- {title} ---\n")


# ── Default training params (match train_tcn_ddp.py) ────────────────────────
DATA_STEP = 10
NSUB_RAW = 781_250
TWARN = 300_000
BASELINE_LEN = 40_000
EXCLUDE_LAST_MS = 0.0
# Model
INPUT_CHANNELS = 160
LEVELS = 4
NHID = 80
KERNEL_SIZE = 15
DILATION_BASE = 10
DROPOUT = 0.2
NRECEPT_TARGET = 30_000


def main():
    p = argparse.ArgumentParser(description="Debug TCN training pipeline")
    p.add_argument("--root", type=str, default=None, help="Disruptive shots root (meta.csv + .h5)")
    p.add_argument("--decimated-root", type=str, default=None)
    p.add_argument("--clear-root", type=str, default=None)
    p.add_argument("--clear-decimated-root", type=str, default=None)
    p.add_argument("--norm-stats", type=str, default=None)
    p.add_argument("--pca-components", type=int, default=0, choices=[0, 1, 4, 8, 16])
    args = p.parse_args()

    use_real_data = args.root is not None and Path(args.root).exists()

    # ─────────────────────────────────────────────────────────────────────
    #  1. INPUT — DATA LAYOUT AND HOW THE DATASET FORMS SUBSEQUENCES
    # ─────────────────────────────────────────────────────────────────────
    section("1. INPUT — Data layout and how the dataset forms subsequences")

    print("""
Raw data (disruptive shots):
  root/
    meta.csv     columns: shot, split, t_disruption (ms), ...
    {shot}.h5    key 'LFS': shape (20, 8, T) at 1 MHz, or (C, T) for PCA with C in {1,4,8,16}

Optional clear (non-disruptive) shots:
  clear_root/   same layout; whole shot is labeled 0 (clear).

Decimated data (when --decimated-root is used):
  Pre-decimated HDF5: LFS shape (20, 8, T/data_step) or (C, T/data_step).
  Offset already removed; no decimation in __getitem__. data_step=10 → 100 kHz.
""")

    print("Time parameters (all in raw 1 MHz samples unless noted):")
    print(f"  Twarn           = {TWARN:,}  ({TWARN/1e6*1000:.1f} ms before disruption)")
    print(f"  baseline_length = {BASELINE_LEN:,}  (first N samples used for DC offset)")
    print(f"  data_step       = {DATA_STEP}  (decimation: keep every Nth sample → 100 kHz)")
    print(f"  nsub            = {NSUB_RAW:,}  (~{NSUB_RAW/1e6*1000:.1f} ms window)")
    T_sub = NSUB_RAW // DATA_STEP
    print(f"  → T_sub         = nsub // data_step = {T_sub:,}  (output length per subsequence)")
    print()

    print("Subsequence tiling (per shot):")
    print("  Each shot is split into windows of length nsub (in data space: nsub//q with q=data_step if decimated).")
    print("  Windows start at 'start_idx', then start += stride until start + nsub > stop_idx.")
    print("  stride (raw) = (nsub//data_step - nrecept + 1) * data_step  (nrecept from model below).")
    print("  DataLoader yields (X, target, weight) with X (B, C, T_sub), target (B, T_sub), weight (B, T_sub).")
    print("  For each window we store: shot_idx, start, stop, disrupt_local, positive_end_local.")
    print("  disrupt_local:  index (in window) of start of Twarn; -1 if window is fully clear.")
    print("  positive_end_local:  end index of 'label 1' region in the window.")
    print()

    if use_real_data:
        subsection("Real dataset — building and one sample")
        # Stride must match training; we use default nrecept for stride formula
        _, nrecept_init, _ = build_model(
            INPUT_CHANNELS if args.pca_components == 0 else args.pca_components,
            1, LEVELS, NHID, KERNEL_SIZE, DILATION_BASE, DROPOUT,
            nrecept_target=NRECEPT_TARGET,
        )
        stride_raw = (NSUB_RAW // DATA_STEP - nrecept_init + 1) * DATA_STEP
        ds = ECEiTCNDataset(
            root=args.root,
            decimated_root=args.decimated_root or None,
            clear_root=args.clear_root or None,
            clear_decimated_root=args.clear_decimated_root or None,
            Twarn=TWARN,
            baseline_length=BASELINE_LEN,
            data_step=DATA_STEP,
            nsub=NSUB_RAW,
            stride=stride_raw,
            normalize=True,
            norm_stats_path=args.norm_stats or ("norm_stats_pca1.npz" if args.pca_components else "norm_stats.npz"),
            exclude_last_ms=EXCLUDE_LAST_MS,
            ignore_twarn=False,
            n_input_channels=args.pca_components if args.pca_components > 0 else None,
        )
        print(f"  Stride (raw): {stride_raw:,}")
        print(f"  Number of subsequences: {len(ds)}")
        print(f"  Subseq length (T_sub): {ds._T_sub:,}")
        print(f"  data_step in getitem: {ds._step_in_getitem}")
        # Get one disruptive and one clear if available
        disruptive_idx = np.where(ds.seq_has_disrupt)[0]
        clear_idx = np.where(~ds.seq_has_disrupt)[0]
        idx_dis = int(disruptive_idx[0]) if len(disruptive_idx) else 0
        idx_clear = int(clear_idx[0]) if len(clear_idx) else None
        X_d, t_d, w_d = ds[idx_dis]
        print(f"\n  Sample index {idx_dis} (disruptive):")
        print(f"    X shape: {X_d.shape}  (C, T_sub)")
        print(f"    target shape: {t_d.shape}")
        print(f"    weight shape: {w_d.shape}")
        if idx_clear is not None:
            X_c, t_c, w_c = ds[idx_clear]
            print(f"  Sample index {idx_clear} (clear):")
            print(f"    X shape: {X_c.shape}, target unique: {t_c.unique().tolist()}, weight unique: {w_c.unique().tolist()}")
    else:
        subsection("Synthetic data (no paths provided)")
        C = INPUT_CHANNELS if args.pca_components == 0 else args.pca_components
        X_syn = torch.randn(2, C, T_sub, dtype=torch.float32) * 0.5
        t_syn = torch.zeros(2, T_sub, dtype=torch.float32)
        t_syn[0, T_sub//2:T_sub//2+1000] = 1.0  # one positive segment
        w_syn = torch.ones(2, T_sub, dtype=torch.float32)
        print(f"  Synthetic X shape: (B=2, C={C}, T_sub={T_sub})")
        print(f"  Synthetic target: 0 everywhere except one segment of 1s (e.g. [T/2 : T/2+1000]).")
        print(f"  Synthetic weight: all 1.0 (for illustration).")

    # ─────────────────────────────────────────────────────────────────────
    #  2. NORMALIZATION AND LABELS
    # ─────────────────────────────────────────────────────────────────────
    section("2. Normalization and label definition")

    print("""
Preprocessing in __getitem__ (when reading from raw; skip if using pre-decimated):
  1. DC offset: baseline = mean(X[..., :baseline_length], axis=-1); X = X - baseline.
  2. Decimation: X = X[..., ::data_step]  → 1 MHz to 100 kHz.
  3. Z-score:   X = (X - norm_mean) / norm_std  (per-channel; norm_mean/norm_std from training split).
""")

    print("Label construction (per timestep, in output space):")
    print("  target[t] = 0 (clear) for t before the Twarn window; 1 (disruptive) inside Twarn.")
    print("  Twarn window: (t_disrupt - Twarn, t_disrupt - exclude_last_ms] in raw time.")
    print("  In __getitem__ we have disrupt_local (dl) and positive_end_local (el) in *data* space.")
    print("  Step to output space: step = data_step (if raw) or 1 (if decimated).")
    print("  d = (dl + 1) // step   (start of '1' in output steps; +1 matches disruptcnn boundary)")
    print("  e = (el + step - 1) // step   (end of '1' in output steps)")
    print("  target[d:e] = 1,  target elsewhere = 0.")
    print("  weight[0:d] = neg_weight,  weight[d:e] = pos_weight,  weight[e:T] = 0 (excluded).")
    print()

    subsection("Label segments for a disruptive subsequence")
    print("  Timeline (output steps 0 .. T_sub-1):")
    print("  ")
    print("     [ 0 ......... d-1 ] [ d ......... e-1 ] [ e ......... T_sub-1 ]")
    print("       label 0 (clear)     label 1 (Twarn)     excluded (weight=0)")
    print("       weight=neg_weight   weight=pos_weight   (e.g. last exclude_last_ms)")
    print("  ")
    print("  Example (T_sub=78125, d=40000, e=43000):")
    T_ex = 78125
    d_ex, e_ex = 40000, 43000
    print(f"    Steps [0, {d_ex})  → label 0 (clear),  count = {d_ex}")
    print(f"    Steps [{d_ex}, {e_ex}) → label 1 (disruptive), count = {e_ex - d_ex}")
    print(f"    Steps [{e_ex}, {T_ex}) → excluded (weight=0),  count = {T_ex - e_ex}")

    # ─────────────────────────────────────────────────────────────────────
    #  3. MODEL AND RECEPTIVE FIELD PER LAYER
    # ─────────────────────────────────────────────────────────────────────
    section("3. Model and per-layer receptive field")

    model, nrecept, dilation_sizes = build_model(
        INPUT_CHANNELS, 1, LEVELS, NHID,
        KERNEL_SIZE, DILATION_BASE, DROPOUT,
        nrecept_target=NRECEPT_TARGET,
    )
    print(f"  Target receptive field: {NRECEPT_TARGET:,} samples")
    print(f"  Achieved receptive field: {nrecept:,} samples")
    print(f"  Dilation sizes: {dilation_sizes}")
    print()

    subsection("Receptive field per layer")
    print("  Formula (causal): RF = 1 + 2 * (kernel_size - 1) * sum(dilations so far).")
    print("  Each TemporalBlock has two convs with same dilation; block i adds 2*(k-1)*dilation_i to the receptive field.")
    cum = 0
    for i, d in enumerate(dilation_sizes):
        add = 2 * (KERNEL_SIZE - 1) * d
        cum += add
        rf_layer = 1 + cum
        print(f"    Level {i}: dilation={d}, cumulative RF = 1 + {cum} = {rf_layer:,}")
    print(f"  Total (all levels): {calc_receptive_field(KERNEL_SIZE, dilation_sizes):,}")
    print()

    subsection("Output and receptive-field crop")
    print("  Model forward: input (B, C, T_sub) → TCN → linear → sigmoid → (B, T_sub).")
    print("  output[t] is causal: it only sees input[0..t]. So the first (nrecept - 1) outputs")
    print("  do not have full receptive field. Training and eval use only output[nrecept-1:]")
    print("  (and target/weight[nrecept-1:]), so each predicted step has full context.")
    print(f"  Usable length for loss = T_sub - (nrecept - 1) = {T_sub} - {nrecept - 1} = {T_sub - (nrecept - 1)}.")

    # ─────────────────────────────────────────────────────────────────────
    #  4. HOW PREDICTION IS MADE
    # ─────────────────────────────────────────────────────────────────────
    section("4. How the prediction is made")

    B, C = 2, INPUT_CHANNELS
    T_forward = min(T_sub, 4000)  # smaller T for dummy forward to avoid OOM
    X_dummy = torch.randn(B, C, T_forward) * 0.1
    try:
        out = model(X_dummy)
        print(f"  Input shape:  (B={B}, C={C}, T={T_forward})")
        print(f"  Output shape: {out.shape}  (B, T)")
        out_crop = out[:, nrecept - 1:]
        print(f"  After crop out[:, nrecept-1:]: shape {out_crop.shape}")
    except Exception as e:
        print(f"  (Dummy forward skipped: {e})")
        print(f"  In training: input (B, C, T_sub={T_sub}) -> model -> (B, T_sub), crop -> (B, T_sub - nrecept + 1).")
    print("  Prediction at threshold 0.5: pred = (out_crop >= 0.5).float()")

    # ─────────────────────────────────────────────────────────────────────
    #  5. HOW WEIGHTING IS DONE (CRYSTAL CLEAR)
    # ─────────────────────────────────────────────────────────────────────
    section("5. Loss weighting (current rolled-back version)")

    print("""
The dataset returns (X, target, weight). The third tensor 'weight' holds
per-timestep BCE weights (pos_weight on positive steps, neg_weight on negative,
0 on excluded). In the current training code this is NOT used in the loss.

Training uses only batch_weights(tgt_v):
  wgt_v = batch_weights(tgt_v)   # tgt_v = target[:, nrecept-1:]
  loss = F.binary_cross_entropy(out_v, tgt_v, weight=wgt_v)

batch_weights(tgt_v) — per-batch 50/50 balance:
  n_total = numel(tgt_v),  n_pos = sum(tgt_v),  n_neg = n_total - n_pos
  For positive timesteps: weight = 0.5 * n_total / n_pos
  For negative timesteps: weight = 0.5 * n_total / n_neg
  So total weight on positives = 0.5*n_total, total on negatives = 0.5*n_total.
""")

    subsection("Numerical example: batch_weights")
    # Small example
    tgt = torch.tensor([0., 0, 0, 0, 0, 1., 1, 0, 0, 0.], dtype=torch.float32)
    w = batch_weights(tgt)
    n_total = tgt.numel()
    n_pos = int(tgt.sum().item())
    n_neg = n_total - n_pos
    pw = 0.5 * n_total / max(n_pos, 1)
    nw = 0.5 * n_total / max(n_neg, 1)
    print(f"  tgt (10 steps): {tgt.tolist()}")
    print(f"  n_pos={n_pos}, n_neg={n_neg}, n_total={n_total}")
    print(f"  pos_weight per step = 0.5*{n_total}/{n_pos} = {pw:.4f}")
    print(f"  neg_weight per step = 0.5*{n_total}/{n_neg} = {nw:.4f}")
    print(f"  batch_weights(tgt): pos steps -> {w[tgt==1].unique().item():.4f}, neg steps -> {w[tgt==0].unique().item():.4f}")
    print(f"  Sum of weights on pos: {(w[tgt==1].sum().item()):.4f}, on neg: {(w[tgt==0].sum().item()):.4f}")

    subsection("Full training step (conceptual)")
    print("  1. Get batch (X, target, _weight); _weight is ignored.")
    print("  2. out = model(X)  → (B, T_sub)")
    print("  3. out_v = out[:, nrecept-1:],  tgt_v = target[:, nrecept-1:]")
    print("  4. wgt_v = batch_weights(tgt_v)")
    print("  5. loss = BCE(out_v, tgt_v, weight=wgt_v)")
    print("  6. loss.backward(), clip_grad_norm_, optimizer.step()")
    print("\n  Do not multiply dataset weight by batch_weights (double weighting causes collapse).")

    print("\n" + "=" * 80)
    print("  Debug pipeline summary complete.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
