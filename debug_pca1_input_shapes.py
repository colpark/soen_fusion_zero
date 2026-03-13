#!/usr/bin/env python3
"""
Debug PCA1 TCN input shapes: trace data from H5 → dataset → wrapper → batch → view → model input.

Run from soen_fusion_zero (same as run_tcn_baseline_pca1_decimated_subsample.sh):
  python debug_pca1_input_shapes.py
  python debug_pca1_input_shapes.py --decimated-root /path/to/dsrpt_decimated_pca1 --norm-stats ./norm_stats_pca1.npz

No GPU or torchrun needed. Uses first available shot and first subsequence index.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from soen_fusion_zero
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

def main():
    p = argparse.ArgumentParser(description="Debug PCA1 input shapes through the pipeline")
    p.add_argument("--root", type=str, default="/home/idies/workspace/Storage/yhuang2/persistent/ecei/dsrpt",
                   help="Root for shot list (disrupt)")
    p.add_argument("--decimated-root", type=str, default="/home/idies/workspace/Storage/yhuang2/persistent/ecei/dsrpt_decimated_pca1")
    p.add_argument("--norm-stats", type=str, default=None,
                   help="norm_stats_pca1.npz path (default: script_dir/norm_stats_pca1.npz)")
    p.add_argument("--nsub", type=int, default=781300)
    p.add_argument("--data-step", type=int, default=10)
    p.add_argument("--decimate-extra", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=3, help="Number of samples to trace (1 batch)")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    norm_stats = args.norm_stats or str(script_dir / "norm_stats_pca1.npz")
    disrupt_file = str(script_dir / "disruptcnn" / "shots" / "d3d_disrupt_ecei.final.txt")

    print("=" * 60)
    print("PCA1 input shape debug")
    print("=" * 60)
    print(f"  decimated_root   = {args.decimated_root}")
    print(f"  norm_stats       = {norm_stats}")
    print(f"  nsub={args.nsub} data_step={args.data_step} decimate_extra={args.decimate_extra}")
    print()

    # 1) Inspect raw H5 LFS shape (first available file)
    import h5py
    dec_root = Path(args.decimated_root)
    if not dec_root.exists():
        print(f"[SKIP] decimated_root does not exist: {dec_root}")
        h5_shape = None
    else:
        h5_files = list(dec_root.glob("*.h5"))
        if not h5_files:
            print(f"[SKIP] No .h5 files in {dec_root}")
            h5_shape = None
        else:
            one_file = h5_files[0]
            with h5py.File(one_file, "r") as f:
                if "LFS" not in f:
                    print(f"[WARN] No 'LFS' in {one_file.name}; keys: {list(f.keys())}")
                    h5_shape = None
                else:
                    LFS = f["LFS"]
                    h5_shape = LFS.shape
                    # Same slice as _read_data for first segment: we need start_idxi, stop_idxi, step
                    # Use a plausible slice: 0:78130:10 -> 7813
                    seg_len_file = 78130
                    step = args.decimate_extra
                    slice_obj = np.s_[..., 0:seg_len_file:step]
                    sample_slice = LFS[slice_obj]
                    print("1) Raw H5 (first file)")
                    print(f"   File: {one_file.name}")
                    print(f"   LFS.shape = {h5_shape}")
                    print(f"   LFS[..., 0:78130:10].shape = {sample_slice.shape}")
                    print()

    # 2) Build dataset (same as train script)
    from disruptcnn.dataset_original import EceiDatasetOriginal, OriginalStyleDatasetForDDP

    # nrecept for tiling: trainer passes nrecept_raw = nrecept * data_step = 3001*10 = 30010
    nrecept_raw = 3001 * 10  # 30010

    try:
        inner = EceiDatasetOriginal(
            root=args.root,
            disrupt_file=disrupt_file,
            clear_file=None,
            flattop_only=True,
            normalize=Path(norm_stats).exists(),
            data_step=args.data_step,
            nsub=args.nsub,
            nrecept=nrecept_raw,
            decimated_root=args.decimated_root,
            clear_decimated_root=None,
            norm_stats_path=norm_stats,
            decimate_extra=args.decimate_extra,
        )
    except Exception as e:
        print(f"2) EceiDatasetOriginal init failed: {e}")
        return

    print("2) EceiDatasetOriginal")
    print(f"   len(inner) = {len(inner)}")
    print(f"   inner.nsub = {inner.nsub}, _step_in_getitem = {getattr(inner, '_step_in_getitem', 1)}")
    print(f"   Expected output length per window = nsub // step = {inner.nsub // getattr(inner, '_step_in_getitem', 1)}")
    print()

    if len(inner) == 0:
        print("No subsequences; cannot trace __getitem__.")
        return

    # 3) One sample from inner (before wrapper)
    idx = 0
    try:
        # Call _read_data directly to see raw read shape
        X_raw = inner._read_data(idx)
        print("3) inner._read_data(0) -> X shape (before wrapper)")
        print(f"   X.shape = {X_raw.shape}  (expected (1, 7813) or (C, T))")
        print()
    except Exception as e:
        print(f"3) inner._read_data(0) failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4) Full __getitem__ from inner (what wrapper receives)
    X_inner, target_inner, _, weight_inner = inner[idx]
    print("4) inner[0] (X, target, _, weight)")
    print(f"   X.shape = {X_inner.shape}")
    print(f"   target.shape = {target_inner.shape}")
    print()

    # 5) Wrapper
    wrapper = OriginalStyleDatasetForDDP(inner)
    print("5) OriginalStyleDatasetForDDP")
    print(f"   _T_fixed = {wrapper._T_fixed}")
    X_wrap, target_wrap, weight_wrap = wrapper[idx]
    print(f"   wrapper[0] -> X.shape = {X_wrap.shape}")
    print()

    # 6) Batch (simulate DataLoader)
    num_samples = min(args.num_samples, len(wrapper))
    batch_X = torch.stack([wrapper[i][0] for i in range(num_samples)], dim=0)
    batch_target = torch.stack([wrapper[i][1] for i in range(num_samples)], dim=0)
    print("6) Batched (stack of wrapper[i][0])")
    print(f"   batch_X.shape = {batch_X.shape}")
    print()

    # 7) Training view + layout fix (same as train_tcn_ddp_original.py)
    B = batch_X.shape[0]
    X = batch_X.view(B, -1, batch_X.shape[-1])
    print("7) After view(B, -1, X.shape[-1])")
    print(f"   X.shape = {X.shape}  (B, C, T) desired (B, 1, 7813)")
    print()

    input_channels = 1
    if input_channels is not None and X.dim() == 3:
        if X.shape[2] == input_channels and X.shape[1] != input_channels:
            X = X.permute(0, 2, 1)
            print("8) After permute(0, 2, 1) [ (B,T,C) -> (B,C,T) ]")
        elif input_channels == 1 and X.shape[1] != 1:
            X = X[:, 0:1, :]
            print("8) After X[:, 0:1, :] [ force 1 channel ]")
        else:
            print("8) No layout fix applied")
    print(f"   Final X.shape = {X.shape}  -> model expects (B, 1, 7813)")
    print()

    if X.shape[1] != 1 or X.shape[2] != 7813:
        print(">>> MISMATCH: model expects (B, 1, 7813). Fix dataset/wrapper or layout logic.")
    else:
        print(">>> OK: shape (B, 1, 7813) ready for model.")
    print("=" * 60)

if __name__ == "__main__":
    main()
