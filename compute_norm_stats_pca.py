#!/usr/bin/env python3
"""
Compute per-component normalization statistics over the **whole** training split
for a given PCA n_component, and save to norm_stats_pca{N}.npz.

Use this to precompute stats from all training shots (not a subsample) so
training can load them with --norm-stats norm_stats_pca{N}.npz.

Example:
  python compute_norm_stats_pca.py --pca-components 1
  python compute_norm_stats_pca.py --pca-components 8 --output my_norm_pca8.npz
"""

import argparse
from pathlib import Path

from dataset_ecei_tcn import ECEiTCNDataset


def main():
    p = argparse.ArgumentParser(description='Compute norm stats over full train split for PCA data')
    p.add_argument('--pca-components', type=int, required=True, choices=[1, 4, 8, 16],
                   help='Number of PCA components (dirs dsrpt_decimated_pca{N}, clear_decimated_pca{N})')
    p.add_argument('--base', type=str,
                   default='/home/idies/workspace/Storage/yhuang2/persistent/ecei',
                   help='Base path; disruptive meta from {base}/dsrpt')
    p.add_argument('--output', type=str, default=None,
                   help='Output .npz path (default: norm_stats_pca{N}.npz)')
    p.add_argument('--split', type=str, default='train',
                   help='Split to use for computing stats (default: train)')
    args = p.parse_args()

    base = Path(args.base)
    root = base / 'dsrpt'
    decimated_root = base / f'dsrpt_decimated_pca{args.pca_components}'
    clear_root = base / f'clear_decimated_pca{args.pca_components}'
    output = args.output or f'norm_stats_pca{args.pca_components}.npz'

    if not root.exists() or not (root / 'meta.csv').exists():
        raise FileNotFoundError(f'Meta not found: {root / "meta.csv"}')
    if not decimated_root.exists():
        raise FileNotFoundError(f'Decimated PCA dir not found: {decimated_root}')

    # Build dataset with normalize=True but stats path that does not exist yet,
    # and norm_max_shots=None so we use the entire split.
    ds = ECEiTCNDataset(
        root=str(root),
        decimated_root=str(decimated_root),
        clear_root=str(clear_root) if clear_root.exists() else None,
        clear_decimated_root=str(clear_root) if clear_root.exists() else None,
        normalize=True,
        norm_stats_path=output,
        norm_train_split=args.split,
        norm_max_shots=None,  # whole dataset
        n_input_channels=args.pca_components,
    )
    # Dataset constructor already computed and saved when path didn't exist
    print(f'Norm stats saved to {output}  (mean/std shape {ds.norm_mean.shape})')


if __name__ == '__main__':
    main()
