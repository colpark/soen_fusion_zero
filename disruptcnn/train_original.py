#!/usr/bin/env python
"""
Training script that uses the original DisruptCNN data setting (dataset_original.py).

Uses shot list segment/label logic (flattop_only, tend = max(tdisrupt, min(tlast, t_flat_stop)), Twarn=300 ms)
and runs the same training loop as main.py.

By default uses **disrupt data only** (no clear shot list), since clear is not in the original data list.

Usage (from soen_fusion_zero project root or with PYTHONPATH):
  python -m disruptcnn.train_original [OPTIONS]
  python -m disruptcnn.train_original --data-root /path/to/data --disrupt-file disruptcnn/shots/d3d_disrupt_ecei.final.txt

Options: same as main.py (--epochs, --batch-size, --flattop-only, etc.).
  --disrupt-only is added by default; pass --no-disrupt-only to use a clear list if you have one.
"""

import sys


def main():
    if "--use-original-dataloader" not in sys.argv:
        sys.argv.insert(1, "--use-original-dataloader")
    if "--disrupt-only" not in sys.argv and "--no-disrupt-only" not in sys.argv:
        sys.argv.insert(1, "--disrupt-only")
    from disruptcnn.main import main as run_main
    run_main()


if __name__ == "__main__":
    main()
