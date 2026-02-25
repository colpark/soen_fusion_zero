#!/usr/bin/env python
"""
Training script that uses the original DisruptCNN data setting only.

Segment and label logic: shot list files (d3d_*_ecei.final.txt), flattop_only,
tend = max(tdisrupt, min(tlast, t_flat_stop)), Twarn=300 ms. See dataset_original.py.

Usage:
  python -m disruptcnn.train_original [main.py args...]
  python -m disruptcnn.train_original --data-root /path/to/data --epochs 10

This script adds --use-original-dataloader and then runs the same training loop as main.py.
"""

import sys


def main():
    # Prepend the flag so the original-dataloader path is used when main parses args
    if "--use-original-dataloader" not in sys.argv:
        sys.argv.insert(1, "--use-original-dataloader")
    from disruptcnn.main import main as run_main
    run_main()


if __name__ == "__main__":
    main()
