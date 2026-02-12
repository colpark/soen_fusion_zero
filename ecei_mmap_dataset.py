#!/usr/bin/env python3
"""
Example PyTorch Dataset for loading memory-mapped chunks.

This demonstrates how to use the converted data for neural network training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from mmap_ninja import numpy as np_ninja
from mmap_ninja.ragged import RaggedMmap
from pathlib import Path
from typing import Optional, Callable


class ChunkedMemMapDataset(Dataset):
    """PyTorch Dataset for loading chunks from memory-mapped files."""

    def __init__(self,
                 mmap_path: str,
                 split: str,
                 dsrpt_threshold: int,
                 clear_threshold: int = None):
        """
        Initialize the dataset.

        Args:
            mmap_path: Path to memory map file (train.mmap and test.mmap)
            split: 'train' or 'test'
            dsrpt_threshold: Chunks with time_to_end <= dsrpt_threshold
                are labeled as class 1
            clear_threshold: Chunks with time_to_end > clear_threshold
                are labeled as class 0.
                If not provided, clear_threshold == dsrpt_threshold
        """
        self.mmap_path = Path(mmap_path)

        # Load and process metadata
        meta = pd.read_csv(self.mmap_path/'meta.csv')
        meta = meta[meta['split'] == split].reset_index(drop=True)

        if clear_threshold is None:
            clear_threshold = dsrpt_threshold

        meta = meta[ (meta['time_to_end'] <= dsrpt_threshold) |
                     (meta['time_to_end'] >  clear_threshold) ]
        meta['label'] = 0
        meta.loc[meta['time_to_end'] <= dsrpt_threshold, 'label'] = 1

        self.meta = meta

        num_dsrpt = meta['label'].sum()
        num_clear = len(meta) - num_dsrpt

        # Load memory-mapped data
        # self.data = np_ninja.open_existing(str(self.mmap_path/f'{split}.mmap'))
        self.data = RaggedMmap(str(self.mmap_path/f'{split}.mmap'))


        print(f"Loaded {split} dataset:")
        print(f"\tChunks: {len(self.meta)}")
        print(f'\tClear: {num_clear}')
        print(f'\tDsrpt: {num_dsrpt}')

    def __len__(self) -> int:
        """Return the number of chunks."""
        return len(self.meta)

    def __getitem__(self, idx: int):
        """
        Get a chunk and optionally its label.

        Args:
            idx: Index of the chunk

        Returns:
            If time_to_end_threshold is None: chunk tensor
            Otherwise: (chunk tensor, label)
        """
        # Get the global index and label for this chunk
        global_idx = self.meta.iloc[idx]['global_idx']
        label = self.meta.iloc[idx]['label']

        # Load chunk from memory map
        chunk = self.data[global_idx]

        # Convert to tensor
        chunk = torch.from_numpy(chunk).float()

        return chunk, label


def example_usage():
    """Example of how to use the dataset."""

    # Paths
    MMAP_PATH = "dsrpt_mmap"
    DSRPT_THRESHOLD = 300000,
    CLEAR_THRESHOLD = 600000,

    # Create datasets
    train_dataset = ChunkedMemMapDataset(
        mmap_path       = MMAP_PATH,
        split           = 'train',
        dsrpt_threshold = DSRPT_THRESHOLD,
        clear_threshold = CLEAR_THRESHOLD,
    )

    test_dataset = ChunkedMemMapDataset(
        mmap_path       = MMAP_PATH,
        split           = 'test',
        dsrpt_threshold = DSRPT_THRESHOLD,
        clear_threshold = CLEAR_THRESHOLD,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = 4,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = 4,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True
    )

    # Example training loop snippet
    print("\nExample batch:")
    for chunks, labels in train_loader:
        print(f"Batch shape: {chunks.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")


if __name__ == "__main__":
    example_usage()
