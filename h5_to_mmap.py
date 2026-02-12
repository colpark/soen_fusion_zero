#!/usr/bin/env python3
"""
Convert HDF5 files to memory-mapped format for fast neural network training.

This script:
1. Loads HDF5 files with key 'LFS' containing data of shape (T, H, W)
2. Processes data by baseline subtraction
3. Chunks data into overlapping segments from the end
4. Generates metadata CSV with train/test split and time-to-end information
5. Saves chunks to memory-mapped files using mmap_ninja
"""

from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import h5py
from mmap_ninja import numpy as np_ninja
from mmap_ninja.ragged import RaggedMmap


class HDF5ToMemMapConverter:
    """
    Converts HDF5 files to memory-mapped format with chunking.
    """

    def __init__(self,
                 hdf5_dir: str,
                 meta_path: str,
                 output_dir: str,
                 chunk_length: int,
                 chunk_step: int,
                 baseline_length: int = 50000):
        """
        Initialize the converter.

        Args:
            hdf5_dir: Directory containing HDF5 files
            meta_path: Path to CSV file with 'td' and 'split' columns
            output_dir: Directory to save memory maps and metadata
            chunk_length: Length L of each chunk
            chunk_step: Step length between chunks
            baseline_length: Length of baseline section (default 50,000)
        """
        self.hdf5_dir   = Path(hdf5_dir)
        self.meta_path  = Path(meta_path)
        self.output_dir = Path(output_dir)

        self.chunk_length    = chunk_length
        self.chunk_step      = chunk_step
        self.baseline_length = baseline_length

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata CSV
        self.meta = pd.read_csv(meta_path).set_index('shot')
        self.meta.index = self.meta.index.astype(str)
        print(f"Loaded metadata for {len(self.meta)} files")

    def load_and_process_hdf5(self, hdf5_path: Path) -> Tuple[np.ndarray, str]:
        """
        Load HDF5 file and process data with baseline subtraction.

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            Tuple of (processed_data_1, split)
        """
        stem = hdf5_path.stem

        # Get metadata for this file
        if stem not in self.meta.index:
            raise ValueError(f"File {stem} not found in metadata CSV")

        # time of disruption is given as milliseconds
        t_dis = int(self.meta.loc[stem, 't_disruption'] * 1000)
        split = self.meta.loc[stem, 'split']

        # Load HDF5 data
        with h5py.File(hdf5_path, 'r') as handle:
            data = handle['LFS'][:]  # Shape: (Height, Width, Time)

        # Extract baseline and meaningful data
        data_0 = data[..., :self.baseline_length]
        data_1 = data[..., self.baseline_length: self.baseline_length + t_dis]

        # Calculate baseline average and subtract from data_1
        baseline = np.mean(data_0, axis=-1, keepdims=True)  # Shape: (H, W, 1)
        data_1 = data_1 - baseline

        return data_1, split

    def create_chunks(self, data_1: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Create overlapping chunks from the end of data_1.

        Args:
            data_1: Processed meaningful data of shape (T, H, W)

        Returns:
            Tuple of (chunks_list, time_to_end_list)
        """
        length = data_1.shape[-1]

        chunks = []
        time_to_end = []

        # Start from the end and work backwards
        offset = 0
        while True:
            start_idx = length - self.chunk_length - offset
            end_idx = length - offset

            # Break if we don't have enough data for a full chunk
            if start_idx < 0:
                break

            chunk = data_1[..., start_idx:end_idx]

            chunks.append(chunk)
            time_to_end.append(offset)

            # Move to next chunk
            offset += self.chunk_step

        return chunks, time_to_end

    def process_all_files(self) -> Tuple[Dict[str, List], pd.DataFrame]:
        """
        Process all HDF5 files and create chunks.

        Returns:
            Tuple of (chunks_by_split, meta)
        """
        # Storage for chunks by split
        chunks_by_split = {'train': [], 'test': []}

        # Metadata for chunks
        chunk_metadata = {
            'shot'       : [],
            'chunk_idx'  : [],
            'time_to_end': [],
            'split'      : [],
            'global_idx' : []  # Index within the split
        }

        # Counters for global indices
        global_idx_counters = {'train': 0, 'test': 0}

        # Get all HDF5 files
        hdf5_files = sorted(self.hdf5_dir.glob('*.h5'))
        if len(hdf5_files) == 0:
            hdf5_files = sorted(self.hdf5_dir.glob('*.hdf5'))

        print(f"Found {len(hdf5_files)} HDF5 files")

        # Process each file
        for hdf5_path in tqdm(hdf5_files, desc="Processing HDF5 files"):
            try:
                # Load and process
                data_1, split = self.load_and_process_hdf5(hdf5_path)

                # Create chunks
                chunks, time_to_end_list = self.create_chunks(data_1)

                # Add to storage
                for chunk_idx, (chunk, tte) in enumerate(zip(chunks, time_to_end_list)):
                    chunks_by_split[split].append(chunk)

                    chunk_metadata['shot'].append(hdf5_path.stem)
                    chunk_metadata['chunk_idx'].append(chunk_idx)
                    chunk_metadata['time_to_end'].append(tte)
                    chunk_metadata['split'].append(split)
                    chunk_metadata['global_idx'].append(global_idx_counters[split])

                    global_idx_counters[split] += 1

            except Exception as e:
                print(f"Error processing {hdf5_path}: {e}")
                continue

        # Create metadata DataFrame
        meta = pd.DataFrame(chunk_metadata)

        print("\nChunk statistics:")
        print(f"\tTrain chunks: {len(chunks_by_split['train'])}")
        print(f"\tTest chunks: {len(chunks_by_split['test'])}")

        return chunks_by_split, meta

    def save_to_mmap(self, chunks_by_split: Dict[str, List], meta: pd.DataFrame):
        """
        Save chunks to memory-mapped files using mmap_ninja.

        Args:
            chunks_by_split: Dictionary with 'train' and 'test' chunk lists
            meta: DataFrame with chunk metadata
        """
        # Save train chunks
        if len(chunks_by_split['train']) > 0:
            print("\nSaving train chunks")
            # np_ninja.from_generator(
            RaggedMmap.from_generator(
                out_dir          = str(self.output_dir/'train.mmap'),
                sample_generator = chunks_by_split['train'],
                batch_size       = 64,
                verbose          = True
            )

        # Save test chunks
        if len(chunks_by_split['test']) > 0:
            print("\nSaving test chunks")
            # np_ninja.from_generator(
            RaggedMmap.from_generator(
                out_dir          = str(self.output_dir/'test.mmap'),
                sample_generator = chunks_by_split['test'],
                batch_size       = 64,
                verbose          = True
            )

        # Save metadata
        metadata_path = self.output_dir/'meta.csv'
        meta.to_csv(metadata_path, index=False)
        print(f"\nSaved metadata to {metadata_path}")

    def convert(self):
        """
        Run the full conversion pipeline.
        """
        print("Starting HDF5 to Memory Map conversion...")

        # Process all files
        chunks_by_split, meta = self.process_all_files()

        # Save to memory maps
        self.save_to_mmap(chunks_by_split, meta)

        print("\nConversion complete!")


def main():
    """
    Gather parameters and run conversion
    """
    parser = argparse.ArgumentParser(
        description='Convert HDF5 files to memory-mapped format for neural network training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--hdf5-dir',
                        type=str,
                        help='Directory containing HDF5 files')
    parser.add_argument('--meta-path',
                        type=str,
                        help=('Path to the meta file with time to disruption '
                              'and split columns'))
    parser.add_argument('--output-dir',
                        type=str,
                        help='Directory to save memory maps and metadata')
    parser.add_argument('--chunk-length',
                        type=int,
                        help='Length of each chunk (in timesteps)')
    parser.add_argument('--chunk-step',
                        type=int,
                        help='Step between chunks (in timesteps)')

    # Optional arguments
    parser.add_argument('--baseline-length',
                        type    = int,
                        default = 50000,
                        help    = 'Length of baseline section (in timesteps)')

    args = parser.parse_args()

    # Create converter
    converter = HDF5ToMemMapConverter(
        hdf5_dir        = args.hdf5_dir,
        meta_path       = args.meta_path,
        output_dir      = args.output_dir,
        chunk_length    = args.chunk_length,
        chunk_step      = args.chunk_step,
        baseline_length = args.baseline_length,
    )

    # Run conversion
    converter.convert()

if __name__ == "__main__":
    main()
