# FILEPATH: src/soen_toolkit/training/distillation/teacher_data_generator.py

"""Generate teacher state trajectories for knowledge distillation.

This module provides functionality to run a teacher model on a dataset and
capture its output state trajectories, which can then be used as regression
targets for training a student model.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from soen_toolkit.core import SOENModelCore
from soen_toolkit.training.data.dataloaders import GenericHDF5Dataset

logger = logging.getLogger(__name__)


def generate_teacher_trajectories(
    teacher_model_path: Path,
    source_data_path: Path,
    output_path: Path,
    subset_fraction: float = 1.0,
    max_samples: int | None = None,
    batch_size: int = 32,
    device: str | None = None,
    num_workers: int = 0,
    target_seq_len: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> None:
    """Run teacher model on dataset and save output state trajectories.

    This function loads a teacher model, runs it on the source dataset,
    and saves the output layer state trajectories as regression targets.

    Args:
        teacher_model_path: Path to the teacher model (.pth or .soen file).
        source_data_path: Path to the source HDF5 dataset.
        output_path: Path where the distillation HDF5 file will be saved.
        subset_fraction: Fraction of dataset to use (0.0-1.0). Default 1.0 uses all data.
        max_samples: Maximum samples to use (applied after subset_fraction). None = no limit.
        batch_size: Batch size for teacher inference.
        device: Device for inference ('cpu', 'cuda', 'mps'). Auto-detected if None.
        num_workers: Number of DataLoader workers.
        target_seq_len: Resample inputs to this sequence length (same as training).
        scale_min: Minimum value for input scaling (same as training).
        scale_max: Maximum value for input scaling (same as training).

    The output HDF5 file has structure:
        - train/data: [N, T, input_dim]  (inputs, potentially resampled)
        - train/labels: [N, T+1, output_dim]  (teacher output states including t=0)
        - val/data, val/labels (same structure)
        - test/data, test/labels (same structure)
    """
    # Validate paths
    if not teacher_model_path.exists():
        msg = f"Teacher model not found: {teacher_model_path}"
        raise FileNotFoundError(msg)

    if not source_data_path.exists():
        msg = f"Source dataset not found: {source_data_path}"
        raise FileNotFoundError(msg)

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # Load teacher model
    logger.info(f"Loading teacher model from: {teacher_model_path}")
    teacher_model = SOENModelCore.load(str(teacher_model_path))
    teacher_model.eval()
    teacher_model.to(device)
    logger.info(f"Teacher model loaded: {len(teacher_model.layers)} layers")

    # Check which splits exist in the source dataset
    with h5py.File(source_data_path, "r") as f:
        available_splits = [s for s in ["train", "val", "test"] if s in f]
        if not available_splits:
            # No split groups - data is at root level
            available_splits = [None]
            logger.info("Source dataset has no split groups, using root level data")
        else:
            logger.info(f"Found splits in source dataset: {available_splits}")

    # Create output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating distillation dataset at: {output_path}")

    with h5py.File(output_path, "w") as out_f:
        for split in available_splits:
            split_name = split if split is not None else "train"
            logger.info(f"Processing split: {split_name}")

            # Load source dataset for this split with same preprocessing as training
            dataset = GenericHDF5Dataset(
                hdf5_path=str(source_data_path),
                split=split,
                cache_in_memory=True,  # Faster for iteration
                target_seq_len=target_seq_len,
                scale_min=scale_min,
                scale_max=scale_max,
            )

            # Apply subset fraction and max_samples cap
            total_samples = len(dataset)
            num_samples = int(total_samples * subset_fraction)
            if max_samples is not None:
                num_samples = min(num_samples, max_samples)
            if num_samples < total_samples:
                logger.info(f"Using {num_samples}/{total_samples} samples")
                indices = np.random.choice(total_samples, size=num_samples, replace=False)
                indices = np.sort(indices)  # Keep order for reproducibility
                dataset = torch.utils.data.Subset(dataset, indices.tolist())

            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device == "cuda"),
            )

            # Collect inputs and teacher outputs
            all_inputs = []
            all_teacher_states = []

            with torch.no_grad():
                for batch_idx, (inputs, _labels) in enumerate(dataloader):
                    inputs = inputs.to(device)

                    # Run teacher forward pass
                    # Returns (final_state, all_states) where final_state is [B, T+1, dim]
                    final_state, _all_states = teacher_model(inputs)

                    # Store results
                    all_inputs.append(inputs.cpu().numpy())
                    all_teacher_states.append(final_state.cpu().numpy())

                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"  Processed {(batch_idx + 1) * batch_size} samples...")

            # Concatenate all batches
            inputs_array = np.concatenate(all_inputs, axis=0)
            teacher_states_array = np.concatenate(all_teacher_states, axis=0)

            logger.info(
                f"  {split_name}: inputs shape {inputs_array.shape}, "
                f"teacher states shape {teacher_states_array.shape}"
            )

            # Create split group and save
            grp = out_f.create_group(split_name)
            grp.create_dataset("data", data=inputs_array, compression="gzip")
            grp.create_dataset("labels", data=teacher_states_array, compression="gzip")

    logger.info(f"Distillation dataset saved to: {output_path}")


def validate_distillation_dataset(dataset_path: Path) -> dict:
    """Validate a distillation dataset and return metadata.

    Args:
        dataset_path: Path to the distillation HDF5 file.

    Returns:
        Dictionary with dataset metadata (shapes, splits, etc.)
    """
    if not dataset_path.exists():
        msg = f"Dataset not found: {dataset_path}"
        raise FileNotFoundError(msg)

    metadata = {"path": str(dataset_path), "splits": {}}

    with h5py.File(dataset_path, "r") as f:
        for split in ["train", "val", "test"]:
            if split in f:
                grp = f[split]
                metadata["splits"][split] = {
                    "num_samples": grp["data"].shape[0],
                    "input_shape": grp["data"].shape[1:],
                    "target_shape": grp["labels"].shape[1:],
                }

    return metadata
