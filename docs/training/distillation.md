[Home](../index.md) | [Training Models](../Training_Models.md)

---

# Knowledge Distillation Training

This guide explains how to use the knowledge distillation paradigm to train a student model to match the output dynamics of a teacher model.

## Overview

**Knowledge distillation** trains a student model to replicate the temporal state trajectories of a teacher model. Unlike traditional distillation that matches final outputs, SOEN distillation matches the full time-series of internal states, enabling the student to learn the teacher's dynamical behavior.

### Key Concepts

- **Teacher Model**: A trained SOEN model with desired dynamics
- **Student Model**: A model (typically smaller or simpler) being trained to match the teacher
- **Distillation Dataset**: HDF5 file containing teacher's state trajectories as regression targets
- **Trajectory Matching**: Student learns via MSE loss over the full time series `[B, T+1, D]`

## Quick Start

### Generate and Train in One Run

```yaml
# distillation_config.yaml
model:
  backend: jax  # or torch
  model_path: /path/to/student_model.soen

data:
  data_path: /path/to/source_data.hdf5
  target_seq_len: 100
  min_scale: -1.0
  max_scale: 1.0

training:
  paradigm: distillation
  max_epochs: 1000
  batch_size: 32

  distillation:
    teacher_model_path: /path/to/teacher.soen
    subset_fraction: 1.0  # Use 100% of data
    max_samples: null     # No limit
    batch_size: 32        # Batch size for teacher inference

    # OPTIONAL: Reuse existing distillation data to skip generation
    # distillation_data_path: /path/to/existing/distillation_data.hdf5
```

Run training:
```bash
train distillation_config.yaml
```

**What happens:**
1. Teacher model runs on source dataset to generate state trajectories (if not reusing)
2. Trajectories saved to `{log_dir}/distillation_data.hdf5`
3. Student model trains on these trajectories using MSE loss

### Reusing Existing Distillation Data

If you've already generated teacher trajectories, you can reuse them by uncommenting the `distillation_data_path` line in the config above. This skips trajectory generation and jumps straight to training, saving significant time.

## Configuration Reference

### DistillationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `teacher_model_path` | `str \| Path \| None` | `None` | Path to teacher model (`.soen` or `.pth`). Required if generating new trajectories. |
| `distillation_data_path` | `str \| Path \| None` | `None` | Path to existing distillation dataset. If provided and exists, skips generation. |
| `subset_fraction` | `float` | `1.0` | Fraction of dataset to use (0.0-1.0). Samples randomly from dataset. |
| `max_samples` | `int \| None` | `None` | Hard cap on samples. Applied after `subset_fraction`. `None` = no limit. |
| `batch_size` | `int` | `32` | Batch size for teacher inference during trajectory generation. |

### How subset_fraction and max_samples Work Together

These two parameters work in sequence to control dataset size:

1. **`subset_fraction`**: First, takes a random fraction of the full dataset
   - `1.0` = use 100% of data (all samples)
   - `0.5` = randomly sample 50% of data
   - `0.1` = randomly sample 10% of data

2. **`max_samples`**: Then, caps the result at a maximum number
   - Applied AFTER `subset_fraction`
   - Useful when you want "at most N samples" regardless of dataset size

**Examples:**
- Dataset has 10,000 samples
- `subset_fraction: 0.5, max_samples: null` → uses 5,000 random samples
- `subset_fraction: 1.0, max_samples: 1000` → uses 1,000 random samples
- `subset_fraction: 0.5, max_samples: 1000` → uses 1,000 random samples (50% = 5000, then capped at 1000)

**Note:** Sampling is random using `np.random.choice(..., replace=False)`, so you get a random subset, not just the first N samples.

### Validation Rules

- **Either** `teacher_model_path` **or** `distillation_data_path` must be provided
- If `distillation_data_path` is provided and exists → uses that dataset (fast path)
- If `distillation_data_path` doesn't exist and no `teacher_model_path` → error
- `subset_fraction` must be in range (0.0, 1.0]
- `max_samples` must be positive if specified

## Distillation Dataset Format

The generated HDF5 file has this structure:

```
distillation_data.hdf5
├── train/
│   ├── data: [N, T, input_dim]      # Inputs (potentially resampled)
│   └── labels: [N, T+1, output_dim] # Teacher states (including t=0)
├── val/
│   ├── data: [N, T, input_dim]
│   └── labels: [N, T+1, output_dim]
└── test/ (if present in source)
    ├── data: [N, T, input_dim]
    └── labels: [N, T+1, output_dim]
```

**Key points:**
- Labels have shape `[N, T+1, output_dim]` (one extra timestep for initial state at t=0)
- Inputs are preprocessed with same `target_seq_len`, `min_scale`, `max_scale` as training
- Labels are **float32** (not int64 like classification)

## Backend Support

Distillation works with both PyTorch and JAX backends:

### PyTorch Backend
```yaml
model:
  backend: torch
training:
  paradigm: distillation
  distillation:
    teacher_model_path: /path/to/teacher.soen
```
- Uses `SOENLightningModule` trainer
- Teacher inference on GPU if available
- MSE loss computed via PyTorch

### JAX Backend
```yaml
model:
  backend: jax
training:
  paradigm: distillation
  distillation:
    teacher_model_path: /path/to/teacher.soen
```
- Uses JAX-native trainer
- Teacher model converted to JAX for inference
- MSE loss computed via JAX
- Supports same teacher generation and training

## Common Workflows

### Workflow 1: One-Shot Distillation
Generate trajectories and train in one run:
```bash
train distillation_config.yaml
```

### Workflow 2: Generate Once, Train Many Times
```bash
# First run: Generate trajectories
train gen_trajectories.yaml

# Subsequent runs: Reuse trajectories with different student configs
train student_v1.yaml
train student_v2.yaml
```

Where `student_v1.yaml` and `student_v2.yaml` both specify:
```yaml
training:
  distillation:
    distillation_data_path: /path/to/distillation_data.hdf5
```

### Workflow 3: Subset for Fast Iteration
Use a subset during development, full data for final training:
```yaml
# Development config
training:
  distillation:
    teacher_model_path: /path/to/teacher.soen
    subset_fraction: 0.1   # Randomly sample 10% for fast iteration
    max_samples: 1000      # Additionally cap at 1000 samples

# Production config
training:
  distillation:
    teacher_model_path: /path/to/teacher.soen
    subset_fraction: 1.0   # Full dataset
    max_samples: null      # No cap
```

## Architecture Requirements

### Teacher and Student Must Match in Key Dimensions

The student model **must** have:
- Same input dimension as teacher
- Same output dimension as teacher
- Same sequence length handling

The student **can differ** in:
- Number of layers
- Layer types (e.g., teacher uses `SingleDendrite`, student uses `BasicLayer`)
- Hidden layer dimensions
- Connection patterns

### Example: Distilling Complex → Simple

```python
# Teacher: 3-layer network with dendrites
teacher_layers = [
    {"type": "SingleDendrite", "dim": 8},
    {"type": "SingleDendrite", "dim": 25},
    {"type": "SingleDendrite", "dim": 2}
]

# Student: 2-layer simpler network
student_layers = [
    {"type": "BasicLayer", "dim": 8},
    {"type": "BasicLayer", "dim": 2}
]
```

Both have input_dim=8, output_dim=2, so distillation works!

## Troubleshooting

### Problem: Low loss but wrong dynamics

**Cause:** Labels were converted to int64, destroying trajectory data.

**Fix:** Ensure you're using the latest version where paradigm detection correctly sets `labels_as_int=False`.

**Verify in logs:**
```
Paradigm 'distillation' detected: loading labels as float32
```

### Problem: TypeError about `distillation_data_path`

**Cause:** Old config class doesn't have the new field.

**Fix:** Pull latest code with updated `DistillationConfig` class.

### Problem: Teacher trajectory generation is slow

**Solutions:**
1. Use smaller `subset_fraction` during development
2. Generate once, then reuse with `distillation_data_path`
3. Use JAX backend for teacher inference

### Problem: Student doesn't learn teacher dynamics

**Debugging steps:**
1. Verify student architecture can express teacher's computation
2. Check learning rate isn't too high/low
3. Ensure preprocessing matches (same `target_seq_len`, `scale_min`, `scale_max`)
4. Check that labels are float32 in first-batch logs

## API Reference

### Python API

```python
from pathlib import Path
from soen_toolkit.training.distillation import generate_teacher_trajectories

# Generate trajectories programmatically
generate_teacher_trajectories(
    teacher_model_path=Path("teacher.soen"),
    source_data_path=Path("source.hdf5"),
    output_path=Path("distillation.hdf5"),
    subset_fraction=1.0,
    max_samples=None,
    batch_size=32,
    target_seq_len=100,
    scale_min=-1.0,
    scale_max=1.0,
    backend="jax"  # or "torch"
)
```

### Config Classes

```python
from soen_toolkit.training.configs import DistillationConfig

config = DistillationConfig(
    teacher_model_path="/path/to/teacher.soen",
    subset_fraction=1.0,
    max_samples=None,
    batch_size=32
)

# Or reuse existing data
config_reuse = DistillationConfig(
    distillation_data_path="/path/to/distillation_data.hdf5"
)
```

---

[Home](../index.md) | [Training Models](../Training_Models.md)
