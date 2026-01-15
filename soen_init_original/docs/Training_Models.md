---
layout: default
title: Training Models - Complete Guide
---
# Training Models: Complete Guide

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Building_Models.md" style="margin-right: 2em;">&#8592; Previous: Building Models</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="GUI_Tools.md" style="margin-left: 2em;">Next: GUI Tools &#8594;</a>
</div>

This guide covers the SOEN training framework built on PyTorch Lightning. By the end, you'll understand how to configure experiments, define losses, schedule learning rates, and use callbacks for dynamic training control.

**Note:** This training framework is optional. Once you have a `.soen` model file, you can train it using standard PyTorch workflows if preferred. This framework was developed to streamline our internal training experiments and prioritize rapid iteration. It focuses on features we've needed in practice—if you require functionality not covered here, please feel free to use whichever alternative training method you prefer.

---

## Table of Contents

- [Framework Overview](#framework-overview)
- [Training Configuration Structure](#configuration-structure)
  - [Training Section](#training-section)
  - [Data Section](#data-section)
  - [Model Section](#model-section)
  - [Logging Section](#logging-section)
  - [Profiler Section](#profiler-section)
  - [Cloud Training](#cloud-training)
  - [Callbacks Section](#callbacks-section)
- [Core Components](#core-components)
- [Loss Functions Catalog](#loss-functions-catalog)
- [Callbacks Catalog](#callbacks-catalog)
- [Running Training](#running-training)
- [Troubleshooting](#troubleshooting)

---

## Framework Overview

The training framework uses PyTorch Lightning for orchestration and is configured entirely through YAML files.

### What is PyTorch Lightning?

PyTorch Lightning is a lightweight wrapper around PyTorch that handles training boilerplate (training loops, validation, logging, checkpointing) so you can focus on the model itself. Instead of writing nested loops and manual GPU transfers, you define what happens in one training step, and Lightning orchestrates everything else. This framework uses Lightning to standardize experiment structure and enable features like distributed training, gradient accumulation, and automatic mixed precision with minimal configuration.

![Framework Overview](Figures/Training_Models/training_models_fig1.jpg)
*SOEN Toolkit Training pipeline. What happens when you run: `soen_toolkit.train`. See expanded description below.*

**Flow:**
1. `training_config.yaml` defines all settings for the entire experiment—batch size, model path, dataset, learning rate, and so on
2. `ExperimentRunner` parses the config and creates:
   - `SOENDataModule` - handles data loading and splitting
   - `SOENLightningModule` - wraps `SOENModelCore` and defines training logic
   - `Callbacks` - plugins for scheduling, logging, and dynamic adjustments
3. `pl.Trainer` receives these components and runs the training loop
4. Outputs: metrics logged to TensorBoard/MLflow, checkpoints saved to disk

---

## Training Configuration Structure

![High-level Training Configuration Structure](Figures/Training_Models/training_models_fig4.png)
*Diagram illustrating the high-level structure of the training experiment configuration/settings yaml. To get a simple training run going you can omit the last few of these sections and keep just `training`, `data`, `model`, and `logging`.*

The configuration file is organized into logical sections. Let's walk through each one.

### Training Section

This section controls the optimization process and the training loop itself.

```yaml
training:
  paradigm: supervised        # 'supervised' or 'self_supervised' - see note below
  mapping: seq2static         # 'seq2static' or 'seq2seq' - see note below
  max_epochs: 100             # Number of training epochs (iterations over the full dataset). "max" because early stopping may stop training beforehand
  batch_size: 64              # Number of samples used per optimization step (parallelized on GPU)
  num_repeats: 1              # Run the experiment N times with different seeds. Each repeat creates a separate training run with its own logs and checkpoints (auto-named repeat_0, repeat_1, etc.). Each repeat uses an incremented seed for both data sampling and model initialization.
  accelerator: "gpu"          # 'auto', 'cpu', 'gpu', 'mps'
  devices: [0]
  precision: "16-mixed"       # '32-true', '16-mixed'
  backend: torch # options are torch and jax. Use torch for reliability. Jax with training is still in development and many features such as callbacks will not work.
  early_stopping_patience: 10  # Stop if val_loss doesn't improve for N epochs (omit or null to disable)
  
  optimizer:
    name: "adamw"             # 'adamw', 'adam', 'sgd', 'lion'...
    lr: 0.001
    kwargs:                   # Any keyword args specific to the optimizer
      weight_decay: 0.0001

  loss:
    # You can define multiple losses here, each weighted differently
    losses:
      - name: cross_entropy
        weight: 1.0

  train_from_checkpoint: null  # Path to .ckpt to resume
```

#### Important Notes on paradigm and mapping

These parameters control key aspects of the training pipeline:

**`mapping` (seq2static vs seq2seq):**
- Determines whether the loss uses pooled outputs or the full sequence
- Drives target/output alignment and shape handling in training/validation/test steps
- In the JAX backend, controls MSE/CE alignment and sequence handling
- Disables time pooling when seq2seq is used (no pooling needed for sequence-to-sequence)
- `seq2static`: Classification/regression on pooled output (e.g., one label per sequence)
- `seq2seq`: Full sequence output (e.g., one label per timestep)

**`paradigm` (supervised vs self_supervised vs distillation):**
- Enables the self-supervised reconstruction path (e.g., targets = inputs) when applicable
- Used by config validation to check compatible losses/data shapes and emit suggestions/warnings
- Also affects whether time pooling will be ignored (e.g., self_supervised + seq2seq)
- **`distillation`**: Trains a student model to match teacher model's state trajectories. See [Distillation Guide](training/distillation.md) for details.
- **Recommendation:** Use `supervised` for most use cases. Self-supervised training is still experimental and has had limited testing.

| Parameter | Description |
|-----------|-------------|
| `paradigm` | `supervised` for labeled data, `self_supervised` for reconstruction tasks (experimental), `distillation` for knowledge distillation ([guide](training/distillation.md)) |
| `mapping` | `seq2static` (pooled output) or `seq2seq` (full sequence output). Controls time pooling behavior and loss alignment. |
| `num_repeats` | Number of independent training runs with incremented seeds. Each repeat creates separate log/checkpoint directories. No averaging is performed. |
| `optimizer` | Optimizer type and learning rate |
| `loss.losses` | List of weighted loss components (see Loss Catalog) |
| `train_from_checkpoint` | Resume from a `.ckpt` file (restores model, optimizer, epoch) |

#### Backend selection (Torch vs JAX)

You can choose the execution backend via:

```yaml
training:
  backend: torch  # or: jax
```

- **Torch** is the default and the most feature-complete path.
- **JAX** is supported, but some features are still limited and evolve faster.

If you’re using JAX, read these two pages first:

- JAX backend overview: `jax/README.md`
- Simulation fundamentals (Torch + JAX): `Simulation.md`

#### Truncated BPTT

Truncated Backpropagation Through Time (TBPTT) can help when dealing with very long sequences. It splits sequences into smaller chunks to reduce memory usage during training. This is particularly useful when sequences don't fit in GPU memory, or when gradient computation becomes prohibitively expensive. Each chunk is processed independently, with gradients computed only within that chunk. You'll most often use this when gradients vanish or explode with long sequences.

```yaml
training:
  use_tbptt: true
  tbptt_steps: 128        # Chunk size (number of timesteps per chunk)
  tbptt_stride: 128       # Step between chunks (can be < steps for overlap)
```

---

### Data Section

This section tells the framework where to find your data and how to prepare it for training. Please refer to [DATASETS](DATASETS.md) for detailed information about the expected dataset format.

```yaml
data:
  data_path: "datasets/my_data.h5"
  cache_data: true              # Load into RAM, should always be true for small datasets, otherwise training can be very slow!
  target_seq_len: 256           # Resample sequences to this length, leave commented out if you wish to use the original dataset file
  min_scale: 0.0                # Optional scaling range
  max_scale: 1.0                # If you comment this out, no scaling will be applied to the samples
  input_encoding: "raw"         # 'raw' or 'one_hot'. This depends on how you created the dataset
```

**Tip:** Always set `cache_data: true` for datasets that fit in memory. The speedup is dramatic—without caching, the training loop spends most of its time reading from disk rather than actually training.

---

### Model Section

Here's where you specify which model to use and how to configure it.

```yaml
model:
  # Option 1: Load from saved model
  base_model_path: "models/base.soen"
  load_exact_model_state: false  # false = load architecture only and initialize weights using the training run's seed
                                  # true = load exact weights from the saved model

  # Option 2: Build from architecture/spec YAML
  architecture_yaml: "configs/arch.yaml"  # Builds a fresh model from YAML spec
                                          # Note: load_exact_model_state is not applicable here

  # Time pooling (for seq2static only - automatically disabled for seq2seq)
  # Used when we want our model to output a fixed scalar per node rather than a full time series
  # Aggregates the output layer's state trajectory over time
  time_pooling:
    name: "max"              # 'max', 'mean', 'rms', 'final', 'mean_last_n', 'ewa'
    params:
      scale: 1.0             # Multiplicative scaling factor applied to pooled outputs
                             # No rule of thumb - adjust if training is unstable or convergence is slow
                             # Different scales can affect gradient magnitudes and loss landscapes

  # Override simulation time-step
  dt: 37                     # Only has physical meaning when using physical SOEN layers
                             # Not used for virtual layers (e.g., RNN)
                             # For SOEN layers, dt corresponds to real physical time as determined by omega_c in physics/constants.py
  dt_learnable: false        # Experimental feature. Set to false by default
```

#### Important: Seed Behavior

When building models during training, the framework ensures deterministic weight initialization across repeats:

- The training run's seed (base seed + repeat number) **always** controls model initialization
- If you load a model from `base_model_path` with `load_exact_model_state: false`, the architecture is loaded but weights are initialized using the training seed, not any seed stored in the model file
- If you build from `architecture_yaml`, the architecture YAML's seed (if present) is **ignored**, and the training seed is used instead
- This ensures that `num_repeats` produces different weight initializations deterministically, even if the base model or architecture YAML contains its own seed

**Example:** If you set `seed: 1234` and `num_repeats: 3`, you get:
- Repeat 0: weights initialized with seed 1234
- Repeat 1: weights initialized with seed 1235
- Repeat 2: weights initialized with seed 1236

All repeats use the same architecture but different random initializations.

#### Time Pooling Methods

When you're doing seq2static tasks (like classifying an entire sequence), you need to somehow reduce the full time series output to a single value. That's where time pooling comes in:

| Method | Description | Parameters |
|--------|-------------|------------|
| `max` | Max over time | `scale` |
| `mean` | Mean over time | `scale` |
| `rms` | Root mean square | `scale` |
| `final` | Last timestep | `scale` |
| `mean_last_n` | Mean of last N steps | `n`, `scale` |
| `ewa` | Exponentially weighted average | `min_weight`, `scale` |

**Note on `scale`:** This parameter multiplies the pooled output. There's no universal rule for setting it—but you may need to adjust if training is unstable or converges slowly. Different scales affect gradient magnitudes and can help match the output distribution to your target distribution.

---

### Logging Section

#### How Logging Works

The framework automatically logs training metrics, hyperparameters, and artifacts throughout training. Training writes TensorBoard event files by default via a lightweight logger wrapper—you'll always get step/epoch metrics at the cadence set by `logging.log_freq`, plus optional histograms and stats when `track_connections`/`track_layer_params` are enabled. 

If `logging.mlflow_active` is enabled, the same metrics and selected artifacts (checkpoints, `.soen` sidecars) are also sent to MLflow; when MLflow is off, everything still works locally with TensorBoard. Lightning orchestrates callback timing and epoch/batch hooks, while the runner handles the actual metric emission and artifact logging to ensure consistent behavior across backends.

Each training run writes to the following directory layout:

```
project_dir/
└── project_{project_name}/
    └── experiment_{experiment_name}/
        └── group_{group_name}/
            ├── checkpoints/
            │   └── repeat_{k}/
            └── logs/
                └── repeat_{k}/
```

- **Event files** (scalars, histograms) are written under `logs/repeat_{k}/`
- **Checkpoints** are written under `checkpoints/repeat_{k}/`
- **Hyperparameters** are captured from your config file

View logs in real-time by launching TensorBoard, for example: `tensorboard --logdir project_dir/`.

#### Reading Logs Programmatically

For analysis in notebooks or scripts, use `TBReader` to load TensorBoard logs into pandas DataFrames:

```python
from soen_toolkit.utils.tb_reader import TBReader

# Point at any level of your project hierarchy
reader = TBReader("experiments/project_MyProject")

# Get scalar metrics as a DataFrame
df = reader.scalars()  # Columns: step, tag, value, wall_time, dir_name, group, repeat

# Filter by tag (regex) and group
accuracy_df = reader.scalars(tags="val_accuracy", groups="audio_8")

# Summary of available data
print(reader.summary())  # {'runs': [...], 'groups': [...], 'scalars': {'count': ..., 'tags': [...]}}
print(reader.groups())   # ['group_foo', 'group_bar', ...]
print(reader.tags())     # ['val_accuracy', 'train_loss', ...]

# Export to CSV
reader.to_csv("metrics.csv")
reader.to_csv("output_dir/", per_run=True)  # One CSV per group/repeat
```

The `group` and `repeat` columns are automatically parsed from the directory structure, making it easy to aggregate across repeats with error bars in seaborn/matplotlib.

```yaml
logging:
  project_name: "MyProject"
  experiment_name: "Run1"
  group_name: "InitialTests"
  log_level: "INFO"
  log_freq: 50                # Log every N training steps
  metrics: ["accuracy"]
  
  track_connections: true     # Log connection weight histograms
  track_layer_params: true    # Log layer parameter histograms
  
  # MLflow integration (optional)
  mlflow_active: false
  mlflow_password: "xxx"      # Not needed for local MLFlow, only if you require access to the shared team server. 
                              # Ask team for password if using the EC2 MLFlow server (https://mlflow-greatsky.duckdns.org)

  # Or use a locally hosted MLFlow service:
  mlflow_tracking_uri: "file:./mlruns"  # Prefix with 'file:' for local storage
```

#### What is MLflow?

MLflow is an open-source platform for tracking machine learning experiments. It logs metrics, parameters, and artifacts (like model checkpoints) to a central server, making it easy to compare runs, visualize training curves, and reproduce results. While TensorBoard logs to local files, MLflow provides a centralized database accessible via a web UI—useful for team collaboration. The framework supports both local MLflow instances (`file:./mlruns`) and remote tracking servers.

**Note:** MLflow is optional. If not configured, metrics are logged to TensorBoard by default.

For more detailed notes on MLFlow, see the [MLFlow Documentation](MLFLOW.md).

---

### Profiler Section

The profiler helps you understand where your training time is actually going. Is it data loading? Forward pass? Backward pass?

```yaml
profiler:
  active: false                   # Enable profiling
  type: "simple"                  # 'simple', 'advanced', 'pytorch'. 'simple' is the most reliable
  num_train_batches: 10          # Profile first N training batches
  num_val_batches: 5             # Profile first N validation batches
  record_shapes: true            # Log tensor shapes
  profile_memory: true           # Track memory usage
  with_stack: true               # Include stack traces
  output_filename: "profile_results"
```

**When to use it:** Enable profiling when you suspect performance bottlenecks. The `simple` profiler gives you a quick overview, while `pytorch` provides detailed GPU/CPU breakdowns. Just remember to disable it for actual training runs—profiling adds overhead.

---

### Cloud Training

For training on AWS SageMaker with GPU acceleration, see the dedicated [Cloud Training Guide](Cloud_Training.md).

---

### Callbacks Section

#### What are Callbacks?

Callbacks are hooks that run at specific points in the training loop (e.g., at the start of each epoch, after each batch, at validation time). They let you inject custom behavior without modifying the core training code. For example:
- **Learning rate schedulers** adjust the optimizer's learning rate over time
- **Noise annealers** gradually reduce noise injection during training
- **Early stopping** halts training when validation loss stops improving

*Note:* Early stopping is configured via `training.early_stopping_patience` (no separate callback block required).

Lightning calls these callback methods automatically at the right moments, and you configure them entirely through the YAML file. Think of them as plugins that extend training behavior declaratively.

```yaml
callbacks:
  # Learning rate scheduler
  lr_scheduler:
    type: "cosine"
    max_lr: 0.001
    min_lr: 0.000001
    warmup_epochs: 5
    cycle_epochs: 50

  # Noise annealing
  noise_annealers:
    - key: "bias_current"
      target: "perturb"          # 'noise' or 'perturb'
      start_value: 0.1
      end_value: 0.0
      start_epoch: 0
      end_epoch: 100

  # Loss weight scheduling
  loss_weight_schedulers:
    - loss_name: "reg_J_loss"
      scheduler_type: "linear"
      params:
        min_weight: 0.01
        max_weight: 0.1

  # QAT (Quantization-Aware Training)
  qat:
    active: false
    bits: 4
    min_val: -0.24
    max_val: 0.24
```

We'll cover the full catalog of available callbacks later in this guide.

---

## Core Components

Now that you understand the configuration structure, let's look at how these pieces actually work together during training.

### SOENLightningModule

This is the Lightning wrapper around `SOENModelCore` that defines the training loop.

![SOENLightningModule](Figures/Training_Models/training_models_fig2.jpg)

**Key Methods:**
- `forward(x)` - Runs the model, applies time pooling if needed
- `training_step(batch, batch_idx)` - Defines one training iteration
- `validation_step(batch, batch_idx)` - Validation iteration
- `configure_optimizers()` - Sets up the optimizer

**Time Pooling Flow:**
1. Model outputs full state trajectory `[batch, time+1, features]`
2. If `mapping="seq2static"`, pool to `[batch, features]` using configured method
3. Apply optional scaling
4. Use as input to loss functions

### Loss System

The loss system is straightforward: total loss is a weighted sum of components. This lets you combine multiple objectives during training.

$$L_{total} = \sum_{i} w_i \cdot L_i(\text{outputs}, \text{targets})$$

Each component in `training.loss.losses` has:
- `name` - registered loss function
- `weight` - scaling factor
- `params` - loss-specific parameters

**Example:**
```yaml
loss:
  losses:
    - name: cross_entropy
      weight: 1.0
    - name: reg_J_loss
      weight: 0.01
      params:
        threshold: 0.24
```

This creates: $L_{total} = 1.0 \cdot L_{CE} + 0.01 \cdot L_{reg}$

The cross entropy pushes the model toward correct predictions, while the regularization loss keeps connection weights from getting too large.

### Callbacks

Callbacks hook into the training loop at specific points:

![How do Callbacks work?](Figures/Training_Models/training_models_fig3.jpg)

Callbacks are created by `ExperimentRunner._create_callbacks()` and passed to `pl.Trainer`. They can modify training behavior on-the-fly without touching the core training code.

---

## Loss Functions Catalog

Here's the complete catalog of available loss functions. Choose combinations that match your training goals.

| Loss | Description | Key Parameters | Example Use Case |
|------|-------------|----------------|----------|
| `cross_entropy` | Standard cross-entropy for classification | - | Classification |
| `autoregressive_cross_entropy` | Cross-entropy for next-token prediction | - | Language modeling |
| `mse` | Mean squared error | - | Regression |
| `reg_J_loss` | Penalizes large connection weights exponentially, almost a hard threshold | `threshold`, `scale`, `factor` | Weight constraints for soen-type layers |
| `top_gap_loss` | Encourages margin between correct and incorrect classes | `separation_threshold`, `factor` | Robustness upon fabrication|
| `gravity_quantization_loss` | Pulls weights toward discrete codebook | `bits`/`levels`, `min_val`, `max_val`, `factor` | QAT |
| `get_off_the_ground_loss` | Penalizes small output magnitudes | `threshold`, `min_penalty_offset`, `max_penalty_offset` | Escaping dead neurons |
| `exp_high_state_penalty` | Penalizes large output magnitudes | `threshold`, `scale`, `factor` | Certain linear in s source functions might only be a good fit if s stays < 1. This loss forces the states to stay within the desired range |
| `branching_loss` | Encourages operation near edge of chaos | `target_sigma`, `factor` | Experimental study into criticality |
| `local_expansion_loss` | Controls expansion ratio (criticality) via dual forward passes | `target_value`, `perturbation_scale`, `use_log_loss` | **PyTorch only** - Trains network toward edge of chaos (σ ≈ 1.0) |
| `average_state_magnitude_loss` | Controls average magnitude of states across all layers | `target_magnitude`, `loss_type`, `layer_ids` | **PyTorch only** - Network "aliveness" control, prevents dead or saturated states |
| `ensure_post_sample_decay` | Penalizes non-zero g at final timestep | - | Resetting states between samples on chip |

---

## Callbacks Catalog

### Learning Rate Schedulers

Configured under `callbacks.lr_scheduler`. These control how the learning rate changes during training.

| Type | Description | Key Parameters |
|------|-------------|----------------|
| `constant` | Fixed LR | `lr` |
| `linear` | Linear decay | `max_lr`, `min_lr`, `log_space` |
| `cosine` | Cosine annealing with warmup/restarts | `max_lr`, `min_lr`, `warmup_epochs`, `cycle_epochs`, `enable_restarts` |
| `rex` | Rational exponential decay | `warmup_epochs`, `min_lr`, `max_lr` |
| `greedy` | Adaptive based on validation loss | `factor_increase`, `factor_decrease`, `patience` |
| `adaptive` | Metric-driven adjustments | `monitor_metric`, `increase_factor`, `decrease_factor`, `patience_increase`, `patience_decrease` |

**Tip:** Cosine annealing with warmup is a solid default choice. It's battle-tested and works well for most cases.

### Dynamic Callbacks

These callbacks modify training behavior dynamically as training progresses.

| Callback | Purpose | Configuration |
|----------|---------|---------------|
| `NoiseAnnealingCallback` | Anneal noise/perturbation magnitude | `key`, `target`, `start_value`, `end_value` |
| `LossWeightScheduler` | Schedule loss component weights | `loss_name`, `scheduler_type`, `params` |
| `TargetSeqLenScheduler` | Adjust sequence length over time | `start_len`, `end_len`, `end_epoch` |
| `TimePoolingScaleScheduler` | Adjust time pooling scale | `start_scale`, `end_scale`, `end_epoch` |
| `ConnectionNoiseCallback` | Add fixed noise to connection weights per forward | `connections`, `std`, `relative` |
| `StatefulTrainingCallback` | Carry forward layer states across batches | `enable_for_training`, `enable_for_validation`, `sample_selection` |

### Monitoring Callbacks

These callbacks help you track and visualize what's happening during training.

| Callback | Purpose | Configuration |
|----------|---------|---------------|
| `MetricsTracker` | Log metrics, gradients, LR (enabled by default) | `logging.metrics`, `logging.track_gradients` |
| `ConnectionParameterProbeCallback` | Log weight histograms | `logging.track_connections: true` |
| `StateTrajectoryLoggerCallback` | Plot state trajectories | `logging.state_trajectories.active: true` |
| `QuantizedAccuracyCallback` | Evaluate with quantized weights | `callbacks.metrics.quantized_accuracy` |
| `QATStraightThroughCallback` | Enable QAT with STE | `callbacks.qat.active: true` |

### Stateful Training

The `StatefulTrainingCallback` enables temporal continuity across training batches by carrying forward the final layer states from one batch to initialize the next batch. This can be useful for continual learning scenarios or when temporal dependencies exist across samples.

**Key Features:**
- States reset at epoch boundaries (continuity only within epochs)
- Supports both training and validation modes independently
- Extracts states from a selected sample (random, first, or last)
- Automatically handles MultiplierNOCC s1/s2 auxiliary states

**Configuration:**
```yaml
callbacks:
  stateful_training:
    enable_for_training: true      # Enable for training batches
    enable_for_validation: false   # Enable for validation batches
    sample_selection: "random"     # Which sample to extract: "random", "first", or "last"
    verbose: false                 # Log state carryover operations
```

**How It Works:**

1. **State Extraction:** After each batch's forward pass, the callback extracts the final timestep states from all layers for one sample in the batch
2. **State Storage:** These states are stored internally in the callback
3. **State Injection:** Before the next batch's forward pass, the stored states are set as initial states for all layers
4. **Epoch Reset:** At the start of each epoch, all stored states are cleared

**Use Cases:**
- Continual learning where the model should maintain state across samples
- Sequential data processing where batches represent consecutive segments
- Studying the effect of temporal persistence on learning dynamics

**Example:**

With stateful training enabled, instead of starting every batch from zero initial states, the network maintains a "memory" from the previous batch. This can help the model learn temporal dependencies that span across batch boundaries.

```yaml
# Training config with stateful training
callbacks:
  stateful_training:
    enable_for_training: true
    enable_for_validation: true
    sample_selection: "random"
    verbose: true
```

**Note:** This feature is currently supported for PyTorch Lightning training. JAX backend support is planned for a future release.

---

## Running Training

Ready to start training? Here's how to launch it. (ensure the venv is active)

**Command line:**
```bash
train path/to/training_config.yaml
```

**Resume from checkpoint:**
If training gets interrupted, just point to the checkpoint file:
```yaml
training:
  train_from_checkpoint: "path/to/checkpoint.ckpt"
```

The framework will restore the model, optimizer state, and current epoch, then continue where it left off.

---

## JAX Training & Checkpoint Management

### JAX Backend Checkpoints

When training with the JAX backend, the system saves two checkpoint formats:

1. **`.pkl` (JAX-specific)**: Complete training state including architecture, parameters, optimizer state, and metadata
2. **`.soen` (PyTorch)**: Universal model format for cross-backend compatibility


### Verifying Checkpoint Conversion

After training, verify that `.pkl` and `.soen` checkpoints match:

```bash
# Verify a single checkpoint
python scripts/verify_jax_checkpoint.py \
    --pkl checkpoints/repeat_0/last.pkl \
    --soen checkpoints/repeat_0/last.soen \
    --verbose

# Verify all checkpoints in directory
python scripts/verify_jax_checkpoint.py \
    --dir checkpoints/repeat_0 \
    --verbose
```

**Verification Output:**
```
Status: PASS
Summary: ✓ All parameters match within tolerance (rtol=1e-05, atol=1e-06)
```

The verification compares:
- Layer parameters (phi_offset, bias_current, gamma_plus, etc.)
- Connection weights and masks
- Internal connectivity (internal_J)

**Custom Tolerances:**
```bash
python scripts/verify_jax_checkpoint.py \
    --pkl last.pkl \
    --soen last.soen \
    --rtol 1e-4 \
    --atol 1e-5
```

### Migrating Old Checkpoints

Checkpoints from older versions (pre-v1.0.0) only stored parameters. Migrate them to the enhanced format:

```bash
# Migrate single checkpoint
python scripts/migrate_jax_checkpoint.py \
    --pkl old_checkpoint.pkl \
    --soen model.soen

# Migrate all checkpoints in directory
python scripts/migrate_jax_checkpoint.py \
    --dir checkpoints/repeat_0
```

After migration, verify the conversion:
```bash
python scripts/verify_jax_checkpoint.py --pkl last.pkl --soen last.soen
```

**Note:** Old checkpoints may have minor parameter mismatches due to conversion limitations. For critical applications, retrain with the enhanced checkpoint system.

### Resuming JAX Training

To resume JAX training from a checkpoint:

```python
from soen_toolkit.utils.port_to_jax.jax_training.callbacks.checkpointing import resume_training

# Load checkpoint and resume
jax_model, params, opt_state, start_epoch, global_step = resume_training("checkpoints/last.pkl")

# Continue training from start_epoch...
```

Or specify in your training config:
```yaml
training:
  train_from_checkpoint: "checkpoints/repeat_0/last.pkl"
  # Will automatically detect and load JAX checkpoint
```

---

## Troubleshooting

Training not going as planned? Here are the most common issues and how to fix them.

### NaN Losses

Your loss suddenly becomes NaN and training crashes.

**Causes:**
- Learning rate too high
- Unstable model parameters (`gamma` values)
- Numerical instability in solver

**Solutions:**
1. Lower `optimizer.lr` (try cutting it in half)
2. Enable gradient clipping: `training.gradient_clip_val: 1.0`
3. Add `reg_J_loss` to constrain weights
4. Check base model parameters (especially `gamma_plus`, `gamma_minus`)
5. Reduce `model.dt` if dynamics are stiff
6. If you're using a large recurrently connected layer sometimes making the connections sparse helps with stability

### Shape Mismatch Errors

**Error:** `Expected target size [B, C] got [B, T]`

This means your outputs and targets have incompatible shapes.

**Causes:**
- Wrong `mapping` setting
- Incorrect data format

**Solutions:**
1. For classification with per-sequence labels: `mapping: seq2static`
2. For classification with per-timestep labels: `mapping: seq2seq`
3. Check `data.num_classes` matches your labels
4. Inspect HDF5 file: `h5ls -r datasets/data.h5`

### Slow Training

Training taking forever? There are several common bottlenecks.

**Causes:**
- Data loading bottleneck
- CPU-bound operations
- Inefficient model configuration

**Solutions:**
1. Set `data.cache_data: true` if dataset fits in RAM (huge speedup!)
2. Use `accelerator: "gpu"` if available
3. Enable mixed precision: `precision: "16-mixed"`
4. Ensure `network_evaluation_method: "layerwise"` for feedforward networks
5. Disable unnecessary tracking: `track_phi: false`, `track_s: false`

### Callback Configuration Errors

**Error:** `KeyError: 'lr_scheduler'`

**Solution:** LR scheduler is optional but must be under `callbacks.lr_scheduler` if specified.

**Error:** Loss weight not updating

**Solution:** Ensure `loss_weight_schedulers` targets a loss in `training.loss.losses` by exact name. Typos will silently fail.

### Cloud Launch Failures

**Error:** `SageMaker role not found`

**Solutions:**
1. Verify IAM role ARN in `cloud.role`
2. Ensure role has `SageMakerFullAccess` policy
3. Check S3 bucket permissions

### Memory Errors

Your GPU runs out of memory mid-training.

**Causes:**
- Batch size too large
- Long sequences
- History tracking enabled

**Solutions:**
1. Reduce `training.batch_size`
2. Enable TBPTT: `use_tbptt: true`, `tbptt_steps: 128`
3. Disable tracking: `sim_config.track_s: false`, `track_phi: false`
4. Reduce `data.target_seq_len`
5. Accumulate gradients across batches

---

## Next Steps

1. **GUI Tools** → [GUI_Tools](GUI_Tools.md) - Explore visual interfaces for model building and analysis
2. **Advanced Features** → [Advanced_Features](Advanced_Features.md)

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Building_Models.md" style="margin-right: 2em;">&#8592; Previous: Building Models</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="GUI_Tools.md" style="margin-left: 2em;">Next: GUI Tools &#8594;</a>
</div>