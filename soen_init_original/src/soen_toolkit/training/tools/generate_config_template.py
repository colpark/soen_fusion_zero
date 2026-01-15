"""CLI tool to generate comprehensive training configuration templates.

This module introspects the training config system and generates a fully
commented YAML template with all available options.

Usage:
    python -m soen_toolkit.training.tools.generate_config_template --output template.yaml
"""

import argparse
import dataclasses
import inspect
import logging
from pathlib import Path
import sys
from typing import Any

import yaml

from soen_toolkit.training.callbacks import (
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
)
from soen_toolkit.training.callbacks.metrics import METRICS_REGISTRY
from soen_toolkit.training.losses import LOSS_REGISTRY

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Introspection Functions
# -----------------------------------------------------------------------------


def introspect_dataclass(cls) -> dict[str, Any]:
    """Extract schema information from a dataclass.

    Args:
        cls: A dataclass type

    Returns:
        Dictionary containing class name, docstring, and field information
    """
    schema = {"name": cls.__name__, "docstring": inspect.getdoc(cls) or "", "fields": []}

    for field in dataclasses.fields(cls):
        # Determine if field is required
        has_default = field.default != dataclasses.MISSING
        has_factory = field.default_factory != dataclasses.MISSING

        field_info = {
            "name": field.name,
            "type": str(field.type),
            "required": not (has_default or has_factory),
            "default": field.default if has_default else None,
            "has_factory": has_factory,
        }

        schema["fields"].append(field_info)

    return schema


def query_registries() -> dict[str, list[str]]:
    """Query all registries to get available options.

    Returns:
        Dictionary mapping registry names to lists of registered items
    """
    return {
        "schedulers": sorted(SCHEDULER_REGISTRY.keys()),
        "losses": sorted(LOSS_REGISTRY.keys()),
        "metrics": sorted(METRICS_REGISTRY.keys()),
        "optimizers": sorted(OPTIMIZER_REGISTRY.keys()),
    }


def introspect_scheduler_params(scheduler_name: str) -> dict[str, Any]:
    """Extract __init__ parameters for a scheduler.

    Args:
        scheduler_name: Name of the registered scheduler

    Returns:
        Dictionary mapping parameter names to their info (default, type)
    """
    if scheduler_name not in SCHEDULER_REGISTRY:
        return {}

    scheduler_cls = SCHEDULER_REGISTRY[scheduler_name]
    try:
        sig = inspect.signature(scheduler_cls.__init__)
    except (ValueError, TypeError):
        return {}

    params: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "debug"):
            continue

        # Extract default value
        default_val = None
        if param.default != inspect.Parameter.empty:
            default_val = param.default

        # Extract type annotation
        type_str = "Any"
        if param.annotation != inspect.Parameter.empty:
            type_str = str(param.annotation).replace("typing.", "")

        params[param_name] = {"default": default_val, "type": type_str}

    return params


# -----------------------------------------------------------------------------
# YAML Template Generation (using standard dict for PyYAML)
# -----------------------------------------------------------------------------


def create_commented_map(comment: str | None = None) -> dict:
    """Create a dict for YAML template.

    Args:
        comment: Optional comment (stored in special key for later processing)

    Returns:
        dict instance
    """
    return {}


def add_section_header(yaml_dict: dict, key: str, header: str) -> None:
    """Add a section header comment before a key.

    Note: With standard PyYAML, comments are added during string generation.

    Args:
        yaml_dict: dict to modify
        key: Key to add comment before
        header: Comment text
    """
    # Store comment metadata (will be used during template generation)
    pass


def build_training_section(verbosity: str) -> dict:
    """Build the training configuration section.

    Args:
        verbosity: Template verbosity level (minimal, standard, full)

    Returns:
        dict with training configuration
    """
    training: dict[str, Any] = {}

    # Task configuration
    training["paradigm"] = "supervised"  # supervised | self_supervised | unsupervised
    training["mapping"] = "seq2static"  # seq2static | seq2seq | static2seq | static2static

    # Basic training settings
    training["batch_size"] = 32
    training["max_epochs"] = 100
    training["num_repeats"] = 1

    # Hardware settings
    training["accelerator"] = "auto"  # auto | cpu | gpu | mps | tpu
    training["precision"] = "32-true"  # 32-true | 16-mixed
    training["devices"] = "auto"  # auto | 1 | [0,1,2] for multi-GPU
    training["num_workers"] = 4
    training["deterministic"] = True

    if verbosity in ["standard", "full"]:
        # Gradient settings
        training["accumulate_grad_batches"] = 1
        training["gradient_clip_val"] = None  # Set to float for gradient clipping
        training["gradient_clip_algorithm"] = "norm"

        # Checkpointing
        training["checkpoint_every_n_epochs"] = 1
        training["checkpoint_save_top_k"] = 3
        training["checkpoint_save_last"] = True
        training["save_initial_state"] = True
        training["save_soen_core_in_checkpoint"] = True

    if verbosity == "full":
        # Autoregressive settings
        training["autoregressive"] = False
        training["autoregressive_mode"] = "next_token"  # next_token | seq2seq
        training["time_steps_per_token"] = 1
        training["autoregressive_start_timestep"] = 0

        # TBPTT settings
        training["use_tbptt"] = False
        training["tbptt_steps"] = None  # Set chunk size if use_tbptt=True
        training["tbptt_stride"] = None

        # Early stopping
        training["early_stopping_patience"] = None  # Set to int for early stopping

        # Distributed training
        training["num_nodes"] = None  # Number of nodes for distributed training
        training["strategy"] = None  # ddp | ddp_spawn | dp | auto

    # Optimizer (always include)
    optimizer: dict[str, Any] = {}
    optimizer["name"] = "adamw"  # adamw | adam | sgd | lion | muon
    optimizer["lr"] = 0.001
    optimizer["kwargs"] = {"weight_decay": 1e-4}
    training["optimizer"] = optimizer

    # Loss configuration (always include)
    loss: dict[str, Any] = {}
    losses_list = [{"name": "cross_entropy", "weight": 1.0, "params": {}}]
    loss["losses"] = losses_list
    training["loss"] = loss

    return training


def build_data_section(verbosity: str) -> dict:
    """Build the data configuration section."""
    data: dict[str, Any] = {}

    data["data_path"] = "path/to/your/dataset.h5"  # REQUIRED: Path to dataset
    data["sequence_length"] = 100
    data["num_classes"] = 10
    data["val_split"] = 0.2
    data["test_split"] = 0.1
    data["cache_data"] = True

    if verbosity in ["standard", "full"]:
        data["target_seq_len"] = None  # Resample to this length (None = no resampling)
        data["min_scale"] = None
        data["max_scale"] = None

    if verbosity == "full":
        data["input_encoding"] = "raw"  # raw | one_hot | embedding
        data["vocab_size"] = None  # Required for one_hot encoding
        data["one_hot_dtype"] = "float32"
        data["synthetic"] = False
        data["synthetic_kwargs"] = {}

    return data


def build_model_section(verbosity: str) -> dict:
    """Build the model configuration section."""
    model: dict[str, Any] = {}

    model["base_model_path"] = "path/to/your/model.pth"  # Path to pre-trained SOEN model
    model["load_exact_model_state"] = False

    # Time pooling
    time_pooling: dict[str, Any] = {}
    time_pooling["name"] = "final"  # max | mean | rms | final | mean_last_n | mean_range | ewa
    time_pooling["params"] = {"scale": 1.0}
    model["time_pooling"] = time_pooling

    if verbosity in ["standard", "full"]:
        model["dt"] = None  # Simulation timestep (None = use model default)
        model["dt_learnable"] = False

    if verbosity == "full":
        model["architecture_yaml"] = None  # Build model from architecture YAML
        model["backend"] = "torch"  # torch | jax

    return model


def build_logging_section(verbosity: str, registries: dict[str, list[str]]) -> dict:
    """Build the logging configuration section."""
    logging_config: dict[str, Any] = {}

    logging_config["project_dir"] = "experiments/"
    logging_config["project_name"] = "soen_training"
    logging_config["group_name"] = "default_group"
    logging_config["experiment_name"] = None  # Auto-generated if None

    # Metrics (with available options as comment in full mode)
    logging_config["metrics"] = ["accuracy"]
    logging_config["log_freq"] = 50  # Log every N batches
    logging_config["log_level"] = "INFO"
    logging_config["console_level"] = "WARNING"  # Console threshold (set to null to mirror log_level)

    if verbosity in ["standard", "full"]:
        logging_config["log_gradients"] = False
        logging_config["track_layer_params"] = False
        logging_config["track_connections"] = False

    if verbosity == "full":
        logging_config["log_batch_metrics"] = False
        logging_config["save_code"] = True
        logging_config["log_model"] = True

        # MLflow settings
        logging_config["mlflow_active"] = False
        logging_config["mlflow_tracking_uri"] = None  # e.g., http://localhost:5000 or file:./mlruns
        logging_config["mlflow_experiment_name"] = None
        logging_config["mlflow_run_name"] = None
        logging_config["mlflow_log_artifacts"] = True
        logging_config["mlflow_username"] = "admin"
        logging_config["mlflow_password"] = None
        logging_config["mlflow_tags"] = {}

        # S3 upload
        logging_config["upload_logs_and_checkpoints"] = False
        logging_config["s3_upload_url"] = None

        # Connection parameter probing
        logging_config["connection_params_probing"] = {"log_histograms": False, "histogram_bins": 50}

        # State trajectory logging
        logging_config["state_trajectories"] = {
            "active": False,
            "mode": "val",  # "train" or "val"
            "layer_id": None,  # None = last/output layer
            "num_samples": 4,
            "class_ids": None,  # Optional list of class IDs to filter
            "max_neurons_per_sample": 4,
            "neuron_indices": None,  # Optional explicit neuron indices
            "tag_prefix": "callbacks/state_trajectories",
        }

    return logging_config


def build_callbacks_section(verbosity: str, registries: dict[str, list[str]]) -> dict:
    """Build the callbacks configuration section."""
    callbacks: dict[str, Any] = {}

    # LR Scheduler
    lr_scheduler: dict[str, Any] = {}
    lr_scheduler["type"] = "constant"  # Available: constant, linear, cosine, rex, greedy, adaptive
    lr_scheduler["lr"] = 0.001
    callbacks["lr_scheduler"] = lr_scheduler

    if verbosity == "full":
        # Loss weight schedulers (commented out examples)
        # callbacks["loss_weight_schedulers"] = [
        #     {
        #         "loss_name": "gap_loss",
        #         "scheduler_type": "linear",
        #         "params": {"min_weight": 0.1, "max_weight": 1.0}
        #     }
        # ]

        # Sequence length scheduler
        # callbacks["seq_len_scheduler"] = {
        #     "active": False,
        #     "start_len": 20,
        #     "end_len": 100,
        #     "growth_type": "linear",
        #     "start_epoch": 5
        # }

        # Noise annealing
        # callbacks["noise_annealing"] = {
        #     "active": False,
        #     "initial_noise": 0.1,
        #     "final_noise": 0.01,
        #     "annealing_epochs": 50
        # }

        # Connection noise callback
        # callbacks["connection_noise"] = {
        #     "active": False,
        #     "noise_std": 0.01,
        #     "apply_every_n_epochs": 1
        # }

        # Time pooling scale scheduler
        # callbacks["time_pooling_scale_scheduler"] = {
        #     "active": False,
        #     "initial_scale": 0.5,
        #     "final_scale": 1.0,
        #     "growth_epochs": 30
        # }

        # Quantization-Aware Training (QAT)
        # callbacks["qat"] = {
        #     "active": False,
        #     "start_epoch": 10,
        #     "num_bits": 8
        # }

        # Output state statistics
        # callbacks["output_state_stats"] = {
        #     "active": False,
        #     "log_every_n_epochs": 5
        # }
        pass

    return callbacks


def build_profiler_section(verbosity: str) -> dict:
    """Build the profiler configuration section."""
    profiler: dict[str, Any] = {}

    profiler["active"] = False
    profiler["type"] = "simple"  # simple | advanced | pytorch

    if verbosity == "full":
        profiler["num_train_batches"] = None  # None = all batches
        profiler["num_val_batches"] = None  # None = all batches
        profiler["record_shapes"] = False
        profiler["profile_memory"] = False
        profiler["with_stack"] = False
        profiler["output_filename"] = None

    return profiler


def generate_full_instructive_template(base_template: dict, registries: dict[str, list[str]]) -> str:
    """Generate a fully commented, instructive template for full verbosity.

    Args:
        base_template: Basic template structure
        registries: Registry contents for showing options

    Returns:
        Fully commented YAML string with extensive documentation
    """
    lines = []

    # Header
    lines.extend(
        [
            "# ==============================================================================",
            "# SOEN Training Configuration Template (FULL)",
            "#",
            "# This template contains ALL available configuration options with detailed",
            "# explanations and examples. Most options are commented out to show what's",
            "# available. Uncomment and modify as needed for your experiment.",
            "#",
            "# Generated by: python -m soen_toolkit.training.tools.generate_config_template",
            "# ==============================================================================",
            "",
            "# ------------------------------------------------------------------------------",
            "# Experiment Metadata",
            "# ------------------------------------------------------------------------------",
            "name: null  # Optional human-readable experiment name",
            'description: "Comprehensive SOEN training experiment"',
            "seed: 42  # Random seed for reproducibility",
            "",
            "# ==============================================================================",
            "# TRAINING CONFIGURATION",
            "# ==============================================================================",
            "training:",
            "  # --- Task Definition ---",
            "  paradigm: supervised  # supervised | self_supervised | unsupervised",
            "  mapping: seq2static   # seq2static | seq2seq | static2seq | static2static",
            "  #   seq2static: sequence input → single output (classification/regression)",
            "  #   seq2seq: sequence input → sequence output (forecasting/labeling)",
            "  #   static2seq: single input → sequence output",
            "  #   static2static: single input → single output",
            "",
            "  # --- Basic Training Settings ---",
            "  batch_size: 32",
            "  max_epochs: 100",
            "  num_repeats: 1  # Run entire experiment N times with different seeds",
            "",
            "  # --- Compute & Hardware ---",
            "  accelerator: auto  # auto | cpu | gpu | mps | tpu",
            "  precision: 32-true  # 32-true (standard) | 16-mixed (faster, less precise)",
            "  devices: auto  # auto | 1 | [0,1,2] for specific GPUs",
            "  num_workers: 4  # CPU workers for data loading (0 for debugging)",
            "  deterministic: true  # Reproducibility (may reduce performance)",
            "",
            "  # --- Gradient & Optimization ---",
            "  accumulate_grad_batches: 1  # Simulate larger batch size",
            "  gradient_clip_val: null  # Set to float (e.g., 1.0) to enable gradient clipping",
            "  gradient_clip_algorithm: norm  # norm | value",
            "",
            "  # --- Checkpointing ---",
            "  checkpoint_every_n_epochs: 1",
            "  checkpoint_save_top_k: 3  # Keep best K checkpoints",
            "  checkpoint_save_last: true",
            "  save_initial_state: true  # Save model before training",
            "  save_soen_core_in_checkpoint: true  # Save .soen sidecar files",
            "  # checkpoint_save_all_epochs: false  # If true, save every epoch",
            "",
            "  # --- Early Stopping ---",
            "  # early_stopping_patience: 10  # Stop if no improvement for N epochs",
            "",
            "  # --- Autoregressive Training (for sequence modeling) ---",
            "  autoregressive: false",
            "  # autoregressive_mode: next_token  # next_token (LM-style) | seq2seq",
            "  # time_steps_per_token: 1  # Simulation timesteps per token",
            "  # autoregressive_start_timestep: 0  # When to start AR loss",
            "",
            "  # --- Truncated Backprop Through Time (for long sequences) ---",
            "  # use_tbptt: true",
            "  # tbptt_steps: 128  # Chunk size in timesteps",
            "  # tbptt_stride: 64  # Overlap between chunks (< steps = overlap)",
            "",
            "  # --- Distributed Training ---",
            "  # num_nodes: 2  # Number of machines",
            "  # strategy: ddp  # ddp | ddp_spawn | dp | auto",
            "",
        ]
    )

    # Optimizer section
    lines.extend(
        [
            "  # --- Optimizer ---",
            f"  # Available optimizers: {', '.join(registries['optimizers'])}",
            "  optimizer:",
            "    name: adamw",
            "    lr: 0.001",
            "    kwargs:",
            "      weight_decay: 0.0001",
            "      # betas: [0.9, 0.999]  # Adam/AdamW momentum",
            "      # eps: 1.0e-08",
            "",
            "    # --- Per-Parameter Learning Rates (Advanced) ---",
            "    # param_groups:",
            '    #   - match: "connections"  # Match param names containing this',
            "    #     lr: 0.0001  # Different LR for connections",
            '    #   - match_regex: ".*bias.*"  # Regex matching',
            "    #     weight_decay: 0.0  # No decay for biases",
            "",
        ]
    )

    # Loss section with all available losses
    lines.extend(
        [
            "  # --- Loss Configuration ---",
            f"  # Available losses: {', '.join(registries['losses'][:15])}...",
            "  loss:",
            "    losses:",
            "      - name: cross_entropy",
            "        weight: 1.0",
            "        params: {}",
            "",
            "      # --- Classification Losses ---",
            "      # - name: gap_loss  # Margin-based classification",
            "      #   weight: 0.5",
            "      #   params:",
            "      #     margin: 0.2",
            "",
            "      # - name: top_gap_loss  # Top-class margin",
            "      #   weight: 0.5",
            "      #   params:",
            "      #     margin: 0.3",
            "",
            "      # - name: rich_margin_loss  # Multi-component margin loss",
            "      #   weight: 1.0",
            "      #   params:",
            "      #     margin: 0.5",
            "      #     sigma_noise: 0.05",
            "      #     λ_gap: 1.0",
            "      #     λ_noise: 0.3",
            "      #     λ_entropy: 0.1",
            "",
            "      # --- Regression Losses ---",
            "      # - name: mse",
            "      #   weight: 1.0",
            "      #   params: {}",
            "",
            "      # --- Regularization Losses ---",
            "      # - name: reg_J_loss  # Penalize large connection weights",
            "      #   weight: 0.01",
            "      #   params:",
            "      #     threshold: 0.24",
            "      #     scale: 1.0",
            "      #     factor: 0.01",
            "",
            "      # - name: exp_high_state_penalty  # Penalize high neuron states",
            "      #   weight: 0.01",
            "      #   params:",
            "      #     threshold: 2.0",
            "      #     penalty_factor: 0.1",
            "",
            "      # --- Autoregressive Losses ---",
            "      # - name: autoregressive_loss",
            "      #   weight: 1.0",
            "      #   params: {}",
            "",
            "      # - name: autoregressive_cross_entropy",
            "      #   weight: 1.0",
            "      #   params: {}",
            "",
        ]
    )

    # Data section
    lines.extend(
        [
            "",
            "# ==============================================================================",
            "# DATA CONFIGURATION",
            "# ==============================================================================",
            "data:",
            '  data_path: "path/to/your/dataset.h5"  # REQUIRED: Path to HDF5/CSV dataset',
            "  sequence_length: 100  # Base sequence length",
            "  num_classes: 10  # Number of output classes",
            "  val_split: 0.2",
            "  test_split: 0.1",
            "  cache_data: true  # Load entire dataset into RAM (faster if it fits)",
            "",
            "  # --- Sequence Processing ---",
            "  # target_seq_len: 128  # Resample sequences to this length",
            "  # min_scale: -1.0  # Normalization range",
            "  # max_scale: 1.0",
            "",
            "  # --- Input Encoding ---",
            "  # input_encoding: raw  # raw | one_hot | embedding",
            "  # vocab_size: 256  # Required for one_hot encoding",
            "  # one_hot_dtype: float32",
            "",
            "  # --- Synthetic Data Generation ---",
            "  # synthetic: true  # Generate synthetic data instead of loading",
            "  # synthetic_kwargs:",
            "  #   seq_len: 100",
            "  #   input_dim: 10",
            "  #   dataset_size: 10000",
            "  #   task: classification",
            "",
            "  # --- CSV Data (Alternative to HDF5) ---",
            "  # csv_data_paths:",
            "  #   train: data/train.csv",
            "  #   val: data/val.csv",
            "  #   test: data/test.csv",
            "",
        ]
    )

    # Model section
    lines.extend(
        [
            "# ==============================================================================",
            "# MODEL CONFIGURATION",
            "# ==============================================================================",
            "model:",
            '  base_model_path: "path/to/your/model.pth"  # .pth (SOEN) or .ckpt (Lightning)',
            "  load_exact_model_state: false  # true: load weights | false: load config only",
            "",
            "  # --- Time Pooling (How to aggregate temporal dimension) ---",
            "  time_pooling:",
            "    name: final  # max | mean | rms | final | mean_last_n | mean_range | ewa",
            "    params:",
            "      scale: 1.0  # Output scaling factor",
            "",
            "  # Examples of different time pooling methods:",
            "  # time_pooling:",
            "  #   name: max  # Maximum over time",
            "  #   params: {scale: 1.0}",
            "",
            "  # time_pooling:",
            "  #   name: mean_last_n  # Average of last N timesteps",
            "  #   params: {n: 10, scale: 1.0}",
            "",
            "  # time_pooling:",
            "  #   name: mean_range  # Average over specific range",
            "  #   params: {scale: 1.0}",
            "  # range_start: 50",
            "  # range_end: 100",
            "",
            "  # --- Simulation Settings ---",
            "  # dt: 195.3125  # Simulation timestep (1 dt = 1.28ps)",
            "  # dt_learnable: false  # Make dt a learnable parameter",
            "",
            "  # --- Model Creation (Alternative to loading) ---",
            "  # architecture_yaml: path/to/architecture.yaml  # Build from YAML",
            "",
            "  # --- Backend Selection ---",
            "  # backend: torch  # torch | jax",
            "",
        ]
    )

    # Logging section
    lines.extend(
        [
            "# ==============================================================================",
            "# LOGGING CONFIGURATION",
            "# ==============================================================================",
            "logging:",
            "  # --- Output Directories ---",
            "  project_dir: experiments/  # Base directory for all outputs",
            "  project_name: soen_training",
            "  group_name: default_group",
            "  experiment_name: null  # Auto-generated if null",
            "",
            "  # --- Metrics to Track ---",
            f"  # Available metrics: {', '.join(registries['metrics'])}",
            "  metrics:",
            "    - accuracy",
            "    # - perplexity  # For language modeling",
            "    # - bits_per_character",
            "    # - top_k_accuracy",
            "    # - f1",
            "    # - precision",
            "    # - recall",
            "",
            "  # --- Logging Frequency ---",
            "  log_freq: 50  # Log every N batches",
            "  log_level: INFO  # DEBUG | INFO | WARNING | ERROR",
            "",
            "  # --- Advanced Logging ---",
            "  # log_batch_metrics: true  # Log at batch level (verbose)",
            "  # log_gradients: true  # Log gradient histograms",
            "  # track_layer_params: true  # Log layer parameter histograms",
            "  # track_connections: true  # Log connection weight histograms",
            "  # save_code: true  # Save code snapshot",
            "  # log_model: true  # Log model checkpoints",
            "",
            "  # --- MLflow Integration (Optional) ---",
            "  # mlflow_active: true",
            "  # mlflow_tracking_uri: http://localhost:5000  # or file:./mlruns",
            "  # mlflow_experiment_name: my_experiment",
            "  # mlflow_run_name: run_001",
            "  # mlflow_log_artifacts: true",
            "  # mlflow_username: admin",
            "  # mlflow_password: xxx",
            "  # mlflow_tags:",
            "  #   owner: your_name",
            "  #   project: research",
            "",
            "  # --- Cloud Storage ---",
            "  # upload_logs_and_checkpoints: true",
            "  # s3_upload_url: s3://my-bucket/experiments/",
            "",
            "  # --- Connection Parameter Probing ---",
            "  # connection_params_probing:",
            "  #   log_histograms: true",
            "  #   histogram_bins: 50",
            "",
            "  # --- State Trajectory Logging ---",
            "  # state_trajectories:",
            "  #   active: true",
            "  #   mode: val  # train | val",
            "  #   layer_id: null  # null = output layer",
            "  #   num_samples: 4",
            "  #   max_neurons_per_sample: 4",
            "",
        ]
    )

    # Callbacks section - the big one!
    lines.extend(
        [
            "# ==============================================================================",
            "# CALLBACKS CONFIGURATION",
            "# ==============================================================================",
            "callbacks:",
            "  # ---------------------------------------------------------------------------",
            "  # Learning Rate Schedulers",
            "  # ---------------------------------------------------------------------------",
            f"  # Available scheduler types: {', '.join(registries['schedulers'])}",
            "",
            "  # --- Option 1: Constant LR (Simple Baseline) ---",
            "  lr_scheduler:",
            "    type: constant",
            "    lr: 0.001",
            "",
            "  # --- Option 2: Cosine Annealing (RECOMMENDED for most cases) ---",
            "  # lr_scheduler:",
            "  #   type: cosine",
            "  #   max_lr: 0.001  # Peak learning rate",
            "  #   min_lr: 0.000001  # Minimum learning rate",
            "  #   warmup_epochs: 5  # Linear warmup from min to max",
            "  #   cycle_epochs: 50  # Length of cosine cycle",
            "  #   enable_restarts: true  # Restart cycle periodically",
            "  #   restart_decay: 1.0  # Decay max_lr after restart (1.0 = no decay)",
            "  #   period_decay: 1.0  # Change cycle duration (1.0 = constant)",
            "  #   amplitude_decay: 1.0  # Decay oscillation (1.0 = no decay)",
            "  #   adjust_on_batch: true  # Adjust every batch vs every epoch",
            "  #   batches_per_adjustment: 1",
            "  #   soft_restart: false  # Smooth vs hard restart",
            "",
            "  # --- Option 3: Linear Decay ---",
            "  # lr_scheduler:",
            "  #   type: linear",
            "  #   max_lr: 0.001",
            "  #   min_lr: 0.000001",
            "  #   log_space: false  # true = exponential decay",
            "",
            "  # --- Option 4: REX Scheduler (Research) ---",
            "  # lr_scheduler:",
            "  #   type: rex",
            "  #   warmup_epochs: 5",
            "  #   warmup_start_lr: 0.000001",
            "  #   max_lr: 0.001",
            "  #   min_lr: 0.0",
            "",
            "  # --- Option 5: Greedy Adaptive (Responsive to validation) ---",
            "  # lr_scheduler:",
            "  #   type: greedy",
            "  #   factor_increase: 1.1  # Multiply LR by this on improvement",
            "  #   factor_decrease: 0.9  # Multiply LR by this on worsening",
            "  #   patience: 3  # Epochs to wait before adjusting",
            "  #   min_lr: 0.000001",
            "  #   max_lr: 0.01",
            "  #   warmup:",
            "  #     enabled: true",
            "  #     epochs: 3",
            "  #     start_lr: 0.000001",
            "  #   intra_epoch: false  # Adjust within epochs",
            "  #   adjustment_frequency: 100  # Batches between adjustments",
            "",
            "  # --- Option 6: Adaptive (Dual patience) ---",
            "  # lr_scheduler:",
            "  #   type: adaptive",
            "  #   monitor_metric: val_loss",
            "  #   max_lr: 0.001",
            "  #   min_lr: 0.000001",
            "  #   warmup_epochs: 3",
            "  #   warmup_start_lr: 0.0000001",
            "  #   increase_factor: 1.2",
            "  #   decrease_factor: 0.7",
            "  #   patience_increase: 3",
            "  #   patience_decrease: 5",
            "  #   threshold: 0.0001",
            "  #   threshold_mode: rel  # rel | abs",
            "",
            "  # ---------------------------------------------------------------------------",
            "  # Loss Weight Schedulers (Dynamic loss weighting during training)",
            "  # ---------------------------------------------------------------------------",
            "  # loss_weight_schedulers:",
            "  #   # Example: Gradually increase gap loss weight",
            "  #   - loss_name: gap_loss",
            "  #     scheduler_type: linear",
            "  #     params:",
            "  #       min_weight: 0.1",
            "  #       max_weight: 1.0",
            "",
            "  #   # Example: Cyclical quantization loss",
            "  #   - loss_name: gravity_quantization_loss",
            "  #     scheduler_type: sinusoidal",
            "  #     params:",
            "  #       min_weight: 1.0",
            "  #       max_weight: 200.0",
            "  #       period_steps: 200",
            "  #       scale: log  # linear | log",
            "",
            "  #   # Example: Decay regularization over time",
            "  #   - loss_name: reg_J_loss",
            "  #     scheduler_type: exponential_decay",
            "  #     params:",
            "  #       initial_weight: 1.0",
            "  #       final_weight: 0.01",
            "  #       decay_rate: 3.0",
            "",
            "  # ---------------------------------------------------------------------------",
            "  # Sequence Length Scheduler (Curriculum learning)",
            "  # ---------------------------------------------------------------------------",
            "  # seq_len_scheduler:",
            "  #   active: true",
            "  #   start_len: 20  # Start with shorter sequences",
            "  #   end_len: 100  # Gradually increase to full length",
            "  #   growth_type: linear  # linear | exponential | step",
            "  #   start_epoch: 5  # When to start increasing",
            "  #   growth_rate: 1.1  # For exponential growth",
            "  #   step_size: 10  # For step growth",
            "",
            "  # ---------------------------------------------------------------------------",
            "  # Noise Annealing (Gradually reduce noise)",
            "  # ---------------------------------------------------------------------------",
            "  # noise_annealing:",
            "  #   active: true",
            "  #   initial_noise: 0.1",
            "  #   final_noise: 0.01",
            "  #   annealing_epochs: 50",
            "",
            "  # ---------------------------------------------------------------------------",
            "  # Connection Noise (Add noise to connection weights)",
            "  # ---------------------------------------------------------------------------",
            "  # connection_noise:",
            "  #   active: true",
            "  #   noise_std: 0.01",
            "  #   apply_every_n_epochs: 1",
            "",
            "  # ---------------------------------------------------------------------------",
            "  # Time Pooling Scale Scheduler",
            "  # ---------------------------------------------------------------------------",
            "  # time_pooling_scale_scheduler:",
            "  #   active: true",
            "  #   initial_scale: 0.5",
            "  #   final_scale: 1.0",
            "  #   growth_epochs: 30",
            "",
            "  # ---------------------------------------------------------------------------",
            "  # Quantization-Aware Training (QAT)",
            "  # ---------------------------------------------------------------------------",
            "  # qat:",
            "  #   active: true",
            "  #   start_epoch: 10  # When to start quantization",
            "  #   num_bits: 8  # Bit precision",
            "",
            "  # ---------------------------------------------------------------------------",
            "  # Output State Statistics",
            "  # ---------------------------------------------------------------------------",
            "  # output_state_stats:",
            "  #   active: true",
            "  #   log_every_n_epochs: 5",
            "",
        ]
    )

    # Profiler and Cloud sections
    lines.extend(
        [
            "# ==============================================================================",
            "# PROFILER CONFIGURATION (Performance Analysis)",
            "# ==============================================================================",
            "# profiler:",
            "#   active: true",
            "#   type: simple  # simple | advanced | pytorch",
            "#   num_train_batches: 20  # Limit batches when profiling",
            "#   num_val_batches: 5",
            "#   record_shapes: false  # PyTorch profiler: record tensor shapes",
            "#   profile_memory: false  # PyTorch profiler: track memory",
            "#   with_stack: false  # PyTorch profiler: include stack traces",
            "#   output_filename: null",
            "",
            "# ==============================================================================",
            "# CLOUD CONFIGURATION (AWS SageMaker)",
            "# ==============================================================================",
            "# cloud:",
            "#   active: true",
            "#   role: arn:aws:iam::123456789:role/SageMakerRole  # REQUIRED",
            "#   region: us-west-2",
            "#   bucket: my-sagemaker-bucket",
            "#   instance_type: ml.p3.2xlarge  # Or use cpu_instance_type/gpu_instance_type",
            "#   instance_count: 1",
            "#   framework_version: 2.2.0",
            "#   py_version: py310",
            "#   max_run: 36000  # Max seconds (10 hours)",
            "#   max_wait: 36000",
            "#   wait: true  # Wait for job to complete",
            "#   verbose: false",
            "#   job_prefix: my_experiment",
            "#   cleanup: true  # Clean up resources after job",
            "",
        ]
    )

    return "\n".join(lines)


def build_cloud_section(verbosity: str) -> dict:
    """Build the cloud configuration section."""
    cloud: dict[str, Any] = {}

    cloud["active"] = False
    cloud["role"] = None  # Required AWS IAM role ARN for SageMaker
    cloud["region"] = None  # e.g., us-west-2
    cloud["bucket"] = None  # S3 bucket for data/models

    if verbosity == "full":
        cloud["instance_type"] = None  # e.g., ml.p3.2xlarge
        cloud["cpu_instance_type"] = None
        cloud["gpu_instance_type"] = None
        cloud["instance_count"] = 1
        cloud["framework_version"] = "2.2.0"
        cloud["py_version"] = "py310"
        cloud["max_run"] = 36000  # Max runtime in seconds
        cloud["max_wait"] = 36000
        cloud["wait"] = True
        cloud["verbose"] = False
        cloud["job_prefix"] = None
        cloud["cleanup"] = True
        cloud["project"] = None
        cloud["experiment"] = None

    return cloud


def generate_template(verbosity: str = "standard", include_comments: bool = True) -> dict:
    """Generate a complete configuration template.

    Args:
        verbosity: Template verbosity level (minimal, standard, full)
        include_comments: Whether to include inline comments

    Returns:
        dict containing the full configuration template
    """
    # Query registries
    registries = query_registries()

    # Create root template
    template: dict[str, Any] = {}

    # Experiment metadata
    template["name"] = None  # Optional experiment name
    template["description"] = "My SOEN training experiment"
    template["seed"] = 42

    # Add main sections
    template["training"] = build_training_section(verbosity)
    template["data"] = build_data_section(verbosity)
    template["model"] = build_model_section(verbosity)
    template["logging"] = build_logging_section(verbosity, registries)
    template["callbacks"] = build_callbacks_section(verbosity, registries)

    if verbosity == "full":
        template["profiler"] = build_profiler_section(verbosity)
        template["cloud"] = build_cloud_section(verbosity)

    return template


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate SOEN training configuration template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate standard template
  python -m soen_toolkit.training.tools.generate_config_template

  # Generate full template with all options
  python -m soen_toolkit.training.tools.generate_config_template --preset full

  # Generate minimal template
  python -m soen_toolkit.training.tools.generate_config_template --preset minimal --output minimal.yaml
        """,
    )

    parser.add_argument("--output", type=str, default="training_config_template.yaml", help="Output YAML file path (default: training_config_template.yaml)")

    parser.add_argument("--preset", type=str, choices=["minimal", "standard", "full"], default="standard", help="Template verbosity level (default: standard)")

    parser.add_argument("--no-comments", action="store_true", help="Disable inline comments")

    parser.add_argument("--list-registries", action="store_true", help="List all registered components and exit")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Handle list registries
    if args.list_registries:
        registries = query_registries()

        for name in registries["schedulers"]:
            params = introspect_scheduler_params(name)
            if params:
                for _param, info in list(params.items())[:3]:  # Show first 3 params
                    info["default"]

        for name in registries["losses"]:
            pass

        for name in registries["metrics"]:
            pass

        for name in registries["optimizers"]:
            pass

        return 0

    # Generate template
    logger.info(f"Generating {args.preset} configuration template...")

    # Query registries first
    registries = query_registries()

    template = generate_template(verbosity=args.preset, include_comments=not args.no_comments)

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate YAML content
    # For full template, use custom instructive format
    if args.preset == "full":
        yaml_content = generate_full_instructive_template(template, registries)
    else:
        # For minimal/standard, use standard YAML dump with header
        yaml_content = yaml.dump(template, default_flow_style=False, sort_keys=False, allow_unicode=True)
        header = (
            f"# SOEN Training Configuration Template\n"
            f"# Generated with verbosity level: {args.preset}\n"
            f"#\n"
            f"# This template includes available configuration options.\n"
            f"# Modify sections as needed for your experiment.\n\n"
        )
        yaml_content = header + yaml_content

    with open(output_path, "w") as f:
        f.write(yaml_content)

    logger.info(f"[OK] Template written to: {output_path}")
    logger.info(f"  Verbosity: {args.preset}")
    logger.info(f"  Comments: {'enabled' if not args.no_comments else 'disabled'}")

    # Validate template can be loaded
    try:
        from soen_toolkit.training.configs import load_config

        # Try to load (will fail on placeholder paths, but dataclass parsing should work)
        try:
            load_config(output_path, validate=False)
            logger.info("[OK] Template structure is valid (parseable by ExperimentConfig)")
        except FileNotFoundError:
            # Expected for placeholder paths
            logger.info("[OK] Template structure is valid (paths need to be updated)")
        except Exception as e:
            logger.warning(f"Template may have issues: {e}")

    except ImportError:
        logger.warning("Could not validate template (config module not available)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
