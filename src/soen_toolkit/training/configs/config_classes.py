"""Configuration Classes for SOEN Model Training Experiments.

This module defines a set of dataclasses that structure the configuration
for training SOEN models. Think of these as the blueprints for your experiment's
settings, making it easy to define everything from data paths and batch sizes
to intricate learning rate schedules and custom loss combinations in a clear,
type-safe way.

These classes are designed to be easily created from YAML files, allowing for
flexible and reproducible experimentation.
"""

from collections.abc import Callable
import contextlib
from dataclasses import dataclass, field
import logging  # For potential warnings/info in post_init
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, cast
import warnings

if TYPE_CHECKING:
    from soen_toolkit.core.noise import NoiseSettings

logger = logging.getLogger(__name__)
# -----------------------------------------------------------------------------
# Profiler Configuration (Trainer-level)
# -----------------------------------------------------------------------------


@dataclass
class ProfilerConfig:
    """Training profiler settings.

    - active: Enable/disable profiling
    - type: "simple" | "advanced" | "pytorch"
    - num_train_batches: Limit training batches during profiling (None = all batches)
    - num_val_batches: Limit validation batches during profiling (None = all batches)
    - record_shapes/profile_memory/with_stack: PyTorch profiler options
    - output_filename: Optional basename for profiler output
    """

    active: bool = False
    type: str = "simple"
    num_train_batches: int | None = None
    num_val_batches: int | None = None
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    output_filename: str | None = None
    # JAX-specific: fine-grained trace control
    trace_start_batch: int = 5
    trace_duration_batches: int = 5

    def __post_init__(self) -> None:
        """Post-initialization to handle type conversions from YAML."""
        if isinstance(self.active, str):
            self.active = self.active.lower() in {"true", "1", "yes", "on"}
        if isinstance(self.record_shapes, str):
            self.record_shapes = self.record_shapes.lower() in {"true", "1", "yes", "on"}
        if isinstance(self.profile_memory, str):
            self.profile_memory = self.profile_memory.lower() in {"true", "1", "yes", "on"}
        if isinstance(self.with_stack, str):
            self.with_stack = self.with_stack.lower() in {"true", "1", "yes", "on"}


# -----------------------------------------------------------------------------
# Core Configuration Components (Often Reused or Nested)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseConfig:
    """Stochastic noise levels applied each timestep."""

    phi: float = 0.0
    g: float = 0.0
    s: float = 0.0
    bias_current: float = 0.0
    j: float = 0.0
    relative: bool = False
    extras: dict[str, float] = field(default_factory=dict)

    def to_settings(
        self,
        perturb: "PerturbationConfig | None" = None,
    ) -> "NoiseSettings":
        from soen_toolkit.core.noise import build_noise_strategies

        return build_noise_strategies(self, perturb)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PerturbationConfig:
    """Deterministic offsets with optional Gaussian spread."""

    phi_mean: float = 0.0
    phi_std: float = 0.0
    g_mean: float = 0.0
    g_std: float = 0.0
    s_mean: float = 0.0
    s_std: float = 0.0
    bias_current_mean: float = 0.0
    bias_current_std: float = 0.0
    j_mean: float = 0.0
    j_std: float = 0.0
    extras_mean: dict[str, float] = field(default_factory=dict)
    extras_std: dict[str, float] = field(default_factory=dict)

    def to_settings(self, noise: NoiseConfig | None = None) -> "NoiseSettings":
        from soen_toolkit.core.noise import build_noise_strategies

        return build_noise_strategies(noise, self)  # type: ignore[arg-type]


@dataclass
class WarmupConfig:
    """Settings for a learning rate warmup phase at the beginning of training.
    Warming up the learning rate (gradually increasing it from a small value)
    can help stabilize training, especially for complex models or sensitive optimizers.
    """

    epochs: int = 0  # How many epochs should the warmup last?
    start_lr: float = 1e-6  # What's the initial learning rate for the warmup?
    enabled: bool = True  # Is warmup active? (Useful for scheduler configs that embed this)

    def __str__(self) -> str:
        if not self.enabled or self.epochs == 0:
            return "no_warmup"
        return f"warmup_{self.epochs}ep_{self.start_lr:.0e}"


@dataclass
class OptimizerConfig:
    """Specifies the optimizer to be used for training and its parameters.
    For example, AdamW is a common choice.
    """

    name: str = "adamw"  # Which optimizer? e.g., "adam", "adamw", "sgd"
    lr: float = 1e-3  # The main learning rate for the optimizer.
    # You can pass any other optimizer-specific arguments here, like weight_decay.
    kwargs: dict[str, Any] = field(default_factory=lambda: {"weight_decay": 1e-4})
    # Optional parameter groups to allow per-parameter learning rates or options.
    # Each entry can specify matching rules (e.g., 'match', 'match_regex') and
    # optimizer options like 'lr', 'weight_decay', etc.
    param_groups: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        kw_str = "_".join(f"{k}{v}" for k, v in self.kwargs.items())
        return f"{self.name}_lr{self.lr}_{kw_str}"


# -----------------------------------------------------------------------------
# Loss Function Configuration
# -----------------------------------------------------------------------------


@dataclass
class LossItemConfig:
    """Configuration for a single loss component.

    - name: identifier registered in LOSS_REGISTRY (or built-in where applicable)
    - weight: scaling factor for the loss
    - params: kwargs to pass when constructing the loss (for nn.Module losses) or
      when calling functional losses
    """

    name: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """Groups all loss-related configurations for the training process.

    Only a flat list of losses is supported. The legacy "base + additional" style
    has been removed to simplify configuration and reduce branching in the code.
    """

    losses: list[LossItemConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Parse list items (dict -> LossItemConfig)
        processed_losses = []
        for i, item in enumerate(self.losses):
            if isinstance(item, dict):
                try:
                    processed_losses.append(LossItemConfig(**item))
                except TypeError as e:
                    logger.exception(
                        f"Error parsing 'losses' item at index {i}: {item}. Details: {e}",
                    )
                    raise
            elif isinstance(item, LossItemConfig):
                processed_losses.append(item)
            else:
                msg = f"Invalid type for losses item at index {i}: {type(item)}"
                raise ValueError(
                    msg,
                )
        self.losses = processed_losses


# -----------------------------------------------------------------------------
# Learning Rate Scheduler Configurations
# -----------------------------------------------------------------------------
# These define how the learning rate changes over the course of training.
# Each scheduler has its own specific parameters.
# -----------------------------------------------------------------------------


@dataclass
class ConstantConfig:
    """Keeps the learning rate constant. Good for baseline experiments."""

    # If None, the optimizer's initial learning rate is used throughout.
    lr: float | None = None

    def __str__(self) -> str:
        lr_str = f"_{self.lr}" if self.lr is not None else "_defaultLR"
        return f"constant{lr_str}"


@dataclass
class LinearConfig:
    """Linearly decays the learning rate from a max value to a min value."""

    max_lr: float = 1e-3  # Starting learning rate.
    min_lr: float = 1e-6  # Learning rate at the end of training.
    log_space: bool = False  # If True, decay linearly in log-space (exponential decay).

    def __str__(self) -> str:
        log_str = "_log" if self.log_space else ""
        return f"linear{log_str}_max{self.max_lr}_min{self.min_lr}"


@dataclass
class CosineConfig:
    """Cosine annealing learning rate schedule, optionally with warmup and restarts.
    This is a popular and effective scheduler.
    """

    max_lr: float = 1e-3  # Peak learning rate after warmup / at restarts.
    min_lr: float = 1e-6  # Lowest learning rate.
    warmup_epochs: int = 5  # Linear warmup from min_lr to max_lr.
    cycle_epochs: int = 50  # Duration of the first cosine cycle.
    enable_restarts: bool = True  # Whether to restart the cosine cycle.
    restart_decay: float = 1.0  # Factor to decay max_lr after each restart (1.0 = no decay).
    period_decay: float = 1.0  # Factor to change cycle duration after restart (1.0 = constant).
    amplitude_decay: float = 1.0  # Factor to decay (max_lr - min_lr) after restart (1.0 = no decay).
    adjust_on_batch: bool = True  # True: adjust LR every batch. False: every epoch.
    batches_per_adjustment: int = 1  # How many batches between LR adjustments if adjust_on_batch=True.
    soft_restart: bool = False  # True: smoother restart (full cosine cycle). False: hard restart (half cycle).

    def __str__(self) -> str:
        restart_str = f"restarts{self.restart_decay:.1f}" if self.enable_restarts else "norestarts"
        adjust = "batch" if self.adjust_on_batch else "epoch"
        cycle = "soft" if self.soft_restart else "hard"
        return f"cosine_w{self.warmup_epochs}_c{self.cycle_epochs}_{restart_str}_{adjust}_{cycle}"


@dataclass
class RexConfig:
    """Rational EXponential (REX) scheduler with optional warmup."""

    warmup_epochs: int = 0  # Number of epochs for linear warmup.
    warmup_start_lr: float = 1e-6  # Starting LR for warmup.
    min_lr: float = 0.0  # Minimum LR to decay to.
    max_lr: float | None = None  # Peak LR after warmup (if None, uses optimizer's initial LR).

    def __str__(self) -> str:
        warmup_str = f"w{self.warmup_epochs}" if self.warmup_epochs > 0 else "nowarmup"
        return f"rex_{warmup_str}_start{self.warmup_start_lr}_max{self.max_lr}_min{self.min_lr}"


@dataclass
class GreedyConfig:
    """Adaptively adjusts LR based on validation loss trends (increases on improvement,
    decreases on worsening), with optional warmup and intra-epoch adjustments.
    """

    factor_increase: float = 1.1  # Factor to multiply LR by on improvement.
    factor_decrease: float = 0.9  # Factor to multiply LR by on worsening.
    patience: int = 3  # Epochs to wait before adjusting LR based on validation.
    min_lr: float = 1e-6
    max_lr: float = 0.01
    # Warmup settings can be provided as a dictionary matching WarmupConfig fields
    warmup: dict[str, Any] | None = None
    intra_epoch: bool = False  # Adjust LR within epochs based on training loss EMA?
    adjustment_frequency: int = 100  # Batches between intra-epoch adjustments.
    ema_beta: float = 0.9  # EMA beta for training loss if intra_epoch=True.
    debug: bool = False  # Enable verbose logging for this scheduler.

    def __str__(self) -> str:
        intra = "intra" if self.intra_epoch else "epoch"
        return f"greedy_{intra}_p{self.patience}_f{self.factor_increase}_{self.factor_decrease}_{self.warmup}"


@dataclass
class AdaptiveConfig:
    """Monitors a metric and adjusts LR if it improves or worsens, with different
    patience settings for each direction. Also supports warmup.
    """

    monitor_metric: str = "val_loss"  # Which metric to watch (e.g., "val_loss").
    max_lr: float = 1e-3
    min_lr: float = 1e-6
    warmup_epochs: int = 3
    warmup_start_lr: float = 1e-7
    increase_factor: float = 1.2  # LR multiplier when metric improves.
    decrease_factor: float = 0.7  # LR multiplier when metric worsens.
    patience_increase: int = 3  # Epochs of improvement before increasing LR.
    patience_decrease: int = 5  # Epochs of no improvement before decreasing LR.
    threshold: float = 1e-4  # Minimum change to qualify as improvement.
    threshold_mode: str = "rel"  # 'rel' or 'abs' for interpreting threshold.
    cooldown: int = 0  # Epochs to wait after an LR change before another.
    debug: bool = False

    def __str__(self) -> str:
        warmup_str = f"_w{self.warmup_epochs}" if self.warmup_epochs > 0 else ""
        return f"adaptive{warmup_str}_max{self.max_lr}_min{self.min_lr}_inc{self.increase_factor}_dec{self.decrease_factor}"


# A type hint for any of the above scheduler configurations.
SchedulerConfig = Union[
    ConstantConfig,
    LinearConfig,
    CosineConfig,
    RexConfig,
    GreedyConfig,
    AdaptiveConfig,
]

# -----------------------------------------------------------------------------
# Main Sections of the Experiment Configuration
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    """All settings related to data loading, preprocessing, and dataset properties."""

    sequence_length: int = 100  # Base sequence length for input features.
    min_scale: float | None = None  # Min value for spectrogram normalization.
    max_scale: float | None = None  # Max value for spectrogram normalization.
    # Path to your data file (e.g., an HDF5 file). Can be relative or absolute.
    data_path: str | Path = "./data"
    cache_data: bool = True  # Load entire dataset into RAM? (Faster if it fits).
    # Number of classes for classification tasks. This influences the model's output layer.
    num_classes: int = 10
    val_split: float = 0.2  # Fraction of data for validation.
    test_split: float = 0.1  # Fraction of data for testing.

    # If using a sequence length scheduler, this is the *target* length for the current epoch.
    target_seq_len: int | None = None

    # One-hot encoding settings for sequence/character data
    input_encoding: str = "raw"  # "raw" (default), "one_hot", or "embedding"
    vocab_size: int | None = None  # Number of unique tokens/characters (required for one_hot)
    one_hot_dtype: str = "float32"  # Data type for one-hot vectors

    # Dataset-less/synthetic options
    synthetic: bool = False  # If True, generate synthetic inputs and dummy targets
    synthetic_kwargs: dict[str, Any] = field(default_factory=dict)  # seq_len, input_dim, dataset_size, task

    # Future features (not yet fully implemented based on provided code)
    augmentation: bool = False
    augmentation_kwargs: dict[str, Any] = field(default_factory=dict)

    # Optional explicit CSV split paths
    csv_data_paths: dict[str, str | Path] | None = None  # keys: train, val, test

    def __post_init__(self) -> None:
        """Convert path strings to Path objects and set defaults."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)

        # Normalize csv split paths to Path
        if self.csv_data_paths:
            # Accept dict or list of single-key dicts
            raw = self.csv_data_paths
            as_dict: dict[str, str | Path] = {}
            if isinstance(raw, dict):
                as_dict = raw
            elif isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            as_dict[k] = v
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        k, v = item
                        as_dict[str(k)] = v
                    else:
                        msg = "csv_data_paths list entries must be dicts or (key, value) pairs"
                        raise TypeError(msg)
            else:
                msg = "csv_data_paths must be a dict or list of dicts"
                raise TypeError(msg)

            normalized: dict[str, str | Path] = {}
            for k, v in as_dict.items():
                if isinstance(v, str):
                    normalized[k] = Path(v)
                elif isinstance(v, Path):
                    normalized[k] = v
                else:
                    normalized[k] = Path(str(v))
            self.csv_data_paths = normalized

        # Validation for one-hot encoding
        if self.input_encoding == "one_hot":
            if self.vocab_size is None:
                msg = "vocab_size must be specified when input_encoding='one_hot'"
                raise ValueError(
                    msg,
                )
            if self.vocab_size <= 0:
                msg = "vocab_size must be positive"
                raise ValueError(msg)

        # Set vocab_size from num_classes if not specified and using one_hot
        if self.input_encoding == "one_hot" and self.vocab_size is None:
            self.vocab_size = self.num_classes
            logger.info(
                f"Using num_classes ({self.num_classes}) as vocab_size for one-hot encoding",
            )

        # Note: num_classes remains here as it's often tied to the dataset itself
        # and needed for configuring the model's output head correctly.



@dataclass
class AutoregressiveConfig:
    """Configuration for autoregressive training.

    Autoregressive training is used for sequence generation tasks where the model
    predicts the next token given previous tokens (e.g., language modeling, time series).

    For multi-timestep AR, each token can be processed over multiple simulation timesteps
    to allow the model's dynamics to settle before making a prediction.
    """

    enabled: bool = False
    """Enable autoregressive training mode."""

    mode: str = "next_token"
    """AR mode: 'next_token' (standard language modeling) or 'seq2seq' (use provided labels)."""

    # Multi-timestep settings
    time_steps_per_token: int = 1
    """Number of simulation timesteps per token.

    For standard AR: 1 timestep per token
    For multi-timestep AR: N timesteps per token (e.g., 4)

    Example:
        time_steps_per_token: 4
        Input:  ["h", "h", "h", "h", "i", "i", "i", "i"]
        Output: Pool 4 timesteps -> 1 prediction per token
    """

    start_timestep: int = 0
    """Which timestep to start computing AR loss from (usually 0)."""

    # Token-level pooling (for multi-timestep)
    token_pooling: dict[str, Any] = field(default_factory=lambda: {
        "method": "final",
        "params": {}
    })
    """How to pool timesteps within each token for multi-timestep AR.

    Methods:
        - "final": Use last timestep (default, most common)
        - "mean": Average all timesteps
        - "max": Max over timesteps
        - "mean_last_n": Average last N timesteps (requires params: {"n": 2})

    Example:
        token_pooling:
          method: "mean_last_n"
          params:
            n: 2  # Average last 2 timesteps
    """

    # Loss configuration
    loss: str = "autoregressive_cross_entropy"
    """Loss function for AR training. Usually 'autoregressive_cross_entropy'."""

    loss_weight: float = 1.0
    """Weight for the AR loss (if using multiple losses)."""

    def __post_init__(self) -> None:
        """Validate autoregressive configuration."""
        # Validate mode
        if self.mode not in {"next_token", "seq2seq"}:
            raise ValueError(
                f"Invalid autoregressive mode: '{self.mode}'. "
                f"Valid options: 'next_token', 'seq2seq'"
            )

        # Validate time_steps_per_token
        if self.enabled and self.time_steps_per_token <= 0:
            raise ValueError(
                f"time_steps_per_token must be positive when autoregressive is enabled, "
                f"got {self.time_steps_per_token}"
            )

        # Validate start_timestep
        if self.enabled and self.start_timestep < 0:
            raise ValueError(
                f"start_timestep must be non-negative, got {self.start_timestep}"
            )

        # Parse and validate token_pooling
        if isinstance(self.token_pooling, dict):
            method = self.token_pooling.get("method", "final")
            params = self.token_pooling.get("params", {})

            # Validate pooling method
            valid_methods = {"final", "mean", "max", "mean_last_n"}
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid token_pooling method: '{method}'. "
                    f"Valid options: {', '.join(sorted(valid_methods))}"
                )

            # Validate method-specific params
            if method == "mean_last_n":
                n = params.get("n")
                if n is None:
                    raise ValueError(
                        "token_pooling method 'mean_last_n' requires 'n' parameter. "
                        "Example: token_pooling: {method: 'mean_last_n', params: {n: 2}}"
                    )
                if not isinstance(n, int) or n <= 0:
                    raise ValueError(
                        f"token_pooling 'mean_last_n' requires positive integer 'n', got {n}"
                    )
                if n > self.time_steps_per_token:
                    raise ValueError(
                        f"token_pooling 'mean_last_n' n={n} cannot exceed "
                        f"time_steps_per_token={self.time_steps_per_token}"
                    )

        # Log configuration if enabled
        if self.enabled:
            logger.info(
                f"Autoregressive training enabled: "
                f"mode={self.mode}, "
                f"time_steps_per_token={self.time_steps_per_token}, "
                f"token_pooling={self.token_pooling.get('method', 'final')}"
            )


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training.

    Distillation trains a student model to match the output state trajectories
    of a teacher model. The teacher model is run on the dataset to generate
    target trajectories, which are then used as regression targets for the student.

    Either teacher_model_path (to generate trajectories) or distillation_data_path
    (to use pre-generated trajectories) must be provided.
    """

    teacher_model_path: str | Path | None = None
    """Path to the teacher model (.pth or .soen file). Required if generating new trajectories."""

    distillation_data_path: str | Path | None = None
    """Optional path to existing distillation dataset (.hdf5). If provided and exists,
    skips teacher trajectory generation and uses this dataset directly."""

    subset_fraction: float = 1.0
    """Fraction of dataset to use for distillation (0.0-1.0). Default uses all data."""

    max_samples: int | None = None
    """Maximum number of samples to use (applied after subset_fraction). None = no limit."""

    batch_size: int = 32
    """Batch size for teacher inference during trajectory generation."""

    def __post_init__(self) -> None:
        """Validate distillation configuration."""
        # Convert paths to Path objects
        if isinstance(self.teacher_model_path, str):
            self.teacher_model_path = Path(self.teacher_model_path)
        if isinstance(self.distillation_data_path, str):
            self.distillation_data_path = Path(self.distillation_data_path)

        # Validate that at least one path is provided
        if self.teacher_model_path is None and self.distillation_data_path is None:
            msg = "Either teacher_model_path or distillation_data_path must be provided"
            raise ValueError(msg)

        # If using existing distillation data, teacher_model_path is optional
        if self.distillation_data_path is not None:
            if not self.distillation_data_path.exists():
                # Warn but don't fail - might be intentional to generate it
                if self.teacher_model_path is None:
                    msg = (
                        f"distillation_data_path does not exist: {self.distillation_data_path}\n"
                        "and teacher_model_path is not provided. Cannot generate trajectories."
                    )
                    raise ValueError(msg)

        if not 0.0 < self.subset_fraction <= 1.0:
            msg = f"subset_fraction must be in (0.0, 1.0], got {self.subset_fraction}"
            raise ValueError(msg)

        if self.max_samples is not None and self.max_samples <= 0:
            msg = f"max_samples must be positive, got {self.max_samples}"
            raise ValueError(msg)

        if self.batch_size <= 0:
            msg = f"batch_size must be positive, got {self.batch_size}"
            raise ValueError(msg)


@dataclass
class TrainingConfig:

    """Parameters that directly control the training loop and optimization process.

    The ``autoregressive_mode`` field selects between ``next_token`` language-model
    style shifting and ``seq2seq`` where targets already contain the full
    prediction sequence.
    """

    batch_size: int = 32
    max_epochs: int = 100
    # Save the full SOENModelCore state with checkpoints? (Uses model.save_soen).
    save_soen_core_in_checkpoint: bool = False
    # Path to a .ckpt file to resume training from.
    train_from_checkpoint: str | None = None

    # ---- Compute Settings ----
    accelerator: str = "auto"  # e.g., "cpu", "gpu", "mps", "tpu"
    precision: str = "32-true"  # e.g., "32-true", "16-mixed"
    deterministic: bool = True  # For reproducibility.
    num_workers: int = 4  # DataLoader workers.
    persistent_workers: bool | None = None  # If None, choose a sensible default for HDF5 (False)
    prefetch_factor: int | None = None  # Per-worker prefetch, set small (1) for HDF5
    multiprocessing_context: str | None = None  # 'spawn'|'fork'|'forkserver' (platform dependent)
    devices: int | list[int] | str = "auto"  # e.g., 1 (for 1 GPU), [0,1] (for 2 GPUs), "auto"
    # Distributed training parameters (optional, pass-through to Lightning Trainer)
    num_nodes: int | None = None  # e.g., 2 for multi-node
    strategy: str | None = None  # e.g., "ddp", "ddp_spawn", "dp", "auto"

    num_repeats: int = 1  # How many times to repeat the entire experiment (with different seeds).

    # ---- Autoregressive Training Settings (NEW) ----
    # Use this for new configs - all AR settings in one place
    ar: AutoregressiveConfig | None = None
    """Autoregressive training configuration (recommended).

    Example:
        ar:
          enabled: true
          time_steps_per_token: 4
          token_pooling:
            method: "final"
    """

    # ---- Autoregressive Training Settings (DEPRECATED) ----
    # These are kept for backward compatibility only
    # New configs should use the 'ar:' section above
    autoregressive: bool = False  # DEPRECATED: Use ar.enabled instead
    time_steps_per_token: int = 1  # DEPRECATED: Use ar.time_steps_per_token instead
    autoregressive_start_timestep: int = 0  # DEPRECATED: Use ar.start_timestep instead
    autoregressive_mode: str = "next_token"  # DEPRECATED: Use ar.mode instead

    # ---- Minimal new axes (non-breaking; optional) ----
    paradigm: str = "supervised"  # supervised | self_supervised | unsupervised | distillation
    mapping: str = "seq2static"  # seq2seq | seq2static | static2seq | static2static

    # ---- Distillation Training Settings ----
    distillation: DistillationConfig | None = None
    """Distillation configuration for training a student to match teacher outputs.

    Example:
        distillation:
          teacher_model_path: "path/to/teacher.pth"
          subset_fraction: 0.5
          output_layer_only: true
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Unified loss configuration: a single list of losses under TrainingConfig.loss.losses
    loss: LossConfig = field(default_factory=LossConfig)

    # Deprecated fields for backward compatibility
    losses: list[Any] | None = None

    accumulate_grad_batches: int = 1  # Accumulate gradients over N batches.
    gradient_clip_val: float | None = None  # Value for gradient clipping.
    gradient_clip_algorithm: str = "norm"  # "norm" or "value" for clipping.
    # Stop training early if validation metric doesn't improve for N epochs.
    early_stopping_patience: int | None = None

    # --- Checkpointing options ---
    checkpoint_every_n_epochs: int = 1  # How often to save checkpoints
    checkpoint_save_top_k: int = 3  # How many best checkpoints to keep
    checkpoint_save_last: bool = True  # Always keep a 'last' checkpoint
    checkpoint_save_all_epochs: bool = False  # If True, save a checkpoint for every epoch (disables top-k)
    save_initial_state: bool = True  # Save model state before training begins

    # Loss alternation remains as an option for multiple components
    alternate_losses: bool = False  # Alternate loss components batch-wise
    batches_per_loss: int = 1  # Number of batches before switching to the next loss

    # ---- Truncated Backprop Through Time (TBPTT) ----
    # Enable Lightning's TBPTT and set chunk length (in time steps)
    use_tbptt: bool = False
    tbptt_steps: int | None = None
    # Optional stride (step) between chunk starts; defaults to tbptt_steps (no overlap)
    tbptt_stride: int | None = None
    # ---- Cloud execution toggle ----
    # When true, the training entrypoint will invoke the SageMaker integration
    # instead of running locally.
    use_cloud: bool = False

    def __post_init__(self) -> None:
        """Convert path strings to Path objects and ensure nested configs are typed."""
        # Validate autoregressive settings
        if self.autoregressive and self.time_steps_per_token <= 0:
            msg = "time_steps_per_token must be positive when autoregressive=True"
            raise ValueError(
                msg,
            )
        # Handle backward compatibility and sync flags
        if self.ar is not None:
            # If new config is present, sync old flags to match it
            if isinstance(self.ar, dict):
                ar_dict = cast(dict[str, Any], self.ar)
                self.autoregressive = bool(ar_dict.get("enabled", False))
                self.time_steps_per_token = int(ar_dict.get("time_steps_per_token", 1))
                self.autoregressive_start_timestep = int(ar_dict.get("start_timestep", 0))
                self.autoregressive_mode = str(ar_dict.get("mode", "next_token"))
                # Ensure it's an object
                self.ar = AutoregressiveConfig(**ar_dict)
            else:
                self.autoregressive = self.ar.enabled
                self.time_steps_per_token = self.ar.time_steps_per_token
                self.autoregressive_start_timestep = self.ar.start_timestep
                self.autoregressive_mode = self.ar.mode
        elif self.autoregressive:
            # If old config is used, populate new config
            warnings.warn(
                "Using deprecated autoregressive config format. "
                "Please migrate to new format:\n"
                "  ar:\n"
                "    enabled: true\n"
                "    time_steps_per_token: ...",
                DeprecationWarning, stacklevel=2
            )
            self.ar = AutoregressiveConfig(
                enabled=True,
                time_steps_per_token=self.time_steps_per_token,
                start_timestep=self.autoregressive_start_timestep,
                mode=self.autoregressive_mode
            )

        if self.autoregressive and self.autoregressive_start_timestep < 0:
            msg = "autoregressive_start_timestep must be non-negative"
            raise ValueError(msg)

        if self.autoregressive:
            logger.info(
                f"Autoregressive training enabled: {self.time_steps_per_token} timesteps/token, start at step {self.autoregressive_start_timestep}",
            )
            if self.autoregressive_mode not in {"next_token", "seq2seq"}:
                msg = f"Invalid autoregressive_mode '{self.autoregressive_mode}'. Expected 'next_token' or 'seq2seq'."
                raise ValueError(
                    msg,
                )
            logger.info(f"Autoregressive mode: {self.autoregressive_mode}")

        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)

        # Handle backward compatibility for 'losses' list
        if self.losses is not None and len(self.losses) > 0:
            # If new config is empty, migrate old config
            if not self.loss.losses:
                warnings.warn(
                    "Using deprecated 'losses' list in TrainingConfig. "
                    "Please migrate to 'loss.losses' structure.",
                    DeprecationWarning, stacklevel=2
                )
                # Convert items to LossItemConfig
                migrated_losses = []
                for item in self.losses:
                    if isinstance(item, dict):
                        migrated_losses.append(LossItemConfig(**item))
                    elif isinstance(item, LossItemConfig):
                        migrated_losses.append(item)
                    # Try to handle generic object if possible, or skip
                    elif hasattr(item, "name"):
                        migrated_losses.append(LossItemConfig(
                            name=item.name,
                            weight=getattr(item, "weight", 1.0),
                            params=getattr(item, "params", {})
                        ))
                self.loss.losses = migrated_losses


        # Allow providing the loss config as a dict (e.g., {"losses": [...]})
        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)

        if isinstance(self.checkpoint_every_n_epochs, str):
            try:
                self.checkpoint_every_n_epochs = int(self.checkpoint_every_n_epochs)
            except ValueError:
                self.checkpoint_every_n_epochs = 1

        if isinstance(self.checkpoint_save_top_k, str):
            try:
                self.checkpoint_save_top_k = int(self.checkpoint_save_top_k)
            except ValueError:
                self.checkpoint_save_top_k = 3

        if isinstance(self.checkpoint_save_last, str):
            self.checkpoint_save_last = self.checkpoint_save_last.lower() in {
                "true",
                "1",
                "yes",
            }

        if isinstance(self.save_initial_state, str):
            self.save_initial_state = self.save_initial_state.lower() in {
                "true",
                "1",
                "yes",
            }

        # Normalize optional DataLoader tuning fields
        try:
            if isinstance(self.prefetch_factor, str):
                self.prefetch_factor = int(self.prefetch_factor)
        except Exception:
            self.prefetch_factor = None
        if isinstance(self.persistent_workers, str):
            self.persistent_workers = self.persistent_workers.lower() in {"true", "1", "yes", "on"}

        # Normalize TBPTT settings
        if isinstance(self.use_tbptt, str):
            self.use_tbptt = self.use_tbptt.lower() in {"true", "1", "yes", "on"}
        if isinstance(self.tbptt_steps, str):
            try:
                self.tbptt_steps = int(float(self.tbptt_steps))
            except Exception:
                self.tbptt_steps = None
        if isinstance(self.tbptt_stride, str):
            try:
                self.tbptt_stride = int(float(self.tbptt_stride))
            except Exception:
                self.tbptt_stride = None
        if self.use_tbptt:
            if not self.tbptt_steps or int(self.tbptt_steps) <= 0:
                msg = "use_tbptt=True requires a positive integer tbptt_steps in training config"
                raise ValueError(msg)
            # Default stride to steps (non-overlapping) when not specified or invalid
            if self.tbptt_stride is None or int(self.tbptt_stride) <= 0:
                self.tbptt_stride = int(self.tbptt_steps)

        # Normalize cloud toggle if provided as string
        if isinstance(self.use_cloud, str):
            self.use_cloud = self.use_cloud.lower() in {"true", "1", "yes", "on"}

        # Handle distillation configuration
        if self.distillation is not None:
            if isinstance(self.distillation, dict):
                dist_dict = cast(dict[str, Any], self.distillation)
                self.distillation = DistillationConfig(**dist_dict)
            # Validate paradigm is set correctly for distillation
            if self.paradigm != "distillation":
                logger.info(
                    f"Distillation config present but paradigm='{self.paradigm}'. "
                    "Setting paradigm='distillation' and mapping='seq2seq'."
                )
                self.paradigm = "distillation"
                self.mapping = "seq2seq"
        elif self.paradigm == "distillation":
            msg = "paradigm='distillation' requires a 'distillation:' config block with teacher_model_path"
            raise ValueError(msg)

        # (backend moved to ModelConfig)


@dataclass
class ModelConfig:
    """Configuration related to the SOEN model architecture and its components,
    including how to load a base model.
    """

    # Path to a pre-trained .pth SOENModelCore file or a .ckpt Lightning checkpoint.
    base_model_path: str | Path | None = None
    # time_pooling can be a string (e.g., "max") or a dict
    # (e.g., {"name": "max", "params": {"scale": 100.0}})
    time_pooling: str | dict[str, Any] = "max"
    # Tracks whether the user explicitly provided time_pooling in the YAML.
    # Used to avoid warnings when only the default is present.
    user_set_time_pooling: bool = False
    # For "mean_range": specific start timestep for averaging.
    range_start: int | None = None
    # For "mean_range": specific end timestep for averaging.
    range_end: int | None = None
    # If base_model_path is a .pth SOEN file, should we load its exact state
    # or just its configuration to rebuild it (potentially with modifications)?
    load_exact_model_state: bool = False
    # Optional override for the simulation timestep (dt) of the SOEN model.
    # If None, the dt from the loaded model/config is used.
    dt: float | None = None
    dt_learnable: bool = False

    # New: build from YAML architecture when base_model_path is not provided
    architecture_yaml: str | Path | None = None
    # Alternatively allow inline dict for programmatic runners
    architecture: dict[str, Any] | None = None

    # Backend selection for training engine ('torch' | 'jax')
    backend: str = "torch"

    # --- ADDED __post_init__ for parsing ---
    def __post_init__(self) -> None:
        """Convert path string to Path object and parse time_pooling."""
        if self.base_model_path is not None and isinstance(self.base_model_path, str):
            self.base_model_path = Path(self.base_model_path)
        if self.architecture_yaml is not None and isinstance(self.architecture_yaml, str):
            self.architecture_yaml = Path(self.architecture_yaml)

        # Store parsed method name and params for easier access later
        # These attributes are not part of the dataclass fields but are convenient instance attributes.
        if isinstance(self.time_pooling, dict):
            self._method_name = self.time_pooling.get("name", "max")
            self._method_params = self.time_pooling.get("params", {})
        else:  # It's a string
            self._method_name = self.time_pooling
            self._method_params = {}

        # Ensure scale defaults to 1.0 if not provided
        if "scale" not in self._method_params:
            self._method_params["scale"] = 1.0

        if isinstance(self.dt_learnable, str):
            self.dt_learnable = self.dt_learnable.lower() in {"true", "1", "yes"}

        # Normalize backend selection (model.backend)
        try:
            b = getattr(self, "backend", "torch")
        except Exception:
            b = "torch"
        if not hasattr(self, "backend"):
            # add attribute dynamically if missing
            self.backend = "torch"
        try:
            if isinstance(b, str):
                b_low = b.strip().lower()
                if b_low in {"torch", "pytorch", "pl", "lightning", ""}:
                    self.backend = "torch"
                elif b_low in {"jax"}:
                    self.backend = "jax"
                else:
                    self.backend = "torch"
        except Exception:
            self.backend = "torch"

    # --- END ADDITION ---

    @property
    def parsed_time_pooling_name(self) -> str:
        """Returns the name of the time pooling method."""
        if hasattr(self, "_method_name"):
            return self._method_name
        # Fallback if __post_init__ wasn't called (e.g. direct instantiation without from_dict)
        # Ensure this fallback logic is robust, especially if time_pooling can be a string
        if isinstance(self.time_pooling, dict):
            return self.time_pooling.get("name", "max")
        return self.time_pooling  # It's a string

    @property
    def parsed_time_pooling_params(self) -> dict[str, Any]:
        """Returns the parameters for the time pooling method."""
        if hasattr(self, "_method_params"):
            return self._method_params
        # Fallback
        if isinstance(self.time_pooling, dict):
            params = self.time_pooling.get("params", {})
            if "scale" not in params:
                params["scale"] = 1.0
            return params
        return {"scale": 1.0}

    @property
    def output_scaling_factor(self) -> float:
        """Returns the scaling factor from the time pooling parameters."""
        return self.parsed_time_pooling_params.get("scale", 1.0)


# --- CONFIGS FOR PROBING (LOGGING SPECIFIC MODEL INTERNALS) ---
@dataclass
class ConnectionParamProbingConfig:
    """Configuration for logging statistics about connection parameters (weights)."""

    log_histograms: bool = False  # Log histograms of connection weights at validation end.
    histogram_bins: int = 50  # Number of bins for the histograms.
    # Future: log_mean_std: bool = False


@dataclass
class StateTrajectoryLoggingConfig:
    """Configuration for logging per-timestep state trajectories to TensorBoard."""

    active: bool = False
    mode: str = "val"  # "train" or "val"
    layer_id: int | None = None  # None => last/output layer
    num_samples: int = 4
    class_ids: list[int] | None = None  # optional filtering for classification tasks
    max_neurons_per_sample: int = 4
    neuron_indices: list[int] | None = None  # optional explicit neuron indices to plot
    tag_prefix: str = "callbacks/state_trajectories"


@dataclass
class GradientStatsLoggingConfig:
    """Configuration for collecting structured gradient statistics."""

    active: bool = False
    log_every_n_steps: int = 1
    max_steps_per_param: int | None = 200
    summary_only: bool = False
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    output_filename: str = "gradient_stats.json"

    def __post_init__(self) -> None:
        if isinstance(self.log_every_n_steps, str):
            try:
                self.log_every_n_steps = int(self.log_every_n_steps)
            except Exception as exc:
                raise ValueError("log_every_n_steps must be an integer") from exc
        self.log_every_n_steps = max(1, int(self.log_every_n_steps))
        if isinstance(self.max_steps_per_param, str):
            self.max_steps_per_param = int(float(self.max_steps_per_param))
        if isinstance(self.summary_only, str):
            self.summary_only = self.summary_only.lower() in {"true", "1", "yes", "on"}


@dataclass
class LoggingConfig:
    """Settings for logging, now using TensorBoard."""

    project_name: str = "soen_training"  # Logging project name.
    group_name: str = "default_group"  # Optional grouping of runs.
    # Base directory for the entire project (contains both checkpoints and logs)
    project_dir: str | Path | None = None
    # Specific name for this run. If None, ExperimentRunner might generate one.
    experiment_name: str | None = None
    log_freq: int = 50  # How often (in batches) to log metrics.
    # List of metrics to track (beyond loss). See METRICS_REGISTRY for options.
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])
    save_code: bool = True  # Save a snapshot of the code alongside logs?
    log_model: bool = True  # Log model checkpoints via logger.
    log_level: str = "INFO"  # Logging level for file logs.
    console_level: str | None = "WARNING"  # Console logging threshold ("OFF" disables console output).
    log_gradients: bool = False  # Log histograms of gradients? (Can be verbose).
    track_layer_params: bool = False  # Log histograms of layer-specific parameters (excluding connections).
    track_connections: bool = False  # Log histograms of connection weights using TensorBoard native histograms.

    log_batch_metrics: bool = False  # Log metrics computed at each batch step?
    # For advanced users: define custom metrics for the logger.
    custom_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- S3 Upload Options ---
    upload_logs_and_checkpoints: bool = False
    s3_upload_url: str | None = None

    # --- MLflow Options (Optional) ---
    # Default OFF so minimal configs don't accidentally require credentials
    mlflow_active: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None  # Defaults to project_name when None
    mlflow_run_name: str | None = None  # Defaults to repeat dir (e.g., repeat_0)
    mlflow_tags: dict[str, str] = field(default_factory=dict)
    mlflow_log_artifacts: bool = True  # Log checkpoints/sidecars as MLflow artifacts
    # Optional HTTP Basic Auth for MLflow server (used only if env vars are not already set)
    mlflow_username: str | None = "admin"  # Default username for team server (requires matching password when server enforces auth)
    mlflow_password: str | None = None  # Provide alongside mlflow_username or disable MLflow when the server requires credentials

    # --- New Probing Sections ---
    connection_params_probing: ConnectionParamProbingConfig = field(
        default_factory=ConnectionParamProbingConfig,
    )
    state_trajectories: StateTrajectoryLoggingConfig = field(
        default_factory=StateTrajectoryLoggingConfig,
    )
    gradient_stats: GradientStatsLoggingConfig = field(
        default_factory=GradientStatsLoggingConfig,
    )

    def __post_init__(self) -> None:
        """Ensure 'metrics' is a list and nested probing configs are instantiated."""
        if isinstance(self.metrics, str):  # Allow single string for convenience
            self.metrics = [self.metrics]

        if isinstance(self.project_dir, str) and self.project_dir.strip():
            self.project_dir = Path(self.project_dir)
        elif not isinstance(self.project_dir, Path):
            self.project_dir = None

        # Normalise console level strings if provided
        if isinstance(self.console_level, str):
            self.console_level = self.console_level.upper()

        # Instantiate nested probing configurations if they are provided as dictionaries
        # This is crucial for when ExperimentConfig.from_dict constructs LoggingConfig
        if isinstance(self.connection_params_probing, dict):
            probing_dict = cast(dict[str, Any], self.connection_params_probing)
            self.connection_params_probing = ConnectionParamProbingConfig(
                **probing_dict,
            )
        if isinstance(self.state_trajectories, dict):
            traj_dict = cast(dict[str, Any], self.state_trajectories)
            self.state_trajectories = StateTrajectoryLoggingConfig(
                **traj_dict,
            )
        if isinstance(self.gradient_stats, dict):
            stats_dict = cast(dict[str, Any], self.gradient_stats)
            self.gradient_stats = GradientStatsLoggingConfig(
                **stats_dict,
            )

        # Accept truthy strings for track_layer_params
        if isinstance(self.track_layer_params, str):
            self.track_layer_params = self.track_layer_params.lower() in {
                "true",
                "1",
                "yes",
                "on",
            }

        # "loss" is typically always logged by LightningModule,
        # but good to ensure it's not missing if user provides an empty list.
        # However, we explicitly log 'train_loss/total', 'val_loss', etc.
        # So, explicit inclusion here might be for user's reference or specific handling in MetricsTracker.
        # For now, let's assume the explicit logs like 'val_loss' are sufficient.

        # Normalize MLflow flags provided as strings
        if isinstance(self.mlflow_active, str):
            self.mlflow_active = self.mlflow_active.lower() in {"true", "1", "yes", "on"}
        if isinstance(self.mlflow_log_artifacts, str):
            self.mlflow_log_artifacts = self.mlflow_log_artifacts.lower() in {"true", "1", "yes", "on"}

        # Ensure tags is a dict
        if self.mlflow_tags is None:
            self.mlflow_tags = {}


# -----------------------------------------------------------------------------
# The All-Encompassing Experiment Configuration
# -----------------------------------------------------------------------------


@dataclass
class CloudConfig:
    """Cloud execution settings for training.

    Place these under a top-level `cloud:` block in the YAML.
    """

    active: bool = False  # When true, run via cloud integration instead of locally
    role: str | None = None  # Required
    # Location and storage
    region: str | None = None
    bucket: str | None = None
    # Optional compute overrides (defaults are chosen from training.accelerator)
    instance_type: str | None = None
    cpu_instance_type: str | None = None
    gpu_instance_type: str | None = None
    instance_count: int = 1  # Single-node by default
    # Optional SageMaker training image/runtime details
    framework_version: str = "2.2.0"
    py_version: str = "py310"
    # Optional job runtime controls
    use_spot: bool = True  # Use spot instances (cheaper but can be interrupted)
    max_run: int = 36000  # Max runtime in seconds (legacy)
    max_wait: int = 36000
    max_runtime_hours: float | None = None  # Max runtime in hours (preferred)
    wait: bool = True
    verbose: bool = False
    job_prefix: str | None = None
    cleanup: bool = True
    # Optional naming (also derivable from logging.*)
    project: str | None = None
    experiment: str | None = None
    # Docker images (new cloud system)
    docker_image_pytorch: str | None = None
    docker_image_jax: str | None = None
    # MLflow settings for cloud
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.active, str):
            self.active = self.active.lower() in {"true", "1", "yes", "on"}
        if isinstance(self.instance_count, str):
            try:
                self.instance_count = int(float(self.instance_count))
            except Exception:
                self.instance_count = 1
        # Normalize optional booleans
        for attr in ["wait", "verbose", "cleanup", "use_spot"]:
            val = getattr(self, attr, None)
            if isinstance(val, str):
                setattr(self, attr, val.lower() in {"true", "1", "yes", "on"})
        # Normalize optional integers
        for attr in ["max_run", "max_wait"]:
            val = getattr(self, attr, None)
            if isinstance(val, str):
                with contextlib.suppress(Exception):
                    setattr(self, attr, int(float(val)))
        # Convert max_runtime_hours to max_run if provided
        if self.max_runtime_hours is not None:
            self.max_run = int(self.max_runtime_hours * 3600)
            self.max_wait = max(self.max_wait, self.max_run)


@dataclass
class ExperimentConfig:
    """The master configuration for a single experiment.
    It brings together all other configuration components (training, data, model, etc.).
    Think of this as the main recipe for your experiment.
    """

    name: str | None = None  # An optional, human-readable name for this specific experiment setup.
    # Not used for critical pathing by ExperimentRunner anymore.
    description: str = ""  # A brief description of what this experiment is trying to achieve.
    seed: int = 42  # The base random seed for reproducibility.

    # ---- Nested Configuration Sections ----
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)

    # --- START REFACTOR: General Callbacks Section ---
    # Callbacks are plugins for custom logic. The LR scheduler is now a callback.
    callbacks: dict[str, Any] = field(
        default_factory=lambda: {"lr_scheduler": {"type": "constant", "lr": 0.001}},
    )
    # --- END REFACTOR ---

    # An advanced feature: a function to programmatically modify model configurations
    # (SimulationConfig, LayerConfig, ConnectionConfig) before model creation.
    # This function is not directly serializable to YAML.
    model_modifier_fn: Callable[..., Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Perform any final validations or adjustments after all fields are initialized.
        For example, ensuring consistency between different config parts.
        This method also handles instantiation of nested dataclasses if the main config
        is created from a dictionary (e.g., loaded from YAML).
        """
        # Ensure nested configurations that might have been loaded as dicts are actual dataclass instances.
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(getattr(self, "profiler", None), dict):
            self.profiler = ProfilerConfig(**self.profiler)  # type: ignore[arg-type]
        if isinstance(getattr(self, "cloud", None), dict):
            self.cloud = CloudConfig(**self.cloud)  # type: ignore[arg-type]

        # Note: The logic to parse the `callbacks` dict and instantiate the actual
        # callback objects (including the LR scheduler) will now reside in the
        # training script (e.g., `run_trial.py`) that consumes this config.
        # This keeps the config class clean and focused on data structure.

        # Distillation defaults to no time pooling unless the user explicitly set it.
        if getattr(self.training, "paradigm", "supervised") == "distillation" and not getattr(
            self.model,
            "user_set_time_pooling",
            False,
        ):
            self.model.time_pooling = {"name": "none", "params": {"scale": 1.0}}
            # Keep parsed helpers in sync when we override time_pooling here.
            self.model._method_name = "none"
            self.model._method_params = {"scale": 1.0}

    def to_dict(self, serializable: bool = False) -> dict[str, Any]:
        """Converts the ExperimentConfig instance to a dictionary, suitable for saving to YAML.

        Args:
            serializable (bool): If True, attempts to convert non-YAML-friendly types
                                 (like Path objects or Callables) to string representations
                                 or remove them.

        Returns:
            Dict[str, Any]: A dictionary representation of the configuration.

        """
        from dataclasses import asdict  # Local import

        # Using asdict for recursive conversion
        config_dict = asdict(self)

        if serializable:
            # Remove fields that are not easily YAML-serializable or meant for runtime only
            if "model_modifier_fn" in config_dict:
                del config_dict["model_modifier_fn"]  # Callable is not serializable

            # Convert Path objects to strings
            if config_dict.get("logging"):
                if isinstance(config_dict["logging"].get("project_dir"), Path):
                    config_dict["logging"]["project_dir"] = str(
                        config_dict["logging"]["project_dir"],
                    )

            if config_dict.get("data"):
                if isinstance(config_dict["data"].get("data_path"), Path):
                    config_dict["data"]["data_path"] = str(
                        config_dict["data"]["data_path"],
                    )

            if config_dict.get("model"):
                if isinstance(config_dict["model"].get("base_model_path"), Path):
                    config_dict["model"]["base_model_path"] = str(
                        config_dict["model"]["base_model_path"],
                    )

            # No special handling needed for profiler

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ExperimentConfig":
        """Creates an ExperimentConfig instance from a dictionary (e.g., loaded from YAML).
        This method handles the reconstruction of nested dataclass objects.
        """
        training_dict = config_dict.get("training", {}) or {}
        logging_dict = config_dict.get("logging", {}) or {}

        # Backwards compatibility: allow project_dir under training
        for legacy_key in ["project_dir"]:
            if legacy_key in training_dict and legacy_key not in logging_dict:
                logging_dict[legacy_key] = training_dict.pop(legacy_key)

        # --- START: Accept flat training.losses without a nested 'loss' block ---
        # If user specifies:
        # training:
        #   losses: [ {name: ...}, ... ]
        # map it to the unified LossConfig structure: training.loss = { losses: [...] }
        if "losses" in training_dict and "loss" not in training_dict:
            training_dict["loss"] = {"losses": training_dict.pop("losses")}
        # Also allow training.loss being provided directly as a list
        if isinstance(training_dict.get("loss"), list):
            training_dict["loss"] = {"losses": training_dict["loss"]}
        # --- END: Accept flat training.losses ---

        # --- START REFACTOR: Handle removal of top-level scheduler ---
        # Backwards compatibility: if a top-level `scheduler` key exists,
        # move it into the `callbacks` dictionary.
        callbacks_dict = config_dict.get("callbacks", {})
        if "scheduler" in config_dict and "lr_scheduler" not in callbacks_dict:
            logger.warning(
                "Found top-level 'scheduler' key. Moving it to 'callbacks.lr_scheduler' for compatibility.",
            )
            callbacks_dict["lr_scheduler"] = config_dict["scheduler"]
        # --- END REFACTOR ---

        # Capture model dict to detect explicitly set fields
        model_dict = config_dict.get("model", {}) or {}
        user_set_time_pooling = "time_pooling" in model_dict

        return cls(
            name=config_dict.get("name"),
            description=config_dict.get("description", ""),
            seed=config_dict.get("seed", 42),
            training=TrainingConfig(**training_dict),
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**{**model_dict, "user_set_time_pooling": user_set_time_pooling}),
            logging=LoggingConfig(**logging_dict),
            callbacks=callbacks_dict,
            profiler=ProfilerConfig(**(config_dict.get("profiler", {}) or {})),
            cloud=CloudConfig(**(config_dict.get("cloud", {}) or {})),
        )
