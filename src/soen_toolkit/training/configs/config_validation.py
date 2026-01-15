"""Configuration validation and auto-detection for training paradigms and mappings.
Provides comprehensive validation with helpful error messages and suggestions.
"""

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config_classes import ExperimentConfig

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""


class ConfigValidator:
    """Validates training configurations and provides helpful suggestions."""

    # Valid paradigm and mapping combinations
    VALID_PARADIGMS = {"supervised", "self_supervised", "unsupervised", "distillation"}  # unsupervised for backward compatibility
    VALID_MAPPINGS = {"seq2static", "seq2seq", "static2seq", "static2static"}

    # Loss function compatibility matrix
    CLASSIFICATION_LOSSES = {"cross_entropy", "gap_loss", "top_gap_loss", "rich_margin_loss"}
    REGRESSION_LOSSES = {"mse", "mae", "huber", "smooth_l1", "mse_gradient_cutoff"}
    SELF_SUPERVISED_LOSSES = {"mse", "mae", "autoregressive_loss", "autoregressive_cross_entropy"}

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def validate_all(self) -> tuple[list[str], list[str]]:
        """Perform comprehensive validation of the configuration.

        Returns:
            Tuple of (warnings, errors)

        """
        self.warnings.clear()
        self.errors.clear()

        # Basic paradigm/mapping validation
        self._validate_paradigm_and_mapping()

        # Loss function compatibility
        self._validate_loss_functions()

        # Data-specific validation (if data path exists)
        if hasattr(self.config.data, "data_path") and self.config.data.data_path:
            self._validate_data_compatibility()

        # Time pooling validation
        self._validate_time_pooling()

        # Sequence length validation
        self._validate_sequence_lengths()

        return self.warnings.copy(), self.errors.copy()

    def _validate_paradigm_and_mapping(self) -> None:
        """Validate paradigm and mapping values and combinations."""
        paradigm = getattr(self.config.training, "paradigm", "supervised")
        mapping = getattr(self.config.training, "mapping", "seq2static")

        # Check valid values
        if paradigm not in self.VALID_PARADIGMS:
            self.errors.append(
                f"Invalid paradigm '{paradigm}'. Valid options: {', '.join(sorted(self.VALID_PARADIGMS))}.\n"
                f"  • Use 'supervised' for labeled data\n"
                f"  • Use 'self_supervised' for reconstruction/autoencoding tasks\n"
                f"  • Use 'distillation' for knowledge distillation (student matches teacher outputs)\n"
                f"  • Use 'unsupervised' (deprecated, use 'self_supervised' instead)",
            )

        if mapping not in self.VALID_MAPPINGS:
            self.errors.append(
                f"Invalid mapping '{mapping}'. Valid options: {', '.join(sorted(self.VALID_MAPPINGS))}.\n"
                f"  • Use 'seq2static' for sequence → single output (classification/regression)\n"
                f"  • Use 'seq2seq' for sequence → sequence output (forecasting/labeling)\n"
                f"  • Use 'static2seq' for single input → sequence output\n"
                f"  • Use 'static2static' for single input → single output",
            )

        # Backward compatibility warning
        if paradigm == "unsupervised":
            self.warnings.append(
                "paradigm='unsupervised' is deprecated. Use 'self_supervised' instead.\n  Both work identically, but 'self_supervised' is more accurate terminology.",
            )

        # Distillation-specific validation
        if paradigm == "distillation":
            distillation_cfg = getattr(self.config.training, "distillation", None)
            if distillation_cfg is None:
                self.errors.append(
                    "paradigm='distillation' requires a 'distillation:' config block.\n"
                    "  Example:\n"
                    "    distillation:\n"
                    "      teacher_model_path: 'path/to/teacher.pth'\n"
                    "      subset_fraction: 1.0",
                )
            else:
                teacher_path = getattr(distillation_cfg, "teacher_model_path", None)
                data_path = getattr(distillation_cfg, "distillation_data_path", None)

                # Must have either teacher model OR pre-generated data
                if not teacher_path and not data_path:
                    self.errors.append(
                        "distillation requires either:\n"
                        "  - teacher_model_path: path to teacher model (.pth or .soen)\n"
                        "  - distillation_data_path: path to pre-generated dataset (.h5)",
                    )

                # Validate paths exist if provided
                if teacher_path and not Path(teacher_path).exists():
                    self.errors.append(
                        f"Teacher model not found: {teacher_path}\n"
                        "  Ensure the teacher model file exists.",
                    )
                if data_path and not Path(data_path).exists():
                    self.errors.append(
                        f"Distillation data not found: {data_path}\n"
                        "  Ensure the distillation dataset file exists.",
                    )

        # Validate paradigm-mapping combinations
        if paradigm in ["unsupervised", "self_supervised"]:
            if mapping not in ["seq2seq", "seq2static"]:
                self.errors.append(
                    f"Self-supervised learning (paradigm='{paradigm}') only supports "
                    f"seq2seq or seq2static mappings, not '{mapping}'.\n"
                    f"  • Use 'seq2seq' for autoencoder/reconstruction tasks\n"
                    f"  • Use 'seq2static' for learning sequence summaries",
                )

    def _validate_loss_functions(self) -> None:
        """Validate loss function compatibility with paradigm and task type."""
        paradigm = getattr(self.config.training, "paradigm", "supervised")
        getattr(self.config.training, "mapping", "seq2static")

        if not hasattr(self.config.training, "loss") or not hasattr(self.config.training.loss, "losses"):
            self.errors.append("No loss functions specified in training.loss.losses")
            return

        loss_names = {loss.name for loss in self.config.training.loss.losses}

        # Check for self-supervised tasks
        if paradigm in ["unsupervised", "self_supervised"]:
            invalid_losses = loss_names - self.SELF_SUPERVISED_LOSSES - self.REGRESSION_LOSSES
            if invalid_losses:
                self.warnings.append(
                    f"Self-supervised tasks typically use reconstruction losses, but found: {', '.join(invalid_losses)}.\n"
                    f"  Recommended losses: {', '.join(sorted(self.SELF_SUPERVISED_LOSSES))}\n"
                    f"  • Use 'mse' for most reconstruction tasks\n"
                    f"  • Use 'mae' for more robust reconstruction",
                )

        # Check for classification vs regression losses
        has_classification_losses = bool(loss_names & self.CLASSIFICATION_LOSSES)
        has_regression_losses = bool(loss_names & self.REGRESSION_LOSSES)

        if has_classification_losses and has_regression_losses:
            self.warnings.append(
                "Mixing classification and regression losses. This is unusual but may be intentional.\n"
                f"  Classification losses: {', '.join(loss_names & self.CLASSIFICATION_LOSSES)}\n"
                f"  Regression losses: {', '.join(loss_names & self.REGRESSION_LOSSES)}",
            )

        # Check num_classes for classification
        if has_classification_losses:
            num_classes = getattr(self.config.data, "num_classes", None)
            if not num_classes or num_classes <= 1:
                self.errors.append(
                    f"Classification losses require data.num_classes > 1.\n  Current num_classes: {num_classes}\n  Set data.num_classes to the number of classes in your dataset",
                )

    def _validate_data_compatibility(self) -> None:
        """Validate data file compatibility with paradigm and mapping."""
        data_path = Path(self.config.data.data_path)

        if not data_path.exists():
            self.errors.append(f"Data file not found: {data_path}")
            return

        if data_path.suffix.lower() not in [".h5", ".hdf5"]:
            self.warnings.append(
                f"Data file '{data_path}' is not HDF5 format. Only HDF5 validation is currently supported.",
            )
            return

        try:
            self._validate_hdf5_data(data_path)
        except Exception as e:
            self.warnings.append(f"Could not validate data file '{data_path}': {e}")

    def _validate_hdf5_data(self, data_path: Path) -> None:
        """Validate HDF5 data structure and suggest corrections."""
        paradigm = getattr(self.config.training, "paradigm", "supervised")
        mapping = getattr(self.config.training, "mapping", "seq2static")
        from soen_toolkit.training.data.dataloaders import open_hdf5_with_consistent_locking

        with open_hdf5_with_consistent_locking(str(data_path)) as f:
            # Determine data structure (split vs single dataset)
            if "train" in f:
                data_shape = f["train/data"].shape
                has_labels = "labels" in f["train"]
                label_shape = f["train/labels"].shape if has_labels else None
            else:
                if "data" not in f:
                    self.errors.append(
                        "HDF5 file missing 'data' dataset. Expected structure:\n"
                        "  Option 1: root/data, root/labels\n"
                        "  Option 2: root/train/{data,labels}, root/val/{data,labels}, root/test/{data,labels}",
                    )
                    return

                data_shape = f["data"].shape
                has_labels = "labels" in f
                label_shape = f["labels"].shape if has_labels else None

            # Validate data shape
            if len(data_shape) != 3:
                self.errors.append(
                    f"Data shape {data_shape} is not valid. Expected [N, T, D] format:\n  • N: number of samples\n  • T: sequence length\n  • D: feature dimension",
                )
                return

            _N, T, _D = data_shape

            # Validate labels based on paradigm and mapping
            if paradigm == "supervised":
                if not has_labels:
                    self.errors.append(
                        "Supervised learning requires labels, but none found in dataset.\n  Add 'labels' dataset to your HDF5 file.",
                    )
                    return

                assert label_shape is not None  # has_labels check guarantees this
                self._validate_label_shapes(label_shape, data_shape, mapping)

            elif paradigm in ["unsupervised", "self_supervised"]:
                if mapping == "seq2static" and not has_labels:
                    self.warnings.append(
                        "Self-supervised seq2static tasks typically need target labels (e.g., sequence statistics to predict).\n  If intentional, the model will use dummy targets.",
                    )

            # Validate sequence length compatibility
            # Skip for distillation since it generates its own dataset with teacher outputs
            target_seq_len = getattr(self.config.data, "target_seq_len", None)
            if target_seq_len and target_seq_len != T and paradigm != "distillation":
                if mapping == "seq2seq":
                    self.warnings.append(
                        f"Sequence length mismatch: data has T={T}, but target_seq_len={target_seq_len}.\n"
                        f"  For seq2seq tasks, these should match to avoid shape issues.\n"
                        f"  • Set data.target_seq_len={T}, or\n"
                        f"  • Ensure your labels have length {target_seq_len}",
                    )
                else:
                    self.warnings.append(f"Data will be resampled from T={T} to target_seq_len={target_seq_len}.\n")

    def _validate_label_shapes(self, label_shape: tuple[int, ...], data_shape: tuple[int, ...], mapping: str) -> None:
        """Validate label shapes against data shape and mapping."""
        N, T, _D = data_shape

        # Check if autoregressive mode is enabled
        is_autoregressive = getattr(self.config.training, "autoregressive", False)

        # For autoregressive training, labels are ignored (targets derived from inputs)
        # So we can be lenient with label shapes
        if is_autoregressive:
            logger.info(
                "Autoregressive mode detected. Label shapes will not be strictly validated "
                "since targets are auto-generated from input sequences."
            )
            # Still check that we have the right loss function
            loss_names = {loss.name for loss in self.config.training.loss.losses}
            if "autoregressive_cross_entropy" not in loss_names and "autoregressive_loss" not in loss_names:
                self.warnings.append(
                    "Autoregressive mode is enabled but no autoregressive loss function found.\n"
                    "  Recommended: use 'autoregressive_cross_entropy' for character/token prediction\n"
                    f"  Current losses: {', '.join(loss_names) if loss_names else 'none'}"
                )
            return  # Skip label shape validation for AR mode

        if mapping == "seq2static":
            # Expect [N] for classification or [N, K] for regression
            if len(label_shape) == 1 and label_shape[0] == N:
                # Classification: [N]
                self._suggest_loss_function("classification")
            elif len(label_shape) == 2 and label_shape[0] == N:
                # Regression: [N, K]
                self._suggest_loss_function("regression")
            else:
                self.errors.append(
                    f"seq2static labels shape {label_shape} doesn't match data.\n"
                    f"  Expected shapes:\n"
                    f"  • Classification: [{N}] (class indices)\n"
                    f"  • Regression: [{N}, K] (K target values per sample)\n"
                    f"  Current data shape: {data_shape}",
                )

        elif mapping == "seq2seq":
            if len(label_shape) == 2 and label_shape == (N, T):
                # Sequence classification: [N, T]
                logger.info(
                    "Detected seq2seq classification (token-level translation). This is aligned classification where each timestep has a label. Different from autoregressive next-token prediction."
                )
                self._suggest_loss_function("seq2seq_classification")
            elif len(label_shape) == 3 and label_shape[:2] == (N, T):
                # Sequence regression: [N, T, K]
                self._suggest_loss_function("seq2seq_regression")
            else:
                self.errors.append(
                    f"seq2seq labels shape {label_shape} doesn't match data.\n"
                    f"  Expected shapes:\n"
                    f"  • Sequence classification: [{N}, {T}] (class per timestep)\n"
                    f"  • Sequence regression: [{N}, {T}, K] (K values per timestep)\n"
                    f"  Current data shape: {data_shape}\n"
                    f"  \n"
                    f"  Tip: For autoregressive training (next-token prediction), set:\n"
                    f"     training.autoregressive: true\n"
                    f"     This will skip label shape validation since targets are auto-generated.",
                )

    def _suggest_loss_function(self, task_type: str) -> None:
        """Suggest appropriate loss functions based on detected task type."""
        current_losses = {loss.name for loss in self.config.training.loss.losses}

        suggestions = {
            "classification": {"cross_entropy"},
            "regression": {"mse"},
            "seq2seq_classification": {"cross_entropy"},
            "seq2seq_regression": {"mse"},
        }

        recommended = suggestions.get(task_type, set())
        if not (current_losses & recommended):
            task_description = {
                "seq2seq_classification": "seq2seq classification (token-level translation)",
                "seq2seq_regression": "seq2seq regression",
                "classification": "classification",
                "regression": "regression",
            }.get(task_type, task_type)

            msg = f"Detected {task_description} task but no typical loss functions found.\n"
            if task_type == "seq2seq_classification":
                msg += "  This is aligned translation where each timestep has a label (not autoregressive).\n  Recommended metrics: accuracy, perplexity, bits_per_character\n"
            msg += f"  Recommended loss: {', '.join(recommended)}\n"
            msg += f"  Current losses: {', '.join(current_losses) if current_losses else 'none'}"

            self.warnings.append(msg)

    def _validate_time_pooling(self) -> None:
        """Validate time pooling configuration."""
        paradigm = getattr(self.config.training, "paradigm", "supervised")
        mapping = getattr(self.config.training, "mapping", "seq2static")

        has_time_pooling = hasattr(self.config.model, "time_pooling") and self.config.model.time_pooling
        bool(getattr(self.config.model, "user_set_time_pooling", False))

        # Get actual pooling method name
        tp = getattr(self.config.model, "time_pooling", None)
        pooling_method = None
        if isinstance(tp, dict):
            pooling_method = tp.get("name")
        elif isinstance(tp, str):
            pooling_method = tp

        # For distillation, time_pooling should be 'none' (or the default behavior handles it)
        # Don't warn about time_pooling for distillation paradigm
        if paradigm == "distillation":
            return

        # Warn for seq2seq if time_pooling is present (it is ignored for seq2seq)
        # Tests expect this warning regardless of whether it came from defaults or user input.
        if mapping == "seq2seq" and has_time_pooling and pooling_method != "none":
            self.warnings.append(
                "time_pooling is specified but will be ignored for seq2seq tasks.\n"
                "  seq2seq tasks use sequence outputs directly, not pooled outputs.\n"
                "  For seq2seq classification: model outputs [B, T, num_classes] for per-timestep classification.\n"
                "  For seq2seq regression: model outputs [B, T, output_dim] for per-timestep prediction.",
            )

        if mapping == "seq2static" and not has_time_pooling:
            self.warnings.append(
                "No time_pooling specified for seq2static task.\n  Recommended: set model.time_pooling (e.g., 'final', 'mean', 'max')\n  Default 'final' will be used if not specified.",
            )

        if paradigm in ["unsupervised", "self_supervised"] and mapping == "seq2seq" and has_time_pooling:
            self.warnings.append(
                "time_pooling specified for self-supervised seq2seq task but will be ignored.\n  Self-supervised seq2seq uses sequence outputs for reconstruction.",
            )

    def _validate_sequence_lengths(self) -> None:
        """Validate sequence length configurations."""
        target_seq_len = getattr(self.config.data, "target_seq_len", None)
        mapping = getattr(self.config.training, "mapping", "seq2static")

        # Only warn about missing target_seq_len for mappings where resampling is commonly desired.
        # For seq2seq tasks, using the dataset T is expected and should not warn.
        if not target_seq_len and mapping != "seq2seq":
            self.warnings.append(
                "No target_seq_len specified. Data will be used at original sequence length.\n  Set data.target_seq_len to control input sequence length.",
            )

        # Check for TBPTT settings
        use_tbptt = getattr(self.config.training, "use_tbptt", False)
        tbptt_steps = getattr(self.config.training, "tbptt_steps", None)

        if use_tbptt:
            if not tbptt_steps:
                self.errors.append(
                    "use_tbptt=True but tbptt_steps not specified.\n  Set training.tbptt_steps to chunk size for truncated backprop.",
                )
            elif target_seq_len and tbptt_steps >= target_seq_len:
                self.warnings.append(
                    f"tbptt_steps ({tbptt_steps}) >= target_seq_len ({target_seq_len}).\n  TBPTT is most useful when chunk size < sequence length.",
                )


def validate_config(config: ExperimentConfig, raise_on_error: bool = True) -> tuple[list[str], list[str]]:
    """Validate a training configuration and return warnings and errors.

    Args:
        config: The configuration to validate
        raise_on_error: Whether to raise ConfigValidationError if errors found

    Returns:
        Tuple of (warnings, errors)

    Raises:
        ConfigValidationError: If errors found and raise_on_error=True

    """
    validator = ConfigValidator(config)
    warnings, errors = validator.validate_all()

    # Log warnings
    for warning in warnings:
        logger.warning(f"Config validation warning:\n{warning}")

    # Handle errors
    if errors:
        error_msg = "Configuration validation failed:\n\n" + "\n\n".join(f"[ERROR] {error}" for error in errors)
        if warnings:
            error_msg += "\n\nWarnings:\n\n" + "\n\n".join(f"[WARNING] {warning}" for warning in warnings)

        if raise_on_error:
            raise ConfigValidationError(error_msg)
        logger.error(error_msg)

    return warnings, errors


def auto_detect_task_type(data_path: str) -> dict[str, Any]:
    """Auto-detect task type and suggest configuration based on data structure.

    Args:
        data_path: Path to HDF5 data file

    Returns:
        Dictionary with suggested configuration values

    """
    suggestions: dict[str, Any] = {
        "paradigm": None,
        "mapping": None,
        "losses": [],
        "num_classes": None,
        "confidence": "low",
    }

    if not Path(data_path).exists():
        return suggestions

    try:
        with h5py.File(data_path, "r") as f:
            # Get data and label shapes
            if "train" in f:
                data_shape = f["train/data"].shape
                has_labels = "labels" in f["train"]
                label_shape = f["train/labels"].shape if has_labels else None
                label_dtype = f["train/labels"].dtype if has_labels else None
            else:
                data_shape = f["data"].shape
                has_labels = "labels" in f
                label_shape = f["labels"].shape if has_labels else None
                label_dtype = f["labels"].dtype if has_labels else None

            if len(data_shape) != 3:
                return suggestions

            _N, T, _D = data_shape

            # Auto-detect paradigm
            if not has_labels:
                suggestions["paradigm"] = "self_supervised"
                suggestions["mapping"] = "seq2seq"
                suggestions["losses"] = [{"name": "mse", "weight": 1.0, "params": {}}]
                suggestions["confidence"] = "medium"

            elif has_labels:
                suggestions["paradigm"] = "supervised"

                # Auto-detect mapping and task type from label shape
                if label_shape is None:
                    return suggestions

                if len(label_shape) == 1:
                    # [N] - seq2static classification
                    suggestions["mapping"] = "seq2static"
                    if np.issubdtype(label_dtype, np.integer):
                        suggestions["losses"] = [{"name": "cross_entropy", "weight": 1.0, "params": {}}]
                        suggestions["num_classes"] = int(np.max([f[f"{'train/' if 'train' in f else ''}labels"][...] for _ in [0]])) + 1
                        suggestions["confidence"] = "high"
                    else:
                        suggestions["losses"] = [{"name": "mse", "weight": 1.0, "params": {}}]
                        suggestions["confidence"] = "medium"

                elif len(label_shape) == 2:
                    if label_shape[1] == T:
                        # [N, T] - seq2seq classification
                        suggestions["mapping"] = "seq2seq"
                        suggestions["losses"] = [{"name": "cross_entropy", "weight": 1.0, "params": {}}]
                        if np.issubdtype(label_dtype, np.integer):
                            suggestions["num_classes"] = int(np.max([f[f"{'train/' if 'train' in f else ''}labels"][...] for _ in [0]])) + 1
                        suggestions["confidence"] = "high"
                    else:
                        # [N, K] - seq2static regression
                        suggestions["mapping"] = "seq2static"
                        suggestions["losses"] = [{"name": "mse", "weight": 1.0, "params": {}}]
                        suggestions["confidence"] = "high"

                elif len(label_shape) == 3:
                    # [N, T, K] - seq2seq regression
                    suggestions["mapping"] = "seq2seq"
                    suggestions["losses"] = [{"name": "mse", "weight": 1.0, "params": {}}]
                    suggestions["confidence"] = "high"

    except Exception as e:
        logger.warning(f"Auto-detection failed for {data_path}: {e}")

    return suggestions
