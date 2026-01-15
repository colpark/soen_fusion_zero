# FILEPATH: soen_toolkit/training/callbacks/__init__.py

"""Callbacks for SOEN model training.

This module provides callbacks for learning rate scheduling, metrics tracking,
and other training utilities.
"""

from .connection_noise import ConnectionNoiseCallback

# Import dynamic loss weight callback
from .dynamic_loss_weight import DynamicLossWeightCallback
from .loss_weight_scheduler import LossWeightScheduler
from .metrics import QuantizedAccuracyCallback
from .metrics_tracker import MetricsTracker
from .noise_annealer import NoiseAnnealingCallback
from .output_state_stats import OutputStateStatsCallback

# Import probing callbacks
from .probing import (
    ConnectionParameterProbeCallback,
)
from .qat import QATStraightThroughCallback
from .scheduler_core import (
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    BaseLRScheduler,
    register_optimizer,
    register_scheduler,
)
from .schedulers import (
    ConstantLRScheduler,
    CosineAnnealingScheduler,
    GreedyScheduler,
    LinearDecayScheduler,
    RexScheduler,
)
from .stateful_training import StatefulTrainingCallback
from .time_pooling_scale_scheduler import TimePoolingScaleScheduler

__all__ = [
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    # Base classes and registry
    "BaseLRScheduler",
    # Connection noise
    "ConnectionNoiseCallback",
    # Probing callbacks
    "ConnectionParameterProbeCallback",
    "ConstantLRScheduler",
    # Scheduler implementations
    "CosineAnnealingScheduler",
    # Dynamic loss weight adjustment
    "DynamicLossWeightCallback",
    "GreedyScheduler",
    "LinearDecayScheduler",
    "LossWeightScheduler",
    # Metrics tracking
    "MetricsTracker",
    # Noise annealing
    "NoiseAnnealingCallback",
    # Output state distributions/stats
    "OutputStateStatsCallback",
    # Quantization-aware training (STE)
    "QATStraightThroughCallback",
    # Quantized accuracy evaluation
    "QuantizedAccuracyCallback",
    "RexScheduler",
    # Stateful training (state carryover across batches)
    "StatefulTrainingCallback",
    # Time pooling scale scheduling
    "TimePoolingScaleScheduler",
    "register_optimizer",
    "register_scheduler",
]
