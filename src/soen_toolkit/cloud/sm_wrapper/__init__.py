"""SageMaker-specific implementations.

Provides wrappers around SageMaker SDK for training, processing,
and batch transform jobs.
"""

from .channels import InputChannel, OutputChannel, prepare_channels
from .estimator import create_estimator, create_training_job

__all__ = [
    "InputChannel",
    "OutputChannel",
    "prepare_channels",
    "create_estimator",
    "create_training_job",
]

