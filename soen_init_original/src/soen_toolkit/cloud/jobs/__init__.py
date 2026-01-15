"""Cloud job type definitions.

Provides abstract base class and concrete implementations for different
job types: training, inference, and processing.
"""

from .base import JobResult, JobSpec, JobStatus, SubmittedJob
from .inference import InferenceJob
from .processing import ProcessingJob
from .training import TrainingJob

__all__ = [
    "JobSpec",
    "JobStatus",
    "JobResult",
    "SubmittedJob",
    "TrainingJob",
    "InferenceJob",
    "ProcessingJob",
]

