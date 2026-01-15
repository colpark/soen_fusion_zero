"""SOEN Cloud - Cloud infrastructure for training, inference, and processing.

This module provides a clean, robust interface for running SOEN toolkit
workloads on AWS SageMaker with:

- Training jobs with MLflow integration
- Batch inference using trained models
- Data processing jobs
- Cost estimation before submission
- Pre-flight validation of all resources

Quick Start
-----------

CLI usage:

    # Submit a training job
    soen cloud train --config experiment.yaml

    # Show cost estimate first
    soen cloud train --config experiment.yaml --estimate

    # Check job status
    soen cloud status <job-id>

    # List available instances
    soen cloud instances

Python API:

    from soen_toolkit.cloud import CloudSession, TrainingJob
    from soen_toolkit.cloud.config import load_config

    # Load configuration
    config = load_config("cloud_config.yaml")

    # Create session with validation
    session = CloudSession(config)

    # Create and submit job
    job = TrainingJob(job_config)
    job_name = job.submit(session)

Environment Variables
--------------------

    SOEN_SM_ROLE: SageMaker execution role ARN
    SOEN_SM_BUCKET: S3 bucket for artifacts
    AWS_REGION: AWS region (default: us-east-1)
    MLFLOW_TRACKING_URI: MLflow tracking server URI (optional)
"""

# Suppress SyntaxWarnings from AWS packages that have invalid escape sequences
# These are cosmetic issues in sagemaker_core and smdebug_rulesconfig that AWS
# hasn't fixed for Python 3.12+ compatibility
import warnings

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"sagemaker_core\..*",
)
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"smdebug_rulesconfig\..*",
)

from .cli import cli, main  # noqa: E402
from .config import (  # noqa: E402
    AWSConfig,
    Backend,
    CloudConfig,
    DataConfig,
    InstanceConfig,
    JobConfig,
    JobType,
    MLflowConfig,
    load_config,
    load_job_config,
)
from .cost import CostEstimate, CostEstimator  # noqa: E402
from .jobs import (  # noqa: E402
    InferenceJob,
    JobResult,
    JobSpec,
    JobStatus,
    ProcessingJob,
    SubmittedJob,
    TrainingJob,
)
from .session import (  # noqa: E402
    AWSIdentity,
    BucketAccessError,
    CloudSession,
    CloudSessionError,
    CredentialsError,
    ImageValidationError,
    RoleValidationError,
)

__all__ = [
    # CLI
    "cli",
    "main",
    # Config
    "CloudConfig",
    "JobConfig",
    "AWSConfig",
    "InstanceConfig",
    "DataConfig",
    "MLflowConfig",
    "JobType",
    "Backend",
    "load_config",
    "load_job_config",
    # Session
    "CloudSession",
    "CloudSessionError",
    "CredentialsError",
    "BucketAccessError",
    "RoleValidationError",
    "ImageValidationError",
    "AWSIdentity",
    # Cost
    "CostEstimator",
    "CostEstimate",
    # Jobs
    "JobSpec",
    "JobStatus",
    "JobResult",
    "SubmittedJob",
    "TrainingJob",
    "InferenceJob",
    "ProcessingJob",
]

__version__ = "0.1.0"

