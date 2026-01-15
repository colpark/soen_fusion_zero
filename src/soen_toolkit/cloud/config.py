"""Cloud configuration schemas with strict validation.

Uses Pydantic for fail-fast validation of all cloud job configurations.
All validation errors are raised immediately before any resources are provisioned.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobType(str, Enum):
    """Supported cloud job types."""

    TRAINING = "training"
    INFERENCE = "inference"
    PROCESSING = "processing"


class Backend(str, Enum):
    """ML framework backend."""

    PYTORCH = "pytorch"
    JAX = "jax"


class AWSConfig(BaseModel):
    """AWS-specific configuration with validation."""

    model_config = ConfigDict(frozen=True)

    role: Annotated[str, Field(description="SageMaker execution role ARN")]
    bucket: Annotated[str, Field(description="S3 bucket for artifacts")]
    region: str = "us-east-1"

    @field_validator("role")
    @classmethod
    def validate_role_arn(cls, v: str) -> str:
        """Validate that role is a proper IAM ARN."""
        if not v.startswith("arn:aws:iam::"):
            raise ValueError(
                f"role must be a valid IAM ARN starting with 'arn:aws:iam::'. Got: {v}"
            )
        # Basic ARN format: arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME
        pattern = r"^arn:aws:iam::\d{12}:role/.+$"
        if not re.match(pattern, v):
            raise ValueError(
                f"role ARN format invalid. Expected: arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME. Got: {v}"
            )
        return v

    @field_validator("bucket")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate S3 bucket naming rules."""
        if not v:
            raise ValueError("bucket name cannot be empty")
        # S3 bucket naming rules (simplified)
        if len(v) < 3 or len(v) > 63:
            raise ValueError(f"bucket name must be 3-63 characters. Got: {len(v)}")
        if not re.match(r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$", v):
            raise ValueError(
                f"bucket name must start/end with letter or number, contain only lowercase letters, numbers, hyphens, periods. Got: {v}"
            )
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate AWS region format."""
        valid_regions = {
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
            "eu-west-1",
            "eu-west-2",
            "eu-central-1",
            "ap-northeast-1",
            "ap-southeast-1",
            "ap-southeast-2",
        }
        if v not in valid_regions:
            raise ValueError(f"region '{v}' not in supported regions: {valid_regions}")
        return v


class InstanceConfig(BaseModel):
    """Compute instance configuration."""

    model_config = ConfigDict(frozen=True)

    instance_type: str = "ml.g5.xlarge"
    instance_count: Annotated[int, Field(ge=1, le=32)] = 1
    use_spot: bool = True
    max_runtime_hours: Annotated[float, Field(gt=0, le=72)] = 24.0

    @field_validator("instance_type")
    @classmethod
    def validate_instance_type(cls, v: str) -> str:
        """Validate SageMaker instance type format."""
        if not v.startswith("ml."):
            raise ValueError(
                f"instance_type must start with 'ml.' for SageMaker. Got: {v}"
            )
        # Valid SageMaker instance families
        # GPU: g4, g5, g6, p3, p4, p5
        # CPU: c5, c6i, c7i, m5, m6i, m7i, r5, r6i
        # Inference/Training: inf1, inf2, trn1
        valid_prefixes = (
            # GPU instances
            "ml.g4", "ml.g5", "ml.g6",
            "ml.p3", "ml.p4", "ml.p5",
            # CPU instances (compute-optimized)
            "ml.c5", "ml.c6i", "ml.c7i",
            # CPU instances (general-purpose)
            "ml.m5", "ml.m6i", "ml.m7i",
            # CPU instances (memory-optimized)
            "ml.r5", "ml.r6i", "ml.r7i",
            # Inference/Training accelerators
            "ml.inf1", "ml.inf2", "ml.trn1",
        )
        if not any(v.startswith(p) for p in valid_prefixes):
            raise ValueError(
                f"instance_type '{v}' not recognized. "
                f"GPU: ml.g5.xlarge, ml.p4d.24xlarge. "
                f"CPU: ml.c7i.4xlarge, ml.m7i.xlarge"
            )
        return v


class DataConfig(BaseModel):
    """Data input/output configuration."""

    model_config = ConfigDict(frozen=True)

    # Input data - can be S3 URI or local path (will be uploaded)
    data_path: str | None = None
    # For inference: model artifact location
    model_path: str | None = None
    # Output location (defaults to bucket/outputs/job_name)
    output_path: str | None = None

    @field_validator("data_path", "model_path", "output_path")
    @classmethod
    def validate_path(cls, v: str | None) -> str | None:
        """Validate path format."""
        if v is None:
            return v
        # S3 paths
        if v.startswith("s3://"):
            if len(v) < 10:  # s3://x/y minimum
                raise ValueError(f"Invalid S3 URI: {v}")
            return v
        # Local paths - must exist for inputs
        return v


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    model_config = ConfigDict(frozen=True)

    tracking_uri: str | None = None
    experiment_name: str = "soen-cloud"
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("tracking_uri")
    @classmethod
    def validate_tracking_uri(cls, v: str | None) -> str | None:
        """Validate MLflow tracking URI format."""
        if v is None:
            return v
        if not (v.startswith("http://") or v.startswith("https://") or v.startswith("s3://")):
            raise ValueError(
                f"tracking_uri must be http://, https://, or s3:// URI. Got: {v}"
            )
        return v


class JobConfig(BaseModel):
    """Complete job configuration."""

    model_config = ConfigDict(frozen=True)

    job_type: JobType
    job_name: str | None = None
    backend: Backend = Backend.PYTORCH

    # Nested configs
    aws: AWSConfig
    instance: InstanceConfig = Field(default_factory=InstanceConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    # Docker image - auto-selected based on backend if None
    docker_image: str | None = None

    # Training-specific config path
    training_config_path: Path | None = None

    # Processing-specific script path
    processing_script: Path | None = None

    # Environment variables to inject
    environment: dict[str, str] = Field(default_factory=dict)

    @field_validator("job_name")
    @classmethod
    def validate_job_name(cls, v: str | None) -> str | None:
        """Validate SageMaker job name format."""
        if v is None:
            return v
        # SageMaker job name: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}
        if len(v) > 63:
            raise ValueError(f"job_name must be <= 63 characters. Got: {len(v)}")
        if not re.match(r"^[a-zA-Z0-9](-*[a-zA-Z0-9])*$", v):
            raise ValueError(
                f"job_name must match pattern [a-zA-Z0-9](-*[a-zA-Z0-9])*. Got: {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_job_requirements(self) -> JobConfig:
        """Validate job-type-specific requirements."""
        if self.job_type == JobType.TRAINING and self.training_config_path is None:
            raise ValueError("training jobs require training_config_path")
        if self.job_type == JobType.INFERENCE and self.data.model_path is None:
            raise ValueError("inference jobs require data.model_path")
        if self.job_type == JobType.PROCESSING and self.processing_script is None:
            raise ValueError("processing jobs require processing_script")
        return self


class CloudConfig(BaseModel):
    """Top-level cloud configuration, typically loaded from YAML."""

    model_config = ConfigDict(frozen=True)

    # AWS configuration
    aws: AWSConfig

    # Default instance settings (can be overridden per-job)
    instance: InstanceConfig = Field(default_factory=InstanceConfig)

    # MLflow settings
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    # Project organization
    project: str = "soen-project"
    experiment: str = "default"

    # Docker images (backend -> image URI)
    docker_images: dict[str, str] = Field(default_factory=dict)

    @field_validator("project", "experiment")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        """Validate project/experiment identifiers."""
        if not v:
            raise ValueError("identifier cannot be empty")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                f"identifier must contain only alphanumeric, underscore, hyphen. Got: {v}"
            )
        return v

    def get_docker_image(self, backend: Backend) -> str | None:
        """Get Docker image for a backend."""
        return self.docker_images.get(backend.value)


def load_config(path: Path | str) -> CloudConfig:
    """Load cloud configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Validated CloudConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config validation fails.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Handle 'cloud' section wrapper if present
    if "cloud" in data and isinstance(data["cloud"], dict):
        data = data["cloud"]

    return CloudConfig.model_validate(data)


def load_job_config(path: Path | str) -> JobConfig:
    """Load job configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Validated JobConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config validation fails.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return JobConfig.model_validate(data)

