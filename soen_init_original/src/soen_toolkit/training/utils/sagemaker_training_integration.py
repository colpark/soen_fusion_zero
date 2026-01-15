"""Integration to run training on cloud when cloud.active: true in training YAML.

This module provides seamless cloud training integration. When a training config
has `cloud: active: true`, training is automatically submitted to AWS SageMaker
instead of running locally.

Flow:
1. Detect cloud.active or legacy training.use_cloud in the YAML
2. Build a CloudConfig from YAML settings + environment variables
3. Submit via the new soen_toolkit.cloud system
4. Return True to signal the training entrypoint that cloud handled the job

Configuration in training YAML:
```yaml
cloud:
  active: true
  role: "arn:aws:iam::123456789012:role/SageMakerRole"  # or env SOEN_SM_ROLE
  bucket: "my-bucket"  # or env SOEN_SM_BUCKET
  # Optional overrides:
  instance_type: ml.g5.xlarge
  instance_count: 1
  use_spot: true
  max_runtime_hours: 24.0
```

Required environment variables (if not in YAML):
- SOEN_SM_ROLE: SageMaker execution role ARN
- SOEN_SM_BUCKET: S3 bucket for artifacts
- AWS_REGION: AWS region (default: us-east-1)

Optional environment variables:
- SOEN_DOCKER_PYTORCH: PyTorch Docker image URI
- SOEN_DOCKER_JAX: JAX Docker image URI
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from soen_toolkit.training.configs.config_classes import ExperimentConfig

logger = logging.getLogger(__name__)


def _fail(message: str, hint: str | None = None) -> None:
    """Log a clear error and abort cloud launch."""
    if hint:
        logger.error(f"{message}\nHint: {hint}")
    else:
        logger.error(message)
    raise SystemExit(2)


def _output(message: str = "") -> None:
    """Write a message directly to stdout (avoids print())."""
    sys.stdout.write(f"{message}\n")


def _get_backend_from_config(config: ExperimentConfig) -> str:
    """Determine the ML backend from config."""
    try:
        backend = getattr(getattr(config, "model", None), "backend", None)
        if isinstance(backend, str) and backend.strip():
            return backend.strip().lower()
    except Exception:
        pass
    return "pytorch"  # Default


def _build_cloud_config(
    config_yaml_path: Path,
    config: ExperimentConfig,
) -> dict:
    """Build cloud configuration from training YAML and environment."""
    # Load raw YAML to get cloud section
    with open(config_yaml_path) as f:
        raw_yaml = yaml.safe_load(f) or {}

    cloud_section = raw_yaml.get("cloud", {}) or {}
    logging_section = raw_yaml.get("logging", {}) or {}

    # Required settings: role and bucket
    role = (
        cloud_section.get("role")
        or os.environ.get("SOEN_SM_ROLE")
    )
    bucket = (
        cloud_section.get("bucket")
        or os.environ.get("SOEN_SM_BUCKET")
    )
    region = (
        cloud_section.get("region")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )

    # Validate required fields
    if not role:
        _fail(
            "Missing required cloud setting: role",
            hint="Add 'cloud.role' to your YAML or set SOEN_SM_ROLE environment variable"
        )
    if not bucket:
        _fail(
            "Missing required cloud setting: bucket",
            hint="Add 'cloud.bucket' to your YAML or set SOEN_SM_BUCKET environment variable"
        )

    # Backend detection
    backend = _get_backend_from_config(config)

    # Docker images
    docker_images = {}
    pytorch_image = (
        cloud_section.get("docker_image_pytorch")
        or os.environ.get("SOEN_DOCKER_PYTORCH")
    )
    jax_image = (
        cloud_section.get("docker_image_jax")
        or os.environ.get("SOEN_DOCKER_JAX")
    )
    if pytorch_image:
        docker_images["pytorch"] = pytorch_image
    if jax_image:
        docker_images["jax"] = jax_image

    # Instance configuration
    instance_type = cloud_section.get("instance_type")
    if not instance_type:
        # Auto-select based on accelerator/backend
        accelerator = getattr(getattr(config, "training", None), "accelerator", "gpu")
        if accelerator == "cpu":
            instance_type = cloud_section.get("cpu_instance_type") or "ml.c7i.4xlarge"
        else:
            instance_type = cloud_section.get("gpu_instance_type") or "ml.g5.xlarge"

    instance_count = int(cloud_section.get("instance_count", 1))
    use_spot = cloud_section.get("use_spot", True)
    max_runtime_hours = float(cloud_section.get("max_runtime_hours", 24.0))

    # Project/experiment names
    project = (
        cloud_section.get("project")
        or logging_section.get("project_name")
        or "soen-training"
    )
    experiment = (
        cloud_section.get("experiment")
        or logging_section.get("experiment_name")
        or "default"
    )

    # MLflow settings
    mlflow_uri = (
        cloud_section.get("mlflow_tracking_uri")
        or raw_yaml.get("mlflow_tracking_uri")
        or logging_section.get("mlflow_tracking_uri")
        or os.environ.get("MLFLOW_TRACKING_URI")
    )
    mlflow_experiment = (
        cloud_section.get("mlflow_experiment")
        or experiment
    )

    # Build mlflow config only if URI is provided
    mlflow_config = {}
    if mlflow_uri:
        mlflow_config = {
            "tracking_uri": mlflow_uri,
            "experiment_name": mlflow_experiment,
        }

    return {
        "aws": {
            "role": role,
            "bucket": bucket,
            "region": region,
        },
        "instance": {
            "instance_type": instance_type,
            "instance_count": instance_count,
            "use_spot": use_spot,
            "max_runtime_hours": max_runtime_hours,
        },
        "mlflow": mlflow_config,
        "project": project,
        "experiment": experiment,
        "docker_images": docker_images,
        "backend": backend,
    }


def maybe_launch_on_sagemaker(
    config_yaml_path: Path,
    config: ExperimentConfig,
    *,
    script_dir: Path,
) -> bool:
    """If cloud.active is true, submit to cloud and return True.

    Returns False if cloud not requested, so normal local training should proceed.

    Args:
        config_yaml_path: Path to the training configuration YAML
        config: Parsed ExperimentConfig object
        script_dir: Project root directory

    Returns:
        True if job was submitted to cloud, False if local training should proceed
    """
    # Check if cloud is active
    cloud_active = False
    try:
        cloud_active = bool(getattr(getattr(config, "cloud", None), "active", False))
    except Exception:
        pass

    # Legacy support: training.use_cloud
    legacy_toggle = False
    try:
        legacy_toggle = bool(getattr(config.training, "use_cloud", False))
    except Exception:
        pass

    if not (cloud_active or legacy_toggle):
        return False

    logger.info("Cloud execution active. Submitting to AWS SageMaker...")

    # Import cloud components (lazy to avoid circular imports)
    try:
        from soen_toolkit.cloud import CloudSession, TrainingJob
        from soen_toolkit.cloud.config import Backend, CloudConfig, JobConfig, JobType
        from soen_toolkit.cloud.cost import CostEstimator
    except ImportError as e:
        _fail(
            f"Failed to import cloud module: {e}",
            hint="Ensure soen_toolkit.cloud is installed and sagemaker package is available"
        )

    # Build cloud configuration
    cloud_dict = _build_cloud_config(config_yaml_path, config)

    # Check for Docker images
    backend = cloud_dict["backend"]
    docker_images = cloud_dict.get("docker_images", {})

    if backend not in docker_images:
        # Try to construct from account ID
        try:
            import boto3
            sts = boto3.client("sts", region_name=cloud_dict["aws"]["region"])
            account_id = sts.get_caller_identity()["Account"]
            region = cloud_dict["aws"]["region"]

            if backend == "jax":
                default_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/soen-toolkit:v2-jax"
            else:
                default_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/soen-toolkit:v2-pytorch"

            docker_images[backend] = default_image
            logger.info(f"Using default Docker image: {default_image}")
        except Exception:
            _fail(
                f"No Docker image configured for backend '{backend}'",
                hint=f"Set SOEN_DOCKER_{backend.upper()} or add 'cloud.docker_image_{backend}' to your YAML"
            )

    cloud_dict["docker_images"] = docker_images

    # Create CloudConfig
    try:
        config_kwargs = {
            "aws": cloud_dict["aws"],
            "instance": cloud_dict["instance"],
            "project": cloud_dict["project"],
            "experiment": cloud_dict["experiment"],
            "docker_images": cloud_dict["docker_images"],
        }
        # Only include mlflow if it has a tracking_uri
        if cloud_dict.get("mlflow") and cloud_dict["mlflow"].get("tracking_uri"):
            config_kwargs["mlflow"] = cloud_dict["mlflow"]

        cloud_config = CloudConfig(**config_kwargs)
    except Exception as e:
        _fail(f"Invalid cloud configuration: {e}")

    # Create session (validates credentials)
    try:
        session = CloudSession(cloud_config)
    except Exception as e:
        error_msg = str(e)
        if "credentials" in error_msg.lower():
            _fail(
                "AWS credentials not found or invalid",
                hint="Run 'aws configure' or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
            )
        elif "role" in error_msg.lower():
            _fail(
                f"Invalid SageMaker role: {cloud_dict['aws']['role']}",
                hint="Verify the role ARN exists and your credentials can assume it"
            )
        elif "bucket" in error_msg.lower():
            _fail(
                f"Cannot access S3 bucket: {cloud_dict['aws']['bucket']}",
                hint="Verify the bucket exists and you have access"
            )
        else:
            _fail(f"Cloud session validation failed: {e}")

    # Estimate costs
    try:
        estimator = CostEstimator(region=cloud_dict["aws"]["region"])
        estimate = estimator.estimate(cloud_config)
        cost_str = f"${estimate.spot_estimate:.2f}" if cloud_dict["instance"]["use_spot"] else f"${estimate.on_demand_estimate:.2f}"
        logger.info(f"Estimated cost: {cost_str} (max {cloud_dict['instance']['max_runtime_hours']}h)")
    except Exception:
        pass  # Cost estimation is optional

    # Create job config
    backend_enum = Backend.JAX if backend == "jax" else Backend.PYTORCH
    job_config = JobConfig(
        job_type=JobType.TRAINING,
        aws=cloud_config.aws,
        instance=cloud_config.instance,
        mlflow=cloud_config.mlflow,
        backend=backend_enum,
        training_config_path=str(config_yaml_path),
    )

    # Create and submit training job
    try:
        job = TrainingJob(job_config)
        job.validate()
        job_name = job.submit(session)

        # Print clear feedback to user (not just logging)
        region = cloud_dict['aws']['region']
        _output()
        _output("=" * 60)
        _output(f"  Cloud job submitted: {job_name}")
        _output("=" * 60)
        _output(f"  Instance: {cloud_dict['instance']['instance_type']}")
        _output(f"  Backend:  {backend}")
        _output(f"  Region:   {region}")
        _output()
        _output(f"  Console:  https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
        _output()
        _output("  Monitor with:")
        _output(f"    python -m soen_toolkit.cloud status {job_name}")
        _output("    python -m soen_toolkit.cloud_gui")
        _output("=" * 60)
        _output()

        return True

    except Exception as e:
        error_msg = str(e)
        if "image" in error_msg.lower():
            _fail(
                f"Docker image issue: {e}",
                hint="Build and push Docker images with: ./src/soen_toolkit/cloud/docker/build.sh"
            )
        else:
            _fail(f"Job submission failed: {e}")

    return True
