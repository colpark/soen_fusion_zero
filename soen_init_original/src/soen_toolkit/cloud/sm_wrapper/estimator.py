"""SageMaker Estimator creation and job submission.

Provides functions to create properly configured SageMaker Estimators
for training jobs with support for spot instances, multi-node, and
custom Docker images.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sagemaker import Session as SageMakerSession
    from sagemaker.estimator import Estimator

    from ..config import Backend, InstanceConfig
    from ..session import CloudSession

logger = logging.getLogger(__name__)


def create_sagemaker_session(session: CloudSession) -> SageMakerSession:
    """Create a SageMaker session from CloudSession.

    Args:
        session: Cloud session with AWS configuration.

    Returns:
        Configured SageMaker session.
    """
    import sagemaker

    return sagemaker.Session(
        boto_session=session.boto_session,
        sagemaker_client=session.sagemaker_client(),
    )


def get_ddp_distribution_config(instance_count: int) -> dict[str, Any]:
    """Get distribution configuration for multi-node training.

    Args:
        instance_count: Number of instances.

    Returns:
        Distribution configuration dict for Estimator.
    """
    if instance_count <= 1:
        return {}

    return {
        "pytorchddp": {
            "enabled": True,
        }
    }


def get_environment_for_backend(
    backend: Backend,
    instance_count: int = 1,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Get environment variables for a backend.

    Args:
        backend: ML framework backend (pytorch/jax).
        instance_count: Number of instances for multi-node settings.
        extra_env: Additional environment variables.

    Returns:
        Environment variable dictionary.
    """
    env = {
        "PYTHONPATH": "/opt/ml/code:/opt/ml/code/src",
        "SOEN_BACKEND": backend.value,
    }

    # Multi-node NCCL settings
    if instance_count > 1:
        env.update({
            "NCCL_DEBUG": "WARN",
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "eth0",
            # Disable SageMaker DDP wrapper (use native PyTorch DDP)
            "SMDDP_ENABLED": "false",
        })

    if extra_env:
        env.update(extra_env)

    return env


def create_estimator(
    session: CloudSession,
    *,
    image_uri: str,
    instance_config: InstanceConfig,
    backend: Backend,
    environment: dict[str, str] | None = None,
    job_name_prefix: str = "soen",
    output_path: str | None = None,
) -> Estimator:
    """Create a SageMaker Estimator for training.

    Args:
        session: Cloud session with AWS configuration.
        image_uri: Docker image URI.
        instance_config: Instance configuration.
        backend: ML framework backend.
        environment: Additional environment variables.
        job_name_prefix: Prefix for generated job names.
        output_path: S3 path for model artifacts.

    Returns:
        Configured Estimator ready for .fit().
    """
    from sagemaker.estimator import Estimator

    sm_session = create_sagemaker_session(session)
    max_run_seconds = int(instance_config.max_runtime_hours * 3600)

    # Build environment
    env = get_environment_for_backend(
        backend,
        instance_count=instance_config.instance_count,
        extra_env=environment,
    )

    # Build distribution config
    distribution = get_ddp_distribution_config(instance_config.instance_count)

    # Default output path
    if output_path is None:
        cloud_cfg = session.cloud_config
        project = cloud_cfg.project if cloud_cfg else "soen"
        experiment = cloud_cfg.experiment if cloud_cfg else "default"
        output_path = f"s3://{session.bucket}/soen/{project}/{experiment}/output"

    # Estimator kwargs
    estimator_kwargs: dict[str, Any] = {
        "image_uri": image_uri,
        "role": session.config.role,
        "instance_count": instance_config.instance_count,
        "instance_type": instance_config.instance_type,
        "sagemaker_session": sm_session,
        "environment": env,
        "max_run": max_run_seconds,
        "output_path": output_path,
        "use_spot_instances": instance_config.use_spot,
        "disable_profiler": True,
        "base_job_name": job_name_prefix,
    }

    # Add spot-specific settings
    if instance_config.use_spot:
        estimator_kwargs["max_wait"] = max_run_seconds

    # Add distribution config if multi-node
    if distribution:
        estimator_kwargs["distribution"] = distribution

    logger.info(
        f"Creating Estimator: {instance_config.instance_type} x "
        f"{instance_config.instance_count}, spot={instance_config.use_spot}"
    )

    return Estimator(**estimator_kwargs)


def create_training_job(
    session: CloudSession,
    *,
    image_uri: str,
    instance_config: InstanceConfig,
    backend: Backend,
    inputs: dict[str, str],
    environment: dict[str, str] | None = None,
    job_name: str | None = None,
    wait: bool = False,
) -> str:
    """Create and submit a training job.

    This is a convenience function that creates an Estimator and
    calls .fit() in one step.

    Args:
        session: Cloud session with AWS configuration.
        image_uri: Docker image URI.
        instance_config: Instance configuration.
        backend: ML framework backend.
        inputs: Input channel mapping (name -> S3 URI).
        environment: Additional environment variables.
        job_name: Explicit job name (auto-generated if None).
        wait: If True, wait for job completion.

    Returns:
        Job name of the submitted training job.
    """
    from sagemaker.inputs import TrainingInput

    estimator = create_estimator(
        session,
        image_uri=image_uri,
        instance_config=instance_config,
        backend=backend,
        environment=environment,
    )

    # Convert inputs to TrainingInput
    sm_inputs = {name: TrainingInput(uri) for name, uri in inputs.items()}

    # Generate job name if not provided
    if job_name is None:
        timestamp = time.strftime("%Y%m%d%H%M%S")
        job_name = f"soen-{backend.value}-{timestamp}"

    logger.info(f"Submitting training job: {job_name}")

    # Submit job
    estimator.fit(
        inputs=sm_inputs,
        job_name=job_name,
        wait=wait,
        logs="None" if not wait else "All",
    )

    latest_job = estimator.latest_training_job
    if latest_job is None:
        msg = "No training job was created"
        raise RuntimeError(msg)
    actual_job_name = latest_job.name
    logger.info(f"Training job submitted: {actual_job_name}")

    return actual_job_name


def wait_for_training_job(
    session: CloudSession,
    job_name: str,
    poll_interval: int = 30,
) -> dict[str, Any]:
    """Wait for a training job to complete.

    Args:
        session: Cloud session for AWS operations.
        job_name: SageMaker training job name.
        poll_interval: Seconds between status checks.

    Returns:
        Job description dictionary from SageMaker.

    Raises:
        RuntimeError: If job fails or is stopped.
    """
    sm = session.sagemaker_client()
    terminal_states = {"Completed", "Failed", "Stopped"}

    while True:
        response = sm.describe_training_job(TrainingJobName=job_name)
        status = response.get("TrainingJobStatus", "Unknown")
        secondary = response.get("SecondaryStatus", "")

        logger.info(f"Job {job_name}: {status} ({secondary})")

        if status in terminal_states:
            if status == "Completed":
                return response
            elif status == "Failed":
                reason = response.get("FailureReason", "Unknown")
                raise RuntimeError(f"Training job failed: {reason}")
            else:
                raise RuntimeError(f"Training job stopped: {status}")

        time.sleep(poll_interval)


def get_training_job_artifacts(
    session: CloudSession,
    job_name: str,
) -> dict[str, str]:
    """Get artifact locations for a completed training job.

    Args:
        session: Cloud session for AWS operations.
        job_name: SageMaker training job name.

    Returns:
        Dictionary with artifact S3 URIs.
    """
    sm = session.sagemaker_client()
    response = sm.describe_training_job(TrainingJobName=job_name)

    artifacts = {}

    # Model artifacts
    model_artifacts = response.get("ModelArtifacts", {})
    if model_artifacts.get("S3ModelArtifacts"):
        artifacts["model"] = model_artifacts["S3ModelArtifacts"]

    # Output data (if configured)
    output_config = response.get("OutputDataConfig", {})
    if output_config.get("S3OutputPath"):
        artifacts["output"] = output_config["S3OutputPath"]

    return artifacts

