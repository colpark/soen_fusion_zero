"""Abstract base class for cloud jobs.

Defines the interface that all job types (training, inference, processing)
must implement. Enables consistent handling across different job types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import Backend, JobConfig
    from ..cost import CostEstimate
    from ..session import CloudSession


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class JobResult:
    """Result of a job execution."""

    job_name: str
    status: JobStatus
    start_time: float | None = None
    end_time: float | None = None
    duration_seconds: float | None = None
    output_path: str | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED


class JobSpec(ABC):
    """Abstract base class for all cloud job types.

    Subclasses must implement validation, cost estimation, and
    SageMaker conversion methods.
    """

    def __init__(self, config: JobConfig) -> None:
        """Initialize job specification.

        Args:
            config: Job configuration.
        """
        self._config = config
        self._cost_estimate: CostEstimate | None = None

    @property
    def config(self) -> JobConfig:
        """Get job configuration."""
        return self._config

    @property
    def job_name(self) -> str:
        """Get or generate job name."""
        if self._config.job_name:
            return self._config.job_name
        return self._generate_job_name()

    @property
    def backend(self) -> Backend:
        """Get ML backend (pytorch/jax)."""
        return self._config.backend

    @property
    def cost_estimate(self) -> CostEstimate | None:
        """Get cached cost estimate."""
        return self._cost_estimate

    def _generate_job_name(self) -> str:
        """Generate a unique job name.

        Format: soen-{job_type}-{timestamp}
        Ensures compliance with SageMaker job name constraints:
        - Max 63 characters
        - Pattern: [a-zA-Z0-9](-*[a-zA-Z0-9])*
        """
        job_type = self._config.job_type.value
        timestamp = time.strftime("%Y%m%d%H%M%S")
        base = f"soen-{job_type}-{timestamp}"

        # Ensure max 63 chars
        if len(base) > 63:
            base = base[:63]

        # Ensure valid characters
        base = re.sub(r"[^a-zA-Z0-9-]", "-", base)
        base = re.sub(r"-+", "-", base).strip("-")

        return base

    @abstractmethod
    def validate(self) -> None:
        """Validate job configuration.

        Should check all required fields and raise ValueError with
        clear messages for any validation failures.

        Raises:
            ValueError: If validation fails.
            FileNotFoundError: If required files are missing.
        """
        pass

    @abstractmethod
    def estimate_cost(self) -> CostEstimate:
        """Estimate job cost.

        Returns:
            CostEstimate with on-demand and spot pricing.
        """
        pass

    @abstractmethod
    def prepare_inputs(self, session: CloudSession) -> dict[str, str]:
        """Prepare input channels for SageMaker.

        Uploads local files to S3 if needed and returns channel mapping.

        Args:
            session: Cloud session for S3 operations.

        Returns:
            Dictionary mapping channel names to S3 URIs.
        """
        pass

    @abstractmethod
    def get_environment(self) -> dict[str, str]:
        """Get environment variables for the job container.

        Returns:
            Dictionary of environment variable name -> value.
        """
        pass

    @abstractmethod
    def submit(self, session: CloudSession) -> str:
        """Submit job to SageMaker.

        Args:
            session: Cloud session for AWS operations.

        Returns:
            Job name/ID of the submitted job.

        Raises:
            CloudSessionError: If submission fails.
        """
        pass

    def get_docker_image(self, session: CloudSession) -> str:
        """Get Docker image URI for the job.

        Uses explicit image from config, or selects based on backend.

        Args:
            session: Cloud session (may have default images configured).

        Returns:
            Docker image URI.

        Raises:
            ValueError: If no image is configured for the backend.
        """
        # Explicit image in job config takes precedence
        if self._config.docker_image:
            return self._config.docker_image

        # Check cloud config for backend-specific image
        cloud_config = session.cloud_config
        if cloud_config:
            image = cloud_config.get_docker_image(self._config.backend)
            if image:
                return image

        raise ValueError(
            f"No Docker image configured for backend '{self._config.backend.value}'. "
            f"Set docker_image in job config or docker_images in cloud config."
        )


class SubmittedJob:
    """Represents a submitted job that can be monitored.

    Provides methods to check status, wait for completion, and
    retrieve results.
    """

    def __init__(
        self,
        job_name: str,
        session: CloudSession,
        job_spec: JobSpec,
    ) -> None:
        """Initialize submitted job.

        Args:
            job_name: SageMaker job name.
            session: Cloud session for AWS operations.
            job_spec: Original job specification.
        """
        self._job_name = job_name
        self._session = session
        self._job_spec = job_spec
        self._cached_status: JobStatus | None = None

    @property
    def job_name(self) -> str:
        """Get job name."""
        return self._job_name

    @property
    def job_spec(self) -> JobSpec:
        """Get original job specification."""
        return self._job_spec

    def get_status(self) -> JobStatus:
        """Get current job status.

        Returns:
            Current JobStatus.
        """
        try:
            sm = self._session.sagemaker_client()
            response = sm.describe_training_job(TrainingJobName=self._job_name)
            status_str = response.get("TrainingJobStatus", "Unknown")

            status_map = {
                "InProgress": JobStatus.RUNNING,
                "Completed": JobStatus.COMPLETED,
                "Failed": JobStatus.FAILED,
                "Stopping": JobStatus.RUNNING,
                "Stopped": JobStatus.STOPPED,
            }
            self._cached_status = status_map.get(status_str, JobStatus.UNKNOWN)
            return self._cached_status
        except Exception:
            return JobStatus.UNKNOWN

    def is_terminal(self) -> bool:
        """Check if job has reached a terminal state."""
        status = self.get_status()
        return status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED)

    def wait(self, poll_interval: int = 30, timeout: int | None = None) -> JobResult:
        """Wait for job to complete.

        Args:
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait (None = no timeout).

        Returns:
            JobResult with final status and metrics.

        Raises:
            TimeoutError: If timeout is reached before completion.
        """
        import logging

        logger = logging.getLogger(__name__)

        start_time = time.time()
        job_start: float | None = None

        while True:
            status = self.get_status()

            if job_start is None and status == JobStatus.RUNNING:
                job_start = time.time()
                logger.info(f"Job {self._job_name} started running")

            if self.is_terminal():
                end_time = time.time()
                duration = end_time - (job_start or start_time)

                result = JobResult(
                    job_name=self._job_name,
                    status=status,
                    start_time=job_start,
                    end_time=end_time,
                    duration_seconds=duration,
                )

                # Try to get additional details
                try:
                    sm = self._session.sagemaker_client()
                    response = sm.describe_training_job(TrainingJobName=self._job_name)
                    result.output_path = response.get("ModelArtifacts", {}).get(
                        "S3ModelArtifacts"
                    )
                    if status == JobStatus.FAILED:
                        result.error_message = response.get("FailureReason")
                except Exception:
                    pass

                return result

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Job {self._job_name} did not complete within {timeout} seconds"
                )

            logger.info(f"Job {self._job_name} status: {status.value}")
            time.sleep(poll_interval)

    def stop(self) -> None:
        """Stop the running job."""
        sm = self._session.sagemaker_client()
        sm.stop_training_job(TrainingJobName=self._job_name)

    def get_logs_url(self) -> str:
        """Get CloudWatch logs URL for the job."""
        region = self._session.region
        log_group = "/aws/sagemaker/TrainingJobs"
        return (
            f"https://{region}.console.aws.amazon.com/cloudwatch/home"
            f"?region={region}#logsV2:log-groups/log-group/{log_group}"
            f"/log-events/{self._job_name}"
        )

