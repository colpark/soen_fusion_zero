"""MLflow integration for cloud jobs.

Provides utilities for setting up MLflow tracking in cloud environments
and logging job metadata, metrics, and artifacts.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import JobConfig, MLflowConfig
    from ..cost import CostEstimate

logger = logging.getLogger(__name__)


# Check if mlflow is available
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore


@dataclass
class MLflowState:
    """Current MLflow tracking state."""

    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    is_active: bool = False


_state = MLflowState()


def is_available() -> bool:
    """Check if MLflow is available."""
    return MLFLOW_AVAILABLE


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> bool:
    """Configure MLflow from parameters or environment.

    Args:
        tracking_uri: MLflow tracking server URI (or use MLFLOW_TRACKING_URI env).
        experiment_name: Experiment name (or use MLFLOW_EXPERIMENT_NAME env).

    Returns:
        True if MLflow was configured successfully.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("mlflow not installed, tracking disabled")
        return False

    # Get from params or environment
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    exp_name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME", "soen-cloud")

    if not uri:
        logger.info("No MLflow tracking URI configured, tracking disabled")
        return False

    try:
        mlflow.set_tracking_uri(uri)
        _state.tracking_uri = uri
        logger.info(f"MLflow tracking URI: {uri}")

        mlflow.set_experiment(exp_name)
        _state.experiment_name = exp_name
        logger.info(f"MLflow experiment: {exp_name}")

        return True

    except Exception as e:
        logger.warning(f"Failed to setup MLflow: {e}")
        return False


def setup_from_config(config: MLflowConfig) -> bool:
    """Configure MLflow from MLflowConfig.

    Args:
        config: MLflow configuration.

    Returns:
        True if MLflow was configured successfully.
    """
    return setup_mlflow(
        tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
    )


@contextmanager
def start_run(
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[str | None, None, None]:
    """Context manager to start and end an MLflow run.

    Args:
        run_name: Optional run name.
        tags: Optional tags to set on the run.

    Yields:
        Run ID if MLflow is configured, None otherwise.
    """
    if not MLFLOW_AVAILABLE or not _state.tracking_uri:
        yield None
        return

    try:
        run = mlflow.start_run(run_name=run_name)
        _state.run_id = run.info.run_id
        _state.is_active = True

        if tags:
            mlflow.set_tags(tags)

        logger.info(f"MLflow run started: {_state.run_id}")
        yield _state.run_id

    finally:
        if _state.is_active:
            mlflow.end_run()
            _state.is_active = False
            logger.info(f"MLflow run ended: {_state.run_id}")


def log_job_metadata(
    job_config: JobConfig,
    cost_estimate: CostEstimate | None = None,
) -> None:
    """Log job configuration and cost estimate to MLflow.

    Args:
        job_config: Job configuration.
        cost_estimate: Optional cost estimate.
    """
    if not MLFLOW_AVAILABLE or not _state.is_active:
        return

    try:
        # Log job parameters
        params = {
            "job_type": job_config.job_type.value,
            "backend": job_config.backend.value,
            "instance_type": job_config.instance.instance_type,
            "instance_count": job_config.instance.instance_count,
            "use_spot": job_config.instance.use_spot,
            "max_runtime_hours": job_config.instance.max_runtime_hours,
            "region": job_config.aws.region,
        }
        mlflow.log_params(params)

        # Log cost estimate
        if cost_estimate:
            mlflow.log_params({
                "estimated_cost_spot": f"${cost_estimate.spot_total:.2f}",
                "estimated_cost_on_demand": f"${cost_estimate.on_demand_total:.2f}",
            })

        # Log tags
        mlflow.set_tags({
            "cloud.backend": job_config.backend.value,
            "cloud.instance_type": job_config.instance.instance_type,
            "cloud.job_type": job_config.job_type.value,
        })

        logger.info("Logged job metadata to MLflow")

    except Exception as e:
        logger.warning(f"Failed to log job metadata: {e}")


def log_metrics(
    metrics: dict[str, float],
    step: int | None = None,
) -> None:
    """Log metrics to MLflow.

    Args:
        metrics: Dictionary of metric name -> value.
        step: Optional step number.
    """
    if not MLFLOW_AVAILABLE or not _state.is_active:
        return

    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


def log_params(params: dict[str, Any]) -> None:
    """Log parameters to MLflow.

    Args:
        params: Dictionary of parameter name -> value.
    """
    if not MLFLOW_AVAILABLE or not _state.is_active:
        return

    try:
        # Convert all values to strings (MLflow requirement)
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)
    except Exception as e:
        logger.warning(f"Failed to log params: {e}")


def log_model_artifact(
    model_path: str,
    artifact_name: str = "model",
) -> None:
    """Log a model artifact to MLflow.

    Args:
        model_path: Path to model file or directory.
        artifact_name: Name for the artifact in MLflow.
    """
    if not MLFLOW_AVAILABLE or not _state.is_active:
        return

    try:
        import os

        if os.path.isdir(model_path):
            mlflow.log_artifacts(model_path, artifact_name)
        else:
            mlflow.log_artifact(model_path, artifact_name)

        logger.info(f"Logged model artifact: {artifact_name}")

    except Exception as e:
        logger.warning(f"Failed to log model artifact: {e}")


def log_artifact(
    local_path: str,
    artifact_path: str | None = None,
) -> None:
    """Log an artifact file to MLflow.

    Args:
        local_path: Path to local file.
        artifact_path: Optional path within artifact directory.
    """
    if not MLFLOW_AVAILABLE or not _state.is_active:
        return

    try:
        mlflow.log_artifact(local_path, artifact_path)
    except Exception as e:
        logger.warning(f"Failed to log artifact: {e}")


def set_tag(key: str, value: str) -> None:
    """Set a tag on the current run.

    Args:
        key: Tag key.
        value: Tag value.
    """
    if not MLFLOW_AVAILABLE or not _state.is_active:
        return

    try:
        mlflow.set_tag(key, value)
    except Exception as e:
        logger.warning(f"Failed to set tag: {e}")


def end_run(status: str = "FINISHED") -> None:
    """End the current MLflow run.

    Args:
        status: Run status (FINISHED, FAILED, KILLED).
    """
    if not MLFLOW_AVAILABLE or not _state.is_active:
        return

    try:
        mlflow.end_run(status=status)
        _state.is_active = False
        logger.info(f"MLflow run ended with status: {status}")
    except Exception as e:
        logger.warning(f"Failed to end run: {e}")


class MLflowTracker:
    """High-level MLflow tracker for cloud jobs.

    Provides a clean interface for tracking experiments with
    automatic setup and cleanup.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        """Initialize tracker.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: Experiment name.
        """
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._configured = False
        self._run_id: str | None = None

    def setup(self) -> bool:
        """Configure MLflow tracking.

        Returns:
            True if configured successfully.
        """
        self._configured = setup_mlflow(
            tracking_uri=self._tracking_uri,
            experiment_name=self._experiment_name,
        )
        return self._configured

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Start a new MLflow run.

        Args:
            run_name: Optional run name.
            tags: Optional tags.

        Returns:
            Run ID if successful.
        """
        if not self._configured:
            return None

        if not MLFLOW_AVAILABLE:
            return None

        try:
            run = mlflow.start_run(run_name=run_name)
            self._run_id = run.info.run_id
            _state.run_id = self._run_id
            _state.is_active = True

            if tags:
                mlflow.set_tags(tags)

            return self._run_id
        except Exception as e:
            logger.warning(f"Failed to start run: {e}")
            return None

    def log_job_start(
        self,
        job_config: JobConfig,
        cost_estimate: CostEstimate | None = None,
    ) -> None:
        """Log job start metadata.

        Args:
            job_config: Job configuration.
            cost_estimate: Optional cost estimate.
        """
        log_job_metadata(job_config, cost_estimate)
        set_tag("status", "started")

    def log_job_complete(
        self,
        output_path: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log job completion.

        Args:
            output_path: Path to output artifacts.
            metrics: Final metrics to log.
        """
        set_tag("status", "completed")
        if output_path:
            log_params({"output_path": output_path})
        if metrics:
            log_metrics(metrics)

    def log_job_failed(self, error: str) -> None:
        """Log job failure.

        Args:
            error: Error message.
        """
        set_tag("status", "failed")
        # Truncate long error messages
        log_params({"error": error[:250] if len(error) > 250 else error})

    def end_run(self, success: bool = True) -> None:
        """End the current run.

        Args:
            success: Whether the job succeeded.
        """
        status = "FINISHED" if success else "FAILED"
        end_run(status)
        self._run_id = None

    @property
    def run_id(self) -> str | None:
        """Get current run ID."""
        return self._run_id

    @property
    def is_tracking(self) -> bool:
        """Check if tracking is active."""
        return self._configured and _state.is_active

