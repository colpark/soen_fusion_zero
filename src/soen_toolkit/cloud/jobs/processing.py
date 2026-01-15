"""Data processing job implementation for SageMaker.

Handles data preprocessing, postprocessing, and other custom
processing jobs using SageMaker Processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from .base import JobSpec

if TYPE_CHECKING:
    from ..config import JobConfig
    from ..cost import CostEstimate
    from ..session import CloudSession

logger = logging.getLogger(__name__)


class ProcessingJob(JobSpec):
    """Data processing job for SageMaker.

    Uses SageMaker Processing to run custom Python scripts for
    data preprocessing, transformation, or analysis.

    Handles:
    - Custom script execution
    - Input/output data management
    - Environment configuration
    """

    def __init__(self, config: JobConfig) -> None:
        """Initialize processing job.

        Args:
            config: Job configuration with job_type=PROCESSING.
        """
        super().__init__(config)
        self._input_s3_uri: str | None = None
        self._output_s3_uri: str | None = None

    @property
    def processing_script(self) -> Path:
        """Get path to processing script."""
        if self._config.processing_script is None:
            raise ValueError("processing_script is required for processing jobs")
        return self._config.processing_script

    @property
    def input_path(self) -> str | None:
        """Get input data path."""
        return self._config.data.data_path

    @property
    def output_path(self) -> str | None:
        """Get output data path."""
        return self._config.data.output_path

    def validate(self) -> None:
        """Validate processing job configuration.

        Raises:
            ValueError: If configuration is invalid.
            FileNotFoundError: If required files are missing.
        """
        # Check processing script exists
        script_path = self.processing_script
        if not script_path.exists():
            raise FileNotFoundError(f"Processing script not found: {script_path}")

        # Validate script is a Python file
        if script_path.suffix != ".py":
            raise ValueError(f"Processing script must be a .py file: {script_path}")

        # Check input path if provided
        input_path = self.input_path
        if input_path and not input_path.startswith("s3://"):
            local_path = Path(input_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Input path not found: {input_path}")

        logger.info(f"Processing job validation passed: {script_path}")

    def estimate_cost(self) -> CostEstimate:
        """Estimate processing job cost.

        Returns:
            CostEstimate with pricing information.
        """
        from ..cost import CostEstimator

        estimator = CostEstimator(region=self._config.aws.region)
        self._cost_estimate = estimator.estimate(self._config)
        return self._cost_estimate

    def prepare_inputs(self, session: CloudSession) -> dict[str, str]:
        """Prepare input data and script for processing.

        Args:
            session: Cloud session for S3 operations.

        Returns:
            Dictionary with script, input, and output S3 URIs.
        """
        cloud_cfg = session.cloud_config
        project = cloud_cfg.project if cloud_cfg else "soen"
        experiment = cloud_cfg.experiment if cloud_cfg else "default"
        timestamp = time.strftime("%Y%m%d%H%M%S")
        s3_prefix = f"soen/{project}/{experiment}/{self.job_name}/{timestamp}"

        inputs: dict[str, str] = {}

        # Upload processing script
        script_path = self.processing_script
        script_s3_key = f"{s3_prefix}/code/{script_path.name}"
        inputs["script"] = session.upload_file(str(script_path), script_s3_key)

        # Upload input data if local
        input_path = self.input_path
        if input_path:
            if input_path.startswith("s3://"):
                self._input_s3_uri = input_path
            else:
                local_path = Path(input_path)
                if local_path.is_file():
                    input_s3_key = f"{s3_prefix}/input/{local_path.name}"
                    self._input_s3_uri = session.upload_file(str(local_path), input_s3_key)
                else:
                    input_s3_key = f"{s3_prefix}/input"
                    session.upload_directory(str(local_path), input_s3_key)
                    self._input_s3_uri = f"s3://{session.bucket}/{input_s3_key}"

            inputs["input"] = self._input_s3_uri

        # Set output path
        output_path = self.output_path
        if output_path:
            self._output_s3_uri = output_path
        else:
            self._output_s3_uri = f"s3://{session.bucket}/{s3_prefix}/output"

        inputs["output"] = self._output_s3_uri

        logger.info(f"Prepared processing inputs: {inputs}")
        return inputs

    def get_environment(self) -> dict[str, str]:
        """Get environment variables for processing container.

        Returns:
            Dictionary of environment variables.
        """
        env = {
            "SOEN_JOB_TYPE": "processing",
            "SOEN_JOB_NAME": self.job_name,
            "SOEN_BACKEND": self._config.backend.value,
        }

        # MLflow configuration
        mlflow_cfg = self._config.mlflow
        if mlflow_cfg.tracking_uri:
            env["MLFLOW_TRACKING_URI"] = mlflow_cfg.tracking_uri
            env["MLFLOW_EXPERIMENT_NAME"] = mlflow_cfg.experiment_name

        env.update(self._config.environment)
        return env

    def submit(self, session: CloudSession) -> str:
        """Submit processing job to SageMaker.

        Args:
            session: Cloud session for AWS operations.

        Returns:
            Processing job name.

        Raises:
            CloudSessionError: If submission fails.
        """
        import sagemaker
        from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

        logger.info(f"Submitting processing job: {self.job_name}")

        # Validate and prepare
        self.validate()
        inputs = self.prepare_inputs(session)
        docker_image = self.get_docker_image(session)

        # Validate Docker image
        session.validate_ecr_image(docker_image)

        # Create SageMaker session
        sm_session = sagemaker.Session(
            boto_session=session.boto_session,
            sagemaker_client=session.sagemaker_client(),
        )

        # Create processor
        instance_cfg = self._config.instance
        max_runtime = int(instance_cfg.max_runtime_hours * 3600)

        env: dict[str, Any] = self.get_environment()
        processor = ScriptProcessor(
            image_uri=docker_image,
            command=["python3"],
            instance_count=instance_cfg.instance_count,
            instance_type=instance_cfg.instance_type,
            role=self._config.aws.role,
            sagemaker_session=sm_session,
            max_runtime_in_seconds=max_runtime,
            env=env,
        )

        # Build processing inputs and outputs
        processing_inputs = []
        if inputs.get("input"):
            processing_inputs.append(
                ProcessingInput(
                    source=inputs["input"],
                    destination="/opt/ml/processing/input",
                    input_name="input",
                )
            )

        processing_outputs = [
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=inputs["output"],
                output_name="output",
            )
        ]

        # Run processing job
        processor.run(
            code=inputs["script"],
            inputs=processing_inputs,
            outputs=processing_outputs,
            job_name=self.job_name,
            wait=False,
        )

        logger.info(f"Processing job submitted: {self.job_name}")
        return self.job_name

    def get_output_path(self) -> str | None:
        """Get output S3 path.

        Returns:
            S3 URI for output data.
        """
        return self._output_s3_uri

