"""Batch inference job implementation for SageMaker.

Handles batch transform jobs for running inference on datasets
using trained SOEN models.
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


class InferenceJob(JobSpec):
    """Batch inference job for SOEN models.

    Uses SageMaker Batch Transform to run inference on datasets.

    Handles:
    - Model artifact download/setup
    - Data splitting for parallel inference
    - Output assembly and upload
    """

    def __init__(self, config: JobConfig) -> None:
        """Initialize inference job.

        Args:
            config: Job configuration with job_type=INFERENCE.
        """
        super().__init__(config)
        self._model_s3_uri: str | None = None
        self._data_s3_uri: str | None = None
        self._output_s3_uri: str | None = None

    @property
    def model_path(self) -> str:
        """Get model artifact path."""
        if self._config.data.model_path is None:
            raise ValueError("model_path is required for inference jobs")
        return self._config.data.model_path

    @property
    def data_path(self) -> str | None:
        """Get input data path."""
        return self._config.data.data_path

    def validate(self) -> None:
        """Validate inference job configuration.

        Raises:
            ValueError: If configuration is invalid.
            FileNotFoundError: If required files are missing.
        """
        # Check model path
        model_path = self.model_path
        if not model_path.startswith("s3://"):
            local_path = Path(model_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

        # Check data path if provided
        data_path = self.data_path
        if data_path and not data_path.startswith("s3://"):
            local_path = Path(data_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Data path not found: {data_path}")

        logger.info("Inference job validation passed")

    def estimate_cost(self) -> CostEstimate:
        """Estimate inference job cost.

        Returns:
            CostEstimate with pricing information.
        """
        from ..cost import CostEstimator

        estimator = CostEstimator(region=self._config.aws.region)
        self._cost_estimate = estimator.estimate(self._config)
        return self._cost_estimate

    def prepare_inputs(self, session: CloudSession) -> dict[str, str]:
        """Prepare input data for batch transform.

        Args:
            session: Cloud session for S3 operations.

        Returns:
            Dictionary with model and data S3 URIs.
        """
        cloud_cfg = session.cloud_config
        project = cloud_cfg.project if cloud_cfg else "soen"
        experiment = cloud_cfg.experiment if cloud_cfg else "default"
        timestamp = time.strftime("%Y%m%d%H%M%S")
        s3_prefix = f"soen/{project}/{experiment}/{self.job_name}/{timestamp}"

        inputs: dict[str, str] = {}

        # Upload model if local
        model_path = self.model_path
        if model_path.startswith("s3://"):
            self._model_s3_uri = model_path
        else:
            local_path = Path(model_path)
            model_s3_key = f"{s3_prefix}/model/{local_path.name}"
            self._model_s3_uri = session.upload_file(str(local_path), model_s3_key)

        inputs["model"] = self._model_s3_uri

        # Upload data if local
        data_path = self.data_path
        if data_path:
            if data_path.startswith("s3://"):
                self._data_s3_uri = data_path
            else:
                local_path = Path(data_path)
                if local_path.is_file():
                    data_s3_key = f"{s3_prefix}/data/{local_path.name}"
                    self._data_s3_uri = session.upload_file(str(local_path), data_s3_key)
                else:
                    data_s3_key = f"{s3_prefix}/data"
                    session.upload_directory(str(local_path), data_s3_key)
                    self._data_s3_uri = f"s3://{session.bucket}/{data_s3_key}"

            inputs["data"] = self._data_s3_uri

        # Set output path
        output_path = self._config.data.output_path
        if output_path:
            self._output_s3_uri = output_path
        else:
            self._output_s3_uri = f"s3://{session.bucket}/{s3_prefix}/output"

        inputs["output"] = self._output_s3_uri

        logger.info(f"Prepared inference inputs: {inputs}")
        return inputs

    def get_environment(self) -> dict[str, str]:
        """Get environment variables for inference container.

        Returns:
            Dictionary of environment variables.
        """
        env = {
            "SOEN_JOB_TYPE": "inference",
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
        """Submit batch transform job to SageMaker.

        Args:
            session: Cloud session for AWS operations.

        Returns:
            Transform job name.

        Raises:
            CloudSessionError: If submission fails.
        """
        import sagemaker
        from sagemaker.transformer import Transformer

        logger.info(f"Submitting inference job: {self.job_name}")

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

        # Create and register model in SageMaker
        from sagemaker.model import Model

        model_name = f"{self.job_name}-model"
        env: dict[str, Any] = self.get_environment()
        sagemaker_model = Model(
            image_uri=docker_image,
            model_data=inputs["model"],
            role=self._config.aws.role,
            sagemaker_session=sm_session,
            name=model_name,
            env=env,
        )
        # Create the model in SageMaker (required before creating transformer)
        sagemaker_model.create(instance_type=self._config.instance.instance_type)

        # Create transformer
        instance_cfg = self._config.instance
        transformer = Transformer(
            model_name=model_name,
            instance_count=instance_cfg.instance_count,
            instance_type=instance_cfg.instance_type,
            output_path=inputs["output"],
            sagemaker_session=sm_session,
            max_concurrent_transforms=instance_cfg.instance_count,
            max_payload=6,  # MB
        )

        # Start transform job
        if inputs.get("data"):
            transformer.transform(
                data=inputs["data"],
                job_name=self.job_name,
                content_type="application/x-npy",
                split_type="None",
                wait=False,
            )
        else:
            raise ValueError("data path required for batch transform")

        logger.info(f"Inference job submitted: {self.job_name}")
        return self.job_name

    def get_output_path(self) -> str | None:
        """Get output S3 path.

        Returns:
            S3 URI for output data.
        """
        return self._output_s3_uri

