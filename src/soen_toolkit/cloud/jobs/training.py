"""Training job implementation for SageMaker.

Handles SOEN toolkit training jobs including config upload, channel
setup, and integration with the training module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import yaml

from .base import JobSpec, SubmittedJob

if TYPE_CHECKING:
    from ..config import JobConfig
    from ..cost import CostEstimate
    from ..session import CloudSession

logger = logging.getLogger(__name__)


class TrainingJob(JobSpec):
    """Training job for SOEN toolkit models.

    Handles:
    - Training configuration upload to S3
    - Dataset channel setup
    - Model checkpoint channel (optional)
    - MLflow integration via environment variables
    - Multi-node DDP configuration
    """

    def __init__(self, config: JobConfig) -> None:
        """Initialize training job.

        Args:
            config: Job configuration with job_type=TRAINING.
        """
        super().__init__(config)
        self._training_config: dict[str, Any] | None = None
        self._config_s3_uri: str | None = None

    @property
    def training_config_path(self) -> Path:
        """Get path to training configuration file."""
        if self._config.training_config_path is None:
            raise ValueError("training_config_path is required for training jobs")
        return self._config.training_config_path

    @property
    def training_config(self) -> dict[str, Any]:
        """Load and cache training configuration."""
        if self._training_config is None:
            with open(self.training_config_path) as f:
                self._training_config = yaml.safe_load(f) or {}
        return self._training_config

    def validate(self) -> None:
        """Validate training job configuration.

        Raises:
            ValueError: If configuration is invalid.
            FileNotFoundError: If required files are missing.
        """
        # Check training config exists
        config_path = self.training_config_path
        if not config_path.exists():
            raise FileNotFoundError(
                f"Training configuration file not found: {config_path}"
            )

        # Load and validate training config
        training_cfg = self.training_config

        # Check for required sections
        if "training" not in training_cfg:
            raise ValueError(
                f"Training config '{config_path}' missing required 'training' section"
            )

        # Check data configuration
        data_cfg = training_cfg.get("data", {})
        data_path = data_cfg.get("data_path")

        if data_path:
            # If it's a local path, check it exists
            if not data_path.startswith("s3://"):
                local_path = Path(data_path)
                if not local_path.exists():
                    raise FileNotFoundError(
                        f"Data path not found: {data_path}. "
                        f"Use an S3 URI or ensure the local path exists."
                    )

        # Validate multi-node configuration
        instance_count = self._config.instance.instance_count
        training_section = training_cfg.get("training", {})

        if instance_count > 1:
            # Warn if num_nodes doesn't match (we'll override it)
            config_nodes = training_section.get("num_nodes", 1)
            if config_nodes != instance_count:
                logger.warning(
                    f"Training config has num_nodes={config_nodes}, but "
                    f"instance_count={instance_count}. Will override to match."
                )

            # Check strategy is set for multi-node
            strategy = training_section.get("strategy")
            if not strategy:
                logger.info(
                    f"Multi-node training ({instance_count} nodes) without strategy. "
                    f"Will default to 'ddp'."
                )

        logger.info(f"Training job validation passed: {config_path}")

    def estimate_cost(self) -> CostEstimate:
        """Estimate training job cost.

        Returns:
            CostEstimate with pricing information.
        """
        from ..cost import CostEstimator

        estimator = CostEstimator(region=self._config.aws.region)
        self._cost_estimate = estimator.estimate(self._config)
        return self._cost_estimate

    def prepare_inputs(self, session: CloudSession) -> dict[str, str]:
        """Prepare input channels for SageMaker.

        Uploads training config and optionally data to S3.

        Args:
            session: Cloud session for S3 operations.

        Returns:
            Dictionary mapping channel names to S3 URIs.
        """
        inputs: dict[str, str] = {}
        cloud_cfg = session.cloud_config

        # Generate S3 prefix for this job
        project = cloud_cfg.project if cloud_cfg else "soen"
        experiment = cloud_cfg.experiment if cloud_cfg else "default"
        timestamp = time.strftime("%Y%m%d%H%M%S")
        s3_prefix = f"soen/{project}/{experiment}/{self.job_name}/{timestamp}"

        # Upload training config
        config_s3_key = f"{s3_prefix}/config/training_config.yaml"

        # Modify config for container environment
        container_config = self._prepare_container_config()

        # Write modified config to temp file and upload
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            yaml.safe_dump(container_config, tmp)
            tmp_path = tmp.name

        try:
            self._config_s3_uri = session.upload_file(tmp_path, config_s3_key)
            inputs["config"] = self._config_s3_uri
        finally:
            os.unlink(tmp_path)

        # Handle data channel
        data_cfg = self.training_config.get("data", {})
        data_path = data_cfg.get("data_path")

        if data_path:
            if data_path.startswith("s3://"):
                # Already on S3
                inputs["training"] = data_path
            else:
                # Upload local data
                local_path = Path(data_path)
                if local_path.is_file():
                    data_s3_key = f"{s3_prefix}/data/{local_path.name}"
                    inputs["training"] = session.upload_file(str(local_path), data_s3_key)
                else:
                    data_s3_key = f"{s3_prefix}/data"
                    session.upload_directory(str(local_path), data_s3_key)
                    inputs["training"] = f"s3://{session.bucket}/{data_s3_key}"

        # Handle model checkpoint channel (for fine-tuning)
        model_cfg = self.training_config.get("model", {})
        base_model_path = model_cfg.get("base_model_path")

        if base_model_path:
            if base_model_path.startswith("s3://"):
                inputs["model"] = base_model_path
            else:
                local_path = Path(base_model_path)
                if local_path.exists():
                    model_s3_key = f"{s3_prefix}/model/{local_path.name}"
                    inputs["model"] = session.upload_file(str(local_path), model_s3_key)

        logger.info(f"Prepared {len(inputs)} input channels for training job")
        return inputs

    def _prepare_container_config(self) -> dict[str, Any]:
        """Prepare training config for container environment.

        Modifies paths to point to SageMaker container mount points.

        Returns:
            Modified training configuration.
        """
        import copy

        config = copy.deepcopy(self.training_config)

        # Update data paths for container
        # SageMaker downloads files into /opt/ml/input/data/{channel_name}/
        # If original was a file, we preserve the filename
        data_cfg = config.get("data", {})
        original_data_path = data_cfg.get("data_path")
        if original_data_path:
            if not original_data_path.startswith("s3://"):
                original_path = Path(original_data_path)
                if original_path.is_file():
                    # File is downloaded to /opt/ml/input/data/training/{filename}
                    data_cfg["data_path"] = f"/opt/ml/input/data/training/{original_path.name}"
                else:
                    # Directory contents are downloaded to /opt/ml/input/data/training/
                    data_cfg["data_path"] = "/opt/ml/input/data/training"
            else:
                # S3 paths - assume directory structure is preserved
                data_cfg["data_path"] = "/opt/ml/input/data/training"
            config["data"] = data_cfg

        # Update model paths for container
        model_cfg = config.get("model", {})
        original_model_path = model_cfg.get("base_model_path")
        if original_model_path:
            if not original_model_path.startswith("s3://"):
                original_path = Path(original_model_path)
                if original_path.is_file():
                    # Model file is downloaded to /opt/ml/input/data/model/{filename}
                    model_cfg["base_model_path"] = f"/opt/ml/input/data/model/{original_path.name}"
                else:
                    model_cfg["base_model_path"] = "/opt/ml/input/data/model"
            else:
                model_cfg["base_model_path"] = "/opt/ml/input/data/model"
            config["model"] = model_cfg

        # Update training settings for multi-node
        training_cfg = config.get("training", {})
        instance_count = self._config.instance.instance_count

        if instance_count > 1:
            training_cfg["num_nodes"] = instance_count
            if not training_cfg.get("strategy"):
                training_cfg["strategy"] = "ddp"
            config["training"] = training_cfg

        # Set project directory to SageMaker model artifacts path
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["project_dir"] = "/opt/ml/model"

        return config

    def get_environment(self) -> dict[str, str]:
        """Get environment variables for training container.

        Returns:
            Dictionary of environment variables.
        """
        env = {
            # Job identification
            "SOEN_JOB_TYPE": "training",
            "SOEN_JOB_NAME": self.job_name,
            "SOEN_BACKEND": self._config.backend.value,
            # Config path in container
            "SOEN_CONFIG_PATH": "/opt/ml/input/data/config/training_config.yaml",
            # Python path for soen_toolkit
            "PYTHONPATH": "/opt/ml/code:/opt/ml/code/src",
        }

        # MLflow configuration
        mlflow_cfg = self._config.mlflow
        if mlflow_cfg.tracking_uri:
            env["MLFLOW_TRACKING_URI"] = mlflow_cfg.tracking_uri
            env["MLFLOW_EXPERIMENT_NAME"] = mlflow_cfg.experiment_name
            if mlflow_cfg.run_name:
                env["MLFLOW_RUN_NAME"] = mlflow_cfg.run_name
            # Check training config for MLflow credentials
            if self.training_config:
                mlflow_password = self.training_config.get("mlflow_password")
                if mlflow_password:
                    env["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
                mlflow_user = self.training_config.get("mlflow_user")
                if mlflow_user:
                    env["MLFLOW_TRACKING_USERNAME"] = mlflow_user

        # Multi-node DDP settings
        instance_count = self._config.instance.instance_count
        if instance_count > 1:
            env.update({
                "NCCL_DEBUG": "WARN",
                "NCCL_IB_DISABLE": "1",
                "NCCL_P2P_DISABLE": "1",
                "NCCL_SOCKET_IFNAME": "eth0",
            })

        # Merge user-provided environment variables
        env.update(self._config.environment)

        return env

    def submit(self, session: CloudSession) -> str:
        """Submit training job to SageMaker.

        Args:
            session: Cloud session for AWS operations.

        Returns:
            Job name of the submitted training job.

        Raises:
            CloudSessionError: If submission fails.
        """
        import sagemaker
        from sagemaker.estimator import Estimator
        from sagemaker.inputs import TrainingInput

        logger.info(f"Submitting training job: {self.job_name}")

        # Validate and prepare
        self.validate()
        inputs = self.prepare_inputs(session)
        env = self.get_environment()
        docker_image = self.get_docker_image(session)

        # Validate Docker image
        session.validate_ecr_image(docker_image)

        # Create SageMaker session
        sm_session = sagemaker.Session(
            boto_session=session.boto_session,
            sagemaker_client=session.sagemaker_client(),
        )

        # Configure instance
        instance_cfg = self._config.instance
        max_run_seconds = int(instance_cfg.max_runtime_hours * 3600)

        # Create estimator
        estimator_kwargs: dict[str, Any] = {
            "image_uri": docker_image,
            "role": self._config.aws.role,
            "instance_count": instance_cfg.instance_count,
            "instance_type": instance_cfg.instance_type,
            "sagemaker_session": sm_session,
            "environment": env,
            "max_run": max_run_seconds,
            "use_spot_instances": instance_cfg.use_spot,
            "disable_profiler": True,
        }

        if instance_cfg.use_spot:
            # Max wait time for spot capacity (same as max run)
            estimator_kwargs["max_wait"] = max_run_seconds

        estimator = Estimator(**estimator_kwargs)

        # Prepare input channels
        sm_inputs = {}
        for channel_name, s3_uri in inputs.items():
            sm_inputs[channel_name] = TrainingInput(s3_uri)

        # Submit job
        estimator.fit(
            inputs=sm_inputs,
            job_name=self.job_name,
            wait=False,
            logs=None,
        )

        latest_job = estimator.latest_training_job
        if latest_job is None:
            msg = "No training job was created"
            raise RuntimeError(msg)
        actual_job_name = latest_job.name
        logger.info(f"Training job submitted: {actual_job_name}")

        return actual_job_name

    def submit_and_wait(
        self,
        session: CloudSession,
        poll_interval: int = 30,
    ) -> SubmittedJob:
        """Submit job and return handle for monitoring.

        Args:
            session: Cloud session for AWS operations.
            poll_interval: Seconds between status checks.

        Returns:
            SubmittedJob for monitoring and waiting.
        """
        job_name = self.submit(session)
        return SubmittedJob(job_name, session, self)

