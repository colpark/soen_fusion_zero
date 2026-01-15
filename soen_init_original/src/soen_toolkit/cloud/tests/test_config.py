"""Tests for cloud configuration validation."""

from pydantic import ValidationError
import pytest

from soen_toolkit.cloud.config import (
    AWSConfig,
    Backend,
    CloudConfig,
    InstanceConfig,
    JobConfig,
    JobType,
    MLflowConfig,
)


class TestAWSConfig:
    """Test AWS configuration validation."""

    def test_valid_config(self):
        """Test valid AWS configuration."""
        config = AWSConfig(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            bucket="my-training-bucket",
            region="us-east-1",
        )
        assert config.role.startswith("arn:aws:iam::")
        assert config.bucket == "my-training-bucket"
        assert config.region == "us-east-1"

    def test_invalid_role_arn(self):
        """Test that invalid role ARN raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AWSConfig(
                role="not-a-valid-arn",
                bucket="my-bucket",
            )
        assert "must be a valid IAM ARN" in str(exc_info.value)

    def test_invalid_role_format(self):
        """Test that malformed role ARN raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AWSConfig(
                role="arn:aws:iam::invalid:role/",
                bucket="my-bucket",
            )
        assert "ARN format invalid" in str(exc_info.value)

    def test_invalid_bucket_name_too_short(self):
        """Test that bucket name must be 3+ characters."""
        with pytest.raises(ValidationError) as exc_info:
            AWSConfig(
                role="arn:aws:iam::123456789012:role/TestRole",
                bucket="ab",
            )
        assert "3-63 characters" in str(exc_info.value)

    def test_invalid_bucket_name_characters(self):
        """Test that bucket name must have valid characters."""
        with pytest.raises(ValidationError) as exc_info:
            AWSConfig(
                role="arn:aws:iam::123456789012:role/TestRole",
                bucket="My_Bucket",  # Uppercase and underscore not allowed
            )
        assert "lowercase letters" in str(exc_info.value)

    def test_invalid_region(self):
        """Test that unsupported region raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AWSConfig(
                role="arn:aws:iam::123456789012:role/TestRole",
                bucket="my-bucket",
                region="invalid-region-1",
            )
        assert "not in supported regions" in str(exc_info.value)


class TestInstanceConfig:
    """Test instance configuration validation."""

    def test_default_values(self):
        """Test default instance configuration."""
        config = InstanceConfig()
        assert config.instance_type == "ml.g5.xlarge"
        assert config.instance_count == 1
        assert config.use_spot is True
        assert config.max_runtime_hours == 24.0

    def test_valid_gpu_instance(self):
        """Test valid GPU instance type."""
        config = InstanceConfig(instance_type="ml.g5.2xlarge")
        assert config.instance_type == "ml.g5.2xlarge"

    def test_valid_cpu_instance_c7i(self):
        """Test valid CPU instance type (compute-optimized c7i)."""
        config = InstanceConfig(instance_type="ml.c7i.4xlarge")
        assert config.instance_type == "ml.c7i.4xlarge"

    def test_valid_cpu_instance_m7i(self):
        """Test valid CPU instance type (general-purpose m7i)."""
        config = InstanceConfig(instance_type="ml.m7i.xlarge")
        assert config.instance_type == "ml.m7i.xlarge"

    def test_valid_cpu_instance_c6i(self):
        """Test valid CPU instance type (compute-optimized c6i)."""
        config = InstanceConfig(instance_type="ml.c6i.2xlarge")
        assert config.instance_type == "ml.c6i.2xlarge"

    def test_invalid_instance_type_prefix(self):
        """Test that non-ml. instance types raise error."""
        with pytest.raises(ValidationError) as exc_info:
            InstanceConfig(instance_type="g5.xlarge")  # Missing ml. prefix
        assert "must start with 'ml.'" in str(exc_info.value)

    def test_invalid_instance_count_zero(self):
        """Test that instance count must be >= 1."""
        with pytest.raises(ValidationError):
            InstanceConfig(instance_count=0)

    def test_invalid_instance_count_too_high(self):
        """Test that instance count must be <= 32."""
        with pytest.raises(ValidationError):
            InstanceConfig(instance_count=100)

    def test_invalid_runtime_zero(self):
        """Test that runtime must be > 0."""
        with pytest.raises(ValidationError):
            InstanceConfig(max_runtime_hours=0)

    def test_invalid_runtime_too_high(self):
        """Test that runtime must be <= 72 hours."""
        with pytest.raises(ValidationError):
            InstanceConfig(max_runtime_hours=100)


class TestMLflowConfig:
    """Test MLflow configuration validation."""

    def test_default_values(self):
        """Test default MLflow configuration."""
        config = MLflowConfig()
        assert config.tracking_uri is None
        assert config.experiment_name == "soen-cloud"
        assert config.run_name is None
        assert config.tags == {}

    def test_valid_http_uri(self):
        """Test valid HTTP tracking URI."""
        config = MLflowConfig(tracking_uri="http://mlflow.example.com:5000")
        assert config.tracking_uri == "http://mlflow.example.com:5000"

    def test_valid_https_uri(self):
        """Test valid HTTPS tracking URI."""
        config = MLflowConfig(tracking_uri="https://mlflow.example.com")
        assert config.tracking_uri == "https://mlflow.example.com"

    def test_valid_s3_uri(self):
        """Test valid S3 tracking URI."""
        config = MLflowConfig(tracking_uri="s3://my-bucket/mlflow")
        assert config.tracking_uri == "s3://my-bucket/mlflow"

    def test_invalid_uri_scheme(self):
        """Test that invalid URI scheme raises error."""
        with pytest.raises(ValidationError) as exc_info:
            MLflowConfig(tracking_uri="file:///local/path")
        assert "http://, https://, or s3://" in str(exc_info.value)


class TestCloudConfig:
    """Test complete cloud configuration."""

    def test_valid_config(self):
        """Test valid complete configuration."""
        config = CloudConfig(
            aws=AWSConfig(
                role="arn:aws:iam::123456789012:role/SageMakerRole",
                bucket="my-bucket",
            ),
            project="my-project",
            experiment="experiment-1",
        )
        assert config.project == "my-project"
        assert config.experiment == "experiment-1"

    def test_invalid_project_name(self):
        """Test that invalid project name raises error."""
        with pytest.raises(ValidationError) as exc_info:
            CloudConfig(
                aws=AWSConfig(
                    role="arn:aws:iam::123456789012:role/SageMakerRole",
                    bucket="my-bucket",
                ),
                project="my project!",  # Space and ! not allowed
            )
        assert "alphanumeric" in str(exc_info.value)

    def test_docker_image_lookup(self):
        """Test Docker image lookup by backend."""
        config = CloudConfig(
            aws=AWSConfig(
                role="arn:aws:iam::123456789012:role/SageMakerRole",
                bucket="my-bucket",
            ),
            docker_images={
                "pytorch": "123456789012.dkr.ecr.us-east-1.amazonaws.com/soen:pytorch",
                "jax": "123456789012.dkr.ecr.us-east-1.amazonaws.com/soen:jax",
            },
        )
        assert config.get_docker_image(Backend.PYTORCH) is not None
        assert config.get_docker_image(Backend.JAX) is not None


class TestJobConfig:
    """Test job configuration validation."""

    def test_training_job_requires_config_path(self):
        """Test that training job requires training_config_path."""
        with pytest.raises(ValidationError) as exc_info:
            JobConfig(
                job_type=JobType.TRAINING,
                aws=AWSConfig(
                    role="arn:aws:iam::123456789012:role/SageMakerRole",
                    bucket="my-bucket",
                ),
                # Missing training_config_path
            )
        assert "training_config_path" in str(exc_info.value)

    def test_inference_job_requires_model_path(self):
        """Test that inference job requires model_path."""
        from soen_toolkit.cloud.config import DataConfig

        with pytest.raises(ValidationError) as exc_info:
            JobConfig(
                job_type=JobType.INFERENCE,
                aws=AWSConfig(
                    role="arn:aws:iam::123456789012:role/SageMakerRole",
                    bucket="my-bucket",
                ),
                data=DataConfig(),  # Missing model_path
            )
        assert "model_path" in str(exc_info.value)

    def test_valid_job_name(self):
        """Test valid job name format."""
        from pathlib import Path

        config = JobConfig(
            job_type=JobType.TRAINING,
            job_name="soen-training-20240101",
            aws=AWSConfig(
                role="arn:aws:iam::123456789012:role/SageMakerRole",
                bucket="my-bucket",
            ),
            training_config_path=Path("/tmp/config.yaml"),
        )
        assert config.job_name == "soen-training-20240101"

    def test_invalid_job_name_too_long(self):
        """Test that job name > 63 chars raises error."""
        from pathlib import Path

        with pytest.raises(ValidationError) as exc_info:
            JobConfig(
                job_type=JobType.TRAINING,
                job_name="a" * 64,  # Too long
                aws=AWSConfig(
                    role="arn:aws:iam::123456789012:role/SageMakerRole",
                    bucket="my-bucket",
                ),
                training_config_path=Path("/tmp/config.yaml"),
            )
        assert "63 characters" in str(exc_info.value)

