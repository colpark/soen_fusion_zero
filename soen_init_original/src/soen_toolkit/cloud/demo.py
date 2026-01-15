#!/usr/bin/env python
"""Demo script showing cloud infrastructure usage.

This script demonstrates the cloud module's capabilities without
requiring AWS credentials (uses --estimate mode).

Run:
    python -m soen_toolkit.cloud.demo
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import yaml


def _output(message: str = "") -> None:
    """Write a message directly to stdout (avoids the print() lint)."""
    sys.stdout.write(f"{message}\n")


def create_demo_configs() -> tuple[Path, Path]:
    """Create temporary demo configuration files."""
    # Create training config
    training_config = {
        "training": {
            "max_epochs": 10,
            "batch_size": 32,
            "accelerator": "gpu",
            "devices": 1,
        },
        "data": {
            "data_path": "s3://my-bucket/datasets/mnist.h5",
        },
        "model": {
            "name": "SimpleNet",
            "hidden_size": 128,
        },
        "logging": {
            "project_name": "demo-project",
            "experiment_name": "demo-experiment",
        },
    }

    # Create cloud config
    cloud_config = {
        "aws": {
            "role": "arn:aws:iam::123456789012:role/SageMakerRole",
            "bucket": "demo-training-bucket",
            "region": "us-east-1",
        },
        "instance": {
            "instance_type": "ml.g5.xlarge",
            "instance_count": 1,
            "use_spot": True,
            "max_runtime_hours": 4.0,
        },
        "mlflow": {
            "tracking_uri": "http://mlflow.example.com:5000",
            "experiment_name": "demo-experiment",
        },
        "project": "demo-project",
        "experiment": "demo-run",
        "docker_images": {
            "pytorch": "123456789012.dkr.ecr.us-east-1.amazonaws.com/soen:pytorch",
            "jax": "123456789012.dkr.ecr.us-east-1.amazonaws.com/soen:jax",
        },
    }

    # Write to temp files
    tmpdir = Path(tempfile.mkdtemp(prefix="soen_cloud_demo_"))

    training_path = tmpdir / "training_config.yaml"
    with open(training_path, "w") as f:
        yaml.safe_dump(training_config, f)

    cloud_path = tmpdir / "cloud_config.yaml"
    with open(cloud_path, "w") as f:
        yaml.safe_dump(cloud_config, f)

    return training_path, cloud_path


def demo_config_loading() -> None:
    """Demonstrate configuration loading and validation."""
    _output("\n" + "=" * 60)
    _output("DEMO: Configuration Loading and Validation")
    _output("=" * 60)

    from soen_toolkit.cloud.config import AWSConfig, CloudConfig, InstanceConfig, load_config

    # Create config programmatically
    _output("\n1. Creating configuration programmatically:")
    config = CloudConfig(
        aws=AWSConfig(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            bucket="my-bucket",
            region="us-east-1",
        ),
        instance=InstanceConfig(
            instance_type="ml.g5.2xlarge",
            instance_count=2,
            use_spot=True,
        ),
        project="my-project",
        experiment="experiment-001",
    )
    _output(f"   AWS Role: {config.aws.role}")
    _output(f"   Instance: {config.instance.instance_type} x {config.instance.instance_count}")
    _output(f"   Spot: {config.instance.use_spot}")

    # Load from file
    _output("\n2. Loading configuration from YAML file:")
    _, cloud_path = create_demo_configs()
    loaded_config = load_config(cloud_path)
    _output(f"   Project: {loaded_config.project}")
    _output(f"   Experiment: {loaded_config.experiment}")

    # Validation demo
    _output("\n3. Validation (fail-fast on errors):")
    try:
        AWSConfig(role="invalid-arn", bucket="x")
    except Exception as e:
        _output(f"   Caught validation error: {type(e).__name__}")
        _output(f"   Message: {str(e)[:100]}...")


def demo_cost_estimation() -> None:
    """Demonstrate cost estimation."""
    _output("\n" + "=" * 60)
    _output("DEMO: Cost Estimation")
    _output("=" * 60)

    from soen_toolkit.cloud.cost import CostEstimator

    estimator = CostEstimator()

    # Single instance estimate
    _output("\n1. Single instance cost estimate:")
    estimate = estimator.estimate_from_params(
        instance_type="ml.g5.xlarge",
        instance_count=1,
        max_runtime_hours=8.0,
        use_spot=True,
    )
    _output(estimate.format())

    # Multi-instance estimate
    _output("\n2. Multi-instance cost estimate (4x ml.p4d.24xlarge):")
    estimate = estimator.estimate_from_params(
        instance_type="ml.p4d.24xlarge",
        instance_count=4,
        max_runtime_hours=24.0,
        use_spot=True,
    )
    _output(estimate.format())

    # Instance comparison
    _output("\n3. GPU Instance Pricing Comparison:")
    instances = estimator.list_instances(gpu_only=True)[:5]
    _output(f"   {'Instance':<20} {'On-Demand':<12} {'Spot (~65% off)'}")
    _output(f"   {'-'*20} {'-'*12} {'-'*15}")
    for instance_type, price in instances:
        spot = price * 0.35
        _output(f"   {instance_type:<20} ${price:<11.3f} ${spot:.3f}")


def demo_job_creation() -> None:
    """Demonstrate job creation."""
    _output("\n" + "=" * 60)
    _output("DEMO: Training Job Creation")
    _output("=" * 60)

    from soen_toolkit.cloud.config import (
        AWSConfig,
        Backend,
        InstanceConfig,
        JobConfig,
        JobType,
        MLflowConfig,
    )
    from soen_toolkit.cloud.jobs import TrainingJob

    training_path, _ = create_demo_configs()

    # Create job configuration
    job_config = JobConfig(
        job_type=JobType.TRAINING,
        backend=Backend.PYTORCH,
        aws=AWSConfig(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            bucket="my-bucket",
        ),
        instance=InstanceConfig(
            instance_type="ml.g5.xlarge",
            instance_count=1,
            use_spot=True,
            max_runtime_hours=8.0,
        ),
        mlflow=MLflowConfig(
            tracking_uri="http://mlflow.example.com:5000",
            experiment_name="demo",
        ),
        training_config_path=training_path,
        docker_image="123456789012.dkr.ecr.us-east-1.amazonaws.com/soen:pytorch",
    )

    # Create job
    job = TrainingJob(job_config)

    _output(f"\n1. Job Name: {job.job_name}")
    _output(f"   Backend: {job.backend.value}")
    _output(f"   Training Config: {job.training_config_path}")

    # Cost estimate
    _output("\n2. Cost Estimate:")
    estimate = job.estimate_cost()
    _output(estimate.format())

    # Environment variables
    _output("\n3. Container Environment Variables:")
    env = job.get_environment()
    for key, value in sorted(env.items()):
        _output(f"   {key}={value}")


def demo_cli() -> None:
    """Demonstrate CLI usage."""
    _output("\n" + "=" * 60)
    _output("DEMO: CLI Commands")
    _output("=" * 60)

    _output("\n1. Available CLI commands:")
    _output("   python -m soen_toolkit.cloud --help")
    _output("   python -m soen_toolkit.cloud train --help")
    _output("   python -m soen_toolkit.cloud status --help")
    _output("   python -m soen_toolkit.cloud list --help")
    _output("   python -m soen_toolkit.cloud instances")

    _output("\n2. Example usage:")
    _output("   # Submit a training job")
    _output("   python -m soen_toolkit.cloud train --config experiment.yaml --cloud-config cloud.yaml")
    _output("")
    _output("   # Show cost estimate without submitting")
    _output("   python -m soen_toolkit.cloud train --config experiment.yaml --cloud-config cloud.yaml --estimate")
    _output("")
    _output("   # Use JAX backend")
    _output("   python -m soen_toolkit.cloud train --config experiment.yaml --cloud-config cloud.yaml --backend jax")
    _output("")
    _output("   # Multi-node training")
    _output("   python -m soen_toolkit.cloud train --config experiment.yaml --cloud-config cloud.yaml --instance-count 2")


def main() -> None:
    """Run all demos."""
    _output("\n" + "#" * 60)
    _output("#" + " " * 15 + "SOEN CLOUD DEMO" + " " * 16 + "#")
    _output("#" * 60)

    demo_config_loading()
    demo_cost_estimation()
    demo_job_creation()
    demo_cli()

    _output("\n" + "=" * 60)
    _output("DEMO COMPLETE")
    _output("=" * 60)
    _output("\nTo run the tests:")
    _output("  pytest src/soen_toolkit/cloud/tests/ -v")
    _output("\nTo see instance pricing:")
    _output("  python -m soen_toolkit.cloud instances")
    _output("")


if __name__ == "__main__":
    main()

