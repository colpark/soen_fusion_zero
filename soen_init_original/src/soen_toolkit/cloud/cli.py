"""Command-line interface for SOEN cloud operations.

Provides a clean CLI for submitting jobs, checking status, and
managing cloud resources.

Usage:
    soen cloud train --config experiment.yaml
    soen cloud train --config experiment.yaml --estimate
    soen cloud status <job-id>
    soen cloud list --project my-project
    soen cloud instances
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import click

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def setup_verbose_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("botocore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """SOEN Cloud - Submit and manage cloud training jobs."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_verbose_logging(verbose)


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to training configuration YAML",
)
@click.option(
    "--cloud-config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to cloud configuration YAML (optional, uses defaults if not provided)",
)
@click.option(
    "--estimate",
    is_flag=True,
    help="Show cost estimate without submitting",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for job completion",
)
@click.option(
    "--instance-type",
    default=None,
    help="Override instance type (e.g., ml.g5.xlarge)",
)
@click.option(
    "--instance-count",
    default=None,
    type=int,
    help="Override instance count",
)
@click.option(
    "--spot/--on-demand",
    default=True,
    help="Use spot instances (default) or on-demand",
)
@click.option(
    "--backend",
    type=click.Choice(["pytorch", "jax"]),
    default="pytorch",
    help="ML framework backend",
)
@click.option(
    "--job-name",
    default=None,
    help="Custom job name (auto-generated if not provided)",
)
@click.pass_context
def train(
    ctx: click.Context,
    config: Path,
    cloud_config: Path | None,
    estimate: bool,
    wait: bool,
    instance_type: str | None,
    instance_count: int | None,
    spot: bool,
    backend: str,
    job_name: str | None,
) -> None:
    """Submit a training job to SageMaker.

    Examples:

        # Basic training job
        soen cloud train --config experiment.yaml

        # Show cost estimate first
        soen cloud train --config experiment.yaml --estimate

        # Use specific instance
        soen cloud train --config experiment.yaml --instance-type ml.g5.2xlarge

        # Multi-node training
        soen cloud train --config experiment.yaml --instance-count 2

        # JAX backend
        soen cloud train --config experiment.yaml --backend jax
    """
    from .config import (
        AWSConfig,
        Backend,
        CloudConfig,
        InstanceConfig,
        JobConfig,
        JobType,
        MLflowConfig,
        load_config,
    )
    from .cost import CostEstimator
    from .jobs import TrainingJob
    from .session import CloudSession

    click.echo(f"Training config: {config}")

    # Load or create cloud config
    if cloud_config:
        click.echo(f"Cloud config: {cloud_config}")
        cloud_cfg = load_config(cloud_config)
    else:
        # Try to load from environment or defaults
        import os

        role = os.environ.get("SOEN_SM_ROLE")
        bucket = os.environ.get("SOEN_SM_BUCKET")

        if not role or not bucket:
            click.echo(
                click.style(
                    "Error: No cloud config provided and SOEN_SM_ROLE/SOEN_SM_BUCKET not set.\n"
                    "Either provide --cloud-config or set environment variables.",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        cloud_cfg = CloudConfig(
            aws=AWSConfig(
                role=role,
                bucket=bucket,
                region=os.environ.get("AWS_REGION", "us-east-1"),
            ),
            mlflow=MLflowConfig(
                tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            ),
        )

    # Build instance config with overrides
    instance_cfg = InstanceConfig(
        instance_type=instance_type or cloud_cfg.instance.instance_type,
        instance_count=instance_count or cloud_cfg.instance.instance_count,
        use_spot=spot,
        max_runtime_hours=cloud_cfg.instance.max_runtime_hours,
    )

    # Build job config
    backend_enum = Backend.PYTORCH if backend == "pytorch" else Backend.JAX

    job_cfg = JobConfig(
        job_type=JobType.TRAINING,
        job_name=job_name,
        backend=backend_enum,
        aws=cloud_cfg.aws,
        instance=instance_cfg,
        mlflow=cloud_cfg.mlflow,
        training_config_path=config,
        docker_image=cloud_cfg.get_docker_image(backend_enum),
    )

    # Create job
    job = TrainingJob(job_cfg)

    # Show cost estimate
    estimator = CostEstimator(region=cloud_cfg.aws.region)
    cost = estimator.estimate(job_cfg)

    click.echo("")
    click.echo(cost.format())
    click.echo("")

    if estimate:
        click.echo("Use --no-estimate to submit the job.")
        return

    # Validate job
    try:
        job.validate()
    except (ValueError, FileNotFoundError) as e:
        click.echo(click.style(f"Validation error: {e}", fg="red"), err=True)
        sys.exit(1)

    # Confirm submission
    if not click.confirm("Submit training job?"):
        click.echo("Cancelled.")
        return

    # Create session and submit
    try:
        session = CloudSession(cloud_cfg)
    except Exception as e:
        click.echo(click.style(f"AWS session error: {e}", fg="red"), err=True)
        sys.exit(1)

    try:
        job_name = job.submit(session)
        click.echo(click.style(f"\nJob submitted: {job_name}", fg="green"))

        # Show useful links
        region = cloud_cfg.aws.region
        console_url = (
            f"https://{region}.console.aws.amazon.com/sagemaker/home"
            f"?region={region}#/jobs/{job_name}"
        )
        click.echo(f"Console: {console_url}")

        if wait:
            click.echo("\nWaiting for job completion...")
            from .jobs.base import SubmittedJob

            submitted = SubmittedJob(job_name, session, job)
            result = submitted.wait(poll_interval=30)

            if result.succeeded:
                click.echo(click.style("\nJob completed successfully!", fg="green"))
                if result.output_path:
                    click.echo(f"Output: {result.output_path}")
            else:
                click.echo(click.style(f"\nJob failed: {result.status}", fg="red"))
                if result.error_message:
                    click.echo(f"Error: {result.error_message}")
                sys.exit(1)

    except Exception as e:
        click.echo(click.style(f"Submission error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.argument("job_name")
@click.option(
    "--cloud-config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to cloud configuration YAML",
)
@click.pass_context
def status(
    ctx: click.Context,
    job_name: str,
    cloud_config: Path | None,
) -> None:
    """Check status of a training job.

    Examples:

        soen cloud status soen-training-20240101120000
    """
    import os

    import boto3

    # Get region from config or environment
    region = os.environ.get("AWS_REGION", "us-east-1")

    if cloud_config:
        from .config import load_config

        cloud_cfg = load_config(cloud_config)
        region = cloud_cfg.aws.region

    try:
        sm = boto3.client("sagemaker", region_name=region)
        response = sm.describe_training_job(TrainingJobName=job_name)

        # Header
        click.echo("=" * 60)
        click.echo(f"Job: {job_name}")
        click.echo("=" * 60)

        # Status
        status = response.get("TrainingJobStatus", "Unknown")
        secondary = response.get("SecondaryStatus", "N/A")
        status_color = {
            "Completed": "green",
            "Failed": "red",
            "Stopped": "yellow",
            "InProgress": "blue",
        }.get(status, "white")
        click.echo(f"Status: {click.style(status, fg=status_color)} ({secondary})")

        if response.get("FailureReason"):
            click.echo(click.style(f"Failure: {response['FailureReason']}", fg="red"))

        # Instance info
        click.echo("")
        click.echo("Instance:")
        resource_cfg = response.get("ResourceConfig", {})
        click.echo(f"  Type: {resource_cfg.get('InstanceType', 'N/A')}")
        click.echo(f"  Count: {resource_cfg.get('InstanceCount', 'N/A')}")
        click.echo(f"  Volume: {resource_cfg.get('VolumeSizeInGB', 'N/A')} GB")

        # Spot instance info
        if response.get("EnableManagedSpotTraining"):
            click.echo("  Spot: Yes")

        # Timing info
        click.echo("")
        click.echo("Timing:")
        if response.get("CreationTime"):
            click.echo(f"  Created: {response['CreationTime']}")
        if response.get("TrainingStartTime"):
            click.echo(f"  Started: {response['TrainingStartTime']}")
        if response.get("TrainingEndTime"):
            click.echo(f"  Ended: {response['TrainingEndTime']}")

        billable = response.get("BillableTimeInSeconds", 0)
        if billable:
            minutes = billable / 60
            click.echo(f"  Billable: {minutes:.1f} minutes")

        # Cost estimate
        instance_type = resource_cfg.get("InstanceType", "")
        if billable and instance_type:
            # Rough cost estimate
            hourly_prices = {
                "ml.g5.xlarge": 1.006,
                "ml.g5.2xlarge": 1.515,
                "ml.g5.4xlarge": 2.27,
                "ml.p3.2xlarge": 3.825,
            }
            hourly = hourly_prices.get(instance_type, 1.0)
            cost = (billable / 3600) * hourly
            if response.get("EnableManagedSpotTraining"):
                cost *= 0.35  # Spot discount
            click.echo(f"  Est. Cost: ${cost:.2f}")

        # Docker image
        click.echo("")
        click.echo(f"Image: {response.get('AlgorithmSpecification', {}).get('TrainingImage', 'N/A')}")

        # Artifacts
        artifacts = response.get("ModelArtifacts", {})
        if artifacts.get("S3ModelArtifacts"):
            click.echo("")
            click.echo(f"Output: {artifacts['S3ModelArtifacts']}")

        # Input channels
        input_config = response.get("InputDataConfig", [])
        if input_config:
            click.echo("")
            click.echo("Input Channels:")
            for channel in input_config:
                name = channel.get("ChannelName", "unknown")
                uri = channel.get("DataSource", {}).get("S3DataSource", {}).get("S3Uri", "N/A")
                click.echo(f"  {name}: {uri}")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command("list")
@click.option(
    "--project",
    default=None,
    help="Filter by project name",
)
@click.option(
    "--limit",
    default=10,
    type=int,
    help="Maximum number of jobs to show",
)
@click.option(
    "--cloud-config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to cloud configuration YAML",
)
@click.pass_context
def list_jobs(
    ctx: click.Context,
    project: str | None,
    limit: int,
    cloud_config: Path | None,
) -> None:
    """List recent training jobs.

    Examples:

        soen cloud list
        soen cloud list --project my-project --limit 20
    """
    import os

    import boto3

    from .config import load_config

    # Get region from config or environment
    if cloud_config:
        cloud_cfg = load_config(cloud_config)
        region = cloud_cfg.aws.region
    else:
        region = os.environ.get("AWS_REGION", "us-east-1")

    try:
        sm = boto3.client("sagemaker", region_name=region)

        # Build filters
        filters = []
        if project:
            filters.append({
                "Name": "TrainingJobName",
                "Operator": "Contains",
                "Value": project,
            })

        kwargs = {"MaxResults": min(limit, 100), "SortBy": "CreationTime", "SortOrder": "Descending"}
        if filters:
            kwargs["Filters"] = filters

        response = sm.list_training_jobs(**kwargs)
        jobs = response.get("TrainingJobSummaries", [])

        if not jobs:
            click.echo("No training jobs found.")
            return

        # Format output
        click.echo(f"{'Job Name':<50} {'Status':<12} {'Created'}")
        click.echo("-" * 80)

        for job in jobs[:limit]:
            name = job.get("TrainingJobName", "")[:48]
            status = job.get("TrainingJobStatus", "Unknown")
            created = job.get("CreationTime", "")
            if created:
                created = created.strftime("%Y-%m-%d %H:%M")

            # Color code status
            if status == "Completed":
                status_str = click.style(status, fg="green")
            elif status == "Failed":
                status_str = click.style(status, fg="red")
            elif status == "InProgress":
                status_str = click.style(status, fg="yellow")
            else:
                status_str = status

            click.echo(f"{name:<50} {status_str:<21} {created}")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--gpu-only",
    is_flag=True,
    default=True,
    help="Show only GPU instances (default)",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Show all instance types including CPU-only",
)
@click.pass_context
def instances(ctx: click.Context, gpu_only: bool, show_all: bool) -> None:
    """List available SageMaker instance types with pricing.

    Examples:

        soen cloud instances
        soen cloud instances --all
    """
    from .cost import format_instance_table

    # Pass gpu_only=False if --all is set
    click.echo(format_instance_table())


@cli.command()
@click.argument("job_name")
@click.option(
    "--cloud-config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to cloud configuration YAML",
)
@click.pass_context
def stop(
    ctx: click.Context,
    job_name: str,
    cloud_config: Path | None,
) -> None:
    """Stop a running training job.

    Examples:

        soen cloud stop soen-training-20240101120000
    """
    import os

    import boto3

    from .config import load_config

    # Get region from config or environment
    if cloud_config:
        cloud_cfg = load_config(cloud_config)
        region = cloud_cfg.aws.region
    else:
        region = os.environ.get("AWS_REGION", "us-east-1")

    if not click.confirm(f"Stop job {job_name}?"):
        click.echo("Cancelled.")
        return

    try:
        sm = boto3.client("sagemaker", region_name=region)
        sm.stop_training_job(TrainingJobName=job_name)
        click.echo(click.style(f"Stop request sent for: {job_name}", fg="yellow"))

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()

