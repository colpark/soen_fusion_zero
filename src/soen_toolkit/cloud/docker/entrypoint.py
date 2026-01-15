#!/usr/bin/env python
"""Container entrypoint for SOEN cloud jobs.

This is a clean, focused entrypoint that:
1. Sets up MLflow tracking (if configured)
2. Dispatches to the appropriate job handler (training/inference/processing)
3. Handles errors and ensures logs are captured

Environment variables:
    SOEN_JOB_TYPE: Job type (training, inference, processing)
    SOEN_CONFIG_PATH: Path to job configuration YAML
    SOEN_BACKEND: ML backend (pytorch, jax)
    MLFLOW_TRACKING_URI: MLflow tracking server URI (optional)
    MLFLOW_EXPERIMENT_NAME: MLflow experiment name (optional)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import traceback

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("soen.entrypoint")


def setup_mlflow() -> None:
    """Configure MLflow from environment variables."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.info("MLflow tracking not configured (MLFLOW_TRACKING_URI not set)")
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")

        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "soen-cloud")
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")

        # Start run if run name is provided
        run_name = os.environ.get("MLFLOW_RUN_NAME")
        if run_name:
            mlflow.start_run(run_name=run_name)
            logger.info(f"MLflow run started: {run_name}")

        # Log environment info
        mlflow.log_params({
            "job_type": os.environ.get("SOEN_JOB_TYPE", "unknown"),
            "backend": os.environ.get("SOEN_BACKEND", "unknown"),
            "job_name": os.environ.get("SOEN_JOB_NAME", "unknown"),
        })

    except ImportError:
        logger.warning("mlflow not installed, skipping tracking setup")
    except Exception as e:
        logger.warning(f"Failed to setup MLflow: {e}")


def log_environment() -> None:
    """Log relevant environment information for debugging."""
    logger.info("=" * 60)
    logger.info("SOEN Cloud Job Starting")
    logger.info("=" * 60)

    # Log key environment variables
    env_vars = [
        "SOEN_JOB_TYPE",
        "SOEN_CONFIG_PATH",
        "SOEN_BACKEND",
        "SOEN_JOB_NAME",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
    ]
    for var in env_vars:
        value = os.environ.get(var, "<not set>")
        logger.info(f"  {var}: {value}")

    # Log Python info
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  Working directory: {os.getcwd()}")

    # Log GPU info for debugging
    backend = os.environ.get("SOEN_BACKEND", "").lower()
    if backend == "jax":
        try:
            import jax
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == "gpu"]
            logger.info(f"  JAX devices: {len(devices)} total, {len(gpu_devices)} GPU")
            for d in devices:
                logger.info(f"    - {d}")
        except Exception as e:
            logger.warning(f"  Could not query JAX devices: {e}")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"  CUDA available: {torch.cuda.device_count()} device(s)")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                logger.info("  CUDA: not available")
        except Exception as e:
            logger.warning(f"  Could not query CUDA devices: {e}")

    logger.info("=" * 60)


def run_training(config_path: str) -> None:
    """Run training job.

    Args:
        config_path: Path to training configuration YAML.
    """
    logger.info(f"Starting training with config: {config_path}")

    # Import here to catch import errors with full traceback
    try:
        from soen_toolkit.training.trainers.experiment import run_from_config
    except Exception as e:
        logger.error(f"Failed to import training module: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Run training with explicit error handling
    try:
        run_from_config(config_path, script_dir=Path.cwd())
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        # Print to stderr so it appears in CloudWatch even if file logging is active
        traceback.print_exc()
        raise


def run_inference(config_path: str) -> None:
    """Run batch inference job.

    Args:
        config_path: Path to inference configuration YAML.
    """
    logger.info(f"Starting inference with config: {config_path}")

    # TODO: Implement batch inference
    # from soen_toolkit.inference import run_batch_inference
    # run_batch_inference(config_path)

    raise NotImplementedError(
        "Batch inference not yet implemented. "
        "Use training job type for now."
    )


def run_processing(config_path: str) -> None:
    """Run data processing job.

    Args:
        config_path: Path to processing configuration YAML.
    """
    logger.info(f"Starting processing with config: {config_path}")

    # TODO: Implement data processing
    # from soen_toolkit.processing import run_processing
    # run_processing(config_path)

    raise NotImplementedError(
        "Data processing not yet implemented. "
        "Use training job type for now."
    )


def main() -> int:
    """Main entrypoint function.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Log environment for debugging
    log_environment()

    # Get job configuration
    job_type = os.environ.get("SOEN_JOB_TYPE", "training").lower()
    config_path = os.environ.get("SOEN_CONFIG_PATH")

    # Validate config path
    if not config_path:
        # Try to find config in standard SageMaker location
        standard_paths = [
            "/opt/ml/input/data/config/training_config.yaml",
            "/opt/ml/input/data/config/config.yaml",
            "/opt/ml/input/config.yaml",
        ]
        for path in standard_paths:
            if os.path.exists(path):
                config_path = path
                logger.info(f"Found config at standard location: {config_path}")
                break

    if not config_path:
        logger.error(
            "No configuration file found. Set SOEN_CONFIG_PATH or place "
            "config in /opt/ml/input/data/config/training_config.yaml"
        )
        return 1

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    # Setup MLflow tracking
    setup_mlflow()

    # Dispatch to job handler
    try:
        if job_type == "training":
            run_training(config_path)
        elif job_type == "inference":
            run_inference(config_path)
        elif job_type == "processing":
            run_processing(config_path)
        else:
            logger.error(f"Unknown job type: {job_type}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Job failed with error: {e}")
        logger.error(traceback.format_exc())

        # Try to log error to MLflow
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_param("error", str(e)[:250])  # Truncate long errors
                mlflow.set_tag("status", "failed")
        except Exception:
            pass

        return 1

    finally:
        # End MLflow run if active
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

