"""MLflow utilities for SOEN Toolkit.

This package provides scripts and utilities for setting up and managing
a shared MLflow tracking server on AWS for the SOEN team.

Key components:
- setup-mlflow-server.sh: Creates AWS infrastructure and MLflow server
- cleanup-mlflow-server.sh: Removes all created resources
- README.md: Complete setup and usage guide
- example_training_config.yaml: Example configuration for training

The MLflow integration is already built into the training pipeline via
the SafeMLFlowLogger in ExperimentRunner.
"""

__version__ = "1.0.0"
