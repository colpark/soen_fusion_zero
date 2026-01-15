"""Cloud Management GUI for SOEN Toolkit.

A graphical interface for:
- Configuring AWS credentials
- Submitting training/inference/processing jobs
- Monitoring job status
- Viewing cost estimates
- Managing cloud resources

Launch:
    python -m soen_toolkit.cloud_gui
"""

from .main import main

__all__ = ["main"]

