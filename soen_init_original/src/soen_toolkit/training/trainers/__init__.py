# FILEPATH: src/soen_toolkit/training/trainers/__init__.py

"""Training utilities for SOEN models.

This module provides experiment runners and training utilities.
"""

from .experiment import ExperimentRunner, run_from_config

__all__ = ["ExperimentRunner", "run_from_config"]
