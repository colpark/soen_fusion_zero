# FILEPATH: src/soen_toolkit/training/distillation/__init__.py

"""Distillation training module for SOEN models.

This module provides utilities for knowledge distillation, where a student model
learns to match the output state trajectories of a teacher model.
"""

from .teacher_data_generator import generate_teacher_trajectories

__all__ = [
    "generate_teacher_trajectories",
]
