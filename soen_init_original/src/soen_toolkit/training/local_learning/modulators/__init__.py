"""Modulator sources for 3-factor learning rules."""

from soen_toolkit.training.local_learning.modulators.base import AbstractModulator
from soen_toolkit.training.local_learning.modulators.error_based import (
    CrossEntropyErrorModulator,
    MSEErrorModulator,
)

__all__ = [
    "AbstractModulator",
    "MSEErrorModulator",
    "CrossEntropyErrorModulator",
]
