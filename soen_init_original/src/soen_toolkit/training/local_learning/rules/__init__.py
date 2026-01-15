"""Local learning rules."""

from soen_toolkit.training.local_learning.rules.base import AbstractLocalRule
from soen_toolkit.training.local_learning.rules.stdp import STDP
from soen_toolkit.training.local_learning.rules.three_factor import (
    RewardModulatedHebbianRule,
)
from soen_toolkit.training.local_learning.rules.two_factor import (
    BCMRule,
    HebbianRule,
    OjaRule,
)

__all__ = [
    "AbstractLocalRule",
    "HebbianRule",
    "OjaRule",
    "BCMRule",
    "STDP",
    "RewardModulatedHebbianRule",
]
