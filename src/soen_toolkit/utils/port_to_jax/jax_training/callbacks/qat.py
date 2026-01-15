from __future__ import annotations

from dataclasses import dataclass
from typing import Any

"""Quantization-Aware Training (QAT) utilities for the JAX backend.

For the JAX trainer, we implement a lightweight STE-based snapping of
connection parameters to a uniform codebook during the forward pass, while
passing gradients straight-through. This mirrors the Torch path semantics at a
high-level for connection matrices.
"""


def _calc_num_levels(bits: int | None, levels: int | None) -> int:
    assert (bits is None) ^ (levels is None), "Specify either bits or levels (exclusively)."
    if levels is not None:
        return int(levels)
    b = int(bits) if bits is not None else 0
    assert b >= 0
    # include zero level for parity with Torch path
    return (2**b) + 1


@dataclass
class QATStraightThroughJAX:
    min_val: float
    max_val: float
    bits: int | None = None
    levels: int | None = None
    connections: list[str] | None = None  # not used in JAX path presently
    update_on_train_epoch_start: bool = False
    stochastic_rounding: bool = False  # not used in JAX path presently

    def enable(self, trainer: Any) -> None:
        """Enable QAT on the low-level JAX trainer by setting a codebook."""
        num_levels = _calc_num_levels(self.bits, self.levels)
        trainer.enable_qat_ste_jax(
            min_val=float(self.min_val),
            max_val=float(self.max_val),
            num_levels=int(num_levels),
        )

    def disable(self, trainer: Any) -> None:
        trainer.disable_qat_ste_jax()


__all__ = ["QATStraightThroughJAX"]
