"""Zero padding operations for input sequences."""

from __future__ import annotations

import torch

from .settings import TimeMode, ZeroPaddingSpec
from .timebase import Timebase


def _steps_from_spec(spec: ZeroPaddingSpec, time_mode: TimeMode, tb: Timebase) -> int:
    """Convert padding specification to number of steps based on time mode.

    Args:
        spec: Padding specification
        time_mode: Current time mode (dt or total_ns)
        tb: Timebase for time conversions

    Returns:
        Number of steps to pad (0 if disabled)
    """
    if not spec.enabled:
        return 0

    if time_mode == TimeMode.DT:
        return spec.count_steps
    else:
        # Total time mode: convert nanoseconds to steps
        if tb.step_ns > 0:
            return int(round(spec.time_ns / tb.step_ns))
        return 0


def apply_prepend(x: torch.Tensor, spec: ZeroPaddingSpec, time_mode: TimeMode, tb: Timebase) -> torch.Tensor:
    """Prepend zeros to input sequence.

    Args:
        x: Input tensor of shape [seq_len, dim]
        spec: Prepend padding specification
        time_mode: Current time mode
        tb: Timebase for conversions

    Returns:
        Padded tensor of shape [seq_len + k, dim] where k is the number of prepended steps
    """
    k = _steps_from_spec(spec, time_mode, tb)
    if k <= 0:
        return x

    # Ensure x is 2D
    if x.ndim == 1:
        x = x.unsqueeze(1)

    dim = x.shape[1]
    pad = torch.zeros((k, dim), dtype=x.dtype, device=x.device)
    return torch.cat([pad, x], dim=0)


def apply_append(x: torch.Tensor, spec: ZeroPaddingSpec, time_mode: TimeMode, tb: Timebase) -> torch.Tensor:
    """Append zeros or held values to input sequence.

    Args:
        x: Input tensor of shape [seq_len, dim]
        spec: Append padding specification
        time_mode: Current time mode
        tb: Timebase for conversions

    Returns:
        Padded tensor of shape [seq_len + k, dim] where k is the number of appended steps
    """
    k = _steps_from_spec(spec, time_mode, tb)
    if k <= 0:
        return x

    # Ensure x is 2D
    if x.ndim == 1:
        x = x.unsqueeze(1)

    dim = x.shape[1]

    if spec.mode == "hold_last":
        # Repeat the last value
        pad = x[-1:].repeat(k, 1)
    else:
        # Append zeros
        pad = torch.zeros((k, dim), dtype=x.dtype, device=x.device)

    return torch.cat([x, pad], dim=0)
