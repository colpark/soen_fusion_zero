from collections.abc import Callable

import torch
import torch.nn.functional as F

# Import your loss functions from loss_functions.py
from .loss_functions import (
    autoregressive_cross_entropy,
    autoregressive_loss,
    average_state_magnitude_loss,
    branching_loss,
    ensure_post_sample_decay,
    exp_high_state_penalty,
    final_timestep_zero_mse,
    gap_loss,
    get_off_the_ground_loss,
    gravity_quantization_loss,
    local_expansion_loss,
    mse,
    reg_J_loss,
    rich_margin_loss,
    top_gap_loss,
)

LOSS_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {}


def register_loss(name: str) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        LOSS_REGISTRY[name] = fn
        return fn

    return decorator


register_loss("reg_J_loss")(reg_J_loss)
register_loss("get_off_the_ground_loss")(get_off_the_ground_loss)
register_loss("top_gap_loss")(top_gap_loss)
register_loss("exp_high_state_penalty")(exp_high_state_penalty)
register_loss("gap_loss")(gap_loss)
register_loss("rich_margin_loss")(rich_margin_loss)
register_loss("autoregressive_loss")(autoregressive_loss)
register_loss("autoregressive_cross_entropy")(autoregressive_cross_entropy)
register_loss("gravity_quantization_loss")(gravity_quantization_loss)
register_loss("branching_loss")(branching_loss)
register_loss("final_timestep_zero_mse")(final_timestep_zero_mse)
register_loss("ensure_post_sample_decay")(ensure_post_sample_decay)
register_loss("local_expansion_loss")(local_expansion_loss)
register_loss("average_state_magnitude_loss")(average_state_magnitude_loss)

# -----------------------------------------------------------------------------
# Functional cross-entropy wrapper so it can be used as an additional loss
# -----------------------------------------------------------------------------


def cross_entropy(outputs, targets):
    """CrossEntropyLoss that handles both static and sequence targets.

    For sequence targets (2D), reshapes inputs and targets appropriately.
    For static targets (1D), uses standard cross entropy.
    """
    if targets.dim() == 2:
        # Sequence classification: reshape [B, T, C] -> [B*T, C] and [B, T] -> [B*T]
        B, T = targets.shape
        if outputs.dim() == 3:
            # outputs: [B, T, C] -> [B*T, C]
            outputs = outputs.reshape(B * T, -1)
        targets = targets.reshape(B * T)

    return F.cross_entropy(outputs, targets)


# -----------------------------------------------------------------------------
register_loss("cross_entropy")(cross_entropy)
register_loss("mse")(mse)

# Stub loss (returns 0) to allow neutral base when using new 'losses' style


def stub_loss(**kwargs):
    return torch.tensor(0.0)


register_loss("stub")(stub_loss)

__all__ = [
    "LOSS_REGISTRY",
    "autoregressive_cross_entropy",
    "autoregressive_loss",
    "average_state_magnitude_loss",
    "cross_entropy",
    "ensure_post_sample_decay",
    "exp_high_state_penalty",
    "final_timestep_zero_mse",
    "gap_loss",  # this coupled with CE works well
    "get_off_the_ground_loss",
    "local_expansion_loss",
    "mse",
    "reg_J_loss",
    "register_loss",
    "rich_margin_loss",  # this, even alone works well!
    "top_gap_loss",  # can remove
]
