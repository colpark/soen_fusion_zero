from __future__ import annotations

"""Input provider registry for HPO objectives.

This light-weight layer makes input generation pluggable and keeps the
optuna runner agnostic to the specific input kind.
"""

from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch


class InputProvider:
    """Interface for generating batched input sequences for trials.

    Implementations should avoid heavy side-effects in __init__. Use
    only cached/prepared state and return a torch.Tensor shaped [B, T, D]
    in get_batch.
    """

    def __init__(self, config: dict, base_spec: dict, device: torch.device) -> None:
        self.config = config
        self.base_spec = base_spec
        self.device = device

    def get_batch(self, *, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        raise NotImplementedError


_REGISTRY: dict[str, Callable[[dict, dict, torch.device], InputProvider]] = {}


def register(name: str, factory: Callable[[dict, dict, torch.device], InputProvider]) -> None:
    _REGISTRY[name] = factory


def create(name: str, config: dict, base_spec: dict, device: torch.device) -> InputProvider | None:
    fn = _REGISTRY.get(name)
    if fn is None:
        return None
    return fn(config, base_spec, device)


# Register builtins
from .builtin import (  # noqa: E402
    ColoredNoiseInput,
    GPRBFInput,
    HDF5DatasetInput,
    LogSlopeNoiseInput,
    WhiteNoiseInput,
)

register("white_noise", lambda cfg, spec, dev: WhiteNoiseInput(cfg, spec, dev))
register("colored_noise", lambda cfg, spec, dev: ColoredNoiseInput(cfg, spec, dev))
register("gp_rbf", lambda cfg, spec, dev: GPRBFInput(cfg, spec, dev))
register("log_slope_noise", lambda cfg, spec, dev: LogSlopeNoiseInput(cfg, spec, dev))
register("hdf5_dataset", lambda cfg, spec, dev: HDF5DatasetInput(cfg, spec, dev))
