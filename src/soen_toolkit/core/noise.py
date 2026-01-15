from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .configs import NoiseConfig, PerturbationConfig

# Maybe we should make the bias current an extra, because not all layers have it. For now it is core, alongside g,phi, etc


class NoiseStrategy(ABC):
    """Base class for noise or perturbation strategies."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.offset(tensor)

    @abstractmethod
    def offset(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the noise/perturbation offset to add to ``tensor``."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state (for perturbations)."""


@dataclass(frozen=True)
class GaussianNoise(NoiseStrategy):
    std: float
    relative: bool = False

    def offset(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std == 0.0:
            return torch.zeros_like(tensor)
        scale = self.std * (tensor.abs() if self.relative else 1.0)
        return torch.randn_like(tensor) * scale


@dataclass(frozen=True)
class GaussianPerturbation(NoiseStrategy):
    """Deterministic offset drawn once per forward pass.

    The offset is sampled for each sample in the batch and for each node, then
    reused for all timesteps in the forward pass.
    """

    mean: float = 0.0
    std: float = 0.0
    _offset: torch.Tensor = field(default=None, init=False, repr=False, compare=False) # type: ignore[assignment]

    def offset(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._offset is None or self._offset.shape != tensor.shape:
            device = tensor.device
            dtype = tensor.dtype
            if tensor.dim() <= 1:
                shape = tensor.shape
            else:
                shape = (tensor.shape[0], tensor.shape[-1]) # type: ignore[assignment]

            if self.std == 0.0:
                per_sample = torch.full(shape, self.mean, device=device, dtype=dtype)
            else:
                per_sample = torch.randn(shape, device=device, dtype=dtype) * self.std + self.mean

            if tensor.dim() > 2:
                view_shape = (tensor.shape[0],) + (1,) * (tensor.dim() - 2) + (tensor.shape[-1],)
                offset = per_sample.view(view_shape).expand(tensor.shape)
            else:
                offset = per_sample

            object.__setattr__(self, "_offset", offset)

        return self._offset

    def reset(self) -> None:
        object.__setattr__(self, "_offset", None)


@dataclass(frozen=True)
class CompositeNoise(NoiseStrategy):
    """Combine multiple noise strategies by summing their offsets."""

    strategies: list[NoiseStrategy]

    def offset(self, tensor: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(tensor)
        for strat in self.strategies:
            total += strat.offset(tensor)
        return total

    def reset(self) -> None:
        for strat in self.strategies:
            if hasattr(strat, "reset"):
                strat.reset()


@dataclass(frozen=True)
class NoiseSettings:
    """Container mapping noise keys to strategy objects."""

    phi: NoiseStrategy | None = None
    g: NoiseStrategy | None = None
    s: NoiseStrategy | None = None
    bias_current: NoiseStrategy | None = None
    j: NoiseStrategy | None = None
    extra: dict[str, NoiseStrategy] = field(default_factory=dict)

    def apply(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        strat = getattr(self, key, None)
        if strat is None:
            strat = self.extra.get(key)
        if strat is None:
            return tensor
        return strat(tensor)

    def is_trivial(self) -> bool:
        for attr in ("phi", "g", "s", "bias_current", "j"):
            if getattr(self, attr) is not None:
                return False
        return not any(self.extra.values())

    def reset(self) -> None:
        for attr in ("phi", "g", "s", "bias_current", "j"):
            strat = getattr(self, attr)
            if strat and hasattr(strat, "reset"):
                strat.reset()
        for strat in self.extra.values():
            if hasattr(strat, "reset"):
                strat.reset()


def build_noise_strategies(
    noise: dict | NoiseConfig | NoiseSettings | None = None,
    perturb: dict | PerturbationConfig | None = None,
) -> NoiseSettings:
    """Convert config objects into :class:`NoiseSettings`."""
    if isinstance(noise, NoiseSettings) and perturb is None:
        return noise
    if isinstance(noise, tuple) and perturb is None:
        noise, perturb = noise

    noise_data: dict = {}
    pert_data: dict = {}

    if noise is not None:
        if hasattr(noise, "__dict__") and not isinstance(noise, dict):
            noise_data = noise.__dict__
        else:
            noise_data = dict(noise)  # type: ignore[arg-type]

    if perturb is not None:
        if hasattr(perturb, "__dict__") and not isinstance(perturb, dict):
            pert_data = perturb.__dict__
        else:
            pert_data = dict(perturb)  # type: ignore[arg-type]

    rel = noise_data.get("relative", False)

    if rel:
        has_pert = any(pert_data.get(f"{k}_std", 0.0) != 0.0 or pert_data.get(f"{k}_mean", 0.0) != 0.0 for k in ("phi", "g", "s", "bias_current", "j"))
        if has_pert:
            msg = "Relative scaling cannot be combined with perturbations"
            raise ValueError(msg)

    def make(key: str) -> NoiseStrategy | None:
        strategies: list[NoiseStrategy] = []

        val_n = noise_data.get(key, 0.0)
        if val_n != 0.0:
            strategies.append(GaussianNoise(val_n, rel))

        mean_p = pert_data.get(f"{key}_mean", 0.0)
        std_p = pert_data.get(f"{key}_std", 0.0)
        if mean_p != 0.0 or std_p != 0.0:
            strategies.append(GaussianPerturbation(mean_p, std_p))

        if not strategies:
            return None
        if len(strategies) == 1:
            return strategies[0]
        return CompositeNoise(strategies)

    base_settings = {
        "phi": make("phi"),
        "g": make("g"),
        "s": make("s"),
        "bias_current": make("bias_current"),
        "j": make("j"),
    }

    # Collect extra keys from provided dicts
    extra_noise = noise_data.get("extras", {}) if isinstance(noise_data.get("extras", {}), dict) else {}
    extra_pert_mean = pert_data.get("extras_mean", {}) if isinstance(pert_data.get("extras_mean", {}), dict) else {}
    extra_pert_std = pert_data.get("extras_std", {}) if isinstance(pert_data.get("extras_std", {}), dict) else {}

    # Include any unrecognized keys as extras
    known_noise = {"phi", "g", "s", "bias_current", "j", "relative", "extras"}
    known_pert_prefixes = {"phi", "g", "s", "bias_current", "j"}
    for k, v in noise_data.items():
        if k not in known_noise:
            extra_noise[k] = v
    for k, v in pert_data.items():
        if k not in {f"{p}_mean" for p in known_pert_prefixes} | {f"{p}_std" for p in known_pert_prefixes} | {"extras_mean", "extras_std"}:
            if k.endswith("_mean"):
                extra_pert_mean[k[:-5]] = v
            elif k.endswith("_std"):
                extra_pert_std[k[:-4]] = v
            else:
                extra_noise[k] = v

    extra_keys = set(extra_noise.keys()) | set(extra_pert_mean.keys()) | set(extra_pert_std.keys())
    extras: dict[str, NoiseStrategy] = {}
    for key in extra_keys:
        strategies: list[NoiseStrategy] = []
        val_n = extra_noise.get(key, 0.0)
        if val_n != 0.0:
            strategies.append(GaussianNoise(val_n, rel))
        mean_p = extra_pert_mean.get(key, 0.0)
        std_p = extra_pert_std.get(key, 0.0)
        if mean_p != 0.0 or std_p != 0.0:
            strategies.append(GaussianPerturbation(mean_p, std_p))
        if strategies:
            extras[key] = strategies[0] if len(strategies) == 1 else CompositeNoise(strategies)

    return NoiseSettings(
        phi=base_settings["phi"],
        g=base_settings["g"],
        s=base_settings["s"],
        bias_current=base_settings["bias_current"],
        j=base_settings["j"],
        extra=extras,
    )
