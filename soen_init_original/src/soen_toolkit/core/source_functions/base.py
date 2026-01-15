# FILEPATH: src/soen_toolkit/core/source_functions/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch


@dataclass(slots=True)
class SourceFunctionInfo:
    key: str
    title: str
    description: str
    category: str
    tags: tuple[str, ...] = ()
    uses_squid_current: bool = False
    supports_coefficients: bool = True


class SourceFunctionBase(ABC):
    info: SourceFunctionInfo | None = None
    _abstract = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "_abstract", False) or inspect.isabstract(cls):
            return
        if getattr(cls, "info", None) is None:
            msg = f"{cls.__name__} must define an 'info' attribute"
            raise TypeError(msg)

    @abstractmethod
    def g(self, phi: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def uses_squid_current(self) -> bool:
        info = getattr(self, "info", None)
        return bool(getattr(info, "uses_squid_current", False))

    @property
    def uses_effective_bias(self) -> bool:
        """Deprecated: Use uses_squid_current instead."""
        return self.uses_squid_current

    def get_coefficients(
        self,
        phi: torch.Tensor,
        gamma_plus: torch.Tensor,
        gamma_minus: torch.Tensor,
        dt: float,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def default_coefficients(
    phi: torch.Tensor,
    gamma_plus: torch.Tensor,
    gamma_minus: torch.Tensor,
    dt: float,
    g_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_t = 1.0 - dt * gamma_minus.unsqueeze(1).expand(-1, phi.shape[1], -1)
    b_t = dt * gamma_plus.unsqueeze(1).expand(-1, phi.shape[1], -1) * g_values
    return a_t, b_t


F = TypeVar("F", bound="SourceFunctionBase")


def build_source(fn: type[F], *args, **kwargs) -> F:  # noqa: UP047
    return fn(*args, **kwargs)


def bulk_register(classes: Iterable[type[SourceFunctionBase]]) -> dict[str, type[SourceFunctionBase]]:
    registry: dict[str, type[SourceFunctionBase]] = {}
    for cls in classes:
        key = cls.info.key # type: ignore[union-attr]
        if key in registry:
            msg = f"Duplicate source function key '{key}'"
            raise ValueError(msg)
        registry[key] = cls
    return registry
