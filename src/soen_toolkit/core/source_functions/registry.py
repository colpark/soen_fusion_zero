# FILEPATH: src/soen_toolkit/core/source_functions/registry.py

from __future__ import annotations

from typing import TYPE_CHECKING

from .analytic import (
    ReLUSourceFunction,
    SimpleGELUSourceFunction,
    TanhGauss1p7IBFitSourceFunction,
    TanhSourceFunction,
    TeLUSourceFunction,
)
from .heaviside import HeavisideFitStateDep
from .rate_array import RateArraySource

if TYPE_CHECKING:
    from .base import SourceFunctionBase, SourceFunctionInfo

ALL_SOURCES: tuple[type[SourceFunctionBase], ...] = (
    ReLUSourceFunction,
    SimpleGELUSourceFunction,
    TanhSourceFunction,
    TeLUSourceFunction,
    TanhGauss1p7IBFitSourceFunction,
    HeavisideFitStateDep,
    RateArraySource,
)

SOURCE_FUNCTIONS: dict[str, type[SourceFunctionBase]] = {}
for cls in ALL_SOURCES:
    SOURCE_FUNCTIONS[cls.info.key] = cls

# Compatibility aliases
SOURCE_FUNCTIONS["Heaviside_fit_state_dep"] = HeavisideFitStateDep


SOURCE_FUNCTION_CATALOG: dict[str, SourceFunctionInfo] = {cls.info.key: cls.info for cls in ALL_SOURCES}

# Default source functions for layer types in the GUI
SOURCE_FUNCTION_DEFAULTS = {
    "SingleDendrite": "RateArray",
    "Multiplier": "RateArray",
    "MultiplierNOCC": "RateArray",
    "DendriteReadout": "RateArray",
    "NonLinear": "Tanh",
}


def group_source_functions_by_category() -> dict[str, tuple[SourceFunctionInfo, ...]]:
    groups: dict[str, list[SourceFunctionInfo]] = {}
    for info in SOURCE_FUNCTION_CATALOG.values():
        groups.setdefault(info.category, []).append(info)
    return {k: tuple(sorted(v, key=lambda i: i.title)) for k, v in groups.items()}
