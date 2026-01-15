#!/usr/bin/env python3
"""Metric registry and an Objective composer that sums weighted costs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .metrics_builtin import (
    AutocorrMetric,
    AvalancheMetric,
    BranchingMetric,
    JacobianMetric,
    LyapunovMetric,
    SpatialPSDMetric,
    SusceptibilityMetric,
    TemporalPSDMetric,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .metrics_base import Metric

_FACTORIES: dict[str, Callable[..., Metric]] = {
    "branching": BranchingMetric,
    "psd_temporal": TemporalPSDMetric,
    "psd_spatial": SpatialPSDMetric,
    "susceptibility": SusceptibilityMetric,
    "avalanche": AvalancheMetric,
    "autocorr": AutocorrMetric,
    "jacobian": JacobianMetric,
    "lyapunov": LyapunovMetric,
}


def register_metric(name: str, factory: Callable[..., Metric]) -> None:
    _FACTORIES[str(name)] = factory


class Objective:
    def __init__(self, *, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        w = {
            "branching": float(self.config.get("w_branch", 1.0)),
            "psd_temporal": float(self.config.get("w_psd_t", 0.0)),
            "psd_spatial": float(self.config.get("w_psd_spatial", 0.0)),
            "susceptibility": float(self.config.get("w_chi_inv", 0.0)),
            "avalanche": float(self.config.get("w_avalanche", 0.0)),
            "autocorr": float(self.config.get("w_autocorr", 0.0)),
            "jacobian": float(self.config.get("w_jacobian", 0.0)),
            "lyapunov": float(self.config.get("w_lyapunov", 0.0)),
        }
        self._metrics: list[Metric] = [
            _FACTORIES["branching"](weight=w["branching"], target_layers=self.config.get("target_layers")),
            _FACTORIES["psd_temporal"](weight=w["psd_temporal"], target_beta=float(self.config.get("target_beta_t", 2.0))),
            _FACTORIES["psd_spatial"](weight=w["psd_spatial"], target_beta=float(self.config.get("target_beta_spatial", 2.0))),
            _FACTORIES["susceptibility"](weight=w["susceptibility"], target_layers=self.config.get("target_layers")),
            _FACTORIES["avalanche"](weight=w["avalanche"]),
            _FACTORIES["autocorr"](weight=w["autocorr"]),
            _FACTORIES["jacobian"](weight=w["jacobian"], target_layers=self.config.get("target_layers")),
            _FACTORIES["lyapunov"](weight=w["lyapunov"], target_layers=self.config.get("target_layers")),
        ]

    def evaluate(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        total = 0.0
        for m in self._metrics:
            res = m.compute(context, model, s_histories) or {}
            # Extract cost separately
            c = float(res.get("cost", 0.0))
            total += c
            # Merge without overwriting total
            for k, v in res.items():
                if k != "cost":
                    out[k] = float(v)
        out["total_cost"] = float(total) if total == total else float("nan")
        return out
