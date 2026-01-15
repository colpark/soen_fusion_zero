#!/usr/bin/env python3
"""Built-in metrics adapting existing functions behind small Metric classes.

Each returns a dict of outputs including a single 'cost' key used by Objective.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .metrics_base import Metric


class BranchingMetric(Metric):
    def __init__(self, *, weight: float = 1.0, target_layers: list[int] | None = None) -> None:
        self.weight = float(weight)
        self.target_layers = target_layers

    def compute(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]:
        # Import from new metrics core module
        from soen_toolkit.utils.hpo.study.objective.core import (
            compute_branching_ratio,
            criticality_branching_cost,
        )

        if self.weight <= 0.0:
            return {"branching_sigma": float("nan"), "branching_cost": float("nan"), "branching_cost_v2": float("nan"), "cost": 0.0}
        sigma = compute_branching_ratio(model, s_histories, target_layers=self.target_layers)
        cost_simple = (1.0 - sigma) ** 2
        cost_v2 = criticality_branching_cost(sigma)
        return {
            "branching_sigma": float(sigma),
            "branching_cost": float(cost_simple),
            "branching_cost_v2": float(cost_v2),
            # Weight the improved v2 cost to prefer its gradients while keeping both fields for analysis
            "cost": float(self.weight * cost_v2),
        }


class TemporalPSDMetric(Metric):
    def __init__(self, *, weight: float = 0.0, target_beta: float = 2.0) -> None:
        self.weight = float(weight)
        self.target_beta = float(target_beta)

    def compute(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]:
        from soen_toolkit.utils.hpo.study.objective.core import (
            temporal_psd_beta_and_cost,
        )

        if self.weight <= 0.0:
            return {"beta_temporal": float("nan"), "psd_temporal_cost": float("nan"), "cost": 0.0}
        beta_t, cost_t = temporal_psd_beta_and_cost(s_histories, float(context.get("dt_s", 0.0)), target_beta=self.target_beta)
        return {"beta_temporal": float(beta_t), "psd_temporal_cost": float(cost_t), "cost": float(self.weight * cost_t)}


class SpatialPSDMetric(Metric):
    def __init__(self, *, weight: float = 0.0, target_beta: float = 2.0) -> None:
        self.weight = float(weight)
        self.target_beta = float(target_beta)

    def compute(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]:
        from soen_toolkit.utils.hpo.study.objective.core import (
            spatial_psd_beta_for_layer,
        )

        if self.weight <= 0.0:
            return {"beta_spatial_mean": float("nan"), "psd_spatial_cost": float("nan"), "cost": 0.0}
        betas, costs = [], []
        for s in s_histories:
            b, c = spatial_psd_beta_for_layer(s, target_beta=self.target_beta)
            if not math.isnan(c):
                betas.append(b)
                costs.append(c)
        beta_mean = float(np.mean(betas)) if betas else float("nan")
        cost_mean = float(np.mean(costs)) if costs else float("nan")
        return {"beta_spatial_mean": beta_mean, "psd_spatial_cost": cost_mean, "cost": float(self.weight * cost_mean) if not math.isnan(cost_mean) else 0.0}


class SusceptibilityMetric(Metric):
    def __init__(self, *, weight: float = 0.0, target_layers: list[int] | None = None) -> None:
        self.weight = float(weight)
        self.target_layers = target_layers

    def compute(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]:
        from soen_toolkit.utils.hpo.study.objective.core import susceptibility_variance

        if self.weight <= 0.0:
            return {"chi_variance": float("nan"), "chi_inv_cost": float("nan"), "cost": 0.0}
        chi = susceptibility_variance(model, s_histories, layers=self.target_layers)
        inv_cost = (1.0 / chi) if (chi and chi > 0 and not math.isnan(chi)) else float("nan")
        return {"chi_variance": float(chi), "chi_inv_cost": float(inv_cost), "cost": float(self.weight * inv_cost) if not math.isnan(inv_cost) else 0.0}


class AvalancheMetric(Metric):
    def __init__(self, *, weight: float = 0.0) -> None:
        self.weight = float(weight)

    def compute(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]:
        from soen_toolkit.utils.hpo.study.objective.core import (
            criticality_avalanche_size_cost,
        )

        if self.weight <= 0.0:
            return {"avalanche_cost": float("nan"), "cost": 0.0}
        c = criticality_avalanche_size_cost(s_histories)
        return {"avalanche_cost": float(c), "cost": float(self.weight * c) if not math.isnan(c) else 0.0}


class AutocorrMetric(Metric):
    def __init__(self, *, weight: float = 0.0) -> None:
        self.weight = float(weight)

    def compute(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]:
        from soen_toolkit.utils.hpo.study.objective.core import (
            criticality_autocorr_cost,
        )

        if self.weight <= 0.0:
            return {"autocorr_cost": float("nan"), "cost": 0.0}
        c = criticality_autocorr_cost(s_histories, float(context.get("dt_s", 0.0)))
        return {"autocorr_cost": float(c), "cost": float(self.weight * c) if not math.isnan(c) else 0.0}


class JacobianMetric(Metric):
    def __init__(self, *, weight: float = 0.0, target_layers: list[int] | None = None) -> None:
        self.weight = float(weight)
        self.target_layers = target_layers

    def compute(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]:
        from soen_toolkit.utils.hpo.study.objective.core import jacobian_spectral_radius

        if self.weight <= 0.0:
            return {"jac_spectral_radius": float("nan"), "jacobian_cost": float("nan"), "cost": 0.0}
        rho, info = jacobian_spectral_radius(model, s_histories, target_layers=self.target_layers)
        cost = abs(rho - 1.0) if (rho == rho) else float("nan")
        out = {"jac_spectral_radius": float(rho), "jacobian_cost": float(cost), "cost": float(self.weight * cost) if not math.isnan(cost) else 0.0}
        # Surface diagnostics for underdetermined/NaN cases
        try:
            if isinstance(info, dict):
                for k, v in info.items():
                    out[f"jacobian_info_{k}"] = float(v) if isinstance(v, (int, float)) else v
        except Exception:
            pass
        return out


class LyapunovMetric(Metric):
    def __init__(self, *, weight: float = 0.0, target_layers: list[int] | None = None) -> None:
        self.weight = float(weight)
        self.target_layers = target_layers

    def compute(self, context: dict[str, Any], model: Any, s_histories: Any) -> dict[str, float]:
        from soen_toolkit.utils.hpo.study.objective.core import (
            lyapunov_largest_exponent,
        )

        if self.weight <= 0.0:
            return {"lyap_per_step": float("nan"), "lyap_per_sec": float("nan"), "lyapunov_cost": float("nan"), "cost": 0.0}
        dt_s = float(context.get("dt_s", 0.0))
        lam_step, lam_sec = lyapunov_largest_exponent(model, s_histories, dt_s, target_layers=self.target_layers)
        # Cost: pseudo-Huber centered at 0 on per-step exponent
        # c(λ) = δ^2 (sqrt(1 + (λ/δ)^2) − 1)
        if lam_step != lam_step:  # NaN guard
            cost = float("nan")
        else:
            delta = float(context.get("lyap_delta_per_step", 0.01))
            x = float(lam_step)
            cost = (delta * delta) * (math.sqrt(1.0 + (x / max(delta, 1e-12)) ** 2) - 1.0)
        return {
            "lyap_per_step": float(lam_step),
            "lyap_per_sec": float(lam_sec),
            "lyapunov_cost": float(cost) if cost == cost else float("nan"),
            "cost": float(self.weight * cost) if (cost == cost) else 0.0,
        }
