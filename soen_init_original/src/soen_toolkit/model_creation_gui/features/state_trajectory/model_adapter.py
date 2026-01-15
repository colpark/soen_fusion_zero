"""Model adapter: façade for model operations, isolating implementation details."""

from __future__ import annotations

import torch

from soen_toolkit.physics.constants import DEFAULT_IC, DEFAULT_PHI0, get_omega_c
from soen_toolkit.utils.power_tracking import convert_energy_to_physical, convert_power_to_physical


class ModelAdapter:
    """Façade for model operations.

    Encapsulates model-specific details and provides a stable interface for
    simulation backends. This isolates the controller and backends from model
    implementation quirks.
    """

    def set_dt(self, model, dt: float) -> None:
        """Set model time step (dimensionless)."""
        model.set_dt(dt)

    def enable_full_tracking(self, model) -> None:
        """Enable tracking of all metrics (phi, g, s, power)."""
        model.set_tracking(track_phi=True, track_g=True, track_s=True, track_power=True)

    def get_tracking_flags(self, model) -> dict[str, bool]:
        """Get current tracking flags."""
        cfg = model.sim_config
        return {
            "track_phi": getattr(cfg, "track_phi", False),
            "track_g": getattr(cfg, "track_g", False),
            "track_s": getattr(cfg, "track_s", False),
            "track_power": getattr(cfg, "track_power", False),
        }

    def set_tracking_flags(self, model, flags: dict[str, bool]) -> None:
        """Set tracking flags."""
        model.set_tracking(**flags)

    def restore_tracking(self, model, sim_config) -> None:
        """Restore tracking settings from simulation config."""
        model.set_tracking(
            track_phi=getattr(sim_config, "track_phi", False),
            track_g=getattr(sim_config, "track_g", False),
            track_s=getattr(sim_config, "track_s", False),
            track_power=getattr(sim_config, "track_power", False),
        )

    def reset_state(self, model) -> None:
        """Reset model state and clear histories if available."""
        for layer in model.layers:
            if hasattr(layer, "_clear_histories"):
                try:
                    layer._clear_histories()
                except Exception:
                    pass

    def collect_metric_histories(
        self,
        model,
        metric: str,
        include_s0: bool,
        raw_state_histories: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Extract histories for the selected metric.

        Args:
            model: Model instance
            metric: Metric name (state, phi, g, power, energy)
            include_s0: For state metric, whether to include initial state
            raw_state_histories: Per-layer state histories [batch, T+1, dim]

        Returns:
            List of history tensors per layer, format depends on metric
        """
        metric_lower = metric.lower()

        if metric_lower.startswith("state") or metric_lower == "s":
            # For state, use raw histories and optionally drop initial state
            if include_s0:
                return raw_state_histories
            else:
                return [h[:, 1:, :] for h in raw_state_histories]

        elif metric_lower.startswith("phi") or metric_lower.startswith("flux"):
            return model.get_phi_history()

        elif metric_lower.startswith("g") or metric_lower.startswith("non"):
            return model.get_g_history()

        elif metric_lower.startswith("power"):
            return self._collect_power_histories(model)

        elif metric_lower.startswith("energy"):
            return self._collect_energy_histories(model)

        else:
            # Unknown metric, return empty list
            return []

    def _collect_power_histories(self, model) -> list[torch.Tensor]:
        """Collect power histories from layers and convert to physical units (nW)."""
        histories = []
        for lyr in model.layers:
            dimless_bias = getattr(lyr, "power_bias_dimensionless", None)
            dimless_diss = getattr(lyr, "power_diss_dimensionless", None)

            if dimless_bias is None or dimless_diss is None:
                histories.append(None)
                continue

            dimless_total = dimless_bias + dimless_diss

            # Get physical parameters
            Phi0 = getattr(lyr, "Phi0", getattr(lyr, "PHI0", DEFAULT_PHI0))
            Ic = getattr(lyr, "Ic", getattr(lyr, "IC", DEFAULT_IC))
            wc = getattr(lyr, "wc", getattr(lyr, "WC", float(get_omega_c())))

            # Convert to physical units (Watts) then to nW
            phys_total = convert_power_to_physical(dimless_total, Ic, Phi0, wc) * 1e9
            histories.append(phys_total)

        return histories

    def _collect_energy_histories(self, model) -> list[torch.Tensor]:
        """Collect energy histories from layers and convert to physical units (nJ)."""
        histories = []
        for lyr in model.layers:
            eb = getattr(lyr, "energy_bias_dimensionless", None)
            ed = getattr(lyr, "energy_diss_dimensionless", None)

            if eb is None or ed is None:
                histories.append(None)
                continue

            energy_total = eb + ed

            # Get physical parameters
            Phi0 = getattr(lyr, "Phi0", getattr(lyr, "PHI0", DEFAULT_PHI0))
            Ic = getattr(lyr, "Ic", getattr(lyr, "IC", DEFAULT_IC))

            # Convert to physical units (Joules) then to nJ
            phys_total = convert_energy_to_physical(energy_total, Ic, Phi0) * 1e9
            histories.append(phys_total)

        return histories
