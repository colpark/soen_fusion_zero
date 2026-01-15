# FILEPATH: src/soen_toolkit/core/mixins/config_update.py

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS

if TYPE_CHECKING:
    from soen_toolkit.core.configs import LayerConfig, SimulationConfig


class ConfigUpdateMixin:
    """Mixin providing configuration update methods."""

    if TYPE_CHECKING:
        # Attributes expected from the composed class
        layers_config: list[LayerConfig]
        sim_config: SimulationConfig
        layers: nn.ModuleList
        dt: float | nn.Parameter
    def update_layer_source_function(self, layer_id: int, new_source_func: str) -> None:
        for idx, cfg in enumerate(self.layers_config):
            if cfg.layer_id == layer_id:
                cfg.params["source_func"] = new_source_func
                if new_source_func not in SOURCE_FUNCTIONS:
                    msg = f"Unknown source function: {new_source_func}"
                    raise ValueError(msg)
                new_func = SOURCE_FUNCTIONS[new_source_func]()
                self.layers[idx].source_function = new_func
                return
        msg = f"Layer with id {layer_id} not found."
        raise ValueError(msg)

    def update_noise_config(self, layer_id: int, new_noise_params: dict) -> None:
        for _idx, cfg in enumerate(self.layers_config):
            if cfg.layer_id == layer_id:
                for key, value in new_noise_params.items():
                    if hasattr(cfg.noise, key):
                        setattr(cfg.noise, key, value)
                    else:
                        msg = f"Unknown noise parameter: {key}"
                        raise ValueError(msg)
                return
        msg = f"Layer with id {layer_id} not found."
        raise ValueError(msg)

    def set_dt(self, new_dt: float | torch.Tensor, propagate_to_layers: bool = True) -> None:
        """Set the global dt.

        - If model.dt is a Parameter, update its data in-place safely.
        - Otherwise set a scalar value.
        - Propagate by binding layers to the shared dt when tensor‑based,
          or assigning a local scalar when not.
        """
        if isinstance(self.dt, torch.nn.Parameter):
            with torch.no_grad():
                self.dt.data = torch.tensor(float(new_dt), dtype=self.dt.dtype, device=self.dt.device)
        else:
            self.dt = float(new_dt)
        self.sim_config.dt = float(new_dt)
        if propagate_to_layers:
            for layer in self.layers:
                try:
                    if hasattr(layer, "set_dt_reference") and isinstance(self.dt, torch.Tensor):
                        layer.set_dt_reference(self.dt)
                    else:
                        # fall back to assigning a scalar value
                        layer.set_dt_reference(None) if hasattr(layer, "set_dt_reference") else None
                        layer.dt = float(self.sim_config.dt)
                except Exception:
                    pass

    def set_dt_learnable(self, learnable: bool, propagate_to_layers: bool = True) -> None:
        """Toggle whether dt is a single learnable parameter shared by all layers."""
        if learnable:
            if not isinstance(self.dt, torch.nn.Parameter):
                self.dt = nn.Parameter(torch.tensor(float(self.dt), dtype=torch.float32), requires_grad=True)
            else:
                self.dt.requires_grad_(True)
        elif isinstance(self.dt, torch.nn.Parameter):
            self.dt = float(self.dt.detach().item())
        self.sim_config.dt_learnable = learnable
        if propagate_to_layers:
            for layer in self.layers:
                try:
                    if hasattr(layer, "set_dt_reference") and isinstance(self.dt, torch.Tensor):
                        # Bind shared dt reference when learnable
                        layer.set_dt_reference(self.dt)
                    else:
                        # Clear binding and assign scalar value
                        layer.set_dt_reference(None) if hasattr(layer, "set_dt_reference") else None
                        layer.dt = float(self.sim_config.dt)
                except Exception:
                    pass

    def set_tracking(
        self,
        *,
        track_phi: bool | None = None,
        track_power: bool | None = None,
        track_g: bool | None = None,
        track_s: bool | None = None,
    ) -> None:
        if track_phi is not None:
            self.sim_config.track_phi = track_phi
            for layer in self.layers:
                layer.track_phi = track_phi
                if not track_phi:
                    layer.phi_input_history = []
        if track_power is not None:
            self.sim_config.track_power = track_power
            for layer in self.layers:
                # Only enable power tracking for SingleDendrite layers
                layer_type = getattr(layer, "layer_type", layer.__class__.__name__)
                if track_power and layer_type != "SingleDendrite":
                    import warnings

                    warnings.warn(
                        f"Power tracking is only supported for SingleDendrite layers, not {layer_type}. Skipping power tracking for layer {getattr(layer, 'layer_id', 'unknown')}.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                if hasattr(layer, "set_tracking_flags"):
                    try:
                        layer.set_tracking_flags(power=track_power)
                    except Exception:
                        layer.track_power = track_power
                else:
                    layer.track_power = track_power
        if track_g is not None:
            self.sim_config.track_g = track_g
            for layer in self.layers:
                layer.track_g = track_g
        if track_s is not None:
            self.sim_config.track_s = track_s
            for layer in self.layers:
                layer.track_s = track_s

    def get_phi_history(self):
        histories = []
        for layer in self.layers:
            histories.append(layer.get_phi_history())
        return histories

    # (Removed initial state helpers; initial conditions are supplied per-forward only.)

    def get_g_history(self):
        histories = []
        for layer in self.layers:
            histories.append(layer.get_g_history())
        return histories

    def get_state_history(self):
        histories = []
        for layer in self.layers:
            histories.append(layer.get_state_history())
        return histories

    def get_power_history(self):
        histories = []
        for layer in self.layers:
            pb_phys = getattr(layer, "power_bias", None)
            pd_phys = getattr(layer, "power_diss", None)
            if pb_phys is not None and pd_phys is not None:
                try:
                    total = pb_phys + pd_phys
                except Exception:
                    total = None
                histories.append(total)
            else:
                histories.append(None)
        return histories
