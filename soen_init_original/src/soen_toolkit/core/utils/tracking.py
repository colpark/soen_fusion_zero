from __future__ import annotations

import contextlib
import warnings

import torch


def begin_step_accumulators(layers_config):
    lids = [cfg.layer_id for cfg in layers_config]
    return (
        {lid: [] for lid in lids},  # phi_accum
        {lid: [] for lid in lids},  # g_accum
        {lid: [] for lid in lids},  # pb_accum
        {lid: [] for lid in lids},  # pd_accum
    )


def collect_step_for_layer(lid: int, layer_mod, sim_config, phi_accum, g_accum, pb_accum, pd_accum) -> None:
    # φ
    if getattr(sim_config, "track_phi", False):
        try:
            phi_step_hist = layer_mod.get_phi_history()
            if phi_step_hist is not None and phi_step_hist.dim() == 3 and phi_step_hist.shape[1] >= 1:
                phi_accum[lid].append(phi_step_hist[:, -1, :])
        except Exception:
            pass
    # g
    if getattr(sim_config, "track_g", False):
        try:
            g_step_hist = layer_mod.get_g_history()
            if g_step_hist is not None and g_step_hist.dim() == 3 and g_step_hist.shape[1] >= 1:
                g_accum[lid].append(g_step_hist[:, -1, :])
        except Exception:
            pass
    # power
    if getattr(sim_config, "track_power", False) and getattr(layer_mod, "track_power", False):
        try:
            pb_t = getattr(layer_mod, "power_bias_dimensionless", None)
            pd_t = getattr(layer_mod, "power_diss_dimensionless", None)
            if pb_t is not None and pb_t.ndim == 3 and pb_t.shape[1] >= 1:
                pb_accum[lid].append(pb_t[:, -1, :])
            if pd_t is not None and pd_t.ndim == 3 and pd_t.shape[1] >= 1:
                pd_accum[lid].append(pd_t[:, -1, :])
        except Exception:
            pass


def rebuild_histories_stepwise(
    model,
    *,
    seq_len: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    s_histories: dict[int, torch.Tensor],
    phi_accum: dict[int, list[torch.Tensor]],
    g_accum: dict[int, list[torch.Tensor]],
    pb_accum: dict[int, list[torch.Tensor]],
    pd_accum: dict[int, list[torch.Tensor]],
    first_layer_id: int,
    input_type: str,
) -> None:
    # Rebuild per‑layer histories for φ/g/s and power/energy
    for cfg in model.layers_config:
        lid = cfg.layer_id
        layer_idx = {c.layer_id: i for i, c in enumerate(model.layers_config)}[lid]
        layer_mod = model.layers[layer_idx]

        if getattr(model.sim_config, "track_phi", False):
            if (input_type == "state") and (lid == first_layer_id):
                layer_mod._clear_phi_history()
            else:
                layer_mod._clear_phi_history()
                if len(phi_accum[lid]) == seq_len:
                    for t in range(seq_len):
                        layer_mod._add_phi_to_history(phi_accum[lid][t])
                else:
                    if len(phi_accum[lid]) != 0:
                        warnings.warn(
                            f"stepwise mode: φ history length mismatch for layer {lid}: expected {seq_len}, got {len(phi_accum[lid])}. Backfilling with zeros.",
                            UserWarning,
                            stacklevel=2,
                        )
                    zeros = torch.zeros(batch, model.layer_nodes[lid], device=device, dtype=dtype)
                    for _ in range(seq_len):
                        layer_mod._add_phi_to_history(zeros)

        if getattr(model.sim_config, "track_g", False):
            layer_mod._clear_g_history()
            if len(g_accum[lid]) == seq_len:
                for t in range(seq_len):
                    layer_mod._add_g_to_history(g_accum[lid][t])
            else:
                if len(g_accum[lid]) != 0:
                    warnings.warn(
                        f"stepwise mode: g history length mismatch for layer {lid}: expected {seq_len}, got {len(g_accum[lid])}. Backfilling with zeros.",
                        UserWarning,
                        stacklevel=2,
                    )
                zeros = torch.zeros(batch, model.layer_nodes[lid], device=device, dtype=dtype)
                for _ in range(seq_len):
                    layer_mod._add_g_to_history(zeros)

        if getattr(model.sim_config, "track_s", False):
            layer_mod._clear_state_history()
            for t in range(seq_len):
                layer_mod._add_state_to_history(s_histories[lid][:, t + 1, :])

        if getattr(model.sim_config, "track_power", False) and getattr(layer_mod, "track_power", False):
            if len(pb_accum[lid]) == seq_len and len(pd_accum[lid]) == seq_len:
                pb = torch.stack(pb_accum[lid], dim=1)
                pd = torch.stack(pd_accum[lid], dim=1)
                if isinstance(model.dt, torch.Tensor):
                    dt_val = model.dt.to(device=pb.device, dtype=pb.dtype)
                else:
                    dt_val = torch.tensor(float(model.dt), device=pb.device, dtype=pb.dtype)
                eb = dt_val * pb.cumsum(dim=1)
                ed = dt_val * pd.cumsum(dim=1)
                layer_mod.power_bias_dimensionless = pb
                layer_mod.power_diss_dimensionless = pd
                layer_mod.energy_bias_dimensionless = eb
                layer_mod.energy_diss_dimensionless = ed
                with contextlib.suppress(Exception):
                    layer_mod._convert_power_and_energy()
            else:
                try:
                    layer_mod._set_zero_power_tracking(batch, seq_len, device, dtype)
                except Exception:
                    shape = (batch, seq_len, model.layer_nodes[lid])
                    layer_mod.power_bias_dimensionless = torch.zeros(shape, device=device, dtype=dtype)
                    layer_mod.power_diss_dimensionless = torch.zeros_like(layer_mod.power_bias_dimensionless)
                    layer_mod.energy_bias_dimensionless = torch.zeros_like(layer_mod.power_bias_dimensionless)
                    layer_mod.energy_diss_dimensionless = torch.zeros_like(layer_mod.power_bias_dimensionless)
                    with contextlib.suppress(Exception):
                        layer_mod._convert_power_and_energy()
