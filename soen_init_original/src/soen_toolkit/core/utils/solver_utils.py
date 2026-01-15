from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import torch

if TYPE_CHECKING:
    from soen_toolkit.core.noise import NoiseSettings


def process_first_layer(
    model,
    external_phi: torch.Tensor,
    batch: int,
    seq_len: int,
    device: torch.device,
    noise_settings: NoiseSettings,
    initial_state: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    """Process the first layer based on input type (flux or state).

    Mirrors SOENModelCore._process_first_layer but provided as a utility.
    """
    s_histories: dict[int, torch.Tensor] = {}
    input_type = getattr(model.sim_config, "input_type", "flux")
    first_layer_config = model.layers_config[0]

    if input_type.lower() == "flux":
        # Ensure external input matches first layer dimension
        first_layer_dim = model.layer_nodes[first_layer_config.layer_id]
        if external_phi.shape[-1] != first_layer_dim:
            in_dim = external_phi.shape[-1]
            if in_dim > first_layer_dim:
                warnings.warn(
                    f"External input for 'flux' mode has {in_dim} channels, but first layer {first_layer_config.layer_id} has dim {first_layer_dim}. Slicing input.",
                    UserWarning,
                    stacklevel=2,
                )
                external_phi = external_phi[..., :first_layer_dim]
            else:  # Pad with zeros if input has fewer channels
                pad_amount = first_layer_dim - in_dim
                padding = torch.zeros(
                    *external_phi.shape[:-1],
                    pad_amount,
                    device=external_phi.device,
                    dtype=external_phi.dtype,
                )
                external_phi = torch.cat([external_phi, padding], dim=-1)

        # Inject noise into the external flux if configured
        if noise_settings.phi is not None:
            external_phi = model.layers[0]._apply_noise(external_phi, noise_settings, "phi")

        # First layer processes external input as flux (optionally with initial state)
        try:
            s_hist = model.layers[0](external_phi, noise_config=noise_settings, initial_state=initial_state)
        except TypeError:
            s_hist = model.layers[0](external_phi, noise_config=noise_settings)

        s_histories[first_layer_config.layer_id] = s_hist

    elif input_type.lower() == "state":
        # Check if the first layer is of the appropriate type for 'state' input
        if first_layer_config.layer_type not in {"Linear", "Input"}:
            warnings.warn(
                f"Input type is 'state', but the first layer (ID: {first_layer_config.layer_id}) "
                f"is of type '{first_layer_config.layer_type}'. For clarity, it is recommended to use "
                f"'LinearLayer' (formerly 'InputLayer') as the first layer when input_type is 'state'.",
                UserWarning,
                stacklevel=2,
            )

        # In state mode, external_phi represents the desired state values
        first_layer_dim = model.layer_nodes[first_layer_config.layer_id]

        # Ensure external_phi matches the first layer's dimension, pad or slice if necessary
        if external_phi.shape[-1] != first_layer_dim:
            if external_phi.shape[-1] < first_layer_dim:
                pad_amount = first_layer_dim - external_phi.shape[-1]
                padding = torch.zeros(
                    *external_phi.shape[:-1],
                    pad_amount,
                    device=external_phi.device,
                    dtype=external_phi.dtype,
                )
                external_phi_adjusted = torch.cat([external_phi, padding], dim=-1)
            else:  # external_phi.shape[-1] > first_layer_dim
                warnings.warn(
                    f"External input for 'state' mode has {external_phi.shape[-1]} channels, but first layer {first_layer_config.layer_id} has dim {first_layer_dim}. Slicing input.",
                    stacklevel=2,
                )
                external_phi_adjusted = external_phi[..., :first_layer_dim]
        else:
            external_phi_adjusted = external_phi

        s_hist = torch.zeros(
            batch,
            seq_len + 1,
            first_layer_dim,
            device=device,
            dtype=external_phi_adjusted.dtype,
        )
        # States s_1..s_T are the provided input; s_0 remains zeros
        s_hist[:, 1:, :] = external_phi_adjusted

        # Add noise to states s_1...s_T if configured
        if noise_settings.s is not None:
            noise_slice = model.layers[0]._apply_noise(external_phi_adjusted, noise_settings, "s")
            s_hist[:, 1:, :] = noise_slice

        s_histories[first_layer_config.layer_id] = s_hist

        # Tracking for state-input mode
        first_layer_module = model.layers[0]
        first_layer_module._clear_phi_history()
        first_layer_module._clear_state_history()
        if getattr(first_layer_module, "track_s", False):
            for t in range(seq_len):
                first_layer_module._add_state_to_history(s_hist[:, t + 1, :])

    else:
        msg = f"Unknown input_type: {input_type}. Must be 'flux' or 'state'."
        raise ValueError(msg)

    return s_histories


def _apply_connection_noise_layerwise(
    J: torch.Tensor,
    conn_noise,
    seq_len: int,
) -> torch.Tensor:
    """Apply per-timestep noise to connection weight for layerwise mode.

    GaussianNoise: Generates [T, D, F] with different noise each timestep
    GaussianPerturbation: Uses same offset for all timesteps (calls offset() once)

    Args:
        J: Weight matrix [D, F]
        conn_noise: NoiseSettings with j attribute
        seq_len: Number of timesteps

    Returns:
        Noisy weight [T, D, F] or original J [D, F] if no noise
    """
    from soen_toolkit.core.noise import GaussianNoise, GaussianPerturbation

    j_noise = getattr(conn_noise, "j", None)
    if j_noise is None:
        return J

    if isinstance(j_noise, GaussianNoise):
        # Per-timestep noise: expand J to [T, D, F] with different noise each timestep
        D, F = J.shape
        if j_noise.std == 0.0:
            return J
        if j_noise.relative:
            scale = j_noise.std * J.abs()  # [D, F]
            noise = torch.randn(seq_len, D, F, device=J.device, dtype=J.dtype) * scale.unsqueeze(0)
        else:
            noise = torch.randn(seq_len, D, F, device=J.device, dtype=J.dtype) * j_noise.std
        return J.unsqueeze(0) + noise  # [T, D, F]

    elif isinstance(j_noise, GaussianPerturbation):
        # Fixed perturbation: apply once, same for all timesteps
        return conn_noise.apply(J, "j")  # Returns [D, F]

    else:
        # Unknown type - just apply directly
        return conn_noise.apply(J, "j")


def collect_upstream_contributions(
    model,
    curr_id: int,
    batch: int,
    seq_len: int,
    curr_dim: int,
    device: torch.device,
    s_histories: dict,
    noise_settings: NoiseSettings,
) -> torch.Tensor:
    """Collect input contributions from all upstream layers.

    Mirrors SOENModelCore._collect_upstream_contributions.
    """
    upstream_phi = None

    for prev_cfg in model.layers_config:
        if prev_cfg.layer_id < curr_id:
            key = f"J_{prev_cfg.layer_id}_to_{curr_id}"
            if key in model.connections:
                s_prev = s_histories[prev_cfg.layer_id][:, 1 : seq_len + 1, :]

                J = model.connections[key]
                J = model._apply_qat_ste_if_enabled(key, J)

                # Apply connection noise (per-timestep for GaussianNoise, fixed for Perturbation)
                conn_noise = model.connection_noise_settings.get(key)
                if conn_noise and getattr(conn_noise, "j", None) is not None:
                    J_noisy = _apply_connection_noise_layerwise(J, conn_noise, seq_len)
                else:
                    J_noisy = J

                # Use unified connection helper
                from soen_toolkit.core.utils.connection_ops import apply_connection_layerwise

                mode = getattr(model, "_connection_modes", {}).get(key, "fixed")
                params = getattr(model, "_connection_params", {}).get(key, {})
                edge_indices = getattr(model, "_connection_edge_maps", {}).get(key, (None, None))

                # Handle per-timestep noise (J_noisy is [T, D, F])
                if J_noisy.dim() == 3:
                    # Per-timestep noise: use einsum for [B, T, F] @ [T, D, F] -> [B, T, D]
                    contrib = torch.einsum("btf,tdf->btd", s_prev, J_noisy)
                else:
                    # Standard case: J is [D, F]
                    contrib = apply_connection_layerwise(s_prev, J_noisy, mode, params, edge_indices, model.dt)

                upstream_phi = contrib if upstream_phi is None else (upstream_phi + contrib)

    if noise_settings.phi is not None:
        layer_idx = next(i for i, cfg in enumerate(model.layers_config) if cfg.layer_id == curr_id)
        if upstream_phi is None:
            upstream_phi = torch.zeros(batch, seq_len, curr_dim, device=device)
        upstream_phi = model.layers[layer_idx]._apply_noise(upstream_phi, noise_settings, "phi")

    if upstream_phi is None:
        upstream_phi = torch.zeros(batch, seq_len, curr_dim, device=device)
    return upstream_phi
