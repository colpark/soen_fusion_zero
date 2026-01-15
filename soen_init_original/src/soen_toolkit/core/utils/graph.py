from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def build_inbound_by_target(connections_config, connections) -> dict[int, list[tuple[int, str]]]:
    """Map target layer_id -> list of (source_id, connection_key) for inter-layer connections."""
    inbound_by_target: dict[int, list[tuple[int, str]]] = {}
    for conn_cfg in connections_config:
        if conn_cfg.from_layer != conn_cfg.to_layer:
            key = f"J_{conn_cfg.from_layer}_to_{conn_cfg.to_layer}"
            if key in connections:
                inbound_by_target.setdefault(conn_cfg.to_layer, []).append((conn_cfg.from_layer, key))
    return inbound_by_target


def precompute_external_J(model, apply_noise: bool = True) -> dict[str, torch.Tensor]:
    """Build effective external connection matrices with QAT-STE applied.

    For GaussianNoise: noise is NOT applied here (must be applied per timestep).
    For GaussianPerturbation: noise IS applied here (fixed per forward pass).

    Args:
        model: SOENModelCore instance
        apply_noise: If False, skip all noise application (for manual per-step handling)

    Skips internal connections - those are handled by layer solvers.
    """
    from soen_toolkit.core.noise import GaussianPerturbation

    J_eff_by_key: dict[str, torch.Tensor] = {}
    for conn_cfg in model.connections_config:
        if conn_cfg.from_layer == conn_cfg.to_layer:
            continue
        key = f"J_{conn_cfg.from_layer}_to_{conn_cfg.to_layer}"
        J = model.connections.get(key)
        if J is None:
            continue
        J_eff = model._apply_qat_ste_if_enabled(key, J)

        # Apply noise only for perturbation (fixed per forward)
        # GaussianNoise should be applied per timestep, not here
        if apply_noise:
            conn_noise = model.connection_noise_settings.get(key)
            if conn_noise and getattr(conn_noise, "j", None) is not None:
                j_noise = conn_noise.j
                if isinstance(j_noise, GaussianPerturbation):
                    # Fixed perturbation: apply once here
                    J_eff = conn_noise.apply(J_eff, "j")
                # GaussianNoise is NOT applied here - applied per timestep in stepwise loop

        J_eff_by_key[key] = J_eff
    return J_eff_by_key


def apply_connection_noise_step(J: torch.Tensor, conn_noise) -> torch.Tensor:
    """Apply per-timestep GaussianNoise to connection weight.

    This should be called inside the stepwise loop for GaussianNoise connections.
    GaussianPerturbation is already applied in precompute_external_J.

    Args:
        J: Weight matrix [D, F]
        conn_noise: NoiseSettings with j attribute

    Returns:
        Noisy weight matrix [D, F]
    """
    from soen_toolkit.core.noise import GaussianNoise

    if conn_noise is None:
        return J

    j_noise = getattr(conn_noise, "j", None)
    if j_noise is None:
        return J

    if isinstance(j_noise, GaussianNoise):
        # Apply fresh noise for this timestep
        return conn_noise.apply(J, "j")

    # For other types (perturbation), J already has it applied
    return J


def has_per_step_connection_noise(model, key: str) -> bool:
    """Check if a connection has GaussianNoise (per-timestep) configured."""
    from soen_toolkit.core.noise import GaussianNoise

    conn_noise = model.connection_noise_settings.get(key)
    if conn_noise is None:
        return False
    j_noise = getattr(conn_noise, "j", None)
    return isinstance(j_noise, GaussianNoise) and j_noise.std > 0.0
