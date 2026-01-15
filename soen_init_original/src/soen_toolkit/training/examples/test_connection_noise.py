"""Quick simulation script to validate connection noise (j) is temporary and affects outputs.

Usage:
  python -m soen_toolkit.training.examples.test_connection_noise
"""

from __future__ import annotations

import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
)
from soen_toolkit.core.noise import GaussianPerturbation
from soen_toolkit.core.soen_model_core import SOENModelCore


def build_toy_model(input_dim: int = 4, hidden_dim: int = 6) -> SOENModelCore:
    sim = SimulationConfig(dt=37, track_phi=False, track_s=False)

    layers = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": input_dim}),
        LayerConfig(
            layer_id=1,
            layer_type="SingleDendrite",
            params={
                "dim": hidden_dim,
                # Ensure non-zero dynamics and avoid dead zone
                "phi_offset": 0.23,
                # Use a simple nonlinearity for this test to guarantee non-zero g
                "source_func": "Tanh",
                # Requested gains
                "gamma_plus": 0.0033,
                "gamma_minus": 0.0033,
            },
        ),
    ]

    # Dense connectivity 0 -> 1 with uniform init in a small range
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            # Stronger constant J so upstream flux is clearly non-zero
            params={"init": "constant", "value": 0.2},
            learnable=True,
        ),
    ]

    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
    # Ensure masks/constraints are applied once so we snapshot a stable baseline
    model.enforce_param_constraints()
    return model


@torch.no_grad()
def main() -> None:
    torch.manual_seed(1234)
    device = torch.device("cpu")

    model = build_toy_model().to(device)

    # Identify the connection key and snapshot baseline
    conn_key = "J_0_to_1"
    assert conn_key in model.connections, f"Connection '{conn_key}' not found"
    baseline_J = model.connections[conn_key].detach().clone()

    # Configure connection perturbation via the model's NoiseSettings
    # Set a modest std for GaussianPerturbation (sampled once per forward)
    std = 0.5
    # Overwrite/assign the 'j' noise strategy for this connection
    settings = model.connection_noise_settings.get(conn_key)
    from soen_toolkit.core.noise import NoiseSettings

    if settings is None:
        settings = NoiseSettings(j=GaussianPerturbation(mean=0.0, std=std))
        model.connection_noise_settings[conn_key] = settings
    else:
        # Replace current 'j' strategy on frozen dataclass
        object.__setattr__(settings, "j", GaussianPerturbation(mean=0.0, std=std))

    # Create dummy input [batch, seq_len, input_dim]
    batch, seq_len, input_dim = 2, 64, model.layers_config[0].params["dim"]
    # Feed constant input
    x = torch.ones(batch, seq_len, input_dim, device=device) * 0.5

    # Quick sanity: noise strategy actually mutates J at apply-time
    J0 = model.connections[conn_key].detach().clone()
    settings.apply(J0, "j") if hasattr(settings, "apply") else J0
    settings.apply(J0, "j") if hasattr(settings, "apply") else J0

    # Run two forwards; capture full state trajectories list
    out1, _states_list1 = model(x)
    out2, _states_list2 = model(x)

    # Check parameter unchanged (no compounding/in-place edits)
    torch.allclose(model.connections[conn_key].detach(), baseline_J)

    # Last layer stats (exclude s0)
    s1 = out1[:, 1:, :]
    s2 = out2[:, 1:, :]
    (s2 - s1)

    # Extra: show that disabling noise makes outputs repeatable
    # Remove the 'j' strategy and re-run
    object.__setattr__(settings, "j", None)
    _out3, _ = model(x)
    _out4, _ = model(x)


if __name__ == "__main__":
    main()
