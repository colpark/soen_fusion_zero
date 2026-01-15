import pytest
import torch

from soen_toolkit.core import LayerConfig, SimulationConfig, SOENModelCore


def build_single_dendrite(dim=4):
    sim = SimulationConfig(dt=37, input_type="flux")
    layer_cfgs = [
        LayerConfig(
            layer_id=0,
            layer_type="SingleDendrite",
            params={"dim": dim},
        ),
    ]
    return SOENModelCore(sim_config=sim, layers_config=layer_cfgs, connections_config=[])


@pytest.mark.parametrize("target", ["noise", "perturb"])  # sanity: core phi works as doc'd
def test_core_phi_noise_vs_perturb_time_variation(target) -> None:
    torch.manual_seed(0)
    model = build_single_dendrite(dim=3)
    B, T, D = 2, 16, 3
    x = torch.zeros(B, T, D)

    # Configure layer 0 core phi noise vs perturb via its LayerConfig
    cfg = model.layers_config[0]
    # dataclasses are frozen; replace with new instances to modify
    from soen_toolkit.training.configs import NoiseConfig, PerturbationConfig

    if target == "noise":
        cfg.noise = NoiseConfig(phi=0.5)
    else:
        cfg.perturb = PerturbationConfig(phi_std=0.5)

    # Assert on φ directly to avoid dynamics confounds
    model.set_tracking(track_phi=True)
    _ = model(x)
    phi_hist = model.get_phi_history()[0]  # list per layer, take first
    if target == "noise":
        assert not torch.allclose(phi_hist[:, 0, :], phi_hist[:, -1, :])
    else:
        assert torch.allclose(phi_hist[:, 0, :], phi_hist[:, -1, :])
