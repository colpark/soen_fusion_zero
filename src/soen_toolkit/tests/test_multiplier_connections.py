import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)


def _simple_model(sim_mode: str, mode: str):
    sim = SimulationConfig(network_evaluation_method=sim_mode, input_type="flux", dt=37)
    layers = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 2}),
        LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 2}),
    ]
    # 2x2 identity J
    J = torch.eye(2)
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={
                "J": {"min": -1.0, "max": 1.0},
                "mode": mode,
                "connection_params": {"gamma_plus": 0.0, "bias_current": 0.0, "source_func": "Tanh"},
                "init": "constant",
                "min": 0.0,
                "max": 0.0,
            },
            learnable=False,
        ),
    ]
    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
    # Overwrite J to exact identity (ensures mask dense and matches dims)
    model.connections["J_0_to_1"].data.copy_(J)
    return model


def test_multiplier_zero_gamma_produces_zero_contribution_layerwise() -> None:
    torch.manual_seed(0)
    model_fixed = _simple_model("layerwise", mode="fixed")
    model_mult = _simple_model("layerwise", mode="WICC")
    # With gamma_plus=0 the multiplier edge states do not change, so contribution is zero.
    x = torch.randn(2, 5, 2)
    _, h_fixed = model_fixed(x)
    _, h_mult = model_mult(x)
    # Second layer history (after t=0) should be all zeros for multiplier mode.
    assert torch.allclose(h_mult[1][:, 1:, :], torch.zeros_like(h_mult[1][:, 1:, :]))
    # And fixed should generally be non-zero for the same input (sanity).
    assert not torch.allclose(h_fixed[1][:, 1:, :], torch.zeros_like(h_fixed[1][:, 1:, :]))


def test_multiplier_stepwise_allocates_state_and_runs() -> None:
    model_mult = _simple_model("stepwise_gauss_seidel", mode="WICC")
    x = torch.randn(1, 3, 2)
    _y, h = model_mult(x)
    assert len(h) == 2
    assert h[1].shape == (1, 4, 2)


def test_internal_connectivity_multiplier_mode_runs() -> None:
    # SingleDendrite with internal connectivity using dynamic (programmable) mode
    sim = SimulationConfig(network_evaluation_method="layerwise", input_type="flux", dt=37)
    layers = [
        LayerConfig(
            layer_id=0,
            layer_type="SingleDendrite",
            params={
                "dim": 2,
                "connectivity": torch.eye(2) * 0.1,
                "connectivity_mode": "WICC",
                "connectivity_dynamic": {"gamma_plus": 0.0, "bias_current": 0.0, "source_func": "Tanh"},
            },
        ),
    ]
    conns = []
    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
    x = torch.randn(1, 5, 2)
    _y, h = model(x)
    # With gamma_plus=0 the internal multiplier contributes zero, so output equals input (with noise off)
    assert not torch.allclose(h[0][:, 1:, :], torch.zeros_like(h[0][:, 1:, :]))
