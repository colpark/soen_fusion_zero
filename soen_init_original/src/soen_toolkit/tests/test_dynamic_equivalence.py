import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)


def _linear_multiplier_linear(phi_y: float, gamma_plus: float, bias_current: float, *, source: str = "RateArray"):
    sim = SimulationConfig(network_evaluation_method="layerwise", input_type="state", dt=37)
    layers = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 1}),
        LayerConfig(
            layer_id=1,
            layer_type="Multiplier",
            params={
                "dim": 1,
                "solver": "FE",
                "source_func": source,
                # Force exact params
                "phi_y": {"value": phi_y, "learnable": False},
                "gamma_plus": {"value": gamma_plus, "learnable": False},
                "gamma_minus": {"value": 0.0, "learnable": False},  # Set to 0 for equivalence test
                "bias_current": {"value": bias_current, "learnable": False},
            },
        ),
        LayerConfig(layer_id=2, layer_type="Linear", params={"dim": 1}),
    ]
    conns = [
        ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={"structure": {"type": "dense"}}),
        ConnectionConfig(from_layer=1, to_layer=2, connection_type="dense", params={"structure": {"type": "dense"}}),
    ]
    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
    # Set J=1 for both edges
    model.connections["J_0_to_1"].data.fill_(1.0)
    model.connections["J_1_to_2"].data.fill_(1.0)
    return model


def _linear_dynamic_linear(phi_y: float, gamma_plus: float, bias_current: float, *, source: str = "RateArray"):
    sim = SimulationConfig(network_evaluation_method="layerwise", input_type="state", dt=37)
    layers = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 1}),
        LayerConfig(layer_id=1, layer_type="Linear", params={"dim": 1}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={
                "structure": {"type": "dense"},
                "mode": "WICC",
                "connection_params": {
                    "source_func": source,
                    "gamma_plus": gamma_plus,
                    "gamma_minus": 0.0,  # Set to 0 for equivalence test
                    "bias_current": bias_current,
                    "j_in": 1.0,
                    "j_out": 1.0,
                },
            },
        ),
    ]
    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
    # Set J (phi_y) = constant on the dynamic edge
    model.connections["J_0_to_1"].data.fill_(phi_y)
    return model


def test_linear_multiplier_matches_linear_dynamic_layerwise() -> None:
    torch.manual_seed(0)
    phi_y = 0.1
    gamma_plus = 1e-3
    bias_current = 2.0
    source = "RateArray"

    m1 = _linear_multiplier_linear(phi_y, gamma_plus, bias_current, source=source)
    m2 = _linear_dynamic_linear(phi_y, gamma_plus, bias_current, source=source)

    # Input sequence
    x = torch.linspace(-0.3, 0.3, steps=21).view(1, 21, 1)

    _y1, h1 = m1(x)
    _y2, h2 = m2(x)

    # Net1 layer 2 should equal Net2 layer 1 (full trajectories)
    assert h1[-1].shape == h2[-1].shape
    assert torch.allclose(h1[-1], h2[-1], atol=1e-6, rtol=1e-5)

    # In Net1, Multiplier state equals downstream Linear (J=1)
    assert torch.allclose(h1[1], h1[2], atol=1e-6, rtol=1e-5)


def test_dynamic_equivalence_stepwise_gauss_seidel() -> None:
    torch.manual_seed(0)
    phi_y = 0.1
    gamma_plus = 1e-3
    bias_current = 2.0
    source = "RateArray"

    m1 = _linear_multiplier_linear(phi_y, gamma_plus, bias_current, source=source)
    m2 = _linear_dynamic_linear(phi_y, gamma_plus, bias_current, source=source)

    m1.sim_config.network_evaluation_method = "stepwise_gauss_seidel"
    m2.sim_config.network_evaluation_method = "stepwise_gauss_seidel"

    x = torch.linspace(-0.3, 0.3, steps=21).view(1, 21, 1)
    _y1, h1 = m1(x)
    _y2, h2 = m2(x)

    # Net1 layer 2 equals Net2 layer 1
    assert h1[-1].shape == h2[-1].shape
    assert torch.allclose(h1[-1], h2[-1], atol=1e-6, rtol=1e-5)

    # Gauss–Seidel uses freshest values within the step: downstream Linear equals Multiplier
    assert torch.allclose(h1[1], h1[2], atol=1e-6, rtol=1e-5)


def test_dynamic_equivalence_stepwise_jacobi() -> None:
    torch.manual_seed(0)
    phi_y = 0.1
    gamma_plus = 1e-3
    bias_current = 2.0
    source = "RateArray"

    m1 = _linear_multiplier_linear(phi_y, gamma_plus, bias_current, source=source)
    m2 = _linear_dynamic_linear(phi_y, gamma_plus, bias_current, source=source)

    m1.sim_config.network_evaluation_method = "stepwise_jacobi"
    m2.sim_config.network_evaluation_method = "stepwise_jacobi"

    x = torch.linspace(-0.3, 0.3, steps=21).view(1, 21, 1)
    _y1, h1 = m1(x)
    _y2, h2 = m2(x)

    # Net1 layer 2 equals Net2 layer 1
    assert h1[-1].shape == h2[-1].shape
    assert torch.allclose(h1[-1], h2[-1], atol=1e-6, rtol=1e-5)

    # Jacobi uses previous-step φ for inbound edges, so downstream Linear is one-step delayed
    # relative to the Multiplier's state trajectory.
    assert torch.allclose(h1[1][:, :-1, :], h1[2][:, 1:, :], atol=1e-6, rtol=1e-5)
