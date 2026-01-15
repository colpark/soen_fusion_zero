import warnings

import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)


def _build_ff_model(dim0=3, dim1=3, weight=0.1) -> SOENModelCore:
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": int(dim0)}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": int(dim1)}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "constant", "value": float(weight)},
            learnable=False,
        ),
    ]
    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def test_stepwise_matches_parallel_feedforward_no_noise() -> None:
    torch.manual_seed(0)
    m = _build_ff_model(dim0=4, dim1=5, weight=0.05)
    x = torch.randn(2, 6, 4)

    # Run standard forward (parallel per layer) with stepwise solver off
    m.sim_config.network_evaluation_method = "layerwise"
    y_parallel, _ = m(x)

    # Run stepwise solver
    m.sim_config.network_evaluation_method = "stepwise_gauss_seidel"
    y_stepwise, _ = m(x)
    assert torch.allclose(y_parallel, y_stepwise, atol=1e-6, rtol=1e-6), "Feedforward outputs must match within numerical precision"


def test_flux_input_pre_adjust_warns_on_slice() -> None:
    m = _build_ff_model(dim0=3, dim1=3)
    # External input has more channels than first layer; should warn and slice
    x = torch.randn(1, 2, 5)
    m.sim_config.network_evaluation_method = "stepwise_gauss_seidel"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _y, _ = m(x)
        assert any("Slicing input" in str(wi.message) for wi in w), "Expected slicing warning for flux mode"
