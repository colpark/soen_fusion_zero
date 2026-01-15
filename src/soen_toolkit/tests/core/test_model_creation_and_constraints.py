import pytest
import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.tests.utils.test_helpers_fixture import (
    build_small_model,
    make_random_series,
)


def test_connectivity_mask_application_preserves_zeros() -> None:
    m = build_small_model(dims=(4, 5), connectivity_type="one_to_one", init="constant", init_value=1.0)

    # Weight should be masked to diagonal up to min(4,5)=4
    w = m.connections["J_0_to_1"].detach()
    # Positions outside diagonal should be zeros
    off_diag = w.clone()
    off_diag.fill_diagonal_(0)  # zero out diagonal entries
    assert torch.count_nonzero(off_diag).item() == 0

    # After a forward pass, masks are re-applied; ensure still zero
    x = make_random_series(batch=2, seq_len=3, dim=4, seed=1)
    _ = m(x)
    w2 = m.connections["J_0_to_1"].detach()
    off_diag2 = w2.clone()
    off_diag2.fill_diagonal_(0)
    assert torch.count_nonzero(off_diag2).item() == 0


def test_connection_constraints_clamp_after_forward() -> None:
    # Set tight constraints to force clamping
    constraints = {"min": -0.1, "max": 0.1}
    m = build_small_model(dims=(3, 3), connectivity_type="dense", init="normal", init_value=0.5, constraints=constraints)

    # Drive a forward pass to trigger enforce_param_constraints at end
    x = make_random_series(batch=1, seq_len=2, dim=3, seed=2)
    _ = m(x)

    w = m.connections["J_0_to_1"].detach()
    assert torch.all(w <= 0.1 + 1e-7)
    assert torch.all(w >= -0.1 - 1e-7)


def test_internal_j_constraints_propagated_and_clamped() -> None:
    # Build model with internal connection and constraints
    m = build_small_model(
        dims=(3, 3),
        with_internal_first=True,
        connectivity_type="dense",
        init="constant",
        init_value=0.5,
        constraints=None,
    )

    # Inject internal constraints via connection config by rebuilding with constraints on internal_0
    # Easiest: directly set layer's constraints and value here, then run forward and check clamp
    layer0 = m.layers[0]
    if getattr(layer0, "connectivity", None) is not None:
        layer0.connectivity.weight.data.fill_(1.0)
        layer0.connectivity.constraints = {"min": -0.2, "max": 0.2}

    x = make_random_series(batch=1, seq_len=2, dim=3, seed=3)
    _ = m(x)

    if getattr(layer0, "connectivity", None) is not None:
        Jd = layer0.connectivity.materialised().detach()
    else:
        pytest.skip("Layer does not expose internal connectivity")
    assert torch.all(Jd <= 0.2 + 1e-7)
    assert torch.all(Jd >= -0.2 - 1e-7)


def test_learnable_layer_params_respect_constraints_during_training() -> None:
    torch.manual_seed(0)

    # Configure a SingleDendrite layer with explicit constraints on all params and mark them learnable
    d0, d1 = 4, 4
    layer0_params = {
        "dim": d0,
        # Non-log parameter with bounds
        "phi_offset": {
            "distribution": "constant",
            "params": {"value": 0.0},
            "constraints": {"min": -0.2, "max": 0.2},
            "learnable": True,
        },
        # Non-log parameter with bounds
        "bias_current": {
            "distribution": "constant",
            "params": {"value": 0.5},
            "constraints": {"min": 0.1, "max": 0.8},
            "learnable": True,
        },
        # Log-parameters with real-space bounds (converted to log-space internally)
        "gamma_plus": {
            "distribution": "loguniform",
            "params": {"min": 1e-3, "max": 1e-2},
            "constraints": {"min": 1e-4, "max": 1.0},
            "learnable": True,
        },
        "gamma_minus": {
            "distribution": "loguniform",
            "params": {"min": 1e-3, "max": 1e-2},
            "constraints": {"min": 1e-4, "max": 1.0},
            "learnable": True,
        },
    }

    layers = [
        LayerConfig(layer_id=0, layer_type="SingleDendrite", params=layer0_params),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": d1}),
    ]

    # Simple inter-layer connection
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "constant", "value": 0.1},
            learnable=True,
        ),
    ]

    sim = SimulationConfig(dt=37, input_type="flux")
    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

    # Grab the first layer for assertions
    layer0 = model.layers[0]

    # Sanity: all targeted params should be learnable
    assert layer0.phi_offset.requires_grad is True
    assert layer0.bias_current.requires_grad is True
    assert layer0.log_gamma_plus.requires_grad is True
    assert layer0.log_gamma_minus.requires_grad is True

    # Optimizer and a tiny training loop
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    batch, seq_len = 2, 6
    x = torch.randn(batch, seq_len, d0)
    target = torch.zeros(batch, seq_len + 1, d1)

    for _ in range(8):
        opt.zero_grad()
        y_hist, _ = model(x)
        loss = torch.nn.functional.mse_loss(y_hist, target)
        loss.backward()
        opt.step()
        # Enforce constraints after optimizer step (as done in training hooks)
        model.enforce_param_constraints()

        # Check bounds in real-space for non-log and via property for log-params
        assert torch.all(layer0.phi_offset.detach() <= 0.2 + 1e-7)
        assert torch.all(layer0.phi_offset.detach() >= -0.2 - 1e-7)

        assert torch.all(layer0.bias_current.detach() <= 0.8 + 1e-7)
        assert torch.all(layer0.bias_current.detach() >= 0.1 - 1e-7)

        # Properties expose exp(log_param) so comparisons are in real space
        assert torch.all(layer0.gamma_plus.detach() <= 1.0 + 1e-7)
        assert torch.all(layer0.gamma_plus.detach() >= 1e-4 - 1e-10)
        assert torch.all(layer0.gamma_minus.detach() <= 1.0 + 1e-7)
        assert torch.all(layer0.gamma_minus.detach() >= 1e-4 - 1e-10)
