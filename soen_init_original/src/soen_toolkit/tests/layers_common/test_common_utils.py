import math

import pytest
import torch

from soen_toolkit.core.layers.common import (
    FeatureHook,
    ForwardEulerSolver,
    PowerTrackingFeature,
    StepPayload,
    broadcast_initial_state,
    validate_sequence_input,
)
from soen_toolkit.core.layers.physical import (
    MultiplierLayer,
    ReadoutLayer,
    SingleDendriteLayer,
)
from soen_toolkit.core.layers.physical.dynamics import MultiplierDynamics
from soen_toolkit.core.source_functions.rate_array import RateArraySource


def test_validate_sequence_input_success_and_error() -> None:
    tensor = torch.zeros(2, 3, 4)
    assert validate_sequence_input(tensor, dim=4) == (2, 3, 4)

    with pytest.raises(ValueError):
        validate_sequence_input(torch.zeros(2, 3), dim=3)

    with pytest.raises(ValueError):
        validate_sequence_input(tensor, dim=5)


def test_broadcast_initial_state_variants() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    out = broadcast_initial_state(None, batch=2, dim=3, device=device, dtype=dtype)
    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.zeros_like(out))

    init_vec = torch.arange(3, dtype=dtype)
    out_vec = broadcast_initial_state(init_vec, batch=2, dim=3, device=device, dtype=dtype)
    assert out_vec.shape == (2, 3)
    assert torch.allclose(out_vec[0], init_vec)
    assert torch.allclose(out_vec[1], init_vec)

    init_batch = torch.ones(2, 3)
    out_batch = broadcast_initial_state(init_batch, batch=2, dim=3, device=device, dtype=dtype)
    assert out_batch.shape == (2, 3)

    with pytest.raises(ValueError):
        broadcast_initial_state(torch.ones(4), batch=2, dim=3, device=device, dtype=dtype)


class _RecorderFeature(FeatureHook):
    def __init__(self) -> None:
        self.start_called = False
        self.before_calls: list[int] = []
        self.after_states: list[torch.Tensor] = []

    def on_integration_start(self, **kwargs) -> None:
        self.start_called = True

    def on_before_step(self, *, context, step_index, payload: StepPayload):
        self.before_calls.append(step_index)
        payload.extras["scaled_phi"] = payload.phi * 2
        return payload

    def on_after_step(self, *, context, step_index, payload: StepPayload) -> None:
        self.after_states.append(payload.state.detach().clone())


class _LinearDynamics:
    def __call__(self, state: torch.Tensor, phi: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        return params["gamma_plus"] * phi - params["gamma_minus"] * state


class _StateWrapper:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values


def test_forward_euler_invokes_features() -> None:
    phi = torch.ones(1, 3, 1)
    params = {
        "gamma_plus": torch.tensor([1.0]),
        "gamma_minus": torch.tensor([0.0]),
    }

    feature = _RecorderFeature()
    solver = ForwardEulerSolver(dynamics=_LinearDynamics(), feature=feature)
    state = _StateWrapper(torch.zeros(1, 1))

    history = solver.integrate(state=state, phi=phi, params=params, dt=torch.tensor(0.5))

    assert history.shape == (1, 4, 1)
    assert feature.start_called
    assert feature.before_calls == [0, 1, 2]
    assert len(feature.after_states) == 3
    assert torch.allclose(feature.after_states[-1], torch.full((1, 1), 1.5))


def test_power_tracking_feature_accumulates() -> None:
    batch, steps, dim = 1, 3, 1
    phi = torch.ones(batch, steps, dim)
    params = {
        "gamma_plus": torch.full((dim,), 1.0),
        "gamma_minus": torch.full((dim,), 0.1),
        "bias_current": torch.full((dim,), 0.5),
    }

    from soen_toolkit.physics.constants import DEFAULT_PHI0, get_omega_c

    power_feature = PowerTrackingFeature(Ic=1e-4, Phi0=DEFAULT_PHI0, wc=float(get_omega_c()))
    solver = ForwardEulerSolver(dynamics=_LinearDynamics(), feature=power_feature)
    state = _StateWrapper(torch.zeros(batch, dim))
    dt = torch.tensor(0.1)

    history = solver.integrate(state=state, phi=phi, params=params, dt=dt)

    storage = power_feature.storage
    assert storage.power_bias_dimensionless is not None
    assert storage.power_bias_dimensionless.shape == (batch, steps, dim)
    assert storage.energy_bias is not None
    assert storage.energy_bias.shape == (batch, steps, dim)
    assert torch.all(history[:, 0] == 0)


def test_single_dendrite_connectivity_contribution() -> None:
    dim = 2
    dt = 37
    connectivity = torch.eye(dim) * 0.2
    layer = SingleDendriteLayer(
        dim=dim,
        dt=dt,
        source_func_type="Tanh",
        connectivity=connectivity,
    )

    layer.phi_offset.data.zero_()
    layer.log_gamma_plus.data.fill_(0.0)
    layer.log_gamma_minus.data.fill_(math.log(1e-6))

    batch, steps = 1, 1
    phi = torch.zeros(batch, steps, dim)
    init_state = torch.ones(dim)

    history = layer.forward(phi, initial_state=init_state)

    gamma_plus = layer.parameter_values()["gamma_plus"]
    gamma_minus = layer.parameter_values()["gamma_minus"]
    phi_eff = torch.matmul(init_state.view(1, -1), connectivity.t())
    g_val = torch.tanh(phi_eff)
    expected_next = init_state + dt * (gamma_plus * g_val.squeeze(0) - gamma_minus * init_state)

    assert history.shape == (batch, steps + 1, dim)
    assert torch.allclose(history[0, -1], expected_next, atol=1e-5)


def test_multiplier_basic_step() -> None:
    layer = MultiplierLayer(dim=1, dt=37, source_func_type="Tanh")
    layer.phi_y.data.fill_(0.2)
    layer.bias_current.data.fill_(0.0)
    layer.log_gamma_plus.data.fill_(0.0)

    batch, steps = 1, 1
    phi = torch.zeros(batch, steps, 1)
    init_state = torch.zeros(1)

    history = layer.forward(phi, initial_state=init_state)

    gamma_plus = layer.parameter_values()["gamma_plus"]
    phi_y = layer.parameter_values()["phi_y"]
    g_a = torch.tanh(phi_y)
    g_b = torch.tanh(-phi_y)
    expected_next = init_state + 37 * gamma_plus * (g_a - g_b)
    assert torch.allclose(history[0, -1], expected_next, atol=1e-5)


def test_multiplier_with_connectivity() -> None:
    connectivity = torch.tensor([[0.1]])
    layer = MultiplierLayer(
        dim=1,
        dt=37,
        source_func_type="Tanh",
        connectivity=connectivity,
    )
    layer.phi_y.data.fill_(0.2)
    layer.bias_current.data.fill_(0.0)
    layer.log_gamma_plus.data.fill_(0.0)
    layer.log_gamma_minus.data.fill_(0.0)

    phi = torch.zeros(1, 1, 1)
    init_state = torch.ones(1)

    history = layer.forward(phi, initial_state=init_state)

    gamma_plus = layer.parameter_values()["gamma_plus"]
    gamma_minus = layer.parameter_values()["gamma_minus"]
    phi_eff = torch.matmul(init_state.view(1, -1), connectivity.t())
    phi_y = layer.parameter_values()["phi_y"]
    g_a = torch.tanh(phi_eff + phi_y)
    g_b = torch.tanh(phi_eff - phi_y)
    expected_next = init_state + 37 * (gamma_plus * (g_a - g_b).squeeze(0) - gamma_minus * init_state)
    assert torch.allclose(history[0, -1], expected_next, atol=1e-5)


def test_multiplier_rate_array_uses_squid_current() -> None:
    dynamics = MultiplierDynamics(source_function=RateArraySource())
    params = {
        "phi_y": torch.tensor([[0.1]]),
        "gamma_plus": torch.tensor([[0.1]]),
        "gamma_minus": torch.tensor([[0.001]]),
        "bias_current": torch.tensor([[2.0]]),
    }
    phi = torch.tensor([[0.3]])

    dsdt_zero_state = dynamics(torch.tensor([[0.0]]), phi, params).item()
    dsdt_positive_state = dynamics(torch.tensor([[0.5]]), phi, params).item()

    assert dsdt_zero_state > 0.0
    assert dsdt_positive_state < 0.0


def test_single_dendrite_connectivity_builder_spec() -> None:
    spec = {"type": "dense"}
    layer = SingleDendriteLayer(
        dim=2,
        dt=37,
        source_func_type="Tanh",
        connectivity_spec=spec,
    )
    assert layer.connectivity is not None
    assert torch.all(layer.connectivity.materialised() == 1)


def test_single_dendrite_parallel_scan_matches_fe() -> None:
    phi = torch.linspace(-0.2, 0.2, steps=5).view(1, 5, 1)
    init_state = torch.tensor([0.05])

    layer_fe = SingleDendriteLayer(dim=1, dt=37, source_func_type="Tanh", solver="FE")
    layer_ps = SingleDendriteLayer(dim=1, dt=37, source_func_type="Tanh", solver="PS")

    for layer in (layer_fe, layer_ps):
        layer.phi_offset.data.fill_(0.3)
        layer.bias_current.data.fill_(0.0)
        layer.log_gamma_plus.data.fill_(0.0)
        layer.log_gamma_minus.data.fill_(torch.log(torch.tensor(0.01)))

    out_fe = layer_fe.forward(phi, initial_state=init_state)
    out_ps = layer_ps.forward(phi, initial_state=init_state)

    assert torch.allclose(out_fe, out_ps, atol=1e-5)


def test_readout_layer_outputs_g_values() -> None:
    layer = ReadoutLayer(dim=1, dt=0.1, source_func_type="Tanh")
    phi = torch.linspace(-0.5, 0.5, steps=4).view(1, 4, 1)
    history = layer.forward(phi)

    expected = torch.tanh(phi)
    zeros = torch.zeros(1, 1, 1)
    expected_history = torch.cat([zeros, expected], dim=1)
    assert torch.allclose(history, expected_history)
