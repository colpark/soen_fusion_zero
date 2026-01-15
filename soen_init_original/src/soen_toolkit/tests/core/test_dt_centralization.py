import torch

from soen_toolkit.core import LayerConfig, SimulationConfig, SOENModelCore


def _build_single_dendrite_model(dim=4, *, dt=37, learnable=False, source_func="Tanh"):
    layers = [
        LayerConfig(
            layer_id=0,
            layer_type="SingleDendrite",
            params={
                "dim": int(dim),
                "solver": "FE",
                "source_func": source_func,
                # Amplify dynamics so dt has a visible effect
                "gamma_plus": 0.001,
                "gamma_minus": 0.001,
                "phi_offset": 0.23,
                "bias_current": 1.7,
            },
        ),
    ]
    sim = SimulationConfig(dt=float(dt), dt_learnable=bool(learnable), input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=[])


def test_learnable_dt_receives_gradients() -> None:
    torch.manual_seed(0)
    m = _build_single_dendrite_model(dim=3, dt=37, learnable=True, source_func="Tanh")
    assert isinstance(m.dt, torch.Tensor)
    assert m.dt.requires_grad

    x = torch.randn(2, 5, 3) * 3.0
    y, _ = m(x)
    loss = y.sum()
    loss.backward()

    assert m.dt.grad is not None, "Expected gradient on model.dt when learnable"
    assert torch.isfinite(m.dt.grad), "Gradient on model.dt should be finite"


def test_set_dt_changes_dynamics_nonlearnable() -> None:
    torch.manual_seed(0)
    m = _build_single_dendrite_model(dim=2, dt=37, learnable=False, source_func="Tanh")
    x = torch.randn(1, 4, 2) * 3.0

    y1, _ = m(x)
    m.set_dt(155.8, propagate_to_layers=True)
    y2, _ = m(x)

    # Dynamics should differ when dt changes (use a strict difference check)
    max_diff = (y2 - y1).abs().max().item()
    assert max_diff > 1e-6, f"Changing dt should alter the state trajectory (max diff {max_diff})"
