import json
import os
from pathlib import Path
import tempfile

import torch

from soen_toolkit.core import ConnectionConfig, LayerConfig, SimulationConfig
from soen_toolkit.core.soen_model_core import SOENModelCore
from soen_toolkit.utils.model_tools import export_model_to_json, model_from_json


def _tmp_path(tmp_path_factory) -> Path:
    return Path(tmp_path_factory.mktemp("soen_json_roundtrip"))


def _build_comprehensive_model() -> SOENModelCore:
    # Simulation config with tracking enabled to touch optional paths
    sim = SimulationConfig(
        dt=50.0,
        dt_learnable=False,
        input_type="flux",
        track_power=True,
        track_phi=True,
        track_g=True,
        track_s=True,
    )

    # Two linear layers surrounding a SingleDendrite to exercise inter/intra connections
    layers = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 6}),
        LayerConfig(
            layer_id=1,
            layer_type="SingleDendrite",
            params={
                "dim": 10,
                "solver": "FE",
                # Use table-based source function to ensure buffers (g_table) are present
                "source_func": "RateArray",
                # Provide some distributions to hit init code paths
                "gamma_plus": {"distribution": "fan_out", "params": {"scale": 303.85}},
                "gamma_minus": {"distribution": "loglinear", "params": {"min": 1e-4, "max": 1e-2}},
                "phi_offset": {"distribution": "uniform", "params": {"min": -0.1, "max": 0.1}},
                "bias_current": {"distribution": "constant", "params": {"value": 1.7}},
                # Per-parameter learnability mix
                "learnable_params": {"gamma_plus": True, "gamma_minus": True},
            },
        ),
        LayerConfig(layer_id=2, layer_type="Linear", params={"dim": 2}),
    ]

    connections = [
        # Inter-layer
        ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={"min": -0.2, "max": 0.2}),
        ConnectionConfig(from_layer=1, to_layer=2, connection_type="dense", params={"min": -0.2, "max": 0.2}),
        # Intra-layer (internal)
        ConnectionConfig(from_layer=1, to_layer=1, connection_type="dense", params={"min": -0.24, "max": 0.24}, learnable=False),
    ]

    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=connections)

    # Touch forward once to ensure buffers/shapes are valid and constraints applied
    with torch.no_grad():
        x = torch.zeros(1, 5, 6)
        model(x)

    return model


def _state_dict_signature(m: SOENModelCore) -> dict:
    sig = {}
    for k, v in m.state_dict().items():
        if isinstance(v, torch.Tensor):
            sig[k] = {"shape": tuple(v.shape), "sum": float(v.sum().detach().cpu().item())}
        else:
            sig[k] = str(type(v))
    return sig


def test_json_roundtrip_matches_soen(tmp_path_factory) -> None:
    tmpdir = _tmp_path(tmp_path_factory)
    # Input layer warns about unsupported g tracking; capture to keep test output clean
    model = _build_comprehensive_model()

    # 1) Export to .soen (torch)
    soen_path = tmpdir / "model.soen"
    model.save(str(soen_path))

    # 2) Export to JSON string + file
    json_path = tmpdir / "model.json"
    s = export_model_to_json(model, str(json_path))
    data = json.loads(s)
    assert "layers" in data
    assert "connections" in data
    # Ensure buffers were exported for the RateArray source function
    # Find the layer_1 entry
    assert any("buffers" in v for k, v in data["layers"].items())

    # 3) Load back from JSON and compare against original .soen signature
    model_from_j = model_from_json(str(json_path))

    # Compare key aspects
    assert isinstance(model_from_j, SOENModelCore)
    assert len(model_from_j.layers_config) == len(model.layers_config)
    assert len(model_from_j.connections_config) == len(model.connections_config)

    sig_a = _state_dict_signature(model)
    sig_b = _state_dict_signature(model_from_j)

    # shapes must match exactly; sums should be close (allow tiny float diff)
    assert sig_a.keys() == sig_b.keys()
    for k in sig_a:
        assert sig_a[k]["shape"] == sig_b[k]["shape"], f"shape mismatch for {k}"
        assert abs(sig_a[k]["sum"] - sig_b[k]["sum"]) < 1e-5, f"value mismatch for {k}"

    # 4) Export JSON-loaded model back to .soen and reload to ensure both paths are consistent
    soen2 = tmpdir / "model_from_json.soen"
    model_from_j.save(str(soen2))
    model_from_soen2 = SOENModelCore.load(str(soen2), show_logs=False)

    sig_c = _state_dict_signature(model_from_soen2)
    assert sig_b.keys() == sig_c.keys()
    for k in sig_b:
        assert sig_b[k]["shape"] == sig_c[k]["shape"], f"shape mismatch on second save for {k}"
        assert abs(sig_b[k]["sum"] - sig_c[k]["sum"]) < 1e-5


from soen_toolkit.tests.utils.test_helpers_fixture import (  # noqa: E402
    build_small_model,
)


def test_export_model_to_json_contains_required_sections() -> None:
    m = build_small_model(dims=(2, 3), connectivity_type="dense", init="constant", init_value=0.21)
    s = export_model_to_json(m)
    data = json.loads(s)

    assert "version" in data
    assert "metadata" in data
    assert "simulation" in data
    assert isinstance(data["simulation"], dict)
    assert "layers" in data
    assert isinstance(data["layers"], dict)
    assert "connections" in data
    assert isinstance(data["connections"], dict)
    assert "config" in data["connections"]
    assert "matrices" in data["connections"]
    assert "global_matrix" in data["connections"]


def test_model_from_json_round_trip() -> None:
    m = build_small_model(dims=(3, 3), connectivity_type="dense", init="constant", init_value=0.4, with_internal_first=True)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model.json")
        export_model_to_json(m, p)
        m2 = model_from_json(p)

    # Compare sim config
    assert m2.sim_config.dt == m.sim_config.dt
    # Compare connections keys and values
    assert set(m2.connections.keys()) == set(m.connections.keys())
    for k in m.connections:
        assert torch.allclose(m.connections[k].detach(), m2.connections[k].detach())
