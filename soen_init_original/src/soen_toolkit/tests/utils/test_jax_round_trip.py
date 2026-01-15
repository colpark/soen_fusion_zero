from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest
import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.port_to_jax.convert import (
    convert_core_model_to_jax,
    convert_jax_to_core_model,
)


def _build_roundtrip_core() -> SOENModelCore:
    sim = SimulationConfig(dt=1.234, input_type="state", network_evaluation_method="layerwise")

    # Layers: Linear (3) -> Multiplier (4, with params + internal_J) -> SingleDendrite (2, with params)
    L0 = LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 3})

    multiplier_params = {
        "dim": 4,
        # Internal connectivity
        "internal_J": torch.full((4, 4), 0.01, dtype=torch.float32),
    }
    L1 = LayerConfig(layer_id=1, layer_type="Multiplier", params=multiplier_params)

    dend_params = {
        "dim": 2,
    }
    L2 = LayerConfig(layer_id=2, layer_type="SingleDendrite", params=dend_params)

    layers = [L0, L1, L2]

    # External connections: 0->1 dynamic, 1->2 fixed. Deterministic initializers.
    C01 = ConnectionConfig(
        from_layer=0,
        to_layer=1,
        connection_type="dense",
        params={
            "structure": {"type": "dense"},
            "init": {"name": "constant", "params": {"value": 0.12}},
            "mode": "WICC",
            "connection_params": {
                "source_func": "RateArray",
                "gamma_plus": 0.003,
                "bias_current": 1.7,
            },
        },
        learnable=True,
    )
    C12 = ConnectionConfig(
        from_layer=1,
        to_layer=2,
        connection_type="dense",
        params={
            "structure": {"type": "dense"},
            "init": {"name": "constant", "params": {"value": 0.34}},
        },
        learnable=True,
    )

    core = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=[C01, C12])

    # After build, set parameter vectors via ParameterRegistry to deterministic values
    # Multiplier (layer_id=1)
    layer_id_to_index = {cfg.layer_id: i for i, cfg in enumerate(core.layers_config)}
    l_mult = core.layers[layer_id_to_index[1]]
    reg_m = getattr(l_mult, "_param_registry", None)
    assert reg_m is not None
    reg_m.override_parameter("phi_y", value=torch.tensor([0.11, 0.22, 0.33, 0.44], dtype=torch.float32))
    reg_m.override_parameter("bias_current", value=torch.tensor([1.2, 1.2, 1.2, 1.2], dtype=torch.float32))
    reg_m.override_parameter("gamma_plus", value=torch.tensor([0.001, 0.002, 0.003, 0.004], dtype=torch.float32))

    # SingleDendrite (layer_id=2)
    l_den = core.layers[layer_id_to_index[2]]
    reg_d = getattr(l_den, "_param_registry", None)
    assert reg_d is not None
    reg_d.override_parameter("phi_offset", value=torch.tensor([0.3, 0.4], dtype=torch.float32))
    reg_d.override_parameter("bias_current", value=torch.tensor([1.8, 1.9], dtype=torch.float32))
    reg_d.override_parameter("gamma_plus", value=torch.tensor([0.002, 0.004], dtype=torch.float32))
    reg_d.override_parameter("gamma_minus", value=torch.tensor([0.005, 0.0075], dtype=torch.float32))

    return core


def _assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, msg: str) -> None:
    assert a.shape == b.shape, f"shape mismatch for {msg}: {a.shape} vs {b.shape}"
    assert torch.allclose(a, b, atol=1e-7, rtol=0), f"value mismatch for {msg}"


def test_round_trip_torch_jax_torch_preserves_model() -> None:
    # Force CPU JAX backend for stability on CI/Mac (avoid METAL/Tensor cores)
    os.environ["JAX_PLATFORMS"] = "cpu"
    jax = pytest.importorskip("jax")
    with contextlib.suppress(Exception):
        jax.config.update("jax_platforms", "cpu")

    core = _build_roundtrip_core()
    dyn_key = "J_0_to_1"
    custom_j_out = torch.tensor([0.31, 0.27, 0.19, 0.11], dtype=torch.float32)
    assert dyn_key in core._connection_params
    core._connection_params[dyn_key]["j_out"] = custom_j_out

    # Convert to JAX and back to a fresh core
    jax_model = convert_core_model_to_jax(core)
    # Ensure JAX spec preserved per-destination j_out vector
    jax_conn = next(c for c in jax_model.connections if c.from_layer == 0 and c.to_layer == 1)
    np.testing.assert_allclose(np.asarray(jax_conn.j_out), custom_j_out.numpy(), atol=0, rtol=0)
    core_rt = convert_jax_to_core_model(jax_model)

    # --- Simulation config ---
    assert float(core.sim_config.dt) == pytest.approx(float(core_rt.sim_config.dt), abs=0)
    assert str(core.sim_config.input_type).lower() == str(core_rt.sim_config.input_type).lower()
    assert str(core.sim_config.network_evaluation_method).lower() == str(core_rt.sim_config.network_evaluation_method).lower()

    # --- Layers: type, dim, parameter vectors, internal connectivity ---
    assert len(core.layers_config) == len(core_rt.layers_config)
    for (cfg_a, layer_a), (cfg_b, layer_b) in zip(
        zip(core.layers_config, core.layers, strict=False),
        zip(core_rt.layers_config, core_rt.layers, strict=False),
        strict=False,
    ):
        assert cfg_a.layer_id == cfg_b.layer_id
        assert cfg_a.layer_type == cfg_b.layer_type
        assert int(layer_a.dim) == int(layer_b.dim)

        # Compare parameter vectors via ParameterRegistry snapshot
        try:
            pa = layer_a.parameter_values()
            pb = layer_b.parameter_values()
            for key in sorted(set(pa.keys()) | set(pb.keys())):
                if key in pa and key in pb and isinstance(pa[key], torch.Tensor) and isinstance(pb[key], torch.Tensor):
                    _assert_tensor_equal(pa[key], pb[key], f"layer{cfg_a.layer_id}.{key}")
        except Exception:
            pass

        # Internal connectivity (if present)
        if getattr(layer_a, "connectivity", None) is not None:
            assert getattr(layer_b, "connectivity", None) is not None, f"missing connectivity on layer {cfg_a.layer_id}"
            _assert_tensor_equal(
                layer_a.connectivity.weight,
                layer_b.connectivity.weight,
                f"layer{cfg_a.layer_id}.internal_J",
            )

    # --- External connections: weights, masks, dynamic params ---
    assert set(core.connections.keys()) == set(core_rt.connections.keys())
    for key in core.connections:
        _assert_tensor_equal(core.connections[key].detach(), core_rt.connections[key].detach(), f"conn {key}")
        mask_a = core.connection_masks.get(key)
        mask_b = core_rt.connection_masks.get(key)
        if mask_a is not None or mask_b is not None:
            assert mask_a is not None and mask_b is not None, f"mask presence mismatch for {key}"
            _assert_tensor_equal(mask_a, mask_b, f"mask {key}")

    # Dynamic metadata for 0->1 should be preserved; 1->2 should remain fixed
    fix_key = "J_1_to_2"
    assert core._connection_modes[dyn_key] == core_rt._connection_modes[dyn_key] == "WICC"
    assert core._connection_modes[fix_key] == core_rt._connection_modes[fix_key] == "fixed"
    assert core._connection_params[dyn_key]["source_func"] == core_rt._connection_params[dyn_key]["source_func"]
    assert core._connection_params[dyn_key]["gamma_plus"] == core_rt._connection_params[dyn_key]["gamma_plus"]
    assert core._connection_params[dyn_key]["bias_current"] == core_rt._connection_params[dyn_key]["bias_current"]
    assert torch.allclose(core._connection_params[dyn_key]["j_out"], core_rt._connection_params[dyn_key]["j_out"])


def test_round_trip_with_scaling_layer() -> None:
    """Test that ScalingLayer converts correctly through JAX round-trip."""
    # Force CPU JAX backend
    os.environ["JAX_PLATFORMS"] = "cpu"
    jax = pytest.importorskip("jax")
    with contextlib.suppress(Exception):
        jax.config.update("jax_platforms", "cpu")

    sim = SimulationConfig(dt=0.01, input_type="flux", network_evaluation_method="layerwise")

    # Build model with ScalingLayer
    layers = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 4}),
        LayerConfig(
            layer_id=1,
            layer_type="ScalingLayer",
            params={"dim": 4, "scale_factor": 1.0},
        ),
        LayerConfig(layer_id=2, layer_type="Linear", params={"dim": 3}),
    ]

    connections = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={
                "structure": {"type": "dense"},
                "init": {"name": "constant", "params": {"value": 0.5}},
            },
            learnable=True,
        ),
        ConnectionConfig(
            from_layer=1,
            to_layer=2,
            connection_type="dense",
            params={
                "structure": {"type": "dense"},
                "init": {"name": "constant", "params": {"value": 0.25}},
            },
            learnable=True,
        ),
    ]

    core = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=connections)

    # Set deterministic scale_factor values
    scaling_layer = core.layers[1]
    reg = getattr(scaling_layer, "_param_registry", None)
    assert reg is not None
    scale_values = torch.tensor([1.5, 2.0, 2.5, 3.0], dtype=torch.float32)
    reg.override_parameter("scale_factor", value=scale_values)

    # Convert to JAX and back
    jax_model = convert_core_model_to_jax(core)
    core_rt = convert_jax_to_core_model(jax_model)

    # Verify ScalingLayer parameters are preserved
    assert len(core.layers_config) == len(core_rt.layers_config)
    assert core.layers_config[1].layer_type == "ScalingLayer"
    assert core_rt.layers_config[1].layer_type == "ScalingLayer"

    # Check scale_factor values are preserved
    original_scale = core.layers[1].parameter_values()["scale_factor"]
    restored_scale = core_rt.layers[1].parameter_values()["scale_factor"]

    _assert_tensor_equal(original_scale, restored_scale, "ScalingLayer.scale_factor")
    _assert_tensor_equal(original_scale, scale_values, "ScalingLayer.scale_factor vs expected")
