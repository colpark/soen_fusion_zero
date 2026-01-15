import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)


def _build_model_dense_3x4_learning() -> SOENModelCore:
    # 3 -> 4 dense inter-layer, learnable; no internal J
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": 3}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": 4}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "constant", "value": 0.1},
            learnable=True,
        ),
    ]
    sim = SimulationConfig(dt=1.0, input_type="flux")
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def _build_model_one_to_one_5x3_learning() -> SOENModelCore:
    # 5 -> 3 one_to_one inter-layer, learnable; 3 ones on diagonal
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": 5}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": 3}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="one_to_one",
            params={"init": "constant", "value": 0.1},
            learnable=True,
        ),
    ]
    sim = SimulationConfig(dt=1.0, input_type="flux")
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def _build_model_with_internal(dim: int = 4, learn_internal: bool = True) -> SOENModelCore:
    # Single layer with internal J (v2 layers) by specifying self-connection in connections
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": dim}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": dim}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "constant", "value": 0.1},
            learnable=True,
        ),
        ConnectionConfig(
            from_layer=0,
            to_layer=0,
            connection_type="one_to_one",
            params={"init": "constant", "value": 0.2},
            learnable=learn_internal,
        ),
    ]
    sim = SimulationConfig(dt=1.0, input_type="flux")
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


@torch.no_grad()
def test_mask_aware_trainable_for_dense_matches_numel() -> None:
    m = _build_model_dense_3x4_learning()
    info = m.compute_summary()
    k = info["kpis"]
    # Dense 4x3 has 12 mask ones; trainable should equal 12 + any other learnable params (none expected for RNN defaults)
    # Count expected: exact mask sum for J plus zero for non-connection learnables in this minimal model
    J = m.connections["J_0_to_1"]
    assert J.requires_grad
    expected = 12
    assert k["trainable_parameters"] >= expected
    # The masked portion should be 12; ensure not larger than numel for connection part
    assert k["trainable_parameters"] <= expected + (sum(p.numel() for p in m.parameters() if p.requires_grad) - J.numel())


@torch.no_grad()
def test_mask_aware_trainable_for_one_to_one_uses_mask_ones() -> None:
    m = _build_model_one_to_one_5x3_learning()
    info = m.compute_summary()
    k = info["kpis"]
    # One-to-one from 5 to 3 gives min(5,3)=3 ones
    expected_mask_ones = 3
    # Baseline unmasked learnables for connection is numel=3x5=15
    unmasked = m.connections["J_0_to_1"].numel()
    assert expected_mask_ones < unmasked
    # Mask-aware should be strictly less than unmasked total (ignoring other params)
    assert k["trainable_parameters"] < unmasked + (sum(p.numel() for p in m.parameters() if p.requires_grad) - unmasked)
    # And at least the mask ones
    assert k["trainable_parameters"] >= expected_mask_ones


@torch.no_grad()
def test_mask_aware_internal_self_connection_counts_mask_ones() -> None:
    m = _build_model_with_internal(dim=4, learn_internal=True)
    info = m.compute_summary()
    k = info["kpis"]
    # Internal one_to_one on 4x4 contributes 4 masked ones
    expected_internal = 4
    assert k["trainable_parameters"] >= expected_internal
