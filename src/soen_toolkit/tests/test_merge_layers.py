import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.merge_layers import MergeSpec, apply_merge_layers


def _sd_params(dim: int, solver: str = "FE"):
    return {
        "dim": dim,
        "solver": solver,
        "source_func": "Tanh",  # simple deterministic source
        # small deterministic params for easy assertions
        "phi_offset": {"distribution": "constant", "params": {"value": 0.0}},
        "bias_current": {"distribution": "constant", "params": {"value": 1.7}},
        "gamma_plus": {"distribution": "constant", "params": {"value": 0.01}},
        "gamma_minus": {"distribution": "constant", "params": {"value": 0.02}},
    }


def _mk_sd_layer(layer_id: int, dim: int) -> LayerConfig:
    return LayerConfig(layer_id=layer_id, layer_type="SingleDendrite", params=_sd_params(dim))


def _conn(from_id: int, to_id: int, t: str, rows: int, cols: int, fill: float) -> ConnectionConfig:
    # Use constant init; we'll override weight after build
    return ConnectionConfig(
        from_layer=from_id,
        to_layer=to_id,
        connection_type=t,
        params={"init": "constant", "value": fill},
        learnable=True,
    )


def _assign(model: SOENModelCore, src: int, dst: int, mat: torch.Tensor) -> None:
    key = f"J_{src}_to_{dst}"
    assert key in model.connections, f"missing {key}"
    with torch.no_grad():
        p = model.connections[key]
        assert tuple(p.shape) == tuple(mat.shape)
        p.copy_(mat.to(device=p.device, dtype=p.dtype))


def test_merge_three_sd_layers_preserves_weights_and_params() -> None:
    # Build a 5-layer toy model:
    # 0 (input) -> 1,2,3 (SD group) -> 4 (readout-like)
    sim = SimulationConfig(dt=1.0)
    layers = [
        _mk_sd_layer(0, 2),
        _mk_sd_layer(1, 3),
        _mk_sd_layer(2, 2),
        _mk_sd_layer(3, 1),
        _mk_sd_layer(4, 4),
    ]

    conns = [
        _conn(0, 1, "dense", rows=3, cols=2, fill=0.0),
        _conn(0, 2, "dense", rows=2, cols=2, fill=0.0),
        _conn(0, 3, "dense", rows=1, cols=2, fill=0.0),
        _conn(1, 1, "dense", rows=3, cols=3, fill=0.0),
        _conn(1, 2, "dense", rows=2, cols=3, fill=0.0),
        _conn(2, 3, "dense", rows=1, cols=2, fill=0.0),
        _conn(3, 3, "dense", rows=1, cols=1, fill=0.0),
        _conn(1, 4, "dense", rows=4, cols=3, fill=0.0),
        _conn(2, 4, "dense", rows=4, cols=2, fill=0.0),
        _conn(3, 4, "dense", rows=4, cols=1, fill=0.0),
    ]

    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

    # Manually set numeric weights to distinctive blocks for easy verification
    J_01 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
    J_02 = torch.tensor([[7.0, 8.0], [9.0, 10.0]])  # 2x2
    J_03 = torch.tensor([[11.0, 12.0]])  # 1x2
    _assign(model, 0, 1, J_01)
    _assign(model, 0, 2, J_02)
    _assign(model, 0, 3, J_03)

    J_11 = torch.eye(3) * 0.1
    J_12 = torch.full((2, 3), 0.2)
    J_23 = torch.full((1, 2), 0.3)
    J_33 = torch.full((1, 1), 0.4)
    _assign(model, 1, 1, J_11)
    _assign(model, 1, 2, J_12)
    _assign(model, 2, 3, J_23)
    _assign(model, 3, 3, J_33)

    J_14 = torch.arange(12, dtype=torch.float32).view(4, 3)  # 4x3
    J_24 = torch.full((4, 2), -1.0)
    J_34 = torch.full((4, 1), 5.0)
    _assign(model, 1, 4, J_14)
    _assign(model, 2, 4, J_24)
    _assign(model, 3, 4, J_34)

    # Merge group {1,2,3} into new layer id 10 with order [1,2,3]
    spec = MergeSpec(group_ids=[1, 2, 3], new_layer_id=10, node_order=[1, 2, 3], preserve_state=True)
    result = apply_merge_layers(model, spec)
    m2 = result.model

    # Check dims: new layer has 3+2+1 = 6
    dims = {cfg.layer_id: cfg.params["dim"] for cfg in m2.layers_config}
    assert dims[10] == 6
    assert 1 not in dims
    assert 2 not in dims
    assert 3 not in dims

    # Internal block must assemble as diag and cross blocks: shape 6x6
    J_10_10 = m2.connections["J_10_to_10"].detach().cpu()
    assert tuple(J_10_10.shape) == (6, 6)
    # Top-left 3x3 block == J_11
    assert torch.allclose(J_10_10[0:3, 0:3], J_11)
    # Block (rows 3:5, cols 0:3) == J_12
    assert torch.allclose(J_10_10[3:5, 0:3], J_12)
    # Block (rows 5:6, cols 3:5) == J_23
    assert torch.allclose(J_10_10[5:6, 3:5], J_23)
    # Bottom-right 1x1 == J_33
    assert torch.allclose(J_10_10[5:6, 5:6], J_33)

    # Inbound 0->10 is vertical stack of (0->1,0->2,0->3)
    J_0_10 = m2.connections["J_0_to_10"].detach().cpu()
    stacked = torch.vstack([J_01, J_02, J_03])
    assert torch.allclose(J_0_10, stacked)

    # Outbound 10->4 is horizontal concat of (1->4,2->4,3->4)
    J_10_4 = m2.connections["J_10_to_4"].detach().cpu()
    cat = torch.hstack([J_14, J_24, J_34])
    assert torch.allclose(J_10_4, cat)

    # Node-wise params concatenated: sample gamma_plus
    idx_by_id = {cfg.layer_id: i for i, cfg in enumerate(model.layers_config)}
    g1 = model.layers[idx_by_id[1]].gamma_plus.detach().cpu()
    g2 = model.layers[idx_by_id[2]].gamma_plus.detach().cpu()
    g3 = model.layers[idx_by_id[3]].gamma_plus.detach().cpu()

    idx2_by_id = {cfg.layer_id: i for i, cfg in enumerate(m2.layers_config)}
    g_merged = m2.layers[idx2_by_id[10]].gamma_plus.detach().cpu()
    assert g_merged.shape[0] == 6
    assert torch.allclose(g_merged, torch.cat([g1, g2, g3], dim=0))


def test_merge_preserves_bidirectional_cross_layer_weights() -> None:
    """Test that merging layers with bidirectional connections preserves all weights.

    This test covers the bug where connectivity.weight (same tensor as model.connections)
    was incorrectly being overwritten with a block-diagonal matrix, zeroing out
    cross-layer weights like 1->2 and 2->1.
    """
    sim = SimulationConfig(dt=1.0)

    # Two SD layers with bidirectional connections
    layers = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 2}),
        _mk_sd_layer(1, 4),
        _mk_sd_layer(2, 4),
        LayerConfig(layer_id=3, layer_type="Linear", params={"dim": 2}),
    ]

    conns = [
        _conn(0, 1, "dense", rows=4, cols=2, fill=0.0),
        _conn(1, 1, "dense", rows=4, cols=4, fill=0.0),  # self-connection layer 1
        _conn(2, 2, "dense", rows=4, cols=4, fill=0.0),  # self-connection layer 2
        _conn(1, 2, "dense", rows=4, cols=4, fill=0.0),  # 1 -> 2 (forward)
        _conn(2, 1, "dense", rows=4, cols=4, fill=0.0),  # 2 -> 1 (backward)
        _conn(2, 3, "dense", rows=2, cols=4, fill=0.0),
    ]

    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

    # Set distinctive weights for each connection
    J_11 = torch.eye(4) * 0.1  # diagonal self
    J_22 = torch.eye(4) * 0.2  # diagonal self
    J_12 = torch.full((4, 4), 0.3)  # cross 1->2
    J_21 = torch.full((4, 4), 0.4)  # cross 2->1

    _assign(model, 1, 1, J_11)
    _assign(model, 2, 2, J_22)
    _assign(model, 1, 2, J_12)
    _assign(model, 2, 1, J_21)

    # Merge layers 1 and 2 into new layer 1
    spec = MergeSpec(group_ids=[1, 2], new_layer_id=1, node_order=[1, 2], preserve_state=True)
    result = apply_merge_layers(model, spec)
    m2 = result.model

    # Internal block J_1_to_1 should be 8x8 with all 4 sub-blocks preserved
    J_merged = m2.connections["J_1_to_1"].detach().cpu()
    assert tuple(J_merged.shape) == (8, 8)

    # Layout:
    #   [0:4, 0:4] = J_11 (1->1)
    #   [0:4, 4:8] = J_21 (2->1)
    #   [4:8, 0:4] = J_12 (1->2)
    #   [4:8, 4:8] = J_22 (2->2)

    # Check all 4 blocks are preserved
    assert torch.allclose(J_merged[0:4, 0:4], J_11), "Block 1->1 not preserved"
    assert torch.allclose(J_merged[0:4, 4:8], J_21), "Block 2->1 not preserved"
    assert torch.allclose(J_merged[4:8, 0:4], J_12), "Block 1->2 not preserved"
    assert torch.allclose(J_merged[4:8, 4:8], J_22), "Block 2->2 not preserved"

    # Verify total nonzero count: each block has 4 or 16 nonzero depending on pattern
    # J_11: 4 nonzero (diagonal), J_22: 4 nonzero (diagonal)
    # J_12: 16 nonzero (full), J_21: 16 nonzero (full)
    # Total: 4 + 4 + 16 + 16 = 40
    assert torch.count_nonzero(J_merged).item() == 40
