import copy

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
from soen_toolkit.utils.quantization import generate_uniform_codebook


def test_quantise_preserves_mask_zeros_and_snaps_diag() -> None:
    # One-to-one mask yields zeros off diagonal; ensure they remain zero after quantise
    m = build_small_model(dims=(4, 5), connectivity_type="one_to_one", init="constant", init_value=0.12)
    W_before = m.connections["J_0_to_1"].detach().clone()
    assert W_before.shape == (5, 4)

    # Quantise copy (not in-place)
    m_q = m.quantise(bits=1, min=-0.24, max=0.24, in_place=False)
    W_after = m_q.connections["J_0_to_1"].detach()

    # Off-diagonal zeros should remain zero
    (torch.ones_like(W_after) - torch.eye(W_after.shape[0], W_after.shape[1]))
    # off_diag built is rectangular; construct properly
    off_diag_mask = torch.ones_like(W_after)
    for i in range(min(W_after.shape)):
        off_diag_mask[i, i] = 0
    assert torch.count_nonzero(W_after * (off_diag_mask > 0)).item() == 0

    # Diagonal values should snap to nearest codebook level (±0.24, 0)
    cb = generate_uniform_codebook(-0.24, 0.24, 3)
    diag_vals = torch.diag(W_after[: W_after.shape[1], : W_after.shape[1]]) if W_after.shape[0] >= W_after.shape[1] else torch.diag(W_after)
    for v in diag_vals:
        diffs = (cb - v).abs()
        j = int(diffs.argmin().item())
        assert torch.isclose(v, cb[j], atol=1e-7)

    # Original model should be unchanged
    assert torch.allclose(m.connections["J_0_to_1"].detach(), W_before)


def test_quantise_in_place_and_copy_equivalence_for_targeted_connection() -> None:
    m = build_small_model(dims=(3, 3), connectivity_type="dense", init="constant", init_value=0.1)
    W_before = m.connections["J_0_to_1"].detach().clone()
    generate_uniform_codebook(-0.24, 0.24, 5)

    # Non in-place
    m_copy = m.quantise(levels=5, min=-0.24, max=0.24, connections=["J_0_to_1"], in_place=False)
    W_copy = m_copy.connections["J_0_to_1"].detach()

    # In-place
    m_inp = copy.deepcopy(m)
    _ = m_inp.quantise(levels=5, min=-0.24, max=0.24, connections=["J_0_to_1"], in_place=True)
    W_inp = m_inp.connections["J_0_to_1"].detach()

    # Both snapped to same codebook values
    assert torch.allclose(W_copy, W_inp)

    # Original unchanged
    assert torch.allclose(m.connections["J_0_to_1"].detach(), W_before)


def test_quantise_connections_filter_and_non_learnable() -> None:
    # Model with J_0_to_0 (make it non-learnable) and J_0_to_1 (learnable)
    d = 3
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": d}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": d}),
    ]
    conns = [
        ConnectionConfig(from_layer=0, to_layer=0, connection_type="dense", params={"init": "constant", "value": 0.11}, learnable=False),
        ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={"init": "constant", "value": 0.11}, learnable=True),
    ]
    sim = SimulationConfig(dt=37, input_type="flux")
    m = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

    W_int_before = m.connections["J_0_to_0"].detach().clone()
    W_ext_before = m.connections["J_0_to_1"].detach().clone()

    # Quantise only external, exclude non-learnable by default
    m_q = m.quantise(min=-0.24, max=0.24, bits=2, connections=["J_0_to_1"], include_non_learnable=False, in_place=False)
    assert torch.allclose(m_q.connections["J_0_to_0"].detach(), W_int_before)
    assert not torch.allclose(m_q.connections["J_0_to_1"].detach(), W_ext_before)

    # Now include non-learnable and target internal explicitly
    m_q2 = m.quantise(min=-0.24, max=0.24, bits=2, connections=["J_0_to_0"], include_non_learnable=True, in_place=False)
    assert not torch.allclose(m_q2.connections["J_0_to_0"].detach(), W_int_before)


def test_quantise_preserves_masks_after_forward() -> None:
    # Build with mask and run forward, then quantise and ensure zeros persist
    m = build_small_model(dims=(4, 5), connectivity_type="one_to_one", init="constant", init_value=0.07)
    x = make_random_series(batch=2, seq_len=3, dim=4, seed=5)
    _ = m(x)
    m_q = m.quantise(min=-0.24, max=0.24, bits=1, in_place=False)
    W = m_q.connections["J_0_to_1"].detach()

    # Off-diagonal must be zero
    off_diag_mask = torch.ones_like(W)
    for i in range(min(W.shape)):
        off_diag_mask[i, i] = 0
    assert torch.count_nonzero(W * (off_diag_mask > 0)).item() == 0
