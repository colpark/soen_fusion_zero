import copy

import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.quantization import generate_uniform_codebook, snapped_copy


def build_two_layer_rnn(dim=4):
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": dim}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": dim}),
    ]
    conns = [
        ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}, learnable=True),
    ]
    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def test_qat_applies_to_inter_layer_without_mutation() -> None:
    torch.manual_seed(0)
    m = build_two_layer_rnn(dim=3)

    # Cache original weights
    J_orig = m.connections["J_0_to_1"].detach().clone()

    # Input
    x = torch.randn(2, 5, m.layer_nodes[0])

    # Codebook
    cb = generate_uniform_codebook(-0.24, 0.24, 9).to(J_orig.device)

    # Hard-snapped baseline: exact clone of model, only snapping targeted connection
    m_hard = copy.deepcopy(m)
    with torch.no_grad():
        m_hard.connections["J_0_to_1"].copy_(snapped_copy(J_orig, cb))

    # Run with QAT active for the inter-layer connection on the original model
    m.enable_qat_ste(min_val=-0.24, max_val=0.24, bits=3, connections=["J_0_to_1"])
    y_qat, _ = m(x)

    y_hard, _ = m_hard(x)

    assert torch.allclose(y_qat, y_hard, atol=1e-6)
    # Ensure parameter was not mutated by QAT forward
    assert torch.allclose(m.connections["J_0_to_1"].detach(), J_orig)


def test_qat_applies_to_internal_j_without_mutation_rnn_layer() -> None:
    torch.manual_seed(0)
    dim = 3
    # Single RNN layer with internal_J as layer parameter
    layers = [LayerConfig(layer_id=0, layer_type="RNN", params={"dim": dim, "internal_J": torch.randn(dim, dim)})]
    sim = SimulationConfig(dt=37, input_type="flux")
    model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=[])

    x = torch.randn(2, 5, dim)
    # Snapshot the layer parameter
    J_int_before = model.layers[0].internal_J.detach().clone()

    cb = generate_uniform_codebook(-0.24, 0.24, 9).to(J_int_before.device)

    # Hard-snapped baseline: exact clone, snap only internal_J
    model_hard = copy.deepcopy(model)
    with torch.no_grad():
        model_hard.layers[0].internal_J.copy_(snapped_copy(J_int_before, cb))

    # Enable QAT for the internal connection key name; flags propagate to layer
    model.enable_qat_ste(min_val=-0.24, max_val=0.24, bits=3, connections=["internal_0"])
    y_qat, _ = model(x)

    y_hard, _ = model_hard(x)

    assert torch.allclose(y_qat, y_hard, atol=1e-6)
    # Ensure parameter not mutated
    assert torch.allclose(model.layers[0].internal_J.detach(), J_int_before)
