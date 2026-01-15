import copy

import pytest
import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.quantization import generate_uniform_codebook, snapped_copy


def build_single_dendrite(dim=3):
    layers = [
        LayerConfig(layer_id=0, layer_type="SingleDendrite", params={"dim": dim}),
    ]
    sim = SimulationConfig(dt=37, input_type="flux")
    # Include internal connectivity via connection config
    conns = [ConnectionConfig(from_layer=0, to_layer=0, connection_type="dense", params={}, learnable=True)]
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


@pytest.mark.parametrize("builder", [build_single_dendrite])
def test_internal_j_qat_ste_matches_hard_snapped_dendrite(builder) -> None:
    torch.manual_seed(0)
    dim = 3
    m = builder(dim)

    # Input
    x = torch.randn(2, 5, dim)

    # Snapshot internal and codebook
    J_int = m.connections["J_0_to_0"].detach().clone()
    cb = generate_uniform_codebook(-0.24, 0.24, 9).to(J_int.device)

    # Hard baseline: deep copy then snap internal_0 only
    m_hard = copy.deepcopy(m)
    with torch.no_grad():
        m_hard.connections["J_0_to_0"].copy_(snapped_copy(J_int, cb))

    # QAT enabled
    m.enable_qat_ste(min_val=-0.24, max_val=0.24, bits=3, connections=["J_0_to_0"])
    y_qat, _ = m(x)
    y_hard, _ = m_hard(x)

    assert torch.allclose(y_qat, y_hard, atol=1e-6)
    assert torch.allclose(m.connections["J_0_to_0"].detach(), J_int)
