import copy

import pytest
import torch

from soen_toolkit.core.layers.virtual import GRULayer, LSTMLayer, MinGRULayer, RNNLayer
from soen_toolkit.utils.quantization import generate_uniform_codebook, snapped_copy


@pytest.mark.parametrize(
    "layer_ctor",
    [RNNLayer, GRULayer, LSTMLayer, MinGRULayer],
)
def test_internal_j_qat_ste_matches_hard_snapped(layer_ctor) -> None:
    torch.manual_seed(0)
    dim = 4

    # Construct layer with internal connectivity
    init_matrix = torch.randn(dim, dim)
    layer = layer_ctor(dim=dim, dt=1.0, connectivity=init_matrix.clone())

    x = torch.randn(2, 6, dim)
    cb = generate_uniform_codebook(-0.24, 0.24, 9)

    # Hard-snapped baseline: deep copy layer, snap only internal_J
    layer_hard = copy.deepcopy(layer)
    with torch.no_grad():
        layer_hard.internal_J.copy_(snapped_copy(layer.internal_J, cb))

    # Enable QAT flags directly on the original layer
    layer._qat_ste_active = True
    layer._qat_internal_active = True
    layer._qat_codebook = cb

    y_qat = layer(x)
    y_hard = layer_hard(x)

    assert torch.allclose(y_qat, y_hard, atol=1e-6)
    # Ensure original internal_J not mutated by the QAT path
    assert torch.allclose(layer.internal_J.detach(), init_matrix)
