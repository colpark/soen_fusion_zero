import torch

from soen_toolkit.core.layers.virtual import LeakyGRULayer


def test_leakygru_layer_forward_shape_and_initial_state() -> None:
    layer = LeakyGRULayer(
        dim=4,
        dt=1.0,
        tau_init=(2.0, 20.0),
        tau_spacing="geometric",
        candidate_diag="zero",
        train_alpha=True,
    )

    phi = torch.randn(2, 3, 4)
    history = layer(phi)

    assert history.shape == (2, 4, 4)
    assert torch.allclose(history[:, 0, :], torch.zeros(2, 4), atol=1e-6)

    init = torch.tensor([1.0, 2.0, 3.0, 4.0])
    history2 = layer(phi, initial_state=init)
    assert history2.shape == (2, 4, 4)
    assert torch.allclose(history2[:, 0, :], init.view(1, -1).expand(2, -1), atol=1e-6)


def test_leakygru_freeze_and_unfreeze_alpha() -> None:
    layer = LeakyGRULayer(dim=3, dt=1.0, train_alpha=True)
    assert layer.core.bias_z.requires_grad is True

    layer.freeze_alpha()
    assert layer.core.bias_z.requires_grad is False

    layer.unfreeze_alpha()
    assert layer.core.bias_z.requires_grad is True


def test_leakygru_candidate_diag_constraint() -> None:
    layer_zero = LeakyGRULayer(dim=5, dt=1.0, candidate_diag="zero")
    with torch.no_grad():
        layer_zero.core.weight_hh.normal_(0.0, 1.0)
    W0 = layer_zero.core._effective_candidate_hh()
    assert torch.allclose(torch.diagonal(W0), torch.zeros(5, device=W0.device, dtype=W0.dtype), atol=1e-6)

    layer_one = LeakyGRULayer(dim=5, dt=1.0, candidate_diag="one")
    with torch.no_grad():
        layer_one.core.weight_hh.normal_(0.0, 1.0)
    W1 = layer_one.core._effective_candidate_hh()
    assert torch.allclose(torch.diagonal(W1), torch.ones(5, device=W1.device, dtype=W1.dtype), atol=1e-6)




