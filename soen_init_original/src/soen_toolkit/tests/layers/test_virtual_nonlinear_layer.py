import torch

from soen_toolkit.core.layers.virtual import NonLinearLayer


def test_nonlinear_layer_tanh_matches_direct_activation() -> None:
    layer = NonLinearLayer(dim=2, dt=1.0, source_func_type="Tanh")
    phi = torch.tensor([[[0.0, 1.0], [0.5, -0.5]]], dtype=torch.float32)

    history = layer(phi)

    assert history.shape == (1, 3, 2)
    expected = torch.tanh(phi)
    assert torch.allclose(history[:, 1:, :], expected, atol=1e-6)
    assert torch.allclose(history[:, 0, :], torch.zeros(1, 2), atol=1e-6)


def test_nonlinear_layer_respects_bias_current_parameter() -> None:
    layer = NonLinearLayer(dim=1, dt=1.0, source_func_type="Heaviside_state_dep")
    with torch.no_grad():
        layer.bias_current.fill_(2.0)

    phi = torch.tensor([[[0.25], [0.5]]], dtype=torch.float32)
    history = layer(phi)

    # Manual calculation of Heaviside response
    # squid_current = 2.0 (from fill)
    # bias_diff = clamp(2.0 - 1.06435066, min=1e-6) = 0.93564934
    # cos_term = abs(cos(pi * phi)) + 1e-6
    # disc = A * bias_diff^K - B * cos_term^M
    # activation = sigmoid(100 * disc) * clamp(disc, min=1e-6)^(1/N)

    A, B, C, K, M, N = 0.37091212, 0.31903101, 1.06435066, 1.92138556, 2.50322787, 2.62706077
    bias_diff = max(2.0 - C, 1e-6)
    cos_term = torch.abs(torch.cos(torch.pi * phi)) + 1e-6
    disc = A * (bias_diff**K) - B * (cos_term**M)
    activation = torch.sigmoid(100.0 * disc)
    expected = activation * (torch.clamp(disc, min=1e-6) ** (1 / N))

    assert history.shape == (1, 3, 1)
    assert torch.allclose(history[:, 1:, :], expected, atol=1e-5)
    assert torch.allclose(history[:, 0, :], torch.zeros(1, 1), atol=1e-6)
