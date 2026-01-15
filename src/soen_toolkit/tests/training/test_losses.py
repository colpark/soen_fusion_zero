import pytest
import torch

from soen_toolkit.tests.utils.test_helpers_fixture import build_small_model
from soen_toolkit.training.losses.loss_functions import (
    autoregressive_cross_entropy,
    gravity_quantization_loss,
)


def test_autoregressive_cross_entropy_skips_t0_when_outputs_have_seq_len_plus_one() -> None:
    B, T, V = 2, 4, 5
    # outputs with seq_len+1
    outputs = torch.randn(B, T + 1, V, requires_grad=True)
    # targets with seq_len=T
    targets = torch.randint(low=0, high=V, size=(B, T))

    loss = autoregressive_cross_entropy(outputs, targets)
    # Backprop works
    loss.backward()
    assert loss.requires_grad


def test_autoregressive_cross_entropy_mismatch_returns_zero_with_grad() -> None:
    B, T, V = 2, 4, 6
    outputs = torch.randn(B, T + 2, V, requires_grad=True)
    targets = torch.randint(low=0, high=V, size=(B, T))
    loss = autoregressive_cross_entropy(outputs, targets)
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_gravity_quantization_loss_modes_and_filtering() -> None:
    # Build a small model with two connections; make one non-learnable
    d = 3
    m = build_small_model(dims=(d, d), connectivity_type="dense", init="constant", init_value=0.12)
    # Add a second connection by creating a new model with internal
    # Easiest: manually adjust requires_grad on the inter-layer
    m.connections["J_0_to_1"].requires_grad_(True)

    # MAE mode
    loss_mae = gravity_quantization_loss(m, min_val=-0.24, max_val=0.24, bits=2, mode="mae")
    assert loss_mae.requires_grad
    assert torch.isfinite(loss_mae)

    # MSE mode
    loss_mse = gravity_quantization_loss(m, min_val=-0.24, max_val=0.24, bits=2, mode="mse")
    assert loss_mse.requires_grad
    assert torch.isfinite(loss_mse)

    # Target only specific connection
    loss_filtered = gravity_quantization_loss(
        m,
        min_val=-0.24,
        max_val=0.24,
        bits=2,
        mode="mae",
        connections=["J_0_to_1"],
    )
    assert loss_filtered.requires_grad
    assert torch.isfinite(loss_filtered)


def test_gravity_quantization_loss_errors_when_no_learnable() -> None:
    d = 3
    m = build_small_model(dims=(d, d), connectivity_type="dense", init="constant", init_value=0.12)
    # Make all connections non-learnable
    for p in m.connections.values():
        p.requires_grad_(False)

    with pytest.raises(ValueError):
        _ = gravity_quantization_loss(m, min_val=-0.24, max_val=0.24, bits=2)

    # If targeting a non-learnable only, also raises
    for p in m.connections.values():
        p.requires_grad_(False)
    with pytest.raises(ValueError):
        _ = gravity_quantization_loss(
            m,
            min_val=-0.24,
            max_val=0.24,
            bits=2,
            connections=["J_0_to_1"],
        )
