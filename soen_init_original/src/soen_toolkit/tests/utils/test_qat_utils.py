import math

import pytest
import torch

from soen_toolkit.utils.quantization import (
    calculate_num_levels,
    generate_uniform_codebook,
    snapped_copy,
    ste_snap,
)


def test_calculate_num_levels_bits_vs_levels() -> None:
    assert calculate_num_levels(bits=0) == 2  # includes a single zero level (2**0 + 1)
    assert calculate_num_levels(bits=1) == 3  # {-a, 0, +a}
    assert calculate_num_levels(bits=3) == 9
    assert calculate_num_levels(levels=9) == 9
    with pytest.raises(ValueError):
        calculate_num_levels(bits=3, levels=9)
    with pytest.raises(ValueError):
        calculate_num_levels()


def test_generate_uniform_codebook_includes_single_zero_and_size() -> None:
    cb = generate_uniform_codebook(-0.24, 0.24, 9)
    assert cb.numel() == 9
    zeros = (cb == 0.0).nonzero().flatten()
    assert zeros.numel() == 1


def test_generate_uniform_codebook_zero_outside_range_still_included() -> None:
    # Range fully positive
    cb_pos = generate_uniform_codebook(0.1, 0.2, 5)
    assert (cb_pos == 0.0).any()
    assert cb_pos.numel() == 5
    # Range fully negative
    cb_neg = generate_uniform_codebook(-0.2, -0.1, 5)
    assert (cb_neg == 0.0).any()
    assert cb_neg.numel() == 5


def test_snapped_copy_selects_nearest_neighbor() -> None:
    torch.manual_seed(0)
    w = torch.tensor([[-0.23, -0.01, 0.02, 0.21]], dtype=torch.float32)
    cb = generate_uniform_codebook(-0.24, 0.24, 9)
    snapped = snapped_copy(w, cb)
    # Verify each element is one of the codebook entries and is nearest
    for i in range(w.numel()):
        v = w.view(-1)[i]
        diffs = (cb - v).abs()
        j = int(diffs.argmin().item())
        assert math.isclose(float(snapped.view(-1)[i].item()), float(cb[j].item()), rel_tol=0, abs_tol=1e-7)


def test_ste_snap_forward_equals_snapped_and_grad_identity() -> None:
    torch.manual_seed(0)
    w = torch.tensor([[-0.23, -0.01, 0.02, 0.21]], dtype=torch.float32, requires_grad=True)
    cb = generate_uniform_codebook(-0.24, 0.24, 9)
    y = ste_snap(w, cb)
    # Forward equivalence with snapped_copy
    assert torch.allclose(y.detach(), snapped_copy(w.detach(), cb), atol=1e-8)
    # Gradient should pass through as identity
    loss = y.sum()
    loss.backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_ste_snap_stochastic_matches_rng_thresholding() -> None:
    # Ensure stochastic path selects between adjacent codebook values with probability p
    # equal to the fractional position between neighbours, using a fixed RNG.
    N = 1000
    # Codebook with two bins [0.0, 1.0] simplifies neighbour logic
    cb = torch.tensor([0.0, 1.0], dtype=torch.float32)
    # Choose values at position p=0.3 between 0 and 1
    p = 0.3
    x = torch.full((N,), float(p), dtype=torch.float32)

    # Two identical-seed generators: one for expected, one passed into ste_snap
    g_expected = torch.Generator(device="cpu")
    g_expected.manual_seed(123456)
    g_call = torch.Generator(device="cpu")
    g_call.manual_seed(123456)

    # Draw the same randoms the function will consume and compute expected choices
    rnd = torch.rand(N, generator=g_expected)
    choose_hi = rnd < p
    expected = torch.where(choose_hi, torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32))

    # Call stochastic STE; forward value equals snapped choice
    y = ste_snap(x, cb, stochastic=True, rng=g_call)
    assert torch.allclose(y, expected)

    # Also verify gradient path remains identity in stochastic mode
    x_req = x.clone().detach().requires_grad_(True)
    y_req = ste_snap(x_req, cb, stochastic=True, rng=g_call)
    y_req.sum().backward()
    assert torch.allclose(x_req.grad, torch.ones_like(x_req))
