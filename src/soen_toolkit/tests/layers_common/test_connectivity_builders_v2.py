import pytest
import torch

from soen_toolkit.core.layers.common import (
    build_chain,
    build_connectivity,
    build_constant,
    build_dense,
    build_exponential,
    build_one_to_one,
    build_power_law,
    build_sparse,
)


def test_dense_returns_ones() -> None:
    mat = build_dense(3, 2)
    assert mat.shape == (2, 3)
    assert torch.all(mat == 1)


def test_one_to_one_handles_rectangular() -> None:
    mat = build_one_to_one(4, 2)
    assert mat.shape == (2, 4)
    assert torch.all(mat[0, 0] == 1)
    assert torch.all(mat[1, 1] == 1)
    assert torch.sum(mat) == 2


def test_chain_creates_forward_links() -> None:
    mat = build_chain(5, 5)
    expected = torch.zeros(5, 5)
    indices = torch.arange(4)
    expected[indices + 1, indices] = 1.0
    assert torch.equal(mat, expected)

    rectangular = build_chain(5, 3)
    expected_rect = torch.zeros(3, 5)
    expected_rect[1, 0] = 1.0
    expected_rect[2, 1] = 1.0
    assert torch.equal(rectangular, expected_rect)


def test_sparse_requires_param() -> None:
    try:
        build_sparse(2, 2, params=None)
    except ValueError:
        pass
    else:
        msg = "Expected ValueError when sparsity missing"
        raise AssertionError(msg)

    torch.manual_seed(0)
    mat = build_sparse(3, 2, params={"sparsity": 0.5})
    assert mat.shape == (2, 3)
    assert torch.all((mat == 0) | (mat == 1))


def test_constant_respects_expected_fan_out() -> None:
    torch.manual_seed(1)
    mat = build_constant(4, 3, params={"expected_fan_out": 2, "allow_self_connections": False})
    assert mat.shape == (3, 4)
    assert torch.all(mat.sum(dim=0) <= 2)


def test_power_law_and_exponential_shapes() -> None:
    torch.manual_seed(2)
    power = build_power_law(5, 5, params={"expected_fan_out": 2})
    assert power.shape == (5, 5)
    assert torch.all(power.sum(dim=0) <= 2)

    expo = build_exponential(5, 4, params={"expected_fan_out": 2, "d_0": 1.5})
    assert expo.shape == (4, 5)
    assert torch.all(expo.sum(dim=0) <= 2)


def test_build_connectivity_dispatch() -> None:
    mat = build_connectivity("dense", from_nodes=2, to_nodes=1)
    assert mat.shape == (1, 2)
    try:
        build_connectivity("unknown", from_nodes=1, to_nodes=1)
    except ValueError:
        pass
    else:
        msg = "Expected ValueError for unknown builder"
        raise AssertionError(msg)


def test_build_connectivity_all_to_all_alias_warns() -> None:
    with pytest.warns(DeprecationWarning):
        mat = build_connectivity("all_to_all", from_nodes=2, to_nodes=1)
    assert mat.shape == (1, 2)
