import torch

from soen_toolkit.core.layers.common import build_connectivity


def test_dense_shape_and_ones() -> None:
    mask = build_connectivity("dense", from_nodes=4, to_nodes=3, params={})
    assert mask.shape == (3, 4)
    assert torch.all((mask == 0) | (mask == 1))
    assert mask.sum().item() == 12


def test_one_to_one_diagonal_up_to_min() -> None:
    mask = build_connectivity("one_to_one", from_nodes=5, to_nodes=3, params={})
    assert mask.shape == (3, 5)
    # Diagonal ones up to min(5,3)=3
    expected = torch.zeros(3, 5)
    expected[0, 0] = 1
    expected[1, 1] = 1
    expected[2, 2] = 1
    assert torch.equal(mask, expected)


def test_one_to_one_with_source_range() -> None:
    # Test basic range connectivity: connect nodes 4-9 from a 10-node source to a 6-node target
    mask = build_connectivity(
        "one_to_one",
        from_nodes=10,
        to_nodes=6,
        params={"source_start_node_id": 4, "source_end_node_id": 9}
    )
    assert mask.shape == (6, 10)

    # Should have diagonal connections from source[4:10] to target[0:6]
    expected = torch.zeros(6, 10)
    for i in range(6):
        expected[i, 4 + i] = 1.0
    assert torch.equal(mask, expected)

    # Test another range: nodes 0-3 from a 10-node source to a 4-node target
    mask2 = build_connectivity(
        "one_to_one",
        from_nodes=10,
        to_nodes=4,
        params={"source_start_node_id": 0, "source_end_node_id": 3}
    )
    assert mask2.shape == (4, 10)

    expected2 = torch.zeros(4, 10)
    for i in range(4):
        expected2[i, i] = 1.0
    assert torch.equal(mask2, expected2)


def test_one_to_one_range_validation() -> None:
    # Test that both parameters must be specified together
    try:
        build_connectivity(
            "one_to_one",
            from_nodes=10,
            to_nodes=4,
            params={"source_start_node_id": 0}
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Both source_start_node_id and source_end_node_id must be specified together" in str(e)

    # Test invalid range: start > end
    try:
        build_connectivity(
            "one_to_one",
            from_nodes=10,
            to_nodes=4,
            params={"source_start_node_id": 5, "source_end_node_id": 3}
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must be <=" in str(e)

    # Test invalid range: negative start
    try:
        build_connectivity(
            "one_to_one",
            from_nodes=10,
            to_nodes=4,
            params={"source_start_node_id": -1, "source_end_node_id": 3}
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must be >= 0" in str(e)

    # Test invalid range: end >= from_nodes
    try:
        build_connectivity(
            "one_to_one",
            from_nodes=10,
            to_nodes=4,
            params={"source_start_node_id": 0, "source_end_node_id": 10}
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must be < from_nodes" in str(e)


def test_one_to_one_range_fits_target() -> None:
    # Test that range size must not exceed target nodes
    try:
        build_connectivity(
            "one_to_one",
            from_nodes=10,
            to_nodes=3,
            params={"source_start_node_id": 0, "source_end_node_id": 5}  # 6 nodes, but target only has 3
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Source range size" in str(e) and "exceeds target nodes" in str(e)

    # Test that exact match works
    mask = build_connectivity(
        "one_to_one",
        from_nodes=10,
        to_nodes=3,
        params={"source_start_node_id": 5, "source_end_node_id": 7}  # Exactly 3 nodes
    )
    assert mask.shape == (3, 10)
    expected = torch.zeros(3, 10)
    expected[0, 5] = 1.0
    expected[1, 6] = 1.0
    expected[2, 7] = 1.0
    assert torch.equal(mask, expected)


def test_sparse_density_param() -> None:
    # Use a non-trivial size and check approximate density
    sparsity = 0.2
    mask = build_connectivity("sparse", from_nodes=50, to_nodes=40, params={"sparsity": sparsity, "seed": 123})
    density = mask.mean().item()
    assert 0.1 < density < 0.3


def test_block_structure_shape() -> None:
    params = {
        "block_count": 2,
        "connection_mode": "diagonal",  # or "full"
        "within_block_density": 1.0,
        "cross_block_density": 0.0,
    }
    mask = build_connectivity("block_structure", from_nodes=4, to_nodes=4, params=params)
    assert mask.shape == (4, 4)
    assert torch.all((mask == 0) | (mask == 1))


def test_power_law_shape_and_binary() -> None:
    params = {"alpha": 2.0, "expected_fan_out": 3, "allow_self_connections": False}
    mask = build_connectivity("power_law", from_nodes=10, to_nodes=8, params=params)
    assert mask.shape == (8, 10)
    assert torch.all((mask == 0) | (mask == 1))


def test_power_law_expected_fan_out_per_source() -> None:
    N = 12
    K = 4
    params = {"alpha": 2.0, "expected_fan_out": K, "allow_self_connections": False}
    mask = build_connectivity("power_law", from_nodes=N, to_nodes=N, params=params)
    # Each column (source) should have exactly K ones, except possibly reduced by self-connection rule.
    col_sums = mask.sum(dim=0)
    # Since allow_self_connections=False, with square mask, each source still has K targets (excluding itself)
    # Clamp check within {K-1, K} to be tolerant if K>N-1
    assert torch.all((col_sums == K) | (col_sums == max(0, min(K, N - 1))))


def test_power_law_locality_bias_small_grid() -> None:
    # 3x3 grid (N=9). Nodes in center have closer neighbors than corners.
    N = 9
    K = 3
    params = {"alpha": 3.0, "expected_fan_out": K, "allow_self_connections": False}
    mask = build_connectivity("power_law", from_nodes=N, to_nodes=N, params=params)

    # For the center node (index 4 in our deterministic 3x3 layout), expect more nearby connections than far ones.
    j = 4
    # Deterministic layout: indices row-major; neighbors at distance 1 should be preferred
    # Count connections to immediate 4-neighborhood vs. corners
    immediate = {1, 3, 5, 7}
    corners = {0, 2, 6, 8}
    col = mask[:, j].nonzero(as_tuple=False).view(-1).tolist()
    near_hits = sum(1 for t in col if t in immediate)
    far_hits = sum(1 for t in col if t in corners)
    assert near_hits >= far_hits
