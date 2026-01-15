"""Tests for Sequential API."""

import torch

from soen_toolkit.nn import Sequential, layers


def test_sequential_basic() -> None:
    """Test basic Sequential construction."""
    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=5,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=1e-3,
                gamma_minus=1e-3,
            ),
            layers.DendriteReadout(dim=3, source_func_type="RateArray", bias_current=1.7),
        ]
    )

    x = torch.randn(2, 10, 10)
    output = net(x)

    assert output.shape == (2, 11, 3)


def test_sequential_auto_connect() -> None:
    """Test that Sequential auto-connects layers."""
    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=5,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=1e-3,
                gamma_minus=1e-3,
            ),
        ]
    )

    # Should have auto-created connection 0->1
    assert len(net._connection_specs) == 1
    assert net._connection_specs[0].from_layer == 0
    assert net._connection_specs[0].to_layer == 1


def test_sequential_append() -> None:
    """Test appending layers to Sequential."""
    net = Sequential(
        [
            layers.Linear(dim=10),
        ]
    )

    net.append(
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        )
    )

    # Should have 2 layers and 1 connection
    assert len(net._layer_specs) == 2
    assert len(net._connection_specs) == 1

    x = torch.randn(2, 10, 10)
    output = net(x)
    assert output.shape == (2, 11, 5)


def test_sequential_no_auto_connect() -> None:
    """Test Sequential with auto_connect=False."""
    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=5,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=1e-3,
                gamma_minus=1e-3,
            ),
        ],
        auto_connect=False,
    )

    # Should have no connections
    assert len(net._connection_specs) == 0

    # Manually add connection
    from soen_toolkit.nn import init, structure

    net.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    x = torch.randn(2, 10, 10)
    output = net(x)
    assert output.shape == (2, 11, 5)


def test_sequential_custom_init() -> None:
    """Test Sequential with custom initialization."""
    from soen_toolkit.nn import init

    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=5,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=1e-3,
                gamma_minus=1e-3,
            ),
        ],
        connection_init=init.constant(0.5),
    )

    core = net.compile()

    # Check that connections use constant init
    J = core.connections["J_0_to_1"]
    # All non-zero entries should be 0.5 (within mask)
    assert torch.allclose(J[J != 0], torch.tensor(0.5))


def test_sequential_parameters() -> None:
    """Test that Sequential exposes parameters correctly."""
    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=5,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=1e-3,
                gamma_minus=1e-3,
            ),
        ]
    )

    net.compile()

    params = list(net.parameters())
    assert len(params) > 0

    # Should work with optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    assert optimizer is not None
