"""Tests for Graph API."""

import os
import tempfile

import pytest
import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.nn import Graph, dynamic, dynamic_v2, init, layers, structure


def test_graph_basic_construction() -> None:
    """Test basic graph construction and compilation."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    # Should compile on first forward
    x = torch.randn(2, 10, 10)
    output = g(x)

    assert output.shape == (2, 11, 5)  # batch, time+1, dim
    assert g._compiled_core is not None


def test_graph_parity_with_spec() -> None:
    """Test that Graph produces same model as spec-based API."""
    # Build with Graph API
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.constant(0.5))

    # Build with spec API
    sim_config = SimulationConfig(dt=37, network_evaluation_method="layerwise")
    layer_configs = [
        LayerConfig(layer_id=0, layer_type="Linear", params={"dim": 10}),
        LayerConfig(
            layer_id=1,
            layer_type="SingleDendrite",
            params={
                "dim": 5,
                "solver": "FE",
                "source_func_type": "RateArray",
                "bias_current": 1.7,
                "gamma_plus": 1e-3,
                "gamma_minus": 1e-3,
            },
        ),
    ]
    connection_configs = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={
                "structure": {"type": "dense", "params": {}},
                "init": {"name": "constant", "params": {"value": 0.5}},
            },
        ),
    ]
    spec_model = SOENModelCore(sim_config, layer_configs, connection_configs)

    # Compile graph
    graph_model = g.compile()

    # Compare shapes
    assert len(graph_model.layers) == len(spec_model.layers)
    assert len(graph_model.connections) == len(spec_model.connections)

    # Compare connection weights
    graph_J = graph_model.connections["J_0_to_1"]
    spec_J = spec_model.connections["J_0_to_1"]
    assert graph_J.shape == spec_J.shape
    assert torch.allclose(graph_J, spec_J)


def test_graph_dynamic_connections_new_api() -> None:
    """Test dynamic connection mode with new simplified API."""
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(
        0,
        1,
        structure=structure.sparse(sparsity=0.5),
        init=init.uniform(-0.2, 0.2),
        mode="WICC",
        connection_params={"source_func": "RateArray", "gamma_plus": 1e-3, "bias_current": 2.0},
    )

    core = g.compile()

    # Check that dynamic metadata was stored
    assert hasattr(core, "_connection_modes")
    assert core._connection_modes.get("J_0_to_1") == "WICC"
    assert hasattr(core, "_connection_params")
    assert "J_0_to_1" in core._connection_params


def test_graph_nocc_connections_new_api() -> None:
    """Test NOCC connection mode with new simplified API."""
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )

    g.connect(
        0,
        1,
        structure=structure.dense(),
        init=init.uniform(-0.2, 0.2),
        mode="NOCC",
        connection_params={"alpha": 1.5, "beta": 303.85, "beta_out": 91.156, "bias_current": 2.1},
    )

    core = g.compile()

    assert hasattr(core, "_connection_modes")
    assert core._connection_modes.get("J_0_to_1") == "NOCC"
    assert hasattr(core, "_connection_params")
    assert "J_0_to_1" in core._connection_params
    assert core._connection_params["J_0_to_1"]["alpha"] == 1.5


def test_graph_dynamic_connections_legacy_api() -> None:
    """Test backward compatibility with deprecated dynamic parameter."""
    import warnings

    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )

    # Should emit deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g.connect(
            0,
            1,
            structure=structure.sparse(sparsity=0.5),
            init=init.uniform(-0.2, 0.2),
            dynamic=dynamic(source_func="RateArray", gamma_plus=1e-3, bias_current=2.0),
        )
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)

    core = g.compile()

    # Should still work correctly
    assert hasattr(core, "_connection_modes")
    assert core._connection_modes.get("J_0_to_1") == "WICC"


def test_graph_mode_inference_from_dynamic_legacy() -> None:
    """Test legacy dynamic spec backward compatibility."""
    import warnings

    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )

    # Should emit deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g.connect(
            0,
            1,
            structure=structure.dense(),
            init=init.uniform(-0.2, 0.2),
            dynamic=dynamic_v2(),
        )
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)

    core = g.compile()

    assert hasattr(core, "_connection_modes")
    assert core._connection_modes.get("J_0_to_1") == "NOCC"


def test_graph_histories() -> None:
    """Test that histories are populated when tracking enabled."""
    g = Graph(dt=37, network_evaluation_method="layerwise", track_phi=True, track_s=True)
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    x = torch.randn(2, 10, 10)
    g(x)

    # Check histories were populated
    assert g.phi_history is not None
    assert g.s_history is not None
    assert len(g.phi_history) == 2  # Two layers
    assert len(g.s_history) == 2


def test_graph_save_load() -> None:
    """Test save and load round-trip."""
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.constant(0.5))

    # Compile and run forward
    x = torch.randn(2, 10, 10)
    output1 = g(x)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.soen")
        g.save(path)

        # Load
        g2 = Graph.load(path)
        output2 = g2(x)

        # Should produce same output
        assert torch.allclose(output1, output2)


def test_graph_parameters() -> None:
    """Test that parameters() works correctly."""
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    g.compile()

    # Should have parameters
    params = list(g.parameters())
    assert len(params) > 0

    # Should be able to create optimizer
    optimizer = torch.optim.Adam(g.parameters(), lr=1e-3)
    assert optimizer is not None


def test_graph_to_device() -> None:
    """Test moving graph to device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    g = g.to("cuda")

    x = torch.randn(2, 10, 10, device="cuda")
    output = g(x)

    assert output.device.type == "cuda"


def test_graph_constraints() -> None:
    """Test that constraints are applied."""
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(
        0,
        1,
        structure=structure.dense(),
        init=init.uniform(-1.0, 1.0),
        constraints={"min": -0.5, "max": 0.5},
    )

    core = g.compile()

    # Check constraints were stored
    assert "J_0_to_1" in core.connection_constraints
    assert core.connection_constraints["J_0_to_1"]["min"] == -0.5
    assert core.connection_constraints["J_0_to_1"]["max"] == 0.5


def test_graph_learnable_flag() -> None:
    """Test that learnable flag controls requires_grad."""
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(
        0,
        1,
        structure=structure.dense(),
        init=init.xavier_uniform(),
        learnable=False,
    )

    core = g.compile()

    # Check that connection is not learnable
    assert not core.connections["J_0_to_1"].requires_grad


def test_graph_multiple_connections() -> None:
    """Test graph with multiple connections."""
    g = Graph(dt=37, network_evaluation_method="stepwise_gauss_seidel")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.add_layer(
        2,
        layers.SingleDendrite(
            dim=3,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )

    # Forward connections
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
    g.connect(1, 2, structure=structure.dense(), init=init.xavier_uniform())

    # Feedback connection
    g.connect(2, 1, structure=structure.sparse(0.3), init=init.uniform(-0.1, 0.1))

    x = torch.randn(2, 10, 10)
    output = g(x)

    assert output.shape == (2, 11, 3)
