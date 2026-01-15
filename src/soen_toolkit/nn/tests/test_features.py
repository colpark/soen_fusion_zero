"""Tests verifying all SOENModelCore features are accessible via Graph/Sequential."""

import os
import tempfile

import torch

from soen_toolkit.nn import Graph, Sequential, init, layers, structure


def test_has_visualize() -> None:
    """Test that visualize method exists and is callable."""
    g = Sequential([layers.Linear(dim=10), layers.NonLinear(dim=5, source_func_type="Tanh")])

    assert hasattr(g, "visualize"), "Graph should have visualize method"
    assert callable(g.visualize), "visualize should be callable"


def test_has_summary() -> None:
    """Test that summary method exists and is callable."""
    g = Sequential([layers.Linear(dim=10), layers.NonLinear(dim=5, source_func_type="Tanh")])

    assert hasattr(g, "summary"), "Graph should have summary method"
    assert callable(g.summary), "summary should be callable"


def test_has_compute_summary() -> None:
    """Test that compute_summary method exists and returns dict."""
    g = Sequential([layers.Linear(dim=10), layers.NonLinear(dim=5, source_func_type="Tanh")])

    assert hasattr(g, "compute_summary"), "Graph should have compute_summary method"

    g.compile()
    stats = g.compute_summary()
    assert isinstance(stats, dict), "compute_summary should return dict"


def test_has_save() -> None:
    """Test that save method exists and works."""
    g = Sequential([layers.Linear(dim=10), layers.NonLinear(dim=5, source_func_type="Tanh")])

    assert hasattr(g, "save"), "Graph should have save method"
    assert callable(g.save), "save should be callable"

    # Test save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test.soen")
        g.save(save_path)
        assert os.path.exists(save_path), "File should be created"


def test_has_load() -> None:
    """Test that load class method exists and works."""
    g = Sequential([layers.Linear(dim=10), layers.NonLinear(dim=5, source_func_type="Tanh")])

    assert hasattr(Graph, "load"), "Graph should have load class method"
    assert callable(Graph.load), "load should be callable"

    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test.soen")
        g.save(save_path)
        g_loaded = Graph.load(save_path)
        assert isinstance(g_loaded, Graph), "load should return Graph instance"


def test_save_load_roundtrip() -> None:
    """Test that save/load preserves model state."""
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=20,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=0.001,
            gamma_minus=0.001,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    # Initialize
    x = torch.randn(2, 10, 10)
    out1 = g(x)

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test.soen")
        g.save(save_path)
        g_loaded = Graph.load(save_path)

    # Verify outputs match
    out2 = g_loaded(x)
    assert torch.allclose(out1, out2, atol=1e-6), "Outputs should match after load"


def test_visualize_runs() -> None:
    """Test that visualize actually runs without error."""
    g = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=20,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=0.001,
                gamma_minus=0.001,
            ),
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_network")
        result = g.visualize(save_path=save_path, file_format="png")

        # Check that a file was created
        assert isinstance(result, str), "visualize should return file path"
        assert os.path.exists(result), "Visualization file should exist"


def test_summary_runs() -> None:
    """Test that summary runs without error."""
    g = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=20,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=0.001,
                gamma_minus=0.001,
            ),
        ]
    )

    # Should not raise
    g.summary(print_summary=False)

    # Should return DataFrame
    df = g.summary(return_df=True, print_summary=False)
    assert df is not None, "summary should return DataFrame when requested"


def test_state_dict_compatible() -> None:
    """Test that state_dict is compatible with PyTorch."""
    g = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=20,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=0.001,
                gamma_minus=0.001,
            ),
        ]
    )

    g.compile()

    # Get state dict
    state = g.state_dict()
    assert isinstance(state, dict), "state_dict should return dict"

    # Save and load with torch.save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "state.pth")
        torch.save(state, save_path)
        loaded_state = torch.load(save_path)

        # Should be able to load back
        g.load_state_dict(loaded_state)


def test_device_management() -> None:
    """Test that to() works for device management."""
    g = Sequential(
        [
            layers.Linear(dim=10),
            layers.NonLinear(dim=5, source_func_type="Tanh"),
        ]
    )

    # Should not raise
    g_cpu = g.to("cpu")
    assert g_cpu is not None

    # Test forward on CPU
    x = torch.randn(2, 10, 10)
    out = g_cpu(x)
    assert out.device.type == "cpu"


def test_parameters_accessible() -> None:
    """Test that parameters() returns parameters."""
    g = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=20,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=0.001,
                gamma_minus=0.001,
            ),
        ]
    )

    g.compile()

    # Should have parameters
    params = list(g.parameters())
    assert len(params) > 0, "Should have parameters"

    # All should be tensors
    for p in params:
        assert isinstance(p, (torch.nn.Parameter, torch.Tensor))


def test_access_compiled_core() -> None:
    """Test that underlying core is accessible."""
    g = Sequential([layers.Linear(dim=10), layers.NonLinear(dim=5, source_func_type="Tanh")])

    g.compile()

    assert hasattr(g, "_compiled_core"), "Should have _compiled_core attribute"
    assert g._compiled_core is not None, "Core should be compiled"

    # Core should have expected attributes
    core = g._compiled_core
    assert hasattr(core, "layers"), "Core should have layers"
    assert hasattr(core, "connections"), "Core should have connections"
    assert hasattr(core, "visualize"), "Core should have visualize"
    assert hasattr(core, "summary"), "Core should have summary"
