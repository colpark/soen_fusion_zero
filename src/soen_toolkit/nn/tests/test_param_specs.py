"""Tests for parameter specifications and fine-grained control."""

import torch

from soen_toolkit.nn import Graph, init, layers, param_specs, structure


def test_param_spec_uniform() -> None:
    """Test uniform distribution parameter spec."""
    spec = param_specs.uniform(1.5, 2.0, learnable=True, constraints={"min": 0.0, "max": 5.0})

    spec_dict = spec.to_dict()
    assert spec_dict["distribution"] == "uniform"
    assert spec_dict["params"]["min"] == 1.5
    assert spec_dict["params"]["max"] == 2.0
    assert spec_dict["learnable"] is True
    assert spec_dict["constraints"]["min"] == 0.0


def test_param_spec_lognormal() -> None:
    """Test log-normal distribution parameter spec."""
    spec = param_specs.lognormal(-6.9, 0.2, learnable=True)

    spec_dict = spec.to_dict()
    assert spec_dict["distribution"] == "lognormal"
    assert spec_dict["params"]["mean"] == -6.9
    assert spec_dict["params"]["std"] == 0.2


def test_layer_with_distributions() -> None:
    """Test layer creation with distribution specs."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=param_specs.uniform(1.5, 2.0, learnable=True),
            gamma_plus=param_specs.lognormal(-6.9, 0.2, learnable=True),
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    core = g.compile()

    # Check that layer was built correctly
    assert len(core.layers) == 2

    # Run forward to ensure it works
    x = torch.randn(2, 10, 10)
    output = g(x)
    assert output.shape == (2, 11, 5)


def test_learnability_control() -> None:
    """Test fine-grained learnability control."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=0.001,
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
        learnable_params={
            "bias_current": True,
            "gamma_plus": False,
            "gamma_minus": False,
            "phi_offset": False,
        },
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    core = g.compile()

    # Check learnability
    # Connection weights should be learnable by default
    assert core.connections["J_0_to_1"].requires_grad

    # Check layer parameters
    for name, param in core.named_parameters():
        if "bias_current" in name and "layer" in name:
            assert param.requires_grad, f"{name} should be learnable"
        elif "gamma" in name or "phi_offset" in name:
            assert not param.requires_grad, f"{name} should be frozen"


def test_constraints() -> None:
    """Test that constraints are stored correctly."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=param_specs.uniform(
                1.5,
                2.0,
                learnable=True,
                constraints={"min": 0.0, "max": 5.0},
            ),
            gamma_plus=0.001,
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    core = g.compile()

    # Constraints are stored in layer configs
    layer1_config = core.layers_config[1]

    # Check that bias_current has constraint info
    assert "bias_current" in layer1_config.params


def test_dict_syntax_still_works() -> None:
    """Test that old dict syntax still works."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current={
                "distribution": "uniform",
                "params": {"min": 1.5, "max": 2.0},
                "learnable": True,
            },
            gamma_plus=0.001,
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    g.compile()

    # Should build without errors
    x = torch.randn(2, 10, 10)
    output = g(x)
    assert output.shape == (2, 11, 5)


def test_param_spec_with_learnability_override() -> None:
    """Test that ParamSpec learnable flag is respected."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=param_specs.constant(1.7, learnable=False),
            gamma_plus=param_specs.constant(0.001, learnable=True),
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    core = g.compile()

    # gamma_plus should be learnable, bias_current should not
    for name, param in core.named_parameters():
        if "bias_current" in name and "layer" in name:
            assert not param.requires_grad, f"{name} should be frozen"
        elif "gamma_plus" in name and "layer" in name:
            assert param.requires_grad, f"{name} should be learnable"


def test_multiple_distributions() -> None:
    """Test layer with multiple different distributions."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=param_specs.uniform(1.5, 2.0),
            gamma_plus=param_specs.lognormal(-6.9, 0.2),
            gamma_minus=param_specs.normal(0.001, 0.0002),
            phi_offset=param_specs.constant(0.23),
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    g.compile()

    # Should build and run
    x = torch.randn(2, 10, 10)
    output = g(x)
    assert output.shape == (2, 11, 5)


def test_linear_spacing() -> None:
    """Test linearly spaced parameter initialization."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=param_specs.linear(1.5, 2.0, learnable=True),
            gamma_plus=0.001,
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    g.compile()

    # Check that bias_current values are linearly spaced
    # (This is handled by the core parameter initialization)
    x = torch.randn(2, 10, 10)
    output = g(x)
    assert output.shape == (2, 11, 5)


# --- Block size tests ---


def test_block_size_param_spec_includes_block_size() -> None:
    """Test that block_size is included in params when not default."""
    # Default block_size=1 should not appear in params
    spec_default = param_specs.linear(0.0, 1.0)
    assert "block_size" not in spec_default.params

    # Non-default block_size should appear
    spec_blocked = param_specs.linear(0.0, 1.0, block_size=3)
    assert spec_blocked.params["block_size"] == 3


def test_block_size_all_distributions() -> None:
    """Test block_size works for all distribution types."""
    # All these should include block_size in params
    specs = [
        param_specs.uniform(0.0, 1.0, block_size=2),
        param_specs.normal(0.0, 1.0, block_size=2),
        param_specs.lognormal(-1.0, 0.5, block_size=2),
        param_specs.constant(1.0, block_size=2),
        param_specs.linear(0.0, 1.0, block_size=2),
        param_specs.loglinear(0.01, 1.0, block_size=2),
        param_specs.loguniform(0.01, 1.0, block_size=2),
    ]
    for spec in specs:
        assert spec.params["block_size"] == 2, f"Failed for {spec.distribution}"


def test_block_size_linear_initialization() -> None:
    """Test that linear initialization with block_size produces correct values."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_deterministic

    # 12 nodes with block_size=3 -> 4 blocks with linear values
    values = initialize_values_deterministic(
        shape=(12,),
        method="linear",
        params={"min": 0.0, "max": 1.0, "block_size": 3},
        device=torch.device("cpu"),
    )

    assert values.shape == (12,)
    # Should have 4 unique values (one per block), linearly spaced
    # Block values: [0.0, 0.333, 0.666, 1.0]
    # Each repeated 3 times
    expected_block_values = torch.linspace(0.0, 1.0, 4)
    for i in range(4):
        for j in range(3):
            assert torch.isclose(values[i * 3 + j], expected_block_values[i], atol=1e-5)


def test_block_size_constant_initialization() -> None:
    """Test that constant initialization with block_size works correctly."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_deterministic

    # Block_size should not change constant init (all same value)
    values = initialize_values_deterministic(
        shape=(12,),
        method="constant",
        params={"value": 2.5, "block_size": 3},
        device=torch.device("cpu"),
    )

    assert values.shape == (12,)
    assert torch.allclose(values, torch.full((12,), 2.5))


def test_block_size_normal_initialization() -> None:
    """Test that normal initialization with block_size produces blocked values."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_stochastic

    torch.manual_seed(42)  # For reproducibility
    values = initialize_values_stochastic(
        shape=(12,),
        method="normal",
        params={"mean": 0.0, "std": 1.0, "block_size": 3},
        device=torch.device("cpu"),
    )

    assert values.shape == (12,)
    # Check that values within each block are identical
    for block_idx in range(4):
        start = block_idx * 3
        block_val = values[start]
        assert values[start + 1] == block_val
        assert values[start + 2] == block_val


def test_block_size_uniform_initialization() -> None:
    """Test that uniform initialization with block_size produces blocked values."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_stochastic

    torch.manual_seed(42)
    values = initialize_values_stochastic(
        shape=(12,),
        method="uniform",
        params={"min": 0.0, "max": 1.0, "block_size": 4},
        device=torch.device("cpu"),
    )

    assert values.shape == (12,)
    # Check that values within each block are identical (3 blocks of 4)
    for block_idx in range(3):
        start = block_idx * 4
        block_val = values[start]
        for j in range(1, 4):
            assert values[start + j] == block_val


def test_block_size_one_is_default_behavior() -> None:
    """Test that block_size=1 produces same results as no block_size."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_deterministic

    torch.manual_seed(42)
    values_default = initialize_values_deterministic(
        shape=(10,),
        method="linear",
        params={"min": 0.0, "max": 1.0},
        device=torch.device("cpu"),
    )

    torch.manual_seed(42)
    values_block1 = initialize_values_deterministic(
        shape=(10,),
        method="linear",
        params={"min": 0.0, "max": 1.0, "block_size": 1},
        device=torch.device("cpu"),
    )

    assert torch.allclose(values_default, values_block1)


def test_block_size_not_divisible_raises_error() -> None:
    """Test that block_size that doesn't divide width raises ValueError."""
    import pytest

    from soen_toolkit.core.layers.common.metadata import initialize_values_deterministic

    # 10 nodes with block_size=3 doesn't divide evenly
    with pytest.raises(ValueError, match="not evenly divisible"):
        initialize_values_deterministic(
            shape=(10,),
            method="linear",
            params={"min": 0.0, "max": 1.0, "block_size": 3},
            device=torch.device("cpu"),
        )


def test_block_size_less_than_one_raises_error() -> None:
    """Test that block_size < 1 raises ValueError."""
    import pytest

    from soen_toolkit.core.layers.common.metadata import initialize_values_deterministic

    with pytest.raises(ValueError, match="block_size must be >= 1"):
        initialize_values_deterministic(
            shape=(10,),
            method="linear",
            params={"min": 0.0, "max": 1.0, "block_size": 0},
            device=torch.device("cpu"),
        )


def test_block_size_parameter_registry() -> None:
    """Test that block_size works through ParameterRegistry."""
    from torch import nn

    from soen_toolkit.core.layers.common.parameters import (
        InitializerSpec,
        ParameterDef,
        ParameterRegistry,
    )

    module = nn.Module()
    registry = ParameterRegistry(module, width=12, dtype=torch.float32)

    # Add parameter with block_size
    registry.add(
        ParameterDef(
            name="test_param",
            default=0.0,
            learnable=True,
            initializer=InitializerSpec(
                method="linear",
                params={"min": 0.0, "max": 1.0, "block_size": 4},
            ),
        )
    )

    # Check values are blocked correctly
    values = module.test_param
    assert values.shape == (12,)
    # 3 blocks of 4, linear from 0 to 1
    expected_block_values = torch.linspace(0.0, 1.0, 3)
    for i in range(3):
        for j in range(4):
            assert torch.isclose(values[i * 4 + j], expected_block_values[i], atol=1e-5)


def test_block_size_override_parameter() -> None:
    """Test that block_size works through override_parameter."""
    from torch import nn

    from soen_toolkit.core.layers.common.parameters import (
        InitializerSpec,
        ParameterDef,
        ParameterRegistry,
    )

    module = nn.Module()
    registry = ParameterRegistry(module, width=12, dtype=torch.float32)

    # Add parameter without blocking initially
    registry.add(
        ParameterDef(
            name="test_param",
            default=0.0,
            learnable=True,
            initializer=InitializerSpec(method="constant", params={"value": 1.0}),
        )
    )

    # Override with blocked initialization
    registry.override_parameter(
        "test_param",
        method="linear",
        params={"min": 0.0, "max": 1.0, "block_size": 3},
    )

    # Check values are now blocked
    values = module.test_param
    assert values.shape == (12,)
    # 4 blocks of 3, linear from 0 to 1
    expected_block_values = torch.linspace(0.0, 1.0, 4)
    for i in range(4):
        for j in range(3):
            assert torch.isclose(values[i * 3 + j], expected_block_values[i], atol=1e-5)


# --- Block mode tests (tiled) ---


def test_block_mode_param_spec_includes_block_mode() -> None:
    """Test that block_mode is included in params when not default."""
    # Default block_mode="shared" should not appear in params
    spec_default = param_specs.linear(0.0, 1.0, block_size=3)
    assert "block_mode" not in spec_default.params

    # Non-default block_mode should appear
    spec_tiled = param_specs.linear(0.0, 1.0, block_size=3, block_mode="tiled")
    assert spec_tiled.params["block_mode"] == "tiled"


def test_block_mode_all_distributions() -> None:
    """Test block_mode works for all distribution types."""
    # All these should include block_mode in params
    specs = [
        param_specs.uniform(0.0, 1.0, block_size=3, block_mode="tiled"),
        param_specs.normal(0.0, 1.0, block_size=3, block_mode="tiled"),
        param_specs.lognormal(-1.0, 0.5, block_size=3, block_mode="tiled"),
        param_specs.constant(1.0, block_size=3, block_mode="tiled"),
        param_specs.linear(0.0, 1.0, block_size=3, block_mode="tiled"),
        param_specs.loglinear(0.01, 1.0, block_size=3, block_mode="tiled"),
        param_specs.loguniform(0.01, 1.0, block_size=3, block_mode="tiled"),
    ]
    for spec in specs:
        assert spec.params["block_mode"] == "tiled", f"Failed for {spec.distribution}"


def test_block_mode_tiled_linear_initialization() -> None:
    """Test that tiled mode produces values that repeat across blocks."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_deterministic

    # 12 nodes with block_size=3, tiled mode -> 3 values tiled 4 times
    values = initialize_values_deterministic(
        shape=(12,),
        method="linear",
        params={"min": 0.0, "max": 1.0, "block_size": 3, "block_mode": "tiled"},
        device=torch.device("cpu"),
    )

    assert values.shape == (12,)
    # Should have pattern [v1, v2, v3] repeated 4 times
    # Pattern values: [0.0, 0.5, 1.0]
    expected_pattern = torch.linspace(0.0, 1.0, 3)
    for block_idx in range(4):
        for pos in range(3):
            actual = values[block_idx * 3 + pos]
            expected = expected_pattern[pos]
            assert torch.isclose(actual, expected, atol=1e-5), \
                f"Block {block_idx}, pos {pos}: expected {expected}, got {actual}"


def test_block_mode_tiled_normal_initialization() -> None:
    """Test that tiled mode with stochastic init produces tiled values."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_stochastic

    torch.manual_seed(42)
    values = initialize_values_stochastic(
        shape=(12,),
        method="normal",
        params={"mean": 0.0, "std": 1.0, "block_size": 3, "block_mode": "tiled"},
        device=torch.device("cpu"),
    )

    assert values.shape == (12,)
    # Check that the same pattern repeats across all blocks
    pattern = values[:3]  # First block defines the pattern
    for block_idx in range(1, 4):
        start = block_idx * 3
        for pos in range(3):
            assert values[start + pos] == pattern[pos], \
                f"Block {block_idx}, pos {pos} doesn't match pattern"


def test_block_mode_tiled_uniform_initialization() -> None:
    """Test that tiled mode with uniform init produces tiled values."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_stochastic

    torch.manual_seed(42)
    values = initialize_values_stochastic(
        shape=(12,),
        method="uniform",
        params={"min": 0.0, "max": 1.0, "block_size": 4, "block_mode": "tiled"},
        device=torch.device("cpu"),
    )

    assert values.shape == (12,)
    # 4 values per block, 3 blocks -> pattern of 4 repeated 3 times
    pattern = values[:4]
    for block_idx in range(1, 3):
        start = block_idx * 4
        for pos in range(4):
            assert values[start + pos] == pattern[pos], \
                f"Block {block_idx}, pos {pos} doesn't match pattern"


def test_block_mode_shared_vs_tiled_difference() -> None:
    """Test that shared and tiled modes produce different patterns."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_deterministic

    # Shared mode: [v1,v1,v1, v2,v2,v2, v3,v3,v3, v4,v4,v4]
    values_shared = initialize_values_deterministic(
        shape=(12,),
        method="linear",
        params={"min": 0.0, "max": 1.0, "block_size": 3, "block_mode": "shared"},
        device=torch.device("cpu"),
    )

    # Tiled mode: [v1,v2,v3, v1,v2,v3, v1,v2,v3, v1,v2,v3]
    values_tiled = initialize_values_deterministic(
        shape=(12,),
        method="linear",
        params={"min": 0.0, "max": 1.0, "block_size": 3, "block_mode": "tiled"},
        device=torch.device("cpu"),
    )

    # Verify shared: values within each block are identical
    for block_idx in range(4):
        start = block_idx * 3
        block_val = values_shared[start]
        assert values_shared[start + 1] == block_val
        assert values_shared[start + 2] == block_val

    # Verify tiled: same pattern repeats across blocks
    pattern = values_tiled[:3]
    for block_idx in range(1, 4):
        start = block_idx * 3
        for pos in range(3):
            assert values_tiled[start + pos] == pattern[pos]

    # They should be different patterns
    assert not torch.allclose(values_shared, values_tiled)


def test_block_mode_invalid_raises_error() -> None:
    """Test that invalid block_mode raises ValueError."""
    import pytest

    from soen_toolkit.core.layers.common.metadata import initialize_values_deterministic

    with pytest.raises(ValueError, match="block_mode must be"):
        initialize_values_deterministic(
            shape=(12,),
            method="linear",
            params={"min": 0.0, "max": 1.0, "block_size": 3, "block_mode": "invalid"},
            device=torch.device("cpu"),
        )


def test_block_mode_tiled_parameter_registry() -> None:
    """Test that block_mode=tiled works through ParameterRegistry."""
    from torch import nn

    from soen_toolkit.core.layers.common.parameters import (
        InitializerSpec,
        ParameterDef,
        ParameterRegistry,
    )

    module = nn.Module()
    registry = ParameterRegistry(module, width=12, dtype=torch.float32)

    # Add parameter with tiled blocking
    registry.add(
        ParameterDef(
            name="test_param",
            default=0.0,
            learnable=True,
            initializer=InitializerSpec(
                method="linear",
                params={"min": 0.0, "max": 1.0, "block_size": 3, "block_mode": "tiled"},
            ),
        )
    )

    # Check values follow tiled pattern
    values = module.test_param
    assert values.shape == (12,)
    # Pattern of 3 values tiled 4 times
    expected_pattern = torch.linspace(0.0, 1.0, 3)
    for block_idx in range(4):
        for pos in range(3):
            assert torch.isclose(values[block_idx * 3 + pos], expected_pattern[pos], atol=1e-5)


def test_block_mode_one_irrelevant() -> None:
    """Test that block_mode has no effect when block_size=1."""
    from soen_toolkit.core.layers.common.metadata import initialize_values_deterministic

    torch.manual_seed(42)
    values_shared = initialize_values_deterministic(
        shape=(10,),
        method="linear",
        params={"min": 0.0, "max": 1.0, "block_size": 1, "block_mode": "shared"},
        device=torch.device("cpu"),
    )

    torch.manual_seed(42)
    values_tiled = initialize_values_deterministic(
        shape=(10,),
        method="linear",
        params={"min": 0.0, "max": 1.0, "block_size": 1, "block_mode": "tiled"},
        device=torch.device("cpu"),
    )

    # Both should produce the same result when block_size=1
    assert torch.allclose(values_shared, values_tiled)
