from soen_toolkit.core import SOENModelCore


def test_yaml_infer_mode_from_dynamic_block_v2() -> None:
    spec = {
        "simulation": {"dt": 37, "network_evaluation_method": "layerwise"},
        "layers": [
            {"layer_id": 0, "layer_type": "Linear", "params": {"dim": 10}},
            {
                "layer_id": 1,
                "layer_type": "SingleDendrite",
                "params": {
                    "dim": 5,
                    "solver": "FE",
                    "source_func_type": "RateArray",
                    "bias_current": 1.7,
                    "gamma_plus": 1e-3,
                    "gamma_minus": 1e-3,
                },
            },
        ],
        "connections": [
            {
                "from_layer": 0,
                "to_layer": 1,
                "connection_type": "dense",
                "params": {
                    "init": {"name": "uniform", "params": {"min": -0.2, "max": 0.2}},
                    "dynamic": {"alpha": 1.64053},  # no mode provided
                },
            },
        ],
    }

    core = SOENModelCore.build(spec)

    assert hasattr(core, "_connection_modes")
    assert core._connection_modes.get("J_0_to_1") == "NOCC"


def test_yaml_infer_mode_from_dynamic_block_v1() -> None:
    spec = {
        "simulation": {"dt": 37, "network_evaluation_method": "layerwise"},
        "layers": [
            {"layer_id": 0, "layer_type": "Linear", "params": {"dim": 10}},
            {
                "layer_id": 1,
                "layer_type": "SingleDendrite",
                "params": {
                    "dim": 5,
                    "solver": "FE",
                    "source_func_type": "RateArray",
                    "bias_current": 1.7,
                    "gamma_plus": 1e-3,
                    "gamma_minus": 1e-3,
                },
            },
        ],
        "connections": [
            {
                "from_layer": 0,
                "to_layer": 1,
                "connection_type": "dense",
                "params": {
                    "init": {"name": "uniform", "params": {"min": -0.2, "max": 0.2}},
                    "dynamic": {"gamma_plus": 1e-3, "bias_current": 2.0},  # no mode provided
                },
            },
        ],
    }

    core = SOENModelCore.build(spec)

    assert hasattr(core, "_connection_modes")
    assert core._connection_modes.get("J_0_to_1") == "WICC"


def test_yaml_infer_mode_from_empty_dynamic_defaults_v2() -> None:
    spec = {
        "simulation": {"dt": 37, "network_evaluation_method": "layerwise"},
        "layers": [
            {"layer_id": 0, "layer_type": "Linear", "params": {"dim": 10}},
            {
                "layer_id": 1,
                "layer_type": "SingleDendrite",
                "params": {
                    "dim": 5,
                    "solver": "FE",
                    "source_func_type": "RateArray",
                    "bias_current": 1.7,
                    "gamma_plus": 1e-3,
                    "gamma_minus": 1e-3,
                },
            },
        ],
        "connections": [
            {
                "from_layer": 0,
                "to_layer": 1,
                "connection_type": "dense",
                "params": {
                    "init": {"name": "uniform", "params": {"min": -0.2, "max": 0.2}},
                    "dynamic": {},  # infer v2 by default
                },
            },
        ],
    }

    core = SOENModelCore.build(spec)

    assert hasattr(core, "_connection_modes")
    assert core._connection_modes.get("J_0_to_1") == "NOCC"
