from __future__ import annotations

import math

import torch

from soen_toolkit.analysis import GradientStatsCollector
from soen_toolkit.core import SOENModelCore


def _build_tiny_model() -> SOENModelCore:
    config = {
        "simulation": {"dt": 37, "network_evaluation_method": "layerwise"},
        "layers": [
            {"layer_id": 0, "layer_type": "Linear", "params": {"dim": 4}},
            {
                "layer_id": 1,
                "layer_type": "SingleDendrite",
                "params": {
                    "dim": 5,
                    "solver": "FE",
                    "source_func_type": "RateArray",
                    "bias_current": 1.7,
                    "gamma_plus": 0.001,
                    "gamma_minus": 0.001,
                },
            },
        ],
        "connections": [
            {"from_layer": 0, "to_layer": 1, "connection_type": "dense"},
        ],
    }
    return SOENModelCore.build(config)


def test_gradient_stats_collector_tracks_basic_stats():
    collector = GradientStatsCollector(track_per_step=True, max_steps_per_param=4)
    grad = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.0]])
    collector.record("test_param", grad, step=0)

    payload = collector.to_dict()
    assert "test_param" in payload["parameters"]
    summary = payload["parameters"]["test_param"]["summary"]
    assert math.isclose(summary["max"], 2.0, rel_tol=1e-4)
    assert math.isclose(summary["min"], -2.0, rel_tol=1e-4)
    assert summary["steps"] == 1
    assert payload["parameters"]["test_param"]["steps"][0]["total_elements"] == grad.numel()


def test_analyze_gradient_flow_returns_stats():
    torch.manual_seed(0)
    model = _build_tiny_model()
    batch = 6
    seq_len = 8
    inputs = torch.randn(batch, seq_len, 4)
    targets = torch.randn(batch, 5)

    result = model.analyze_gradient_flow(
        inputs,
        targets,
        loss_fn="mse",
        batch_size=2,
        max_batches=2,
        output_reduction="final",
    )

    assert result["batches_processed"] == 2
    assert math.isfinite(result["loss"])
    stats = result["gradient_stats"]["parameters"]
    assert stats, "Expected gradient statistics to be populated"
    assert any(name.startswith("connections") for name in stats.keys())

