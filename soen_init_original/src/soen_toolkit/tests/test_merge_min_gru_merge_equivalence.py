from pathlib import Path

import torch

from soen_toolkit.core import SOENModelCore
from soen_toolkit.utils.merge_layers import MergeSpec, apply_merge_layers


def _run_model(model: SOENModelCore, x: torch.Tensor):
    model.eval()
    with torch.no_grad():
        y, histories = model(x)
    return y.cpu(), [h.cpu() for h in histories]


def test_min_gru_forward_equivalence_after_merge() -> None:
    # Load the provided model with MinGRU layers and funky connectivity
    path = Path(__file__).parent / "temp_minGRU.soen"
    base = SOENModelCore.load(str(path), strict=True, verbose=False, show_logs=False)

    # Identify group: merge all MinGRU layers; keep the first layer (likely Input) separate
    group_ids = [cfg.layer_id for cfg in base.layers_config if cfg.layer_type == "MinGRU"]
    assert len(group_ids) >= 6  # per instructions: many MinGRU layers
    node_order = sorted(group_ids)
    spec = MergeSpec(group_ids=group_ids, new_layer_id=10, node_order=node_order, preserve_state=True)
    merged = apply_merge_layers(base, spec).model

    # Prepare a deterministic input: [batch, seq_len, input_dim]
    batch = 3
    seq_len = 8
    input_dim = base.layers_config[0].params.get("dim", 10)
    torch.manual_seed(123)
    x = torch.randn(batch, seq_len, input_dim)

    # Run both models
    y0, hist0 = _run_model(base, x)
    y1, hist1 = _run_model(merged, x)

    # Final outputs should be identical within numeric tolerance
    # Original final output corresponds to the last layer in base (by layer_id order).
    last_base_id = base.layers_config[-1].layer_id
    assert last_base_id in group_ids, "Expected the last base layer to be part of the merged group"
    # Compute slice for last_base_id within the merged super-layer's output (concatenation order=node_order)
    dims = base.layer_nodes
    start = 0
    for lid in node_order:
        if lid == last_base_id:
            break
        start += int(dims[lid])
    width = int(dims[last_base_id])
    y1_last = y1[:, :, start : start + width]
    assert torch.allclose(y0, y1_last, atol=1e-5, rtol=1e-5)

    # Also check that per-layer histories align in aggregate length (not identity of splits)
    assert len(hist0) == len(base.layers_config)
    assert len(hist1) == len(merged.layers_config)
