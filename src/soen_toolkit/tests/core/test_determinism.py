import torch

from soen_toolkit.tests.utils.test_helpers_fixture import (
    build_small_model,
    make_random_series,
)


def test_forward_determinism_no_noise_fixed_seed() -> None:
    torch.manual_seed(123)
    m = build_small_model(dims=(4, 4), connectivity_type="dense", init="constant", init_value=0.05)
    x = make_random_series(batch=2, seq_len=5, dim=4, seed=999)

    y1, _ = m(x)
    # Reset nothing; just run again with same inputs (no noise in model)
    y2, _ = m(x)
    assert torch.allclose(y1, y2)


def test_forward_determinism_across_instances_same_seed() -> None:
    torch.manual_seed(42)
    m1 = build_small_model(dims=(3, 3), connectivity_type="dense", init="constant", init_value=0.1)
    torch.manual_seed(42)
    m2 = build_small_model(dims=(3, 3), connectivity_type="dense", init="constant", init_value=0.1)

    x = make_random_series(batch=1, seq_len=3, dim=3, seed=7)
    y1, _ = m1(x)
    y2, _ = m2(x)
    assert torch.allclose(y1, y2)
