import torch

from soen_toolkit.training.models.lightning_wrapper import SOENLightningModule


def _make_dummy_wrapper(method: str, params: dict, range_start=None, range_end=None):
    dummy = object.__new__(SOENLightningModule)
    dummy.time_pooling_method_name = method
    dummy.time_pooling_params = dict(params or {})
    dummy.range_start = range_start
    dummy.range_end = range_end
    return dummy


def test_time_pooling_max_and_mean_and_rms() -> None:
    B, T, D = 2, 5, 3
    s = torch.arange(B * T * D, dtype=torch.float32).view(B, T, D)

    for method in ["max", "mean", "rms"]:
        w = _make_dummy_wrapper(method, {"scale": 1.0})
        y = SOENLightningModule.process_output(w, s)
        assert y.shape == (B, D)
        # Basic sanity: max >= mean and rms >= mean for non-negative increasing sequences
        if method == "max":
            m = SOENLightningModule.process_output(_make_dummy_wrapper("mean", {"scale": 1.0}), s)
            assert torch.all(y >= m)
        if method == "rms":
            m = SOENLightningModule.process_output(_make_dummy_wrapper("mean", {"scale": 1.0}), s)
            assert torch.all(y >= m)


def test_time_pooling_final_and_mean_last_n() -> None:
    _B, _T, _D = 1, 4, 2
    s = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        ]
    )  # shape [1,4,2]

    final = SOENLightningModule.process_output(_make_dummy_wrapper("final", {"scale": 1.0}), s)
    assert torch.allclose(final, s[:, -1, :])

    # mean_last_n with n=2
    w = _make_dummy_wrapper("mean_last_n", {"n": 2, "scale": 1.0})
    last2 = SOENLightningModule.process_output(w, s)
    assert torch.allclose(last2, s[:, -2:, :].mean(dim=1))


def test_time_pooling_mean_range_defaults_and_bounds() -> None:
    B, T, D = 1, 6, 2
    s = torch.randn(B, T, D)

    # Default mean_range uses last min(50,T) → whole series here
    w_default = _make_dummy_wrapper("mean_range", {"scale": 1.0})
    y_default = SOENLightningModule.process_output(w_default, s)
    assert torch.allclose(y_default, s.mean(dim=1))

    # Specific bounded range
    w_rng = _make_dummy_wrapper("mean_range", {"scale": 1.0}, range_start=2, range_end=5)
    y_rng = SOENLightningModule.process_output(w_rng, s)
    assert torch.allclose(y_rng, s[:, 2:5, :].mean(dim=1))


def test_time_pooling_ewa_weights_shape_and_effect() -> None:
    B, T, D = 2, 7, 3
    s = torch.ones(B, T, D)
    w = _make_dummy_wrapper("ewa", {"min_weight": 0.2, "scale": 1.0})
    y = SOENLightningModule.process_output(w, s)
    assert y.shape == (B, D)
    # With ones input, weighted sum should be 1 regardless of weights normalization
    assert torch.allclose(y, torch.ones(B, D), atol=1e-6)


def test_time_pooling_scale_applied() -> None:
    B, T, D = 1, 3, 2
    s = torch.ones(B, T, D)
    w = _make_dummy_wrapper("mean", {"scale": 10.0})
    y = SOENLightningModule.process_output(w, s)
    # mean=1, scaled by 10
    assert torch.allclose(y, torch.full((B, D), 10.0))
