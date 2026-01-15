from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from soen_toolkit.training.models.lightning_wrapper import (  # noqa: E402
    SOENLightningModule,
)
from soen_toolkit.utils.port_to_jax.jax_training.pooling import (  # noqa: E402
    apply_time_pooling,
)


def _make_dummy_wrapper(method: str, params: dict | None = None, *, range_start=None, range_end=None):
    dummy = object.__new__(SOENLightningModule)
    dummy.time_pooling_method_name = method
    dummy.time_pooling_params = dict(params or {})
    dummy.range_start = range_start
    dummy.range_end = range_end
    dummy.autoregressive = False
    return dummy


@pytest.mark.parametrize(
    ("method", "params", "range_start", "range_end"),
    [
        ("max", {"scale": 1.0}, None, None),
        ("mean", {"scale": 1.0}, None, None),
        ("rms", {"scale": 1.0}, None, None),
        ("final", {"scale": 1.0}, None, None),
        ("mean_last_n", {"n": 2, "scale": 1.0}, None, None),
        ("mean_range", {"scale": 1.0}, 1, 4),
        ("mean_range", {"scale": 1.0}, None, None),
        ("ewa", {"min_weight": 0.2, "scale": 1.0}, None, None),
        ("mean", {"scale": 10.0}, None, None),
    ],
)
def test_apply_time_pooling_matches_torch(method, params, range_start, range_end) -> None:
    if method == "ewa" and range_start is not None:
        pytest.skip("Not applicable combination")

    batch, timesteps, dim = 2, 5, 3
    arr = np.arange(batch * timesteps * dim, dtype=np.float32).reshape(batch, timesteps, dim)
    torch_tensor = torch.from_numpy(arr)
    jax_tensor = jnp.asarray(arr)

    wrapper = _make_dummy_wrapper(method, params, range_start=range_start, range_end=range_end)
    torch_result = SOENLightningModule.process_output(wrapper, torch_tensor)
    jax_result = apply_time_pooling(
        jax_tensor,
        method,
        params,
        range_start=range_start,
        range_end=range_end,
    )

    torch_np = torch_result.detach().cpu().numpy() if hasattr(torch_result, "detach") else torch_result.numpy()
    np.testing.assert_allclose(torch_np, np.asarray(jax_result), rtol=1e-5, atol=1e-5)
