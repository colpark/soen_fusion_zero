"""Lightweight test for stateful BPTT detachment.

This checks the core failure mode your colleague referred to:
If you carry recurrent state from batch 1 into batch 2 without detaching,
then a loss on batch 2 will (incorrectly) have non-zero gradient w.r.t.
the inputs of batch 1.

With `jax.lax.stop_gradient` applied to the carried state at the batch boundary,
that cross-batch gradient must be exactly zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from soen_toolkit.core import ConnectionConfig, LayerConfig, SimulationConfig, SOENModelCore
from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax


def _build_two_layer_multiplier_model() -> SOENModelCore:
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": 3}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": 3}),
    ]
    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "constant", "value": 0.05},
            learnable=False,
        ),
    ]
    sim = SimulationConfig(dt=1.0, input_type="flux", track_phi=False, track_s=False, track_g=False)
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def test_stateful_bptt_requires_detach_at_batch_boundary() -> None:
    torch_model = _build_two_layer_multiplier_model()
    jax_model = convert_core_model_to_jax(torch_model)

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    # Two "batches" (same shapes). This is intentionally tiny.
    x1 = jax.random.normal(key1, (2, 4, 3))
    x2 = jax.random.normal(key2, (2, 4, 3))

    def loss_without_detach(x1_in: jax.Array, x2_in: jax.Array) -> jax.Array:
        # Run batch 1 to get states
        _y1, h1 = jax_model(x1_in)
        # Carry final layer states into batch 2 (dict of arrays)
        init_states = {0: h1[0][:, -1, :], 1: h1[1][:, -1, :]}

        y2, _h2 = jax_model(x2_in, initial_states=init_states)
        # Scalar loss only on batch 2
        return jnp.mean(y2[:, -1, :])

    def loss_with_detach(x1_in: jax.Array, x2_in: jax.Array) -> jax.Array:
        _y1, h1 = jax_model(x1_in)
        init_states = {0: h1[0][:, -1, :], 1: h1[1][:, -1, :]}
        init_states = jax.tree_util.tree_map(jax.lax.stop_gradient, init_states)

        y2, _h2 = jax_model(x2_in, initial_states=init_states)
        return jnp.mean(y2[:, -1, :])

    # Gradient of batch-2 loss w.r.t. batch-1 inputs:
    # - Without detach: should be non-zero (graph crosses boundary).
    # - With detach: must be zero (boundary severed).
    g_no_detach = jax.grad(lambda x: loss_without_detach(x, x2))(x1)
    g_detach = jax.grad(lambda x: loss_with_detach(x, x2))(x1)

    norm_no_detach = jnp.linalg.norm(g_no_detach)
    norm_detach = jnp.linalg.norm(g_detach)

    assert norm_no_detach > 1e-8, f"Expected cross-batch gradient without detach; got norm {norm_no_detach}"
    assert norm_detach < 1e-12, f"Expected zero cross-batch gradient with detach; got norm {norm_detach}"


