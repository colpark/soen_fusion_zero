import jax
import jax.numpy as jnp
import numpy as np

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.utils.port_to_jax import convert_core_model_to_jax
from soen_toolkit.utils.port_to_jax.jax_model import JAXModel


def make_random_series_jax(batch: int, seq_len: int, dim: int, seed: int = 0) -> jnp.ndarray:
    """Generate random input series for JAX testing."""
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (batch, seq_len, dim))


def build_small_model_jax(
    *,
    dims=(3, 3),
    connectivity_type="dense",
    init="constant",
    init_value=0.5,
    with_internal_first=False,
    constraints=None,
) -> tuple[SOENModelCore, JAXModel]:
    """Build a small two-layer model and convert to JAX.

    Returns:
        Tuple of (PyTorch model, JAX model converted version)
    """
    d0, d1 = int(dims[0]), int(dims[1])
    layers = [
        LayerConfig(layer_id=0, layer_type="Multiplier", params={"dim": d0}),
        LayerConfig(layer_id=1, layer_type="Multiplier", params={"dim": d1}),
    ]

    conn_params = {"init": init, "value": init_value}
    if constraints is not None:
        conn_params["constraints"] = constraints

    conns = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type=connectivity_type,
            params=conn_params,
            learnable=True,
        ),
    ]

    if with_internal_first:
        conns.append(
            ConnectionConfig(
                from_layer=0,
                to_layer=0,
                connection_type="dense",
                params={"init": init, "value": init_value},
                learnable=True,
            ),
        )

    sim = SimulationConfig(dt=37, input_type="flux", track_phi=False, track_s=False, track_g=False)
    torch_model = SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)

    jax_model = convert_core_model_to_jax(torch_model)
    return torch_model, jax_model


def jnp_allclose(a: jnp.ndarray, b: jnp.ndarray, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    """Check if two JAX arrays are close."""
    return bool(jnp.allclose(a, b, atol=atol, rtol=rtol))


def numpy_allclose(a: jnp.ndarray | np.ndarray, b: jnp.ndarray | np.ndarray, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    """Check if two arrays are close, converting JAX to numpy if needed."""
    if isinstance(a, jnp.ndarray):
        a = np.asarray(a)
    if isinstance(b, jnp.ndarray):
        b = np.asarray(b)
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))
