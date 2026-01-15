import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)


def make_random_series(batch: int, seq_len: int, dim: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, seq_len, dim)


def build_small_model(
    *,
    dims=(3, 3),
    connectivity_type="dense",
    init="constant",
    init_value=0.5,
    with_internal_first=False,
    constraints=None,
) -> SOENModelCore:
    """Build a small two-layer model with configurable connectivity and constraints.

    - dims: tuple of (dim0, dim1)
    - connectivity_type: any supported by layers.connectivity
    - init: weight initializer name (e.g., 'constant')
    - init_value: scalar for 'constant' initializer
    - with_internal_first: if True, adds an internal connection for layer 0
    - constraints: optional dict to attach to the inter-layer connection params
    """
    d0, d1 = int(dims[0]), int(dims[1])
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": d0}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": d1}),
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
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)
