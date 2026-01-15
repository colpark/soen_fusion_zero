import os
import tempfile

import torch

from soen_toolkit.core import SOENModelCore
from soen_toolkit.tests.utils.test_helpers_fixture import build_small_model


def test_save_and_load_round_trip_preserves_connections_and_configs() -> None:
    m = build_small_model(dims=(3, 4), connectivity_type="dense", init="constant", init_value=0.33, with_internal_first=True)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model_test.soen")
        m.save(path)
        assert os.path.isfile(path)

        m2 = SOENModelCore.load(path, show_logs=False)

        # Check sim config
        assert m2.sim_config.dt == m.sim_config.dt
        # Layer config count and types
        assert len(m2.layers_config) == len(m.layers_config)
        assert [c.layer_type for c in m2.layers_config] == [c.layer_type for c in m.layers_config]

        # Connections keys equal
        assert set(m2.connections.keys()) == set(m.connections.keys())
        # Weight tensors equal
        for k in m.connections:
            assert torch.allclose(m.connections[k].detach(), m2.connections[k].detach())
