import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.training.callbacks.qat import QATStraightThroughCallback


class DummyPLModule:
    def __init__(self, model) -> None:
        self.model = model


def build_model_with_internal_and_inter(dim=3):
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": dim, "internal_J": torch.randn(dim, dim)}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": dim}),
    ]
    conns = [ConnectionConfig(from_layer=0, to_layer=1, connection_type="dense", params={}, learnable=True)]
    sim = SimulationConfig(dt=37, input_type="flux")
    return SOENModelCore(sim_config=sim, layers_config=layers, connections_config=conns)


def test_qat_callback_toggles_flags() -> None:
    model = build_model_with_internal_and_inter()
    pl_mod = DummyPLModule(model)

    cb = QATStraightThroughCallback(min_val=-0.24, max_val=0.24, bits=3, connections=["internal_0", "J_0_to_1"])
    cb.on_fit_start(None, pl_mod)

    # Model-level flags
    assert model._qat_ste_active is True
    assert model._qat_codebook is not None

    # Layer-level flags
    for layer in model.layers:
        assert getattr(layer, "_qat_ste_active", False) is True
        assert getattr(layer, "_qat_codebook", None) is not None

    cb.on_fit_end(None, pl_mod)
    assert model._qat_ste_active is False
    for layer in model.layers:
        assert getattr(layer, "_qat_ste_active", True) is False


def test_qat_callback_stochastic_rounding_flag_propagates() -> None:
    model = build_model_with_internal_and_inter()
    pl_mod = DummyPLModule(model)

    cb = QATStraightThroughCallback(
        min_val=-0.24,
        max_val=0.24,
        bits=3,
        connections=["internal_0", "J_0_to_1"],
        stochastic_rounding=True,
    )
    cb.on_fit_start(None, pl_mod)

    assert getattr(model, "_qat_stochastic_rounding", False) is True
    for layer in model.layers:
        assert getattr(layer, "_qat_stochastic_rounding", False) is True

    cb.on_fit_end(None, pl_mod)
    assert getattr(model, "_qat_stochastic_rounding", True) is False
    for layer in model.layers:
        assert getattr(layer, "_qat_stochastic_rounding", True) is False
