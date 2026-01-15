"""QAT callback for SOEN model training.

FILEPATH: src/soen_toolkit/training/callbacks/qat.py
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class QATStraightThroughCallback(Callback):
    """Enable straight-through estimation for quantization-aware training.

    This callback enables STE on the SOEN core model at fit start, using
    uniform codebooks that match the robustness/tooling semantics.
    """

    def __init__(
        self,
        *,
        min_val: float,
        max_val: float,
        bits: int | None = None,
        levels: int | None = None,
        connections: list[str] | None = None,
        update_on_train_epoch_start: bool = False,
        stochastic_rounding: bool = False,
    ) -> None:
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.bits = bits
        self.levels = levels
        self.connections = connections
        self.update_on_train_epoch_start = bool(update_on_train_epoch_start)
        # Toggle for probabilistic/stochastic rounding in STE
        self.stochastic_rounding = bool(stochastic_rounding)
        # Sign preconditioning removed; legacy args are ignored upstream

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = getattr(pl_module, "model", None)
        if model is None or not hasattr(model, "enable_qat_ste"):
            return
        model.enable_qat_ste(
            min_val=self.min_val,
            max_val=self.max_val,
            bits=self.bits,
            levels=self.levels,
            connections=self.connections,
            stochastic_rounding=self.stochastic_rounding,
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.update_on_train_epoch_start:
            return
        # Re-emit the enable call in case device/dtype or connections changed after load
        self.on_fit_start(trainer, pl_module)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = getattr(pl_module, "model", None)
        if model is None or not hasattr(model, "disable_qat_ste"):
            return
        model.disable_qat_ste()
