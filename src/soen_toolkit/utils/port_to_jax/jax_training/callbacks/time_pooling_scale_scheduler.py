from __future__ import annotations

from dataclasses import dataclass

"""Time pooling scale scheduler for JAX backend.

This mirrors the behavior of the Torch callback, updating the JAX trainer's
`TrainingConfigJAX.time_pooling_params['scale']` over epochs.
"""


@dataclass
class TimePoolingScaleSchedulerJAX:
    start_scale: float
    end_scale: float
    start_epoch: int = 0
    end_epoch: int | None = None

    def _calc(self, epoch: int, max_epochs: int) -> float:
        se = max(0, int(self.start_epoch))
        ee = int(self.end_epoch) if self.end_epoch is not None else int(max_epochs)
        if epoch <= se:
            return float(self.start_scale)
        if epoch >= ee:
            return float(self.end_scale)
        progress = (epoch - se) / max(1, (ee - se))
        return float(self.start_scale + progress * (self.end_scale - self.start_scale))

    def apply_on_epoch_start(self, *, trainer, epoch: int) -> float:
        scale = self._calc(epoch, trainer.cfg.training.max_epochs)
        try:
            p = dict(trainer.cfg.training.time_pooling_params or {})
            p["scale"] = float(scale)
            trainer.cfg.training.time_pooling_params = p
        except Exception:
            pass
        return float(scale)


__all__ = ["TimePoolingScaleSchedulerJAX"]
