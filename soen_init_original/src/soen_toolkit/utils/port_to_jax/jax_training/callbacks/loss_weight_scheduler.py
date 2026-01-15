from __future__ import annotations

from dataclasses import dataclass
from typing import Any

"""Loss weight scheduler for JAX backend.

This mirrors the Torch `LossWeightScheduler` at a high-level. In JAX runner we
apply it by updating a dictionary of active loss specs inside the low-level
trainer configuration before computing the next batch.
"""


@dataclass
class LossWeightSchedulerJAX:
    loss_name: str
    scheduler_type: str
    params: dict | None = None
    start_epoch: int = 0
    end_epoch: int | None = None
    per_step: bool = True

    def _progress(self, *, epoch: int, step: int, steps_per_epoch: int, max_epochs: int) -> float:
        se = max(0, int(self.start_epoch))
        ee = int(self.end_epoch) if self.end_epoch is not None else int(max_epochs - 1)
        ee = max(ee, se)
        if self.per_step:
            start_step = se * steps_per_epoch
            end_step = ee * steps_per_epoch + (steps_per_epoch - 1)
            if step <= start_step:
                return 0.0
            if step >= end_step:
                return 1.0
            return (step - start_step) / max(1, (end_step - start_step))
        if epoch <= se:
            return 0.0
        if epoch >= ee:
            return 1.0
        return (epoch - se) / max(1, (ee - se))

    def _calc_weight(self, prog: float) -> float:
        p = max(0.0, min(1.0, float(prog)))
        params = dict(self.params or {})
        st = str(self.scheduler_type).lower()
        if st == "linear":
            return float(params.get("min_weight", 0.0)) + p * (float(params.get("max_weight", 1.0)) - float(params.get("min_weight", 0.0)))
        if st in ("exponential_decay", "exp_decay", "exponential"):
            initial = float(params.get("initial_weight", 1.0))
            if "decay_rate" in params:
                rate = float(params.get("decay_rate", 0.0))
                import math

                return float(initial * math.exp(-rate * p))
            final = float(params.get("final_weight", max(1e-12, initial)))
            import math

            rate = -math.log(max(1e-12, final) / max(1e-12, initial))
            return float(initial * math.exp(-rate * p))
        if st in ("cosine", "cosine_decay"):
            import math

            init_w = float(params.get("initial_weight", 1.0))
            final_w = float(params.get("final_weight", 0.0))
            cosine = 0.5 * (1.0 + math.cos(math.pi * p))
            return float(final_w + (init_w - final_w) * cosine)
        if st == "sinusoidal":
            import math

            min_w = float(params.get("min_weight", 0.0))
            max_w = float(params.get("max_weight", 1.0))
            angle = 2.0 * math.pi * p
            return float((min_w + max_w) / 2.0 + (max_w - min_w) * 0.5 * math.cos(angle))
        return 1.0

    def apply(self, *, trainer: Any, epoch: int, step: int, steps_per_epoch: int, max_epochs: int) -> None:
        prog = self._progress(epoch=epoch, step=step, steps_per_epoch=steps_per_epoch, max_epochs=max_epochs)
        weight = self._calc_weight(prog)
        # Update matching loss spec if present in trainer.cfg.training.losses
        try:
            for spec in trainer.cfg.training.losses:
                if str(spec.get("name", "")) == self.loss_name:
                    spec["weight"] = float(weight)
                    break
        except Exception:
            pass


__all__ = ["LossWeightSchedulerJAX"]
