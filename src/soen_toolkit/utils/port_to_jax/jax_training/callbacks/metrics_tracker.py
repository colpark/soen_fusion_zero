from __future__ import annotations

from dataclasses import dataclass, field

"""Minimal metrics tracking for the JAX backend.

The high-level `JaxRunner` already logs step-/epoch-level metrics to the same
TensorBoard and MLflow loggers that the Torch path uses. This module provides a
thin helper class to aggregate and expose history in a similar spirit as the
Torch `MetricsTracker` callback, without tight coupling to Lightning APIs.
"""


@dataclass
class MetricsTrackerJAX:
    metric_names: list[str]
    history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }
    )

    def log_epoch(self, *, epoch: int, train_loss: float, val_loss: float, lr: float, extra: dict[str, float] | None = None) -> None:
        self.history["epoch"].append(float(epoch))
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["learning_rate"].append(float(lr))

        if extra:
            for k, v in extra.items():
                key = str(k)
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(float(v))


__all__ = ["MetricsTrackerJAX"]
