from __future__ import annotations

from dataclasses import dataclass

"""Sequence-length scheduler for the JAX backend.

This mirrors the behavior of the PyTorch Lightning callback
`TargetSeqLenScheduler` but is implemented as a lightweight utility that the
JAX training loop can call at the start of each epoch.

With dt scaling enabled, this scheduler pre-computes all sequence lengths and
corresponding dt values for each epoch. The JAX training loop then maintains
a cache of pre-jitted functions for each unique (seq_len, dt) configuration,
allowing dt to scale with sequence length without breaking JIT compilation.
"""


@dataclass
class TargetSeqLenSchedulerJAX:
    data_module: object
    start_len: int
    end_len: int
    max_epochs: int
    start_epoch: int = 0
    end_epoch: int | None = None
    scale_dt: bool = False
    base_dt: float | None = None  # Original dt from model; required if scale_dt=True

    def __post_init__(self) -> None:
        assert isinstance(self.start_len, int) and self.start_len > 0
        assert isinstance(self.end_len, int) and self.end_len > 0
        assert isinstance(self.max_epochs, int) and self.max_epochs > 0
        assert isinstance(self.start_epoch, int) and self.start_epoch >= 0
        if self.end_epoch is None:
            self.end_epoch = self.max_epochs
        assert isinstance(self.end_epoch, int)
        assert self.end_epoch >= self.start_epoch

        # Validate base_dt is provided if scaling is enabled
        if self.scale_dt:
            assert self.base_dt is not None and self.base_dt > 0.0, "base_dt must be provided and positive when scale_dt=True"

        # Ensure datamodule caching of prebuilt batches for varying lengths
        # Matches Torch path expectation; presence validated by the datamodule
        if hasattr(self.data_module, "config") and hasattr(self.data_module.config, "data"):
            self.data_module.config.data.cache_data = True
            self.data_module.config.data.target_seq_len = int(self.start_len)

        # Pre-compute all epoch configurations for JIT cache preparation
        self._epoch_configs: dict[int, tuple[int, float]] = {}
        for epoch in range(self.max_epochs):
            seq_len = self._compute_len_for_epoch(epoch)
            dt = self._compute_dt_for_epoch_incremental(epoch, seq_len)
            self._epoch_configs[epoch] = (seq_len, dt)

    def _compute_len_for_epoch(self, epoch: int) -> int:
        """Compute the target sequence length for a given epoch."""
        if epoch <= self.start_epoch:
            return int(self.start_len)
        if epoch >= int(self.end_epoch):
            return int(self.end_len)
        window = max(1, int(self.end_epoch) - int(self.start_epoch))
        progress = (epoch - int(self.start_epoch)) / float(window)
        val = round(self.start_len + progress * (self.end_len - self.start_len))
        return int(max(1, val))

    def _compute_dt_for_epoch_incremental(self, epoch: int, seq_len: int) -> float:
        """Compute the scaled dt for a given epoch and sequence length using incremental scaling.

        This method is used during initialization to build the epoch configurations.
        It matches the PyTorch implementation's incremental scaling behavior.
        """
        if not self.scale_dt or self.base_dt is None:
            return float(self.base_dt) if self.base_dt is not None else 1.0

        # For the first epoch, use the base_dt directly
        if epoch == 0:
            return float(self.base_dt)

        # For subsequent epochs, scale based on the previous epoch's length
        # This matches the PyTorch implementation's incremental scaling
        prev_epoch = epoch - 1
        prev_len, prev_dt = self._epoch_configs.get(prev_epoch, (self.start_len, self.base_dt))

        if prev_len > 0:
            scaling_factor = float(prev_len) / float(seq_len)
            return float(prev_dt * scaling_factor)
        else:
            return float(self.base_dt)

    def _compute_dt_for_epoch(self, epoch: int, seq_len: int) -> float:
        """Compute the scaled dt for a given epoch and sequence length.

        This method is used after initialization and simply returns the pre-computed value.
        """
        return self._epoch_configs.get(epoch, (seq_len, self.base_dt or 1.0))[1]

    def get_all_unique_configs(self) -> list[tuple[int, float]]:
        """Return all unique (seq_len, dt) configurations across all epochs.

        Used by the training loop to pre-compile jitted functions for each config.
        """
        unique_configs = list(set(self._epoch_configs.values()))
        return sorted(unique_configs)  # Sort for deterministic ordering

    def get_config_for_epoch(self, epoch: int) -> tuple[int, float]:
        """Get the (seq_len, dt) configuration for a specific epoch."""
        return self._epoch_configs.get(epoch, (self.start_len, self.base_dt or 1.0))

    def apply_on_epoch_start(self, *, epoch: int) -> int:
        """Apply the scheduled sequence length to the datamodule.

        Returns the new target sequence length for logging purposes.

        Note: dt scaling is handled separately by the training loop, which
        selects pre-compiled jitted functions for the current configuration.
        """
        new_len, _new_dt = self.get_config_for_epoch(int(epoch))
        prev_len = getattr(self.data_module.config.data, "target_seq_len", None)
        if prev_len != new_len:
            if hasattr(self.data_module, "update_target_seq_len"):
                self.data_module.update_target_seq_len(int(new_len))
            else:
                self.data_module.config.data.target_seq_len = int(new_len)
        return int(new_len)


__all__ = ["TargetSeqLenSchedulerJAX"]
