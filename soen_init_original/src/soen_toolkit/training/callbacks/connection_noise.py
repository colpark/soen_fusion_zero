# FILEPATH: src/soen_toolkit/training/callbacks/connection_noise.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pytorch_lightning.callbacks import Callback
import torch

# Use the built-in noise strategy machinery for connections
from soen_toolkit.core.noise import GaussianPerturbation, NoiseSettings

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class ConnectionNoiseCallback(Callback):
    """Enable fixed-per-forward Gaussian perturbation on selected connection matrices
    using the core NoiseSettings.

    This leverages the model's per-connection noise strategies (`connection_noise_settings`)
    so noise is applied within the forward pass via `NoiseSettings.apply(..., key='j')`.

    Parameters
    ----------
    - connections: names of connection parameters to perturb (e.g., ["J_0_to_1"]).
    - std: base standard deviation of the Gaussian perturbation (fixed within a forward).
    - relative: if True, scales std by mean(abs(W)) for each targeted parameter at hook time.
    - per_step: retained for API symmetry
    strategies are set at hook time and used each forward pass.
    - log_every_n_steps: step interval for emitting logs.
    - seed: optional RNG seed for reproducibility of noise.

    """

    def __init__(
        self,
        *,
        connections: Sequence[str],
        std: float,
        relative: bool = False,
        per_step: bool = True,
        log_every_n_steps: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(connections, (list, tuple)) or len(connections) == 0:
            msg = "connections must be a non-empty list/tuple of connection names"
            raise ValueError(msg)
        self.connection_names: list[str] = list(connections)
        self.base_std: float = float(std)
        self.relative: bool = bool(relative)
        self.per_step: bool = bool(per_step)
        self.log_every_n_steps: int = int(max(1, log_every_n_steps))
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def _apply_noise(self, pl_module: pl.LightningModule, *, global_step: int, on_epoch: bool, allow_logging: bool = True) -> None:
        if not hasattr(pl_module, "model"):
            return

        soen_model = pl_module.model
        if not hasattr(soen_model, "connections"):
            return

        # Ensure the connection_noise_settings dict exists
        if not hasattr(soen_model, "connection_noise_settings") or not isinstance(soen_model.connection_noise_settings, dict):
            return

        settings_dict: dict = soen_model.connection_noise_settings

        # Assign/overwrite GaussianPerturbation strategy for 'j' on each requested connection
        for name in self.connection_names:
            if name not in soen_model.connections:
                if global_step % self.log_every_n_steps == 0:
                    logger.warning(f"[ConnectionNoise] Connection '{name}' not found; skipping")
                continue

            current_settings: NoiseSettings | None = settings_dict.get(name)
            # Compute effective std. If relative=True, scale by mean(|W|) of the connection.
            try:
                J_param = soen_model.connections[name]
                scale = float(J_param.abs().mean().item()) if self.relative else 1.0
            except Exception:
                scale = 1.0
            eff_std = float(self.base_std) * scale
            strat = GaussianPerturbation(mean=0.0, std=eff_std)

            if isinstance(current_settings, NoiseSettings):
                # Replace/assign the 'j' strategy
                object.__setattr__(current_settings, "j", strat) if getattr(type(current_settings), "__setattr__", None) else setattr(current_settings, "j", strat)
            else:
                # Create a new settings container with only 'j' defined
                settings_dict[name] = NoiseSettings(j=strat)

            # Log the configured std (already includes relative scaling if enabled)
            log_name = f"callbacks/connection_noise/{name}_std"
            if allow_logging:
                pl_module.log(
                    log_name,
                    torch.tensor(float(eff_std), device=pl_module.device if hasattr(pl_module, "device") else torch.device("cpu")),
                    prog_bar=False,
                    logger=True,
                    on_step=self.per_step and not on_epoch,
                    on_epoch=on_epoch,
                )

        if global_step % self.log_every_n_steps == 0:
            mode = "step" if self.per_step and not on_epoch else "epoch"
            logger.info(
                f"[ConnectionNoise] Applied noise to {len(self.connection_names)} connections ({mode}={global_step})",
            )

    # Hooks
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx: int) -> None:
        if not self.per_step:
            return
        # Ensure strategies are set; GaussianNoise samples per forward internally
        self._apply_noise(pl_module, global_step=trainer.global_step, on_epoch=False)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.per_step:
            return
        # Ensure strategies are set at start of epoch (harmless if already set)
        self._apply_noise(pl_module, global_step=trainer.current_epoch, on_epoch=True)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Set strategies once at fit start so they are active immediately (no logging allowed here)
        self._apply_noise(pl_module, global_step=getattr(trainer, "global_step", 0), on_epoch=False, allow_logging=False)
