# FILEPATH: src/soen_toolkit/training/callbacks/stateful_training.py

"""Stateful training callback for carrying forward layer states across batches."""

import logging
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch

logger = logging.getLogger(__name__)


class StatefulTrainingCallback(Callback):
    """Carry forward final layer states from one batch to initialize the next batch.

    This callback enables temporal continuity across training samples by using the
    final states from one batch as initial states for the next batch. This can be
    useful for continual learning scenarios or when temporal dependencies exist
    across samples.

    The states reset at the start of each epoch, so continuity is maintained only
    within an epoch. For MultiplierV2 layers, this also carries forward the s1 and
    s2 auxiliary states.

    Parameters
    ----------
    enable_for_training: bool
        Enable state carryover during training. Defaults to False.
    enable_for_validation: bool
        Enable state carryover during validation. Defaults to False.
    sample_selection: str
        Which batch sample to extract final states from. Options:
        - "random": Pick a random sample from the batch
        - "first": Always use the first sample (index 0)
        - "last": Always use the last sample (index -1)
        Defaults to "random".
    verbose: bool
        Log state carryover operations. Defaults to False.

    """

    def __init__(
        self,
        *,
        enable_for_training: bool = False,
        enable_for_validation: bool = False,
        sample_selection: str = "random",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.enable_for_training = bool(enable_for_training)
        self.enable_for_validation = bool(enable_for_validation)

        if sample_selection not in {"random", "first", "last"}:
            msg = f"sample_selection must be 'random', 'first', or 'last', got {sample_selection}"
            raise ValueError(msg)
        self.sample_selection = str(sample_selection)
        self.verbose = bool(verbose)

        # Training state storage
        self._train_states: dict[int, torch.Tensor] | None = None
        self._train_s1_states: dict[int, torch.Tensor] | None = None
        self._train_s2_states: dict[int, torch.Tensor] | None = None

        # Validation state storage
        self._val_states: dict[int, torch.Tensor] | None = None
        self._val_s1_states: dict[int, torch.Tensor] | None = None
        self._val_s2_states: dict[int, torch.Tensor] | None = None

    def _pick_sample_index(self, batch_size: int) -> int:
        """Select which sample index to extract states from."""
        if self.sample_selection == "first":
            return 0
        if self.sample_selection == "last":
            return batch_size - 1
        # random
        return int(torch.randint(0, batch_size, (1,)).item())

    def _extract_final_states(
        self,
        all_states: list[torch.Tensor],
        layers_config: list,
        sample_idx: int
    ) -> dict[int, torch.Tensor]:
        """Extract final timestep states from all layers for a specific sample.

        Args:
            all_states: List of state histories [batch, time+1, dim] for each layer
            layers_config: List of layer configurations with layer_id
            sample_idx: Which batch sample to extract

        Returns:
            Dict mapping layer_id to final state tensor [dim]
        """
        states = {}
        for layer_hist, cfg in zip(all_states, layers_config, strict=False):
            # Extract final timestep [:, -1, :] for the selected sample
            final_state = layer_hist[sample_idx, -1, :].detach().clone()
            states[cfg.layer_id] = final_state
        return states

    def _extract_multiplier_states(
        self,
        model: Any,
        sample_idx: int,
    ) -> tuple[dict[int, torch.Tensor] | None, dict[int, torch.Tensor] | None]:
        """Extract s1 and s2 final states from MultiplierV2 layers.

        Args:
            model: The SOEN model core
            sample_idx: Which batch sample to extract

        Returns:
            Tuple of (s1_states dict, s2_states dict), or (None, None) if no states found
        """
        s1_states = {}
        s2_states = {}

        for idx, cfg in enumerate(model.layers_config):
            layer = model.layers[idx]
            # Check if this is a MultiplierV2 layer with stored final states
            if hasattr(layer, '_s1_final') and hasattr(layer, '_s2_final'):
                s1_final = layer._s1_final
                s2_final = layer._s2_final
                if s1_final is not None and s2_final is not None:
                    # Extract the sample
                    s1_states[cfg.layer_id] = s1_final[sample_idx, :].detach().clone()
                    s2_states[cfg.layer_id] = s2_final[sample_idx, :].detach().clone()

        # Return None if no states were found
        if not s1_states:
            return None, None
        return s1_states, s2_states

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clear training states at the start of each epoch."""
        if self.enable_for_training:
            self._train_states = None
            self._train_s1_states = None
            self._train_s2_states = None
            if self.verbose:
                logger.info("[StatefulTraining] Reset training states at epoch start")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clear validation states at the start of each validation epoch."""
        if self.enable_for_validation:
            self._val_states = None
            self._val_s1_states = None
            self._val_s2_states = None
            if self.verbose:
                logger.info("[StatefulTraining] Reset validation states at epoch start")

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int
    ) -> None:
        """Inject stored states before training forward pass."""
        if not self.enable_for_training:
            return

        # Set pending states on the module for forward to use
        if self._train_states is not None:
            pl_module._pending_initial_states = self._train_states
            if self.verbose and batch_idx % 10 == 0:
                logger.info(f"[StatefulTraining] Injecting training states at batch {batch_idx}")

        if self._train_s1_states is not None:
            pl_module._pending_s1_states = self._train_s1_states

        if self._train_s2_states is not None:
            pl_module._pending_s2_states = self._train_s2_states

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Inject stored states before validation forward pass."""
        if not self.enable_for_validation:
            return

        # Set pending states on the module for forward to use
        if self._val_states is not None:
            pl_module._pending_initial_states = self._val_states
            if self.verbose and batch_idx % 10 == 0:
                logger.info(f"[StatefulTraining] Injecting validation states at batch {batch_idx}")

        if self._val_s1_states is not None:
            pl_module._pending_s1_states = self._val_s1_states

        if self._val_s2_states is not None:
            pl_module._pending_s2_states = self._val_s2_states

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Extract and store final states after training forward pass."""
        if not self.enable_for_training:
            return

        # Access the stored states from the last forward pass
        all_states = getattr(pl_module, 'latest_all_states', None)
        if all_states is None or len(all_states) == 0:
            return

        # Determine batch size from the first layer's states
        batch_size = all_states[0].shape[0]
        sample_idx = self._pick_sample_index(batch_size)

        # Extract main states
        model = getattr(pl_module, 'model', None)
        if model is None:
            return

        self._train_states = self._extract_final_states(
            all_states,
            model.layers_config,
            sample_idx
        )

        # Extract MultiplierV2 s1/s2 states if present
        s1_states, s2_states = self._extract_multiplier_states(model, sample_idx)
        self._train_s1_states = s1_states
        self._train_s2_states = s2_states

        if self.verbose and batch_idx % 10 == 0:
            num_layers = len(self._train_states)
            has_multiplier = s1_states is not None
            logger.info(
                f"[StatefulTraining] Stored training states from sample {sample_idx} "
                f"({num_layers} layers, MultiplierV2={has_multiplier})"
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Extract and store final states after validation forward pass."""
        if not self.enable_for_validation:
            return

        # Access the stored states from the last forward pass
        all_states = getattr(pl_module, 'latest_all_states', None)
        if all_states is None or len(all_states) == 0:
            return

        # Determine batch size from the first layer's states
        batch_size = all_states[0].shape[0]
        sample_idx = self._pick_sample_index(batch_size)

        # Extract main states
        model = getattr(pl_module, 'model', None)
        if model is None:
            return

        self._val_states = self._extract_final_states(
            all_states,
            model.layers_config,
            sample_idx
        )

        # Extract MultiplierV2 s1/s2 states if present
        s1_states, s2_states = self._extract_multiplier_states(model, sample_idx)
        self._val_s1_states = s1_states
        self._val_s2_states = s2_states

        if self.verbose and batch_idx % 10 == 0:
            num_layers = len(self._val_states)
            has_multiplier = s1_states is not None
            logger.info(
                f"[StatefulTraining] Stored validation states from sample {sample_idx} "
                f"({num_layers} layers, MultiplierV2={has_multiplier})"
            )

