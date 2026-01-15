# FILEPATH: src/soen_toolkit/training/callbacks/metrics.py

"""Metrics for SOEN model training."""

from collections.abc import Callable
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import torch.nn.functional as F

from soen_toolkit.utils.quantization import (
    calculate_num_levels,
    generate_uniform_codebook,
)

logger = logging.getLogger(__name__)


class QuantizedAccuracyCallback(Callback):
    """Evaluate and log accuracy with hard-quantized connections during training.

    Parameters
    ----------
    min_val: float
        Minimum codebook range.
    max_val: float
        Maximum codebook range.
    bits: Optional[int]
        If provided, number of levels is 2**bits.
    levels: Optional[int]
        Number of levels (exclusive of zero-counting requirement handled internally).
    connections: Optional[List[str]]
        Connection keys to quantize (e.g., ["J_1_to_2", "internal_2"]). If None, quantize all learnable.
    log_tag: str
        Metric tag prefix for TensorBoard.
    eval_every_n_steps: Optional[int]
        If an integer N is provided, evaluate every N training steps.
        If None, evaluate once per epoch at validation end only.
    max_eval_batches: Optional[int]
        If provided, limit evaluation to the first N validation batches. If None, use the full
        validation set.

    """

    def __init__(
        self,
        *,
        min_val: float,
        max_val: float,
        bits: int | None = None,
        levels: int | None = None,
        connections: list[str] | None = None,
        log_tag: str = "metrics/quantized_accuracy",
        eval_every_n_steps: int | None = None,
        max_eval_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.bits = bits
        self.levels = levels
        self.connections = set(connections) if connections else None
        self.log_tag = log_tag
        # None => evaluate only at validation epoch end
        if eval_every_n_steps is not None:
            try:
                eval_every_n_steps = int(eval_every_n_steps)
            except Exception:
                eval_every_n_steps = None
        self.eval_every_n_steps = eval_every_n_steps
        # None means evaluate over full validation set
        if max_eval_batches is not None:
            try:
                max_eval_batches = int(max_eval_batches)
            except Exception:
                max_eval_batches = None
            if isinstance(max_eval_batches, int) and max_eval_batches <= 0:
                max_eval_batches = None
        self.max_eval_batches = max_eval_batches

    @staticmethod
    def _generate_codebook(min_val: float, max_val: float, num_levels: int) -> torch.Tensor:
        return generate_uniform_codebook(min_val, max_val, num_levels)

    def _quantize_in_place(self, soen_model, codebook: torch.Tensor, target_connections: set[str] | None) -> None:
        # Hard round each learnable connection to nearest codebook value
        for name, param in soen_model.connections.items():
            if not param.requires_grad:
                continue
            if target_connections is not None and name not in target_connections:
                continue
            with torch.no_grad():
                flat = param.view(-1)
                diffs = (flat.unsqueeze(1) - codebook.unsqueeze(0)).abs()
                idx = diffs.argmin(dim=1)
                snapped = codebook[idx].view_as(param)
                param.copy_(snapped)

    def _eval_quantized(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> float:
        # Build codebook
        if self.bits is not None or self.levels is not None:
            num_levels = calculate_num_levels(bits=self.bits, levels=self.levels)
        else:
            msg = "QuantizedAccuracyCallback requires 'bits' or 'levels'"
            raise ValueError(msg)

        codebook = self._generate_codebook(self.min_val, self.max_val, num_levels).to(pl_module.device)

        # Snapshot original connections
        soen_model = pl_module.model
        originals: dict[str, torch.Tensor] = {}
        for name, param in soen_model.connections.items():
            if not param.requires_grad:
                continue
            if self.connections is not None and name not in self.connections:
                continue
            originals[name] = param.detach().clone()

        # Remember training mode
        was_training = pl_module.training
        accuracy = None
        try:
            # Quantize in-place
            self._quantize_in_place(soen_model, codebook, self.connections)

            # Evaluate on a small validation batch set
            pl_module.eval()
            val_loader = trainer.datamodule.val_dataloader() if hasattr(trainer, "datamodule") else None
            if val_loader is not None:
                correct = 0
                total = 0
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        x = x.to(pl_module.device)
                        y = y.to(pl_module.device)
                        # Call LightningModule.forward which returns (output, final_state, all_states)
                        output, _, _ = pl_module(x)
                        logits = output
                        preds = logits.argmax(dim=-1)
                        correct += (preds == y).sum().item()
                        total += y.numel()
                        if self.max_eval_batches is not None and (i + 1) >= self.max_eval_batches:
                            break
                if total > 0:
                    accuracy = 100.0 * correct / total
        finally:
            # Restore originals and training mode even if evaluation fails
            for name, tensor in originals.items():
                soen_model.connections[name].data.copy_(tensor)
            if was_training:
                pl_module.train()
            else:
                pl_module.eval()

        return accuracy if accuracy is not None else float("nan")

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        # Skip during sanity check
        if getattr(trainer, "sanity_checking", False):
            return
        # If eval_every_n_steps is None, skip batch-level evaluation (epoch-only mode)
        if self.eval_every_n_steps is None:
            return
        # Evaluate every N steps
        if trainer.global_step % max(1, int(self.eval_every_n_steps)) != 0:
            return
        try:
            acc = self._eval_quantized(trainer, pl_module)
            if acc == acc:  # not NaN
                pl_module.log(self.log_tag, acc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        except Exception as e:
            logger.warning(f"QuantizedAccuracyCallback evaluation failed: {e}")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Skip during sanity check to avoid long pre-training runs
        if getattr(trainer, "sanity_checking", False):
            return
        # If eval_every_n_steps is specified, prefer step-based evaluation path
        if self.eval_every_n_steps is not None:
            return
        try:
            acc = self._eval_quantized(trainer, pl_module)
            if acc == acc:  # not NaN
                pl_module.log(self.log_tag, acc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        except Exception as e:
            logger.warning(f"QuantizedAccuracyCallback epoch evaluation failed: {e}")


"""
Metrics registry and common metric implementations for SOEN training framework.

This module defines a simple registry pattern that allows metrics to be
registered via a decorator and retrieved dynamically at runtime.  It also
implements several common metrics used in classification problems.

The registry pattern mirrors the scheduler/optimizer registries already in the
code base, enabling users to extend the available metrics without modifying
core training code.  New metrics can be added simply by decorating a function
with ``@register_metric("my_metric_name")``.
"""


# ---------------------------------------------------------------------------
# Registry logic
# ---------------------------------------------------------------------------

# Global dictionary mapping metric names -> callable implementations
METRICS_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {}


def register_metric(name: str) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Decorator used to register a metric implementation.

    Parameters
    ----------
    name: str
        The name under which the metric will be registered.  This should be the
        string that users refer to in the YAML configuration file.

    """

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        # Register the function and return it unchanged so it can still be used
        # directly if desired.
        METRICS_REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Built-in metric implementations
# ---------------------------------------------------------------------------


@register_metric("accuracy")
def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute standard classification accuracy.

    Handles both seq2static and seq2seq classification:
    - seq2static: outputs [B, C], targets [B]
    - seq2seq: outputs [B, T, C], targets [B, T] (with -100 padding)

    Parameters
    ----------
    outputs : torch.Tensor
        Raw model outputs (logits or probabilities).
        - seq2static: shape ``(batch, num_classes)``
        - seq2seq: shape ``(batch, seq_len, num_classes)``
    targets : torch.Tensor
        Ground-truth class indices.
        - seq2static: shape ``(batch,)``
        - seq2seq: shape ``(batch, seq_len)`` with -100 for padding

    """
    if outputs.dim() == 3:
        # seq2seq: flatten predictions and targets, mask padding
        B, T, num_classes = outputs.shape
        preds = torch.argmax(outputs, dim=-1)  # [B, T]

        # Flatten for comparison
        preds_flat = preds.reshape(-1)  # [B*T]
        targets_flat = targets.reshape(-1)  # [B*T]

        # Mask out padding (-100)
        valid_mask = (targets_flat >= 0) & (targets_flat < num_classes)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=outputs.device, dtype=torch.float32)

        correct = (preds_flat == targets_flat).float() * valid_mask.float()
        return correct.sum() / valid_mask.float().sum()
    else:
        # seq2static: standard accuracy
        preds = torch.argmax(outputs, dim=1)
        return (preds == targets).float().mean()


@register_metric("precision")
def precision(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute precision for binary classification (class 1 is positive).

    If *outputs* contains more than two classes we treat class index 1 as the
    positive class which is the most common convention.  For true multiclass
    precision users should register their own specialised metric.
    """
    preds = torch.argmax(outputs, dim=1)
    true_pos = ((preds == 1) & (targets == 1)).float().sum()
    predicted_pos = (preds == 1).float().sum()
    return true_pos / (predicted_pos + 1e-8)


@register_metric("recall")
def recall(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute recall for binary classification (class 1 is positive)."""
    preds = torch.argmax(outputs, dim=1)
    true_pos = ((preds == 1) & (targets == 1)).float().sum()
    actual_pos = (targets == 1).float().sum()
    return true_pos / (actual_pos + 1e-8)


@register_metric("f1")
def f1_score(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the F1 score for binary classification."""
    prec = precision(outputs, targets)
    rec = recall(outputs, targets)
    return 2 * (prec * rec) / (prec + rec + 1e-8)


@register_metric("mse")
def mse_metric(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error between outputs and targets.

    Shape-smart alignment (mirrors loss.mse):
    - If outputs [B, T+1, D] and targets [B, T, D] -> drop first output timestep
    - If outputs [B, D] and targets [B, T, D] -> pool targets by mean over time
    - If outputs [B, T, D] and targets [B, D] -> pool outputs by mean over time
    """
    try:
        if outputs.dim() == 3 and targets.dim() == 3:
            # Align time when outputs include initial state
            if outputs.size(1) == targets.size(1) + 1:
                outputs = outputs[:, 1:, :]
        elif outputs.dim() == 2 and targets.dim() == 3:
            # Pooled model outputs vs sequence targets -> pool targets
            targets = targets.mean(dim=1)
        elif outputs.dim() == 3 and targets.dim() == 2:
            # Sequence outputs vs pooled targets -> pool outputs (drop initial if present)
            if outputs.size(1) > 1:
                outputs = outputs[:, 1:, :].mean(dim=1)
            else:
                outputs = outputs.mean(dim=1)
    except Exception:
        pass
    return F.mse_loss(outputs, targets)


@register_metric("top_k_accuracy")
def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, *, k: int = 5) -> torch.Tensor:
    """Compute the top-*k* accuracy.

    This metric is **not** meant to be referenced directly from the YAML config.
    Instead, the :pyclass:`~soen_toolkit.training.callbacks.metrics_tracker.MetricsTracker`
    will parse entries of the form "top_5", "top_10", etc., extract *k* and call
    this implementation internally.
    """
    # ``torch.topk`` returns the *k* largest elements for each sample.
    # Shape: (k, batch)
    _, pred = outputs.topk(k, dim=1)
    pred = pred.t()  # transpose for easier comparison
    # Expand targets to match the shape of ``pred``.
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    # ``any(dim=0)`` checks whether the target appears in any of the top-k
    # predictions for each sample.
    return correct.any(dim=0).float().mean()


# Alias for compatibility (some users may specify "topk" instead of "top_k").
METRICS_REGISTRY.setdefault("topk", top_k_accuracy)


@register_metric("autoregressive_accuracy")
def autoregressive_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute autoregressive next-token prediction accuracy.

    Parameters
    ----------
    outputs : torch.Tensor
        Autoregressive model outputs with shape ``(batch, seq_len, vocab_size)``.
    targets : torch.Tensor
        Target sequence with shape ``(batch, seq_len)`` - already properly shifted.

    """
    # Handle shape mismatch (SOEN outputs seq_len+1, we need seq_len)
    _batch_size, seq_len_out, _vocab_size = outputs.shape
    _batch_size_tgt, seq_len_tgt = targets.shape

    if seq_len_out != seq_len_tgt:
        if seq_len_out == seq_len_tgt + 1:
            # Common case: SOEN outputs seq_len+1 (includes initial state at timestep 0)
            # Skip the first timestep (initial state) and use timesteps 1 through seq_len
            outputs = outputs[:, 1:, :]  # Skip first timestep, take timesteps 1 through seq_len
        else:
            # Return 0 accuracy for other mismatches
            return torch.tensor(0.0, device=outputs.device)

    # Get predicted token indices
    pred_tokens = torch.argmax(outputs, dim=-1)  # [batch, seq_len]

    # Calculate accuracy: how often predicted token matches target token
    correct_predictions = (pred_tokens == targets).float()
    return correct_predictions.mean()


@register_metric("perplexity")
def perplexity(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute perplexity for both autoregressive and standard classification.

    Parameters
    ----------
    outputs : torch.Tensor
        Model outputs. Either:
        - Autoregressive: shape ``(batch, seq_len+1, vocab_size)``
        - Standard classification: shape ``(batch, vocab_size)``
    targets : torch.Tensor
        Targets. Either:
        - Autoregressive: sequence targets with shape ``(batch, seq_len)``
        - Standard classification: class indices with shape ``(batch,)``

    """
    if outputs.dim() == 3:
        # Autoregressive mode: targets are already properly constructed
        # outputs: [batch, seq_len, vocab_size]
        # targets: [batch, seq_len] - already shifted

        # Handle shape mismatch (SOEN outputs seq_len+1, we need seq_len)
        _batch_size, seq_len_out, _vocab_size = outputs.shape
        _batch_size_tgt, seq_len_tgt = targets.shape

        if seq_len_out != seq_len_tgt:
            if seq_len_out == seq_len_tgt + 1:
                # Common case: SOEN outputs seq_len+1 (includes initial state at timestep 0)
                # Skip the first timestep (initial state) and use timesteps 1 through seq_len
                outputs = outputs[:, 1:, :]  # Skip first timestep, take timesteps 1 through seq_len
            # For other mismatches, continue with original outputs (will likely fail gracefully)

        # Reshape for cross-entropy: [batch * seq_len, vocab_size] and [batch * seq_len]
        pred_logits_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
    else:
        # Standard classification mode
        pred_logits_flat = outputs  # [batch, vocab_size]
        targets_flat = targets  # [batch]

    # Compute cross-entropy loss
    # Use ignore_index=-100 to mask padding positions (PyTorch default)
    cross_entropy = F.cross_entropy(pred_logits_flat, targets_flat, reduction="mean", ignore_index=-100)

    # Perplexity = exp(cross_entropy)
    return torch.exp(cross_entropy)


@register_metric("bits_per_character")
def bits_per_character(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute bits per character for both autoregressive and standard classification.

    Parameters
    ----------
    outputs : torch.Tensor
        Model outputs. Either:
        - Autoregressive: shape ``(batch, seq_len+1, vocab_size)``
        - Standard classification: shape ``(batch, vocab_size)``
    targets : torch.Tensor
        Targets. Either:
        - Autoregressive: sequence targets with shape ``(batch, seq_len)``
        - Standard classification: class indices with shape ``(batch,)``

    """
    if outputs.dim() == 3:
        # Autoregressive mode: targets are already properly constructed
        # outputs: [batch, seq_len, vocab_size]
        # targets: [batch, seq_len] - already shifted

        # Handle shape mismatch (SOEN outputs seq_len+1, we need seq_len)
        _batch_size, seq_len_out, _vocab_size = outputs.shape
        _batch_size_tgt, seq_len_tgt = targets.shape

        if seq_len_out != seq_len_tgt:
            if seq_len_out == seq_len_tgt + 1:
                # Common case: SOEN outputs seq_len+1 (includes initial state at timestep 0)
                # Skip the first timestep (initial state) and use timesteps 1 through seq_len
                outputs = outputs[:, 1:, :]  # Skip first timestep, take timesteps 1 through seq_len
            # For other mismatches, continue with original outputs (will likely fail gracefully)

        # Reshape for cross-entropy: [batch * seq_len, vocab_size] and [batch * seq_len]
        pred_logits_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
    else:
        # Standard classification mode
        pred_logits_flat = outputs  # [batch, vocab_size]
        targets_flat = targets  # [batch]

    # Compute cross-entropy loss
    # Use ignore_index=-100 to mask padding positions (PyTorch default)
    cross_entropy = F.cross_entropy(pred_logits_flat, targets_flat, reduction="mean", ignore_index=-100)

    # Bits per character = cross_entropy / ln(2)
    return cross_entropy / torch.log(torch.tensor(2.0))
