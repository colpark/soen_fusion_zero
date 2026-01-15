from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

"""JAX-native metrics for the JAX backend.

This mirrors the Torch metrics in `soen_toolkit.training.callbacks.metrics`,
but uses JAX primitives so it can be computed inside jitted training/eval
steps without framework conversions.

All metric functions return a scalar jnp.ndarray (dtype float32/float64) and
accept model ``outputs`` and ``targets`` as jnp.ndarrays.
"""


# ---------------------------------------------------------------------------
# Registry logic
# ---------------------------------------------------------------------------

METRICS_REGISTRY: dict[str, Callable[..., jnp.ndarray]] = {}


def register_metric(name: str) -> Callable[[Callable[..., jnp.ndarray]], Callable[..., jnp.ndarray]]:
    def decorator(fn: Callable[..., jnp.ndarray]) -> Callable[..., jnp.ndarray]:
        METRICS_REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_class_indices(targets: jnp.ndarray) -> jnp.ndarray:
    """Convert targets to integer class indices when one-hot or shape > 1.

    - If targets are 3D [B, T, num_classes], apply argmax (one-hot encoding).
    - If targets are 2D [B, T] with T > 1, preserve shape (seq2seq class indices).
    - If targets are 2D [B, 1], squeeze to [B].
    - Otherwise return as int32.
    """
    if targets.ndim == 3:
        # 3D targets [B, T, num_classes] -> one-hot, apply argmax
        return jnp.argmax(targets, axis=-1).astype(jnp.int32)
    if targets.ndim == 2 and targets.shape[-1] > 1:
        # 2D targets [B, T] -> seq2seq class indices, preserve shape
        return targets.astype(jnp.int32)
    if targets.ndim >= 2 and targets.shape[-1] == 1:
        return jnp.squeeze(targets, axis=-1).astype(jnp.int32)
    return targets.astype(jnp.int32)


def _ce_mean_from_logits(logits: jnp.ndarray, targets_idx: jnp.ndarray) -> jnp.ndarray:
    """Mean cross-entropy for integer targets.

    Shapes:
      - logits [..., C]
      - targets_idx [...]
    """
    num_classes = logits.shape[-1]
    valid_mask = jnp.logical_and(targets_idx >= 0, targets_idx < num_classes)
    safe_targets = jnp.where(valid_mask, targets_idx, 0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1)[..., 0]
    gathered = gathered * valid_mask.astype(gathered.dtype)
    valid_total = jnp.sum(valid_mask.astype(gathered.dtype))
    return jnp.where(
        valid_total > 0,
        -jnp.sum(gathered) / valid_total,
        jnp.asarray(0.0, dtype=log_probs.dtype),
    )


# ---------------------------------------------------------------------------
# Built-in metrics (JAX)
# ---------------------------------------------------------------------------


@register_metric("accuracy")
def accuracy(outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Standard multiclass accuracy.

    Expects pooled logits ``[B, C]``. If a sequence ``[B, T, C]`` is provided,
    the metric is computed across all timesteps after flattening.
    """
    if outputs.ndim == 3:
        B, T, C = outputs.shape
        outputs = outputs.reshape((B * T, C))
        targets_idx = _as_class_indices(targets).reshape((B * T,))
    else:
        outputs = outputs.reshape((-1, outputs.shape[-1]))
        targets_idx = _as_class_indices(targets).reshape((-1,))
    preds = jnp.argmax(outputs, axis=-1).astype(jnp.int32)
    num_classes = outputs.shape[-1]
    valid_mask = jnp.logical_and(targets_idx >= 0, targets_idx < num_classes)
    correct = jnp.logical_and(valid_mask, preds == targets_idx)
    valid_count = jnp.sum(valid_mask.astype(jnp.float32))
    return jnp.where(
        valid_count > 0,
        jnp.sum(correct.astype(jnp.float32)) / valid_count,
        jnp.asarray(0.0, dtype=jnp.float32),
    )


@register_metric("precision")
def precision(outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Binary precision (class 1 is positive)."""
    if outputs.ndim == 3:
        outputs = outputs[:, -1, :]
    preds = jnp.argmax(outputs, axis=-1)
    targets_idx = _as_class_indices(targets)
    true_pos = jnp.sum(((preds == 1) & (targets_idx == 1)).astype(jnp.float32))
    predicted_pos = jnp.sum((preds == 1).astype(jnp.float32))
    return true_pos / (predicted_pos + jnp.asarray(1e-8, dtype=jnp.float32))


@register_metric("recall")
def recall(outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Binary recall (class 1 is positive)."""
    if outputs.ndim == 3:
        outputs = outputs[:, -1, :]
    preds = jnp.argmax(outputs, axis=-1)
    targets_idx = _as_class_indices(targets)
    true_pos = jnp.sum(((preds == 1) & (targets_idx == 1)).astype(jnp.float32))
    actual_pos = jnp.sum((targets_idx == 1).astype(jnp.float32))
    return true_pos / (actual_pos + jnp.asarray(1e-8, dtype=jnp.float32))


@register_metric("f1")
def f1_score(outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Binary F1 using precision/recall with class 1 as positive."""
    p = precision(outputs, targets)
    r = recall(outputs, targets)
    return 2.0 * (p * r) / (p + r + jnp.asarray(1e-8, dtype=jnp.float32))


@register_metric("mse")
def mse_metric(outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error with shape-smart alignment.

    Rules (mirrors Torch implementation):
      - If outputs [B, T+1, D] and targets [B, T, D] -> drop first output timestep
      - If outputs [B, D] and targets [B, T, D] -> pool targets by mean over time
      - If outputs [B, T, D] and targets [B, D] -> pool outputs by mean over time
    """
    x = outputs
    y = targets
    if x.ndim == 3 and y.ndim == 3:
        if x.shape[1] == y.shape[1] + 1:
            x = x[:, 1:, :]
    elif x.ndim == 2 and y.ndim == 3:
        y = jnp.mean(y, axis=1)
    elif x.ndim == 3 and y.ndim == 2:
        if x.shape[1] > 1:
            x = jnp.mean(x[:, 1:, :], axis=1)
        else:
            x = jnp.mean(x, axis=1)
    return jnp.mean(jnp.square(x - y))


@register_metric("top_k_accuracy")
def top_k_accuracy(outputs: jnp.ndarray, targets: jnp.ndarray, *, k: int = 5) -> jnp.ndarray:
    """Top-k accuracy.

    For each sample, checks whether the target index appears in the top-k
    predicted classes.
    """
    if outputs.ndim == 3:
        outputs = outputs[:, -1, :]
    targets_idx = _as_class_indices(targets)
    # Get indices of top-k classes per sample
    topk_idx = jnp.argsort(outputs, axis=-1)[:, -k:]
    # Compare against targets
    match = topk_idx == targets_idx[:, None]
    return jnp.mean(jnp.any(match, axis=1).astype(jnp.float32))


# Alias for compatibility (users may specify "topk").
METRICS_REGISTRY.setdefault("topk", top_k_accuracy)


@register_metric("autoregressive_accuracy")
def autoregressive_accuracy(outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Next-token prediction accuracy for autoregressive outputs.

    outputs: [B, S_out, V]; targets: [B, S]
    If S_out == S+1, drop the first output timestep.
    """
    b, s_out, _ = outputs.shape
    b_t, s_tgt = targets.shape
    # Assume batch sizes align
    if s_out != s_tgt:
        if s_out == s_tgt + 1:
            outputs = outputs[:, 1:, :]
        else:
            raise ValueError(f"AR accuracy shape mismatch: outputs.shape={outputs.shape} targets.shape={targets.shape}")
    pred_tokens = jnp.argmax(outputs, axis=-1)
    targets_idx = _as_class_indices(targets)
    return jnp.mean((pred_tokens == targets_idx).astype(jnp.float32))


@register_metric("perplexity")
def perplexity(outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Perplexity for autoregressive or standard classification.

    - Autoregressive: outputs [B, S_out, V], targets [B, S]
      If S_out == S+1, drop the first output timestep.
    - Standard: outputs [B, V], targets [B]
    """
    if outputs.ndim == 3:
        b, s_out, _ = outputs.shape
        b_t, s_tgt = targets.shape
        if s_out != s_tgt:
            if s_out == s_tgt + 1:
                outputs = outputs[:, 1:, :]
            else:
                raise ValueError(f"Perplexity shape mismatch: outputs.shape={outputs.shape} targets.shape={targets.shape}")
        # Flatten time
        logits_flat = outputs.reshape((-1, outputs.shape[-1]))
        targets_idx = _as_class_indices(targets).reshape((-1,))
        ce = _ce_mean_from_logits(logits_flat, targets_idx)
    else:
        targets_idx = _as_class_indices(targets)
        ce = _ce_mean_from_logits(outputs, targets_idx)
    return jnp.exp(ce)


@register_metric("bits_per_character")
def bits_per_character(outputs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Bits per character for AR or standard classification.

    bpc = CE / ln(2)
    """
    ln2 = jnp.log(jnp.asarray(2.0))
    if outputs.ndim == 3:
        b, s_out, _ = outputs.shape
        b_t, s_tgt = targets.shape
        if s_out != s_tgt:
            if s_out == s_tgt + 1:
                outputs = outputs[:, 1:, :]
            else:
                raise ValueError(f"BPC shape mismatch: outputs.shape={outputs.shape} targets.shape={targets.shape}")
        logits_flat = outputs.reshape((-1, outputs.shape[-1]))
        targets_idx = _as_class_indices(targets).reshape((-1,))
        ce = _ce_mean_from_logits(logits_flat, targets_idx)
    else:
        targets_idx = _as_class_indices(targets)
        ce = _ce_mean_from_logits(outputs, targets_idx)
    return ce / ln2


__all__ = ["METRICS_REGISTRY", "register_metric"]
