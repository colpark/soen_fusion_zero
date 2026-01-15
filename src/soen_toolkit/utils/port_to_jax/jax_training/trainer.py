from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import dataclass, field
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.profiler
import optax

from soen_toolkit.core.soen_model_core import SOENModelCore
from soen_toolkit.utils.port_to_jax.convert import convert_core_model_to_jax
from soen_toolkit.utils.port_to_jax.eqx_model import SoenEqxModel
from soen_toolkit.utils.port_to_jax.pure_forward import (
    build_topology,
    convert_params_to_arrays,
)

from .ar_utils import build_ar_targets_jax, pool_token_timesteps_jax
from .callbacks.metrics import METRICS_REGISTRY
from .losses import LOSS_REGISTRY
from .pooling import apply_time_pooling

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from soen_toolkit.utils.port_to_jax.jax_model import JAXModel


def _build_gradient_clip_transform(clip_val: float | None, algorithm: str = "norm") -> optax.GradientTransformation | None:
    """Build an optax gradient clipping transform.

    Args:
        clip_val: Clipping threshold value. If None, returns None (no clipping).
        algorithm: "norm" for global norm clipping, "value" for element-wise clipping.

    Returns:
        An optax transform or None if clip_val is None.
    """
    if clip_val is None:
        return None

    if algorithm == "value":
        # Element-wise clipping: clip each element to [-clip_val, clip_val]
        return optax.clip(clip_val)
    else:
        # "norm" (default): global norm clipping
        return optax.clip_by_global_norm(clip_val)


def _build_optimizer(
    optimizer_name: str,
    lr_or_schedule: float | Callable[[jnp.ndarray], jnp.ndarray],
    optimizer_kwargs: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> optax.GradientTransformation:
    """Build an optax optimizer based on name.

    Args:
        optimizer_name: Name of the optimizer (adamw, adam, sgd, rmsprop, lion, muon).
        lr_or_schedule: Learning rate (float) or learning rate schedule (callable).
        optimizer_kwargs: Additional optimizer-specific keyword arguments.
        params: Optional parameter tree (required for muon to determine which params are 2D+).

    Returns:
        An optax GradientTransformation.

    Raises:
        ValueError: If optimizer_name is not supported.
        ModuleNotFoundError: If required optimizer package is not installed.
    """
    optimizer_kwargs = optimizer_kwargs or {}
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        weight_decay = optimizer_kwargs.pop("weight_decay", 0.001)
        b1 = optimizer_kwargs.pop("b1", 0.9)
        b2 = optimizer_kwargs.pop("b2", 0.999)
        eps = optimizer_kwargs.pop("eps", 1e-8)
        if optimizer_kwargs:
            logger.warning(f"Unused optimizer kwargs for adamw: {optimizer_kwargs}")
        return optax.adamw(learning_rate=lr_or_schedule, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay)

    elif optimizer_name == "adam":
        b1 = optimizer_kwargs.pop("b1", 0.9)
        b2 = optimizer_kwargs.pop("b2", 0.999)
        eps = optimizer_kwargs.pop("eps", 1e-8)
        if optimizer_kwargs:
            logger.warning(f"Unused optimizer kwargs for adam: {optimizer_kwargs}")
        return optax.adam(learning_rate=lr_or_schedule, b1=b1, b2=b2, eps=eps)

    elif optimizer_name == "sgd":
        momentum = optimizer_kwargs.pop("momentum", 0.0)
        nesterov = optimizer_kwargs.pop("nesterov", False)
        weight_decay = optimizer_kwargs.pop("weight_decay", 0.0)
        if optimizer_kwargs:
            logger.warning(f"Unused optimizer kwargs for sgd: {optimizer_kwargs}")
        return optax.sgd(learning_rate=lr_or_schedule, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)

    elif optimizer_name == "rmsprop":
        decay = optimizer_kwargs.pop("decay", 0.9)
        eps = optimizer_kwargs.pop("eps", 1e-8)
        initial_scale = optimizer_kwargs.pop("initial_scale", 0.0)
        momentum = optimizer_kwargs.pop("momentum", 0.0)
        nesterov = optimizer_kwargs.pop("nesterov", False)
        if optimizer_kwargs:
            logger.warning(f"Unused optimizer kwargs for rmsprop: {optimizer_kwargs}")
        return optax.rmsprop(
            learning_rate=lr_or_schedule,
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            momentum=momentum,
            nesterov=nesterov,
        )

    elif optimizer_name == "lion":
        b1 = optimizer_kwargs.pop("b1", optimizer_kwargs.pop("beta1", 0.9))
        b2 = optimizer_kwargs.pop("b2", optimizer_kwargs.pop("beta2", 0.99))
        weight_decay = optimizer_kwargs.pop("weight_decay", 0.0)
        if optimizer_kwargs:
            logger.warning(f"Unused optimizer kwargs for lion: {optimizer_kwargs}")
        return optax.lion(learning_rate=lr_or_schedule, b1=b1, b2=b2, weight_decay=weight_decay)

    elif optimizer_name == "muon":
        # Muon requires 2D+ parameters. For 1D parameters, we use AdamW as fallback.
        # This matches PyTorch's MuonWithAuxAdam behavior.
        if params is None:
            msg = "Muon optimizer requires params to be passed to _build_optimizer for parameter masking"
            raise ValueError(msg)

        # Extract ns_coeffs - can be tuple/list or None (will use default)
        ns_coeffs_raw = optimizer_kwargs.pop("ns_coeffs", None)
        if ns_coeffs_raw is None:
            ns_coeffs = (3.4445, -4.775, 2.0315)  # Default from empirical work
        elif isinstance(ns_coeffs_raw, (tuple, list)):
            ns_coeffs = tuple(ns_coeffs_raw)
        else:
            raise ValueError(f"muon ns_coeffs must be a tuple or list, got {type(ns_coeffs_raw)}")

        ns_steps = int(optimizer_kwargs.pop("ns_steps", 5))

        # Extract weight_decay for AdamW fallback (muon doesn't support weight_decay directly)
        weight_decay = optimizer_kwargs.pop("weight_decay", 0.001)

        if optimizer_kwargs:
            logger.warning(f"Unused optimizer kwargs for muon: {optimizer_kwargs}")

        try:
            muon_optimizer = optax.contrib.muon(
                learning_rate=lr_or_schedule,
                ns_coeffs=ns_coeffs,
                ns_steps=ns_steps,
            )
        except AttributeError:
            msg = "Muon optimizer requested but optax.contrib.muon is not available. Please upgrade optax to the latest version: pip install --upgrade optax"
            raise ModuleNotFoundError(msg) from None

        # Create predicate function to identify 2D+ parameters (ndim >= 2)
        def is_2d_plus(param: jnp.ndarray) -> bool:
            """Check if parameter has at least 2 dimensions and both dimensions are > 0."""
            if param.ndim < 2:
                return False
            # Also check that dimensions are non-zero to avoid division by zero
            if param.shape[-2] == 0 or param.shape[-1] == 0:
                return False
            return True

        # Create mask tree: True for 2D+ params (use Muon), False for 1D params (use AdamW)
        muon_mask_bool = jax.tree_util.tree_map(is_2d_plus, params)

        # Convert boolean mask to string labels for multi_transform
        def bool_to_label(is_muon: bool) -> str:
            return "muon" if is_muon else "adamw"

        muon_mask = jax.tree_util.tree_map(bool_to_label, muon_mask_bool)

        # Create AdamW optimizer for 1D parameters
        aux_optimizer = optax.adamw(
            learning_rate=lr_or_schedule,
            weight_decay=weight_decay,
        )

        # Use multi_transform to apply Muon to 2D+ params and AdamW to 1D params
        return optax.multi_transform(
            transforms={"muon": muon_optimizer, "adamw": aux_optimizer},
            param_labels=muon_mask,
        )

    else:
        msg = f"Unsupported optimizer: {optimizer_name}. Supported: adamw, adam, sgd, rmsprop, lion, muon"
        raise ValueError(msg)


@dataclass
class TrainingConfigJAX:
    lr: float = 1e-3
    max_epochs: int = 5
    log_every_n_steps: int = 50
    seed: int = 0
    platform: str = "cpu"  # cpu|cuda|metal
    classification: bool = True
    mapping: str = "seq2static"
    paradigm: str = "supervised"  # supervised|distillation|unsupervised|self_supervised
    profile: bool = False
    profile_samples: int = 0
    trace_dir: str | None = None  # if set, write XLA trace to this directory
    # Time pooling controls (used for seq2static classification and parity with Torch path)
    time_pooling_method: str = "max"
    time_pooling_params: dict[str, Any] = field(default_factory=dict)
    time_steps_per_token: int = 1  # DEPRECATED: Use ar.time_steps_per_token instead
    time_pooling_range_start: int | None = None
    time_pooling_range_end: int | None = None

    # Autoregressive settings
    autoregressive: bool = False
    ar_time_steps_per_token: int = 1
    ar_token_pooling_method: str = "final"
    ar_token_pooling_params: dict[str, Any] = field(default_factory=dict)

    # Multiple losses configuration removed - passed separately to avoid JAX tracing issues


@dataclass
class DataConfigJAX:
    train_loader: Any
    val_loader: Any


@dataclass
class ExperimentConfigJAX:
    model_path: str
    data: DataConfigJAX
    training: TrainingConfigJAX
    model_dt: float | None = None  # Optional dt override from config
    model_dt_learnable: bool | None = None  # Optional dt_learnable override from config


class JaxTrainer:
    def __init__(
        self,
        cfg: ExperimentConfigJAX,
        losses: list[dict[str, Any]] | None = None,
        lr_schedule: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        metric_names: list[str] | None = None,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
        accumulate_grad_batches: int = 1,
        optimizer_name: str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._log = logging.getLogger(__name__)
        self.cfg = cfg
        self.losses = losses
        self.lr_schedule = lr_schedule
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self._optimizer_name = optimizer_name.lower()  # Store optimizer name for logging
        os.environ.setdefault("JAX_PLATFORMS", cfg.training.platform)
        jax.config.update("jax_platforms", os.environ["JAX_PLATFORMS"])

        # Build Torch core then convert
        self.core: SOENModelCore = SOENModelCore.load(str(self.cfg.model_path))
        self.core.eval()

        # Apply dt override from config if provided (matches Lightning wrapper behavior)
        if self.cfg.model_dt is not None:
            try:
                self.core.set_dt(self.cfg.model_dt, propagate_to_layers=True)
            except Exception as e:
                raise RuntimeError(f"[JAX] Failed to override dt on SOEN model: {e}") from e
            self._log.info(f"[JAX] Overrode SOEN model dt to {self.cfg.model_dt} via config")

        # Set dt learnability if specified
        if self.cfg.model_dt_learnable is not None:
            try:
                self.core.set_dt_learnable(self.cfg.model_dt_learnable, propagate_to_layers=True)
            except Exception as e:
                raise RuntimeError(f"[JAX] Failed to set dt learnability: {e}") from e

        self.jax_model: JAXModel = convert_core_model_to_jax(self.core)
        # Build and cache layer implementations outside of jit to avoid tracer leaks
        try:
            self.jax_model._ensure_cache()
        except Exception as e:
            raise RuntimeError(f"[JAX] Failed to cache layer implementations: {e}") from e

        # Build topology arrays BEFORE JIT compilation (must be concrete)
        # Topology arrays are critical for performance
        try:
            from soen_toolkit.utils.port_to_jax.topology_arrays import build_topology_arrays
            self.jax_model._topology_arrays = build_topology_arrays(self.jax_model)

            self._log.debug("[JAX] Topology arrays prepared successfully")
        except Exception as e:
            # Fail-fast: topology arrays are required for production training
            self._log.error(f"[JAX] Failed to prepare topology arrays: {e}. Training will be significantly slower without topology arrays. This is likely a configuration or model conversion error.")
            raise RuntimeError(f"Topology arrays are required for JAX training. This error indicates a problem with model conversion or topology construction. Original error: {e}") from e

        # Build initial parameter tree from converted JAXModel specs
        params_tree = self._build_param_tree(self.jax_model)

        # Build static topology for pure functional forward
        self.topology = build_topology(self.jax_model)
        self.layer_params, self.connection_params = convert_params_to_arrays(
            params_tree,
            self.topology,
            connection_constraints=self.topology.connection_constraints,
        )

        # Extract internal connections from param tree (kept as dict for now)
        initial_internal_conns = params_tree.get("internal_connections", {})

        # Stage-2 Equinox: trainable params live as module leaves
        self.eqx_model = SoenEqxModel(
            topology=self.jax_model,
            layer_params=tuple(self.layer_params),  # optax-friendly container
            connection_params=self.connection_params,
            internal_connections=initial_internal_conns,
        )
        # `self.params` is the optimizer parameter pytree (the eqx.Module)
        self.params = self.eqx_model

        # Store original layer parameter arrays for non-learnable parameter restoration
        # This must happen AFTER convert_params_to_arrays since that's when they're in final array form
        # Note: GRU/LSTM layers store params as tuples of arrays, not single arrays
        def _copy_layer_param(param):
            if isinstance(param, tuple):
                return tuple(jnp.copy(arr) for arr in param)
            return jnp.copy(param)
        self._original_layer_params = tuple(_copy_layer_param(p) for p in self.layer_params)

        self._log.info(f"[JAX][debug] dt={self.topology.dt:.6e}")

        # Initialize RNG key for noise application
        # Use seed from training config if available, otherwise use default
        noise_seed = getattr(cfg.training, "noise_seed", 42)
        self.rng_key = jax.random.PRNGKey(noise_seed)
        self._batch_counter = 0  # Track batch number for deterministic key splitting

        # Precompute edge masks (pad-aligned to connection_params) for projection
        if getattr(self.topology, "edge_masks", None) is not None:
            self.edge_masks = jnp.asarray(self.topology.edge_masks, dtype=self.connection_params.dtype)
        else:
            self.edge_masks = jnp.ones_like(self.connection_params)

        # Precompute learnability masks for external connections
        edge_learnable = getattr(self.topology, "edge_learnable", ())
        if edge_learnable and len(edge_learnable) > 0:
            # Convert tuple of bools to JAX array matching connection_params shape [E, max_dst_dim, max_src_dim]
            E, max_dst, max_src = self.connection_params.shape
            learnable_array = jnp.array([float(is_learnable) for is_learnable in edge_learnable], dtype=self.connection_params.dtype)
            # Broadcast to match connection_params shape: [E, 1, 1] so it broadcasts correctly
            self.edge_learnable_mask = learnable_array[:, None, None]
        else:
            # Default to all learnable if not specified
            self.edge_learnable_mask = jnp.ones_like(self.connection_params)

        # Store per-element constraint matrices for polarity enforcement
        self.edge_constraint_mins = getattr(self.topology, "edge_constraint_mins", None)
        self.edge_constraint_maxs = getattr(self.topology, "edge_constraint_maxs", None)

        # Extract internal connection learnability
        internal_learnable_dict = getattr(self.topology, "internal_learnable", None)
        if internal_learnable_dict:
            self.internal_learnable = dict(internal_learnable_dict)
        else:
            # Build from jax_model if not in topology (backward compatibility)
            self.internal_learnable = {}
            for spec in self.jax_model.layers:
                if spec.internal_J is not None:
                    self.internal_learnable[int(spec.layer_id)] = bool(getattr(spec, "internal_J_learnable", True))

        # Store original internal connection values for restoration (needed in JIT functions)
        self._original_internal_conns: dict[int, jnp.ndarray] = {}
        # Cache internal masks for gradient masking and restoration
        self._internal_masks: dict[int, jnp.ndarray] = {}
        # Store internal constraint matrices for polarity enforcement
        self._internal_constraint_mins: dict[int, jnp.ndarray] = {}
        self._internal_constraint_maxs: dict[int, jnp.ndarray] = {}

        internal_mins_list = getattr(self.topology, "internal_constraint_mins", None)
        internal_maxs_list = getattr(self.topology, "internal_constraint_maxs", None)

        for idx, spec in enumerate(self.jax_model.layers):
            if spec.internal_J is not None:
                layer_id = int(spec.layer_id)
                self._original_internal_conns[layer_id] = jnp.asarray(spec.internal_J)
                # Store mask if present, otherwise create ones mask matching J shape
                if getattr(spec, "internal_mask", None) is not None:
                    self._internal_masks[layer_id] = jnp.asarray(spec.internal_mask)
                else:
                    # Create ones mask matching J shape (no masking)
                    self._internal_masks[layer_id] = jnp.ones_like(spec.internal_J)

                # Store constraint matrices if present
                if internal_mins_list and idx < len(internal_mins_list) and internal_mins_list[idx] is not None:
                    self._internal_constraint_mins[layer_id] = internal_mins_list[idx]
                if internal_maxs_list and idx < len(internal_maxs_list) and internal_maxs_list[idx] is not None:
                    self._internal_constraint_maxs[layer_id] = internal_maxs_list[idx]

        # Build optimizer chain with optional gradient clipping
        optimizer_chain = []

        # Add gradient clipping if configured
        if gradient_clip_val is not None and gradient_clip_val > 0:
            if gradient_clip_algorithm == "norm":
                optimizer_chain.append(optax.clip_by_global_norm(gradient_clip_val))
                self._log.info(f"Gradient clipping enabled: clip_by_global_norm={gradient_clip_val}")
            elif gradient_clip_algorithm == "value":
                optimizer_chain.append(optax.clip(gradient_clip_val))
                self._log.info(f"Gradient clipping enabled: clip_by_value={gradient_clip_val}")
            else:
                self._log.warning(f"Unknown gradient_clip_algorithm '{gradient_clip_algorithm}', using 'norm'")
                optimizer_chain.append(optax.clip_by_global_norm(gradient_clip_val))
        # Add main optimizer
        lr_or_schedule = self.lr_schedule if self.lr_schedule is not None else self.cfg.training.lr
        optimizer_kwargs = optimizer_kwargs or {}
        # Pass params for muon optimizer to enable parameter masking
        optimizer = _build_optimizer(optimizer_name, lr_or_schedule, optimizer_kwargs, params=self.params)
        optimizer_chain.append(optimizer)
        self._log.info(f"[JAX] Optimizer: {optimizer_name} (lr={'schedule' if self.lr_schedule is not None else f'{self.cfg.training.lr}'})")

        # Chain optimizers together
        self.tx = optax.chain(*optimizer_chain) if len(optimizer_chain) > 1 else optimizer_chain[0]
        self.opt_state = self.tx.init(self.params)

        # Gradient accumulation state (if accumulate_grad_batches > 1)
        if self.accumulate_grad_batches > 1:
            self._log.info(f"Gradient accumulation enabled: accumulate_grad_batches={self.accumulate_grad_batches}")
            # Initialize accumulated gradients to zero
            self.accumulated_grads = jax.tree_util.tree_map(jnp.zeros_like, self.params)
            self.accum_steps = 0

        # JIT compile step (topology as traced arg to avoid hashability issues)
        self.train_step = jax.jit(self._train_step)
        self.eval_step = jax.jit(self._eval_step)
        with contextlib.suppress(Exception):
            self._log.debug("JAX backend: %s %s", jax.default_backend(), jax.devices())

        # QAT (Quantization-Aware Training) state
        self._qat_active: bool = False
        self._qat_codebook: jnp.ndarray | None = None
        # Metric names to compute inside jitted steps
        self.metric_names: list[str] = list(metric_names or [])
        # Fail-fast: validate configured metrics and losses now (before JIT tracing).
        for name in self.metric_names:
            if name in {"loss", "input_seq_len"}:
                continue
            if name not in METRICS_REGISTRY:
                raise ValueError(
                    f"Unknown JAX metric '{name}'. Available: {sorted(METRICS_REGISTRY.keys())}"
                )
        if self.losses and len(self.losses) > 0:
            for spec in self.losses:
                loss_name = spec.get("loss_function")
                if not loss_name:
                    raise ValueError(f"Loss spec missing 'loss_function': {spec}")
                if loss_name not in LOSS_REGISTRY:
                    raise ValueError(
                        f"Unknown JAX loss '{loss_name}'. Available: {sorted(LOSS_REGISTRY.keys())}"
                    )

    # ------------------------ Noise RNG management ------------------------
    def get_batch_rng_key(self) -> jax.Array | None:
        """Get a fresh RNG key for the current batch.

        Returns None if noise is not configured, otherwise returns a key
        derived from the base key and batch counter for deterministic noise.
        """
        # Check if noise is configured
        if not hasattr(self.jax_model, "connection_noise_settings") or not self.jax_model.connection_noise_settings:
            return None

        # Split key deterministically based on batch counter
        batch_key = jax.random.fold_in(self.rng_key, self._batch_counter)
        self._batch_counter += 1
        return batch_key

    # ------------------------ QAT controls ------------------------
    def enable_qat_ste_jax(self, *, min_val: float, max_val: float, num_levels: int) -> None:
        """Enable STE snapping of connection parameters to a uniform codebook during loss computation.

        Forward uses snapped values; gradients pass through unmodified (identity STE).
        """
        min_v = float(min_val)
        max_v = float(max_val)
        nl = max(1, int(num_levels))
        # Even if zero isn't exactly included, a fine-grained grid approximates well
        self._qat_codebook = jnp.linspace(min_v, max_v, nl, dtype=self.connection_params.dtype)
        self._qat_active = True

    def disable_qat_ste_jax(self) -> None:
        self._qat_active = False
        self._qat_codebook = None

    def _ste_snap(self, x: jnp.ndarray, codebook: jnp.ndarray) -> jnp.ndarray:
        """Straight-through estimator snapping to nearest codebook value.

        y = x + stop_gradient(snapped - x) so forward uses snapped, backward is identity.
        """
        # Compute nearest codebook entry per element
        cb = codebook
        # x[..., None] - cb[None, ...] -> distances over last new axis
        idx = jnp.argmin(jnp.abs(x[..., None] - cb[None, ...]), axis=-1)
        snapped = cb[idx]
        return x + jax.lax.stop_gradient(snapped - x)

    def _maybe_apply_qat(self, model: SoenEqxModel) -> SoenEqxModel:
        if not self._qat_active or self._qat_codebook is None:
            return model
        cp_q = self._ste_snap(model.connection_params, self._qat_codebook)
        internal_q: dict[int, jnp.ndarray] = {}
        for layer_id, internal_J in (model.internal_connections or {}).items():
            internal_q[int(layer_id)] = self._ste_snap(jnp.asarray(internal_J), self._qat_codebook)
        return SoenEqxModel(
            topology=model.topology,
            layer_params=model.layer_params,
            connection_params=cp_q,
            internal_connections=internal_q if internal_q else (model.internal_connections or {}),
        )

    def _apply_layer_param_constraints(self, model: SoenEqxModel) -> SoenEqxModel:
        layer_params = model.layer_params
        if not layer_params:
            return model
        clamped = self.jax_model.clamp_and_apply_layer_param_arrays(layer_params, apply_to_specs=False)
        if clamped is layer_params:
            return model
        return SoenEqxModel(
            topology=model.topology,
            layer_params=clamped,
            connection_params=model.connection_params,
            internal_connections=model.internal_connections,
        )

    def eval_lr(self, step: int | float) -> float:
        """Evaluate the current learning rate at a given global step."""
        if self.lr_schedule is None:
            return float(self.cfg.training.lr)
        return float(self.lr_schedule(jnp.asarray(step, dtype=jnp.float32)))

    # ------------------------ Param tree ------------------------
    def _build_param_tree(self, jax_model: JAXModel):
        from soen_toolkit.core.layers.common.metadata import LAYER_PARAM_CONFIGS
        from soen_toolkit.utils.port_to_jax.parameter_specs import get_param_spec, validate_layer_params

        layers_sorted = sorted(jax_model.layers, key=lambda layer_item: layer_item.layer_id)
        layers: dict[int, dict[str, jnp.ndarray]] = {}

        for spec in layers_sorted:
            # Get parameter specification for this layer type
            param_spec = get_param_spec(spec.kind)

            # Validate that spec.params contains all required parameters
            validate_layer_params(spec.kind, spec.params)

            # Extract ALL parameters (learnable and non-learnable)
            # We need all params for forward pass and convert_params_to_arrays
            # Gradients will be masked later for non-learnable params
            param_dict = {}

            for param_name in param_spec.required_params:
                if param_name not in spec.params:
                    msg = f"Layer {spec.layer_id} ({spec.kind}) missing required parameter '{param_name}'. This should have been filled by convert_core_model_to_jax()."
                    raise ValueError(msg)
                param_dict[param_name] = spec.params[param_name]

            # Include optional parameters if present
            for param_name in param_spec.optional_params:
                if param_name in spec.params and spec.params[param_name] is not None:
                    param_dict[param_name] = spec.params[param_name]

            # CRITICAL FIX: Convert log-space parameters to log domain for optimizer
            # PyTorch stores gamma_plus/gamma_minus as log_gamma_plus/log_gamma_minus internally
            # and the optimizer updates them in log-space, guaranteeing positivity via exp()
            # We replicate this behavior in JAX to prevent gamma params from going negative
            layer_kind = self.jax_model._canonical_layer_kind(spec.kind)
            param_configs = LAYER_PARAM_CONFIGS.get(layer_kind, [])
            for cfg in param_configs:
                if cfg.is_log_param and cfg.name in param_dict:
                    # Convert from real-space to log-space for optimizer
                    param_dict[cfg.name] = jnp.log(jnp.maximum(param_dict[cfg.name], 1e-10))

            layers[spec.layer_id] = param_dict

        conns: dict[tuple[int, int], jnp.ndarray] = {(c.from_layer, c.to_layer): c.J for c in self.jax_model.connections}

        # Extract internal connections (layer_id -> internal_J)
        internal_conns: dict[int, jnp.ndarray] = {}
        for spec in layers_sorted:
            if getattr(spec, "internal_J", None) is not None:
                internal_conns[spec.layer_id] = spec.internal_J

        return {"layers": layers, "connections": conns, "internal_connections": internal_conns}

    # ------------------------ Forward wrapper ------------------------
    def _forward(
        self,
        model: SoenEqxModel,
        x: jnp.ndarray,
        initial_states: dict[int, jnp.ndarray] | None = None,
        s1_inits: dict[int, jnp.ndarray] | None = None,
        s2_inits: dict[int, jnp.ndarray] | None = None,
        rng_key: jax.Array | None = None,
    ):
        """Forward pass for training/eval (pure, traced, eqx.Module params)."""
        with jax.profiler.TraceAnnotation("forward_pass"):
            return model(
                x,
                initial_states=initial_states,
                s1_inits=s1_inits,
                s2_inits=s2_inits,
                rng_key=rng_key,
                return_trace=True,
                track_phi=False,
                track_g=False,
            )

    # ------------------------ Loss & metrics ------------------------
    def _compute_batch_loss(
        self,
        model: SoenEqxModel,
        x_b: jnp.ndarray,
        y_b: jnp.ndarray,
        initial_states: dict[int, jnp.ndarray] | None = None,
        s1_inits: dict[int, jnp.ndarray] | None = None,
        s2_inits: dict[int, jnp.ndarray] | None = None,
        rng_key: jax.Array | None = None,
    ):
        # Apply QAT STE snapping on connection parameters if enabled
        model_q = self._maybe_apply_qat(model)
        final_hist, histories, trace = self._forward(model_q, x_b, initial_states, s1_inits, s2_inits, rng_key=rng_key)
        params_q = model_q.as_params_dict()
        histories_tuple = tuple(histories) if isinstance(histories, list) else histories

        def _finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
            # Include histories for stateful training (stored as tuple for JIT compatibility)
            metrics["_histories"] = histories_tuple
            # Include auxiliary final states for MultiplierNOCC (aligned to layer order)
            metrics["_s1_final_by_layer"] = trace.s1_final_by_layer
            metrics["_s2_final_by_layer"] = trace.s2_final_by_layer
            return metrics
        # final_hist: [B, T+1, D]; drop initial state to get sequence outputs
        # For distillation, keep t=0 to match teacher trajectory
        paradigm = getattr(self.cfg.training, "paradigm", "supervised")
        if paradigm == "distillation":
            y_logits_seq = final_hist  # Include t=0 for distillation
        else:
            y_logits_seq = final_hist[:, 1:, :]
        mapping = getattr(self.cfg.training, "mapping", "seq2static")

        # Helper: MSE with SOEN alignment rules
        def _mse_aligned(seq_logits: jnp.ndarray, y_target: jnp.ndarray) -> jnp.ndarray:
            target = jnp.asarray(y_target)
            if mapping == "seq2seq" and target.ndim >= 2:
                preds = seq_logits
                if target.ndim == 2:
                    target = jnp.broadcast_to(target[:, None, :], preds.shape)
                if target.ndim == 3 and target.shape[1] != preds.shape[1]:
                    min_T = min(preds.shape[1], target.shape[1])
                    preds = preds[:, :min_T, :]
                    target = target[:, :min_T, :]
            else:
                if target.ndim >= 2 and target.shape[0] == seq_logits.shape[0] and target.shape[1] == seq_logits.shape[1]:
                    target = target[:, -1, :]
                preds = seq_logits[:, -1, :]
                if target.ndim == 1 and preds.ndim == 2:
                    target = jnp.broadcast_to(target[:, None], preds.shape)
            return LOSS_REGISTRY["mse"](preds, target)

        # Helper: CE with SOEN pooling rules
        def _cross_entropy(seq_logits: jnp.ndarray, y_target: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            if mapping == "seq2seq":
                logits = seq_logits
                targets = jnp.asarray(y_target)
                if targets.ndim == 3:
                    if targets.shape[-1] == logits.shape[-1]:
                        targets = jnp.argmax(targets, axis=-1)
                    elif targets.shape[-1] == 1:
                        targets = jnp.squeeze(targets, axis=-1)
                if targets.ndim == 1:
                    targets = jnp.broadcast_to(targets[:, None], logits.shape[:-1])
                targets = targets.astype(jnp.int32)
                num_classes = logits.shape[-1]
                valid_mask = jnp.logical_and(targets >= 0, targets < num_classes)
                ignore_val = jnp.asarray(-100, dtype=jnp.int32)
                targets_masked = jnp.where(valid_mask, targets, ignore_val)
                logits_flat = logits.reshape(-1, num_classes)
                targets_flat = targets_masked.reshape(-1)
                loss_ce = LOSS_REGISTRY["cross_entropy"](logits_flat, targets_flat)
                preds = jnp.argmax(logits, axis=-1)
                correct = jnp.logical_and(valid_mask, preds == targets)
                valid_count = jnp.sum(valid_mask.astype(jnp.float32))
                acc = jnp.where(
                    valid_count > 0,
                    jnp.sum(correct.astype(jnp.float32)) / valid_count,
                    jnp.asarray(0.0, dtype=jnp.float32),
                )
                return loss_ce, acc
            method = getattr(self.cfg.training, "time_pooling_method", "max")
            p = getattr(self.cfg.training, "time_pooling_params", {})
            rs = getattr(self.cfg.training, "time_pooling_range_start", None)
            re = getattr(self.cfg.training, "time_pooling_range_end", None)
            pooled = apply_time_pooling(seq_logits, method, p, range_start=rs, range_end=re)
            targets = jnp.asarray(y_target)
            if targets.ndim == 2 and targets.shape[-1] == pooled.shape[-1]:
                targets = jnp.argmax(targets, axis=-1)
            if targets.ndim > 1 and targets.shape[-1] == 1:
                targets = jnp.squeeze(targets, axis=-1)
            targets = targets.astype(jnp.int32)
            loss_ce = LOSS_REGISTRY["cross_entropy"](pooled, targets)
            preds = jnp.argmax(pooled, axis=-1)
            acc = jnp.mean((preds == targets).astype(jnp.float32))
            return loss_ce, acc

        # Handle distillation paradigm.
        # Back-compat: if no explicit losses are configured, default to direct MSE between
        # student/teacher sequences (historical behavior).
        if paradigm == "distillation" and not (self.losses and len(self.losses) > 0):
            targets = jnp.asarray(y_b)
            method = getattr(self.cfg.training, "time_pooling_method", "none")
            params = getattr(self.cfg.training, "time_pooling_params", {})
            rs = getattr(self.cfg.training, "time_pooling_range_start", None)
            re = getattr(self.cfg.training, "time_pooling_range_end", None)

            if method and method != "none":
                if y_logits_seq.ndim != 3 or targets.ndim != 3:
                    msg = "Distillation time pooling expects [B, T, D] tensors for student and teacher."
                    raise ValueError(msg)
                y_logits_seq = apply_time_pooling(y_logits_seq, method, params, range_start=rs, range_end=re)
                targets = apply_time_pooling(targets, method, params, range_start=rs, range_end=re)

            l_mse = LOSS_REGISTRY["mse"](y_logits_seq, targets)
            # Skip accuracy metrics for distillation (it's regression)
            return l_mse, _finalize_metrics({})

        # If multiple losses configured, aggregate; otherwise fall back to default
        if self.losses and len(self.losses) > 0:
            # Handle seq2seq differently: skip time pooling and flatten logits/targets
            if mapping == "seq2seq":
                # For seq2seq, use sequence outputs directly without pooling
                logits_for_loss = y_logits_seq  # [B, T, num_classes]

                # Prepare targets for loss computation.
                #
                # IMPORTANT:
                # - For classification, seq2seq targets are class indices (or one-hot that we
                #   convert to indices) and we flatten to [B*T] for cross-entropy.
                # - For regression/distillation, seq2seq targets are float trajectories
                #   aligned to logits (e.g. [B, T, D]) and we must preserve the feature dim.
                is_classification = bool(self.cfg.training.classification)
                targets_for_loss = jnp.asarray(y_b)
                num_classes = logits_for_loss.shape[-1]

                if is_classification:
                    if targets_for_loss.ndim == 3:
                        # One-hot encoded [B, T, C]: convert to class indices
                        if targets_for_loss.shape[-1] == num_classes:
                            targets_for_loss = jnp.argmax(targets_for_loss, axis=-1)
                        elif targets_for_loss.shape[-1] == 1:
                            targets_for_loss = jnp.squeeze(targets_for_loss, axis=-1)
                    elif targets_for_loss.ndim == 1:
                        # Broadcast [B] to [B, T] if needed
                        targets_for_loss = jnp.broadcast_to(targets_for_loss[:, None], logits_for_loss.shape[:-1])
                    targets_for_loss = targets_for_loss.astype(jnp.int32)
                    valid_mask = jnp.logical_and(targets_for_loss >= 0, targets_for_loss < num_classes)
                    ignore_val = jnp.asarray(-100, dtype=jnp.int32)
                    targets_for_loss = jnp.where(valid_mask, targets_for_loss, ignore_val)

                    # Flatten for loss computation: [B*T, C] and [B*T]
                    logits_flat = logits_for_loss.reshape(-1, num_classes)
                    targets_flat = targets_for_loss.reshape(-1)
                    valid_mask_flat = valid_mask.reshape(-1)
                else:
                    # Regression/distillation: align target rank to logits and preserve feature dim.
                    if targets_for_loss.ndim == 2 and logits_for_loss.ndim == 3 and num_classes == 1:
                        # Allow scalar regression labels [B, T] for logits [B, T, 1]
                        targets_for_loss = targets_for_loss[..., None]
                    if targets_for_loss.ndim == 1 and logits_for_loss.ndim == 3 and num_classes == 1:
                        # Allow scalar regression labels [B] -> [B, T, 1]
                        targets_for_loss = jnp.broadcast_to(targets_for_loss[:, None, None], logits_for_loss.shape)

                    if targets_for_loss.shape != logits_for_loss.shape:
                        msg = (
                            "Seq2seq regression/distillation expects targets to match logits shape. "
                            f"Got logits.shape={tuple(logits_for_loss.shape)} targets.shape={tuple(targets_for_loss.shape)}. "
                            "If this is classification, set training.classification=true; "
                            "if this is distillation trajectories, ensure y has shape [B, T, D]."
                        )
                        raise ValueError(msg)

                    logits_flat = logits_for_loss.reshape(-1, num_classes)
                    targets_flat = targets_for_loss.reshape(-1, num_classes)
                    valid_mask_flat = jnp.ones((logits_flat.shape[0],), dtype=bool)
            elif self.cfg.training.autoregressive:
                # Autoregressive Training (New)
                # 1. Pool timesteps -> tokens
                tspt = self.cfg.training.ar_time_steps_per_token
                method = self.cfg.training.ar_token_pooling_method
                params = self.cfg.training.ar_token_pooling_params

                # y_logits_seq is [B, T, D] (skipping t=0)
                # pool_token_timesteps_jax expects [B, T, D]
                logits_for_loss = pool_token_timesteps_jax(y_logits_seq, tspt, method, params)

                # 2. Build targets
                # y_b is [B, seq_len] (token indices)
                targets_for_loss = build_ar_targets_jax(jnp.asarray(y_b, dtype=jnp.int32))

                # 3. Flatten
                num_classes = logits_for_loss.shape[-1]
                logits_flat = logits_for_loss.reshape(-1, num_classes)
                targets_flat = targets_for_loss.reshape(-1)

                # Valid mask (ignore -100)
                valid_mask = jnp.logical_and(targets_for_loss >= 0, targets_for_loss < num_classes)
                valid_mask_flat = valid_mask.reshape(-1)
            else:
                # For seq2static, apply time pooling
                method = getattr(self.cfg.training, "time_pooling_method", "max")
                p = getattr(self.cfg.training, "time_pooling_params", {})
                rs = getattr(self.cfg.training, "time_pooling_range_start", None)
                re = getattr(self.cfg.training, "time_pooling_range_end", None)
                logits_for_loss = apply_time_pooling(y_logits_seq, method, p, range_start=rs, range_end=re)

                is_classification = bool(self.cfg.training.classification)
                targets_for_loss = jnp.asarray(y_b)

                if is_classification:
                    # Prepare targets for loss computation (convert one-hot to indices if needed)
                    if targets_for_loss.ndim == 2 and targets_for_loss.shape[-1] == logits_for_loss.shape[-1]:
                        # One-hot encoded: convert to class indices
                        targets_for_loss = jnp.argmax(targets_for_loss, axis=-1)
                    elif targets_for_loss.ndim > 1 and targets_for_loss.shape[-1] == 1:
                        # Squeeze single dimension
                        targets_for_loss = jnp.squeeze(targets_for_loss, axis=-1)
                    targets_for_loss = targets_for_loss.astype(jnp.int32)
                    num_classes = logits_for_loss.shape[-1]
                    valid_mask = jnp.logical_and(targets_for_loss >= 0, targets_for_loss < num_classes)
                    ignore_val = jnp.asarray(-100, dtype=jnp.int32)
                    targets_for_loss = jnp.where(valid_mask, targets_for_loss, ignore_val)

                    # No flattening needed for seq2static (already [B, C] and [B])
                    logits_flat = logits_for_loss
                    targets_flat = targets_for_loss
                    valid_mask_flat = valid_mask
                else:
                    # Regression/distillation: pool targets if they are sequences, then require exact shape match.
                    if targets_for_loss.ndim == 3:
                        targets_for_loss = apply_time_pooling(targets_for_loss, method, p, range_start=rs, range_end=re)
                    if targets_for_loss.ndim == 2 and logits_for_loss.ndim == 2 and logits_for_loss.shape[1] == 1:
                        # Allow scalar targets [B, 1] or [B] for logits [B, 1]
                        pass
                    elif targets_for_loss.ndim == 1 and logits_for_loss.ndim == 2 and logits_for_loss.shape[1] == 1:
                        targets_for_loss = targets_for_loss[:, None]

                    if targets_for_loss.shape != logits_for_loss.shape:
                        msg = (
                            "Seq2static regression expects pooled targets to match pooled logits shape. "
                            f"Got logits.shape={tuple(logits_for_loss.shape)} targets.shape={tuple(targets_for_loss.shape)}."
                        )
                        raise ValueError(msg)

                    logits_flat = logits_for_loss
                    targets_flat = targets_for_loss
                    valid_mask_flat = jnp.ones((logits_for_loss.shape[0],), dtype=bool)

            total_loss = None
            acc_val = None
            raw_terms: list[jnp.ndarray] = []
            weighted_terms: list[jnp.ndarray] = []

            for spec in self.losses:
                weight = float(spec.get("weight", 1.0))
                loss_function_name = spec.get("loss_function")
                if loss_function_name not in LOSS_REGISTRY:
                    raise ValueError(
                        f"[JAX] Loss function '{loss_function_name}' not found in registry. "
                        f"Available: {sorted(LOSS_REGISTRY.keys())}"
                    )
                loss_function = LOSS_REGISTRY[loss_function_name]

                # Build kwargs from config and (optionally) runtime traces.
                # We only pass args that the loss explicitly declares to keep this fail-fast
                # and avoid TypeErrors for standard losses.
                loss_kwargs = dict(spec.get("params", {}) or {})
                try:
                    sig = inspect.signature(loss_function)
                except Exception as e:
                    raise RuntimeError(f"Unable to introspect JAX loss '{loss_function_name}': {e}") from e

                if "histories" in sig.parameters:
                    loss_kwargs["histories"] = histories_tuple
                if "final_hist" in sig.parameters:
                    loss_kwargs["final_hist"] = final_hist
                if "params" in sig.parameters:
                    loss_kwargs["params"] = params_q
                if "connection_params" in sig.parameters:
                    loss_kwargs["connection_params"] = params_q.get("connection_params")
                if "internal_connections" in sig.parameters:
                    loss_kwargs["internal_connections"] = params_q.get("internal_connections")
                if "topology" in sig.parameters:
                    # This is a Python dataclass holding JAX arrays/tuples; it is treated as static
                    # when closed over inside the jitted step.
                    loss_kwargs["topology"] = self.topology

                # Special handling for autoregressive_cross_entropy which needs 3D logits
                if loss_function_name == "autoregressive_cross_entropy":
                    l_val = loss_function(logits_for_loss, targets_for_loss, **loss_kwargs)
                else:
                    # Pass properly formatted targets (class indices, not one-hot)
                    l_val = loss_function(logits_flat, targets_flat, **loss_kwargs)
                raw_terms.append(l_val)
                weighted_terms.append(jnp.asarray(weight, dtype=l_val.dtype) * l_val)

            # Sum weighted terms; if none, fall back to default
            if len(weighted_terms) == 0:
                # No valid losses configured, fall through to default behavior
                pass
            else:
                with jax.profiler.TraceAnnotation("loss_aggregation"):
                    from functools import reduce

                    total_loss = reduce(jnp.add, weighted_terms)

                # Compute accuracy for classification tasks
                metrics: dict[str, Any] = {}
                if self.cfg.training.classification:
                    preds = jnp.argmax(logits_flat, axis=-1)
                    valid_mask_float = valid_mask_flat.astype(jnp.float32)
                    correct = (preds == targets_flat).astype(jnp.float32) * valid_mask_float
                    total_valid = jnp.sum(valid_mask_float)
                    acc_val = jnp.where(
                        total_valid > 0,
                        jnp.sum(correct) / total_valid,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    )
                    metrics["acc"] = acc_val
                # Compute additional configured metrics (JAX-native) using appropriate logits
                if len(self.metric_names) > 0:
                    mapping = getattr(self.cfg.training, "mapping", "seq2static")
                    if bool(self.cfg.training.classification) and mapping == "seq2static":
                        method = getattr(self.cfg.training, "time_pooling_method", "max")
                        p = getattr(self.cfg.training, "time_pooling_params", {})
                        rs = getattr(self.cfg.training, "time_pooling_range_start", None)
                        re = getattr(self.cfg.training, "time_pooling_range_end", None)
                        metric_logits = apply_time_pooling(y_logits_seq, method, p, range_start=rs, range_end=re)
                    else:
                        metric_logits = y_logits_seq
                    for name in self.metric_names:
                        if name in {"loss", "input_seq_len"}:
                            continue
                        fn = METRICS_REGISTRY.get(name)
                        if fn is None:
                            raise RuntimeError(f"Metric '{name}' missing from METRICS_REGISTRY (should have been validated).")
                        with jax.profiler.TraceAnnotation("metric_computation"):
                            val = fn(metric_logits, y_b)
                        metrics[name] = val
                # Return per-loss vectors for logging (fixed length at trace time)
                metrics["loss_terms"] = jnp.stack(raw_terms) if len(raw_terms) > 0 else jnp.zeros((0,), dtype=total_loss.dtype)
                metrics["loss_terms_w"] = jnp.stack(weighted_terms) if len(weighted_terms) > 0 else jnp.zeros((0,), dtype=total_loss.dtype)
                return total_loss, _finalize_metrics(metrics)

        # Fallback to default single-loss behavior
        if self.cfg.training.classification:
            l_ce, acc = _cross_entropy(y_logits_seq, y_b)
            metrics = {"acc": acc}
            # Additional metrics for classification
            if len(self.metric_names) > 0:
                mapping = getattr(self.cfg.training, "mapping", "seq2static")
                if mapping == "seq2static":
                    method = getattr(self.cfg.training, "time_pooling_method", "max")
                    p = getattr(self.cfg.training, "time_pooling_params", {})
                    rs = getattr(self.cfg.training, "time_pooling_range_start", None)
                    re = getattr(self.cfg.training, "time_pooling_range_end", None)
                    metric_logits = apply_time_pooling(y_logits_seq, method, p, range_start=rs, range_end=re)
                else:
                    metric_logits = y_logits_seq
                # Metrics computation commented out for now - can be re-enabled if needed
            return l_ce, _finalize_metrics(metrics)
        l_mse = _mse_aligned(y_logits_seq, y_b)
        metrics = {}
        if len(self.metric_names) > 0:
            # Regression metrics likely expect aligned shapes; pass y_logits_seq
            metric_logits = y_logits_seq
            for name in self.metric_names:
                if name in {"loss", "input_seq_len"}:
                    continue
                fn = METRICS_REGISTRY.get(name)
                if fn is None:
                    raise RuntimeError(f"Metric '{name}' missing from METRICS_REGISTRY (should have been validated).")
                with jax.profiler.TraceAnnotation("metric_computation"):
                    val = fn(metric_logits, y_b)
                metrics[name] = val
        return l_mse, _finalize_metrics(metrics)

    # ------------------------ Train / Eval steps ------------------------
    def _mask_non_learnable_gradients(self, grads: SoenEqxModel, model: SoenEqxModel) -> SoenEqxModel:
        """Zero out gradients for non-learnable parameters (layer + connections).

        This mirrors the previous dict-based masking, but operates on `SoenEqxModel`
        leaves so Optax updates the `eqx.Module` directly.
        """
        layers_sorted = sorted(self.jax_model.layers, key=lambda layer_item: layer_item.layer_id)

        # Mask layer parameter gradients (tuple aligned to layer order).
        layer_grads = list(grads.layer_params)
        masked_layer_grads: list[Any] = []
        for i, spec in enumerate(layers_sorted):
            if i >= len(layer_grads):
                masked_layer_grads.append(layer_grads[i] if i < len(layer_grads) else jnp.zeros((0, spec.dim)))
                continue

            grad_item = layer_grads[i]
            learnable_params = getattr(spec, "learnable_params", None) or {}
            kind = str(spec.kind).lower()

            # GRU/LSTM params are tuples; we currently treat them as learnable as a block.
            if isinstance(grad_item, tuple):
                masked_layer_grads.append(grad_item)
                continue

            grad_array = jnp.asarray(grad_item)

            if kind in ("multiplierv2", "multiplier_v2", "multipliernocc", "nocc") and grad_array.ndim == 2 and grad_array.shape[0] == 5:
                masked = grad_array
                for idx, name in enumerate(["phi_y", "bias_current", "alpha", "beta", "beta_out"]):
                    if not learnable_params.get(name, True):
                        masked = masked.at[idx].set(0.0)
                masked_layer_grads.append(masked)
            elif kind == "multiplier" and grad_array.ndim == 2 and grad_array.shape[0] == 4:
                masked = grad_array
                for idx, name in enumerate(["phi_y", "bias_current", "gamma_plus", "gamma_minus"]):
                    if not learnable_params.get(name, True):
                        masked = masked.at[idx].set(0.0)
                masked_layer_grads.append(masked)
            elif kind in ("singledendrite", "single_dendrite", "dendrite") and grad_array.ndim == 2 and grad_array.shape[0] == 4:
                masked = grad_array
                for idx, name in enumerate(["phi_offset", "bias_current", "gamma_plus", "gamma_minus"]):
                    if not learnable_params.get(name, True):
                        masked = masked.at[idx].set(0.0)
                masked_layer_grads.append(masked)
            else:
                masked_layer_grads.append(grad_item)

        # Mask external connection gradients
        masked_conn_grads = grads.connection_params * self.edge_learnable_mask

        # Mask internal connection gradients
        masked_internal_grads: dict[int, jnp.ndarray] = {}
        for layer_id, g in (grads.internal_connections or {}).items():
            lid = int(layer_id)
            if not self.internal_learnable.get(lid, True):
                masked_internal_grads[lid] = jnp.zeros_like(g)
                continue
            mask = self._internal_masks.get(lid)
            g_arr = jnp.asarray(g)
            if mask is not None and g_arr.shape == mask.shape:
                masked_internal_grads[lid] = g_arr * mask
            else:
                masked_internal_grads[lid] = g_arr

        return SoenEqxModel(
            topology=model.topology,
            layer_params=tuple(masked_layer_grads),
            connection_params=masked_conn_grads,
            internal_connections=masked_internal_grads,
        )

    def _restore_non_learnables(self, model: SoenEqxModel) -> SoenEqxModel:
        """Project parameters after an optimizer step (constraints + frozen restoration)."""
        # External connections: restore frozen edges, re-apply masks/constraints
        conn = jnp.where(self.edge_learnable_mask, model.connection_params, self.topology.edge_matrices)
        conn = conn * self.edge_masks
        if self.edge_constraint_mins is not None and self.edge_constraint_maxs is not None:
            conn = jnp.clip(conn, self.edge_constraint_mins, self.edge_constraint_maxs)

        # Internal connections: restore frozen layers; enforce structural masks + constraints
        restored_internal: dict[int, jnp.ndarray] = {}
        for layer_id, updated_internal_J in (model.internal_connections or {}).items():
            lid = int(layer_id)
            if not self.internal_learnable.get(lid, True):
                restored_internal[lid] = self._original_internal_conns.get(lid, jnp.asarray(updated_internal_J))
                continue
            internal_J = jnp.asarray(updated_internal_J)
            mask = self._internal_masks.get(lid)
            if mask is not None and internal_J.shape == mask.shape:
                internal_J = internal_J * mask
            if lid in self._internal_constraint_mins and lid in self._internal_constraint_maxs:
                internal_J = jnp.clip(internal_J, self._internal_constraint_mins[lid], self._internal_constraint_maxs[lid])
            restored_internal[lid] = internal_J

        # Layer params: restore frozen entries (prevents optimizer drift)
        layer_params_list = list(model.layer_params)
        layers_sorted = sorted(self.jax_model.layers, key=lambda layer_item: layer_item.layer_id)
        for i, spec in enumerate(layers_sorted):
            if i >= len(layer_params_list):
                continue
            learnable_params = getattr(spec, "learnable_params", None) or {}
            kind = str(spec.kind).lower()
            updated_item = layer_params_list[i]
            original_item = self._original_layer_params[i]
            if isinstance(updated_item, tuple) or isinstance(original_item, tuple):
                # Tuple-shaped params (e.g. GRU/LSTM) handled as a block for now.
                continue
            updated_array = jnp.asarray(updated_item)
            original_array = jnp.asarray(original_item)
            if kind in ("multiplierv2", "multiplier_v2", "multipliernocc", "nocc") and updated_array.ndim == 2 and updated_array.shape[0] == 5:
                for idx, name in enumerate(["phi_y", "bias_current", "alpha", "beta", "beta_out"]):
                    if not learnable_params.get(name, True):
                        updated_array = updated_array.at[idx].set(original_array[idx])
                layer_params_list[i] = updated_array
            elif kind == "multiplier" and updated_array.ndim == 2 and updated_array.shape[0] == 4:
                for idx, name in enumerate(["phi_y", "bias_current", "gamma_plus", "gamma_minus"]):
                    if not learnable_params.get(name, True):
                        updated_array = updated_array.at[idx].set(original_array[idx])
                layer_params_list[i] = updated_array
            elif kind in ("singledendrite", "single_dendrite", "dendrite") and updated_array.ndim == 2 and updated_array.shape[0] == 4:
                for idx, name in enumerate(["phi_offset", "bias_current", "gamma_plus", "gamma_minus"]):
                    if not learnable_params.get(name, True):
                        updated_array = updated_array.at[idx].set(original_array[idx])
                layer_params_list[i] = updated_array

        return SoenEqxModel(
            topology=model.topology,
            layer_params=tuple(layer_params_list),
            connection_params=conn,
            internal_connections=restored_internal,
        )

    def _train_step(
        self,
        params: SoenEqxModel,
        opt_state,
        x_b,
        y_b,
        initial_states=None,
        s1_inits=None,
        s2_inits=None,
        rng_key=None,
    ):
        """Single training step with gradient computation and parameter update.

        Note: Gradient accumulation is handled at the training loop level,
        not inside this JIT-compiled function.
        """
        # ------------------------------------------------------------------
        # Stateful BPTT safety:
        #
        # If callers pass state from a previous batch (stateful / streaming mode),
        # we MUST detach it at the batch boundary. Otherwise gradients will flow
        # back through all previous batches, leading to unbounded graph growth
        # and memory blow-ups.
        #
        # This is a "do no harm" default: initial states are not learnable params.
        # ------------------------------------------------------------------
        if initial_states is not None:
            initial_states = jax.tree_util.tree_map(jax.lax.stop_gradient, initial_states)
        if s1_inits is not None:
            s1_inits = jax.tree_util.tree_map(jax.lax.stop_gradient, s1_inits)
        if s2_inits is not None:
            s2_inits = jax.tree_util.tree_map(jax.lax.stop_gradient, s2_inits)

        with jax.profiler.TraceAnnotation("grad_compute"):
            (loss, metrics), grads = jax.value_and_grad(self._compute_batch_loss, has_aux=True)(
                params,
                x_b,
                y_b,
                initial_states,
                s1_inits,
                s2_inits,
                rng_key,
            )
            grads = self._mask_non_learnable_gradients(grads, params)
        with jax.profiler.TraceAnnotation("optimizer_update"):
            updates, opt_state = self.tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._apply_layer_param_constraints(params)
            params = self._restore_non_learnables(params)
        return params, opt_state, loss, metrics, grads

    def _accumulate_grads_step(self, params: SoenEqxModel, x_b, y_b, initial_states=None, s1_inits=None, s2_inits=None, rng_key=None):
        """Compute gradients without updating parameters (for accumulation)."""
        with jax.profiler.TraceAnnotation("grad_compute"):
            (loss, metrics), grads = jax.value_and_grad(self._compute_batch_loss, has_aux=True)(
                params,
                x_b,
                y_b,
                initial_states,
                s1_inits,
                s2_inits,
                rng_key,
            )
        grads = self._mask_non_learnable_gradients(grads, params)
        return loss, metrics, grads

    def _apply_accumulated_grads(self, params: SoenEqxModel, opt_state, accumulated_grads, num_accumulation_steps):
        """Apply accumulated gradients to parameters."""
        # Average accumulated gradients
        avg_grads = jax.tree_util.tree_map(lambda g: g / num_accumulation_steps, accumulated_grads)
        with jax.profiler.TraceAnnotation("optimizer_update"):
            updates, opt_state = self.tx.update(avg_grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._apply_layer_param_constraints(params)
            params = self._restore_non_learnables(params)
        return params, opt_state

    def _eval_step(self, params, x_b, y_b, initial_states=None, s1_inits=None, s2_inits=None, rng_key=None):
        # Same detach rule for eval: treat carried states as constants at batch boundaries.
        if initial_states is not None:
            initial_states = jax.tree_util.tree_map(jax.lax.stop_gradient, initial_states)
        if s1_inits is not None:
            s1_inits = jax.tree_util.tree_map(jax.lax.stop_gradient, s1_inits)
        if s2_inits is not None:
            s2_inits = jax.tree_util.tree_map(jax.lax.stop_gradient, s2_inits)
        loss, metrics = self._compute_batch_loss(params, x_b, y_b, initial_states, s1_inits, s2_inits, rng_key)
        return loss, metrics
