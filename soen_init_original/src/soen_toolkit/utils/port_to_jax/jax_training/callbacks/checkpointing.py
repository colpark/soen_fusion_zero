from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
import pickle
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

"""Professional checkpointing system for the JAX backend.

These utilities provide complete save/load/verification capabilities:
- Enhanced .pkl format stores full architecture + params + optimizer state
- Load functions reconstruct JAXModel for training resumption
- Backward compatible with old .pkl format (params-only)
- Verification utilities to ensure JAX↔Torch conversion integrity

The .pkl checkpoint is JAX-specific training state, while .soen remains
the universal interchange format for cross-backend portability.
"""

# Checkpoint format version for future compatibility
CHECKPOINT_VERSION = "1.0.0"

if TYPE_CHECKING:
    from soen_toolkit.utils.port_to_jax.eqx_model import SoenEqxModel
    from soen_toolkit.utils.port_to_jax.jax_model import JAXModel


def _params_to_legacy_dict(params: Any) -> dict[str, Any]:
    """Normalize params to the legacy dict shape used in .pkl checkpoints.

    We intentionally keep the on-disk format stable:
      {"layer_params": ..., "connection_params": ..., "internal_connections": ...}

    This allows Stage-2/3 Equinox training (SoenEqxModel params pytree) while keeping
    backward compatibility with existing checkpoint consumers.
    """
    if hasattr(params, "as_params_dict"):
        params_dict = params.as_params_dict()
        if not isinstance(params_dict, dict):
            raise TypeError(
                "Expected params.as_params_dict() to return a dict, "
                f"got {type(params_dict)}"
            )
        return params_dict
    if isinstance(params, dict):
        return params
    raise TypeError(
        f"Unsupported params type for checkpoint serialization: {type(params)}. "
        "Expected a dict or an object with as_params_dict()."
    )


def _legacy_dict_to_eqx_model(*, topology: JAXModel, params: dict[str, Any]) -> SoenEqxModel:
    """Rebuild a `SoenEqxModel` from a legacy params dict + an existing topology."""
    from soen_toolkit.utils.port_to_jax.eqx_model import SoenEqxModel

    def _to_array(x):
        if isinstance(x, tuple):
            return tuple(_to_array(v) for v in x)
        return jnp.asarray(x)

    layer_params_raw = params.get("layer_params", ()) or ()
    layer_params = tuple(_to_array(v) for v in layer_params_raw)

    connection_params_raw = params.get("connection_params", None)
    if connection_params_raw is None:
        raise ValueError("Checkpoint params missing required key: 'connection_params'")
    connection_params = _to_array(connection_params_raw)

    internal_connections_raw = params.get("internal_connections", {}) or {}
    if not isinstance(internal_connections_raw, dict):
        raise TypeError(
            "Expected 'internal_connections' to be a dict in checkpoint params, "
            f"got {type(internal_connections_raw)}"
        )
    internal_connections = {int(k): _to_array(v) for k, v in internal_connections_raw.items()}

    return SoenEqxModel(
        topology=topology,
        layer_params=layer_params,
        connection_params=connection_params,
        internal_connections=internal_connections,
    )


def _serialize_jax_array(arr: jnp.ndarray | np.ndarray) -> list:
    """Convert JAX/numpy array to nested list for pickling.

    Args:
        arr: JAX or numpy array

    Returns:
        Nested list representation
    """
    if arr is None:
        return None
    return np.asarray(arr).tolist()


def _deserialize_to_jax_array(data: list | None) -> jnp.ndarray | None:
    """Convert nested list back to JAX array.

    Args:
        data: Nested list or None

    Returns:
        JAX array or None
    """
    if data is None:
        return None
    return jnp.array(data)


def _serialize_layer_spec(spec) -> dict:
    """Serialize LayerSpec to dict.

    Args:
        spec: LayerSpec dataclass

    Returns:
        Dictionary with serialized params and arrays
    """

    return {
        "layer_id": int(spec.layer_id),
        "kind": str(spec.kind),
        "dim": int(spec.dim),
        "params": {k: _serialize_jax_array(v) for k, v in (spec.params or {}).items()},
        "internal_J": _serialize_jax_array(spec.internal_J),
        "internal_mask": _serialize_jax_array(getattr(spec, "internal_mask", None)),
        "source_key": spec.source_key,
        "internal_mode": str(spec.internal_mode),
        "internal_dynamic_params": spec.internal_dynamic_params,
    }


def _deserialize_layer_spec(data: dict):
    """Reconstruct LayerSpec from dict.

    Args:
        data: Serialized layer spec dictionary

    Returns:
        LayerSpec instance
    """
    from soen_toolkit.utils.port_to_jax.jax_model import LayerSpec
    from soen_toolkit.utils.port_to_jax.parameter_specs import fill_missing_params, validate_layer_params

    # Deserialize params, filtering out None values
    params_dict = {}
    for k, v in data["params"].items():
        v_array = _deserialize_to_jax_array(v)
        if v_array is not None:
            params_dict[k] = v_array

    # Fill missing required parameters using single source of truth
    kind = data["kind"]
    dim = data["dim"]
    params_dict = fill_missing_params(kind, dim, params_dict)

    # Validate completeness (fail-fast if required params still missing)
    validate_layer_params(kind, params_dict)

    return LayerSpec(
        layer_id=data["layer_id"],
        kind=data["kind"],
        dim=dim,
        params=params_dict if params_dict else None,
        internal_J=_deserialize_to_jax_array(data["internal_J"]),
        internal_mask=_deserialize_to_jax_array(data.get("internal_mask")),
        source_key=data.get("source_key"),
        internal_mode=data.get("internal_mode", "fixed"),
        internal_dynamic_params=data.get("internal_dynamic_params"),
    )


def _serialize_connection_spec(spec) -> dict:
    """Serialize ConnectionSpec to dict.

    Args:
        spec: ConnectionSpec dataclass

    Returns:
        Dictionary with serialized connection data
    """

    def _serialize_scalar_or_array(value):
        if value is None:
            return None
        if isinstance(value, (np.ndarray, jnp.ndarray)) or hasattr(value, "__array__"):
            arr = np.asarray(value)
            if arr.shape == ():
                return float(arr.item())
            return arr.tolist()
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.shape == ():
                return float(arr.item())
            return arr.tolist()
        return float(value)

    return {
        "from_layer": int(spec.from_layer),
        "to_layer": int(spec.to_layer),
        "J": _serialize_jax_array(spec.J),
        "mask": _serialize_jax_array(spec.mask) if spec.mask is not None else None,
        "index": int(spec.index),
        "mode": str(spec.mode),
        "source_key": spec.source_key,
        "gamma_plus": float(spec.gamma_plus),
        "bias_current": float(spec.bias_current),
        "j_in": float(spec.j_in),
        "j_out": _serialize_scalar_or_array(spec.j_out),
        "alpha": float(spec.alpha) if spec.alpha is not None else None,
        "beta": float(spec.beta) if spec.beta is not None else None,
        "beta_out": float(spec.beta_out) if spec.beta_out is not None else None,
        # Note: ib field removed - using bias_current for both WICC and NOCC
        "half_flux_offset": bool(spec.half_flux_offset),
    }


def _deserialize_connection_spec(data: dict):
    """Reconstruct ConnectionSpec from dict.

    Args:
        data: Serialized connection spec dictionary

    Returns:
        ConnectionSpec instance
    """
    from soen_toolkit.utils.port_to_jax.jax_model import ConnectionSpec

    j_out_data = data.get("j_out", 0.38)
    if isinstance(j_out_data, list):
        j_out_value = _deserialize_to_jax_array(j_out_data)
    else:
        j_out_value = j_out_data

    return ConnectionSpec(
        from_layer=data["from_layer"],
        to_layer=data["to_layer"],
        J=_deserialize_to_jax_array(data["J"]),
        mask=_deserialize_to_jax_array(data["mask"]) if data.get("mask") is not None else None,
        index=data.get("index", 0),
        mode=data.get("mode", "fixed"),
        source_key=data.get("source_key"),
        gamma_plus=data.get("gamma_plus", 1e-3),
        # For backwards compatibility, check for old "bias_current" field and use it as bias_current for NOCC
        bias_current=data.get("bias_current", data.get("bias_current", 2.0)),
        j_in=data.get("j_in", 0.38),
        j_out=j_out_value,
        alpha=data.get("alpha"),
        beta=data.get("beta"),
        beta_out=data.get("beta_out"),
        half_flux_offset=data.get("half_flux_offset", False),
    )


def _serialize_jax_model(jax_model) -> dict:
    """Serialize JAXModel to dict.

    Args:
        jax_model: JAXModel instance

    Returns:
        Dictionary with complete architecture
    """
    return {
        "dt": float(jax_model.dt),
        "network_evaluation_method": str(jax_model.network_evaluation_method),
        "input_type": str(jax_model.input_type),
        "layers": [_serialize_layer_spec(spec) for spec in jax_model.layers],
        "connections": [_serialize_connection_spec(spec) for spec in jax_model.connections],
        "connection_constraints": jax_model.connection_constraints,
    }


def _deserialize_jax_model(data: dict):
    """Reconstruct JAXModel from dict.

    Args:
        data: Serialized JAX model dictionary

    Returns:
        JAXModel instance
    """
    from soen_toolkit.utils.port_to_jax.jax_model import JAXModel

    return JAXModel(
        dt=data["dt"],
        layers=[_deserialize_layer_spec(layer_data) for layer_data in data["layers"]],
        connections=[_deserialize_connection_spec(conn_data) for conn_data in data["connections"]],
        network_evaluation_method=data.get("network_evaluation_method", "layerwise"),
        input_type=data.get("input_type", "flux"),
        connection_constraints=data.get("connection_constraints"),
    )


def apply_params_to_jax_model(jax_model, params: dict | None) -> None:
    """Update a JAXModel in-place with layer/connection tensors from a checkpoint."""
    if not params:
        return

    from soen_toolkit.utils.port_to_jax.pure_forward import build_topology

    def _to_array(x):
        # Handle tuples (for GRU/LSTM which store params as tuple of arrays)
        if isinstance(x, tuple):
            return tuple(_to_array(item) for item in x)
        try:
            return jnp.asarray(x)
        except Exception:
            return np.asarray(x)

    # Layers: unpack flattened arrays using parameter spec ordering
    layer_params_list = params.get("layer_params", ()) or ()
    if layer_params_list:
        layer_arrays = tuple(_to_array(arr) for arr in layer_params_list)
        jax_model.clamp_and_apply_layer_param_arrays(layer_arrays, apply_to_specs=True)

    # Connections: map padded matrices back to actual shapes
    conn_arrays = params.get("connection_params", None)
    if conn_arrays is not None:
        conn_arrays = [_to_array(item) for item in conn_arrays]
        topology = build_topology(jax_model)
        edge_from_ids = getattr(topology, "edge_from_ids", ())
        edge_to_ids = getattr(topology, "edge_to_ids", ())

        conn_lookup: dict[tuple[int, int], list[Any]] = defaultdict(list)
        if edge_from_ids and edge_to_ids and len(edge_from_ids) == len(conn_arrays):
            for idx, weights in enumerate(conn_arrays):
                key = (int(edge_from_ids[idx]), int(edge_to_ids[idx]))
                conn_lookup[key].append(weights)

        for idx, conn in enumerate(jax_model.connections):
            trained_weights = None
            key = (int(conn.from_layer), int(conn.to_layer))
            bucket = conn_lookup.get(key)
            if bucket:
                trained_weights = bucket.pop(0)
            elif idx < len(conn_arrays):
                trained_weights = conn_arrays[idx]
            if trained_weights is None:
                continue
            actual_shape = tuple(map(int, conn.J.shape))
            dst_dim, src_dim = actual_shape
            try:
                conn.J = jnp.asarray(trained_weights)[:dst_dim, :src_dim]
            except Exception:
                conn.J = np.asarray(trained_weights)[:dst_dim, :src_dim]

    # Internal connections: update layer internal_J from checkpoint params
    internal_conns = params.get("internal_connections", None)
    if internal_conns is not None and isinstance(internal_conns, dict):
        logger.debug(f"Found internal_connections in params: {list(internal_conns.keys())}")
        for spec in jax_model.layers:
            layer_id = int(spec.layer_id)
            if layer_id in internal_conns:
                internal_J_raw = internal_conns[layer_id]
                # Convert to JAX array, handling various input types
                trained_internal_J = _to_array(internal_J_raw)
                old_internal_J = spec.internal_J

                # Ensure shapes match
                if old_internal_J is not None and trained_internal_J.shape != old_internal_J.shape:
                    logger.warning(f"Shape mismatch for layer {layer_id} internal_J: old={old_internal_J.shape}, new={trained_internal_J.shape}")
                    # Try to reshape or slice to match
                    if trained_internal_J.size >= old_internal_J.size:
                        trained_internal_J = trained_internal_J.reshape(old_internal_J.shape)[: old_internal_J.shape[0], : old_internal_J.shape[1]]
                    else:
                        logger.error(f"Cannot reshape internal_J for layer {layer_id}, skipping update")
                        continue

                spec.internal_J = trained_internal_J
                changed = True
                if old_internal_J is not None:
                    try:
                        changed = not jnp.allclose(old_internal_J, trained_internal_J, rtol=1e-5, atol=1e-6)
                    except Exception:
                        changed = True

                logger.debug(
                    f"Updated internal_J for layer {layer_id}: "
                    f"shape={trained_internal_J.shape}, "
                    f"changed={changed}, "
                    f"sample_old={old_internal_J.flatten()[:3] if old_internal_J is not None else None}, "
                    f"sample_new={trained_internal_J.flatten()[:3]}"
                )
            else:
                logger.debug(f"No internal_connection found for layer {layer_id} in params dict")
    elif internal_conns is None:
        logger.warning("No internal_connections found in checkpoint params")
    else:
        logger.warning(f"internal_connections has unexpected type: {type(internal_conns)}, expected dict")


def _serialize_topology(topology) -> dict:
    """Serialize topology metadata.

    Args:
        topology: Topology object from build_topology()

    Returns:
        Dictionary with topology metadata
    """
    return {
        "layer_param_shapes": [list(s) for s in getattr(topology, "layer_param_shapes", [])],
        "connection_param_shapes": [list(s) for s in getattr(topology, "connection_param_shapes", [])],
        "learnable_connections": list(getattr(topology, "learnable_connections", [])),
        "edge_learnable": list(getattr(topology, "edge_learnable", ())),
        "internal_learnable": dict(getattr(topology, "internal_learnable", {})) if getattr(topology, "internal_learnable", None) is not None else None,
        "dt": float(topology.dt),
    }


def save_last_checkpoint(
    *,
    ckpt_dir: Path,
    params: Any,
    opt_state: Any | None,
    epoch: int,
    global_step: int,
    val_loss: float,
    jax_model=None,
    topology=None,
    source_model_path: str | None = None,
) -> tuple[Path, Path]:
    """Save last checkpoint with full architecture (enhanced format).

    Args:
        ckpt_dir: Checkpoint directory
        params: Parameter dictionary {"layer_params": ..., "connection_params": ...}
        opt_state: Optimizer state
        epoch: Current epoch
        global_step: Global training step
        val_loss: Validation loss
        jax_model: JAXModel instance (optional for backward compatibility)
        topology: Topology metadata (optional)
        source_model_path: Path to original .soen/yaml model (optional)

    Returns:
        Tuple of (pkl_path, soen_path)
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = ckpt_dir / "last.pkl"

    params_dict = _params_to_legacy_dict(params)

    # Build checkpoint data
    checkpoint = {
        "params": params_dict,
        "opt_state": opt_state,
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "checkpoint_version": CHECKPOINT_VERSION,
    }

    # Add architecture if provided (enhanced format)
    if jax_model is not None:
        checkpoint["jax_model"] = _serialize_jax_model(jax_model)
        logger.debug("Saved checkpoint with JAXModel architecture")

    if topology is not None:
        checkpoint["topology"] = _serialize_topology(topology)
        logger.debug("Saved checkpoint with topology metadata")

    if source_model_path is not None:
        checkpoint["source_model_path"] = str(source_model_path)

    # Save checkpoint
    with open(pkl_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=4)

    soen_path = ckpt_dir / "last.soen"
    logger.debug(f"Saved JAX checkpoint: {pkl_path}")

    return pkl_path, soen_path


def save_topk_checkpoint(
    *,
    ckpt_dir: Path,
    params: Any,
    opt_state: Any | None,
    epoch: int,
    global_step: int,
    val_loss: float,
    jax_model=None,
    topology=None,
    source_model_path: str | None = None,
) -> tuple[Path, Path]:
    """Save top-k checkpoint with full architecture (enhanced format).

    Args:
        ckpt_dir: Checkpoint directory
        params: Parameter dictionary {"layer_params": ..., "connection_params": ...}
        opt_state: Optimizer state
        epoch: Current epoch
        global_step: Global training step
        val_loss: Validation loss
        jax_model: JAXModel instance (optional for backward compatibility)
        topology: Topology metadata (optional)
        source_model_path: Path to original .soen/yaml model (optional)

    Returns:
        Tuple of (pkl_path, soen_path)
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"epoch={epoch:02d}-val_loss={val_loss:.4f}"
    pkl_path = ckpt_dir / f"{tag}.pkl"

    params_dict = _params_to_legacy_dict(params)

    # Build checkpoint data
    checkpoint = {
        "params": params_dict,
        "opt_state": opt_state,
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "checkpoint_version": CHECKPOINT_VERSION,
    }

    # Add architecture if provided (enhanced format)
    if jax_model is not None:
        checkpoint["jax_model"] = _serialize_jax_model(jax_model)
        logger.debug("Saved checkpoint with JAXModel architecture")

    if topology is not None:
        checkpoint["topology"] = _serialize_topology(topology)
        logger.debug("Saved checkpoint with topology metadata")

    if source_model_path is not None:
        checkpoint["source_model_path"] = str(source_model_path)

    # Save checkpoint
    with open(pkl_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=4)

    soen_path = Path(ckpt_dir) / f"{tag}.soen"
    logger.debug(f"Saved JAX checkpoint: {pkl_path}")

    return pkl_path, soen_path


def save_initial_checkpoint(
    *,
    ckpt_dir: Path,
    params: Any,
    opt_state: Any | None,
    jax_model=None,
    topology=None,
    source_model_path: str | None = None,
) -> tuple[Path, Path]:
    """Save initial checkpoint before training starts.

    Args:
        ckpt_dir: Checkpoint directory
        params: Parameter dictionary
        opt_state: Optimizer state
        jax_model: JAXModel instance
        topology: Topology metadata
        source_model_path: Path to source model

    Returns:
        Tuple of (pkl_path, soen_path)
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = ckpt_dir / "initial.pkl"

    params_dict = _params_to_legacy_dict(params)

    checkpoint = {
        "params": params_dict,
        "opt_state": opt_state,
        "epoch": 0,
        "global_step": 0,
        "val_loss": float("inf"),
        "checkpoint_version": CHECKPOINT_VERSION,
        "is_initial": True
    }

    if jax_model is not None:
        checkpoint["jax_model"] = _serialize_jax_model(jax_model)

    if topology is not None:
        checkpoint["topology"] = _serialize_topology(topology)

    if source_model_path is not None:
        checkpoint["source_model_path"] = str(source_model_path)

    with open(pkl_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=4)

    soen_path = ckpt_dir / "initial.soen"
    logger.debug(f"Saved initial JAX checkpoint: {pkl_path}")

    return pkl_path, soen_path


def load_checkpoint(pkl_path: Path | str) -> dict:
    """Load complete JAX checkpoint.

    Args:
        pkl_path: Path to .pkl checkpoint file

    Returns:
        Dictionary with checkpoint data

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint is corrupted
    """
    pkl_path = Path(pkl_path)

    if not pkl_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pkl_path}")

    try:
        with open(pkl_path, "rb") as f:
            checkpoint = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {pkl_path}: {e}") from e

    # Check format
    version = checkpoint.get("checkpoint_version", "0.0.0")
    if "jax_model" not in checkpoint:
        logger.warning(f"Loaded legacy checkpoint (no architecture). Version: {version}. Cannot resume JAX training without architecture.")
    else:
        logger.info(f"Loaded enhanced checkpoint with architecture. Version: {version}")

    return checkpoint


def reconstruct_jax_model(checkpoint: dict):
    """Reconstruct JAXModel from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        JAXModel instance

    Raises:
        ValueError: If checkpoint doesn't contain architecture
    """
    if "jax_model" not in checkpoint:
        raise ValueError("Cannot reconstruct JAXModel: checkpoint does not contain architecture. This is likely an old-format checkpoint. Please retrain or convert from .soen.")

    jax_model = _deserialize_jax_model(checkpoint["jax_model"])
    logger.info(f"Reconstructed JAXModel with {len(jax_model.layers)} layers, {len(jax_model.connections)} connections")

    return jax_model


def resume_training(pkl_path: Path | str):
    """Resume JAX training from checkpoint.

    Args:
        pkl_path: Path to .pkl checkpoint file

    Returns:
        Tuple of (jax_model, params, opt_state, epoch, global_step).

        - `params` is the legacy dict by default.
        - If Equinox is available, a `SoenEqxModel` is also provided under key
          `checkpoint["eqx_model"]` for convenience.

    Raises:
        ValueError: If checkpoint cannot be used for resumption
    """
    checkpoint = load_checkpoint(pkl_path)

    # Reconstruct model
    jax_model = reconstruct_jax_model(checkpoint)

    # Extract training state
    params = checkpoint["params"]
    opt_state = checkpoint.get("opt_state")
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)

    # Convenience: optionally materialize a SoenEqxModel from the checkpoint params.
    # Keep this fail-soft (not fail-silent): log why it could not be created.
    try:
        checkpoint["eqx_model"] = _legacy_dict_to_eqx_model(topology=jax_model, params=params)
    except Exception as e:
        logger.debug(f"Could not reconstruct SoenEqxModel from checkpoint params: {e}")

    logger.info(f"Resuming training from epoch {epoch}, step {global_step}")

    return jax_model, params, opt_state, epoch, global_step


__all__ = [
    "save_last_checkpoint",
    "save_topk_checkpoint",
    "save_initial_checkpoint",
    "load_checkpoint",
    "reconstruct_jax_model",
    "resume_training",
    "apply_params_to_jax_model",
    "_params_to_legacy_dict",
    "_legacy_dict_to_eqx_model",
    "CHECKPOINT_VERSION",
]
