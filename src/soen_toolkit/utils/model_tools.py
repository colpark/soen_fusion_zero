"""Conversion, extraction, transformation etc."""

#  FILEPATH: src/soen_toolkit/utils/model_tools.py

from collections.abc import Iterable
import contextlib
from dataclasses import asdict, is_dataclass
from datetime import datetime

# Define UTC for Python < 3.11 compatibility
try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc  # noqa: UP017
import json

import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    NoiseConfig,
    PerturbationConfig,
    SimulationConfig,
    SOENModelCore,
)


def create_global_connection_matrix(model) -> torch.Tensor:
    """Constructs a global block matrix J from the model's connection matrices.

    The global matrix is built as a block matrix where each block corresponds to
    the connection from one layer ("from") to another ("to"). By convention, rows
    correspond to target (to) layers and columns to source (from) layers.

    Connection keys in model.connections use the unified form
    "J_{from}_to_{to}" with matrix shape (n_to x n_from). Internal (self)
    connections are therefore named "J_{i}_to_{i}" with shape (n_i x n_i).
    For backward compatibility, legacy keys of the form "internal_{i}" are also
    supported when encountered.

    Args:
        model: The SOEN model instance.

    Returns:
        A torch.Tensor representing the global connection matrix.

    """
    # Get the layer dimensions; assume model.layer_nodes is a dict mapping layer IDs to dims.
    # Sorting by layer id will ensure a consistent ordering.
    sorted_layers = sorted(model.layer_nodes.keys())

    # Compute start indices for each layer
    start_idx = {}
    current = 0
    for layer in sorted_layers:
        start_idx[layer] = current
        current += model.layer_nodes[layer]
    total_nodes = current

    # Create the global connection matrix (rows: to, columns: from)
    # Determine a safe device to allocate on. Some models may have no
    # registered parameters at this point (yielding StopIteration), so we
    # fall back to the device of any connection tensor, or CPU.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None
    if device is None or str(device) == "cpu":
        for t in getattr(model, "connections", {}).values():
            try:
                device = t.device
                break
            except Exception:
                continue
    if device is None:
        device = torch.device("cpu")

    global_J = torch.zeros(total_nodes, total_nodes, device=device)

    # Iterate over all connections in the model
    for key, weight in model.connections.items():
        # Ensure weight is detached from gradients for assignment
        weight_data = weight.data

        if key.startswith("J_"):
            # Expected key format: "J_{from}_to_{to}"
            try:
                parts = key.split("_")
                from_layer = int(parts[1])
                to_layer = int(parts[3])
            except Exception as e:
                msg = f"Unexpected key format for connection '{key}': {e}"
                raise ValueError(msg)
        elif key.startswith("internal_"):
            # Legacy support: treat as J_{i}_to_{i}
            try:
                layer = int(key.split("_")[1])
            except Exception as e:
                msg = f"Unexpected key format for internal connection '{key}': {e}"
                raise ValueError(msg)
            from_layer = layer
            to_layer = layer
        else:
            msg = f"Unrecognised connection key format: {key}"
            raise ValueError(msg)

        # Compute the block indices
        row_start = start_idx[to_layer]
        row_end = row_start + model.layer_nodes[to_layer]
        col_start = start_idx[from_layer]
        col_end = col_start + model.layer_nodes[from_layer]

        # Check dimensions of the weight matrix
        expected_shape = (model.layer_nodes[to_layer], model.layer_nodes[from_layer])
        if weight_data.shape != expected_shape:
            msg = f"Connection {key} has shape {weight_data.shape}, but expected {expected_shape}."
            raise ValueError(msg)

        global_J[row_start:row_end, col_start:col_end] = weight_data

    return global_J


def create_global_connection_mask(model) -> torch.Tensor:
    """Assemble the global binary mask (1 = allowed edge, 0 = pruned)."""
    sorted_layers = sorted(model.layer_nodes.keys())
    start_idx = {}
    current = 0
    for layer in sorted_layers:
        start_idx[layer] = current
        current += model.layer_nodes[layer]
    total_nodes = current

    device = torch.device("cpu")
    mask_device = getattr(model, "connection_masks", {})
    for m in mask_device.values():
        try:
            device = m.device
            break
        except Exception:
            continue

    global_mask = torch.zeros(total_nodes, total_nodes, device=device)

    for key, mask in getattr(model, "connection_masks", {}).items():
        if key.startswith("J_"):
            try:
                parts = key.split("_")
                from_layer = int(parts[1])
                to_layer = int(parts[3])
            except Exception as e:
                msg = f"Unexpected key format for connection mask '{key}': {e}"
                raise ValueError(msg)
        elif key.startswith("internal_"):
            try:
                layer = int(key.split("_")[1])
            except Exception as e:
                msg = f"Unexpected key format for internal connection mask '{key}': {e}"
                raise ValueError(msg)
            from_layer = layer
            to_layer = layer
        else:
            continue

        row_start = start_idx[to_layer]
        row_end = row_start + model.layer_nodes[to_layer]
        col_start = start_idx[from_layer]
        col_end = col_start + model.layer_nodes[from_layer]

        block = mask.to(device=device, dtype=global_mask.dtype)
        expected_shape = (row_end - row_start, col_end - col_start)
        if block.shape != expected_shape:
            msg = f"Connection mask {key} has shape {block.shape}, expected {expected_shape}."
            raise ValueError(
                msg,
            )
        global_mask[row_start:row_end, col_start:col_end] = block

    return global_mask


def export_model_to_json(
    model: SOENModelCore,
    filename: str | None = None,
) -> str:
    """Serialize a SOENModelCore into a navigable JSON schema.

    JSON layout:
      - version / metadata
      - simulation: SimulationConfig
      - layers: {
          "layer_{id}": {
            config: { type, params_config, model_id, description },
            noise: { phi, g, s, ... },
            perturb: { phi_mean, phi_std, ... },
            parameters: { <param_name>: [...], … },
            buffers: { <buffer_name>: [...], ... }
          }, …
        }
      - connections: {
          config: { <conn_name>: {from,to,type,params_config,learnable,noise,perturb}, … },
          matrices: { <conn_name>: [[…],…], … },
          global_matrix: [[…],…]
        }

    Args:
        model: your loaded SOENModelCore
        filename: if given, writes JSON to this path

    Returns:
        the pretty‑printed JSON string

    """

    # Helper to turn config-like objects into plain dicts safely
    def _to_plain_dict(obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            try:
                return asdict(obj)
            except Exception:
                pass
        # Fallback for simple objects with __dict__
        try:
            return dict(vars(obj))
        except Exception:
            return {}

    # Unified connection naming (J_<from>_to_<to>) is used in the model core.
    # Legacy internal_<i> keys may still appear at runtime for backward compatibility,
    # but JSON export preserves the actual connection tensor names.
    out = {
        "version": "1.0",
        "metadata": {
            "exported_at": datetime.now(UTC).isoformat(),
            "format": "soen-json",
            "torch_version": torch.__version__,
        },
        "simulation": _to_plain_dict(model.sim_config),
        "layers": {},
        "connections": {"config": {}, "matrices": {}, "masks": {}},
    }

    # --- layers ---
    for cfg, layer in zip(model.layers_config, model.layers, strict=False):
        key = f"layer_{cfg.layer_id}"
        # config block
        config = {"type": cfg.layer_type, "params_config": cfg.params}
        # Include model_id for provenance/visualisation when available
        try:
            config["model_id"] = int(getattr(cfg, "model_id", 0))
        except Exception:
            config["model_id"] = 0
        # Include description if present
        try:
            config["description"] = getattr(cfg, "description", "") or ""
        except Exception:
            config["description"] = ""
        # noise block
        noise = _to_plain_dict(getattr(cfg, "noise", NoiseConfig()))
        perturb = _to_plain_dict(getattr(cfg, "perturb", PerturbationConfig()))
        # actual weight tensors
        parameters = {name: param.detach().cpu().tolist() for name, param in layer.named_parameters()}
        # persistent buffers (e.g. lookup tables in source functions)
        # Do not serialize large built-in tables like source_function.g_table
        buffers = {name: buf.detach().cpu().tolist() for name, buf in layer.named_buffers() if name.split(".")[-1] != "g_table"}
        out["layers"][key] = {
            "config": config,
            "noise": noise,
            "perturb": perturb,
            "parameters": parameters,
            "buffers": buffers,
        }

    # --- connections ---
    for name, param, cc in zip(
        model.connections.keys(),
        model.connections.values(),
        model.connections_config,
        strict=False,
    ):
        # configuration
        out["connections"]["config"][name] = {
            "from": cc.from_layer,
            "to": cc.to_layer,
            "type": cc.connection_type,
            "params_config": cc.params,
            "learnable": cc.learnable,
            "noise": _to_plain_dict(getattr(cc, "noise", NoiseConfig())),
            "perturb": _to_plain_dict(getattr(cc, "perturb", PerturbationConfig())),
        }
        # raw matrix
        out["connections"]["matrices"][name] = param.detach().cpu().tolist()
        try:
            mask = model.connection_masks.get(name)
            if mask is not None:
                out["connections"]["masks"][name] = mask.detach().cpu().tolist()
        except Exception:
            pass

    # global
    out["connections"]["global_matrix"] = create_global_connection_matrix(model).detach().cpu().tolist()
    try:
        from soen_toolkit.core.mixins.summary import SummaryMixin  # noqa: F401
        from soen_toolkit.utils.model_tools import (
            create_global_connection_mask as _cgcm,
        )

        out["connections"]["global_mask"] = _cgcm(model).detach().cpu().tolist()
    except Exception:
        pass

    s = json.dumps(out, indent=2)
    if filename:
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "w") as f:
            f.write(s)
    return s


def model_from_json(json_path: str) -> SOENModelCore:
    """Load a SOENModelCore from the JSON schema produced by model_to_json_improved.

    Args:
        json_path: path to the JSON file

    Returns:
        a SOENModelCore with identical sim_config, layers, connections, and weights.

    """
    with open(json_path) as f:
        data = json.load(f)

    # 1) Simulation config (filter unknown/legacy fields)
    sim_dict = data.get("simulation", {})
    try:
        import inspect as _inspect

        allowed = set(_inspect.signature(SimulationConfig.__init__).parameters.keys())
        allowed.discard("self")
        filtered_sim = {k: v for k, v in sim_dict.items() if k in allowed}
    except Exception:
        filtered_sim = sim_dict
    sim_cfg = SimulationConfig(**filtered_sim)

    # 2) Layers: rebuild LayerConfig list in ascending order of layer_id
    layers_cfg = []
    for key, entry in sorted(
        data["layers"].items(),
        key=lambda kv: int(kv[0].split("_")[1]),
    ):
        lid = int(key.split("_")[1])
        cfg = entry["config"]
        noise = NoiseConfig(**entry.get("noise", {}))
        perturb = PerturbationConfig(**entry.get("perturb", {}))
        layers_cfg.append(
            LayerConfig(
                layer_id=lid,
                layer_type=cfg["type"],
                params=cfg["params_config"],
                model_id=int(cfg.get("model_id", 0)),
                description=cfg.get("description", ""),
                noise=noise,
                perturb=perturb,
            ),
        )

    # 3) Connections: rebuild ConnectionConfig list in the same order
    conns_cfg = []
    for name, cc in data["connections"]["config"].items():
        conn_noise = NoiseConfig(**cc.get("noise", {}))
        conn_perturb = PerturbationConfig(**cc.get("perturb", {}))
        conns_cfg.append(
            ConnectionConfig(
                from_layer=cc["from"],
                to_layer=cc["to"],
                connection_type=cc["type"],
                params=cc["params_config"],
                learnable=cc["learnable"],
                noise=conn_noise,
                perturb=conn_perturb,
            ),
        )

    # 4) Instantiate empty model then load weights
    model = SOENModelCore(
        sim_config=sim_cfg,
        layers_config=layers_cfg,
        connections_config=conns_cfg,
    )

    # 5) Build a state_dict from JSON
    sd = {}
    # layers.<i>.<param>
    for key, entry in data["layers"].items():
        i = int(key.split("_")[1])
        for pname, arr in entry["parameters"].items():
            sd[f"layers.{i}.{pname}"] = torch.tensor(arr)
        # restore persistent buffers if present
        for bname, arr in entry.get("buffers", {}).items():
            sd[f"layers.{i}.{bname}"] = torch.tensor(arr)

    # connections.<name>
    for name, mat in data["connections"]["matrices"].items():
        sd[f"connections.{name}"] = torch.tensor(mat)

    # 6) Load into model
    # Use non-strict loading so that built-in buffers (e.g., source_function.g_table)
    # that are not serialized can be left at their default initialized values.
    model.load_state_dict(sd, strict=False)

    # 7) Restore connection masks if present
    try:
        masks_obj = data.get("connections", {}).get("masks", {})
        if isinstance(masks_obj, dict):
            with torch.no_grad():
                for name, arr in masks_obj.items():
                    mask_t = torch.tensor(arr)
                    if name in model.connections:
                        # Ensure shape match
                        p = model.connections[name]
                        if tuple(mask_t.shape) == tuple(p.shape):
                            model.connection_masks[name] = mask_t.to(device=p.device, dtype=p.dtype)
        # If provided, global_mask is ignored (we reconstruct from per-conn masks)
    except Exception:
        pass
    return model


# -----------------------------------------------------------------------------
# Surgical rebuild utilities
# -----------------------------------------------------------------------------


def _layer_index_map_by_id(model: SOENModelCore) -> dict[int, int]:
    """Return a mapping from layer_id to index in model.layers list."""
    return {cfg.layer_id: idx for idx, cfg in enumerate(model.layers_config)}


def _copy_matching_parameters(
    *,
    src_layer: torch.nn.Module,
    dst_layer: torch.nn.Module,
    allow_param_names: set[str] | None = None,
) -> None:
    """Copy parameters by name if present in both layers and with identical shape.

    Args:
        src_layer: source layer module
        dst_layer: destination layer module
        allow_param_names: if provided, only copy these param names

    """
    src_named = dict(src_layer.named_parameters())
    for name, dst_param in dst_layer.named_parameters():
        if allow_param_names is not None and name not in allow_param_names:
            continue
        src_param = src_named.get(name)
        if src_param is None:
            continue
        if tuple(src_param.shape) != tuple(dst_param.shape):
            continue
        # Device/dtype safe copy
        with torch.no_grad():
            dst_param.data.copy_(src_param.data.to(dst_param.device, dtype=dst_param.dtype))


def _connection_key(from_layer: int, to_layer: int) -> str:
    # Unified naming: always J_<from>_to_<to>
    return f"J_{from_layer}_to_{to_layer}"


def rebuild_model_preserving_state(
    *,
    base_model: SOENModelCore | None,
    sim_config: SimulationConfig,
    layers_config: list[LayerConfig],
    connections_config: list[ConnectionConfig],
    preserve_mode: str = "all",
    freeze_layers: Iterable[int] | None = None,
    freeze_connections: Iterable[tuple[int, int]] | None = None,
    freeze_masks: Iterable[tuple[int, int]] | None = None,
    seed: int | None = None,
) -> SOENModelCore:
    """Build a new model from configs while preserving parameters/weights where possible.

    Behavior:
    - If base_model is None: builds a fresh model (ignores preserve options).
    - If preserve_mode == "none": builds fresh regardless of base_model.
    - If preserve_mode == "all": copies every matching parameter/connection weight by name/shape.
    - If preserve_mode == "frozen_only": copies only those for layer IDs in freeze_layers and
      connections in freeze_connections.
    - If a layer's type changed or parameter shapes do not match, that layer is left as freshly
      initialized even under preserve_mode == "all".
    - Connections are preserved if the (from,to) pair still exists and shapes match.

    Args:
        base_model: The previously built model to preserve state from (or None)
        sim_config, layers_config, connections_config: new configuration objects
        preserve_mode: one of {"all", "frozen_only", "none"}
        freeze_layers: collection of layer IDs to force-preserve when preserve_mode == "frozen_only"
        freeze_connections: collection of (from_layer, to_layer) tuples to force-preserve when
                            preserve_mode == "frozen_only". Use (i,i) for internal.
        freeze_masks: collection of (from_layer, to_layer) tuples whose masks should be preserved
                      when preserve_mode == "frozen_only". Use (i,i) for internal.
        seed: optional seed before creating the new model (for reproducible fresh parts)

    Returns:
        Newly built SOENModelCore with state preserved according to the rules.

    """
    # Fast path: no base or explicit fresh build
    if base_model is None or preserve_mode == "none":
        if seed is not None:
            torch.manual_seed(seed)
        return SOENModelCore(
            sim_config=sim_config,
            layers_config=layers_config,
            connections_config=connections_config,
        )

    # Build the destination model first
    if seed is not None:
        torch.manual_seed(seed)
    new_model = SOENModelCore(
        sim_config=sim_config,
        layers_config=layers_config,
        connections_config=connections_config,
    )

    # Determine preservation sets
    preserve_all = preserve_mode == "all"
    preserve_frozen_only = preserve_mode == "frozen_only"
    frozen_layer_ids: set[int] = set(freeze_layers or [])
    frozen_conns: set[tuple[int, int]] = set(freeze_connections or [])
    frozen_masks: set[tuple[int, int]] = set(freeze_masks or [])

    src_masks: dict[str, torch.Tensor] = getattr(base_model, "connection_masks", {})

    # Validate freeze selections exist in base model
    if preserve_frozen_only and base_model is not None:
        base_layer_ids = {cfg.layer_id for cfg in base_model.layers_config}
        invalid_layers = frozen_layer_ids - base_layer_ids
        if invalid_layers:
            msg = f"freeze_layers contains IDs not in base model: {sorted(invalid_layers)}"
            raise ValueError(msg)

        base_conn_pairs = {(cc.from_layer, cc.to_layer) for cc in base_model.connections_config}
        invalid_conns = frozen_conns - base_conn_pairs
        if invalid_conns:
            msg = f"freeze_connections contains pairs not in base model: {sorted(invalid_conns)}"
            raise ValueError(msg)

        if frozen_masks:
            missing_masks = {pair for pair in frozen_masks if _connection_key(*pair) not in src_masks}
            if missing_masks:
                msg = f"freeze_masks contains pairs without masks in base model: {sorted(missing_masks)}"
                raise ValueError(msg)

    src_idx_by_id = _layer_index_map_by_id(base_model)
    dst_idx_by_id = _layer_index_map_by_id(new_model)

    # 1) Preserve layer parameters where possible
    for layer_id, dst_idx in dst_idx_by_id.items():
        if layer_id not in src_idx_by_id:
            continue  # New layer

        src_idx = src_idx_by_id[layer_id]
        src_cfg = base_model.layers_config[src_idx]
        dst_cfg = new_model.layers_config[dst_idx]

        # Respect rule: if layer type changed, do not copy its parameters
        if src_cfg.layer_type != dst_cfg.layer_type:
            continue

        # Decide whether to preserve this layer
        should_copy = preserve_all or (preserve_frozen_only and layer_id in frozen_layer_ids)
        if not should_copy:
            continue

        src_layer = base_model.layers[src_idx]
        dst_layer = new_model.layers[dst_idx]
        _copy_matching_parameters(src_layer=src_layer, dst_layer=dst_layer)

    # 2) Preserve connections where possible
    # Build quick lookup for source connection tensors
    src_conns: dict[str, torch.nn.Parameter] = dict(base_model.connections)

    for conn_cfg in connections_config:
        pair = (conn_cfg.from_layer, conn_cfg.to_layer)
        key = _connection_key(*pair)
        if key not in new_model.connections:
            continue

        # Decide whether to preserve this connection
        should_copy_conn = preserve_all or (preserve_frozen_only and pair in frozen_conns)
        should_copy_mask = preserve_all or (preserve_frozen_only and pair in frozen_masks)

        if should_copy_conn:
            src_param = src_conns.get(key)
            dst_param = new_model.connections.get(key)
            if src_param is not None and dst_param is not None and tuple(src_param.shape) == tuple(dst_param.shape):
                with torch.no_grad():
                    dst_param.data.copy_(src_param.data.to(dst_param.device, dtype=dst_param.dtype))

        # Also preserve the mask if it exists and shapes match
        if should_copy_mask:
            src_mask = src_masks.get(key)
            dst_param = new_model.connections.get(key)
            if src_mask is not None and dst_param is not None:
                try:
                    if tuple(src_mask.shape) == tuple(dst_param.shape):
                        new_model.connection_masks[key] = src_mask.to(device=dst_param.device, dtype=dst_param.dtype)
                except Exception:
                    # If mask copy fails, generate from weight sparsity as fallback
                    with contextlib.suppress(Exception):
                        new_model.connection_masks[key] = (dst_param.data != 0).to(device=dst_param.device, dtype=dst_param.dtype)

    # Enforce constraints and masks after copying
    new_model.enforce_param_constraints()

    return new_model


def rebuild_model_preserving_id_map(
    *,
    base_model: SOENModelCore,
    sim_config: SimulationConfig,
    layers_config: list[LayerConfig],
    connections_config: list[ConnectionConfig],
    id_map_old_to_new: dict[int, int],
) -> SOENModelCore:
    """Rebuild a model and preserve weights using an explicit old->new ID mapping.

    This is designed for workflows like renumbering (normalize IDs) after a merge,
    or layer insertion that shifts existing layer IDs.

    Rules:
    - Layers: copy parameters by name from old_id to new_id when shapes match.
      If a new_id is not in the mapping (i.e., it's a newly inserted layer),
      it is left freshly initialized. Fallback to same-id only when the ID mapping
      is empty (no structural change occurred).
    - Connections: copy matrices by mapping endpoints (from_old,to_old)->(from_new,to_new)
      when shapes match. Same fallback rule applies.
    - Internal connectivity is handled via unified keys J_<i>_to_<i> like any other connection.
    """
    # Build destination model fresh
    new_model = SOENModelCore(
        sim_config=sim_config,
        layers_config=layers_config,
        connections_config=connections_config,
    )

    # Invert mapping for convenience (prefer one-to-one renumbering use case)
    new_to_old: dict[int, int] = {}
    for old_id, new_id in (id_map_old_to_new or {}).items():
        # Do not overwrite if duplicate; keep first assignment
        if new_id not in new_to_old:
            new_to_old[new_id] = old_id

    # Determine whether we have an explicit mapping (structural change)
    # If so, only copy for layers/connections that are explicitly mapped
    has_explicit_mapping = bool(id_map_old_to_new)

    src_idx_by_id = _layer_index_map_by_id(base_model)
    dst_idx_by_id = _layer_index_map_by_id(new_model)

    # 1) Layers: copy params by name/shape from mapped old id
    for new_id, dst_idx in dst_idx_by_id.items():
        if new_id in new_to_old:
            # Explicitly mapped: copy from the mapped old ID
            src_id = new_to_old[new_id]
        elif not has_explicit_mapping:
            # No mapping provided: fall back to same-id preservation
            src_id = new_id
        else:
            # Mapping provided but new_id not in it: this is a new layer, skip
            continue

        src_idx = src_idx_by_id.get(src_id)
        if src_idx is None:
            continue
        src_layer = base_model.layers[src_idx]
        dst_layer = new_model.layers[dst_idx]
        _copy_matching_parameters(src_layer=src_layer, dst_layer=dst_layer)

    # 2) Connections: copy by mapped endpoints when shapes match
    src_conns: dict[str, torch.nn.Parameter] = dict(base_model.connections)
    src_masks: dict[str, torch.Tensor] = getattr(base_model, "connection_masks", {})
    for conn_cfg in connections_config:
        fn = conn_cfg.from_layer
        tn = conn_cfg.to_layer

        # Determine source endpoints using the mapping
        if fn in new_to_old:
            fo = new_to_old[fn]
        elif not has_explicit_mapping:
            fo = fn
        else:
            # from_layer not in mapping: new layer, skip this connection
            continue

        if tn in new_to_old:
            to = new_to_old[tn]
        elif not has_explicit_mapping:
            to = tn
        else:
            # to_layer not in mapping: new layer, skip this connection
            continue

        src_key = _connection_key(fo, to)
        dst_key = _connection_key(fn, tn)
        src_param = src_conns.get(src_key)
        dst_param = new_model.connections.get(dst_key)
        if src_param is None or dst_param is None:
            continue
        if tuple(src_param.shape) != tuple(dst_param.shape):
            continue
        with torch.no_grad():
            dst_param.data.copy_(src_param.data.to(dst_param.device, dtype=dst_param.dtype))

        # Also preserve the mask if it exists and shapes match
        src_mask = src_masks.get(src_key)
        if src_mask is not None:
            try:
                if tuple(src_mask.shape) == tuple(dst_param.shape):
                    new_model.connection_masks[dst_key] = src_mask.to(device=dst_param.device, dtype=dst_param.dtype)
            except Exception:
                # If mask copy fails, generate from weight sparsity as fallback
                with contextlib.suppress(Exception):
                    new_model.connection_masks[dst_key] = (dst_param.data != 0).to(device=dst_param.device, dtype=dst_param.dtype)

    new_model.enforce_param_constraints()
    return new_model


def merge_models_with_mapping(models: list[SOENModelCore]) -> tuple[SOENModelCore, list[dict[int, int]]]:
    """Merge multiple models into a single model by offsetting subsequent layer IDs and
    preserving all parameters and connections from each input model. Also returns
    the per-model ID remapping dictionaries: for each input model, a mapping from
    its original layer_id to the new merged layer_id.

    - The first model's `SimulationConfig` is used for the merged model.
    - For model i>0, each layer_id is remapped by adding an offset of
      1 + max(previous_layer_id). Connections are remapped accordingly.
    - After constructing the merged configs, a new model is built and state from
      each original model is copied into the appropriate layers/connections.

    Returns:
        Tuple[SOENModelCore, List[Dict[int,int]]]: merged model and list of id maps.

    """
    if not models:
        msg = "No models provided to merge_models_with_mapping"
        raise ValueError(msg)

    # Start with the first model's configs
    base_sim = models[0].sim_config
    merged_layers: list[LayerConfig] = []
    merged_conns: list[ConnectionConfig] = []

    # Track current maximum layer id and mapping per model
    current_max_id = -1
    current_max_model_id = -1
    per_model_mapping: list[dict[int, int]] = []

    for m in models:
        # Compute offset
        if merged_layers:
            # Next block starts after the current max id
            offset = current_max_id + 1
        else:
            offset = 0

        # Map layer IDs for this model
        id_map: dict[int, int] = {}
        # Compute a model_id offset to keep different input models visually separable
        model_id_offset = (current_max_model_id + 1) if merged_layers else 0
        for lc in m.layers_config:
            new_id = lc.layer_id + offset
            id_map[lc.layer_id] = new_id
            # Clone LayerConfig with shifted id
            merged_layers.append(
                LayerConfig(
                    layer_id=new_id,
                    model_id=getattr(lc, "model_id", 0) + model_id_offset,
                    layer_type=lc.layer_type,
                    params=dict(lc.params),
                    description=lc.description,
                    noise=lc.noise,
                    perturb=lc.perturb,
                ),
            )
            current_max_id = max(current_max_id, new_id)
            current_max_model_id = max(current_max_model_id, getattr(lc, "model_id", 0) + model_id_offset)

        # Remap connections
        for cc in m.connections_config:
            merged_conns.append(
                ConnectionConfig(
                    from_layer=id_map[cc.from_layer],
                    to_layer=id_map[cc.to_layer],
                    connection_type=cc.connection_type,
                    params=dict(cc.params) if cc.params is not None else None,
                    learnable=cc.learnable,
                    noise=cc.noise,
                    perturb=cc.perturb,
                ),
            )

        per_model_mapping.append(id_map)

    # Build the merged model fresh
    merged = SOENModelCore(
        sim_config=base_sim,
        layers_config=merged_layers,
        connections_config=merged_conns,
    )

    # Copy state from each source model according to the mapping
    for m, id_map in zip(models, per_model_mapping, strict=False):
        # Layers: copy per layer
        src_idx_by_id = _layer_index_map_by_id(m)
        dst_idx_by_id = _layer_index_map_by_id(merged)
        for old_id, new_id in id_map.items():
            src_idx = src_idx_by_id.get(old_id)
            dst_idx = dst_idx_by_id.get(new_id)
            if src_idx is None or dst_idx is None:
                continue
            src_cfg = m.layers_config[src_idx]
            dst_cfg = merged.layers_config[dst_idx]
            if src_cfg.layer_type != dst_cfg.layer_type:
                continue  # should not happen, but guard
            _copy_matching_parameters(src_layer=m.layers[src_idx], dst_layer=merged.layers[dst_idx])

        # Connections: copy by remapped keys
        src_masks: dict[str, torch.Tensor] = getattr(m, "connection_masks", {})
        for name, src_param in m.connections.items():
            if name.startswith("J_"):
                parts = name.split("_")
                old_from = int(parts[1])
                old_to = int(parts[3])
                new_from = id_map[old_from]
                new_to = id_map[old_to]
                new_key = _connection_key(new_from, new_to)
            elif name.startswith("internal_"):
                # Legacy: map to unified key
                old_id = int(name.split("_")[1])
                new_key = _connection_key(id_map[old_id], id_map[old_id])
            else:
                continue

            if new_key not in merged.connections:
                continue
            dst_param = merged.connections[new_key]
            if tuple(dst_param.shape) != tuple(src_param.shape):
                continue
            with torch.no_grad():
                dst_param.data.copy_(src_param.data.to(dst_param.device, dtype=dst_param.dtype))

            # Also copy the mask if it exists
            src_mask = src_masks.get(name)
            if src_mask is not None:
                try:
                    if tuple(src_mask.shape) == tuple(dst_param.shape):
                        merged.connection_masks[new_key] = src_mask.to(device=dst_param.device, dtype=dst_param.dtype)
                except Exception:
                    # If mask copy fails, generate from weight sparsity as fallback
                    with contextlib.suppress(Exception):
                        merged.connection_masks[new_key] = (dst_param.data != 0).to(device=dst_param.device, dtype=dst_param.dtype)

    merged.enforce_param_constraints()
    return merged, per_model_mapping


def merge_models(models: list[SOENModelCore]) -> SOENModelCore:
    """Compatibility wrapper returning only the merged model."""
    merged, _ = merge_models_with_mapping(models)
    return merged
