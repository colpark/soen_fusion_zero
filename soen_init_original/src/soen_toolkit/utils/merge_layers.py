from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING
import warnings

import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SOENModelCore,
)
from soen_toolkit.utils.model_tools import (
    _connection_key,
    _layer_index_map_by_id,
    rebuild_model_preserving_id_map,
    rebuild_model_preserving_state,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)


@dataclass
class MergeSpec:
    """User-facing specification for a layer merge operation.

    Attributes:
        group_ids: Layer IDs to merge. Must all be the same layer_type.
        new_layer_id: ID to assign to the merged layer. Defaults to min(group_ids).
        node_order: Order of group_ids for concatenation. Defaults to sorted(group_ids).
        normalize_ids: If True, renumber layers to 0..N-1 after merge.
        preserve_state: If True, copy numeric parameters/weights into the new model.

    """

    group_ids: list[int]
    new_layer_id: int | None = None
    node_order: list[int] | None = None
    normalize_ids: bool = False
    preserve_state: bool = True


@dataclass
class MergeResult:
    """Result of a merge operation."""

    model: SOENModelCore
    id_map: dict[int, int]
    report: dict[str, object]


@dataclass
class ConnectionPatch:
    """Container for a connection's weight tensor and structural mask."""

    weight: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype) -> ConnectionPatch:
        return ConnectionPatch(
            weight=self.weight.to(device=device, dtype=dtype),
            mask=self.mask.to(device=device, dtype=dtype),
        )


def _validate_and_prepare(model: SOENModelCore, spec: MergeSpec) -> tuple[list[int], int, list[int]]:
    """Validate merge feasibility and derive canonical identifiers.

    Returns:
        (group_ids, new_layer_id, node_order)

    """
    if not spec.group_ids:
        msg = "group_ids must be non-empty"
        raise ValueError(msg)

    group_ids = list(dict.fromkeys(int(x) for x in spec.group_ids))
    layer_by_id = {cfg.layer_id: cfg for cfg in model.layers_config}
    missing = [i for i in group_ids if i not in layer_by_id]
    if missing:
        msg = f"Unknown layer IDs in merge group: {missing}"
        raise ValueError(msg)

    # Require same layer type
    types = {layer_by_id[i].layer_type for i in group_ids}
    if len(types) != 1:
        msg = f"All layers in a merge group must share the same type, got: {sorted(types)}"
        raise ValueError(msg)

    # Canonical new layer ID
    new_layer_id = int(spec.new_layer_id) if spec.new_layer_id is not None else min(group_ids)
    # Prevent collision with an unrelated existing layer id
    if (new_layer_id not in group_ids) and (new_layer_id in layer_by_id):
        msg = f"new_layer_id {new_layer_id} is already used by a non-merged layer"
        raise ValueError(
            msg,
        )

    # Node order for concatenation
    if spec.node_order is None:
        node_order = sorted(group_ids)
    else:
        node_order = list(spec.node_order)
        if set(node_order) != set(group_ids):
            msg = "node_order must be a permutation of group_ids"
            raise ValueError(msg)

    # Soft topology warnings (do not block)
    topo = _topology_intel(model, group_ids)
    if topo["cycle_nodes"]:
        warnings.warn(
            (f"Merging selection will create feedback between the merged layer and node(s) {topo['cycle_nodes']}. Consider using a stepwise global solver."),
            UserWarning,
            stacklevel=2,
        )
    if topo["touches_input"]:
        warnings.warn(
            "Selection includes the first layer by layer ID; external input mapping may change.",
            UserWarning,
            stacklevel=2,
        )
    if topo["touches_output"]:
        warnings.warn(
            "Selection includes the last layer by layer ID; output channel mapping may change.",
            UserWarning,
            stacklevel=2,
        )

    return group_ids, new_layer_id, node_order


# Public helper for GUI/tools
def topology_intel(model: SOENModelCore, group_ids: list[int]) -> dict[str, object]:
    return _topology_intel(model, group_ids)


def _collect_dims(model: SOENModelCore, ids: Iterable[int]) -> dict[int, int]:
    """Return per-layer dimension for the provided IDs."""
    return {i: int(model.layer_nodes[i]) for i in ids}


def _build_new_layer_config(
    model: SOENModelCore,
    group_ids: list[int],
    new_layer_id: int,
    node_order: list[int],
) -> LayerConfig:
    """Create the merged LayerConfig (params copied from the first in order).

    Sets params['dim'] to the concatenated size. Solver is preserved from the
    first layer and only adjusted later if intra‑group edges require FE.

    Also handles polarity merging: extracts polarity from each layer, concatenates,
    and stores as an explicit list in the config to prevent polarity_init override.
    """
    cfg_by_id = {cfg.layer_id: cfg for cfg in model.layers_config}
    first_cfg = cfg_by_id[node_order[0]]
    dims = _collect_dims(model, node_order)
    new_dim = sum(dims[i] for i in node_order)

    # Copy params and override dim; do NOT force solver unless necessary later
    new_params = dict(first_cfg.params)
    new_params["dim"] = new_dim

    # Handle polarity merging
    # Extract polarity from each layer in node_order
    polarity_parts = []
    has_any_polarity = False

    for lid in node_order:
        cfg = cfg_by_id[lid]
        dim = dims[lid]

        # Try to get polarity from config
        polarity_explicit = cfg.params.get("polarity")
        polarity_file = cfg.params.get("polarity_file")
        polarity_init = cfg.params.get("polarity_init")

        polarity = None
        if polarity_explicit is not None:
            # Handle list or numpy array
            if isinstance(polarity_explicit, list):
                polarity = torch.tensor(polarity_explicit, dtype=torch.int8)
            elif isinstance(polarity_explicit, torch.Tensor):
                polarity = polarity_explicit.to(dtype=torch.int8)
            else:
                polarity = torch.as_tensor(polarity_explicit, dtype=torch.int8)
            has_any_polarity = True
        elif polarity_file:
            # Load from file
            from soen_toolkit.core.layers.common.connectivity_metadata import load_neuron_polarity
            polarity = load_neuron_polarity(polarity_file)
            has_any_polarity = True
        elif polarity_init:
            # Generate from init spec
            from soen_toolkit.utils.polarity_utils import (
                generate_alternating_polarity,
                generate_excitatory_polarity,
                generate_inhibitory_polarity,
                generate_random_polarity,
            )

            if isinstance(polarity_init, dict):
                excitatory_ratio = polarity_init.get("excitatory_ratio", 0.8)
                seed = polarity_init.get("seed")
                polarity = generate_random_polarity(dim, excitatory_ratio=excitatory_ratio, seed=seed)
            elif polarity_init in {"alternating", "50_50"}:
                polarity = generate_alternating_polarity(dim)
            elif polarity_init == "excitatory":
                polarity = generate_excitatory_polarity(dim)
            elif polarity_init == "inhibitory":
                polarity = generate_inhibitory_polarity(dim)
            else:
                # Unknown, default to no polarity
                polarity = None

            if polarity is not None:
                polarity = torch.from_numpy(polarity)
                has_any_polarity = True

        # If no polarity, assume unrestricted (0)
        if polarity is None:
            polarity = torch.zeros(dim, dtype=torch.int8)

        polarity_parts.append(polarity)

    # Concatenate polarity
    if has_any_polarity:
        merged_polarity = torch.cat(polarity_parts, dim=0)

        # Store explicit list in params and remove other polarity keys
        new_params["polarity"] = merged_polarity.cpu().tolist()
        new_params.pop("polarity_file", None)
        new_params.pop("polarity_init", None)

        log.debug("[merge] Merged polarity into explicit list of length %d", len(new_params["polarity"]))
    else:
        # No polarity in any layer, remove any polarity keys
        new_params.pop("polarity", None)
        new_params.pop("polarity_file", None)
        new_params.pop("polarity_init", None)

    return LayerConfig(
        layer_id=new_layer_id,
        model_id=getattr(first_cfg, "model_id", 0),
        layer_type=first_cfg.layer_type,
        params=new_params,
        description=getattr(first_cfg, "description", ""),
        noise=getattr(first_cfg, "noise", None),
        perturb=getattr(first_cfg, "perturb", None),
    )


def _topology_intel(model: SOENModelCore, group_ids: list[int]) -> dict[str, object]:
    """Return a small bundle of topology facts useful for warnings/reporting.

    Keys:
      - inbound_sources: list[int]
      - outbound_targets: list[int]
      - cycle_nodes: list[int]  (nodes that both feed the group and are fed by it)
      - touches_input: bool     (selection includes first layer by ID)
      - touches_output: bool    (selection includes last layer by ID)
    """
    inbound_sources = _discover_inbound_sources(model, group_ids)
    outbound_targets = _discover_outbound_targets(model, group_ids)
    cycle_nodes = sorted(set(inbound_sources).intersection(outbound_targets))

    try:
        all_ids = sorted(int(cfg.layer_id) for cfg in model.layers_config)
        first_id = all_ids[0] if all_ids else None
        last_id = all_ids[-1] if all_ids else None
    except Exception:
        first_id = last_id = None

    touches_input = (first_id in group_ids) if first_id is not None else False
    touches_output = (last_id in group_ids) if last_id is not None else False

    return {
        "inbound_sources": inbound_sources,
        "outbound_targets": outbound_targets,
        "cycle_nodes": cycle_nodes,
        "touches_input": touches_input,
        "touches_output": touches_output,
    }


def _group_has_internal_edges(model: SOENModelCore, group_ids: list[int]) -> bool:
    for cc in model.connections_config:
        if (cc.from_layer in group_ids) and (cc.to_layer in group_ids):
            return True
    return False


def _is_connection_param(name: str) -> bool:
    """Check if a parameter name refers to a connection weight managed by the model.

    These params share memory with model.connections and should not be handled
    as layer params during merge (they're handled via connection overrides).
    """
    # Skip internal_J (legacy) and connectivity.* (modern SingleDendrite/MultiDendrite layers)
    if name == "internal_J":
        return True
    if name.startswith("connectivity."):
        return True
    return False


def _classify_mergeable_params(model: SOENModelCore, node_order: list[int]) -> dict[str, str]:
    """Classify parameters for generic merging.

    Returns mapping param_name -> kind where kind ∈ {"1d", "2d_square", "skip"}:
      - 1d:     concatenate along dim 0
      - 2d_square: block‑diagonal assembly (dim×dim per layer)
      - skip:   leave as is (copy from the first layer in order) – covers scalars

    Notes:
      - Connection parameters (internal_J, connectivity.*) are handled via the
        connection override system, not as layer parameters.
      - If shapes mismatch within a kind (e.g., some layer missing a param), we fall back to "skip".

    """
    dims = _collect_dims(model, node_order)
    id_to_idx = _layer_index_map_by_id(model)
    # Intersect the parameter name set across all layers in the group
    # Skip connection-related params as they're handled via conn_overrides
    name_sets = []
    for lid in node_order:
        layer = model.layers[id_to_idx[lid]]
        name_sets.append({n for n, _ in layer.named_parameters() if not _is_connection_param(n)})
    common_names = set.intersection(*name_sets) if name_sets else set()

    classification: dict[str, str] = {}
    for name in common_names:
        kinds: list[str] = []
        ok = True
        for lid in node_order:
            layer = model.layers[id_to_idx[lid]]
            dim_l = dims[lid]
            p = dict(layer.named_parameters()).get(name)
            if p is None:
                ok = False
                break
            if p.dim() == 1 and p.shape[0] == dim_l:
                kinds.append("1d")
            elif p.dim() == 2 and tuple(p.shape) == (dim_l, dim_l):
                kinds.append("2d_square")
            elif p.dim() == 0:
                kinds.append("skip")
            else:
                kinds.append("skip")
        if not ok:
            continue
        # If any layer marks it as skip, mark the whole param skip
        if all(k == "1d" for k in kinds):
            classification[name] = "1d"
        elif all(k == "2d_square" for k in kinds):
            classification[name] = "2d_square"
        else:
            classification[name] = "skip"
    return classification


def _discover_inbound_sources(model: SOENModelCore, group_ids: list[int]) -> list[int]:
    sources: set[int] = set()
    for cc in model.connections_config:
        if (cc.to_layer in group_ids) and (cc.from_layer not in group_ids):
            sources.add(cc.from_layer)
    return sorted(sources)


def _discover_outbound_targets(model: SOENModelCore, group_ids: list[int]) -> list[int]:
    targets: set[int] = set()
    for cc in model.connections_config:
        if (cc.from_layer in group_ids) and (cc.to_layer not in group_ids):
            targets.add(cc.to_layer)
    return sorted(targets)


def _get_conn_param(model: SOENModelCore, src: int, dst: int) -> torch.nn.Parameter | None:
    """Fetch a connection parameter, handling unified and legacy naming.

    Prefers unified keys 'J_<src>_to_<dst>'. Falls back to legacy
    'internal_<i>' when src == dst and unified key is absent.
    """
    key = f"J_{src}_to_{dst}"
    p = model.connections.get(key)
    if p is not None:
        return p
    if src == dst:
        legacy = f"internal_{src}"
        return model.connections.get(legacy)
    return None


def _infer_device_dtype(model: SOENModelCore) -> tuple[torch.device, torch.dtype]:
    """Best-effort inference of a suitable device/dtype based on model params or connections."""
    # Try parameters first
    try:
        first = next(model.parameters())
        return first.device, first.dtype
    except StopIteration:
        pass
    # Fall back to connections
    for t in getattr(model, "connections", {}).values():
        try:
            return t.device, t.dtype
        except Exception:
            continue
    # CPU/float32 fallback
    return torch.device("cpu"), torch.float32


def _get_conn_mask(model: SOENModelCore, src: int, dst: int) -> torch.Tensor | None:
    """Fetch a connection mask tensor from model.connection_masks."""
    key = _connection_key(src, dst)
    m = model.connection_masks.get(key)
    if m is not None:
        return m
    if src == dst:
        legacy = f"internal_{src}"
        return model.connection_masks.get(legacy)
    return None


def _snapshot_masks(model: SOENModelCore) -> dict[tuple[int, int], torch.Tensor]:
    """Unused now; kept for reference/logging if needed later."""
    lookup: dict[tuple[int, int], torch.Tensor] = {}
    for key, mask in model.connection_masks.items():
        try:
            if key.startswith("J_"):
                _, from_id, _, to_id = key.split("_")
                pair = (int(from_id), int(to_id))
            elif key.startswith("internal_"):
                idx = int(key.split("_")[1])
                pair = (idx, idx)
            else:
                continue
            lookup[pair] = mask.detach().clone()
        except Exception:
            continue
    return lookup


def _build_numeric_overrides(
    model: SOENModelCore,
    group_ids: list[int],
    new_layer_id: int,
    node_order: list[int],
) -> tuple[dict[tuple[int, int], ConnectionPatch], dict[str, torch.Tensor]]:
    """Compute raw numeric matrices for new connections and node-wise parameter concatenations.

    Returns:
        (conn_overrides, param_overrides)
        - conn_overrides maps (src_id, dst_id) to a Tensor to be assigned to model.connections[f"J_{src}_to_{dst}"]
        - param_overrides maps new-layer parameter name -> 1D Tensor of length new_dim

    """
    dims = _collect_dims(model, node_order)
    new_dim = sum(dims[i] for i in node_order)

    # Internal block J_{new->new}
    device, dtype = _infer_device_dtype(model)
    conn_overrides: dict[tuple[int, int], ConnectionPatch] = {}

    def _stack_blocks_vert(blocks: list[torch.Tensor], default_rows: int, default_cols: int) -> torch.Tensor:
        if not blocks:
            return torch.zeros(default_rows, default_cols, device=device, dtype=dtype)
        return torch.vstack(blocks)

    def _stack_blocks_horiz(blocks: list[torch.Tensor], default_rows: int, default_cols: int) -> torch.Tensor:
        if not blocks:
            return torch.zeros(default_rows, default_cols, device=device, dtype=dtype)
        return torch.hstack(blocks)

    def _collect_block(src: int, dst: int, rows: int, cols: int) -> torch.Tensor:
        param = _get_conn_param(model, src, dst)
        if param is None:
            return torch.zeros(rows, cols, device=device, dtype=dtype)
        return param.data.to(device=device, dtype=dtype)

    def _collect_mask(src: int, dst: int, rows: int, cols: int) -> torch.Tensor:
        mask = _get_conn_mask(model, src, dst)
        if mask is None:
            return torch.zeros(rows, cols, device=device, dtype=dtype)
        return mask.to(device=device, dtype=dtype)

    # Internal connection patch -------------------------------------------------
    if _group_has_internal_edges(model, group_ids):
        blocks: list[torch.Tensor] = []
        mask_blocks: list[torch.Tensor] = []
        for to_id in node_order:
            row_blocks: list[torch.Tensor] = []
            row_mask_blocks: list[torch.Tensor] = []
            rows = dims[to_id]
            for from_id in node_order:
                cols = dims[from_id]
                row_blocks.append(_collect_block(from_id, to_id, rows, cols))
                row_mask_blocks.append(_collect_mask(from_id, to_id, rows, cols))
            blocks.append(_stack_blocks_horiz(row_blocks, rows, new_dim))
            mask_blocks.append(_stack_blocks_horiz(row_mask_blocks, rows, new_dim))
        weight = _stack_blocks_vert(blocks, new_dim, new_dim)
        mask = _stack_blocks_vert(mask_blocks, new_dim, new_dim)
        conn_overrides[(new_layer_id, new_layer_id)] = ConnectionPatch(weight=weight, mask=mask)

    # Inbound patches -----------------------------------------------------------
    inbound_sources = _discover_inbound_sources(model, group_ids)
    for src in inbound_sources:
        weight_blocks = []
        mask_blocks = []
        cols = int(model.layer_nodes[src])
        for dst in node_order:
            rows = dims[dst]
            weight_blocks.append(_collect_block(src, dst, rows, cols))
            mask_blocks.append(_collect_mask(src, dst, rows, cols))
        weight = _stack_blocks_vert(weight_blocks, new_dim, cols)
        mask = _stack_blocks_vert(mask_blocks, new_dim, cols)
        conn_overrides[(src, new_layer_id)] = ConnectionPatch(weight=weight, mask=mask)

    # Outbound patches ----------------------------------------------------------
    outbound_targets = _discover_outbound_targets(model, group_ids)
    for dst in outbound_targets:
        rows = int(model.layer_nodes[dst])
        weight_blocks = []
        mask_blocks = []
        for src in node_order:
            cols = dims[src]
            weight_blocks.append(_collect_block(src, dst, rows, cols))
            mask_blocks.append(_collect_mask(src, dst, rows, cols))
        weight = _stack_blocks_horiz(weight_blocks, rows, new_dim)
        mask = _stack_blocks_horiz(mask_blocks, rows, new_dim)
        conn_overrides[(new_layer_id, dst)] = ConnectionPatch(weight=weight, mask=mask)

    # Concatenate/assemble parameters generically based on shape classification
    id_to_idx = _layer_index_map_by_id(model)
    classification = _classify_mergeable_params(model, node_order)
    param_overrides: dict[str, torch.Tensor] = {}

    for name, kind in classification.items():
        if kind == "skip":
            continue
        if kind == "1d":
            parts: list[torch.Tensor] = []
            for lid in node_order:
                param = dict(model.layers[id_to_idx[lid]].named_parameters())[name]
                parts.append(param.detach().to(device=device, dtype=dtype).clone())
            param_overrides[name] = torch.cat(parts, dim=0)
        elif kind == "2d_square":
            # Build block-diagonal matrix in node_order
            blocks = []
            for lid in node_order:
                param = dict(model.layers[id_to_idx[lid]].named_parameters())[name]
                blocks.append(param.detach().to(device=device, dtype=dtype).clone())
            sizes = [b.shape[0] for b in blocks]
            total = sum(sizes)
            out = torch.zeros(total, total, device=device, dtype=dtype)
            r = 0
            for b in blocks:
                n = b.shape[0]
                out[r : r + n, r : r + n] = b
                r += n
            param_overrides[name] = out

    return conn_overrides, param_overrides


def _build_mask_overrides(
    model: SOENModelCore,
    group_ids: list[int],
    new_layer_id: int,
    node_order: list[int],
) -> dict[tuple[int, int], torch.Tensor]:
    """Deprecated: kept for compatibility; maintained by ConnectionPatch."""
    return {}


def _build_new_configs(
    model: SOENModelCore,
    group_ids: list[int],
    new_layer_cfg: LayerConfig,
    node_order: list[int],
) -> list[ConnectionConfig]:
    """Create a new connections_config with group edges replaced by super-layer edges.

    ConnectionConfig values are placeholders
    numeric weights are applied via overrides.
    """
    keep: list[ConnectionConfig] = []
    for cc in model.connections_config:
        if (cc.from_layer in group_ids) or (cc.to_layer in group_ids):
            continue
        keep.append(
            ConnectionConfig(
                from_layer=cc.from_layer,
                to_layer=cc.to_layer,
                connection_type=cc.connection_type,
                params=dict(cc.params) if cc.params is not None else None,
                learnable=cc.learnable,
                noise=cc.noise,
                perturb=cc.perturb,
            ),
        )

    # Add internal placeholder for new layer only if the group had intra-group edges
    if _group_has_internal_edges(model, group_ids):
        keep.append(
            ConnectionConfig(
                from_layer=new_layer_cfg.layer_id,
                to_layer=new_layer_cfg.layer_id,
                connection_type="dense",
                params={"init": "normal", "mean": 0.0, "std": 0.1},
                learnable=True,
            ),
        )

    # Inbound and outbound placeholders
    inbound_sources = _discover_inbound_sources(model, group_ids)
    for s in inbound_sources:
        keep.append(
            ConnectionConfig(
                from_layer=s,
                to_layer=new_layer_cfg.layer_id,
                connection_type="dense",
                params={"init": "normal", "mean": 0.0, "std": 0.1},
                learnable=True,
            ),
        )

    outbound_targets = _discover_outbound_targets(model, group_ids)
    for t in outbound_targets:
        keep.append(
            ConnectionConfig(
                from_layer=new_layer_cfg.layer_id,
                to_layer=t,
                connection_type="dense",
                params={"init": "normal", "mean": 0.0, "std": 0.1},
                learnable=True,
            ),
        )

    return keep


def apply_merge_layers(model: SOENModelCore, spec: MergeSpec) -> MergeResult:
    """Merge a group of same-type layers into a single super-layer.

    This performs a config rewrite, rebuilds a new model while preserving all unaffected
    weights, then applies exact numeric overrides for the new connections and concatenated
    node-wise parameters.
    """
    group_ids, new_layer_id, node_order = _validate_and_prepare(model, spec)

    cfg_by_id = {cfg.layer_id: cfg for cfg in model.layers_config}
    base_type = cfg_by_id[group_ids[0]].layer_type

    # New layer config
    new_layer_cfg = _build_new_layer_config(model, group_ids, new_layer_id, node_order)

    # Layer configs: keep others, add new
    new_layers_cfg: list[LayerConfig] = []
    for cfg in model.layers_config:
        if cfg.layer_id in group_ids:
            continue
        new_layers_cfg.append(
            LayerConfig(
                layer_id=cfg.layer_id,
                model_id=getattr(cfg, "model_id", 0),
                layer_type=cfg.layer_type,
                params=dict(cfg.params),
                description=getattr(cfg, "description", ""),
                noise=getattr(cfg, "noise", None),
                perturb=getattr(cfg, "perturb", None),
            ),
        )
    new_layers_cfg.append(new_layer_cfg)

    # Connections configs placeholder set
    new_connections_cfg = _build_new_configs(model, group_ids, new_layer_cfg, node_order)

    # Build numeric overrides and structural mask overrides (masks rebuilt from model directly)
    conn_overrides, param_overrides = _build_numeric_overrides(
        model,
        group_ids,
        new_layer_id,
        node_order,
    )

    # If the group had internal edges and the source layer used an incompatible
    # per-layer solver (e.g. PS for SingleDendrite), switch that layer's solver to FE
    # in the new cfg to mirror stepwise integration semantics.
    solver_adjusted = False
    if _group_has_internal_edges(model, group_ids):
        try:
            solver = str(new_layer_cfg.params.get("solver", "FE")).upper()
            if solver == "PS":
                new_layer_cfg.params["solver"] = "FE"
                solver_adjusted = True
        except Exception:
            pass

    # Rebuild model preserving unaffected state if requested
    preserve_mode = "all" if spec.preserve_state else "none"
    new_model = rebuild_model_preserving_state(
        base_model=model,
        sim_config=model.sim_config,
        layers_config=new_layers_cfg,
        connections_config=new_connections_cfg,
        preserve_mode=preserve_mode,
    )

    # Apply connection overrides
    with torch.no_grad():
        for (src, dst), patch in conn_overrides.items():
            key = _connection_key(src, dst)
            if key not in new_model.connections:
                # Create a clear error to aid debugging rather than silent skip
                msg = f"Missing expected connection '{key}' in rebuilt model"
                raise KeyError(msg)
            param = new_model.connections[key]
            if tuple(param.shape) != tuple(patch.weight.shape):
                msg = f"Shape mismatch for '{key}': have {tuple(param.shape)} but override is {tuple(patch.weight.shape)}"
                raise ValueError(
                    msg,
                )
            param.data.copy_(patch.weight.to(device=param.device, dtype=param.dtype))

            # Preserve structural masks from source edges where available;
            # fallback to numeric sparsity if no mask override exists.
            try:
                override_mask = patch.mask.to(device=param.device, dtype=param.dtype)
                new_model.connection_masks[key] = override_mask
                log.debug("[merge] set mask for %s: shape=%s, nonzero=%d", key, tuple(override_mask.shape), int(torch.count_nonzero(override_mask).item()))
            except Exception as e:
                log.warning("[merge] failed to set mask for %s: %s", key, e)

    # Re-apply masks once so constraints reflect the new structural masks
    try:
        new_model.apply_masks()
        log.debug("[merge] applied masks to new model")
    except Exception as e:
        log.warning("[merge] failed to apply masks: %s", e)

    # Apply parameter overrides for the new layer only
    id_to_idx_new = _layer_index_map_by_id(new_model)
    new_idx = id_to_idx_new[new_layer_id]
    new_layer = new_model.layers[new_idx]
    with torch.no_grad():
        name_to_param = dict(new_layer.named_parameters())
        for name, tensor in param_overrides.items():
            p = name_to_param.get(name)
            if p is None:
                continue
            if tuple(p.shape) != tuple(tensor.shape):
                continue
            p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))

    # Force re-application of constraints to ensure weights are valid immediately
    # (though numeric overrides likely already set valid weights, this is safe)
    with contextlib.suppress(Exception):
        new_model.enforce_param_constraints()

    # Build ID map: merged layers map to the new id; others are identity
    id_map: dict[int, int] = {}
    for cfg in model.layers_config:
        if cfg.layer_id in group_ids:
            id_map[cfg.layer_id] = new_layer_id
        else:
            id_map[cfg.layer_id] = cfg.layer_id

    # Optional: normalize ids within the utility path (preserve weights via mapping)
    if spec.normalize_ids:
        # Build normalized id mapping based on ascending order of current configs
        ordered_ids = sorted([cfg.layer_id for cfg in new_model.layers_config])
        norm_map = {lid: idx for idx, lid in enumerate(ordered_ids)}

        # Apply mapping to configs
        norm_layers = []
        for cfg in new_model.layers_config:
            norm_layers.append(
                LayerConfig(
                    layer_id=norm_map[cfg.layer_id],
                    model_id=getattr(cfg, "model_id", 0),
                    layer_type=cfg.layer_type,
                    params=dict(cfg.params),
                    description=getattr(cfg, "description", ""),
                    noise=getattr(cfg, "noise", None),
                    perturb=getattr(cfg, "perturb", None),
                ),
            )
        norm_conns = []
        for cc in new_model.connections_config:
            norm_conns.append(
                ConnectionConfig(
                    from_layer=norm_map[cc.from_layer],
                    to_layer=norm_map[cc.to_layer],
                    connection_type=cc.connection_type,
                    params=dict(cc.params) if cc.params is not None else None,
                    learnable=cc.learnable,
                    noise=getattr(cc, "noise", None),
                    perturb=getattr(cc, "perturb", None),
                ),
            )

        # Preserve via explicit id map (old->new)
        new_model = rebuild_model_preserving_id_map(
            base_model=new_model,
            sim_config=new_model.sim_config,
            layers_config=norm_layers,
            connections_config=norm_conns,
            id_map_old_to_new=norm_map,
        )
        # Update id_map to reflect the normalization as well (compose)
        id_map = {old: norm_map[new] for old, new in id_map.items()}

        # Re-apply structural masks to normalized model using the same overrides
        with contextlib.suppress(Exception):
            new_model.apply_masks()

    topo = _topology_intel(model, group_ids)
    report = {
        "merged_type": base_type,
        "group_ids": group_ids,
        "new_layer_id": new_layer_id,
        "node_order": node_order,
        "solver_adjusted": solver_adjusted,
        # Soft topology info for callers
        "cycle_nodes": topo["cycle_nodes"],
        "touches_input": topo["touches_input"],
        "touches_output": topo["touches_output"],
        # Sanity: help callers verify structural equivalence (connection set)
        "inbound_sources": topo["inbound_sources"],
        "outbound_targets": topo["outbound_targets"],
    }

    return MergeResult(model=new_model, id_map=id_map, report=report)
