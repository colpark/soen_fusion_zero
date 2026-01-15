# src/soen_toolkit/model_creation_gui/model_manager.py
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field

# import json # No longer needed here for config save/load
import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)
from soen_toolkit.core.layers.common import CONNECTIVITY_BUILDERS
from soen_toolkit.core.layers.common.connectivity_metadata import CONNECTIVITY_ALIASES
from soen_toolkit.core.layers.common.metadata import (
    LAYER_PARAM_CONFIGS,
    get_layer_catalog,
)
from soen_toolkit.core.source_functions import SOURCE_FUNCTION_CATALOG
from soen_toolkit.core.source_functions.registry import SOURCE_FUNCTION_DEFAULTS
from soen_toolkit.model_creation_gui.features.state_trajectory.dataset_service import DatasetService
from soen_toolkit.utils.model_tools import (
    _connection_key,
    merge_models_with_mapping,
    rebuild_model_preserving_state,
)

if TYPE_CHECKING:
    import pathlib

from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# auto‑discovery helpers
# ---------------------------------------------------------------------
def discover_layer_types(layers_impl: str = "v1") -> dict[str, dict[str, object]]:
    """Catalog layer metadata for GUI layers picker."""
    from soen_toolkit.core.layer_registry import LAYER_TYPE_MAP

    catalog = get_layer_catalog()
    layer_map: dict[str, dict[str, object]] = {}
    for name in LAYER_TYPE_MAP:
        if name in {"Input", "Readout", "leakyGRU"}:  # hide legacy aliases from the picker
            # Hide legacy alias from GUI layer picker while keeping metadata for compatibility.
            continue
        params_cfg = {pc.name: pc for pc in LAYERS_METADATA.get(name, [])}
        info = catalog.get(name)
        layer_map[name] = {
            "params": params_cfg,
            "category": info.category if info else "Other",
            "title": info.title if info else name,
            "description": info.description if info else "",
        }
    return layer_map


def discover_source_functions() -> dict[str, dict[str, object]]:
    catalog_map: dict[str, dict[str, object]] = {}
    for key, info in SOURCE_FUNCTION_CATALOG.items():
        catalog_map[key] = {
            "title": info.title,
            "description": info.description,
            "category": info.category,
            "uses_squid_current": info.uses_squid_current,
        }
    catalog_map_defaults = dict(SOURCE_FUNCTION_DEFAULTS)
    catalog_map["__defaults__"] = catalog_map_defaults
    return catalog_map


def discover_connection_types() -> list[str]:
    """Return known connection builder keys from the v2 registry."""
    try:
        canonical = []
        alias_keys = set(CONNECTIVITY_ALIASES.keys())
        for name in sorted(CONNECTIVITY_BUILDERS.keys()):
            if name in alias_keys:
                continue
            canonical.append(name)
        return canonical
    except Exception:
        return ["dense", "one_to_one", "sparse", "block_structure"]


# Cache metadata lazily to avoid partial init cycles
LAYERS_METADATA: dict[str, list] = dict(LAYER_PARAM_CONFIGS)

# Lazy-loaded discovery caches to speed up startup
_LAYER_TYPES_DISCOVERED: dict[str, dict[str, object]] | None = None
_SOURCE_FUNCTIONS_DISCOVERED: dict[str, dict[str, object]] | None = None
_CONN_TYPES_DISCOVERED: list[str] | None = None


def _get_layer_types_discovered() -> dict[str, dict[str, object]]:
    """Lazy-load layer types on first access."""
    global _LAYER_TYPES_DISCOVERED
    if _LAYER_TYPES_DISCOVERED is None:
        _LAYER_TYPES_DISCOVERED = discover_layer_types("v2")
    return _LAYER_TYPES_DISCOVERED


def _get_source_functions_discovered() -> dict[str, dict[str, object]]:
    """Lazy-load source functions on first access."""
    global _SOURCE_FUNCTIONS_DISCOVERED
    if _SOURCE_FUNCTIONS_DISCOVERED is None:
        _SOURCE_FUNCTIONS_DISCOVERED = discover_source_functions()
    return _SOURCE_FUNCTIONS_DISCOVERED


def _get_conn_types_discovered() -> list[str]:
    """Lazy-load connection types on first access."""
    global _CONN_TYPES_DISCOVERED
    if _CONN_TYPES_DISCOVERED is None:
        _CONN_TYPES_DISCOVERED = discover_connection_types()
    return _CONN_TYPES_DISCOVERED


# Legacy module-level accessors for backward compatibility
# These will trigger lazy loading on first import
def _lazy_init():
    """Called by __getattr__ for backward-compatible lazy loading."""
    return {
        "LAYER_TYPES_DISCOVERED": _get_layer_types_discovered(),
        "SOURCE_FUNCTIONS_DISCOVERED": _get_source_functions_discovered(),
        "CONN_TYPES_DISCOVERED": _get_conn_types_discovered(),
    }


# ---------------------------------------------------------------------
@dataclass
class ModelManager(QObject):
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)
    layers: list[LayerConfig] = field(default_factory=list)
    connections: list[ConnectionConfig] = field(default_factory=list)
    model: SOENModelCore | None = None  # built model
    # Shared dataset service for persistence across dialogs
    dataset_service: DatasetService = field(default_factory=DatasetService)
    # Baseline preservation state
    baseline_mode: str = "current"  # current | last_file | choose_file | snapshot
    baseline_file_path: pathlib.Path | None = None
    last_loaded_path: pathlib.Path | None = None
    _baseline_snapshot_model: SOENModelCore | None = None
    # Cache the last successfully built/loaded model so preservation from
    # "current" works even if self.model is temporarily cleared by edits
    _last_built_model: SOENModelCore | None = None
    # Track provenance after merges: list of mappings per merged source
    model_provenance: list[dict[int, int]] = field(default_factory=list)
    model_provenance_labels: list[str] = field(default_factory=list)
    # Track the last merge id_map for mapping-aware preserve/rebuild workflows
    last_merge_id_map: dict[int, int] = field(default_factory=dict)

    # Signal emitted when the model is successfully built or loaded
    model_changed = pyqtSignal()

    # Need __post_init__ to call QObject constructor when using dataclass
    def __post_init__(self):
        QObject.__init__(self)
        self._internal_connectivity_settings: dict[int, dict] = {}

    # ------------------------------ helpers ------------------------------
    def _apply_model(self, core: SOENModelCore, *, cache_last_built: bool = True) -> None:
        """Adopt ``core`` as the current model and optionally cache a deep copy."""
        self.model = core
        self.sim_config = core.sim_config
        self.layers = core.layers_config
        self.connections = core.connections_config
        # Log mask count for debugging
        mask_count = len(getattr(core, "connection_masks", {}))
        log.debug("_apply_model: adopting model with %d connection masks", mask_count)
        if cache_last_built:
            self._cache_last_built_model(core)

    def _cache_last_built_model(self, core: SOENModelCore | None) -> None:
        """Deep-copy ``core`` for preservation workflows; ignore copy failures."""
        if core is None:
            return
        with contextlib.suppress(Exception):
            self._last_built_model = copy.deepcopy(core)

    # -------- persistence --------
    # REMOVED: save_config_json - Superseded by export_model_to_json in MainWindow
    # REMOVED: load_config_json - Superseded by model_from_json in MainWindow

    # ------------------------------------------------------------------
    # Generic save/load helpers (support .pth)
    # ------------------------------------------------------------------

    def save_model_file(self, path: pathlib.Path) -> None:
        """Save model to .pth extension (fallback supports .soen)."""
        if self.model is None:
            log.error("Attempted to save model, but no model is built.")
            msg = "No model built yet."
            raise RuntimeError(msg)
        # Accept both .soen and .pth; if no known extension, default to .soen
        if path.suffix.lower() not in {".pth", ".soen"}:
            path = path.with_suffix(".soen")
        try:
            self.model.save(str(path))
            log.info("Model state and config saved to file: %s", path)
        except Exception:
            log.exception("Error saving model to file: %s", path)
            raise  # Re-raise the exception

    def load_model_file(self, path: pathlib.Path) -> None:
        """Load model from .pth (or legacy .soen) file."""
        # Support both .soen and .pth transparently (no warnings)
        try:
            # Suppress load logs to avoid cluttering the visualization area
            loaded = SOENModelCore.load(str(path), show_logs=False, strict=False)
            self._apply_model(loaded)
            self.last_loaded_path = path
            log.info("Model state and config loaded from PTH: %s", path)
            # If tolerant load reported diffs, log a summary once
            try:
                miss = getattr(self.model, "_load_missing_keys", []) or []
                unexp = getattr(self.model, "_load_unexpected_keys", []) or []
                if miss or unexp:
                    log.warning(
                        "Loaded with differences: %d missing and %d unexpected keys in state_dict.",
                        len(miss),
                        len(unexp),
                    )
            except Exception:
                pass
            self.model_changed.emit()  # Emit signal
        except Exception:
            log.exception("Error loading model from PTH: %s", path)
            self.model = None  # Ensure model is None if loading failed
            raise  # Re-raise the exception

    def load_additional_model_file(self, path: pathlib.Path) -> None:
        """Load a model and merge it into the current model/configs.

        If there is no existing model/config, this becomes the current model.
        Otherwise, its layers/ids are shifted and weights merged in.
        """
        try:
            # Determine loader based on extension
            suffix = path.suffix.lower()
            if suffix in {".yaml", ".yml", ".json"}:
                # Build from config (YAML/JSON). JSON may be exported full-model too; SOENModelCore.build handles both.
                add_model = SOENModelCore.build(str(path))
            else:
                # Torch binary (.soen/.pth)
                add_model = SOENModelCore.load(str(path), show_logs=False)

            if self.model is None and not self.layers and not self.connections:
                # Adopt as the current model
                self._apply_model(add_model)
                self.last_loaded_path = path
                log.info("Loaded initial model from %s", path)
            else:
                # Base is current model if available; else build from configs
                if self.model is None:
                    base = SOENModelCore(
                        sim_config=self.sim_config,
                        layers_config=self.layers,
                        connections_config=self.connections,
                    )
                else:
                    base = self.model

                merged, id_maps = merge_models_with_mapping([base, add_model])
                self._apply_model(merged)
                # Update provenance: append mapping for the added model only
                # id_maps[0] corresponds to the base; id_maps[1] corresponds to add_model
                if len(id_maps) > 1:
                    self.model_provenance.append(id_maps[1])
                    self.model_provenance_labels.append(str(path))
                log.info("Merged additional model from %s (layers now: %d)", path, len(self.layers))

            self.model_changed.emit()
        except Exception:
            log.exception("Error loading additional model from %s", path)
            raise

    # ----------------- Baseline helpers -----------------
    def _resolve_baseline_model(self, mode: str | None, file_path: pathlib.Path | None) -> SOENModelCore | None:
        m = (mode or "current").lower()
        try:
            if m == "current":
                return self.model or self._last_built_model
            if m == "last_file":
                if self.last_loaded_path is None:
                    return None
                return self._load_model_from_path(self.last_loaded_path)
            if m == "choose_file":
                if file_path is None:
                    return None
                return self._load_model_from_path(file_path)
            if m == "snapshot":
                return self._baseline_snapshot_model
        except Exception as e:
            log.exception("Failed to resolve baseline '%s': %s", m, e)
            return None
        return None

    def _load_model_from_path(self, path: pathlib.Path) -> SOENModelCore:
        if path.suffix.lower() == ".json":
            from soen_toolkit.utils.model_tools import model_from_json

            return model_from_json(str(path))
        return SOENModelCore.load(str(path), show_logs=False, strict=False)

    def save_baseline_snapshot(self) -> bool:
        if self.model is None:
            return False
        self._baseline_snapshot_model = copy.deepcopy(self.model)
        # Also refresh last_built cache to this snapshot for convenience
        self._cache_last_built_model(self.model)
        return True

    def compute_preservation_preview(
        self,
        *,
        preserve_mode: str,
        freeze_layers: list[int] | None = None,
        freeze_connections: list[tuple[int, int]] | None = None,
        freeze_masks: list[tuple[int, int]] | None = None,
        baseline_mode: str | None = None,
        baseline_file: pathlib.Path | None = None,
    ) -> dict[str, list]:
        bm = baseline_mode or self.baseline_mode
        bf = baseline_file or self.baseline_file_path
        base_model = self._resolve_baseline_model(bm, bf)
        if base_model is None:
            conns = [(c.from_layer, c.to_layer) for c in self.connections]
            return {
                "preserve_layers": [],
                "reinit_layers": [cfg.layer_id for cfg in self.layers],
                "preserve_conns": [],
                "new_conns": conns,
                "preserve_masks": [],
                "new_masks": conns,
            }

        tgt_layers_by_id = {cfg.layer_id: cfg for cfg in self.layers}
        base_layers_by_id = {cfg.layer_id: cfg for cfg in base_model.layers_config}
        base_masks = getattr(base_model, "connection_masks", {})
        freeze_masks = freeze_masks or []
        if preserve_mode == "frozen_only" and freeze_masks:
            missing = [pair for pair in freeze_masks if _connection_key(*pair) not in base_masks]
            if missing:
                msg = f"freeze_masks contains pairs without masks in base model: {missing}"
                raise ValueError(msg)

        preserve_layers: list[int] = []
        reinit_layers: list[int] = []
        for lid, cfg in tgt_layers_by_id.items():
            base_cfg = base_layers_by_id.get(lid)
            same_type = base_cfg is not None and base_cfg.layer_type == cfg.layer_type
            same_dim = base_cfg is not None and base_cfg.params.get("dim") == cfg.params.get("dim")
            should_preserve = same_type and same_dim
            if preserve_mode == "frozen_only":
                should_preserve = should_preserve and (freeze_layers or []) and (lid in (freeze_layers or []))
            if preserve_mode == "none":
                should_preserve = False
            (preserve_layers if should_preserve else reinit_layers).append(lid)

        tgt_conns = [(c.from_layer, c.to_layer) for c in self.connections]
        base_conns = {(c.from_layer, c.to_layer) for c in base_model.connections_config}
        preserve_conns: list[tuple[int, int]] = []
        new_conns: list[tuple[int, int]] = []
        preserve_masks: list[tuple[int, int]] = []
        new_masks: list[tuple[int, int]] = []
        for pair in tgt_conns:
            exists_in_base = pair in base_conns
            tgt_from = tgt_layers_by_id.get(pair[0])
            base_from = base_layers_by_id.get(pair[0])
            tgt_to = tgt_layers_by_id.get(pair[1])
            base_to = base_layers_by_id.get(pair[1])
            from_dim_same = (tgt_from.params.get("dim") if tgt_from else None) == (base_from.params.get("dim") if base_from else None)
            to_dim_same = (tgt_to.params.get("dim") if tgt_to else None) == (base_to.params.get("dim") if base_to else None)
            ok = exists_in_base and from_dim_same and to_dim_same
            ok_mask = ok and _connection_key(*pair) in base_masks
            if preserve_mode == "frozen_only":
                ok = ok and (freeze_connections or []) and (pair in (freeze_connections or []))
                ok_mask = ok_mask and freeze_masks and (pair in freeze_masks)
            if preserve_mode == "none":
                ok = False
                ok_mask = False
            (preserve_conns if ok else new_conns).append(pair)
            (preserve_masks if ok_mask else new_masks).append(pair)

        return {
            "preserve_layers": sorted(preserve_layers),
            "reinit_layers": sorted(reinit_layers),
            "preserve_conns": sorted(preserve_conns),
            "new_conns": sorted(new_conns),
            "preserve_masks": sorted(preserve_masks),
            "new_masks": sorted(new_masks),
        }

    # Backwards-compat aliases ------------------------------------------------
    def save_model_pth(self, path: pathlib.Path) -> None:
        """Deprecated alias for save_model_file."""
        if path.suffix.lower() != ".pth":
            path = path.with_suffix(".pth")
        self.save_model_file(path)

    def load_model_pth(self, path: pathlib.Path) -> None:
        """Deprecated alias for load_model_file (accepts .pth)."""
        self.load_model_file(path)

    # -------- build --------
    def build_model(
        self,
        use_seed: bool = True,
        preserve_mode: str = "none",
        freeze_layers: list[int] | None = None,
        freeze_connections: list[tuple[int, int]] | None = None,
        freeze_masks: list[tuple[int, int]] | None = None,
        seed_value: int | None = None,
    ) -> None:
        """Builds the SOENModelCore instance from the current configuration.

        Args:
            use_seed: If True, seeds rngs with a fixed value for reproducible fresh init
            preserve_mode: one of {"none", "all", "frozen_only"}
                - "none": build fresh (legacy behavior)
                - "all": copy all matching parameters/weights from the existing model
                - "frozen_only": copy only for layers and connections specified in freeze_* args
            freeze_layers: list of layer IDs to preserve when preserve_mode=="frozen_only"
            freeze_connections: list of (from_layer, to_layer) pairs to preserve when
                                preserve_mode=="frozen_only"
                                use (i,i) for internal
            freeze_masks: list of (from_layer, to_layer) pairs whose masks should be preserved
                          when preserve_mode=="frozen_only"

        """
        # Get progress callback if available (set by GUI during build)
        progress_callback = getattr(self, "_build_progress_callback", None) or (lambda v, m="": None)

        # Prevent layerwise builds when feedback connections exist. Layerwise assumes
        # strictly feedforward topology, so any backwards edge results in undefined
        # behaviour. Enforce this constraint here so callers (GUI or API) receive a
        # clear error instead of a silently incorrect model.
        solver_mode = str(getattr(self.sim_config, "network_evaluation_method", "layerwise")).lower()
        if solver_mode == "layerwise":
            feedback: list[ConnectionConfig] = []
            for cfg in self.connections:
                from_layer = getattr(cfg, "from_layer", None)
                to_layer = getattr(cfg, "to_layer", None)
                if isinstance(from_layer, int) and isinstance(to_layer, int) and to_layer < from_layer:
                    feedback.append(cfg)
            if feedback:
                ids = ", ".join(f"{cfg.from_layer}→{cfg.to_layer}" for cfg in feedback[:5])
                if len(feedback) > 5:
                    ids += ", …"
                raise ValueError(
                    "Layerwise global solver requires a strictly feedforward graph. Feedback connections detected: " + ids + ". Switch to a stepwise solver.",
                )

        # --- Add seeding for deterministic initialization (conditionally) ---
        progress_callback(30, "Setting up random seeds...")
        seed = None
        if use_seed and (seed_value is not None):
            seed = int(seed_value)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            log.info(f"Set random seeds (random, numpy, torch) to {seed} for model building.")
        else:
            log.info("Building model without setting a fixed random seed.")
        # ----------------------------------------------------------------

        try:
            # Ensure layers are sorted by ID before building
            progress_callback(35, "Sorting layer configurations...")
            self.layers.sort(key=lambda x: x.layer_id)

            if preserve_mode in ("all", "frozen_only"):
                progress_callback(40, "Resolving baseline model...")
                base_model = self._resolve_baseline_model(self.baseline_mode, self.baseline_file_path)
                if base_model is not None:
                    progress_callback(50, "Rebuilding model with state preservation...")
                    # Use ID-map-aware rebuild when an explicit mapping exists
                    # (e.g., after layer insertion that shifted IDs)
                    if self.last_merge_id_map:
                        from soen_toolkit.utils.model_tools import rebuild_model_preserving_id_map
                        rebuilt = rebuild_model_preserving_id_map(
                            base_model=base_model,
                            sim_config=self.sim_config,
                            layers_config=self.layers,
                            connections_config=self.connections,
                            id_map_old_to_new=self.last_merge_id_map,
                        )
                        # Clear the map after use to avoid stale mappings
                        self.last_merge_id_map = {}
                    else:
                        rebuilt = rebuild_model_preserving_state(
                            base_model=base_model,
                            sim_config=self.sim_config,
                            layers_config=self.layers,
                            connections_config=self.connections,
                            preserve_mode=preserve_mode,
                            freeze_layers=freeze_layers,
                            freeze_connections=freeze_connections,
                            freeze_masks=freeze_masks,
                            seed=seed,
                        )
                    self._apply_model(rebuilt, cache_last_built=False)
                else:
                    progress_callback(50, "Building fresh model...")
                    built = SOENModelCore(
                        sim_config=self.sim_config,
                        layers_config=self.layers,
                        connections_config=self.connections,
                    )
                    self._apply_model(built, cache_last_built=False)
            else:
                # Fresh build (legacy behavior)
                progress_callback(50, "Building fresh model from configurations...")
                built = SOENModelCore(
                    sim_config=self.sim_config,
                    layers_config=self.layers,
                    connections_config=self.connections,
                )
                self._apply_model(built, cache_last_built=False)

            # Persist the seed used on the built model so exports can include it
            progress_callback(85, "Finalizing model...")
            try:
                if seed is not None:
                    self.model._creation_seed = int(seed)
            except Exception:
                pass
            log.info("Model built successfully – %d parameters", sum(p.numel() for p in self.model.parameters()))
            # Cache a deep copy of the built model so we can preserve from
            # "current" even after subsequent edits clear self.model
            self._cache_last_built_model(self.model)
            progress_callback(95, "Emitting model signals...")
            self.model_changed.emit()  # Emit signal
        except Exception:
            log.exception("Error building model from configuration.")
            self.model = None  # Ensure model is None if build failed
            raise  # Re-raise the exception
