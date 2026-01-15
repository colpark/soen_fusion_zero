# FILEPATH: src/soen_toolkit/model_creation_gui/tabs/tab_model_io.py
from __future__ import annotations

import contextlib
import pathlib
from typing import TYPE_CHECKING

from PyQt6.QtCore import QDir, QSettings, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from soen_toolkit.core.model_yaml import dump_model_to_yaml
from soen_toolkit.core.soen_model_core import SOENModelCore
from soen_toolkit.utils.model_tools import export_model_to_json

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager


class ModelIOTab(QWidget):
    def __init__(self, mgr: ModelManager) -> None:
        super().__init__()
        self._mgr = mgr
        # Persistent settings
        self._settings = QSettings("GreatSky", "SOEN-Toolkit")
        # Keep a session history of imported model paths for convenience
        self._import_history: list[pathlib.Path] = []
        self._ui()
        # Drag-and-drop convenience
        self.setAcceptDrops(True)
        # Restore recents
        self._load_recent_files()

    def _ui(self) -> None:
        # Use a splitter for a more responsive layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)
        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        # ----------------- LOAD PANEL -----------------
        load_group = QGroupBox("Load / Combine Models")
        load_v = QVBoxLayout(load_group)
        load_v.setSpacing(8)
        load_v.setContentsMargins(10, 12, 10, 12)

        # Hint text
        hint = QLabel("Drag a Torch (.soen/.pth) model or a YAML/JSON config here, or use Browse. Choose an action, then Load.")
        hint.setStyleSheet("color: #aaa;")
        load_v.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("File:"))
        self.load_path = QLineEdit()
        self.load_path.setPlaceholderText("Select a .pth/.soen or .json file, or drop here…")
        self.load_path.setToolTip("Full path to model file. Drag & drop works too.")
        row.addWidget(self.load_path, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._pick_load)
        row.addWidget(btn_browse)
        load_v.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Action:"))
        self.load_action = QComboBox()
        self.load_action.addItems(
            [
                "Replace current (reset)",
                "Merge as additional (ID-shift)",
            ]
        )
        self.load_action.setMinimumContentsLength(18)
        self.load_action.setMinimumWidth(240)
        self.load_action.setToolTip("Replace: discard current model and load selected. Merge: append layers with shifted IDs.")
        row2.addWidget(self.load_action)
        btn_do_load = QPushButton("Load")
        btn_do_load.setDefault(True)
        btn_do_load.setToolTip("Perform the selected action using the chosen file")
        btn_do_load.clicked.connect(self._do_load)
        row2.addWidget(btn_do_load)
        row2.addStretch()
        load_v.addLayout(row2)

        # Recent imports row
        recent_row = QHBoxLayout()
        recent_row.addWidget(QLabel("Recent:"))
        self.recent_cb = QComboBox()
        self.recent_cb.setPlaceholderText("No recent files")
        self.recent_cb.setMinimumWidth(280)
        self.recent_cb.setToolTip("Quickly re-select a recently used model")
        self.recent_cb.activated.connect(self._select_recent)
        recent_row.addWidget(self.recent_cb, 1)
        clear_hist_btn = QPushButton("Clear")
        clear_hist_btn.setToolTip("Clear the recent list for this session")
        clear_hist_btn.clicked.connect(self.clear_history)
        recent_row.addWidget(clear_hist_btn)
        load_v.addLayout(recent_row)

        # Session imports list (fills space nicely)
        self.session_list = QListWidget()
        self.session_list.setMaximumHeight(160)
        self.session_list.setToolTip("Models imported in this session. Double-click to select again.")
        self.session_list.itemDoubleClicked.connect(self._use_history_item)
        load_v.addWidget(self.session_list)

        splitter.addWidget(load_group)

        # ----------------- EXPORT PANEL -----------------
        export_group = QGroupBox("Export / Save Model")
        exp_v = QVBoxLayout(export_group)
        exp_v.setSpacing(8)
        exp_v.setContentsMargins(10, 12, 10, 12)

        rowe0 = QHBoxLayout()
        rowe0.addWidget(QLabel("Export type:"))
        self.export_type = QComboBox()
        self.export_type.addItems([
            "Torch (full model)",
            "JSON (config + weights)",
            "YAML (config only)",
        ])
        self.export_type.setMinimumWidth(200)
        self.export_type.setToolTip("Choose save format. Torch is a single-file binary (.soen/.pth).")
        rowe0.addWidget(self.export_type)
        self.export_type.currentIndexChanged.connect(self._on_export_type_changed)
        exp_v.addLayout(rowe0)

        # Intentionally no rebuild/preservation controls here to avoid confusion.
        # Use Model Builder tab for building with preservation.

        # Destination/path
        rowe2 = QHBoxLayout()
        rowe2.addWidget(QLabel("Save to:"))
        self.export_path = QLineEdit()
        self.export_path.setPlaceholderText("Choose an output path…")
        self.export_path.setToolTip("Destination path for saving. If blank after a load, a suggestion will appear here.")
        rowe2.addWidget(self.export_path, 1)
        btn_pick_out = QPushButton("Choose…")
        btn_pick_out.setToolTip("Pick a file path to save to")
        btn_pick_out.clicked.connect(self._pick_export)
        rowe2.addWidget(btn_pick_out)
        exp_v.addLayout(rowe2)

        # Model summary/info label
        self.model_info_lbl = QLabel("")
        self.model_info_lbl.setWordWrap(True)
        exp_v.addWidget(self.model_info_lbl)

        # Collapsible preview tree (used for JSON and Torch views)
        self.json_tree = QTreeWidget()
        self.json_tree.setHeaderHidden(True)
        self.json_tree.setVisible(False)
        self.json_tree.setMinimumHeight(200)
        exp_v.addWidget(self.json_tree)

        self.btn_export = QPushButton("Export")
        self.btn_export.setToolTip("Save the current model in the selected format")
        self.btn_export.clicked.connect(self._do_export)
        exp_v.addWidget(self.btn_export)

        splitter.addWidget(export_group)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter, 1)

        # Initialize with current state
        self._refresh_freeze_lists()
        self._update_export_enabled()
        self._refresh_preview()
        # Show initial model information
        with contextlib.suppress(Exception):
            self._update_model_info()

    # ----------------- helpers -----------------
    def _pick_load(self) -> None:
        # Choose a fast starting directory: current text, last loaded, or CWD
        try:
            start = pathlib.Path(self.load_path.text().strip()) if self.load_path.text().strip() else None
            if not start and hasattr(self._mgr, "last_loaded_path") and self._mgr.last_loaded_path:
                start = pathlib.Path(str(self._mgr.last_loaded_path))
            if not start:
                start = pathlib.Path.cwd()
            if start.is_file():
                start = start.parent
            start_dir = str(start)
        except Exception:
            start_dir = QDir.homePath()

        # Use Qt's non-native dialog to avoid macOS Finder slowness in large dirs
        opts = QFileDialog.Option(0)  # Start with no options
        try:
            opts |= QFileDialog.Option.DontUseNativeDialog
            opts |= QFileDialog.Option.DontResolveSymlinks
            opts |= QFileDialog.Option.ReadOnly
            # Avoid custom icon lookups which can be slow on network/iCloud dirs
            with contextlib.suppress(Exception):
                opts |= QFileDialog.Option.DontUseCustomDirectoryIcons
        except Exception:
            pass

        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select model file",
            start_dir,
            "All supported (*.soen *.pth *.json *.yaml *.yml);;Torch Model (*.soen *.pth);;YAML/JSON Config (*.yaml *.yml *.json);;All files (*)",
            options=opts,
        )
        if fname:
            self.load_path.setText(fname)

    def _do_load(self) -> None:
        path = self.load_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Load", "Please select a file to load.")
            return
        action = self.load_action.currentText()
        try:
            p = pathlib.Path(path)
            if action.startswith("Replace"):
                # Confirm reset
                resp = QMessageBox.question(
                    self,
                    "Replace Current",
                    "This will replace the current configuration and model. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if resp != QMessageBox.StandardButton.Yes:
                    return
                # Accept config-only YAML/JSON as well as full Torch saves
                suffix = p.suffix.lower()
                if suffix in (".yaml", ".yml", ".json"):
                    # Delegate to unified builder – handles YAML and JSON (exported JSON or minimal spec)
                    built_model = SOENModelCore.build(str(p))
                    # Extract seed from spec (top-level or simulation) and prime GUI controls
                    try:
                        import yaml as _yaml

                        data_text = p.read_text()
                        spec = _yaml.safe_load(data_text) or {}
                        cfg_seed = None
                        if isinstance(spec, dict):
                            cfg_seed = spec.get("seed")
                            if cfg_seed is None:
                                sim_block = spec.get("simulation", spec.get("sim_config", {})) or {}
                                cfg_seed = sim_block.get("seed")
                        if cfg_seed is not None:
                            try:
                                cfg_seed = int(cfg_seed)
                            except Exception:
                                cfg_seed = None
                        # Prime the GUI seed toggle/value with spec seed by default
                        from soen_toolkit.model_creation_gui.main_window import (
                            MainWindow,
                        )

                        w = self.window()
                        if isinstance(w, MainWindow) and hasattr(w, "build_tab"):
                            if cfg_seed is not None:
                                w.build_tab.seed_checkbox.setChecked(True)
                                w.build_tab.seed_value_edit.setText(str(cfg_seed))
                            else:
                                # No seed in spec: leave controls untouched
                                pass
                    except Exception:
                        pass
                    # If the GUI has an explicit seed set, persist it; else keep the spec's seed
                    try:
                        # Find the build tab by scanning parent window's tabs if available
                        parent = self.parent()
                        seed_val = None
                        use_seed = False
                        if parent and hasattr(parent, "parent"):
                            pass
                        # Best-effort: if main window is accessible, sync seed
                        try:
                            from soen_toolkit.model_creation_gui.main_window import (
                                MainWindow,
                            )

                            mw = None
                            w = self.window()
                            if isinstance(w, MainWindow):
                                mw = w
                            if mw is not None and hasattr(mw, "build_tab"):
                                use_seed = mw.build_tab.seed_checkbox.isChecked()
                                txt = mw.build_tab.seed_value_edit.text().strip()
                                if use_seed and txt:
                                    seed_val = int(txt)
                        except Exception:
                            pass
                        if use_seed and seed_val is not None:
                            built_model._creation_seed = int(seed_val)
                    except Exception:
                        pass
                    self._mgr.model = built_model
                    self._mgr.sim_config = built_model.sim_config
                    self._mgr.layers = built_model.layers_config
                    self._mgr.connections = built_model.connections_config
                    self._mgr.last_loaded_path = p
                    self._mgr.model_changed.emit()
                else:
                    # Torch binary (.soen/.pth)
                    self._mgr.load_model_file(p)
            else:
                self._mgr.load_additional_model_file(p)
            QMessageBox.information(self, "Load", "Model load/merge completed.")
            self._update_export_enabled()
            with contextlib.suppress(Exception):
                self._update_model_info()
            with contextlib.suppress(Exception):
                self._refresh_preview()
            # Track in history (deduplicate, show only name for clarity)
            with contextlib.suppress(Exception):
                self._add_to_history(p)
            # Suggest export location on first successful load
            with contextlib.suppress(Exception):
                self._maybe_suggest_export_path(p)
        except Exception as e:
            # Provide a clearer error when a non-model JSON (e.g. Optuna report) is selected
            try:
                emsg = str(e)
                p = pathlib.Path(path)
                if p.suffix.lower() == ".json" and ("Unrecognized JSON schema" in emsg or emsg.strip() == "'simulation'" or "'simulation'/'layers'/'connections'" in emsg):
                    emsg = (
                        "The selected JSON does not look like a SOEN model or architecture spec.\n\n"
                        "Expected either:\n"
                        "  • Full model: .soen/.pth or exported JSON (with connections.config/matrices)\n"
                        "  • Spec (config): YAML/JSON with top-level 'simulation', 'layers', 'connections'\n\n"
                        f"File: {p}"
                    )
                QMessageBox.critical(self, "Load failed", emsg)
            except Exception:
                QMessageBox.critical(self, "Load failed", str(e))

    # Back-compat: no-op hooks (main window may connect these)
    def _toggle_freeze_area(self, mode: str) -> None:
        pass

    def _refresh_freeze_lists(self) -> None:
        # Nothing to refresh now (preservation removed from this tab)
        # Keep basic info updated and ensure export button reflects model state
        with contextlib.suppress(Exception):
            self._update_model_info()
        with contextlib.suppress(Exception):
            self._refresh_preview()
        with contextlib.suppress(Exception):
            self._update_export_enabled()

    def _pick_export(self) -> None:
        if self.export_type.currentIndex() == 0:
            fname, _ = QFileDialog.getSaveFileName(
                self,
                "Save Torch Model",
                QDir.homePath(),
                "Torch Model (*.soen *.pth)",
            )
            if fname:
                p = pathlib.Path(fname)
                if p.suffix.lower() not in (".pth", ".soen"):
                    p = p.with_suffix(".soen")
                fname = str(p)
        elif self.export_type.currentIndex() == 1:
            fname, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model JSON",
                QDir.homePath(),
                "JSON (*.json)",
            )
            if fname:
                p = pathlib.Path(fname)
                if p.suffix.lower() != ".json":
                    p = p.with_suffix(".json")
                fname = str(p)
        elif self.export_type.currentIndex() == 2:
            fname, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model YAML",
                QDir.homePath(),
                "YAML (*.yaml *.yml)",
            )
            if fname:
                p = pathlib.Path(fname)
                if p.suffix.lower() not in (".yaml", ".yml"):
                    p = p.with_suffix(".yaml")
                fname = str(p)
        if fname:
            self.export_path.setText(fname)

    def _collect_freezes(self):
        layers = []
        for i in range(self.freeze_layer_list.count()):
            it = self.freeze_layer_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                layers.append(int(it.data(Qt.ItemDataRole.UserRole)))
        conns = []
        for i in range(self.freeze_conn_list.count()):
            it = self.freeze_conn_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                pair = it.data(Qt.ItemDataRole.UserRole)
                conns.append((int(pair[0]), int(pair[1])))
        return layers, conns

    def _do_export(self) -> None:
        path = self.export_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Export", "Please choose an output path.")
            return
        if self._mgr.model is None:
            QMessageBox.warning(self, "Export", "No built model to export. Build or load a model first.")
            return
        try:
            # Save
            idx = self.export_type.currentIndex()
            p = pathlib.Path(path)
            if idx == 0:
                if p.suffix.lower() not in (".pth", ".soen"):
                    p = p.with_suffix(".soen")
                self._mgr.save_model_file(p)
            elif idx == 1:
                if p.suffix.lower() != ".json":
                    p = p.with_suffix(".json")
                export_model_to_json(self._mgr.model, str(p))
            elif idx == 2:
                if p.suffix.lower() not in (".yaml", ".yml"):
                    p = p.with_suffix(".yaml")
                # Ensure seed from GUI is captured before dumping
                try:
                    from soen_toolkit.model_creation_gui.main_window import MainWindow

                    w = self.window()
                    if isinstance(w, MainWindow) and w.build_tab.seed_checkbox.isChecked():
                        txt = w.build_tab.seed_value_edit.text().strip()
                        if txt:
                            self._mgr.model._creation_seed = int(txt)
                except Exception:
                    pass
                dump_model_to_yaml(self._mgr.model, str(p))
            QMessageBox.information(self, "Export", "Export completed.")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    # ------------- drag-and-drop -------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        local = urls[0].toLocalFile()
        if local:
            self.load_path.setText(local)
            self._do_load()
        super().dropEvent(event)

    # ------------- helpers -------------
    def _update_export_enabled(self) -> None:
        has_model = self._mgr.model is not None
        # Enable/disable export controls based on whether a model exists
        self.export_type.setEnabled(True)
        self.export_path.setEnabled(True)
        if hasattr(self, "btn_export"):
            self.btn_export.setEnabled(has_model)
            if not has_model:
                self.btn_export.setToolTip("Build or load a model to enable export")
            else:
                self.btn_export.setToolTip("")

    def _on_export_type_changed(self, index: int) -> None:
        # Toggle preview visibility when switching export type
        self.json_tree.setVisible(True)
        # Live-normalize the export path extension to match the selected type
        try:
            path_txt = self.export_path.text().strip()
            if path_txt:
                p = pathlib.Path(path_txt)
                if index == 0:
                    if p.suffix.lower() not in (".pth", ".soen"):
                        p = p.with_suffix(".soen")
                elif index == 1:
                    if p.suffix.lower() != ".json":
                        p = p.with_suffix(".json")
                elif index == 2:
                    if p.suffix.lower() not in (".yaml", ".yml"):
                        p = p.with_suffix(".yaml")
                else:
                    pass
                self.export_path.setText(str(p))
        except Exception:
            pass
        with contextlib.suppress(Exception):
            self._refresh_preview()

    def _update_model_info(self) -> None:
        try:
            if self._mgr.model is None:
                self.model_info_lbl.setText("No model loaded/built yet.")
                return
            num_layers = len(self._mgr.layers)
            num_conns = len(self._mgr.connections)
            params = sum(p.numel() for p in self._mgr.model.parameters())
            self.model_info_lbl.setText(
                f"Current model: {num_layers} layers, {num_conns} connections, {params} parameters",
            )
        except Exception:
            self.model_info_lbl.setText("")

    def _refresh_preview(self) -> None:
        """Populate the collapsible tree for the selected export type."""
        try:
            if self._mgr.model is None:
                self.json_tree.clear()
                self.json_tree.setVisible(True)
                self.json_tree.addTopLevelItem(QTreeWidgetItem(["No model built or loaded"]))
                return

            self.json_tree.clear()
            if self.export_type.currentIndex() in (1, 2):
                # JSON/YAML view (structure-only preview: no parameters/matrices)
                data = self._make_json_structure_preview()
                # Augment preview with seed if available
                try:
                    seed_val = getattr(self._mgr.model, "_creation_seed", None)
                    if seed_val is not None:
                        if isinstance(data, dict):
                            data.setdefault("simulation", {})
                            data.setdefault("metadata", {})
                            data["metadata"]["seed"] = int(seed_val)
                except Exception:
                    pass
                self._add_obj_to_tree(self.json_tree.invisibleRootItem(), data)
            else:
                # Torch view (structure summary)
                data = self._make_torch_preview_dict()
                self._add_obj_to_tree(self.json_tree.invisibleRootItem(), data)

            self.json_tree.expandToDepth(0)
            self.json_tree.setVisible(True)
        except Exception:
            # Keep UI resilient; just hide preview on errors
            self.json_tree.setVisible(False)

    def _add_obj_to_tree(self, parent: QTreeWidgetItem, obj) -> None:
        """Recursively add dict/list/scalars to a QTreeWidget."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                node = QTreeWidgetItem([str(k)])
                parent.addChild(node)
                self._add_obj_to_tree(node, v)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                node = QTreeWidgetItem([f"[{idx}]"])
                parent.addChild(node)
                self._add_obj_to_tree(node, v)
        else:
            leaf = QTreeWidgetItem([str(obj)])
            parent.addChild(leaf)

    def _make_torch_preview_dict(self) -> dict:
        """Create a compact nested dict describing the PyTorch model structure."""
        m = self._mgr.model
        layers = {}
        try:
            for idx, layer in enumerate(getattr(m, "layers", [])):
                layer_info = {
                    "type": layer.__class__.__name__,
                }
                # Include direct child modules if any
                try:
                    children = [name for name, _ in layer.named_children()]
                    if children:
                        layer_info["children"] = children
                except Exception:
                    pass
                layers[str(idx)] = layer_info
        except Exception:
            pass

        conns = {}
        try:
            for name, param in getattr(m, "connections", {}).items():
                try:
                    shape = list(param.shape)
                except Exception:
                    shape = "?"
                conns[name] = {"shape": shape}
        except Exception:
            pass

        return {
            "SOENModelCore": {
                "layers": layers,
                "connections": conns,
            },
        }


    def _make_json_structure_preview(self) -> dict:
        """Create a lightweight JSON-like dict with only structural info.

        Excludes all tensor data: no layer parameters, buffers, connection matrices,
        or global matrices. Suitable for instant preview even on large models.
        """
        try:
            sim = dict(vars(self._mgr.model.sim_config)) if getattr(self._mgr, "model", None) is not None else {}
        except Exception:
            sim = {}

        # Layers: from configs only
        layers = {}
        try:
            for lc in getattr(self._mgr, "layers", []) or []:
                layers[f"layer_{lc.layer_id}"] = {
                    "config": {
                        "type": lc.layer_type,
                        "params_config": dict(lc.params) if lc.params is not None else {},
                        "model_id": int(getattr(lc, "model_id", 0)),
                    },
                    # Keep noise structure shallow (values are small)
                    "noise": (vars(getattr(lc, "noise", None)) if getattr(lc, "noise", None) is not None else {}),
                }
        except Exception:
            pass

        # Connections: config only
        conns_cfg = {}
        try:
            for cc in getattr(self._mgr, "connections", []) or []:
                # Compute expected shape from dims if available (optional, cheap)
                shape = None
                try:
                    dims_by_id = {c.layer_id: c.params.get("dim", 0) for c in (getattr(self._mgr, "layers", []) or [])}
                    if cc.from_layer == cc.to_layer:
                        d = int(dims_by_id.get(cc.from_layer, 0))
                        shape = [d, d]
                    else:
                        d_from = int(dims_by_id.get(cc.from_layer, 0))
                        d_to = int(dims_by_id.get(cc.to_layer, 0))
                        shape = [d_to, d_from]
                except Exception:
                    shape = None

                conns_cfg_key = f"J_{cc.from_layer}_to_{cc.to_layer}" if cc.from_layer != cc.to_layer else f"internal_{cc.from_layer}"
                entry = {
                    "from": cc.from_layer,
                    "to": cc.to_layer,
                    "type": cc.connection_type,
                    "params_config": dict(cc.params) if cc.params is not None else {},
                    "learnable": bool(cc.learnable),
                }
                if shape is not None:
                    entry["expected_shape"] = shape
                conns_cfg[conns_cfg_key] = entry
        except Exception:
            pass

        return {
            "version": "1.0",
            "metadata": {
                "format": "soen-json-structure-preview",
            },
            "simulation": sim,
            "layers": layers,
            "connections": {
                "config": conns_cfg,
            },
        }

    # ---------- history helpers ----------
    def _load_recent_files(self) -> None:
        """Load recent file list from persistent settings."""
        try:
            # QSettings stores lists as lists of strings
            recents = self._settings.value("recent_model_files", [], type=list)
            # Convert back to Path objects, filtering out invalids
            self._import_history = []
            for r in recents:
                try:
                    p = pathlib.Path(str(r))
                    if p.exists():
                        self._import_history.append(p)
                except Exception:
                    pass
            # Refresh UI
            self._refresh_history_ui()
        except Exception:
            pass

    def _save_recent_files(self) -> None:
        """Save recent file list to persistent settings."""
        try:
            # Convert Path objects to strings for storage
            recents = [str(p) for p in self._import_history]
            self._settings.setValue("recent_model_files", recents)
        except Exception:
            pass

    def _refresh_history_ui(self) -> None:
        """Refresh the recent files combo box and list widget."""
        self.recent_cb.blockSignals(True)
        self.recent_cb.clear()
        for q in self._import_history:
            label = f"{q.name} (…/{q.parent.name})"
            self.recent_cb.addItem(label, str(q))
        self.recent_cb.blockSignals(False)

        try:
            self.session_list.clear()
            for q in self._import_history:
                label = f"{q.name}  —  …/{q.parent.name}"
                it = QListWidgetItem(label)
                it.setData(Qt.ItemDataRole.UserRole, str(q))
                self.session_list.addItem(it)
        except Exception:
            pass

    def _add_to_history(self, p: pathlib.Path) -> None:
        with contextlib.suppress(Exception):
            p = p.resolve()
        # Remove if exists and append to front
        self._import_history = [q for q in self._import_history if q != p]
        self._import_history.insert(0, p)
        # Trim to last 10 entries
        self._import_history = self._import_history[:10]

        # Update UI and persist
        self._refresh_history_ui()
        self._save_recent_files()

    def _select_recent(self, index: int) -> None:
        path_str = self.recent_cb.itemData(index)
        if path_str:
            self.load_path.setText(path_str)

    def _use_history_item(self, item: QListWidgetItem) -> None:
        try:
            path_str = item.data(Qt.ItemDataRole.UserRole)
            if path_str:
                self.load_path.setText(path_str)
        except Exception:
            pass

    def clear_history(self) -> None:
        self._import_history.clear()
        self._refresh_history_ui()
        self._save_recent_files()
        self.recent_cb.setPlaceholderText("No recent files")

    # ---------- suggestions ----------
    def _maybe_suggest_export_path(self, src_path: pathlib.Path) -> None:
        """If export path is blank, suggest saving alongside the last loaded file."""
        if not self.export_path.text().strip():
            try:
                suggested = src_path.with_suffix(".soen")
                self.export_path.setText(str(suggested))
            except Exception:
                pass
