# FILEPATH: src/soen_toolkit/model_creation_gui/tabs/tab_model_building.py
from __future__ import annotations

import contextlib
import copy
import itertools
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QDoubleValidator, QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from soen_toolkit.core import SOENModelCore
from soen_toolkit.core.configs import ConnectionConfig, LayerConfig
from soen_toolkit.model_creation_gui.components.collapsible_section import (
    CollapsibleSection,
)
from soen_toolkit.model_creation_gui.components.connection_config_dialog import (
    ConnectionConfigDialog,
)
from soen_toolkit.model_creation_gui.components.layer_config_dialog import (
    LayerConfigDialog,
)
from soen_toolkit.model_creation_gui.components.merge_layers_dialog import (
    MergeLayersDialog,
)
from soen_toolkit.model_creation_gui.components.quantize_dialog import QuantizeDialog
from soen_toolkit.model_creation_gui.utils.paths import icon
from soen_toolkit.model_creation_gui.utils.ui_helpers import (
    create_double_spinbox,
    show_error,
    show_info,
    show_warning,
)
from soen_toolkit.utils.merge_layers import (
    MergeSpec,
    apply_merge_layers,
    topology_intel,
)
from soen_toolkit.utils.model_tools import rebuild_model_preserving_id_map

if TYPE_CHECKING:
    from soen_toolkit.model_creation_gui.model_manager import ModelManager


class ModelBuildingTab(QWidget):
    def __init__(self, mgr: ModelManager) -> None:
        super().__init__()
        self._mgr = mgr
        self._spin_bindings: list[tuple[object, str, object, object]] = []
        self._tracking_controls: list[tuple[str, QCheckBox]] = []
        self._ui()

    def _bind_spinbox(self, widget, attr: str, default, cast, on_change=None) -> None:
        value = getattr(self._mgr.sim_config, attr, default)
        try:
            widget.setValue(cast(value))
        except Exception:
            widget.setValue(default)

        def _setter(val) -> None:
            casted = cast(val)
            setattr(self._mgr.sim_config, attr, casted)
            if on_change is not None:
                on_change(casted)

        widget.valueChanged.connect(_setter)
        self._spin_bindings.append((widget, attr, cast, default))

    def _init_tracking_checkbox(self, checkbox: QCheckBox, attr: str) -> None:
        checkbox.setChecked(bool(getattr(self._mgr.sim_config, attr, False)))
        checkbox.stateChanged.connect(self._on_tracking_changed)
        self._tracking_controls.append((attr, checkbox))

    def _ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Simulation Settings (Collapsible) ---
        sim_panel = QWidget()
        sim_layout = QVBoxLayout(sim_panel)
        sim_layout.setContentsMargins(0, 0, 0, 0)
        sim_layout.setSpacing(6)

        settings_row = QHBoxLayout()

        # dt control
        lbl_dt = QLabel("dt:")
        lbl_dt.setToolTip(
            "Dimensionless time step used by solvers. Physical Δt seconds = dt / ω_c (rad/s).\nPlots display time in ns using Δt_ns = (dt / ω_c)·1e9.",
        )
        settings_row.addWidget(lbl_dt)
        self.dt_input = QLineEdit()
        self.dt_input.setToolTip("Enter a positive number. Scientific notation is supported, e.g. 7.79e1")
        dt_validator = QDoubleValidator()
        dt_validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        dt_validator.setBottom(1e-9)
        self.dt_input.setValidator(dt_validator)
        self.dt_input.setText(str(37))
        try:
            self._mgr.sim_config.dt = float(self.dt_input.text())
        except ValueError:
            self._mgr.sim_config.dt = 37
        self.dt_input.editingFinished.connect(self._on_dt_changed)
        settings_row.addWidget(self.dt_input)

        # input mode control
        lbl_im = QLabel("Input mode:")
        lbl_im.setToolTip("flux: external inputs add to g; state: inputs directly set the state variable")
        settings_row.addWidget(lbl_im)
        self.input_type_cb = QComboBox()
        self.input_type_cb.setToolTip("Choose how external inputs are interpreted during simulation")
        self.input_type_cb.addItems(["flux", "state"])
        self.input_type_cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.input_type_cb.setMinimumContentsLength(6)
        self.input_type_cb.setMinimumWidth(100)
        current_mode = getattr(self._mgr.sim_config, "input_type", "flux")
        self.input_type_cb.setCurrentText(current_mode)
        self.input_type_cb.currentTextChanged.connect(
            lambda t: setattr(self._mgr.sim_config, "input_type", t),
        )
        settings_row.addWidget(self.input_type_cb)

        # global solver control
        lbl_solver = QLabel("Global solver:")
        lbl_solver.setToolTip(
            "Choose network update strategy. Layerwise runs each layer over the full sequence in order. "
            "Stepwise (Gauss–Seidel) advances all layers one time step using freshest values (feedback allowed, 1×Δt on backward edges). "
            "Stepwise (Jacobi) advances using the previous-step snapshot (fully parallelizable across layers; all edges incur 1×Δt lag).",
        )
        settings_row.addWidget(lbl_solver)
        self.solver_cb = QComboBox()
        self.solver_cb.clear()
        self.solver_cb.addItems(["Layerwise", "Stepwise (Gauss–Seidel)", "Stepwise (Jacobi)"])
        self.solver_cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.solver_cb.setMinimumContentsLength(10)
        self.solver_cb.setMinimumWidth(200)
        # Reflect current global solver
        _gs = str(getattr(self._mgr.sim_config, "network_evaluation_method", "layerwise")).lower()
        if _gs == "stepwise_gauss_seidel":
            self.solver_cb.setCurrentIndex(1)
        elif _gs == "stepwise_jacobi":
            self.solver_cb.setCurrentIndex(2)
        else:
            self.solver_cb.setCurrentIndex(0)

        def _on_solver_changed(text: str) -> None:
            lower = text.lower()
            if "gauss" in lower:
                self._mgr.sim_config.network_evaluation_method = "stepwise_gauss_seidel"
            elif "jacobi" in lower:
                self._mgr.sim_config.network_evaluation_method = "stepwise_jacobi"
            else:
                self._mgr.sim_config.network_evaluation_method = "layerwise"
            # Enforce early-stop availability based on solver mode
            self._update_early_stop_enabled()

        self.solver_cb.currentTextChanged.connect(_on_solver_changed)
        settings_row.addWidget(self.solver_cb)

        # Early stopping controls (stepwise solvers only)
        self.es_chk = QCheckBox("Early stop")
        self.es_chk.setToolTip(
            "Enable early termination of stepwise forward passes once steady state is detected.\n"
            "When enabled, simulation stops early using either windowed or per-step criteria (see below).\n"
            "Not supported for Layerwise solver.",
        )
        try:
            self.es_chk.setChecked(bool(getattr(self._mgr.sim_config, "early_stopping_forward_pass", False)))
        except Exception:
            self.es_chk.setChecked(False)
        self.es_chk.toggled.connect(lambda v: setattr(self._mgr.sim_config, "early_stopping_forward_pass", bool(v)))
        self.es_chk.toggled.connect(lambda _v: self._update_early_stop_enabled())
        settings_row.addWidget(self.es_chk)
        # Early-stop options panel (hidden unless Early stop is enabled)
        self.es_opts_panel = QWidget()
        es_opts_row = QHBoxLayout(self.es_opts_panel)
        es_opts_row.setContentsMargins(0, 0, 0, 0)
        es_opts_row.setSpacing(6)

        lbl_tol = QLabel("Tolerance:")
        lbl_tol.setToolTip("Legacy per-step threshold: stop when max|Δs| ≤ Tolerance for 'Patience' consecutive steps.")
        es_opts_row.addWidget(lbl_tol)
        self.es_tol = create_double_spinbox(min_val=0.0, max_val=1e9, decimals=12, step=1e-6, default=1e-6)
        self.es_tol.setToolTip("Legacy per-step absolute threshold on max |Δs| across all layers.")
        self._bind_spinbox(self.es_tol, "early_stopping_tolerance", 1e-6, float)
        es_opts_row.addWidget(self.es_tol)

        lbl_pat = QLabel("Patience:")
        lbl_pat.setToolTip("Legacy per-step: number of consecutive steps satisfying Tolerance before stopping.")
        es_opts_row.addWidget(lbl_pat)
        self.es_pat = QSpinBox()
        self.es_pat.setRange(1, 1_000_000)
        self.es_pat.setToolTip("Legacy per-step: consecutive steps below Tolerance required to stop.")
        self._bind_spinbox(self.es_pat, "early_stopping_patience", 1, int)
        es_opts_row.addWidget(self.es_pat)

        lbl_min = QLabel("Min steps:")
        lbl_min.setToolTip("Always simulate at least this many steps before applying early-stop checks.")
        es_opts_row.addWidget(lbl_min)
        self.es_min = QSpinBox()
        self.es_min.setRange(0, 10_000_000)
        self.es_min.setToolTip("Minimum number of steps before early-stop checks begin (applies to both modes).")
        self._bind_spinbox(self.es_min, "early_stopping_min_steps", 1, int)
        es_opts_row.addWidget(self.es_min)

        # Windowed steady-state controls
        lbl_win = QLabel("Window:")
        lbl_win.setToolTip("Trailing window size (steps). If > 0, use windowed detection: max|Δs|_window ≤ Abs tol + Rel tol × max|s|_window.")
        es_opts_row.addWidget(lbl_win)
        self.es_win = QSpinBox()
        self.es_win.setRange(0, 10_000_000)
        self.es_win.setToolTip("If > 0, enable windowed detection using Abs/Rel tolerances.")
        self._bind_spinbox(
            self.es_win,
            "steady_window_min",
            50,
            int,
            on_change=lambda _v: self._update_early_stop_enabled(),
        )
        es_opts_row.addWidget(self.es_win)

        lbl_abs = QLabel("Abs tol:")
        lbl_abs.setToolTip("Windowed mode: absolute component of threshold. Used with Rel tol.")
        es_opts_row.addWidget(lbl_abs)
        self.es_abs = create_double_spinbox(min_val=0.0, max_val=1e9, decimals=12, step=1e-6, default=1e-5)
        self.es_abs.setToolTip("Windowed mode: threshold = Abs tol + Rel tol × max|s| over the window.")
        self._bind_spinbox(self.es_abs, "steady_tol_abs", 1e-5, float)
        es_opts_row.addWidget(self.es_abs)

        lbl_rel = QLabel("Rel tol:")
        lbl_rel.setToolTip("Windowed mode: relative component (unitless factor) applied to max|s|.")
        es_opts_row.addWidget(lbl_rel)
        self.es_rel = create_double_spinbox(min_val=0.0, max_val=1.0, decimals=12, step=1e-4, default=1e-3)
        self.es_rel.setToolTip("Windowed mode: threshold = Abs tol + Rel tol × max|s| over the window.")
        self._bind_spinbox(self.es_rel, "steady_tol_rel", 1e-3, float)
        es_opts_row.addWidget(self.es_rel)

        # Assemble sim panel
        settings_row.addStretch()
        sim_layout.addLayout(settings_row)
        sim_layout.addWidget(self.es_opts_panel)

        # Add tracking row (to compact the header)
        tracking_row = QHBoxLayout()
        tracking_row.setSpacing(20)
        tracking_row.addWidget(QLabel("Tracking:"))
        tracking_row.addSpacing(10)
        self.track_phi_chk = QCheckBox("φ (phi)")
        self.track_phi_chk.setToolTip("Record φ(t) per layer during simulation for later analysis/plotting")
        self._init_tracking_checkbox(self.track_phi_chk, "track_phi")
        tracking_row.addWidget(self.track_phi_chk)

        self.track_g_chk = QCheckBox("g (source)")
        self.track_g_chk.setToolTip("Record g(t) source signal per layer")
        self._init_tracking_checkbox(self.track_g_chk, "track_g")
        tracking_row.addWidget(self.track_g_chk)

        self.track_power_chk = QCheckBox("Power")
        self.track_power_chk.setToolTip("Record instantaneous power estimates if supported")
        self._init_tracking_checkbox(self.track_power_chk, "track_power")
        tracking_row.addWidget(self.track_power_chk)

        self.track_s_chk = QCheckBox("State s")
        self.track_s_chk.setToolTip("Record neuron state s(t); can be memory intensive for large models")
        self._init_tracking_checkbox(self.track_s_chk, "track_s")
        tracking_row.addWidget(self.track_s_chk)
        tracking_row.addStretch()
        sim_layout.addLayout(tracking_row)

        # Wrap in collapsible section
        sim_section = CollapsibleSection("General Simulation Settings", collapsed=False)
        sim_section.setToolTip("Global simulation parameters affecting the forward pass and tracking.")
        sim_section.setContent(sim_panel)

        # Create a vertical splitter to allow users to resize
        outer_splitter = QSplitter(Qt.Orientation.Vertical)
        outer_splitter.setChildrenCollapsible(False)
        outer_splitter.addWidget(sim_section)

        # Initialize availability of early-stop controls
        self._update_early_stop_enabled()

        # --- Model Specification (Layers + Connections) ---
        # Build the lower pane (Model Specification)
        spec_root = QWidget()
        spec_layout = QVBoxLayout(spec_root)
        spec_layout.setContentsMargins(0, 0, 0, 0)
        title_spec = QLabel("<b>Model Specification</b>")
        title_spec.setToolTip("Define layers and connections that make up your model")
        spec_layout.addWidget(title_spec)
        # Splitter for Layers and Connections
        splitter = QSplitter()

        # Left panel: Layers
        layers_panel = QWidget()
        layers_layout = QVBoxLayout(layers_panel)
        lbl_layers = QLabel("<b>Layers</b>")
        lbl_layers.setToolTip("Double‑click a layer to edit. IDs must be unique and determine ordering.")
        layers_layout.addWidget(lbl_layers)

        # Layers toolbar
        layers_toolbar = QHBoxLayout()
        btn_add_layer = QPushButton(QIcon(icon("plus")), "Add Layer…")
        btn_add_layer.setToolTip("Create a new layer with a chosen type and parameters")
        btn_add_layer.clicked.connect(self._add_layer)

        btn_merge = QPushButton("Merge Layers…")
        btn_merge.setToolTip("Merge multiple same-type layers into a single super-layer. Node-wise params only.")
        btn_merge.clicked.connect(self._open_merge_dialog)

        btn_auto_merge = QPushButton("Auto-merge…")
        btn_auto_merge.setToolTip(
            "Automatically merge same-type layers. Options: include/exclude input and output; optionally avoid cycles (contiguous-only).",
        )
        btn_auto_merge.clicked.connect(self._auto_merge_layers)

        self.btn_dup = QPushButton("Duplicate Layer")
        self.btn_dup.setToolTip("Duplicate the selected layer using the next available ID")
        self.btn_dup.clicked.connect(self._duplicate_layer)
        self.btn_dup.setEnabled(False)

        btn_sort = QPushButton("Sort by ID")
        btn_sort.setToolTip("Order layers by ascending ID for clarity")
        btn_sort.clicked.connect(self._sort_layers_by_id)

        btn_normalize = QPushButton("Normalize IDs")
        btn_normalize.setToolTip("Renumber layers sequentially (0..N) by current sort; updates connections.")
        btn_normalize.clicked.connect(self._normalize_layer_ids)

        layers_toolbar.addWidget(btn_add_layer)
        layers_toolbar.addWidget(self.btn_dup)
        layers_toolbar.addWidget(btn_merge)
        layers_toolbar.addWidget(btn_auto_merge)
        layers_toolbar.addStretch()
        layers_toolbar.addWidget(btn_sort)
        layers_toolbar.addWidget(btn_normalize)
        layers_layout.addLayout(layers_toolbar)

        self.layer_list = QListWidget()
        self.layer_list.setStyleSheet("border: 2px solid black;")
        self.layer_list.setAlternatingRowColors(True)
        self.layer_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_list.customContextMenuRequested.connect(self._show_layers_menu)
        self.layer_list.itemDoubleClicked.connect(self._edit_layer)
        # Enable duplicate button only when a layer is selected
        with contextlib.suppress(Exception):
            self.layer_list.itemSelectionChanged.connect(
                lambda: self.btn_dup.setEnabled(self.layer_list.currentRow() >= 0),
            )
        layers_layout.addWidget(self.layer_list, 1)
        splitter.addWidget(layers_panel)

        # Right panel: Connections
        conns_panel = QWidget()
        conns_layout = QVBoxLayout(conns_panel)
        lbl_conns = QLabel("<b>Connections</b>")
        lbl_conns.setToolTip("Double‑click a connection to edit. Use Add Connection… to create new links.")
        conns_layout.addWidget(lbl_conns)

        # Connections toolbar
        conns_toolbar = QHBoxLayout()
        btn_add_conn = QPushButton(QIcon(icon("link")), "Add Connection…")
        btn_add_conn.setToolTip("Create a connection between two layers")
        btn_add_conn.clicked.connect(self._add_conn)

        btn_auto_connect = QPushButton("Auto Connect")
        btn_auto_connect.setToolTip("Create feedforward connections (i → i+1) for all adjacent layers using dense connectivity")
        btn_auto_connect.clicked.connect(self._auto_connect_feedforward)

        btn_quantize = QPushButton("Quantize…")
        btn_quantize.setToolTip("Snap selected connection weights to a codebook defined by bits and min/max")
        btn_quantize.clicked.connect(self._open_quantize_dialog)

        conns_toolbar.addWidget(btn_add_conn)
        conns_toolbar.addWidget(btn_auto_connect)
        conns_toolbar.addStretch()
        conns_toolbar.addWidget(btn_quantize)
        conns_layout.addLayout(conns_toolbar)

        self.conn_list = QListWidget()
        self.conn_list.setStyleSheet("border: 2px solid black;")
        self.conn_list.setAlternatingRowColors(True)
        self.conn_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conn_list.customContextMenuRequested.connect(self._show_conns_menu)
        self.conn_list.itemDoubleClicked.connect(self._edit_conn)
        conns_layout.addWidget(self.conn_list, 1)
        splitter.addWidget(conns_panel)

        splitter.setChildrenCollapsible(False)
        spec_layout.addWidget(splitter, 1)

        # --- Rebuild Options (Preservation) ---
        options_row = QHBoxLayout()
        options_row.setContentsMargins(0, 8, 0, 8)
        options_row.addWidget(QLabel("Preserve on rebuild:"))
        self.preserve_cb = QComboBox()
        self.preserve_cb.addItems(["none", "all", "frozen_only"])
        self.preserve_cb.setCurrentText("none")
        self.preserve_cb.setToolTip("Choose what to keep from a baseline model when rebuilding: none (fresh), all (matching only), frozen_only (selected below)")
        self.preserve_cb.currentTextChanged.connect(self._update_freeze_panel_enabled)
        # Ensure the text is fully visible
        self.preserve_cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.preserve_cb.setMinimumContentsLength(12)
        self.preserve_cb.setMinimumWidth(160)
        options_row.addWidget(self.preserve_cb)
        # Baseline selector
        options_row.addWidget(QLabel("Preserve from:"))
        self.baseline_cb = QComboBox()
        self.baseline_cb.addItems(["current", "last_file", "choose_file", "snapshot"])
        self.baseline_cb.setCurrentText(self._mgr.baseline_mode)
        self.baseline_cb.setToolTip("Where to copy weights from when preservation is used")
        # Ensure the text is fully visible
        self.baseline_cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.baseline_cb.setMinimumContentsLength(12)
        self.baseline_cb.setMinimumWidth(180)
        options_row.addWidget(self.baseline_cb)
        self.choose_file_btn = QPushButton("Select file…")
        self.choose_file_btn.setToolTip("Pick a Torch/JSON model file to serve as the preservation baseline")
        self.choose_file_btn.clicked.connect(self._select_baseline_file)
        options_row.addWidget(self.choose_file_btn)
        self.snapshot_btn = QPushButton("Save snapshot")
        self.snapshot_btn.setToolTip("Capture the currently built model as the preservation baseline")
        self.snapshot_btn.clicked.connect(self._save_snapshot)
        options_row.addWidget(self.snapshot_btn)
        options_row.addStretch()
        spec_layout.addLayout(options_row)

        # Freeze selection panel (enabled only for frozen_only)
        self.freeze_panel = QWidget()
        freeze_layout = QVBoxLayout(self.freeze_panel)
        freeze_layout.setContentsMargins(0, 0, 0, 0)

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 4)
        controls_row.addStretch()
        btn_freeze_all = QPushButton("Select all")
        btn_freeze_all.setToolTip("Check every preservation list for a quick 'freeze everything'.")
        btn_freeze_all.setFixedWidth(90)
        btn_freeze_all.clicked.connect(lambda: self._set_freeze_checks(Qt.CheckState.Checked))
        btn_freeze_none = QPushButton("Select none")
        btn_freeze_none.setToolTip("Clear all preservation selections.")
        btn_freeze_none.setFixedWidth(90)
        btn_freeze_none.clicked.connect(lambda: self._set_freeze_checks(Qt.CheckState.Unchecked))
        controls_row.addWidget(btn_freeze_all)
        controls_row.addWidget(btn_freeze_none)
        freeze_layout.addLayout(controls_row)

        lists_row = QHBoxLayout()

        # Freeze Layers
        left = QVBoxLayout()
        lbl_fl = QLabel("Freeze Layers")
        lbl_fl.setToolTip("When using frozen_only, only checked layers will keep parameters from the baseline.")
        left.addWidget(lbl_fl)
        self.freeze_layer_list = QListWidget()
        self.freeze_layer_list.setMaximumHeight(140)
        left.addWidget(self.freeze_layer_list)
        lists_row.addLayout(left)

        # Freeze Connections (weights)
        middle = QVBoxLayout()
        lbl_fc = QLabel("Freeze Connection Weights")
        lbl_fc.setToolTip("When using frozen_only, only checked connections will keep weights from the baseline.")
        middle.addWidget(lbl_fc)
        self.freeze_conn_list = QListWidget()
        self.freeze_conn_list.setMaximumHeight(140)
        middle.addWidget(self.freeze_conn_list)
        lists_row.addLayout(middle)

        # Freeze Masks
        mask_col = QVBoxLayout()
        lbl_mask = QLabel("Freeze Connection Masks")
        lbl_mask.setToolTip("When using frozen_only, only checked masks will be preserved. You can copy masks without copying weights.")
        mask_col.addWidget(lbl_mask)
        self.freeze_mask_list = QListWidget()
        self.freeze_mask_list.setMaximumHeight(140)
        mask_col.addWidget(self.freeze_mask_list)
        lists_row.addLayout(mask_col)

        freeze_layout.addLayout(lists_row)

        spec_layout.addWidget(self.freeze_panel)
        self._update_freeze_panel_enabled(self.preserve_cb.currentText())

        # Build from Model Specification and seed controls
        row = QHBoxLayout()
        btn_build = QPushButton("Build from Model Specification")
        btn_build.setToolTip(
            "Builds a model from current configs. If 'Preserve on rebuild' is 'none',\n"
            "parameters and connections are freshly initialized using their configured distributions.\n"
            "Select 'all' or 'frozen_only' with a baseline to keep weights.",
        )

        # --- Seeding controls (toggle + value) ---
        self.seed_checkbox = QCheckBox("Use Seed:")
        self.seed_checkbox.setToolTip("Enable deterministic initialization for reproducible builds")
        self.seed_value_edit = QLineEdit()
        self.seed_value_edit.setPlaceholderText("e.g. 42")
        self.seed_value_edit.setFixedWidth(80)
        self.seed_value_edit.setText("1")  # default deterministic seed
        self.seed_checkbox.toggled.connect(self.seed_value_edit.setEnabled)
        self.seed_checkbox.setChecked(True)  # default to seeding with value 1
        # ----------------------------------------

        btn_build.clicked.connect(self._build_model)

        row.addWidget(btn_build)
        # --- Add seed controls to the button row ---
        row.addWidget(self.seed_checkbox)
        row.addWidget(self.seed_value_edit)
        # --------------------------------------------
        row.addStretch()
        spec_layout.addLayout(row)

        # Add the lower pane to the outer splitter
        outer_splitter.addWidget(spec_root)
        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)

        # Add splitter to the tab layout
        layout.addWidget(outer_splitter, 1)

        # One-time warning flag for fresh builds that resample distributions
        self._warned_fresh_build = False

        self.refresh_lists()

    def _open_merge_dialog(self) -> None:
        try:
            if len(self._mgr.layers) < 2:
                show_warning(self, "Merge Layers", "Add at least two layers first.")
                return
            dlg = MergeLayersDialog(self, layers=self._mgr.layers)
            if not dlg.exec():
                return
            group = dlg.selected_group()
            if len(group) < 2:
                show_warning(self, "Merge Layers", "Select at least two layers to merge.")
                return

            # Validate same type in UI guardrail (dialog already checks)
            types = {next((layer_item.layer_type for layer_item in self._mgr.layers if layer_item.layer_id == lid), None) for lid in group}
            if None in types:
                show_error(self, "Merge Layers", "Internal error: invalid selection.")
                return
            if len(types) != 1:
                show_warning(self, "Merge Layers", "Selected layers must be of the same type.")
                return

            new_id = dlg.new_layer_id()
            node_order = sorted(group)

            # Try merging on a temporary model instance built from current configs
            temp_model = self._mgr.model or None
            if temp_model is None:
                show_info(
                    self,
                    "Merge Layers",
                    ("Please build the network before merging layers so the current weights can be merged correctly.\nClick Build, then try merging again."),
                )
                return

            try:
                # Soft warnings in UI (never block): feedback creation and input/output involvement
                topo = topology_intel(temp_model, group)
                msgs = []
                if topo.get("cycle_nodes"):
                    msg_nodes = ", ".join(map(str, topo["cycle_nodes"]))
                    msgs.append(
                        "This merge will create feedback between the merged layer and node(s) " + msg_nodes + ". Consider using a stepwise global solver.",
                    )
                if topo.get("touches_input"):
                    msgs.append("Selection includes the input (first) layer. External input semantics will change.")
                if topo.get("touches_output"):
                    msgs.append("Selection includes the output (last) layer. Output channel mapping will change.")
                if msgs:
                    resp = QMessageBox.question(
                        self,
                        "Merge Layers: Warnings",
                        "\n\n".join(msgs) + "\n\nProceed?",
                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                    )
                    if resp != QMessageBox.StandardButton.Ok:
                        return

                spec = MergeSpec(
                    group_ids=node_order,
                    new_layer_id=new_id,
                    node_order=node_order,
                    preserve_state=True,
                    normalize_ids=dlg.normalize_ids(),
                )
                result = apply_merge_layers(temp_model, spec)
                merged_model = result.model
                # Persist the id_map for mapping-aware rebuilds
                try:
                    self._mgr.last_merge_id_map = dict(result.id_map)
                except Exception:
                    self._mgr.last_merge_id_map = {}
            except Exception as e:
                # Show clear reason from utility (e.g., non node-wise params)
                show_error(self, "Cannot Merge", str(e))
                return

            # Success: adopt merged configs back into the manager
            self._mgr.sim_config = merged_model.sim_config
            self._mgr.layers = merged_model.layers_config
            self._mgr.connections = merged_model.connections_config
            self._mgr.model = merged_model
            # Make the merged model the baseline for preservation, so a subsequent Build with
            # Preserve='all' will keep the raw weights and params we just set.
            try:
                self._mgr._last_built_model = copy.deepcopy(merged_model)
            except Exception:
                self._mgr._last_built_model = merged_model
            self._mgr.model_changed.emit()

            # Normalization is handled inside apply_merge_layers when requested

            # Nudge the user: Build is NOT required now; if they choose to rebuild later,
            # Preserve='all' with baseline 'current' will retain weights. Mapping-aware rebuilds
            # are used automatically when IDs changed via this merge flow.
            try:
                self.preserve_cb.setCurrentText("all")
                self.baseline_cb.setCurrentText("current")
            except Exception:
                pass

            show_info(
                self,
                "Merge Layers",
                (
                    f"Merged {len(group)} layer(s) into Layer {new_id}.\n\n"
                    "The merged model is already built; you do not need to click Build now.\n"
                    "If you rebuild later, set Preserve on rebuild = 'all' with baseline 'current' to retain weights."
                ),
            )
            self.refresh_lists()
        except Exception as e:
            show_error(self, "Merge Layers", str(e))

    def _auto_merge_layers(self) -> None:
        try:
            if len(self._mgr.layers) < 2:
                show_info(self, "Auto-merge", "Add at least two layers first.")
                return

            # Settings dialog
            from PyQt6.QtWidgets import (
                QDialog,
                QDialogButtonBox,
                QFormLayout,
                QVBoxLayout,
            )

            dlg = QDialog(self)
            dlg.setWindowTitle("Auto-merge Settings")
            v = QVBoxLayout(dlg)
            form = QFormLayout()
            chk_allow_input = QCheckBox()
            chk_allow_input.setChecked(False)
            form.addRow("Allow merging Input layer:", chk_allow_input)
            chk_allow_output = QCheckBox()
            chk_allow_output.setChecked(False)
            form.addRow("Allow merging Output layer:", chk_allow_output)
            chk_avoid_cycles = QCheckBox()
            chk_avoid_cycles.setChecked(True)
            form.addRow("Avoid Cycles (contiguous-only):", chk_avoid_cycles)
            chk_normalize = QCheckBox()
            chk_normalize.setChecked(True)
            form.addRow("Normalize IDs after merge:", chk_normalize)
            v.addLayout(form)
            btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            v.addWidget(btns)
            if not dlg.exec():
                return

            allow_input = chk_allow_input.isChecked()
            allow_output = chk_allow_output.isChecked()
            avoid_cycles = chk_avoid_cycles.isChecked()
            do_normalize = chk_normalize.isChecked()

            # Prepare model (warn if not built)
            temp_model = self._mgr.model
            if temp_model is None:
                resp = QMessageBox.question(
                    self,
                    "Auto-merge: No Built Model",
                    ("You have not built the model yet. Auto-merge will update configurations only;\nthere are no learned parameters to merge. Proceed?"),
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                )
                if resp != QMessageBox.StandardButton.Ok:
                    return
                temp_model = SOENModelCore(
                    sim_config=self._mgr.sim_config,
                    layers_config=self._mgr.layers,
                    connections_config=self._mgr.connections,
                )

            # Helpers
            def _scc_nodes() -> list[list[int]]:
                # Build adjacency over current connections (directed)
                adj: dict[int, list[int]] = {}
                ids = [cfg.layer_id for cfg in temp_model.layers_config]
                for cc in temp_model.connections_config:
                    adj.setdefault(cc.from_layer, []).append(cc.to_layer)
                    adj.setdefault(cc.to_layer, adj.get(cc.to_layer, []))
                # Tarjan
                index = 0
                stack: list[int] = []
                onstack: set[int] = set()
                idx: dict[int, int] = {}
                low: dict[int, int] = {}
                out: list[list[int]] = []

                def strongconnect(v: int) -> None:
                    nonlocal index
                    idx[v] = index
                    low[v] = index
                    index += 1
                    stack.append(v)
                    onstack.add(v)
                    for w in adj.get(v, []):
                        if w not in idx:
                            strongconnect(w)
                            low[v] = min(low[v], low[w])
                        elif w in onstack:
                            low[v] = min(low[v], idx[w])
                    if low[v] == idx[v]:
                        comp = []
                        while True:
                            w = stack.pop()
                            onstack.remove(w)
                            comp.append(w)
                            if w == v:
                                break
                        out.append(sorted(comp))

                for v in ids:
                    if v not in idx:
                        strongconnect(v)
                return out

            merged_count = 0
            if avoid_cycles:
                # Contiguous-only runs per type (minimizes new cycles)
                changed = True
                while changed:
                    changed = False
                    layers_sorted = sorted(self._mgr.layers, key=lambda x: x.layer_id)
                    if not layers_sorted:
                        break
                    first_id = layers_sorted[0].layer_id
                    last_id = layers_sorted[-1].layer_id
                    runs: list[list[int]] = []
                    cur: list[int] = []
                    cur_type: str | None = None
                    for cfg in layers_sorted:
                        # Respect input/output inclusion toggles
                        if (not allow_input and cfg.layer_id == first_id) or (not allow_output and cfg.layer_id == last_id):
                            if len(cur) >= 2:
                                runs.append(cur)
                            cur = []
                            cur_type = None
                            continue
                        if not cur:
                            cur = [cfg.layer_id]
                            cur_type = cfg.layer_type
                        elif cfg.layer_type == cur_type and cfg.layer_id == (cur[-1] + 1):
                            cur.append(cfg.layer_id)
                        else:
                            if len(cur) >= 2:
                                runs.append(cur)
                            cur = [cfg.layer_id]
                            cur_type = cfg.layer_type
                    if len(cur) >= 2:
                        runs.append(cur)

                    merged_this_pass = False
                    for group in runs:
                        spec = MergeSpec(group_ids=group, new_layer_id=min(group), node_order=group, preserve_state=True)
                        try:
                            result = apply_merge_layers(temp_model, spec)
                        except Exception:
                            continue
                        temp_model = result.model
                        self._mgr.sim_config = temp_model.sim_config
                        self._mgr.layers = temp_model.layers_config
                        self._mgr.connections = temp_model.connections_config
                        self._mgr.model = temp_model
                        merged_count += 1
                        merged_this_pass = True
                    changed = merged_this_pass
                # Check DAG
                sccs = _scc_nodes()
                if any(len(c) > 1 for c in sccs):
                    show_info(
                        self,
                        "Auto-merge",
                        "Avoid Cycles requested, but cycles remain after merges (likely due to mixed layer types).",
                    )
            else:
                # Merge all same-type groups across the network (allow feedback)
                changed = True
                while changed:
                    changed = False
                    # Map type -> list of ids
                    by_type: dict[str, list[int]] = {}
                    for cfg in temp_model.layers_config:
                        by_type.setdefault(cfg.layer_type, []).append(cfg.layer_id)
                    # Apply avoid IO filter per type
                    try:
                        ids_all = sorted(cfg.layer_id for cfg in temp_model.layers_config)
                        first_id = ids_all[0]
                        last_id = ids_all[-1]
                    except Exception:
                        first_id = last_id = None
                    for ids in by_type.values():
                        ids = sorted(ids)
                        # Respect input/output inclusion toggles
                        if not allow_input and first_id is not None:
                            ids = [i for i in ids if i != first_id]
                        if not allow_output and last_id is not None:
                            ids = [i for i in ids if i != last_id]
                        if len(ids) < 2:
                            continue
                        spec = MergeSpec(group_ids=ids, new_layer_id=min(ids), node_order=ids, preserve_state=True)
                        try:
                            result = apply_merge_layers(temp_model, spec)
                        except Exception:
                            continue
                        temp_model = result.model
                        self._mgr.sim_config = temp_model.sim_config
                        self._mgr.layers = temp_model.layers_config
                        self._mgr.connections = temp_model.connections_config
                        self._mgr.model = temp_model
                        merged_count += 1
                        changed = True
                        break  # rescan types after each merge
                # If cycles exist, ensure a stepwise solver is used
                sccs = _scc_nodes()
                if any(len(c) > 1 for c in sccs):
                    with contextlib.suppress(Exception):
                        self._mgr.sim_config.network_evaluation_method = "stepwise_gauss_seidel"

            # Optional normalization with mapping-aware preserve
            if merged_count > 0 and do_normalize:
                try:
                    ids = sorted(cfg.layer_id for cfg in temp_model.layers_config)
                    norm_map = {old: i for i, old in enumerate(ids)}
                    # Build normalized configs
                    norm_layers = []
                    for cfg in temp_model.layers_config:
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
                    for cc in temp_model.connections_config:
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
                    temp_model = rebuild_model_preserving_id_map(
                        base_model=temp_model,
                        sim_config=temp_model.sim_config,
                        layers_config=norm_layers,
                        connections_config=norm_conns,
                        id_map_old_to_new=norm_map,
                    )
                    self._mgr.model = temp_model
                    self._mgr.sim_config = temp_model.sim_config
                    self._mgr.layers = temp_model.layers_config
                    self._mgr.connections = temp_model.connections_config
                except Exception as e:
                    show_warning(self, "Normalize IDs", f"Failed to normalize IDs after auto-merge: {e}")

            if merged_count == 0:
                show_info(self, "Auto-merge", "No mergeable groups were found.")
            else:
                show_info(self, "Auto-merge", f"Merged {merged_count} group(s).")
            self.refresh_lists()
        except Exception as e:
            show_error(self, "Auto-merge", str(e))

    def _auto_connect_feedforward(self) -> None:
        """Create default feedforward connections between adjacent layers.
        Links each sorted layer_id i to the next layer_id j using 'dense'.
        Skips pairs that already exist.
        """
        try:
            if len(self._mgr.layers) < 2:
                show_warning(self, "Not enough layers", "Add at least two layers to auto-connect.")
                return
            # Ensure deterministic order by layer_id
            self._sort_layers_by_id()
            sorted_ids = [cfg.layer_id for cfg in self._mgr.layers]
            existing = {(c.from_layer, c.to_layer) for c in self._mgr.connections}
            created = 0
            for a, b in itertools.pairwise(sorted_ids):
                if (a, b) in existing:
                    continue
                self._mgr.connections.append(
                    ConnectionConfig(
                        from_layer=a,
                        to_layer=b,
                        connection_type="dense",
                        params={},
                        learnable=True,
                    ),
                )
                created += 1
            self.refresh_lists()
            if created == 0:
                show_info(self, "Auto Connect", "No new connections were added.")
            else:
                show_info(self, "Auto Connect", f"Added {created} connection(s).")
        except Exception as e:
            show_error(self, "Auto Connect", str(e))

    def refresh_lists(self) -> None:
        # Always display in ascending ID order
        self.layer_list.clear()
        sorted_layers = sorted(self._mgr.layers, key=lambda x: x.layer_id)
        # Determine input/output layer IDs (first/last by sorted id)
        first_id = sorted_layers[0].layer_id if sorted_layers else None
        last_id = sorted_layers[-1].layer_id if sorted_layers else None
        from PyQt6.QtGui import QBrush, QColor

        for layer_item in sorted_layers:
            dim = layer_item.params.get("dim")
            suffix = ""
            brush = None
            if layer_item.layer_id == first_id:
                suffix = "  (Input Layer)"
                brush = QBrush(QColor("#1b5e20"))  # green
            elif layer_item.layer_id == last_id:
                suffix = "  (Output Layer)"
                brush = QBrush(QColor("#b71c1c"))  # red
            it = QListWidgetItem(f"Layer {layer_item.layer_id} • {layer_item.layer_type} • dim={dim}{suffix}")
            if brush is not None:
                it.setForeground(brush)
            self.layer_list.addItem(it)
        self.conn_list.clear()
        sorted_conns = sorted(self._mgr.connections, key=lambda c: (c.from_layer, c.to_layer))
        for i, c in enumerate(sorted_conns):
            # Store (from,to) in item data to map selection back to the true connection
            it = QListWidgetItem(f"[{i}] {c.from_layer}→{c.to_layer} • {c.connection_type}")
            it.setData(Qt.ItemDataRole.UserRole, (c.from_layer, c.to_layer))
            self.conn_list.addItem(it)

        # Preserve checked selections before repopulating freeze lists
        prev_layer_checks = set()
        for idx in range(self.freeze_layer_list.count()):
            it = self.freeze_layer_list.item(idx)
            if it.checkState() == Qt.CheckState.Checked:
                prev_layer_checks.add(it.data(Qt.ItemDataRole.UserRole))
        prev_conn_checks = set()
        for idx in range(self.freeze_conn_list.count()):
            it = self.freeze_conn_list.item(idx)
            if it.checkState() == Qt.CheckState.Checked:
                pair = it.data(Qt.ItemDataRole.UserRole)
                if isinstance(pair, tuple) and len(pair) == 2:
                    prev_conn_checks.add(tuple(pair))
        prev_mask_checks = set()
        for idx in range(self.freeze_mask_list.count()):
            it = self.freeze_mask_list.item(idx)
            if it.checkState() == Qt.CheckState.Checked:
                pair = it.data(Qt.ItemDataRole.UserRole)
                if isinstance(pair, tuple) and len(pair) == 2:
                    prev_mask_checks.add(tuple(pair))

        # Repopulate freeze lists (checkable)
        self.freeze_layer_list.clear()
        # Build provenance lookup: merged sources annotated by manager
        provenance_tag: dict[int, str] = {}
        try:
            for src_label, id_map in zip(self._mgr.model_provenance_labels, self._mgr.model_provenance, strict=False):
                for new_id in id_map.values():
                    provenance_tag[new_id] = src_label
        except Exception:
            provenance_tag = {}

        for layer_item in sorted_layers:
            src = provenance_tag.get(layer_item.layer_id, "current")
            src_short = src.split("/")[-1] if isinstance(src, str) and "/" in src else src
            text = f"Layer {layer_item.layer_id}: {layer_item.layer_type} [{src_short}]"
            it = QListWidgetItem(text)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if layer_item.layer_id in prev_layer_checks else Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, layer_item.layer_id)
            self.freeze_layer_list.addItem(it)

        self.freeze_conn_list.clear()
        self.freeze_mask_list.clear()
        for c in sorted_conns:
            src_from = provenance_tag.get(c.from_layer, "current")
            src_to = provenance_tag.get(c.to_layer, "current")
            # If both ends were from the same source, show that tag once; else show both
            if src_from == src_to:
                src_tag = src_from.split("/")[-1] if isinstance(src_from, str) and "/" in src_from else src_from
            else:
                s_from = src_from.split("/")[-1] if isinstance(src_from, str) and "/" in src_from else src_from
                s_to = src_to.split("/")[-1] if isinstance(src_to, str) and "/" in src_to else src_to
                src_tag = f"{s_from}|{s_to}"
            text = f"{c.from_layer}→{c.to_layer} ({c.connection_type}) [{src_tag}]"
            it = QListWidgetItem(text)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            pair = (c.from_layer, c.to_layer)
            it.setCheckState(Qt.CheckState.Checked if pair in prev_conn_checks else Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, pair)
            self.freeze_conn_list.addItem(it)
            mask_item = QListWidgetItem(text)
            mask_item.setFlags(mask_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            mask_item.setCheckState(Qt.CheckState.Checked if pair in prev_mask_checks else Qt.CheckState.Unchecked)
            mask_item.setData(Qt.ItemDataRole.UserRole, pair)
            self.freeze_mask_list.addItem(mask_item)

    def _next_id(self) -> int:
        if not self._mgr.layers:
            return 0
        return max(layer_item.layer_id for layer_item in self._mgr.layers) + 1

    def _add_layer(self) -> None:
        dlg = LayerConfigDialog(
            self,
            default_id=self._next_id(),
        )
        if dlg.exec():
            new_cfg = dlg.result_config()
            new_id = new_cfg.layer_id
            existing_ids = {cfg.layer_id for cfg in self._mgr.layers}

            # Check if this insertion would conflict with existing IDs
            # If so, shift existing layers to make room
            if new_id in existing_ids:
                # Conflict: shift all layers >= new_id upward by 1
                id_map = self._build_id_shift_map(new_id, shift=1)
                self._apply_id_shift(id_map)
            # Otherwise: no conflict, preserve any existing ID map from prior operations
            # (e.g., merge or edit). Don't clear it - it's still valid for preservation.

            self._mgr.layers.append(new_cfg)
            self._sort_layers_by_id()
            self.refresh_lists()

    def _build_id_shift_map(self, threshold: int, shift: int) -> dict[int, int]:
        """Build a mapping of old_id -> new_id for layers >= threshold."""
        id_map = {}
        for cfg in self._mgr.layers:
            if cfg.layer_id >= threshold:
                id_map[cfg.layer_id] = cfg.layer_id + shift
            else:
                id_map[cfg.layer_id] = cfg.layer_id
        return id_map

    def _apply_id_shift(self, id_map: dict[int, int]) -> None:
        """Apply ID shifts to layers and connections, and store for preservation."""
        # Shift layer IDs
        for cfg in self._mgr.layers:
            old_id = cfg.layer_id
            if old_id in id_map:
                cfg.layer_id = id_map[old_id]

        # Shift connection endpoints
        for cc in self._mgr.connections:
            if cc.from_layer in id_map:
                cc.from_layer = id_map[cc.from_layer]
            if cc.to_layer in id_map:
                cc.to_layer = id_map[cc.to_layer]

        # Store the mapping for preservation during rebuild
        # The map is old->new, so preservation can find the right source
        self._mgr.last_merge_id_map = id_map

        # Clear the built model since structure changed
        self._mgr.model = None

    def _edit_layer(self) -> None:
        i = self.layer_list.currentRow()
        if i < 0:
            return
        cfg = self._mgr.layers[i]
        dlg = LayerConfigDialog(
            self,
            existing=cfg,
        )
        ret = dlg.exec()
        if getattr(dlg, "deleteRequested", False):
            layer_id = cfg.layer_id
            del self._mgr.layers[i]
            self._mgr.connections = [c for c in self._mgr.connections if layer_id not in (c.from_layer, c.to_layer)]
            self._mgr.model = None
            self._mgr.last_merge_id_map = {}  # Clear stale mapping
            self._mgr.model_changed.emit()
        elif ret:
            new_cfg = dlg.result_config()
            if new_cfg is None:
                return
            old_id = cfg.layer_id
            new_id = new_cfg.layer_id
            self._mgr.layers[i] = new_cfg
            if old_id != new_id:
                # Check for conflict with existing IDs
                other_ids = {c.layer_id for c in self._mgr.layers if c is not new_cfg}
                if new_id in other_ids:
                    # Conflict: this is more complex - for now, just warn
                    show_warning(
                        self,
                        "ID Conflict",
                        f"Layer ID {new_id} already exists. Please choose a different ID.",
                    )
                    # Revert the change
                    new_cfg.layer_id = old_id
                else:
                    # Create ID map for preservation: old_id -> new_id
                    # Only include the changed layer's mapping (other layers keep same id)
                    id_map = {old_id: new_id}
                    # Add identity mappings for other layers (skip the one we just edited)
                    for c in self._mgr.layers:
                        if c is not new_cfg and c.layer_id not in id_map:
                            id_map[c.layer_id] = c.layer_id
                    self._mgr.last_merge_id_map = id_map
                    self._prompt_update_connections_for_layer_id_change(old_id, new_id)
                    self._mgr.model = None
            self._sort_layers_by_id()
        self.refresh_lists()

    def _add_conn(self) -> None:
        if not self._mgr.layers:
            show_warning(self, "No layers", "Define at least one layer first.")
            return
        dlg = ConnectionConfigDialog(
            self,
            layers=self._mgr.layers,
        )
        if dlg.exec():
            self._mgr.connections.append(dlg.result_config())
            self.refresh_lists()

    def _edit_conn(self) -> None:
        item = self.conn_list.currentItem()
        if item is None:
            return
        pair = item.data(Qt.ItemDataRole.UserRole)
        # Find the correct index in manager list
        idx = None
        for k, cc in enumerate(self._mgr.connections):
            if (cc.from_layer, cc.to_layer) == tuple(pair):
                idx = k
                break
        if idx is None:
            return
        cfg = self._mgr.connections[idx]
        dlg = ConnectionConfigDialog(
            self,
            layers=self._mgr.layers,
            existing=cfg,
        )
        ret = dlg.exec()
        if getattr(dlg, "deleteRequested", False):
            del self._mgr.connections[idx]
            self._mgr.model = None
            self._mgr.model_changed.emit()
        elif ret:
            self._mgr.connections[idx] = dlg.result_config()
        self.refresh_lists()

    def _build_model(self) -> None:
        from soen_toolkit.model_creation_gui.components.progress_dialog import OperationProgressDialog

        try:
            use_seed = self.seed_checkbox.isChecked()
            self._on_dt_changed()
            preserve_mode = self.preserve_cb.currentText()
            freeze_layers, freeze_connections, freeze_masks = self._collect_freeze_selections()
            # Only pass freeze lists for frozen_only mode to avoid confusion
            if preserve_mode != "frozen_only":
                freeze_layers = None
                freeze_connections = None
                freeze_masks = None
            # Set baseline mode/file on manager
            self._mgr.baseline_mode = self.baseline_cb.currentText()
            seed_val = None
            if use_seed:
                try:
                    seed_val = int(self.seed_value_edit.text()) if self.seed_value_edit.text().strip() else None
                except Exception:
                    seed_val = None
                if seed_val is None:
                    # Fallback to default seed when the toggle is on but no value is provided
                    seed_val = 1
                    self.seed_value_edit.setText("1")

            # Warn if performing a fresh build that will resample distributions
            if preserve_mode == "none" and not getattr(self, "_warned_fresh_build", False):
                msg = (
                    "This will build a fresh model: layer parameters and connection weights "
                    "will be initialized by their configured distributions (random resampling).\n\n"
                    "To keep existing weights, set 'Preserve on rebuild' to 'all' (matching shapes) "
                    "or 'frozen_only' and choose a baseline under 'Preserve from'."
                )
                if not use_seed:
                    msg += "\n\nTip: Enable 'Use Seed' for reproducibility."
                resp = QMessageBox.question(
                    self,
                    "Fresh Build: Resampling",
                    msg,
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                )
                if resp != QMessageBox.StandardButton.Ok:
                    return
                self._warned_fresh_build = True

            # Show progress dialog during build
            progress = OperationProgressDialog(self, title="Building Model")
            progress.set_title("Building Model...")
            progress.show()  # Make sure dialog is visible
            progress.raise_()  # Bring to front
            progress.activateWindow()  # Give it focus
            from PyQt6.QtWidgets import QApplication

            # Initial progress
            progress.set_progress(5, "Initializing build...")
            QApplication.processEvents()

            # Create a callback for progress updates from the model manager
            def on_build_progress(value: int, step: str = "") -> None:
                progress.set_progress(value, step)
                QApplication.processEvents()

            # Store the callback for the build process
            self._mgr._build_progress_callback = on_build_progress

            try:
                on_build_progress(10, "Loading layer types...")
                QApplication.processEvents()
                on_build_progress(20, "Preparing layer configurations...")
                QApplication.processEvents()

                self._mgr.build_model(
                    use_seed=use_seed,
                    preserve_mode=preserve_mode,
                    freeze_layers=freeze_layers,
                    freeze_connections=freeze_connections,
                    freeze_masks=freeze_masks,
                    seed_value=seed_val,
                )
                on_build_progress(100, "Build complete!")
                QApplication.processEvents()
                progress.close()  # Close dialog after build
            finally:
                # Clean up callback
                if hasattr(self._mgr, "_build_progress_callback"):
                    del self._mgr._build_progress_callback
                progress.close()  # Ensure dialog closes

            show_info(self, "Built", "Model successfully built from model specification.")
        except Exception as e:
            show_error(self, "Error", str(e))

    def _prompt_update_connections_for_layer_id_change(self, old_id: int, new_id: int) -> None:
        """Offer to update connections when a layer ID is modified."""
        impacted = [c for c in self._mgr.connections if old_id in (c.from_layer, c.to_layer)]
        if not impacted:
            return

        resp = QMessageBox.question(
            self,
            "Update connections?",
            (f"Layer ID changed from {old_id} to {new_id}.\nUpdate {len(impacted)} connection(s) to reference the new ID?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if resp != QMessageBox.StandardButton.Yes:
            return

        for conn in impacted:
            if conn.from_layer == old_id:
                conn.from_layer = new_id
            if conn.to_layer == old_id:
                conn.to_layer = new_id

    def _open_quantize_dialog(self) -> None:
        if self._mgr.model is None:
            show_warning(self, "No model", "Build or load a model first to quantize connections.")
            return
        dlg = QuantizeDialog(self, model=self._mgr.model)
        if not dlg.exec():
            return
        bits, min_v, max_v, names = dlg.result_params()
        try:
            # Use in-place quantization on the current model instance
            self._mgr.model.quantize(bits=bits, min=min_v, max=max_v, connections=names, in_place=True)
            show_info(self, "Quantized", f"Quantized {len(names)} connection(s).")
            # Notify other tabs/views
            self._mgr.model_changed.emit()
        except Exception as e:
            show_error(self, "Quantization failed", str(e))

    def _select_baseline_file(self) -> None:
        from PyQt6.QtCore import QDir
        from PyQt6.QtWidgets import QFileDialog

        # Prefer last-used model directory; use non-native dialog for speed
        try:
            start = getattr(self._mgr, "last_loaded_path", None)
            start_dir = str(start.parent) if start else QDir.homePath()
        except Exception:
            start_dir = QDir.homePath()
        opts = QFileDialog.Options()
        try:
            opts |= QFileDialog.Option.DontUseNativeDialog
            opts |= QFileDialog.Option.DontResolveSymlinks
            opts |= QFileDialog.Option.ReadOnly
        except Exception:
            pass
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select baseline model",
            start_dir,
            "SOEN Model (*.pth *.soen);;JSON (*.json)",
            options=opts,
        )
        if fname:
            import pathlib

            self._mgr.baseline_file_path = pathlib.Path(fname)
            self.baseline_cb.setCurrentText("choose_file")

    def _save_snapshot(self) -> None:
        ok = self._mgr.save_baseline_snapshot()
        if ok:
            show_info(self, "Snapshot", "Saved current model snapshot as preservation baseline.")
        else:
            show_warning(self, "Snapshot", "No model to snapshot yet. Build or load a model first.")

    def _on_dt_changed(self) -> None:
        try:
            new_dt_val = float(self.dt_input.text())
            if new_dt_val <= 0:
                show_warning(self, "Invalid dt", "Time step (dt) must be positive.")
                self.dt_input.setText(str(self._mgr.sim_config.dt))
                return

            self._mgr.sim_config.dt = new_dt_val
            if self._mgr.model is not None:
                self._mgr.model.set_dt(new_dt_val)
        except ValueError:
            show_warning(self, "Invalid Input", f"Invalid number format for dt: '{self.dt_input.text()}'. Please use standard or scientific notation.")
            self.dt_input.setText(str(self._mgr.sim_config.dt))

    def _on_model_changed(self) -> None:
        self.refresh_lists()
        self.dt_input.setText(str(self._mgr.sim_config.dt))
        current_mode = getattr(self._mgr.sim_config, "input_type", "flux")
        self.input_type_cb.setCurrentText(current_mode)
        # sync solver dropdown
        _gs2 = str(getattr(self._mgr.sim_config, "network_evaluation_method", "layerwise")).lower()
        if _gs2 == "stepwise_gauss_seidel":
            self.solver_cb.setCurrentIndex(1)
        elif _gs2 == "stepwise_jacobi":
            self.solver_cb.setCurrentIndex(2)
        else:
            self.solver_cb.setCurrentIndex(0)
        for attr, checkbox in self._tracking_controls:
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(getattr(self._mgr.sim_config, attr, False)))
            checkbox.blockSignals(False)
        # Early stopping controls
        self.es_chk.blockSignals(True)
        try:
            self.es_chk.setChecked(bool(getattr(self._mgr.sim_config, "early_stopping_forward_pass", False)))
        finally:
            self.es_chk.blockSignals(False)
        for widget, attr, cast, default in self._spin_bindings:
            widget.blockSignals(True)
            try:
                widget.setValue(cast(getattr(self._mgr.sim_config, attr, default)))
            except Exception:
                widget.setValue(default)
            widget.blockSignals(False)
        self._update_early_stop_enabled()

    def _update_freeze_panel_enabled(self, mode: str) -> None:
        enabled = mode == "frozen_only"
        self.freeze_panel.setEnabled(enabled)
        self.freeze_panel.setVisible(enabled)

    def _set_freeze_checks(self, state: Qt.CheckState) -> None:
        """Apply a check/uncheck state to all freeze lists for quick selection."""
        for lst in (self.freeze_layer_list, self.freeze_conn_list, self.freeze_mask_list):
            if lst is None:
                continue
            for i in range(lst.count()):
                lst.item(i).setCheckState(state)

    def _collect_freeze_selections(self):
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
                if isinstance(pair, tuple) and len(pair) == 2:
                    conns.append((int(pair[0]), int(pair[1])))
        masks = []
        for i in range(self.freeze_mask_list.count()):
            it = self.freeze_mask_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                pair = it.data(Qt.ItemDataRole.UserRole)
                if isinstance(pair, tuple) and len(pair) == 2:
                    masks.append((int(pair[0]), int(pair[1])))
        return layers, conns, masks

    def _remove_layer(self) -> None:
        i = self.layer_list.currentRow()
        if i < 0:
            return
        layer_cfg = self._mgr.layers[i]
        layer_id = layer_cfg.layer_id
        related = [c for c in self._mgr.connections if layer_id in (c.from_layer, c.to_layer)]
        count = len(related)
        if count > 0:
            resp = QMessageBox.question(
                self,
                "Remove Layer",
                f"Removing layer {layer_id} will also remove {count} connection(s). Proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        del self._mgr.layers[i]
        self._mgr.connections = [c for c in self._mgr.connections if layer_id not in (c.from_layer, c.to_layer)]
        self._mgr.model = None
        self._mgr.last_merge_id_map = {}  # Clear stale mapping
        self._mgr.model_changed.emit()

    def _remove_conn(self) -> None:
        item = self.conn_list.currentItem()
        if item is None:
            return
        pair = item.data(Qt.ItemDataRole.UserRole)
        idx = None
        for k, cc in enumerate(self._mgr.connections):
            if (cc.from_layer, cc.to_layer) == tuple(pair):
                idx = k
                break
        if idx is None:
            return
        conn_cfg = self._mgr.connections[idx]
        f = conn_cfg.from_layer
        t = conn_cfg.to_layer
        resp = QMessageBox.question(
            self,
            "Remove Connection",
            f"Remove connection {f} → {t}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        del self._mgr.connections[idx]
        self._mgr.model = None
        # Note: Don't clear last_merge_id_map here - connection removal doesn't
        # change layer IDs, and the map may still be valid for layer preservation
        self._mgr.model_changed.emit()

    def _on_tracking_changed(self) -> None:
        updates = {}
        for attr, checkbox in self._tracking_controls:
            value = bool(checkbox.isChecked())
            setattr(self._mgr.sim_config, attr, value)
            updates[attr] = value
        if self._mgr.model is not None and updates:
            self._mgr.model.set_tracking(**updates)

    def _update_early_stop_enabled(self) -> None:
        """Enable/disable early-stop controls depending on solver mode.
        Disables and unchecks when solver = Layerwise.
        """
        mode_text = self.solver_cb.currentText().lower()
        is_layerwise = "layerwise" in mode_text
        enabled = not is_layerwise
        self.es_chk.setEnabled(enabled)
        # Windowed vs legacy
        use_window = enabled and self.es_chk.isChecked() and (self.es_win.value() > 0)
        use_legacy = enabled and self.es_chk.isChecked() and not use_window

        # Show/hide entire early-stop options row only when enabled and supported
        self.es_opts_panel.setVisible(enabled and self.es_chk.isChecked())

        # Legacy controls
        self.es_tol.setEnabled(use_legacy)
        self.es_pat.setEnabled(use_legacy)
        self.es_min.setEnabled(self.es_chk.isChecked() and enabled)
        # Windowed controls
        self.es_win.setEnabled(enabled and self.es_chk.isChecked())
        self.es_abs.setEnabled(use_window)
        self.es_rel.setEnabled(use_window)
        if is_layerwise and self.es_chk.isChecked():
            # Force off when switching into layerwise mode
            self.es_chk.setChecked(False)
            with contextlib.suppress(Exception):
                self._mgr.sim_config.early_stopping_forward_pass = False

    # ---------- utilities ----------
    def _sort_layers_by_id(self) -> None:
        with contextlib.suppress(Exception):
            self._mgr.layers.sort(key=lambda x: x.layer_id)

    def _normalize_layer_ids(self) -> None:
        # Renumber layers sequentially (0..N) following current sorted order
        try:
            self._sort_layers_by_id()
            old_to_new = {cfg.layer_id: idx for idx, cfg in enumerate(self._mgr.layers)}

            # Check if any IDs actually change
            needs_change = any(old != new for old, new in old_to_new.items())
            if not needs_change:
                show_info(self, "Normalize IDs", "Layer IDs are already sequential.")
                return

            # Store the mapping for preservation during rebuild
            self._mgr.last_merge_id_map = old_to_new

            for cfg in self._mgr.layers:
                cfg.layer_id = old_to_new[cfg.layer_id]
            for cc in self._mgr.connections:
                cc.from_layer = old_to_new.get(cc.from_layer, cc.from_layer)
                cc.to_layer = old_to_new.get(cc.to_layer, cc.to_layer)
            self._mgr.model = None
            self._mgr.model_changed.emit()
            self.refresh_lists()
        except Exception as e:
            show_error(self, "Normalize IDs", str(e))

    # Context menus
    def _show_layers_menu(self, pos: QPoint) -> None:
        i = self.layer_list.indexAt(pos).row()
        if i < 0:
            return
        from PyQt6.QtWidgets import QMenu

        # Get the layer config for the selected row (layers are displayed sorted by ID)
        sorted_layers = sorted(self._mgr.layers, key=lambda x: x.layer_id)
        if i >= len(sorted_layers):
            return
        selected_layer = sorted_layers[i]

        m = QMenu(self)
        act_edit = m.addAction("Edit…")
        act_insert_before = m.addAction("Insert Before…")
        act_insert_after = m.addAction("Insert After…")
        m.addSeparator()
        act_dup = m.addAction("Duplicate")
        act_del = m.addAction("Delete")
        a = m.exec(self.layer_list.mapToGlobal(pos))
        if a == act_edit:
            self._edit_layer()
        elif a == act_insert_before:
            self._insert_layer_before(selected_layer.layer_id)
        elif a == act_insert_after:
            self._insert_layer_after(selected_layer.layer_id)
        elif a == act_dup:
            self._duplicate_layer()
        elif a == act_del:
            self._remove_layer()

    def _duplicate_layer(self) -> None:
        i = self.layer_list.currentRow()
        if i < 0:
            return
        try:
            src = self._mgr.layers[i]
            # Deep copy mutable fields to avoid unintended shared references
            import copy as _cpy

            new_cfg = LayerConfig(
                layer_id=self._next_id(),
                model_id=getattr(src, "model_id", 0),
                layer_type=src.layer_type,
                params=_cpy.deepcopy(dict(src.params) if isinstance(src.params, dict) else src.params),
                description=getattr(src, "description", ""),
                noise=_cpy.deepcopy(getattr(src, "noise", None)),
                perturb=_cpy.deepcopy(getattr(src, "perturb", None)),
            )
            self._mgr.layers.append(new_cfg)
            self._sort_layers_by_id()
            self.refresh_lists()
        except Exception as e:
            show_error(self, "Duplicate Layer", str(e))

    def _insert_layer_before(self, reference_layer_id: int) -> None:
        """Insert a new layer before the specified layer, shifting existing IDs."""
        self._insert_layer_at(target_id=reference_layer_id, position="before")

    def _insert_layer_after(self, reference_layer_id: int) -> None:
        """Insert a new layer after the specified layer, shifting existing IDs."""
        self._insert_layer_at(target_id=reference_layer_id + 1, position="after")

    def _insert_layer_at(self, target_id: int, position: str) -> None:
        """Insert a new layer at the target ID, shifting existing layers if needed.

        Args:
            target_id: The ID for the new layer
            position: 'before' or 'after' - used for dialog title
        """
        # Open the layer config dialog with the target ID pre-filled
        dlg = LayerConfigDialog(
            self,
            default_id=target_id,
        )
        dlg.setWindowTitle(f"Insert Layer {position.capitalize()}")
        if not dlg.exec():
            return

        new_cfg = dlg.result_config()
        new_id = new_cfg.layer_id

        # Check if any existing layers need to be shifted
        existing_ids = {cfg.layer_id for cfg in self._mgr.layers}
        if new_id in existing_ids or any(lid >= new_id for lid in existing_ids):
            # Build ID shift map: all layers >= new_id shift by +1
            id_map = self._build_id_shift_map(new_id, shift=1)
            # Apply the shift to existing layers and connections
            self._apply_id_shift(id_map)

        # Add the new layer
        self._mgr.layers.append(new_cfg)
        self._sort_layers_by_id()
        self.refresh_lists()

        # Inform user about parameter preservation
        show_info(
            self,
            "Layer Inserted",
            f"Layer {new_id} inserted. Existing layers have been renumbered.\n\n"
            "To preserve weights from shifted layers, set 'Preserve on rebuild' to 'all' "
            "with baseline 'current' before rebuilding.",
        )

    def _show_conns_menu(self, pos: QPoint) -> None:
        j = self.conn_list.indexAt(pos).row()
        if j < 0:
            return
        from PyQt6.QtWidgets import QMenu

        m = QMenu(self)
        act_edit = m.addAction("Edit…")
        act_del = m.addAction("Delete")
        a = m.exec(self.conn_list.mapToGlobal(pos))
        if a == act_edit:
            self._edit_conn()
        elif a == act_del:
            self._remove_conn()
