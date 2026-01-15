# FILE: src/soen_toolkit/model_creation_gui/components/connection_config_dialog.py

from __future__ import annotations

import contextlib
import inspect
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from soen_toolkit.core.configs import (
    ConnectionConfig,
    LayerConfig,
    NoiseConfig,
    PerturbationConfig,
)
from soen_toolkit.core.layers.common import CONNECTIVITY_BUILDERS, WEIGHT_INITIALIZERS

# Import descriptions and parameter types (import directly to avoid empty fallbacks)
from soen_toolkit.core.layers.common.connectivity_metadata import (
    CONNECTIVITY_ALIASES,
    CONNECTIVITY_DESCRIPTIONS,
    CONNECTIVITY_PARAM_TYPES,
    WEIGHT_INITIALIZER_DESCRIPTIONS,
    normalize_connectivity_type,
)
from soen_toolkit.model_creation_gui.components.func_param_editor import FuncParamEditor

from ..utils.ui_helpers import create_float_validator

# Define known parameter types that should be treated as enums even if not in CONNECTIVITY_PARAM_TYPES
DEFAULT_ENUM_PARAMS = {
    "connection_mode": ["diagonal", "full", "none", "next"],
}

# Default numeric parameters with specific ranges (kept for reference, but ranges will be overridden)
DEFAULT_NUMERIC_RANGES = {
    # Format: "param_name": (min, max, decimals, step, default)
    "sparsity": (0.0, 1.0, 2, 0.1, 0.5),  # Min/max/decimals will be overridden by QDoubleValidator setup
    "within_block_density": (0.0, 1.0, 2, 0.1, 1.0),  # Min/max/decimals will be overridden
    "cross_block_density": (0.0, 1.0, 2, 0.1, 0.0),  # Min/max/decimals will be overridden
}


class DynamicFloatListWidget(QWidget):
    """Widget that shows N float input boxes based on a dependency parameter."""

    def __init__(
        self,
        count: int = 3,
        defaults: list[float] | None = None,
        min_val: float = 0.0,
        max_val: float = 1.0,
        decimals: int = 2,
        step: float = 0.05,
        existing_values: list[float] | None = None,
    ) -> None:
        super().__init__()
        self._defaults = defaults or [1.0, 0.5, 0.25, 0.125]
        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._step = step
        self._boxes: list[QDoubleSpinBox] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)
        self._rebuild(count, existing_values)

    def _rebuild(self, count: int, existing_values: list[float] | None = None) -> None:
        """Rebuild the widget with the specified number of input boxes."""
        # Clear existing widgets
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()
        self._boxes.clear()

        # Create new spinboxes
        for i in range(count):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)

            label = QLabel(f"Tier {i}:")
            label.setMinimumWidth(50)
            row_layout.addWidget(label)

            spinbox = QDoubleSpinBox()
            spinbox.setRange(self._min, self._max)
            spinbox.setDecimals(self._decimals)
            spinbox.setSingleStep(self._step)
            spinbox.setMinimumWidth(100)

            # Set value: use existing if available, else default, else fallback
            if existing_values and i < len(existing_values):
                spinbox.setValue(float(existing_values[i]))
            elif i < len(self._defaults):
                spinbox.setValue(self._defaults[i])
            else:
                # Fallback for levels beyond defaults
                spinbox.setValue(self._defaults[-1] if self._defaults else 0.5)

            row_layout.addWidget(spinbox)
            row_layout.addStretch()

            self._layout.addWidget(row)
            self._boxes.append(spinbox)

    def update_count(self, count: int) -> None:
        """Update the number of input boxes, preserving existing values where possible."""
        existing = self.get_values()
        self._rebuild(count, existing)

    def get_values(self) -> list[float]:
        """Get the list of values from all spinboxes."""
        return [box.value() for box in self._boxes]

    def set_values(self, values: list[float]) -> None:
        """Set values for the spinboxes."""
        for i, val in enumerate(values):
            if i < len(self._boxes):
                self._boxes[i].setValue(float(val))


class ConnectionConfigDialog(QDialog):
    def __init__(
        self,
        parent=None,
        layers: list[LayerConfig] | None = None,
        existing: ConnectionConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connection Configuration")
        self._layers = layers or []
        self._existing = existing
        # Flag to indicate delete request
        self.deleteRequested = False
        self._init_ui()

    def _init_ui(self) -> None:
        self.resize(1100, 700)
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Basic tab
        basic = QWidget()
        v_layout = QVBoxLayout(basic)
        v_layout.setSpacing(15)
        v_layout.setContentsMargins(10, 10, 10, 10)

        # --- Connection Routing Section ---
        routing_group = QGroupBox("Connection Routing")
        routing_form = QFormLayout(routing_group)
        routing_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        routing_form.setHorizontalSpacing(20)
        routing_form.setVerticalSpacing(12)

        # Create custom dropdown model for better display
        self.from_cb = QComboBox()
        self.to_cb = QComboBox()
        self.from_cb.setMinimumWidth(180)
        self.to_cb.setMinimumWidth(180)

        from_model = QStandardItemModel()
        to_model = QStandardItemModel()

        for layer in self._layers:
            from_item = QStandardItem(f"Layer {layer.layer_id}")
            to_item = QStandardItem(f"Layer {layer.layer_id}")
            from_item.setData(layer.layer_id, Qt.ItemDataRole.UserRole)
            to_item.setData(layer.layer_id, Qt.ItemDataRole.UserRole)
            from_model.appendRow(from_item)
            to_model.appendRow(to_item)

        self.from_cb.setModel(from_model)
        self.to_cb.setModel(to_model)

        routing_form.addRow("From Layer:", self.from_cb)
        routing_form.addRow("To Layer:", self.to_cb)

        if self._existing is None and to_model.rowCount() >= 2:
            with contextlib.suppress(Exception):
                self.to_cb.setCurrentIndex(1)

        v_layout.addWidget(routing_group)

        # --- Connectivity Configuration Section ---
        conn_cfg_group = QGroupBox("Connectivity Configuration")
        conn_cfg_form = QFormLayout(conn_cfg_group)
        conn_cfg_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        conn_cfg_form.setHorizontalSpacing(20)
        conn_cfg_form.setVerticalSpacing(12)

        # Connectivity type
        self.type_cb = QComboBox()
        canonical_types = [name for name in sorted(CONNECTIVITY_BUILDERS.keys()) if name not in CONNECTIVITY_ALIASES]
        for name in canonical_types:
            self.type_cb.addItem(name)
        if self._existing is None:
            try:
                idx_dense = self.type_cb.findText("dense")
                if idx_dense >= 0:
                    self.type_cb.setCurrentIndex(idx_dense)
            except Exception:
                pass
        self.type_cb.currentTextChanged.connect(self._on_connectivity_type_changed)
        self.type_cb.currentTextChanged.connect(self._check_layer_compatibility)
        conn_cfg_form.addRow("Type:", self.type_cb)

        # Connectivity description
        self.conn_desc_label = QLabel()
        self.conn_desc_label.setWordWrap(True)
        self.conn_desc_label.setStyleSheet("font-style: italic; color: #666; margin-top: 5px; margin-left: 10px; background-color: transparent;")
        self.conn_desc_label.setMinimumHeight(40)
        conn_cfg_form.addRow(self.conn_desc_label)

        # Warning label for dimension mismatches
        self.conn_warning_label = QLabel()
        self.conn_warning_label.setWordWrap(True)
        self.conn_warning_label.setStyleSheet("font-weight: bold; color: #ff9500; margin-top: 5px; margin-left: 10px; background-color: transparent;")
        self.conn_warning_label.setMinimumHeight(40)
        self.conn_warning_label.setVisible(False)
        conn_cfg_form.addRow("", self.conn_warning_label)

        # Init method
        self.init_cb = QComboBox()
        self.init_cb.addItems(WEIGHT_INITIALIZERS.keys())
        conn_cfg_form.addRow("Init Method:", self.init_cb)

        # Init method description (full-width row)
        self.init_desc_label = QLabel()
        self.init_desc_label.setWordWrap(True)
        self.init_desc_label.setStyleSheet("font-style: italic; color: #666; margin-top: 5px; margin-left: 10px; background-color: transparent;")
        self.init_desc_label.setMinimumHeight(40)
        conn_cfg_form.addRow(self.init_desc_label)

        v_layout.addWidget(conn_cfg_group)

        # --- Connection Flags Section ---
        flags_group = QGroupBox("Connection Flags")
        flags_layout = QVBoxLayout(flags_group)
        flags_layout.setSpacing(10)

        self.self_conn_cb = QCheckBox("Allow Self-Connections")
        self.self_conn_cb.setChecked(True)
        self.self_conn_cb.setToolTip("Enable/disable diagonal elements in the connectivity matrix")
        flags_layout.addWidget(self.self_conn_cb)

        self.learn_cb = QCheckBox("Learnable")
        self.learn_cb.setChecked(True)
        flags_layout.addWidget(self.learn_cb)

        v_layout.addWidget(flags_group)

        # --- Connection Mode Section ---
        mode_group = QGroupBox("Connection Mode")
        mode_form = QFormLayout(mode_group)
        mode_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        mode_form.setHorizontalSpacing(20)
        mode_form.setVerticalSpacing(12)

        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["Fixed", "Dynamic"])
        mode_form.addRow("Mode:", self.mode_cb)

        self.mode_hint = QLabel("Fixed mode uses standard matrix multiply (fastest, no dynamics).")
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setStyleSheet("color: #999; font-style: italic; margin-left: 10px;")
        mode_form.addRow("", self.mode_hint)

        v_layout.addWidget(mode_group)

        # --- Dynamic Connection Settings removed from here - now in Parameters tab ---

        v_layout.addStretch()
        self.tabs.addTab(basic, "Basic")

        # Initialize structure params BEFORE building params tab
        existing_params = self._existing.params if self._existing and self._existing.params else {}
        self._existing_structure_params = {}
        self._existing_init_params = {}
        self._existing_type_override = None
        if isinstance(existing_params, dict):
            structure_info = existing_params.get("structure")
            if isinstance(structure_info, dict):
                self._existing_type_override = structure_info.get("type")
                self._existing_structure_params = structure_info.get("params", {}) or {}
            init_info = existing_params.get("init")
            if isinstance(init_info, dict):
                self._existing_init_params = init_info.get("params", {}) or {}

        # Connect layer selection changed handlers after Basic tab is set up
        self.from_cb.currentIndexChanged.connect(self._check_layer_compatibility)
        self.to_cb.currentIndexChanged.connect(self._check_layer_compatibility)

        # Load existing connection data if editing
        if self._existing:
            from_model = self.from_cb.model()
            to_model = self.to_cb.model()

            for i in range(from_model.rowCount()):
                if from_model.item(i).data(Qt.ItemDataRole.UserRole) == self._existing.from_layer:
                    self.from_cb.setCurrentIndex(i)
                    break

            for i in range(to_model.rowCount()):
                if to_model.item(i).data(Qt.ItemDataRole.UserRole) == self._existing.to_layer:
                    self.to_cb.setCurrentIndex(i)
                    break

            type_value = self._existing_type_override or self._existing.connection_type
            normalized_value = CONNECTIVITY_ALIASES.get(type_value, type_value)
            self.type_cb.setCurrentText(normalized_value)

            if isinstance(existing_params.get("init"), dict):
                init_name = existing_params["init"].get("name", "normal")
            else:
                init_name = existing_params.get("init", "normal")
            self.init_cb.setCurrentText(init_name)

            if self._existing.params and "allow_self_connections" in self._existing.params:
                self.self_conn_cb.setChecked(self._existing.params["allow_self_connections"])

            self.learn_cb.setChecked(self._existing.learnable)

            try:
                mode_val = str(existing_params.get("mode", existing_params.get("connection_mode", "fixed"))).lower()
                if mode_val in ("multiplier", "dynamic", "wicc", "nocc", "dynamic_v1", "dynamic_v2", "multiplier_v2"):
                    self.mode_cb.setCurrentText("Dynamic")
                else:
                    self.mode_cb.setCurrentText("Fixed")
            except Exception:
                pass

        # Build Parameters tab after all Basic tab setup is complete
        # This ensures correct defaults for new connections and correct values for existing connections
        self._build_params_tab()

        # Constraints tab
        cons = QWidget()
        cons_layout = QVBoxLayout(cons)
        cons_layout.setSpacing(15)
        cons_layout.setContentsMargins(10, 10, 10, 10)

        cons_group = QGroupBox("Weight Constraints")
        cform = QFormLayout(cons_group)
        cform.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        cform.setHorizontalSpacing(20)
        cform.setVerticalSpacing(12)

        self.cons_enable = QCheckBox("Enable constraints")
        self.min_sb = QLineEdit()
        self.max_sb = QLineEdit()

        self.min_sb.setValidator(create_float_validator())
        self.max_sb.setValidator(create_float_validator())

        cform.addRow(self.cons_enable, QWidget())
        cform.addRow("Minimum Value:", self.min_sb)
        cform.addRow("Maximum Value:", self.max_sb)

        if self._existing and "constraints" in (self._existing.params or {}):
            cd = self._existing.params["constraints"]
            if cd:
                self.cons_enable.setChecked(True)
                self.min_sb.setText(str(cd.get("min", 0.0)))
                self.max_sb.setText(str(cd.get("max", 0.0)))

        cons_layout.addWidget(cons_group)
        cons_layout.addStretch()
        self.tabs.addTab(cons, "Constraints")

        # --- Noise Tab ---
        noise_tab = QWidget()
        noise_layout = QVBoxLayout(noise_tab)
        noise_layout.setSpacing(15)
        noise_layout.setContentsMargins(10, 10, 10, 10)

        # Noise settings
        noise_group = QGroupBox("Noise Settings")
        noise_form = QFormLayout(noise_group)
        noise_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        noise_form.setHorizontalSpacing(15)
        noise_form.setVerticalSpacing(10)

        initial_noise = self._existing.noise if self._existing and self._existing.noise else NoiseConfig()

        self.no_j = QLineEdit(str(initial_noise.j))
        self.no_j.setValidator(create_float_validator())
        self.no_rel = QCheckBox("Relative")
        self.no_rel.setChecked(initial_noise.relative)
        j_layout = QHBoxLayout()
        j_layout.addWidget(self.no_j)
        j_layout.addWidget(self.no_rel)

        noise_form.addRow("j:", j_layout)
        noise_layout.addWidget(noise_group)

        # Perturbation settings
        initial_pert = self._existing.perturb if self._existing and self._existing.perturb else PerturbationConfig()

        pert_group = QGroupBox("Perturbation Settings")
        pert_form = QFormLayout(pert_group)
        pert_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        pert_form.setHorizontalSpacing(15)
        pert_form.setVerticalSpacing(10)

        self.per_j_mean = QLineEdit(str(initial_pert.j_mean))
        self.per_j_mean.setValidator(create_float_validator())
        self.per_j_std = QLineEdit(str(initial_pert.j_std))
        self.per_j_std.setValidator(create_float_validator())

        j_layout_p = QHBoxLayout()
        j_layout_p.addWidget(self.per_j_mean)
        j_layout_p.addWidget(self.per_j_std)
        pert_form.addRow("j (mean, std):", j_layout_p)
        noise_layout.addWidget(pert_group)

        noise_layout.addStretch()
        self.tabs.addTab(noise_tab, "Noise")

        layout.addWidget(self.tabs)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        if self._existing is not None:
            delete_btn = btns.addButton("Delete Connection", QDialogButtonBox.ButtonRole.DestructiveRole)
            delete_btn.clicked.connect(self._on_delete)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Rebuild when type or init changes
        self.type_cb.currentTextChanged.connect(self._build_params_tab)
        self.init_cb.currentTextChanged.connect(self._build_params_tab)
        self.mode_cb.currentTextChanged.connect(self._update_mode_visibility)

        # Update init description when init method changes
        self.init_cb.currentTextChanged.connect(self._on_init_method_changed)

        # Initial check for layer compatibility
        self._check_layer_compatibility()

        # Initial connectivity type information
        self._on_connectivity_type_changed(self.type_cb.currentText())
        # Initial init method description
        self._on_init_method_changed(self.init_cb.currentText())
        # Initial mode UI visibility
        self._update_mode_visibility(self.mode_cb.currentText())

    def _on_delete(self) -> None:
        """Handle delete button click to request connection deletion."""
        if not self._existing:
            return
        text = f"Remove connection {self._existing.from_layer} → {self._existing.to_layer}?"
        resp = QMessageBox.question(
            self,
            "Delete Connection",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp == QMessageBox.StandardButton.Yes:
            self.deleteRequested = True
            self.accept()

    def _on_connectivity_type_changed(self, conn_type: str) -> None:
        """Update description when connectivity type changes."""
        normalized = normalize_connectivity_type(conn_type)
        # Update description
        if normalized in CONNECTIVITY_DESCRIPTIONS:
            description = CONNECTIVITY_DESCRIPTIONS[normalized].get("description", "")
            self.conn_desc_label.setText(description)
            self.conn_desc_label.setVisible(bool(description))
        else:
            self.conn_desc_label.setText("")
            self.conn_desc_label.setVisible(False)

    def _on_init_method_changed(self, init_name: str) -> None:
        """Update init method description when init method changes."""
        if init_name in WEIGHT_INITIALIZER_DESCRIPTIONS:
            description = WEIGHT_INITIALIZER_DESCRIPTIONS[init_name].get("description", "")
            self.init_desc_label.setText(description)
            self.init_desc_label.setVisible(bool(description))
        else:
            self.init_desc_label.setText("")
            self.init_desc_label.setVisible(False)

    def _check_layer_compatibility(self) -> None:
        """Check layer compatibility for connectivity types and show warnings if needed."""
        from_layer_id = self.from_cb.currentData(Qt.ItemDataRole.UserRole)
        to_layer_id = self.to_cb.currentData(Qt.ItemDataRole.UserRole)

        if from_layer_id is None or to_layer_id is None:
            return

        is_same_layer = from_layer_id == to_layer_id

        # Only enable the self-connections checkbox if it's an internal connection
        self.self_conn_cb.setEnabled(is_same_layer)

        if not is_same_layer:
            # For external connections, self-connections are not applicable
            self.self_conn_cb.setToolTip("Self-connections only apply to internal connections (same layer)")
        else:
            self.self_conn_cb.setToolTip("Enable/disable diagonal elements in the connectivity matrix")

        # Informative hint for internal connections, but keep the dropdown enabled so users can explore settings.
        if is_same_layer:
            self.mode_hint.setText("Internal programmable connectivity is configured in the layer settings (connectivity_mode).")
        else:
            # Update hint based on current mode
            mode = self.mode_cb.currentText().lower()
            if mode == "dynamic":
                self.mode_hint.setText("Dynamic mode uses full multiplier dynamics integration (slower, exact).")
            else:
                self.mode_hint.setText("Fixed mode uses standard matrix multiply (fastest, no dynamics).")

        # Check for one-to-one dimension mismatch warning
        conn_type = self.type_cb.currentText()
        if conn_type == "one_to_one":
            # Find the layer dimensions
            from_layer = next((layer_item for layer_item in self._layers if layer_item.layer_id == from_layer_id), None)
            to_layer = next((layer_item for layer_item in self._layers if layer_item.layer_id == to_layer_id), None)

            if from_layer and to_layer:
                from_dim = from_layer.params.get("dim", 0)
                to_dim = to_layer.params.get("dim", 0)

                if from_dim != to_dim:
                    min_dim = min(from_dim, to_dim)
                    warning_text = f"⚠️ Warning: One-to-one connectivity with mismatched dimensions (from: {from_dim}, to: {to_dim}). Will create {min_dim} diagonal connections."
                    self.conn_warning_label.setText(warning_text)
                    self.conn_warning_label.setVisible(True)
                else:
                    self.conn_warning_label.setVisible(False)
            else:
                self.conn_warning_label.setVisible(False)
        else:
            self.conn_warning_label.setVisible(False)

    def _get_param_type_info(self, conn_type: str, param_name: str) -> dict[str, Any]:
        """Get parameter type information from registries or defaults."""
        # First check in CONNECTIVITY_PARAM_TYPES (e.g., from connectivity.py)
        if conn_type in CONNECTIVITY_PARAM_TYPES and param_name in CONNECTIVITY_PARAM_TYPES[conn_type]:
            return CONNECTIVITY_PARAM_TYPES[conn_type][param_name]

        # Check known enum parameters
        if param_name in DEFAULT_ENUM_PARAMS:
            return {
                "type": "enum",
                "options": DEFAULT_ENUM_PARAMS[param_name],
                "default": DEFAULT_ENUM_PARAMS[param_name][0],
            }

        # Check known numeric parameters
        if param_name in DEFAULT_NUMERIC_RANGES:
            min_val, max_val, decimals, step, default = DEFAULT_NUMERIC_RANGES[param_name]
            return {
                "type": "float",
                "min": min_val,
                "max": max_val,
                "decimals": decimals,
                "step": step,
                "default": default,
            }

        # Default handling based on parameter name
        if "size" in param_name.lower() or "count" in param_name.lower() or "block" in param_name.lower():
            return {
                "type": "int",
                "min": 1,
                "max": 1000,
                "default": 1,
            }

        # Default to float
        return {
            "type": "float",
            "min": -1e6,
            "max": 1e6,
            "decimals": 3,
            "step": 0.05,
            "default": 0.0,
        }

    def _update_mode_visibility(self, text: str) -> None:
        """Update hint text based on mode selection."""
        mode = str(text).lower()
        if mode == "dynamic":
            self.mode_hint.setText("Dynamic mode uses full multiplier dynamics integration (slower, exact).")
            # Show dynamic settings when in dynamic mode
            if hasattr(self, "dyn_settings_container"):
                self.dyn_settings_container.setVisible(True)
                # Update multiplier version visibility
                self._update_multiplier_version_visibility(self.mult_version_cb.currentText())
        else:  # Fixed
            self.mode_hint.setText("Fixed mode uses standard matrix multiply (fastest, no dynamics).")
            # Hide dynamic settings when in fixed mode
            if hasattr(self, "dyn_settings_container"):
                self.dyn_settings_container.setVisible(False)

    def _update_multiplier_version_visibility(self, version: str) -> None:
        """Update visibility of multiplier WICC/NOCC parameters based on selection."""
        if "NOCC" in version:
            self.mult_v1_label.setVisible(False)
            self.mult_gamma_le.setVisible(False)
            self.mult_bias_le.setVisible(False)
            self.mult_v1_j_in_le.setVisible(False)
            self.mult_v1_j_out_info.setVisible(False)
            self.mult_v2_label.setVisible(True)
            self.mult_alpha_le.setVisible(True)
            self.mult_beta_le.setVisible(True)
            self.mult_beta_out_le.setVisible(True)
            self.mult_ib_le.setVisible(True)
            self.mult_v2_j_in_le.setVisible(True)
            self.mult_v2_j_out_le.setVisible(True)
        else:  # WICC
            self.mult_v1_label.setVisible(True)
            self.mult_gamma_le.setVisible(True)
            self.mult_bias_le.setVisible(True)
            self.mult_v1_j_in_le.setVisible(True)
            self.mult_v1_j_out_info.setVisible(True)
            self.mult_v2_label.setVisible(False)
            self.mult_alpha_le.setVisible(False)
            self.mult_beta_le.setVisible(False)
            self.mult_beta_out_le.setVisible(False)
            self.mult_ib_le.setVisible(False)
            self.mult_v2_j_in_le.setVisible(False)
            self.mult_v2_j_out_le.setVisible(False)

    def _create_file_selector_widget(self, current_value: str = "", file_filter: str = "All Files (*)") -> QWidget:
        """Create a simple file selector widget used for file-type params."""
        from PyQt6.QtWidgets import QFileDialog

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        line_edit = QLineEdit(current_value)
        line_edit.setPlaceholderText("Select file or enter path...")

        browse_btn = QPushButton("Browse...")

        def browse() -> None:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
            if file_path:
                line_edit.setText(file_path)

        browse_btn.clicked.connect(browse)
        layout.addWidget(line_edit)
        layout.addWidget(browse_btn)
        # Expose line_edit for value extraction
        setattr(container, "line_edit", line_edit)
        return container

    def _create_param_widget(self, conn_type: str, param_name: str, existing_value: Any = None) -> QComboBox | QSpinBox | QLineEdit | QWidget:
        """Create appropriate widget based on parameter type information."""
        param_info = self._get_param_type_info(conn_type, param_name)
        widget_type = param_info.get("type", "float")

        if widget_type == "dynamic_float_list":
            # Create dynamic list widget that updates based on another parameter
            depends_on = param_info.get("depends_on")
            defaults = param_info.get("defaults", [1.0, 0.5, 0.25, 0.125])
            min_val = param_info.get("min", 0.0)
            max_val = param_info.get("max", 1.0)
            decimals = param_info.get("decimals", 2)
            step = param_info.get("step", 0.05)

            # Determine initial count from dependency parameter
            initial_count = 3  # default
            if depends_on and depends_on in self.struct_boxes:
                dep_widget = self.struct_boxes[depends_on]
                if isinstance(dep_widget, QSpinBox):
                    initial_count = dep_widget.value()

            # Get existing values if available
            existing_list = None
            if isinstance(existing_value, list):
                existing_list = existing_value

            widget = DynamicFloatListWidget(
                count=initial_count,
                defaults=defaults,
                min_val=min_val,
                max_val=max_val,
                decimals=decimals,
                step=step,
                existing_values=existing_list,
            )

            # Connect to dependency parameter if it exists
            if depends_on and depends_on in self.struct_boxes:
                dep_widget = self.struct_boxes[depends_on]
                if isinstance(dep_widget, QSpinBox):
                    dep_widget.valueChanged.connect(widget.update_count)

            return widget

        if widget_type == "file":
            # Create file selector widget
            extensions = param_info.get("extensions", [])
            if extensions:
                ext_filter = f"Files ({' '.join(extensions)})"
            else:
                ext_filter = "All Files (*)"

            current_val = str(existing_value) if existing_value else ""
            return self._create_file_selector_widget(current_val, ext_filter)

        if widget_type == "enum":
            widget = QComboBox()
            widget.addItems(param_info.get("options", []))

            # Set existing value if available
            if existing_value is not None:
                index = widget.findText(str(existing_value))
                if index >= 0:
                    widget.setCurrentIndex(index)
                else:
                    # Set to default if value not found
                    default_index = widget.findText(str(param_info.get("default", "")))
                    if default_index >= 0:
                        widget.setCurrentIndex(default_index)
            elif "default" in param_info:
                default_index = widget.findText(str(param_info["default"]))
                if default_index >= 0:
                    widget.setCurrentIndex(default_index)

            return widget

        if widget_type == "int":
            widget = QSpinBox()
            widget.setRange(param_info.get("min", 1), param_info.get("max", 1000))
            if "step" in param_info:
                widget.setSingleStep(param_info["step"])

            # Set value
            if existing_value is not None:
                try:
                    widget.setValue(int(existing_value))
                except (ValueError, TypeError):
                    widget.setValue(param_info.get("default", 1))
            else:
                widget.setValue(param_info.get("default", 1))

            return widget

        # float -> Use QLineEdit
        widget = QLineEdit()
        # Use general wide range, ignore specific ranges from param_info for now
        # min_val = param_info.get("min", FLOAT_MIN) # Or use FLOAT_MIN always? Using FLOAT_MIN for now.
        # max_val = param_info.get("max", FLOAT_MAX)
        widget.setValidator(create_float_validator())

        # Set value using setText
        default_val = param_info.get("default", 0.0)
        value_to_set = default_val
        if existing_value is not None:
            try:
                value_to_set = float(existing_value)
            except (ValueError, TypeError):
                value_to_set = default_val  # Fallback to default

        widget.setText(str(value_to_set))

        return widget

    def _build_params_tab(self) -> None:
        # Remove any existing "Parameters" tab
        for idx in range(self.tabs.count()):
            if self.tabs.tabText(idx) == "Parameters":
                self.tabs.removeTab(idx)
                break

        # Create new Parameters tab
        params_tab = QWidget()
        v = QVBoxLayout(params_tab)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(15)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        cv = QVBoxLayout(container)
        cv.setSpacing(15)
        cv.setContentsMargins(0, 0, 0, 0)

        # --- Get connection type ---
        conn_type = self.type_cb.currentText()
        normalized_type = normalize_connectivity_type(conn_type)
        self.struct_boxes = {}
        struct_existing = self._existing_structure_params.copy()
        if not struct_existing and self._existing and self._existing.params:
            struct_existing = self._existing.params

        # --- Get parameters from CONNECTIVITY_PARAM_TYPES ---
        param_keys = []
        if normalized_type in CONNECTIVITY_PARAM_TYPES:
            param_keys = list(CONNECTIVITY_PARAM_TYPES[normalized_type].keys())

            # Sort parameters to ensure dependencies come before dependents
            # Parameters with "depends_on" should come after their dependencies
            # Preserve dictionary order for parameters with same priority
            # Create stable index mapping before sorting
            original_indices = {name: idx for idx, name in enumerate(param_keys)}

            def param_sort_key(param_name: str) -> tuple[int, int]:
                param_info = CONNECTIVITY_PARAM_TYPES[normalized_type].get(param_name, {})
                # If this param depends on another, it should come later
                depends_on = param_info.get("depends_on")
                original_index = original_indices.get(param_name, 999)
                if depends_on:
                    return (1, original_index)  # Dependent parameters second, preserve order
                return (0, original_index)  # Independent parameters first, preserve order

            param_keys.sort(key=param_sort_key)

        # Filter struct_existing to only include valid parameters for this connection type
        # This prevents stale parameters from previous connection type selections from showing
        valid_param_keys = set(param_keys)
        struct_existing = {k: v for k, v in struct_existing.items() if k in valid_param_keys}

        # Get parameter descriptions
        param_descriptions = {}
        if normalized_type in CONNECTIVITY_DESCRIPTIONS:
            param_descriptions = CONNECTIVITY_DESCRIPTIONS[normalized_type].get("params", {})

        # Create the structural parameters interface
        if param_keys:
            # Add header with description
            header = QLabel("Connection Structure Parameters")
            header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            cv.addWidget(header)

            struct_desc = QLabel("Configure the connectivity pattern between layers.")
            struct_desc.setWordWrap(True)
            struct_desc.setStyleSheet("font-style: italic; color: #666; margin-bottom: 10px; background-color: transparent;")
            cv.addWidget(struct_desc)

            # Create group for structure parameters
            struct_group = QGroupBox()
            struct_layout = QVBoxLayout(struct_group)
            struct_layout.setSpacing(12)
            struct_layout.setContentsMargins(10, 10, 10, 10)

            for k in param_keys:
                param_container = QWidget()
                param_layout = QVBoxLayout(param_container)
                param_layout.setContentsMargins(0, 0, 0, 0)
                param_layout.setSpacing(6)

                # Parameter name
                param_name = k.replace("_", " ").capitalize()
                param_label = QLabel(f"{param_name}:")
                param_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                param_layout.addWidget(param_label)

                # Parameter description
                if k in param_descriptions:
                    desc_label = QLabel(param_descriptions[k])
                    desc_label.setWordWrap(True)
                    desc_label.setStyleSheet("font-style: italic; color: #666; margin-left: 10px; font-size: 9pt; background-color: transparent;")
                    desc_label.setMinimumHeight(30)
                    param_layout.addWidget(desc_label)

                # Parameter widget
                existing_value = struct_existing.get(k)
                widget = self._create_param_widget(normalized_type, k, existing_value)
                param_layout.addWidget(widget)

                struct_layout.addWidget(param_container)
                self.struct_boxes[k] = widget

            cv.addWidget(struct_group)
        else:
            # No parameters - show friendly message
            header = QLabel("Connection Structure Parameters")
            header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            cv.addWidget(header)

            no_params_label = QLabel("✓ This connectivity type requires no additional parameters.")
            no_params_label.setWordWrap(True)
            no_params_label.setStyleSheet("color: #4a9; font-style: italic; padding: 10px; background-color: transparent;")
            cv.addWidget(no_params_label)

        # --- Dynamic Connection Settings (wrapped in container for conditional visibility) ---
        self.dyn_settings_container = QWidget()
        dyn_container_layout = QVBoxLayout(self.dyn_settings_container)
        dyn_container_layout.setContentsMargins(0, 0, 0, 0)
        dyn_container_layout.setSpacing(15)

        dyn_header = QLabel("Dynamic Connection Settings")
        dyn_header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        dyn_container_layout.addWidget(dyn_header)

        dyn_description = QLabel("Configure the multiplier circuit behavior for dynamic connections.")
        dyn_description.setWordWrap(True)
        dyn_description.setStyleSheet("font-style: italic; color: #666; margin-bottom: 10px; background-color: transparent;")
        dyn_container_layout.addWidget(dyn_description)

        dyn_group = QGroupBox()
        dyn_form = QFormLayout(dyn_group)
        dyn_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        dyn_form.setHorizontalSpacing(20)
        dyn_form.setVerticalSpacing(12)

        self.mult_source_cb = QComboBox()
        self.mult_source_cb.addItems(["RateArray", "Heaviside_state_dep"])
        dyn_form.addRow("Source Function:", self.mult_source_cb)

        # Multiplier version selector (NOCC or WICC)
        self.mult_version_cb = QComboBox()
        self.mult_version_cb.addItems(["NOCC (No Collection Coil)", "WICC (With Collection Coil)"])
        self.mult_version_cb.setToolTip("NOCC: Dual SQUID states, no collection coil | WICC: Single state, with collection coil")
        self.mult_version_cb.currentTextChanged.connect(self._update_multiplier_version_visibility)
        dyn_form.addRow("Multiplier Version:", self.mult_version_cb)

        # WICC-specific parameters
        self.mult_v1_label = QLabel("WICC Parameters")
        bold_font = QFont()
        bold_font.setBold(True)
        self.mult_v1_label.setFont(bold_font)
        dyn_form.addRow(self.mult_v1_label)

        self.mult_gamma_le = QLineEdit("0.001")
        self.mult_gamma_le.setValidator(create_float_validator())
        self.mult_bias_le = QLineEdit("2.0")
        self.mult_bias_le.setValidator(create_float_validator())
        self.mult_v1_j_in_le = QLineEdit("0.38")
        self.mult_v1_j_in_le.setValidator(create_float_validator())
        self.mult_v1_j_out_info = QLabel("Auto-computed from destination fan-in (not editable).")
        self.mult_v1_j_out_info.setWordWrap(True)
        self.mult_v1_j_out_info.setStyleSheet("color: #666; font-style: italic;")

        dyn_form.addRow("Gamma Plus:", self.mult_gamma_le)
        dyn_form.addRow("Bias Current:", self.mult_bias_le)
        dyn_form.addRow("J_in:", self.mult_v1_j_in_le)
        dyn_form.addRow("J_out:", self.mult_v1_j_out_info)

        # NOCC-specific parameters
        self.mult_v2_label = QLabel("NOCC Parameters")
        self.mult_v2_label.setFont(bold_font)
        dyn_form.addRow(self.mult_v2_label)

        self.mult_alpha_le = QLineEdit("1.64053")
        self.mult_alpha_le.setValidator(create_float_validator())
        self.mult_beta_le = QLineEdit("303.85")
        self.mult_beta_le.setValidator(create_float_validator())
        self.mult_beta_out_le = QLineEdit("91.156")
        self.mult_beta_out_le.setValidator(create_float_validator())
        self.mult_ib_le = QLineEdit("2.1")
        self.mult_ib_le.setValidator(create_float_validator())
        self.mult_v2_j_in_le = QLineEdit("0.38")
        self.mult_v2_j_in_le.setValidator(create_float_validator())
        self.mult_v2_j_out_le = QLineEdit("0.38")
        self.mult_v2_j_out_le.setValidator(create_float_validator())

        dyn_form.addRow("Alpha:", self.mult_alpha_le)
        dyn_form.addRow("Beta:", self.mult_beta_le)
        dyn_form.addRow("Beta Out:", self.mult_beta_out_le)
        dyn_form.addRow("Ib:", self.mult_ib_le)
        dyn_form.addRow("J_in:", self.mult_v2_j_in_le)
        dyn_form.addRow("J_out:", self.mult_v2_j_out_le)

        # Half-flux quantum offset toggle
        self.half_flux_cb = QCheckBox("Half-flux quantum phi_y offset (+0.5)")
        self.half_flux_cb.setToolTip("If enabled, adds +0.5 to phi_y just before source function computation for this connection.")
        self.half_flux_cb.setChecked(True)  # Default to enabled
        dyn_form.addRow(self.half_flux_cb)

        dyn_container_layout.addWidget(dyn_group)
        cv.addWidget(self.dyn_settings_container)

        # Load existing dynamic settings if available
        if self._existing:
            try:
                existing_params = self._existing.params or {}
                # Check both new API ("connection_params") and legacy ("dynamic"/"multiplier") keys
                mult_cfg = (
                    existing_params.get("connection_params")
                    or existing_params.get("dynamic")
                    or existing_params.get("multiplier")
                    or {}
                )

                # Also check the explicit "mode" key to determine WICC vs NOCC
                explicit_mode = existing_params.get("mode", "").upper()

                src = mult_cfg.get("source_func") or "RateArray"
                idx_src = self.mult_source_cb.findText(str(src))
                if idx_src >= 0:
                    self.mult_source_cb.setCurrentIndex(idx_src)

                # Detect whether WICC or NOCC based on:
                # 1. Explicit mode key (preferred)
                # 2. Presence of NOCC-specific params (alpha, beta)
                # 3. Presence of WICC-specific params (gamma_plus)
                is_nocc = (
                    explicit_mode == "NOCC"
                    or mult_cfg.get("alpha") is not None
                    or mult_cfg.get("beta") is not None
                )
                is_wicc = (
                    explicit_mode == "WICC"
                    or mult_cfg.get("gamma_plus") is not None
                )

                # NOCC takes priority if both are somehow set
                if is_nocc:
                    self.mult_version_cb.setCurrentText("NOCC (No Collection Coil)")
                    if mult_cfg.get("alpha") is not None:
                        self.mult_alpha_le.setText(str(mult_cfg.get("alpha")))
                    if mult_cfg.get("beta") is not None:
                        self.mult_beta_le.setText(str(mult_cfg.get("beta")))
                    if mult_cfg.get("beta_out") is not None:
                        self.mult_beta_out_le.setText(str(mult_cfg.get("beta_out")))
                    if mult_cfg.get("bias_current") is not None:
                        self.mult_ib_le.setText(str(mult_cfg.get("bias_current")))
                    if mult_cfg.get("j_in") is not None:
                        self.mult_v2_j_in_le.setText(str(mult_cfg.get("j_in")))
                    if mult_cfg.get("j_out") is not None:
                        self.mult_v2_j_out_le.setText(str(mult_cfg.get("j_out")))
                elif is_wicc:
                    self.mult_version_cb.setCurrentText("WICC (With Collection Coil)")
                    if mult_cfg.get("gamma_plus") is not None:
                        self.mult_gamma_le.setText(str(mult_cfg.get("gamma_plus")))
                    if mult_cfg.get("bias_current") is not None:
                        self.mult_bias_le.setText(str(mult_cfg.get("bias_current")))
                    if mult_cfg.get("j_in") is not None:
                        self.mult_v1_j_in_le.setText(str(mult_cfg.get("j_in")))
                # If neither detected and mode is dynamic, default to NOCC (v2)
                elif self.mode_cb.currentText().lower() == "dynamic":
                    self.mult_version_cb.setCurrentText("NOCC (No Collection Coil)")

                self._update_multiplier_version_visibility(self.mult_version_cb.currentText())
                # Load half-flux toggle
                try:
                    self.half_flux_cb.setChecked(bool(mult_cfg.get("half_flux_offset", False)))
                except Exception:
                    self.half_flux_cb.setChecked(False)
            except Exception:
                pass

        # --- Weight initialization parameters ---
        init_header = QLabel("Weight Initialization Parameters")
        init_header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        cv.addWidget(init_header)

        init_desc = QLabel("Configure how connection weights are initialized.")
        init_desc.setWordWrap(True)
        init_desc.setStyleSheet("font-style: italic; color: #666; margin-bottom: 10px; background-color: transparent;")
        cv.addWidget(init_desc)

        init_group = QGroupBox()
        init_layout = QVBoxLayout(init_group)
        init_layout.setContentsMargins(10, 10, 10, 10)

        init_func = WEIGHT_INITIALIZERS[self.init_cb.currentText()]
        init_method_name = self.init_cb.currentText()

        # Add init method description
        if init_method_name in WEIGHT_INITIALIZER_DESCRIPTIONS:
            method_desc = WEIGHT_INITIALIZER_DESCRIPTIONS[init_method_name].get("description", "")
            if method_desc:
                init_method_desc_label = QLabel(method_desc)
                init_method_desc_label.setWordWrap(True)
                init_method_desc_label.setStyleSheet("font-style: italic; color: #666; margin-bottom: 10px; background-color: transparent;")
                init_layout.addWidget(init_method_desc_label)

        # Get parameter descriptions for init method
        init_param_descriptions = {}
        if init_method_name in WEIGHT_INITIALIZER_DESCRIPTIONS:
            init_param_descriptions = WEIGHT_INITIALIZER_DESCRIPTIONS[init_method_name].get("params", {})

        init_existing = {}
        if self._existing:
            sig = inspect.signature(init_func).parameters
            tmp_existing = self._existing_init_params.copy() if self._existing_init_params else struct_existing.copy()
            if self.init_cb.currentText() == "uniform":
                if ("a" in tmp_existing) and ("min" not in tmp_existing):
                    tmp_existing["min"] = tmp_existing["a"]
                if ("b" in tmp_existing) and ("max" not in tmp_existing):
                    tmp_existing["max"] = tmp_existing["b"]
            init_existing = {k: v for k, v in tmp_existing.items() if k in sig}

        self.init_editor = FuncParamEditor(init_func, existing=init_existing, param_descriptions=init_param_descriptions)
        init_layout.addWidget(self.init_editor)
        cv.addWidget(init_group)

        cv.addStretch()
        scroll.setWidget(container)
        v.addWidget(scroll)

        # Insert at position 1
        self.tabs.insertTab(1, params_tab, "Parameters")

        # Update dynamic settings visibility based on current mode
        if hasattr(self, "dyn_settings_container"):
            current_mode = self.mode_cb.currentText().lower()
            self.dyn_settings_container.setVisible(current_mode == "dynamic")

    def _extract_structure_params(self) -> dict[str, Any]:
        """Extract connectivity structure parameters from UI widgets."""
        structure_params: dict[str, Any] = {}
        normalized_type = normalize_connectivity_type(self.type_cb.currentText())
        param_types = CONNECTIVITY_PARAM_TYPES.get(normalized_type, {})
        valid_param_keys = set(param_types.keys())

        # First pass: collect all values
        collected_values: dict[str, Any] = {}
        for param_name, widget in self.struct_boxes.items():
            # Validate parameter name is expected for this connection type
            if param_name not in valid_param_keys:
                continue

            # Dynamic float list widget
            if isinstance(widget, DynamicFloatListWidget):
                collected_values[param_name] = widget.get_values()
            # File selector is a QWidget container with a line_edit attribute
            elif hasattr(widget, "line_edit"):
                line_edit = getattr(widget, "line_edit")
                if isinstance(line_edit, QLineEdit):
                    collected_values[param_name] = line_edit.text()
            elif isinstance(widget, QLineEdit):
                try:
                    collected_values[param_name] = float(widget.text())
                except ValueError:
                    # Keep as string if conversion fails
                    collected_values[param_name] = widget.text()
            elif isinstance(widget, QComboBox):
                collected_values[param_name] = widget.currentText()
            elif isinstance(widget, QSpinBox):
                collected_values[param_name] = widget.value()

        # Second pass: handle one_to_one node range parameters specially
        # These must be sent together or not at all
        if normalized_type == "one_to_one":
            start_val = collected_values.get("source_start_node_id")
            end_val = collected_values.get("source_end_node_id")
            start_default = param_types.get("source_start_node_id", {}).get("default", 0)
            end_default = param_types.get("source_end_node_id", {}).get("default", 0)

            # Only include node range parameters if either is non-default
            if start_val != start_default or end_val != end_default:
                if start_val is not None:
                    structure_params["source_start_node_id"] = start_val
                if end_val is not None:
                    structure_params["source_end_node_id"] = end_val
        else:
            # For other connectivity types, process normally
            for param_name, value in collected_values.items():
                param_info = param_types.get(param_name, {})
                is_optional = param_info.get("optional", False)
                default_value = param_info.get("default")

                # Skip optional parameters that are at their default value
                if is_optional and value == default_value:
                    continue

                structure_params[param_name] = value

        return structure_params

    def _extract_init_params(self) -> dict[str, Any]:
        """Extract weight initialization parameters from FuncParamEditor."""
        init_params = {}
        if hasattr(self, "init_editor"):
            init_params = self.init_editor.values() or {}
            # Handle uniform init special case (a/b vs min/max)
            if self.init_cb.currentText() == "uniform":
                if ("a" in init_params) and ("min" not in init_params):
                    init_params["min"] = init_params.pop("a")
                if ("b" in init_params) and ("max" not in init_params):
                    init_params["max"] = init_params.pop("b")
        return init_params

    def _extract_dynamic_params(self) -> tuple[str, dict[str, Any] | None]:
        """Extract dynamic mode and multiplier parameters.

        Returns:
            Tuple of (mode, dynamic_params) where mode is 'fixed' or 'dynamic'
        """
        mode_val = self.mode_cb.currentText().strip().lower()

        if mode_val in {"dynamic", "multiplier"}:
            try:
                version_text = self.mult_version_cb.currentText()

                if "NOCC" in version_text:
                    # NOCC multiplier parameters
                    dynamic_params = {
                        "source_func": self.mult_source_cb.currentText(),
                        "alpha": float(self.mult_alpha_le.text()),
                        "beta": float(self.mult_beta_le.text()),
                        "beta_out": float(self.mult_beta_out_le.text()),
                        "bias_current": float(self.mult_ib_le.text()),
                        "j_in": float(self.mult_v2_j_in_le.text()),
                        "j_out": float(self.mult_v2_j_out_le.text()),
                        "half_flux_offset": bool(self.half_flux_cb.isChecked()),
                    }
                    return "NOCC", dynamic_params
                else:
                    # WICC multiplier parameters
                    dynamic_params = {
                        "source_func": self.mult_source_cb.currentText(),
                        "gamma_plus": float(self.mult_gamma_le.text()),
                        "bias_current": float(self.mult_bias_le.text()),
                        "j_in": float(self.mult_v1_j_in_le.text()),
                        "half_flux_offset": bool(self.half_flux_cb.isChecked()),
                    }
                    return "WICC", dynamic_params
            except ValueError:
                # Fallback to minimal config if parameters invalid
                dynamic_params = {
                    "source_func": self.mult_source_cb.currentText(),
                    "half_flux_offset": bool(self.half_flux_cb.isChecked()),
                }

            # Default to NOCC if version detection failed
            return "NOCC", dynamic_params
        else:
            return "fixed", None

    def _extract_constraints(self) -> dict[str, float] | None:
        """Extract weight constraint parameters."""
        if not self.cons_enable.isChecked():
            return None

        constraints = {}
        min_val_str = self.min_sb.text()
        max_val_str = self.max_sb.text()

        if min_val_str:
            constraints["min"] = float(min_val_str)
        if max_val_str:
            constraints["max"] = float(max_val_str)

        return constraints if constraints else None

    def _extract_noise_and_perturbation(self) -> tuple[NoiseConfig, PerturbationConfig]:
        """Extract noise and perturbation configurations."""
        noise_cfg = NoiseConfig(
            j=float(self.no_j.text()),
            relative=self.no_rel.isChecked(),
        )
        pert_cfg = PerturbationConfig(
            j_mean=float(self.per_j_mean.text()),
            j_std=float(self.per_j_std.text()),
        )
        return noise_cfg, pert_cfg

    def result_config(self) -> ConnectionConfig | None:
        """Build and return the connection configuration from UI values."""
        try:
            # Extract all parameter groups
            structure_params = self._extract_structure_params()
            init_params = self._extract_init_params()
            mode, dynamic_params = self._extract_dynamic_params()
            constraints = self._extract_constraints()
            noise_cfg, pert_cfg = self._extract_noise_and_perturbation()

            # Build unified params structure
            params = {
                "structure": {
                    "type": self.type_cb.currentText(),
                    "params": structure_params,
                },
                "init": {
                    "name": self.init_cb.currentText(),
                    "params": init_params,
                },
                "allow_self_connections": self.self_conn_cb.isChecked(),
                "mode": mode,
            }

            # Add dynamic params if present (use "connection_params" for new API)
            if dynamic_params is not None:
                params["connection_params"] = dynamic_params

            # Add constraints if present
            if constraints is not None:
                params["constraints"] = constraints

        except ValueError as e:
            QMessageBox.critical(
                self,
                "Invalid Input",
                f"Invalid numeric input for a parameter: {e}. Please use standard (e.g., 0.01) or scientific (e.g., 1e-2) notation.",
            )
            return None

        # Get layer IDs
        from_layer_id = self.from_cb.currentData(Qt.ItemDataRole.UserRole)
        to_layer_id = self.to_cb.currentData(Qt.ItemDataRole.UserRole)

        return ConnectionConfig(
            from_layer=int(from_layer_id),
            to_layer=int(to_layer_id),
            connection_type=self.type_cb.currentText(),
            params=params,
            learnable=self.learn_cb.isChecked(),
            noise=noise_cfg,
            perturb=pert_cfg,
        )
