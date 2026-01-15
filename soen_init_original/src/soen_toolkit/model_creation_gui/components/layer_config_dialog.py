# FILE: src/soen_toolkit/model_creation_gui/components/layer_config_dialog.py
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QColor,
    QDoubleValidator,
    QStandardItem,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from soen_toolkit.core.configs import LayerConfig, NoiseConfig, PerturbationConfig
from soen_toolkit.core.layers.common.metadata import (
    LAYER_PARAM_CONFIGS,
    noise_keys_for_layer,
)
from soen_toolkit.core.source_functions import SOURCE_FUNCTIONS
from soen_toolkit.model_creation_gui.components.constraints_editor import (
    ConstraintsEditor,
)
from soen_toolkit.model_creation_gui.components.param_editor import ParamEditor
from soen_toolkit.model_creation_gui.model_manager import (
    _get_layer_types_discovered,
    _get_source_functions_discovered,
    discover_layer_types,
)
from soen_toolkit.utils.polarity_utils import (
    POLARITY_ENFORCEMENT_CLIP,
    POLARITY_ENFORCEMENT_DEFAULT,
    POLARITY_ENFORCEMENT_SIGN_FLIP,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

ODE_LAYER_TYPES_REQUIRING_SOLVER = frozenset(
    {
        "SingleDendrite",
        "Multiplier",
        "MultiplierNOCC",
        "DendriteReadout",
    }
)
LAYERS_REQUIRING_SOURCE_FUNCTION = frozenset(
    set(ODE_LAYER_TYPES_REQUIRING_SOLVER) | {"NonLinear"},
)
# Layers that only support Forward Euler solver
LAYERS_ONLY_FE_SOLVER = frozenset(
    {
        "Multiplier",
        "MultiplierNOCC",
    }
)




class LayerConfigDialog(QDialog):
    def __init__(
        self,
        parent=None,
        existing: LayerConfig | None = None,
        default_id: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Layer Configuration")
        self.setModal(True)
        self._existing = existing
        self._default_id = default_id
        self.deleteRequested = False
        # Track previous type to avoid resetting defaults during initial load for existing layers
        self._prev_type: str | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        self.resize(850, 750)
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        self._float_validator = QDoubleValidator()
        self._float_validator.setNotation(QDoubleValidator.Notation.ScientificNotation)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Create dummy type combo box to be used by other tabs during init
        self.type_cb = QComboBox()
        if self._existing:
            # Set text so other components can see it, but don't add items yet
            self.type_cb.setCurrentText(self._existing.layer_type)

        # Create tabs that might depend on the initial type
        self._create_parameters_tab()
        self._create_noise_tab()
        self._create_constraints_tab()
        # This one populates the type_cb and MUST be called after others use the placeholder
        self._create_basic_tab()

        self.tabs.setCurrentIndex(0)

        # Connections
        self.type_cb.currentTextChanged.connect(self._on_type_change)
        self.solver_cb.currentTextChanged.connect(self._on_solver_change)
        self.source_cb.currentTextChanged.connect(self._on_source_change)

        # Initial state setup based on the final, populated type_cb
        # Avoid resetting source function to defaults during initialisation for existing layers
        self._prev_type = self._current_layer_type()
        self._on_type_change(self.type_cb.currentText())
        self._on_solver_change(self.solver_cb.currentText())
        self._on_source_change(self.source_cb.currentText())

        main_layout.addWidget(self.tabs)

        # Dialog Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        if self._existing is not None:
            delete_btn = buttons.addButton("Delete Layer", QDialogButtonBox.ButtonRole.DestructiveRole)
            delete_btn.clicked.connect(self._on_delete)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def _create_basic_tab(self) -> None:
        tab = QWidget()
        v_layout = QVBoxLayout(tab)
        v_layout.setSpacing(15)
        v_layout.setContentsMargins(10, 10, 10, 10)

        # --- Layer Identity Section ---
        identity_group = QGroupBox("Layer Identity")
        identity_form = QFormLayout(identity_group)
        identity_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        identity_form.setHorizontalSpacing(20)
        identity_form.setVerticalSpacing(12)

        # Layer ID
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 9999)
        self.id_spin.setToolTip("Unique identifier for this layer within the network. Used for connections and referencing.")
        if self._existing:
            self.id_spin.setValue(self._existing.layer_id)
        elif self._default_id is not None:
            self.id_spin.setValue(self._default_id)
        identity_form.addRow("Layer ID:", self.id_spin)

        # Model ID (for grouping / provenance)
        self.model_id_spin = QSpinBox()
        self.model_id_spin.setRange(0, 999999)
        self.model_id_spin.setToolTip("Module ID for composing multiple networks into one. Most users can ignore this and leave it as 0.")
        if self._existing and hasattr(self._existing, "model_id"):
            try:
                self.model_id_spin.setValue(int(getattr(self._existing, "model_id", 0)))
            except Exception:
                self.model_id_spin.setValue(0)
        else:
            self.model_id_spin.setValue(0)
        identity_form.addRow("Model ID:", self.model_id_spin)

        v_layout.addWidget(identity_group)

        # --- Layer Configuration Section ---
        config_group = QGroupBox("Layer Configuration")
        config_form = QFormLayout(config_group)
        config_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        config_form.setHorizontalSpacing(20)
        config_form.setVerticalSpacing(12)

        # Layer Type (re-uses self.type_cb and populates it)
        layer_types = self._populate_type_combo()
        self.layer_desc_label = QLabel()
        self.layer_desc_label.setWordWrap(True)
        self.layer_desc_label.setStyleSheet("font-style: italic; color: #999; margin-top: 5px; margin-left: 10px; background-color: transparent;")
        self.layer_desc_label.setMinimumHeight(40)
        self.layer_desc_label.setVisible(False)
        if self._existing:
            self._set_layer_current(self._existing.layer_type)
        elif "Linear" in layer_types:
            self._set_layer_current("Linear")
        self._update_layer_description()
        config_form.addRow("Type:", self.type_cb)
        config_form.addRow("", self.layer_desc_label)
        self.type_cb.currentIndexChanged.connect(self._update_layer_description)
        self.type_cb.currentTextChanged.connect(self._on_type_change)

        # Dimension
        self.dim_spin = QSpinBox()
        self.dim_spin.setRange(1, 1000000000)
        self.dim_spin.setToolTip("Number of neurons/units in this layer. Determines the layer's output size.")
        if self._existing:
            self.dim_spin.setValue(self._existing.params.get("dim", 1))
        config_form.addRow("Dimension:", self.dim_spin)

        # Neuron Polarity (Dale's Principle)
        self.polarity_cb = QComboBox()
        self.polarity_cb.addItem("None (unrestricted)", None)
        self.polarity_cb.addItem("Alternating (50:50)", "alternating")
        self.polarity_cb.addItem("Excitatory (All +1)", "excitatory")
        self.polarity_cb.addItem("Inhibitory (All -1)", "inhibitory")
        self.polarity_cb.setToolTip(
            "Enforce excitatory/inhibitory constraints on neuron outputs.\n"
            "• Alternating: [Excitatory, Inhibitory, Excitatory, ...]\n"
            "• Excitatory: all outgoing weights ≥ 0\n"
            "• Inhibitory: all outgoing weights ≤ 0"
        )
        if self._existing:
            polarity_init = self._existing.params.get("polarity_init")
            if polarity_init in {"alternating", "50_50"}:
                self.polarity_cb.setCurrentIndex(1)
            elif polarity_init == "excitatory":
                self.polarity_cb.setCurrentIndex(2)
            elif polarity_init == "inhibitory":
                self.polarity_cb.setCurrentIndex(3)
            else:
                self.polarity_cb.setCurrentIndex(0)
        config_form.addRow("Neuron Polarity:", self.polarity_cb)

        # Polarity Enforcement Method (only visible when polarity is active)
        self.polarity_enforcement_label = QLabel("Enforcement Method:")
        self.polarity_enforcement_cb = QComboBox()
        self.polarity_enforcement_cb.addItem("Sign Flip (preserve magnitude)", POLARITY_ENFORCEMENT_SIGN_FLIP)
        self.polarity_enforcement_cb.addItem("Clip to Zero", POLARITY_ENFORCEMENT_CLIP)
        self.polarity_enforcement_cb.setToolTip(
            "How weights are adjusted at initialization when polarity is active:\n"
            "• Sign Flip: Preserve weight magnitude, adjust sign to match polarity\n"
            "  (excitatory → abs(weight), inhibitory → -abs(weight))\n"
            "• Clip to Zero: Clip violating weights to zero\n"
            "  (excitatory → max(0, weight), inhibitory → min(0, weight))"
        )
        # Set default to sign_flip (index 0) or restore from existing config
        if self._existing:
            existing_method = self._existing.params.get("polarity_enforcement_method", POLARITY_ENFORCEMENT_DEFAULT)
            if existing_method == POLARITY_ENFORCEMENT_CLIP:
                self.polarity_enforcement_cb.setCurrentIndex(1)
            else:
                self.polarity_enforcement_cb.setCurrentIndex(0)
        # Initially hide if polarity is None
        has_polarity = self.polarity_cb.currentData(Qt.ItemDataRole.UserRole) is not None
        self.polarity_enforcement_label.setVisible(has_polarity)
        self.polarity_enforcement_cb.setVisible(has_polarity)
        config_form.addRow(self.polarity_enforcement_label, self.polarity_enforcement_cb)

        # Connect polarity change to update enforcement visibility
        self.polarity_cb.currentIndexChanged.connect(self._on_polarity_change)

        # Description
        self.desc_edit = QLineEdit()
        self.desc_edit.setToolTip("Optional text description for this layer to help identify its purpose.")
        if self._existing and self._existing.description:
            self.desc_edit.setText(self._existing.description)
        config_form.addRow("Description:", self.desc_edit)

        v_layout.addWidget(config_group)



        # --- Advanced Solver Settings Section ---
        solver_settings_group = QGroupBox("Advanced Solver Settings")
        solver_settings_form = QFormLayout(solver_settings_group)
        solver_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        solver_settings_form.setHorizontalSpacing(20)
        solver_settings_form.setVerticalSpacing(12)

        # Solver
        self.solver_label = QLabel("Solver:")
        self.solver_cb = QComboBox()
        self.solver_cb.addItem("Forward Euler", "FE")
        self.solver_cb.addItem("Parallel Scan", "PS")
        self.solver_cb.addItem("ParaRNN (Newton)", "PARARNN")
        self.solver_cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.solver_cb.setToolTip(
            "Numerical solver for ODE layers.\n"
            "• Forward Euler: Sequential O(T) solver, most compatible\n"
            "• Parallel Scan: O(log T) parallel prefix sum (linear recurrences)\n"
            "• ParaRNN: O(log T) Newton-method parallel solver (nonlinear, fixed connectivity only)"
        )
        fm_solver = self.solver_cb.fontMetrics()
        longest_solver = max((fm_solver.horizontalAdvance(self.solver_cb.itemText(i)) for i in range(self.solver_cb.count())), default=150)
        self.solver_cb.view().setMinimumWidth(longest_solver + 30)
        if self._existing:
            current_solver = self._existing.params.get("solver", "FE")
            self._set_solver_current(current_solver)
        solver_settings_form.addRow(self.solver_label, self.solver_cb)

        # Adaptive Solver Method
        self.adapt_label = QLabel("Adaptive method:")
        self.adapt_cb = QComboBox()
        self.adapt_cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.adapt_cb.setToolTip("Integration method for adaptive solver: dopri5 (Dormand-Prince), rk4 (Runge-Kutta 4th order), euler (Euler method)")
        self.adapt_cb.fontMetrics()
        longest_adapt = 150
        self.adapt_cb.view().setMinimumWidth(longest_adapt + 30)
        solver_settings_form.addRow(self.adapt_label, self.adapt_cb)

        # Non-Linearity
        self.source_label = QLabel("Non-Linearity:")
        self.source_cb = QComboBox()
        self._populate_source_combo()
        self.source_desc_label = QLabel()
        self.source_desc_label.setWordWrap(True)
        self.source_desc_label.setStyleSheet("font-style: italic; color: #999; margin-top: 5px; margin-left: 10px; background-color: transparent;")
        self.source_desc_label.setMinimumHeight(40)
        self.source_desc_label.setVisible(False)
        initial_source = self._determine_default_source()
        self._set_source_current(initial_source)
        solver_settings_form.addRow(self.source_label, self.source_cb)
        solver_settings_form.addRow("", self.source_desc_label)

        v_layout.addWidget(solver_settings_group)
        v_layout.addStretch()
        self.tabs.insertTab(0, tab, "Basic")

    def _create_parameters_tab(self) -> None:
        tab = QWidget()
        self.params_layout = QVBoxLayout(tab)
        self.params_layout.setSpacing(15)
        self.params_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.params_v_layout = QVBoxLayout(container)
        self.params_v_layout.setSpacing(15)
        self.params_v_layout.setContentsMargins(0, 0, 0, 0)

        self.param_editor_container = QWidget()
        self.param_editor_container.setLayout(QVBoxLayout())
        self.params_v_layout.addWidget(self.param_editor_container)

        self.params_v_layout.addStretch()

        container.setLayout(self.params_v_layout)
        scroll.setWidget(container)
        self.params_layout.addWidget(scroll)
        self.tabs.addTab(tab, "Parameters")

    def _create_noise_tab(self) -> None:
        tab = QWidget()
        self.noise_params_layout = QVBoxLayout(tab)
        self.noise_params_layout.setSpacing(15)
        self.noise_params_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.noise_params_v_layout = QVBoxLayout(container)
        self.noise_params_v_layout.setSpacing(15)
        self.noise_params_v_layout.setContentsMargins(0, 0, 0, 0)

        self.noise_param_editor_container = QWidget()
        self.noise_param_editor_container.setLayout(QVBoxLayout())
        self.noise_params_v_layout.addWidget(self.noise_param_editor_container)

        self._noise_param_validator = QDoubleValidator()
        self._noise_param_validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self._noise_boxes = {}
        self._pert_mean_boxes = {}
        self._pert_std_boxes = {}

        self.noise_group = self._create_noise_group()
        self.pert_group = self._create_pert_group()
        self.noise_params_v_layout.addWidget(self.noise_group)
        self.noise_params_v_layout.addWidget(self.pert_group)
        self.noise_params_v_layout.addStretch()

        container.setLayout(self.noise_params_v_layout)
        scroll.setWidget(container)
        self.noise_params_layout.addWidget(scroll)
        self.tabs.addTab(tab, "Noise")

    def _create_noise_group(self) -> QGroupBox:
        group = QGroupBox("Noise Settings")
        self.noise_param_form = QFormLayout(group)
        self.noise_param_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.noise_param_form.setHorizontalSpacing(20)
        self.noise_param_form.setVerticalSpacing(12)
        self.relative_cb = QCheckBox("Apply Noise Relative to Parameter Values")
        self.noise_param_form.addRow(self.relative_cb)
        return group

    def _create_pert_group(self) -> QGroupBox:
        group = QGroupBox("Perturbation Settings")
        self.noise_pert_form = QFormLayout(group)
        self.noise_pert_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.noise_pert_form.setHorizontalSpacing(20)
        self.noise_pert_form.setVerticalSpacing(12)
        return group

    def _create_constraints_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.cons_v_layout = QVBoxLayout(container)
        self.cons_v_layout.setSpacing(15)
        self.cons_v_layout.setContentsMargins(0, 0, 0, 0)

        self.cons_editor_container = QWidget()
        self.cons_editor_container.setLayout(QVBoxLayout())
        self.cons_v_layout.addWidget(self.cons_editor_container)
        self.cons_v_layout.addStretch()

        container.setLayout(self.cons_v_layout)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "Constraints")

    def _on_type_change(self, new_type: str) -> None:
        canonical = self._current_layer_type()
        # Detect true type changes (ignore first call where _prev_type is None)
        is_change = (self._prev_type is not None) and (canonical != self._prev_type)
        self._update_param_editor(canonical)
        self._update_constraints_editor(canonical)
        self._update_dynamic_basic_settings(canonical)
        self._update_noise_and_pert_widgets(canonical)
        self._update_layer_description()
        # Only snap to the layer-type default if:
        #  - this is a new layer (no existing), or
        #  - the user actually changed the layer type
        if (self._existing is None) or is_change:
            self._maybe_reset_source_default(canonical)
        # Update tracking state
        self._prev_type = canonical

    def _on_polarity_change(self, _index: int) -> None:
        """Show/hide polarity enforcement method dropdown based on polarity selection."""
        has_polarity = self.polarity_cb.currentData(Qt.ItemDataRole.UserRole) is not None
        self.polarity_enforcement_label.setVisible(has_polarity)
        self.polarity_enforcement_cb.setVisible(has_polarity)

    def _update_dynamic_basic_settings(self, new_type: str) -> None:
        show_solver = new_type in ODE_LAYER_TYPES_REQUIRING_SOLVER
        self.solver_label.setVisible(show_solver)
        self.solver_cb.setVisible(show_solver)
        if not show_solver:
            self._set_solver_current("FE")

        show_source_func = new_type in LAYERS_REQUIRING_SOURCE_FUNCTION
        self.source_label.setVisible(show_source_func)
        self.source_cb.setVisible(show_source_func)
        if not show_source_func:
            self.source_cb.setCurrentIndex(0)
            self.source_desc_label.clear()
            self.source_desc_label.setVisible(False)
        else:
            # For new ODE layers without an explicit source_func, default to 'RateArray'
            try:
                if (self._existing is None) or (not self._existing.params.get("source_func")):
                    idx = self.source_cb.findText("RateArray")
                    if idx >= 0:
                        self.source_cb.setCurrentIndex(idx)
            except Exception:
                pass
            self._update_source_description()

        # Update solver options based on layer type
        if show_solver:
            current_source = self.source_cb.currentText()
            self._on_source_change(current_source)



    def _update_param_editor(self, layer_type: str) -> None:
        while self.param_editor_container.layout().count():
            item = self.param_editor_container.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        layers_config = LAYER_PARAM_CONFIGS.get(layer_type, [])
        if not layers_config:
            info = QLabel("This layer type has no configurable parameters.")
            info.setWordWrap(True)
            info.setStyleSheet("font-style: italic; color: #555;")
            self.param_editor_container.layout().addWidget(info)
            self.param_editor = None
        else:
            self.param_editor = ParamEditor(
                layer_type=layer_type,
                existing=self._existing.params if self._existing else {},
            )
            self.param_editor_container.layout().addWidget(self.param_editor)

    def _update_constraints_editor(self, layer_type: str) -> None:
        while self.cons_editor_container.layout().count():
            item = self.cons_editor_container.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        param_names = [pc.name for pc in LAYER_PARAM_CONFIGS.get(layer_type, [])]
        if not param_names:
            info = QLabel("No parameters available to apply constraints to.")
            info.setWordWrap(True)
            info.setStyleSheet("font-style: italic; color: #555;")
            self.cons_editor_container.layout().addWidget(info)
            self.cons_editor = None
        else:
            existing_cons = {}
            if self._existing:
                for name, cfg in self._existing.params.items():
                    if isinstance(cfg, dict) and cfg.get("constraints"):
                        existing_cons[name] = cfg["constraints"]
            self.cons_editor = ConstraintsEditor(param_names=param_names, existing=existing_cons)
            self.cons_editor_container.layout().addWidget(self.cons_editor)

    def _on_solver_change(self, solver_text: str) -> None:
        self.adapt_label.setVisible(False)
        self.adapt_cb.setVisible(False)

    def _on_source_change(self, source_text: str) -> None:
        """Enable/disable solver options based on source function semantics and layer type.

        - Disallow PS when the chosen source function depends on s via effective bias.
        - Disallow PS for Multiplier and MultiplierNOCC layers (only FE supported).
        - Keep FE and PS available for other ODE layers unless filtered out.
        """
        try:
            layer_type = self._current_layer_type()
            # Only care for ODE layers that expose a solver
            if layer_type not in ODE_LAYER_TYPES_REQUIRING_SOLVER:
                return

            # Check if layer only supports FE
            if layer_type in LAYERS_ONLY_FE_SOLVER:
                allowed = ["FE"]
            else:
                uses_eff_bias = False
                sf_cls = SOURCE_FUNCTIONS.get(source_text)
                if sf_cls is not None:
                    with contextlib.suppress(Exception):
                        uses_eff_bias = bool(getattr(sf_cls(), "uses_squid_current", False))

                allowed = ["FE", "PARARNN"]  # ParaRNN works with all source functions
                if not uses_eff_bias:
                    allowed.insert(1, "PS")  # PS only works without state-dependent sources

            current = self._current_solver_key()
            if current not in allowed:
                self._set_solver_current(allowed[0])

            # Update combo items to reflect allowed list (idempotent refresh)
            current_items = [self.solver_cb.itemData(i) for i in range(self.solver_cb.count())]
            if current_items != allowed:
                self.solver_cb.blockSignals(True)
                self.solver_cb.clear()
                solver_names = {
                    "FE": "Forward Euler",
                    "PS": "Parallel Scan",
                    "PARARNN": "ParaRNN (Newton)",
                }
                for key in allowed:
                    self.solver_cb.addItem(solver_names.get(key, key), key)
                self.solver_cb.blockSignals(False)
                self._set_solver_current(allowed[0])
        except Exception:
            pass

        self._update_source_description()

    def _clear_form_rows(self, layout: QFormLayout, keep_first: bool = False) -> None:
        """Remove all rows from the given QFormLayout.

        Args:
            layout: The QFormLayout to clear.
            keep_first: If True, keep the first row (index 0)
            useful when the
                first row contains a persistent widget like a checkbox.

        """
        start = 1 if keep_first else 0
        for i in reversed(range(start, layout.rowCount())):
            layout.removeRow(i)

    def _update_noise_and_pert_widgets(self, layer_type: str) -> None:
        # Make sure we always have proper dataclass instances (may arrive as plain dicts when loaded from file)
        if self._existing and self._existing.noise:
            initial_noise_raw = self._existing.noise
            if isinstance(initial_noise_raw, dict):
                initial_noise = NoiseConfig(**initial_noise_raw)
            else:
                initial_noise = initial_noise_raw
        else:
            initial_noise = NoiseConfig()

        if self._existing and self._existing.perturb:
            initial_pert_raw = self._existing.perturb
            if isinstance(initial_pert_raw, dict):
                initial_pert = PerturbationConfig(**initial_pert_raw)
            else:
                initial_pert = initial_pert_raw
        else:
            initial_pert = PerturbationConfig()

        self.relative_cb.setChecked(initial_noise.relative)

        self._clear_form_rows(self.noise_param_form, keep_first=True)
        self._clear_form_rows(self.noise_pert_form, keep_first=False)
        self._noise_boxes.clear()
        self._pert_mean_boxes.clear()
        self._pert_std_boxes.clear()

        keys = noise_keys_for_layer(layer_type)
        has_params = bool(keys)
        self.noise_group.setVisible(has_params)
        self.pert_group.setVisible(has_params)

        for key in keys:
            # Noise
            val = getattr(initial_noise, key, 0.0) if hasattr(initial_noise, key) else initial_noise.extras.get(key, 0.0)
            edit = QLineEdit(str(val))
            edit.setValidator(self._noise_param_validator)
            self.noise_param_form.addRow(f"{key}:", edit)
            self._noise_boxes[key] = edit

            # Perturbation
            mean_val = getattr(initial_pert, f"{key}_mean", 0.0) if hasattr(initial_pert, f"{key}_mean") else initial_pert.extras_mean.get(key, 0.0)
            std_val = getattr(initial_pert, f"{key}_std", 0.0) if hasattr(initial_pert, f"{key}_std") else initial_pert.extras_std.get(key, 0.0)
            mean_edit = QLineEdit(str(mean_val))
            mean_edit.setValidator(self._noise_param_validator)
            std_edit = QLineEdit(str(std_val))
            std_edit.setValidator(self._noise_param_validator)

            pert_layout = QHBoxLayout()
            pert_layout.setContentsMargins(0, 0, 0, 0)
            pert_layout.addWidget(mean_edit)
            pert_layout.addWidget(std_edit)
            self.noise_pert_form.addRow(f"{key} (mean, std):", pert_layout)
            self._pert_mean_boxes[key] = mean_edit
            self._pert_std_boxes[key] = std_edit

    def result_config(self) -> LayerConfig | None:
        params = self.param_editor.values() if self.param_editor else {}
        params["dim"] = self.dim_spin.value()

        # Add polarity if selected
        polarity_value = self.polarity_cb.currentData(Qt.ItemDataRole.UserRole)
        if polarity_value is not None:
            params["polarity_init"] = polarity_value
            # Also save the enforcement method
            enforcement_method = self.polarity_enforcement_cb.currentData(Qt.ItemDataRole.UserRole)
            if enforcement_method:
                params["polarity_enforcement_method"] = enforcement_method

        layer_key = self._current_layer_type()
        if layer_key in ODE_LAYER_TYPES_REQUIRING_SOLVER:
            params["solver"] = self._current_solver_key()
        if layer_key in LAYERS_REQUIRING_SOURCE_FUNCTION:
            params["source_func"] = self._current_source_key()



        if self.cons_editor:
            constraints = self.cons_editor.values()
            for name, cons in constraints.items():
                if name in params and isinstance(params[name], dict):
                    params[name]["constraints"] = cons

        try:
            noise_config = self._get_noise_config()
            perturb_config = self._get_perturb_config()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Input", f"Invalid number format: {e}. Please use standard (e.g., 0.01) or scientific (e.g., 1e-2) notation.")
            return None

        return LayerConfig(
            layer_id=self.id_spin.value(),
            model_id=self.model_id_spin.value(),
            layer_type=layer_key,
            params=params,
            description=self.desc_edit.text().strip(),
            noise=noise_config,
            perturb=perturb_config,
        )

    def _get_noise_config(self) -> NoiseConfig:
        kwargs = {"relative": self.relative_cb.isChecked(), "extras": {}}
        for key, box in self._noise_boxes.items():
            val = float(box.text() or "0.0")
            if key in NoiseConfig.__annotations__:
                kwargs[key] = val
            else:
                kwargs["extras"][key] = val
        return NoiseConfig(**kwargs)

    def _get_perturb_config(self) -> PerturbationConfig:
        kwargs = {"extras_mean": {}, "extras_std": {}}
        for key, box in self._pert_mean_boxes.items():
            mean_val = float(box.text() or "0.0")
            std_val = float(self._pert_std_boxes[key].text() or "0.0")
            if f"{key}_mean" in PerturbationConfig.__annotations__:
                kwargs[f"{key}_mean"] = mean_val
                kwargs[f"{key}_std"] = std_val
            else:
                kwargs["extras_mean"][key] = mean_val
                kwargs["extras_std"][key] = std_val
        return PerturbationConfig(**kwargs)

    def _on_delete(self) -> None:
        layer_id = self._existing.layer_id if self._existing else self.id_spin.value()
        resp = QMessageBox.question(
            self,
            "Delete Layer",
            f"Are you sure you want to delete layer {layer_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp == QMessageBox.StandardButton.Yes:
            self.deleteRequested = True
            self.accept()

    def _populate_type_combo(self) -> dict[str, dict[str, object]]:
        layer_types = discover_layer_types()
        self._populate_grouped_combo(
            self.type_cb,
            layer_types.items(),
            ordered_categories=("Virtual", "Normalisation", "Physical", "Other"),
            default_category="Other",
            header_fmt=lambda cat: f"— {cat} Layers —",
            title_getter=lambda meta, key: str(meta.get("title", key)),
            description_getter=lambda meta: str(meta.get("description", "")),
            tooltip="Select the layer implementation to insert into the network.",
            min_width=250,
            header_color=QColor("#1f6feb"),
        )
        return layer_types

    def _current_layer_type(self) -> str:
        data = self.type_cb.currentData(Qt.ItemDataRole.UserRole)
        if isinstance(data, str) and data:
            return data
        return self.type_cb.currentText()

    def _set_layer_current(self, key: str) -> None:
        model = self.type_cb.model()
        for row in range(model.rowCount()):
            item = model.item(row)
            if item and item.data(Qt.ItemDataRole.UserRole) == key:
                self.type_cb.setCurrentIndex(row)
                return

    def _update_layer_description(self) -> None:
        key = self._current_layer_type()
        meta = _get_layer_types_discovered().get(key, {})
        desc = meta.get("description", "")
        self.layer_desc_label.setVisible(bool(desc))
        self.layer_desc_label.setText(desc)

    def _populate_source_combo(self) -> None:
        self._populate_grouped_combo(
            self.source_cb,
            ((key, meta) for key, meta in _get_source_functions_discovered().items() if key != "__defaults__"),
            ordered_categories=("SOEN", "Analytic", "Other"),
            default_category="Other",
            header_fmt=lambda cat: f"— {cat} Non-Linearities —",
            title_getter=lambda meta, key: str(meta.get("title", key)),
            description_getter=lambda meta: str(meta.get("description", "")),
            tooltip="Select the non-linearity applied by this layer.",
            min_width=260,
        )

    @staticmethod
    def _format_catalog_label(title: str, key: str) -> str:
        title_stripped = title.strip()
        key_stripped = key.strip()
        if not title_stripped or title_stripped.lower() == key_stripped.lower():
            return key_stripped
        return f"{title_stripped} ({key_stripped})"

    def _populate_grouped_combo(
        self,
        combo: QComboBox,
        entries: Iterable[tuple[str, Mapping[str, object]]],
        *,
        ordered_categories: Sequence[str],
        default_category: str,
        header_fmt: Callable[[str], str],
        title_getter: Callable[[Mapping[str, object], str], str],
        description_getter: Callable[[Mapping[str, object]], str],
        tooltip: str,
        min_width: int,
        header_color: QColor | None = None,
    ) -> None:
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        model = QStandardItemModel()
        category_rows: dict[str, list[tuple[str, Mapping[str, object]]]] = {}
        for key, meta in entries:
            category = str(meta.get("category", default_category)) or default_category
            category_rows.setdefault(category, []).append((key, meta))

        order = list(ordered_categories) + sorted(c for c in category_rows if c not in ordered_categories)
        for category in order:
            rows = category_rows.get(category)
            if not rows:
                continue
            header = QStandardItem(header_fmt(category))
            header.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if header_color is not None:
                header.setData(header_color, Qt.ItemDataRole.ForegroundRole)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            model.appendRow(header)
            for key, meta in sorted(rows, key=lambda item: title_getter(item[1], item[0])):
                title = title_getter(meta, key)
                display = self._format_catalog_label(title, key)
                item = QStandardItem(display)
                item.setData(key, Qt.ItemDataRole.UserRole)
                item.setToolTip(description_getter(meta))
                model.appendRow(item)

        combo.setModel(model)
        combo.view().setMinimumWidth(min_width)
        combo.setToolTip(tooltip)

    def _current_source_key(self) -> str:
        data = self.source_cb.currentData(Qt.ItemDataRole.UserRole)
        if isinstance(data, str) and data:
            return data
        return self.source_cb.currentText()

    def _set_source_current(self, key: str) -> None:
        model = self.source_cb.model()
        for row in range(model.rowCount()):
            item = model.item(row)
            if item and item.data(Qt.ItemDataRole.UserRole) == key:
                self.source_cb.setCurrentIndex(row)
                self._update_source_description()
                return

    def _update_source_description(self) -> None:
        key = self._current_source_key()
        meta = _get_source_functions_discovered().get(key, {})
        desc = meta.get("description", "")
        self.source_desc_label.setVisible(bool(desc))
        self.source_desc_label.setText(desc)

    def _current_solver_key(self) -> str:
        data = self.solver_cb.currentData(Qt.ItemDataRole.UserRole)
        if isinstance(data, str) and data:
            return data
        return self.solver_cb.currentText()

    def _set_solver_current(self, key: str) -> None:
        for i in range(self.solver_cb.count()):
            if self.solver_cb.itemData(i) == key:
                self.solver_cb.setCurrentIndex(i)
                return

    def _determine_default_source(self) -> str:
        if self._existing:
            existing = self._existing.params.get("source_func")
            if existing:
                return existing

        defaults = _get_source_functions_discovered().get("__defaults__", {})
        layer_key = self._current_layer_type()
        return defaults.get(layer_key, "RateArray")

    def _maybe_reset_source_default(self, layer_key: str) -> None:
        if layer_key not in LAYERS_REQUIRING_SOURCE_FUNCTION:
            return

        defaults = _get_source_functions_discovered().get("__defaults__", {})
        default_source = defaults.get(layer_key)
        if default_source is None:
            return

        # If editing an existing layer, we still want to snap to the new type's
        # default when the type is changed. However, if the user already has the
        # default selected, do nothing.
        current_key = self._current_source_key()
        if current_key == default_source:
            return

        options = {self.source_cb.itemData(i) for i in range(self.source_cb.count())}
        if default_source in options:
            self._set_source_current(default_source)
