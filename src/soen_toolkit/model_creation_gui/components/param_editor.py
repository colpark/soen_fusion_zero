# FILE: src/soen_toolkit/model_creation_gui/components/param_editor.py
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from soen_toolkit.core.layers.common.metadata import LAYER_PARAM_CONFIGS, ParamConfig

from ..utils.ui_helpers import create_float_validator

_ALL_DISTS = ["constant", "normal", "uniform", "loguniform", "linear", "loglinear", "fan_out"]


class ParamEditor(QWidget):
    """Form-based parameter editor with clear labels and:
    - correct defaults
    - no scale input for fan_out on gamma_plus parameters
    - block_size support for grouping nodes with shared values
    """

    def __init__(self, layer_type: str, existing: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._existing = existing or {}
        self._configs: list[ParamConfig] = LAYER_PARAM_CONFIGS.get(layer_type, [])
        self._combo: dict[str, QComboBox] = {}
        self._stack: dict[str, QStackedWidget] = {}
        self._learn: dict[str, QCheckBox] = {}
        self._block_size: dict[str, QSpinBox] = {}
        self._block_mode: dict[str, QComboBox] = {}
        self._block_enabled: dict[str, QCheckBox] = {}  # Toggle for showing block settings
        self._block_widgets: dict[str, list[QWidget]] = {}  # Widgets to show/hide
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        for pc in self._configs:
            group = QGroupBox(pc.name)
            group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

            # Distribution + Learnable
            dist_cb = QComboBox()
            dists = _ALL_DISTS.copy()
            if not pc.name.startswith("gamma_plus"):
                dists.remove("fan_out")
            dist_cb.addItems(dists)

            # --- Sizing Policy ---
            dist_cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            fm = dist_cb.fontMetrics()
            longest = max((fm.horizontalAdvance(t) for t in dists), default=150)
            dist_cb.view().setMinimumWidth(longest + 30)

            learn_cb = QCheckBox("Learnable")
            dist_row = QWidget()
            hr = QVBoxLayout(dist_row)
            hr.setContentsMargins(0, 0, 0, 0)
            hr.setSpacing(4)
            hr.addWidget(dist_cb)
            hr.addWidget(learn_cb)
            form.addRow("Distribution:", dist_row)

            # Toggle for enabling block mode (hidden by default)
            block_enabled_cb = QCheckBox("Enable Block Mode")
            block_enabled_cb.setToolTip(
                "Enable block-based parameter sharing.\n\n"
                "When disabled, each node gets its own independent value.\n"
                "When enabled, you can group nodes into blocks with shared or tiled values."
            )
            form.addRow("", block_enabled_cb)

            # Block size spinbox (hidden by default)
            block_size_label = QLabel("Block Size:")
            block_spin = QSpinBox()
            block_spin.setRange(1, 10000)
            block_spin.setValue(1)
            block_spin.setToolTip(
                "Group nodes into blocks.\n\n"
                "Example: With 30 nodes and block_size=3, you get 10 blocks of 3 nodes each.\n\n"
                "Requirement: Layer width must be evenly divisible by block_size.\n"
                "Default: 1 (no blocking, each node gets its own value)."
            )
            form.addRow(block_size_label, block_spin)

            # Block mode combo box (hidden by default, defaults to tiled)
            block_mode_label = QLabel("Block Mode:")
            block_mode_cb = QComboBox()
            block_mode_cb.addItems(["tiled", "shared"])  # tiled is default
            block_mode_cb.setToolTip(
                "How values are distributed with blocking:\n\n"
                "tiled (default):\n"
                "  - Values vary within each block (one per position)\n"
                "  - Same pattern repeats across all blocks\n"
                "  - Example: [v1,v2,v3, v1,v2,v3, v1,v2,v3, v1,v2,v3]\n\n"
                "shared:\n"
                "  - Nodes within a block share the same value\n"
                "  - Values vary across blocks\n"
                "  - Example: [v1,v1,v1, v2,v2,v2, v3,v3,v3, v4,v4,v4]"
            )
            form.addRow(block_mode_label, block_mode_cb)

            # Store block widgets for show/hide
            block_widgets = [block_size_label, block_spin, block_mode_label, block_mode_cb]

            # Hide block settings by default
            for w in block_widgets:
                w.setVisible(False)

            # Connect toggle to show/hide block settings
            def toggle_block_widgets(checked: bool, widgets: list[QWidget] = block_widgets) -> None:
                for w in widgets:
                    w.setVisible(checked)

            block_enabled_cb.toggled.connect(toggle_block_widgets)

            # Stacked widget for parameters
            stack = QStackedWidget()

            # --- constant ---
            w0 = QWidget()
            f0 = QFormLayout(w0)
            sb0 = QLineEdit()
            sb0.setValidator(create_float_validator())
            if not isinstance(self._existing.get(pc.name), dict):
                sb0.setText(str(pc.default_value))
            else:
                sb0.setText(str(pc.default_value))
            f0.addRow("Value:", sb0)
            stack.addWidget(w0)

            # --- normal ---
            w1 = QWidget()
            f1 = QFormLayout(w1)
            sb1m = QLineEdit()
            sb1m.setValidator(create_float_validator())
            sb1m.setText("0.0")

            sb1s = QLineEdit()
            sb1s.setValidator(create_float_validator())
            sb1s.setText("0.0")

            f1.addRow("Mean:", sb1m)
            f1.addRow("Std Dev:", sb1s)
            stack.addWidget(w1)

            # --- uniform ---
            w2 = QWidget()
            f2 = QFormLayout(w2)
            sb2min = QLineEdit()
            sb2min.setValidator(create_float_validator())
            sb2min.setText("0.0")

            sb2max = QLineEdit()
            sb2max.setValidator(create_float_validator())
            sb2max.setText("0.0")

            f2.addRow("Min:", sb2min)
            f2.addRow("Max:", sb2max)
            stack.addWidget(w2)
            # --- loguniform ---
            w_lu = QWidget()
            f_lu = QFormLayout(w_lu)
            sb_lu_min = QLineEdit()
            sb_lu_min.setValidator(create_float_validator())
            sb_lu_min.setText("0.0")
            sb_lu_max = QLineEdit()
            sb_lu_max.setValidator(create_float_validator())
            sb_lu_max.setText("0.0")
            f_lu.addRow("Min:", sb_lu_min)
            f_lu.addRow("Max:", sb_lu_max)
            stack.addWidget(w_lu)

            # --- linear ---
            w3 = QWidget()
            f3 = QFormLayout(w3)
            sb3min = QLineEdit()
            sb3min.setValidator(create_float_validator())
            sb3min.setText("0.0")

            sb3max = QLineEdit()
            sb3max.setValidator(create_float_validator())
            sb3max.setText("0.0")

            f3.addRow("Min:", sb3min)
            f3.addRow("Max:", sb3max)
            stack.addWidget(w3)
            # --- loglinear ---
            w_ll = QWidget()
            f_ll = QFormLayout(w_ll)
            sb_ll_min = QLineEdit()
            sb_ll_min.setValidator(create_float_validator())
            sb_ll_min.setText("0.0")
            sb_ll_max = QLineEdit()
            sb_ll_max.setValidator(create_float_validator())
            sb_ll_max.setText("0.0")
            f_ll.addRow("Min:", sb_ll_min)
            f_ll.addRow("Max:", sb_ll_max)
            stack.addWidget(w_ll)

            # --- fan_out ---
            w4 = QWidget()
            f4 = QFormLayout(w4)
            if pc.name.startswith("gamma_plus"):
                # Inductance per fan-out input for gamma_plus parameters
                sb4_inductance = QLineEdit()
                sb4_inductance.setValidator(create_float_validator())
                sb4_inductance.setText("5e-10")  # 0.5nH default
                sb4_inductance.setToolTip(
                    "Inductance value per fan-out connection.\n"
                    "Units: Henries (H)\n"
                    "Default: 0.5nH (5e-10 H)"
                )
                inductance_label = QLabel("Inductance per fan-out (H):")
                inductance_label.setToolTip("Units: Henries (H)")
                f4.addRow(inductance_label, sb4_inductance)
            else:
                sb4 = QLineEdit()
                sb4.setValidator(create_float_validator())
                sb4.setText("0.0")
                f4.addRow("Scale:", sb4)
            stack.addWidget(w4)

            # wire distribution change
            dist_cb.currentIndexChanged.connect(lambda idx, st=stack: st.setCurrentIndex(idx))
            form.addRow("Parameters:", stack)

            # populate existing config
            cfg = self._existing.get(pc.name)
            if isinstance(cfg, dict):
                dist = cfg.get("distribution", "constant")
                if dist in dists:
                    idx = dists.index(dist)
                    dist_cb.setCurrentIndex(idx)
                    stack.setCurrentIndex(idx)
                params = cfg.get("params", {})
                # Load block_size if present
                block_size_val = params.get("block_size", 1)
                block_spin.setValue(int(block_size_val) if block_size_val else 1)
                # Load block_mode if present (default is now tiled)
                block_mode_val = params.get("block_mode", "tiled")
                block_mode_idx = 0 if block_mode_val == "tiled" else 1
                block_mode_cb.setCurrentIndex(block_mode_idx)
                # Enable block mode toggle if non-default block settings exist
                has_block_settings = block_size_val > 1 or block_mode_val == "shared"
                if has_block_settings:
                    block_enabled_cb.setChecked(True)
                edits = stack.currentWidget().findChildren(QLineEdit)
                if dist == "constant" and len(edits) > 0:
                    edits[0].setText(str(params.get("value", pc.default_value)))
                elif dist == "normal" and len(edits) > 1:
                    edits[0].setText(str(params.get("mean", 0.0)))
                    edits[1].setText(str(params.get("std", 0.0)))
                elif dist in ("uniform", "linear", "loguniform", "loglinear") and len(edits) > 1:
                    edits[0].setText(str(params.get("min", 0.0)))
                    edits[1].setText(str(params.get("max", 0.0)))
                elif dist == "fan_out":
                    if pc.name.startswith("gamma_plus") and len(edits) > 0:
                        # Load inductance_per_fan for gamma_plus (default 0.5nH)
                        edits[0].setText(str(params.get("inductance_per_fan", 5e-10)))
                    elif not pc.name.startswith("gamma_plus") and len(edits) > 0:
                        edits[0].setText(str(params.get("scale", 0.0)))
                learn_cb.setChecked(bool(self._existing.get("learnable_params", {}).get(pc.name, False)))
            elif not isinstance(cfg, dict) and dists[0] == "constant":
                constant_widget = stack.widget(0)
                constant_le = constant_widget.findChildren(QLineEdit)[0]
                constant_le.setText(str(cfg if cfg is not None else pc.default_value))
                learn_cb.setChecked(bool(self._existing.get("learnable_params", {}).get(pc.name, False)))

            # finalize
            group.setLayout(form)
            layout.addWidget(group)

            # track widgets
            self._combo[pc.name] = dist_cb
            self._stack[pc.name] = stack
            self._learn[pc.name] = learn_cb
            self._block_size[pc.name] = block_spin
            self._block_mode[pc.name] = block_mode_cb
            self._block_enabled[pc.name] = block_enabled_cb
            self._block_widgets[pc.name] = block_widgets

        layout.addStretch()

    def values(self) -> dict[str, Any]:
        out: dict[str, Any] = {"learnable_params": {}}
        for name, dist_cb in self._combo.items():
            dist = dist_cb.currentText()
            widget = self._stack[name].currentWidget()
            edits = widget.findChildren(QLineEdit)
            params: dict[str, Any] = {}
            try:
                if dist == "constant" and len(edits) > 0:
                    params = {"value": float(edits[0].text() or "0.0")}
                elif dist == "normal" and len(edits) > 1:
                    params = {"mean": float(edits[0].text() or "0.0"), "std": float(edits[1].text() or "0.0")}
                elif dist in ("uniform", "linear", "loguniform", "loglinear") and len(edits) > 1:
                    params = {"min": float(edits[0].text() or "0.0"), "max": float(edits[1].text() or "0.0")}
                elif dist == "fan_out":
                    if name.startswith("gamma_plus") and len(edits) > 0:
                        # Output inductance_per_fan for gamma_plus parameters
                        params = {"inductance_per_fan": float(edits[0].text() or "5e-10")}
                    elif not name.startswith("gamma_plus") and len(edits) > 0:
                        params = {"scale": float(edits[0].text() or "0.0")}
            except ValueError:
                params = {}

            # Include blocking params only if block mode is enabled and non-default
            if self._block_enabled[name].isChecked():
                block_size = self._block_size[name].value()
                block_mode = self._block_mode[name].currentText()
                if block_size > 1:
                    params["block_size"] = block_size
                # Only include block_mode if it's "shared" (non-default)
                if block_mode == "shared":
                    params["block_mode"] = block_mode

            out[name] = {"distribution": dist, "params": params}
            out["learnable_params"][name] = self._learn[name].isChecked()
        return out
