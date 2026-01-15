# FILE: src/soen_toolkit/model_creation_gui/components/constraints_editor.py
from __future__ import annotations

from PyQt6.QtGui import QDoubleValidator, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..utils.constants import FLOAT_DECIMALS, FLOAT_MAX, FLOAT_MIN


class ConstraintsEditor(QWidget):
    """Scrollable constraints editor with clear labels per field."""

    def __init__(self, param_names: list[str], existing: dict[str, dict] | None = None) -> None:
        super().__init__()
        self._existing = existing or {}
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        v = QVBoxLayout(container)
        title = QLabel("Parameter Constraints")
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        v.addWidget(title)

        self._fields: dict[str, tuple[QCheckBox, QLineEdit, QLineEdit]] = {}
        for name in param_names:
            group = QGroupBox(name)
            group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            layout = QHBoxLayout(group)
            layout.setContentsMargins(8, 4, 8, 4)
            layout.setSpacing(10)
            enable = QCheckBox("Enable")

            min_sb = QLineEdit()
            validator_min = QDoubleValidator(FLOAT_MIN, FLOAT_MAX, FLOAT_DECIMALS)
            validator_min.setNotation(QDoubleValidator.Notation.StandardNotation)
            min_sb.setValidator(validator_min)
            min_sb.setText("0.0")

            max_sb = QLineEdit()
            validator_max = QDoubleValidator(FLOAT_MIN, FLOAT_MAX, FLOAT_DECIMALS)
            validator_max.setNotation(QDoubleValidator.Notation.StandardNotation)
            max_sb.setValidator(validator_max)
            max_sb.setText("0.0")

            layout.addWidget(enable)
            layout.addWidget(QLabel("Min:"))
            layout.addWidget(min_sb)
            layout.addWidget(QLabel("Max:"))
            layout.addWidget(max_sb)
            layout.addStretch()
            if name in self._existing:
                enable.setChecked(True)
                cons = self._existing[name]
                try:
                    min_sb.setText(str(float(cons.get("min", 0.0))))
                    max_sb.setText(str(float(cons.get("max", 0.0))))
                except (ValueError, TypeError):
                    min_sb.setText("0.0")
                    max_sb.setText("0.0")
            self._fields[name] = (enable, min_sb, max_sb)
            v.addWidget(group)
        v.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def values(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, (enable, min_sb, max_sb) in self._fields.items():
            if enable.isChecked():
                try:
                    min_val = float(min_sb.text() or "0.0")
                    max_val = float(max_sb.text() or "0.0")
                    out[name] = {"min": min_val, "max": max_val}
                except ValueError:
                    out[name] = {"min": 0.0, "max": 0.0}
        return out
