from __future__ import annotations

import re

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)


class QuantizeDialog(QDialog):
    """Simple dialog to collect quantization parameters and target connections.

    Fields:
    - bits (int)
    - min (float)
    - max (float)
    - checkable list of current model connection keys
    """

    def __init__(self, parent=None, *, model: object | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quantize Connections")
        self._model = model
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Parameters form
        form = QFormLayout()

        self.bits_spin = QSpinBox()
        self.bits_spin.setRange(0, 32)
        self.bits_spin.setValue(3)
        form.addRow("bits:", self.bits_spin)

        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(8)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setSingleStep(0.01)
        self.min_spin.setValue(-0.24)
        form.addRow("min:", self.min_spin)

        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(8)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setSingleStep(0.01)
        self.max_spin.setValue(0.24)
        form.addRow("max:", self.max_spin)

        layout.addLayout(form)

        # Connections list
        layout.addWidget(QLabel("Select connections to quantize:"))
        self.conn_list = QListWidget()
        self.conn_list.setAlternatingRowColors(True)
        layout.addWidget(self.conn_list)

        self._populate_connections()

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_connections(self) -> None:
        self.conn_list.clear()
        if self._model is None or not hasattr(self._model, "connections"):
            return

        # Show only quantizable tensors (torch.Tensor) and indicate frozen status
        for name, param in self._model.connections.items():
            label = self._friendly_label(name, param)
            it = QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Default: check only learnable params
            default_checked = getattr(param, "requires_grad", True)
            it.setCheckState(Qt.CheckState.Checked if default_checked else Qt.CheckState.Unchecked)
            self.conn_list.addItem(it)

    def _friendly_label(self, key: str, param) -> str:
        frozen = hasattr(param, "requires_grad") and not param.requires_grad
        suffix = " [frozen]" if frozen else ""
        # Parse common key patterns for readability
        m = re.match(r"^J_(\d+)_to_(\d+)$", key)
        if m:
            a, b = m.group(1), m.group(2)
            return f"{a}→{b} (external){suffix}"
        m2 = re.match(r"^internal_(\d+)$", key)
        if m2:
            a = m2.group(1)
            return f"{a} (internal){suffix}"
        return key + suffix

    def selected_connections(self) -> list[str]:
        names: list[str] = []
        for i in range(self.conn_list.count()):
            it = self.conn_list.item(i)
            if it is not None and it.checkState() == Qt.CheckState.Checked:
                names.append(str(it.data(Qt.ItemDataRole.UserRole)))
        return names

    def result_params(self) -> tuple[int, float, float, list[str]]:
        return (
            int(self.bits_spin.value()),
            float(self.min_spin.value()),
            float(self.max_spin.value()),
            self.selected_connections(),
        )

    def accept(self) -> None:
        # Validate before closing
        min_v = float(self.min_spin.value())
        max_v = float(self.max_spin.value())
        if max_v <= min_v:
            QMessageBox.warning(self, "Invalid range", "max must be greater than min.")
            return
        if len(self.selected_connections()) == 0:
            QMessageBox.warning(self, "No selection", "Select at least one connection to quantize.")
            return
        super().accept()
