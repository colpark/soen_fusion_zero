from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from soen_toolkit.core.configs import LayerConfig


class MergeLayersDialog(QDialog):
    """Dialog to select layers to merge and choose the new layer ID.

    Guardrails:
      - Requires at least 2 layers selected
      - All selected layers must have the same layer_type
      - Displays helpful messages when constraints are not satisfied
    """

    def __init__(self, parent=None, layers: list[LayerConfig] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge Layers")
        self._layers = list(layers or [])

        layout = QVBoxLayout(self)

        info = QLabel(
            "Select two or more layers of the same type to merge into a single super-layer.\n"
            "Parameters must be node-wise (1D vectors of length dim). Internal connectivity is handled\n"
            "via connections and will be preserved.",
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for cfg in sorted(self._layers, key=lambda c: c.layer_id):
            dim = cfg.params.get("dim")
            it = QListWidgetItem(f"Layer {cfg.layer_id} • {cfg.layer_type} • dim={dim}")
            it.setData(Qt.ItemDataRole.UserRole, (cfg.layer_id, cfg.layer_type))
            self.list_widget.addItem(it)
        self.list_widget.itemSelectionChanged.connect(self._validate)
        layout.addWidget(self.list_widget)

        row = QHBoxLayout()
        row.addWidget(QLabel("New Layer ID:"))
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 999999)
        # default to min(selected) once selection changes
        row.addWidget(self.id_spin)

        self.normalize_chk = QCheckBox("Normalize IDs after merge")
        self.normalize_chk.setChecked(True)
        row.addWidget(self.normalize_chk)

        row.addStretch()
        layout.addLayout(row)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #b00020;")
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # initial validation
        self._validate()

    def selected_group(self) -> list[int]:
        out = []
        for it in self.list_widget.selectedItems():
            lid, _ = it.data(Qt.ItemDataRole.UserRole)
            out.append(int(lid))
        return sorted(out)

    def normalize_ids(self) -> bool:
        return self.normalize_chk.isChecked()

    def new_layer_id(self) -> int:
        return int(self.id_spin.value())

    def _on_accept(self) -> None:
        if self._validate():
            self.accept()

    def _validate(self) -> bool:
        items = self.list_widget.selectedItems()
        if len(items) < 2:
            self.error_label.setText("Select at least two layers.")
            # Update id spin default
            ids = [it.data(Qt.ItemDataRole.UserRole)[0] for it in items]
            self.id_spin.setValue(min(ids) if ids else 0)
            return False

        types = {it.data(Qt.ItemDataRole.UserRole)[1] for it in items}
        if len(types) != 1:
            self.error_label.setText("All selected layers must have the same type.")
            ids = [it.data(Qt.ItemDataRole.UserRole)[0] for it in items]
            self.id_spin.setValue(min(ids) if ids else 0)
            return False

        ids = [it.data(Qt.ItemDataRole.UserRole)[0] for it in items]
        self.id_spin.setValue(min(ids))

        # Validate that chosen new ID does not collide with an unrelated layer
        selected = set(ids)
        chosen = int(self.id_spin.value())
        existing_ids = {cfg.layer_id for cfg in self._layers}
        if (chosen in existing_ids) and (chosen not in selected):
            self.error_label.setText(f"ID {chosen} already exists (outside selection). Choose a free ID.")
            return False

        self.error_label.setText("")
        return True
