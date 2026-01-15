# FILE: src/soen_toolkit/model_creation_gui/components/connection_param_editor.py

from __future__ import annotations

import inspect
import re
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDoubleSpinBox, QFormLayout, QLabel, QSpinBox, QWidget

from soen_toolkit.core.layers.common import CONNECTIVITY_BUILDERS
from soen_toolkit.core.layers.common.connectivity_metadata import (
    normalize_connectivity_type,
)


class ConnectionParamEditor(QWidget):
    """Dynamically show only the params referenced in the chosen builder function:
    - Scans the source of C.connectivity_builders[conn_type]
    - Extracts any params['KEY'] occurrences.
    """

    def __init__(self, existing: dict[str, Any] | None, conn_type: str) -> None:
        super().__init__()
        self._existing = existing or {}
        self._conn_type = conn_type
        self._boxes: dict[str, QDoubleSpinBox | QSpinBox] = {}
        self._init_ui()

    def _get_fields(self) -> list[str]:
        """Inspect the builder source code for params['...'] keys."""
        key = normalize_connectivity_type(self._conn_type)
        builder = CONNECTIVITY_BUILDERS.get(key)
        if not builder:
            return []
        src = inspect.getsource(builder)

        # find all occurrences of params['key'] or params["key"]
        bracket_keys = re.findall(r"params\[['\"]([^'\"]+)['\"]\]", src)

        # find all occurrences of params.get('key') or params.get("key")
        get_keys = re.findall(r"params\.get\(['\"]([^'\"]+)['\"]", src)

        keys = bracket_keys + get_keys

        # remove duplicates and any generic access like 'constraints'
        return sorted({k for k in keys if k != "constraints"})

    def _init_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(10)

        fields = self._get_fields()
        for fld in fields:
            # choose widget type
            if fld.endswith(("size", "Size")) or "block" in fld.lower():
                widget = QSpinBox()
                widget.setRange(1, 10**6)
            elif fld.lower() in ("sparsity",):
                widget = QDoubleSpinBox()
                widget.setRange(0.0, 1.0)
                widget.setDecimals(3)
                widget.setSingleStep(0.05)
            else:
                widget = QDoubleSpinBox()
                widget.setRange(-1e12, 1e12)
                widget.setDecimals(6)

            # populate existing or default
            if fld in self._existing:
                widget.setValue(self._existing[fld])
            # a small guess: default sparsity=0.5, block_size=1
            elif fld.lower() == "sparsity":
                widget.setValue(0.5)
            elif fld.lower().endswith("size"):
                widget.setValue(1)
                # else leave at zero

            layout.addRow(f"{fld.replace('_', ' ').capitalize()}:", widget)
            self._boxes[fld] = widget

        # if no fields, show a label
        if not fields:
            placeholder = QLabel("No init parameters for this connection type.")
            placeholder.setStyleSheet("font-style: italic; color: #555;")
            layout.addRow("", placeholder)

    def values(self) -> dict[str, Any]:
        """Return a dict of whatever fields we discovered."""
        return {fld: box.value() for fld, box in self._boxes.items()}
