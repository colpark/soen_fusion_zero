# FILE: src/soen_toolkit/model_creation_gui/components/func_param_editor.py

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from collections.abc import Callable

# Import weight initializer metadata
from soen_toolkit.core.layers.common.connectivity_metadata import WEIGHT_INITIALIZER_PARAM_TYPES


class FuncParamEditor(QWidget):
    """Introspects a function's signature and builds a QFormLayout with
    one spin‑box per numeric keyword parameter (int or float)
    (skips pure-positional args or non-numeric defaults).
    Uses QSpinBox for ints, QLineEdit+QDoubleValidator for floats.
    """

    def __init__(self, func: Callable, existing: dict[str, Any] | None = None, param_descriptions: dict[str, str] | None = None) -> None:
        super().__init__()
        self._func = func
        self._existing = existing or {}
        self._param_descriptions = param_descriptions or {}
        self._boxes: dict[str, QSpinBox | QLineEdit | QWidget] = {}
        self._build_ui()

    def _is_file_parameter(self, param_name: str, param_default: Any) -> bool:
        """Check if a parameter should be treated as a file selector.

        Uses metadata from WEIGHT_INITIALIZER_PARAM_TYPES first, then falls back to heuristics.
        """
        # Try to get function name from the function object
        func_name = getattr(self._func, "__name__", "").replace("init_", "")

        # Check metadata first (most reliable)
        if func_name in WEIGHT_INITIALIZER_PARAM_TYPES:
            param_info = WEIGHT_INITIALIZER_PARAM_TYPES[func_name].get(param_name, {})
            if param_info.get("type") == "file":
                return True

        # Fallback heuristics for non-registered functions
        # Check if default value is a string (simple and reliable)
        if isinstance(param_default, str):
            return True

        # Check if parameter name suggests it's a file path
        name_lower = param_name.lower()
        if "file" in name_lower or "path" in name_lower:
            return True

        return False

    def _build_ui(self) -> None:
        sig = inspect.signature(self._func)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        for name, param in sig.parameters.items():
            # Skip pure-positional or required args without default
            if param.kind == inspect.Parameter.POSITIONAL_ONLY or (param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and param.default is inspect._empty):
                continue

            default = param.default if param.default is not inspect._empty else 0.0

            # Create a vertical param container to match structure UI
            param_container = QWidget()
            param_layout = QVBoxLayout(param_container)
            param_layout.setContentsMargins(0, 0, 0, 0)
            param_layout.setSpacing(6)

            # Label above
            pretty_name = name.replace("_", " ").capitalize()
            label = QLabel(f"{pretty_name}:")
            from PyQt6.QtGui import QFont as _QFont

            _bold = _QFont()
            _bold.setBold(True)
            label.setFont(_bold)
            param_layout.addWidget(label)

            if self._is_file_parameter(name, default):
                existing_val = self._existing.get(name, default if isinstance(default, str) else "")
                widget = self._create_file_selector(name, str(existing_val) if existing_val else "")
                self._boxes[name] = widget
                param_layout.addWidget(widget)
                # Description under widget
                if name in self._param_descriptions:
                    desc_label = QLabel(self._param_descriptions[name])
                    desc_label.setWordWrap(True)
                    desc_label.setStyleSheet("font-style: italic; color: #666; margin-left: 10px; font-size: 9pt; background-color: transparent;")
                    desc_label.setMinimumHeight(30)
                    param_layout.addWidget(desc_label)
                root.addWidget(param_container)
                continue

            # Only show numeric parameters
            if not isinstance(default, (int, float)):
                continue

            existing_val = self._existing.get(name)
            val = existing_val if isinstance(existing_val, (int, float)) else default

            if isinstance(default, int):
                widget = QSpinBox()
                widget.setRange(-(10**6), 10**6)
                try:
                    widget.setValue(int(val))
                except (ValueError, TypeError):
                    widget.setValue(int(default))
            else:
                widget = QLineEdit()
                validator = QDoubleValidator(-1e18, 1e18, 16)
                validator.setNotation(QDoubleValidator.Notation.StandardNotation)
                widget.setValidator(validator)
                try:
                    widget.setText(str(float(val)))
                except (ValueError, TypeError):
                    widget.setText(str(float(default)))

            self._boxes[name] = widget
            param_layout.addWidget(widget)
            # Description below widget
            if name in self._param_descriptions:
                desc_label = QLabel(self._param_descriptions[name])
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("font-style: italic; color: #666; margin-left: 10px; font-size: 9pt; background-color: transparent;")
                desc_label.setMinimumHeight(30)
                param_layout.addWidget(desc_label)

            root.addWidget(param_container)

        if not self._boxes:
            root.addWidget(QLabel("<i>(no parameters)</i>"))

    def _create_file_selector(self, param_name: str, current_value: str = "") -> QWidget:
        """Create a file selector widget with browse button."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        line_edit = QLineEdit(current_value)
        line_edit.setPlaceholderText("Select file or enter path...")

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(
            lambda: self._browse_file(line_edit, param_name),
        )

        layout.addWidget(line_edit)
        layout.addWidget(browse_btn)

        # Store reference to line_edit for later value retrieval
        setattr(container, "line_edit", line_edit)
        return container

    def _browse_file(self, line_edit: QLineEdit, param_name: str) -> None:
        """Open file dialog and update line edit with selected path."""
        # Determine file filter based on parameter name
        file_filter = "All Files (*)"
        name_lower = param_name.lower()
        if "weights" in name_lower:
            file_filter = "NumPy Files (*.npy *.npz);;All Files (*)"
        elif "mask" in name_lower:
            file_filter = "NPZ Files (*.npz);;All Files (*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            file_filter,
        )
        if file_path:
            line_edit.setText(file_path)

    def values(self) -> dict[str, Any]:
        """Return a dict mapping parameter names to the current widget values."""
        results = {}
        for name, box in self._boxes.items():
            if hasattr(box, "line_edit"):
                # File selector widget
                line_edit = getattr(box, "line_edit")
                results[name] = line_edit.text()
            elif isinstance(box, QLineEdit):
                try:
                    results[name] = float(box.text() or "0.0")
                except ValueError:
                    results[name] = 0.0  # Default on error
            elif isinstance(box, QSpinBox):
                results[name] = box.value()
            # else: handle other types if added later
        return results
