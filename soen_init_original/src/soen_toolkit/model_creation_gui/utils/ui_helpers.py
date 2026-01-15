"""UI helper functions for common PyQt6 patterns.

This module provides reusable functions for common UI operations,
eliminating code duplication across dialogs and tabs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QMessageBox,
)

from .constants import FLOAT_DECIMALS, FLOAT_MAX, FLOAT_MIN

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


def create_float_validator(
    parent: QWidget | None = None,
    min_val: float = FLOAT_MIN,
    max_val: float = FLOAT_MAX,
    decimals: int = FLOAT_DECIMALS,
) -> QDoubleValidator:
    """Create a QDoubleValidator with scientific notation support.

    Args:
        parent: Optional parent widget
        min_val: Minimum allowed value (default: -1e18)
        max_val: Maximum allowed value (default: 1e18)
        decimals: Number of decimal places (default: 16)

    Returns:
        Configured QDoubleValidator

    Example:
        >>> validator = create_float_validator()
        >>> line_edit.setValidator(validator)
    """
    validator = QDoubleValidator(min_val, max_val, decimals, parent)
    validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
    return validator


def create_dialog_buttons(
    parent: QDialog | None = None,
    ok: bool = True,
    cancel: bool = True,
    auto_connect: bool = True,
) -> QDialogButtonBox:
    """Create a standard dialog button box with OK and/or Cancel buttons.

    Args:
        parent: Parent dialog widget (should be a QDialog for auto_connect to work)
        ok: Include OK button (default: True)
        cancel: Include Cancel button (default: True)
        auto_connect: Automatically connect accepted/rejected signals to parent's
                      accept()/reject() methods (default: True)

    Returns:
        Configured QDialogButtonBox

    Example:
        >>> buttons = create_dialog_buttons(self)
        >>> layout.addWidget(buttons)
    """
    buttons_flags = QDialogButtonBox.StandardButton.NoButton

    if ok:
        buttons_flags |= QDialogButtonBox.StandardButton.Ok
    if cancel:
        buttons_flags |= QDialogButtonBox.StandardButton.Cancel

    button_box = QDialogButtonBox(buttons_flags, parent)

    if auto_connect and parent is not None:
        if hasattr(parent, "accept") and hasattr(parent, "reject"):
            button_box.accepted.connect(parent.accept)
            button_box.rejected.connect(parent.reject)

    return button_box


def create_form_group(
    title: str,
    parent: QWidget | None = None,
) -> tuple[QGroupBox, QFormLayout]:
    """Create a QGroupBox with QFormLayout for form-style UI.

    Args:
        title: Title for the group box
        parent: Optional parent widget

    Returns:
        Tuple of (QGroupBox, QFormLayout)

    Example:
        >>> group, form = create_form_group("Settings")
        >>> form.addRow("Name:", name_edit)
        >>> layout.addWidget(group)
    """
    group = QGroupBox(title, parent)
    form = QFormLayout()
    group.setLayout(form)
    return group, form


def create_double_spinbox(
    parent: QWidget | None = None,
    min_val: float = 0.0,
    max_val: float = 100.0,
    decimals: int = 2,
    step: float = 1.0,
    default: float = 0.0,
) -> QDoubleSpinBox:
    """Create a configured QDoubleSpinBox.

    Args:
        parent: Optional parent widget
        min_val: Minimum value (default: 0.0)
        max_val: Maximum value (default: 100.0)
        decimals: Number of decimal places (default: 2)
        step: Single step increment (default: 1.0)
        default: Default/initial value (default: 0.0)

    Returns:
        Configured QDoubleSpinBox

    Example:
        >>> spinbox = create_double_spinbox(min_val=0, max_val=10, step=0.5, default=5)
        >>> layout.addWidget(spinbox)
    """
    spinbox = QDoubleSpinBox(parent)
    spinbox.setRange(min_val, max_val)
    spinbox.setDecimals(decimals)
    spinbox.setSingleStep(step)
    spinbox.setValue(default)
    return spinbox


def show_error(parent: QWidget | None, title: str, message: str) -> None:
    """Show a critical error message dialog.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Error message to display

    Example:
        >>> show_error(self, "Error", "Failed to load model")
    """
    QMessageBox.critical(parent, title, message)


def show_warning(parent: QWidget | None, title: str, message: str) -> None:
    """Show a warning message dialog.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Warning message to display

    Example:
        >>> show_warning(self, "Warning", "Model has unsaved changes")
    """
    QMessageBox.warning(parent, title, message)


def show_info(parent: QWidget | None, title: str, message: str) -> None:
    """Show an information message dialog.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Information message to display

    Example:
        >>> show_info(self, "Success", "Model saved successfully")
    """
    QMessageBox.information(parent, title, message)


def ask_yes_no(parent: QWidget | None, title: str, message: str) -> bool:
    """Ask a yes/no question and return the result.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Question to ask

    Returns:
        True if user clicked Yes, False if No

    Example:
        >>> if ask_yes_no(self, "Confirm", "Delete this layer?"):
        ...     # User clicked Yes
        ...     delete_layer()
    """
    result = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    return result == QMessageBox.StandardButton.Yes
