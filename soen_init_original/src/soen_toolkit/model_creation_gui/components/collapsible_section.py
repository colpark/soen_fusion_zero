from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A simple collapsible section: header button + hideable content.

    Usage:
        sec = CollapsibleSection("Simulation Settings", collapsed=True)
        sec.setContent(your_widget)
        layout.addWidget(sec)
    """

    def __init__(self, title: str = "Section", *, collapsed: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content: QWidget | None = None
        self._collapsed = bool(collapsed)

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)

        self._header_btn = QToolButton(self)
        self._header_btn.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._header_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header_btn.setArrowType(Qt.ArrowType.RightArrow if self._collapsed else Qt.ArrowType.DownArrow)
        self._header_btn.setText(title)
        self._header_btn.setCheckable(True)
        self._header_btn.setChecked(not self._collapsed)
        self._header_btn.toggled.connect(self._on_toggled)
        self._root.addWidget(self._header_btn)

        # Separator line below header
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        self._root.addWidget(sep)

    def setContent(self, widget: QWidget) -> None:
        if self._content is not None:
            self._content.setParent(None)
        self._content = widget
        self._root.addWidget(widget)
        self._apply_collapsed_state()

    def content(self) -> QWidget | None:
        return self._content

    def setCollapsed(self, collapsed: bool) -> None:
        self._collapsed = bool(collapsed)
        self._header_btn.setChecked(not self._collapsed)
        self._header_btn.setArrowType(Qt.ArrowType.RightArrow if self._collapsed else Qt.ArrowType.DownArrow)
        self._apply_collapsed_state()

    def isCollapsed(self) -> bool:
        return self._collapsed

    # slots
    def _on_toggled(self, checked: bool) -> None:
        self._collapsed = not checked
        self._header_btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._apply_collapsed_state()

    def _apply_collapsed_state(self) -> None:
        if self._content is not None:
            self._content.setVisible(not self._collapsed)
